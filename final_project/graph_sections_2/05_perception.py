# ======================================================================
# graph.py section: 05_perception.py
# Depth-Anything-3 and SAM3 inference wrappers
#
# NOTE: not a standalone module. Loaded by graph.py into a shared
# namespace via exec(); names from earlier sections are in scope here.
# ======================================================================
from __future__ import annotations
def load_da3(config: PipelineConfig) -> DepthAnything3:
    """Load the local DA3 model without allowing remote downloads."""
    print("Loading DA3...")
    os.environ.setdefault("HF_HOME", config.hf_cache_dir)
    if config.local_files_only:
        os.environ["HF_HUB_OFFLINE"] = "1"
        if not config.da3_model_dir.exists():
            raise FileNotFoundError(f"DA3 model directory not found: {config.da3_model_dir}")
    model = DepthAnything3.from_pretrained("depth-anything/DA3NESTED-GIANT-LARGE-1.1").to("cuda")
    print("DA3 ready.")
    return model


def run_da3(
    da3_model: DepthAnything3,
    rgb_list: Sequence[np.ndarray],
    T_wc_list: Sequence[np.ndarray],
    K_color: np.ndarray,
    T_cam_color: np.ndarray,
    color_w: int,
    color_h: int,
    chunk_size: int = DA3_CHUNK_SIZE,
) -> List[np.ndarray]:
    """Run DA3 depth estimation in chunks and return resized depth maps.

    ``T_wc_list`` holds head/IMU camera poses (the ``tf_camera`` topic). The
    images we hand DA3 were captured by the color camera, so the extrinsics
    we pass to DA3 must be ``inv(T_wc @ T_cam_color)`` (world-to-color), not
    ``inv(T_wc)`` (world-to-head). The two frames differ by a ~90 deg
    axis swap in this dataset, so getting this wrong systematically biases
    DA3's metric depth.
    """
    def _even_chunks(items: Sequence, size: int) -> List[Sequence]:
        chunks = [items[idx : idx + size] for idx in range(0, len(items), size)]
        if len(chunks) > 1 and len(chunks[-1]) == 1:
            chunks[-2] = list(chunks[-2]) + list(chunks[-1])
            chunks = chunks[:-1]
        return chunks

    T_cam_color_arr = np.asarray(T_cam_color, dtype=np.float32)
    outputs: List[np.ndarray] = []
    for rgb_chunk, pose_chunk in zip(_even_chunks(rgb_list, chunk_size), _even_chunks(T_wc_list, chunk_size)):
        pil_images = [Image.fromarray(rgb) for rgb in rgb_chunk]
        extrinsics = np.stack([
            np.linalg.inv(np.asarray(pose, dtype=np.float32) @ T_cam_color_arr).astype(np.float32)
            for pose in pose_chunk
        ])
        intrinsics = np.stack([K_color.astype(np.float32) for _ in rgb_chunk])
        use_pose = len(rgb_chunk) >= 2
        try:
            with torch.inference_mode():
                pred = da3_model.inference(
                    image=pil_images,
                    extrinsics=extrinsics,
                    intrinsics=intrinsics,
                    use_ray_pose=use_pose,
                    process_res=DA3_PROCESS_RES,
                    process_res_method="upper_bound_resize",
                )
        except Exception as exc:
            if "Degenerate covariance" in str(exc) or "Umeyama" in str(exc):
                with torch.inference_mode():
                    pred = da3_model.inference(
                        image=pil_images,
                        extrinsics=extrinsics,
                        intrinsics=intrinsics,
                        use_ray_pose=False,
                        process_res=DA3_PROCESS_RES,
                        process_res_method="upper_bound_resize",
                    )
            else:
                raise
        depths = pred.depth
        if isinstance(depths, torch.Tensor):
            depths = depths.float().cpu().numpy()
        for depth in depths:
            outputs.append(cv2.resize(depth, (color_w, color_h), interpolation=cv2.INTER_LINEAR))
        torch.cuda.empty_cache()
        gc.collect()
    return outputs


def run_sam3(
    jpeg_dir: Path,
    object_classes: Sequence[str],
    hand_prompts: Sequence[str],
    n_frames: int,
    color_w: int,
    color_h: int,
    predictor: Optional[Any] = None,
) -> Tuple[Dict[str, List[np.ndarray]], List[np.ndarray]]:
    """Track the requested objects and hands across the clip with SAM3 video.

    If ``predictor`` is provided, it is reused (caller owns its lifecycle). Otherwise
    a fresh predictor is built and shut down inside this call (legacy behavior).
    """
    H, W = color_h, color_w
    owns_predictor = predictor is None
    if owns_predictor:
        predictor = build_configured_sam3_video_predictor()
    object_masks = {label: [np.zeros((H, W), dtype=bool) for _ in range(n_frames)] for label in object_classes}
    hand_masks = [np.zeros((H, W), dtype=bool) for _ in range(n_frames)]
    prompts = [(label, object_masks[label]) for label in object_classes] + [(prompt, hand_masks) for prompt in hand_prompts]

    # One session per clip — frames load + GPU encoding happen once and are
    # reused across every text prompt. add_prompt internally resets the per-prompt
    # state, so prompts are processed sequentially within the shared session.
    response = predictor.handle_request({"type": "start_session", "resource_path": str(jpeg_dir)})
    session_id = response["session_id"]
    try:
        for text_prompt, store in tqdm(prompts, desc="  SAM3", leave=False):
            predictor.handle_request({"type": "reset_session", "session_id": session_id})
            predictor.handle_request({"type": "add_prompt", "session_id": session_id, "frame_index": 0, "text": text_prompt})
            for response in predictor.handle_stream_request({"type": "propagate_in_video", "session_id": session_id}):
                frame_index = response.get("frame_index", -1)
                outputs = response.get("outputs")
                if outputs is None or frame_index < 0 or frame_index >= n_frames:
                    continue
                for _, mask_data in outputs.items():
                    if isinstance(mask_data, dict):
                        mask_data = mask_data.get("mask", mask_data.get("masks"))
                    if mask_data is None:
                        continue
                    if isinstance(mask_data, torch.Tensor):
                        mask = mask_data.float().squeeze().cpu().numpy() > 0.5
                    elif isinstance(mask_data, np.ndarray):
                        mask = mask_data.squeeze() > 0.5
                    else:
                        continue
                    mask = np.asarray(mask)
                    if mask.size == 0:
                        continue
                    while mask.ndim > 2:
                        if mask.shape[0] == 0:
                            mask = np.zeros((0, 0), dtype=bool)
                            break
                        mask = mask[0]
                    if mask.ndim != 2 or mask.shape[0] == 0 or mask.shape[1] == 0:
                        continue
                    if mask.shape != (H, W):
                        mask = cv2.resize(mask.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
                    store[frame_index] |= mask
    finally:
        predictor.handle_request({"type": "close_session", "session_id": session_id})

    if owns_predictor:
        predictor.shutdown()
        del predictor
    torch.cuda.empty_cache()
    gc.collect()
    return object_masks, hand_masks



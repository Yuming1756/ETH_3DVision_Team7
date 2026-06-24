# ======================================================================
# graph.py section: 11_vlm_reasoning.py
# Qwen-VL loading, generation, detection and functional reasoning
#
# NOTE: not a standalone module. Loaded by graph.py into a shared
# namespace via exec(); names from earlier sections are in scope here.
# ======================================================================
from __future__ import annotations
def safe_json_extract(raw: str) -> str:
    """Extract the outermost JSON object substring from a model response."""
    if not raw:
        return "{}"
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return "{}"
    return raw[start : end + 1]


def strip_think_blocks(raw: str) -> str:
    """Remove Qwen thinking tags before downstream parsing."""
    return re.sub(r"<think>.*?</think>", "", raw or "", flags=re.DOTALL).strip()


def resolve_local_hf_model_path(model_id: str, cache_dir: str) -> str:
    """Resolve a local Hugging Face snapshot path for offline model loading."""
    def _snapshot_has_weights(snapshot_dir: Path) -> bool:
        if not snapshot_dir.exists() or not (snapshot_dir / "config.json").exists():
            return False
        if (snapshot_dir / "model.safetensors").exists() or (snapshot_dir / "pytorch_model.bin").exists():
            return True
        shard_files = list(snapshot_dir.glob("model-*.safetensors"))
        if shard_files:
            return True
        index_file = snapshot_dir / "model.safetensors.index.json"
        if index_file.exists():
            try:
                with open(index_file, "r", encoding="utf-8") as handle:
                    payload = json.load(handle)
                weight_map = payload.get("weight_map", {})
                shard_names = {name for name in weight_map.values() if isinstance(name, str)}
                if shard_names and all((snapshot_dir / shard_name).exists() for shard_name in shard_names):
                    return True
            except Exception:
                return False
        return False

    candidate = Path(model_id).expanduser()
    if candidate.exists():
        return str(candidate)

    search_roots = []
    env_hf_home = os.environ.get("HF_HOME")
    for root in [Path(cache_dir).expanduser(), Path(env_hf_home).expanduser() if env_hf_home else None, Path("/work/courses/3dv/team7/.cache/huggingface")]:
        if root is not None and root not in search_roots:
            search_roots.append(root)

    repo_dir = f"models--{model_id.replace('/', '--')}"
    for root in search_roots:
        for base in [root, root / "hub"]:
            snapshot_root = base / repo_dir / "snapshots"
            if not snapshot_root.exists():
                continue
            snapshots = [entry for entry in snapshot_root.iterdir() if entry.is_dir()]
            if not snapshots:
                continue
            valid_snapshots = [entry for entry in snapshots if _snapshot_has_weights(entry)]
            if not valid_snapshots:
                continue
            latest = max(valid_snapshots, key=lambda entry: entry.stat().st_mtime)
            return str(latest)
    return model_id


def _format_cuda_mem_snapshot() -> str:
    """Return a compact summary of current CUDA memory usage for debugging."""
    if not torch.cuda.is_available():
        return "cuda unavailable"
    try:
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        allocated = torch.cuda.memory_allocated()
        reserved = torch.cuda.memory_reserved()
        return (
            f"free={free_bytes / (1024**3):.2f} GiB / total={total_bytes / (1024**3):.2f} GiB, "
            f"allocated={allocated / (1024**3):.2f} GiB, reserved={reserved / (1024**3):.2f} GiB"
        )
    except Exception as exc:  # noqa: BLE001
        return f"cuda mem unavailable ({exc})"


# Thijs
def load_qwen_bundle(config: PipelineConfig) -> QwenBundle:
    """Load the configured Qwen VL model using AutoConfig-based class dispatch."""
    print(f"Loading Qwen-VL ({config.qwen_vl_model_id})...")
    skip_allocator_warmup = os.environ.get("GRAPH_QWEN_SKIP_WARMUP", "1") == "1"
    use_cpu_offload = os.environ.get("GRAPH_QWEN_CPU_OFFLOAD", "0") == "1"
    offload_gpu_max_memory = os.environ.get("GRAPH_QWEN_GPU_MAX_MEMORY", "10GiB").strip() or "10GiB"
    offload_cpu_max_memory = os.environ.get("GRAPH_QWEN_CPU_MAX_MEMORY", "60GiB").strip() or "60GiB"
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass
    print(f"Qwen-VL pre-load CUDA memory: {_format_cuda_mem_snapshot()}")
    os.environ.setdefault("HF_HOME", config.hf_cache_dir)
    model_source = resolve_local_hf_model_path(config.qwen_vl_model_id, config.hf_cache_dir)
    if model_source != config.qwen_vl_model_id:
        print(f"Resolved local Qwen path: {model_source}")
    auto_config = AutoConfig.from_pretrained(
        model_source,
        local_files_only=config.local_files_only,
        cache_dir=config.hf_cache_dir,
        trust_remote_code=True,
    )
    model_type = str(getattr(auto_config, "model_type", "")).lower()
    model_id_lower = config.qwen_vl_model_id.lower()
    if model_type in {"qwen3_vl", "qwen3_vl_moe"} or ("qwen3" in model_type and "vl" in model_type):
        model_cls = Qwen3VLForConditionalGeneration
    else:
        raise ValueError(f"Unsupported Qwen VL model_type: {model_type}")

    load_kwargs = {
        "torch_dtype": torch.bfloat16,
        "attn_implementation": "sdpa",
        "local_files_only": config.local_files_only,
        "cache_dir": config.hf_cache_dir,
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
    }
    if use_cpu_offload:
        load_kwargs["device_map"] = "auto"
        load_kwargs["max_memory"] = {0: offload_gpu_max_memory, "cpu": offload_cpu_max_memory}
        load_mode_text = (
            "bf16 with CPU offload "
            f"(gpu_max={offload_gpu_max_memory}, cpu_max={offload_cpu_max_memory})"
        )
    else:
        load_kwargs["device_map"] = {"": 0}
        load_mode_text = "full bf16 on GPU 0 (no CPU offload)"
    import transformers.modeling_utils as hf_modeling_utils
    original_caching_allocator_warmup = getattr(hf_modeling_utils, "caching_allocator_warmup", None)
    if skip_allocator_warmup and original_caching_allocator_warmup is not None:
        print("Qwen-VL loader: skipping transformers caching_allocator_warmup")
        hf_modeling_utils.caching_allocator_warmup = lambda *args, **kwargs: None
    try:
        try:
            model = model_cls.from_pretrained(
                model_source,
                **load_kwargs,
            )
        except TypeError:
            load_kwargs.pop("attn_implementation", None)
            model = model_cls.from_pretrained(
                model_source,
                **load_kwargs,
            )
    finally:
        if skip_allocator_warmup and original_caching_allocator_warmup is not None:
            hf_modeling_utils.caching_allocator_warmup = original_caching_allocator_warmup
    processor = AutoProcessor.from_pretrained(
        model_source,
        local_files_only=config.local_files_only,
        cache_dir=config.hf_cache_dir,
        trust_remote_code=True,
    )
    is_qwen35_vl = ("3.5" in model_id_lower or "qwen3_5" in model_type) and "vl" in (model_type + model_id_lower)
    print(f"Qwen-VL load mode: {load_mode_text}")
    print(
        f"Qwen-VL visual caps: image_max_pixels={int(QWEN_IMAGE_MAX_PIXELS)} "
        f"video_max_pixels={int(QWEN_VIDEO_MAX_PIXELS)}"
    )
    print(f"Qwen-VL post-load CUDA memory: {_format_cuda_mem_snapshot()}")
    print(f"Qwen-VL ready. model_type={model_type}")
    return QwenBundle(model=model, processor=processor, model_type=model_type, is_qwen35_vl=is_qwen35_vl, config=config)


def release_qwen_bundle(bundle: Optional[QwenBundle]) -> None:
    """Release Qwen model resources aggressively before switching back to DA3."""
    if bundle is None:
        return
    print(f"[MemProfile] Qwen release: BEFORE  {_format_cuda_mem_snapshot()}")
    model = getattr(bundle, 'model', None)
    processor = getattr(bundle, 'processor', None)
    # Evict GPU weights to CPU FIRST: del + empty_cache alone does not reclaim the
    # CUDA memory if any reference lingers (accelerate hooks / autocast / closures),
    # which leaked ~2.4GiB per load/release cycle and OOM'd on the 3rd Qwen reload.
    try:
        if model is not None and hasattr(model, 'cpu'):
            model.cpu()
    except Exception:
        pass
    bundle.model = None
    bundle.processor = None
    del model
    del processor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass
    print(f"[MemProfile] Qwen release: AFTER   {_format_cuda_mem_snapshot()}")


def ensure_qwen_bundle_loaded(bundle: Optional[QwenBundle], config: PipelineConfig) -> QwenBundle:
    """Return a Qwen bundle with live model resources, reloading in place if needed."""
    if bundle is not None and getattr(bundle, "model", None) is not None and getattr(bundle, "processor", None) is not None:
        return bundle
    fresh_bundle = load_qwen_bundle(config)
    if bundle is None:
        return fresh_bundle
    bundle.model = fresh_bundle.model
    bundle.processor = fresh_bundle.processor
    bundle.model_type = fresh_bundle.model_type
    bundle.is_qwen35_vl = fresh_bundle.is_qwen35_vl
    bundle.config = fresh_bundle.config
    return bundle


# Thijs
def qwen_generate(bundle: QwenBundle, messages: List[dict], max_tokens: int, enable_reasoning: bool = False) -> str:
    """Run a Qwen generation pass and return the decoded response text."""
    prepared_messages = copy.deepcopy(messages)
    for message in prepared_messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for item in content:
            if not isinstance(item, dict):
                continue
            item_type = str(item.get("type", "")).strip().lower()
            if item_type == "image":
                item.setdefault("max_pixels", int(QWEN_IMAGE_MAX_PIXELS))
            elif item_type == "video":
                item.setdefault("max_pixels", int(QWEN_VIDEO_MAX_PIXELS))
    template_kwargs = {"tokenize": False, "add_generation_prompt": True}
    if enable_reasoning and bundle.is_qwen35_vl:
        template_kwargs["enable_thinking"] = True
    text = bundle.processor.apply_chat_template(prepared_messages, **template_kwargs)
    image_inputs, video_inputs, video_kwargs = process_vision_info(prepared_messages, return_video_kwargs=True)
    device = next(bundle.model.parameters()).device
    processor_kwargs = {"text": [text], "return_tensors": "pt"}
    if image_inputs:
        processor_kwargs["images"] = image_inputs
    if video_inputs:
        processor_kwargs["videos"] = video_inputs
        fps_value = video_kwargs.get("fps")
        if isinstance(fps_value, list) and len(fps_value) == 1:
            video_kwargs = dict(video_kwargs)
            video_kwargs["fps"] = fps_value[0]
        processor_kwargs.update(video_kwargs)
    inputs = bundle.processor(**processor_kwargs).to(device)
    with torch.inference_mode():
        generated = bundle.model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False, repetition_penalty=1.1)
    trimmed = generated[0][inputs.input_ids.shape[1] :]
    return strip_think_blocks(bundle.processor.decode(trimmed, skip_special_tokens=True).strip())


def image_to_data_url(rgb: np.ndarray) -> str:
    """Encode an RGB image as a JPEG data URL for Qwen messages."""
    image = Image.fromarray(rgb)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("utf-8")


def sampled_video_frame_indices(n_frames: int, stride: int = QWEN_ACTIVE_VIDEO_FRAME_STRIDE, max_frames: int = QWEN_ACTIVE_VIDEO_MAX_FRAMES) -> List[int]:
    """Return stride-sampled frame indices for active-clip Qwen video prompts."""
    if n_frames <= 0:
        return []
    stride = max(1, int(stride))
    indices = list(range(0, int(n_frames), stride))
    if indices[-1] != int(n_frames) - 1:
        indices.append(int(n_frames) - 1)
    if len(indices) <= int(max_frames):
        return indices
    positions = np.linspace(0, len(indices) - 1, num=int(max_frames), dtype=int).tolist()
    sampled = [indices[pos] for pos in positions]
    sampled[0] = indices[0]
    sampled[-1] = indices[-1]
    return sorted(set(int(idx) for idx in sampled))


def save_rgb_clip_as_mp4(
    rgb_list: Sequence[np.ndarray],
    out_path: Path,
    fps: int = QWEN_ACTIVE_VIDEO_FPS,
) -> Path:
    """Save one RGB clip as an MP4 video for Qwen temporal reasoning."""
    if not rgb_list:
        raise ValueError("Cannot save Qwen active video: rgb_list is empty.")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frames: List[np.ndarray] = []
    for frame in rgb_list:
        arr = np.asarray(frame)
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        frames.append(arr)
    if not frames:
        raise ValueError("Cannot save Qwen active video: no valid RGB frames.")
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (width, height),
    )
    try:
        for frame in frames:
            if frame.shape[:2] != (height, width):
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()
    if not out_path.exists() or out_path.stat().st_size == 0:
        raise RuntimeError(f"Failed to write Qwen active video: {out_path}")
    return out_path


def vlm_model_name(_bundle: QwenBundle) -> str:
    """Return the short model family name for logging."""
    return "qwen"


def vlm_generate(bundle: QwenBundle, messages: List[dict], max_tokens: int, enable_reasoning: bool = False) -> str:
    """Dispatch generation to the active Qwen backend."""
    return qwen_generate(bundle, messages, max_tokens=max_tokens, enable_reasoning=enable_reasoning)


def is_generic_scene_label(label: str) -> bool:
    """Return whether a label is too generic to trust as a graph node."""
    normalized = str(label or "").strip().lower()
    return normalized in (NON_INTERACTABLE | {"", "object", "unknown", "interactive element"})


def detection_is_valid(result: Optional[dict]) -> bool:
    """Return whether a detection contains usable non-generic labels."""
    if not result:
        return False
    return (
        not is_generic_scene_label(result.get("interactive_element", ""))
        and not is_generic_scene_label(result.get("parent_object", ""))
    )


def qwen_only_comparison(result: Optional[dict]) -> dict:
    """Build a compact comparison record for the single active VLM backend."""
    return {
        "selected_model": "qwen",
        "qwen": result,
    }


def make_multiscale_crop(
    image_rgb: np.ndarray,
    bbox: Sequence[float],
    mask: Optional[np.ndarray] = None,
    paddings: Tuple[float, float, float] = (40, 1.5, 2.5),
    outline: bool = True,
) -> np.ndarray:
    """Create OpenFunGraph-style small, medium, and large crops concatenated horizontally."""
    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    h, w = image_rgb.shape[:2]
    x1 = int(np.clip(x1, 0, w - 1))
    y1 = int(np.clip(y1, 0, h - 1))
    x2 = int(np.clip(max(x2, x1 + 1), 1, w))
    y2 = int(np.clip(max(y2, y1 + 1), 1, h))
    canvas = image_rgb.copy()

    if outline:
        if mask is not None and mask.any():
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        else:
            contours = [np.array([[[x1, y1]], [[x2, y1]], [[x2, y2]], [[x1, y2]]], dtype=np.int32)]
        cv2.drawContours(canvas, contours, -1, (255, 0, 0), 2)

    box_w = max(1, x2 - x1)
    box_h = max(1, y2 - y1)
    crops = []
    for padding in paddings:
        if isinstance(padding, float) and padding > 1.0:
            half_w = int(round(box_w * padding * 0.5))
            half_h = int(round(box_h * padding * 0.5))
            cx = int(round((x1 + x2) * 0.5))
            cy = int(round((y1 + y2) * 0.5))
            crop_x1 = max(0, cx - half_w)
            crop_y1 = max(0, cy - half_h)
            crop_x2 = min(w, cx + half_w)
            crop_y2 = min(h, cy + half_h)
        else:
            pad = int(round(float(padding)))
            crop_x1 = max(0, x1 - pad)
            crop_y1 = max(0, y1 - pad)
            crop_x2 = min(w, x2 + pad)
            crop_y2 = min(h, y2 + pad)
        crops.append(canvas[crop_y1:crop_y2, crop_x1:crop_x2])

    target_height = max(crop.shape[0] for crop in crops)
    resized = []
    for crop in crops:
        scale = target_height / max(1, crop.shape[0])
        target_width = max(1, int(round(crop.shape[1] * scale)))
        resized.append(cv2.resize(crop, (target_width, target_height), interpolation=cv2.INTER_LINEAR))
    stitched = np.concatenate(resized, axis=1)
    return cv2.resize(stitched, FULL_FRAME_QWEN_SIZE, interpolation=cv2.INTER_LINEAR)


def make_object_caption_view(
    image_rgb: np.ndarray,
    bbox: Sequence[float],
    mask: Optional[np.ndarray] = None,
    padding_scale: float = 1.6,
) -> np.ndarray:
    """Return a single padded RGB crop suited for object captioning.

    Unlike ``make_multiscale_crop`` (which builds a 3-panel small/medium/large
    triptych for fine-part identification), this returns ONE medium-context
    crop without contour outlining — better for whole-object captioning where
    we want the model to see the object plus a bit of surrounding context but
    not be biased by a drawn outline.
    """
    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    h, w = image_rgb.shape[:2]
    x1 = int(np.clip(x1, 0, w - 1))
    y1 = int(np.clip(y1, 0, h - 1))
    x2 = int(np.clip(max(x2, x1 + 1), 1, w))
    y2 = int(np.clip(max(y2, y1 + 1), 1, h))
    box_w = max(1, x2 - x1)
    box_h = max(1, y2 - y1)
    half_w = int(round(box_w * float(padding_scale) * 0.5))
    half_h = int(round(box_h * float(padding_scale) * 0.5))
    cx = int(round((x1 + x2) * 0.5))
    cy = int(round((y1 + y2) * 0.5))
    crop_x1 = max(0, cx - half_w)
    crop_y1 = max(0, cy - half_h)
    crop_x2 = min(w, cx + half_w)
    crop_y2 = min(h, cy + half_h)
    crop = image_rgb[crop_y1:crop_y2, crop_x1:crop_x2]
    if crop.size == 0:
        crop = image_rgb.copy()
    return cv2.resize(crop, FULL_FRAME_QWEN_SIZE, interpolation=cv2.INTER_LINEAR)


def detect_interaction_bbox(sample: dict, image_shape: Sequence[int]) -> Tuple[int, int, int, int]:
    """Estimate an interaction-focused bounding box from hand and object detections."""
    h, w = image_shape[:2]
    boxes = []
    for detections in (sample.get("hand_dets"), sample.get("obj_dets")):
        if detections is None:
            continue
        for row in detections:
            if len(row) < 5 or row[4] < DETECTION_THRESH:
                continue
            boxes.append(row[:4])
    if not boxes:
        return 0, 0, w, h
    boxes = np.asarray(boxes, dtype=np.float32)
    return (
        int(np.clip(np.min(boxes[:, 0]), 0, w - 1)),
        int(np.clip(np.min(boxes[:, 1]), 0, h - 1)),
        int(np.clip(np.max(boxes[:, 2]), 1, w)),
        int(np.clip(np.max(boxes[:, 3]), 1, h)),
    )

# Thijs: just modify
def prepare_full_frame_for_qwen(image_rgb: np.ndarray) -> np.ndarray:
    """Resize a frame to the standard full-frame Qwen input resolution."""
    return cv2.resize(image_rgb, FULL_FRAME_QWEN_SIZE, interpolation=cv2.INTER_LINEAR)


def detect_interaction_video(
    bundle: QwenBundle,
    active_frames: Sequence[np.ndarray],
    mode_name: str,
    video_path: Path,
    inactive_context_frames: Optional[Sequence[np.ndarray]] = None,
    sampled_frame_ids: Optional[Sequence[int]] = None,
) -> dict:
    """Ask Qwen to identify the manipulated element/object from an active-clip video."""
    inactive_context_frames = list(inactive_context_frames or [])
    n_inactive_shown = min(2, len(inactive_context_frames))
    sampled_frame_ids = [int(idx) for idx in (sampled_frame_ids or [])]
    prepared_active_frames = [prepare_full_frame_for_qwen(frame) for frame in active_frames]
    video_path = save_rgb_clip_as_mp4(prepared_active_frames, video_path, fps=QWEN_ACTIVE_VIDEO_FPS)
    print(
        f"[Qwen-Detect] mode={mode_name} saved active video -> {video_path} "
        f"({len(prepared_active_frames)} frames @ {QWEN_ACTIVE_VIDEO_FPS} fps)"
    )
    content = [{"type": "image", "image": image_to_data_url(frame)} for frame in inactive_context_frames[:n_inactive_shown]]
    video_payload = {"type": "video", "video": str(video_path)}
    if prepared_active_frames:
        # FFmpeg can lose the trailing frame during encoding, so the saved
        # MP4 sometimes contains one fewer frame than ``prepared_active_frames``.
        # qwen_vl_utils refuses with ValueError if ``nframes`` exceeds the
        # actual ``total_frames`` in the file, so cap against the real count
        # and align to qwen_vl_utils' FRAME_FACTOR=2 by rounding down to an
        # even number.
        requested = int(len(prepared_active_frames))
        actual_total = requested
        try:
            cap = cv2.VideoCapture(str(video_path))
            if cap.isOpened():
                probed = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                if probed > 0:
                    actual_total = probed
            cap.release()
        except Exception:  # noqa: BLE001
            pass
        nframes = min(requested, actual_total)
        # qwen_vl_utils requires nframes to be a multiple of FRAME_FACTOR=2.
        if nframes >= 2:
            nframes = (nframes // 2) * 2
        if nframes >= 2:
            video_payload["nframes"] = int(nframes)
    content.append(video_payload)
    sampled_note = (
        f"The video contains the active interaction sampled every {QWEN_ACTIVE_VIDEO_FRAME_STRIDE} frame(s). "
        f"Sampled active frame ids: {sampled_frame_ids}. "
        if sampled_frame_ids
        else f"The video contains the active interaction sampled every {QWEN_ACTIVE_VIDEO_FRAME_STRIDE} frame(s). "
    )
    inactive_context_note = (
        f"The first {n_inactive_shown} image(s) show the inactive scene immediately before the interaction. "
        if inactive_context_frames
        else ""
    )
    few_shots = [
        {
            "observation": "A hand pulls a drawer using the visible drawer handle.",
            "interactive_element": "drawer handle",
            "parent_object": "drawer",
            "relation_scope": "local",
            "why": "The handle is the touched functional part, and the manipulated parent object is the moving drawer, not the cabinet bank around it.",
        },
        {
            "observation": "A hand opens a cabinet by gripping its handle.",
            "interactive_element": "cabinet handle",
            "parent_object": "cabinet door",
            "relation_scope": "local",
            "why": "The touched part is the handle, and the manipulated parent object is the moving cabinet door, not the whole cabinet body.",
        },
        {
            "observation": "A hand grasps a spoon and carries it out of or away from a bowl.",
            "interactive_element": "spoon handle",
            "parent_object": "spoon",
            "relation_scope": "local",
            "why": "The spoon is the carried rigid object. The bowl is only nearby context or support, not the manipulated parent object.",
        },
        {
            "observation": "A hand presses a light switch to control a ceiling light.",
            "interactive_element": "light switch",
            "parent_object": "ceiling light",
            "relation_scope": "remote",
            "why": "The touched control affects a separate target object's state, so this is a remote functional relation.",
        },
    ]
    shared_rules = (
        "Think in terms of a functional scene graph. "
        "Identify the touched interactive element and the parent object being manipulated or controlled THROUGH that element. "
        "A single parent object can have multiple interactive elements with different functions, so choose the part that is ACTUALLY touched in this clip, not merely the most salient object in view. "
        "The interactive_element is the SMALL graspable functional part the hand makes direct contact with — a handle, knob, pull, grip, button, switch, lever, latch, or rim — NOT the large panel, door, drawer front, lid, or body it is attached to. "
        "It must usually be a DIFFERENT, smaller label than the parent_object (e.g. interactive_element 'cabinet handle' with parent_object 'cabinet door'; NEVER interactive_element 'cabinet door' with parent_object 'cabinet door'). "
        "A hinged door or sliding drawer is opened by gripping its handle, so when a hand opens one, name the handle as the interactive_element EVEN IF the door/drawer is already swung open and the handle is small, foreshortened, or partly hidden — infer the handle from how the part is being opened rather than labelling the whole moving body as the touched element. "
        "Reason about the LOCAL functional relationship: what object does this touched part belong to, and what object is being manipulated through it? "
        "A functional relationship is not a spatial relationship: inside, on, near, touching, holding, stored in, supported by, or next to are NOT valid parent-object choices by themselves. "
        "Use the video to decide whether there is REAL RELATIVE MOTION between a local part and the rest of the object. "
        "If a small object is simply carried through space as one rigid piece, choose the carried object itself as the parent object and do NOT name a lid/cap/door unless the video clearly shows that part moving relative to the rest of the object. "
        "If a utensil or tool is grasped while resting in a bowl, rack, drawer, or holder, the utensil/tool is the parent object if IT is what moves with the hand; the container/support is only context unless it also moves. "
        "Only name a lid, cap, door, drawer, panel, or flap when the active video shows that part moving relative to a larger body. "
        "For local functional relations, choose the specific immediate moving part the touched element is rigidly attached to, OR the carried object itself if the whole object moves together. "
        "DO NOT name the larger static appliance / cabinet / chassis the moving part hinges on. "
        "Examples: drawer handle -> drawer (NOT cabinet); refrigerator door handle -> refrigerator door (NOT refrigerator); cabinet handle -> cabinet door (NOT cabinet); pot lid handle -> pot lid only if the lid moves relative to the pot; spoon handle -> spoon; mug handle -> mug; container rim -> container. "
        "For remote functional relations, choose the object whose state or function is controlled by the interactive element. Example: button -> ceiling light. "
        "Do not choose a nearby container, support, destination, or scene context unless it is the actual local parent object or the actual remote controlled target. "
        f"Few-shot examples: {json.dumps(few_shots)}. "
        "Use specific labels like 'spoon handle', 'spoon', 'drawer handle', 'drawer', 'button', 'ceiling light'. "
        "Avoid generic labels like hand, person, room, floor, wall, appliance, or object unless no better label is possible. "
    )
    if bundle.config.use_cot and bundle.is_qwen35_vl:
        prompt_text = (
            "A hand is touching or manipulating something in these egocentric observations. "
            f"These inputs were prepared with the '{mode_name}' view. "
            + inactive_context_note
            + sampled_note
            + shared_rules
            + "Reason carefully from the temporal evidence: first identify the touched part, then identify the parent object functionally manipulated through that part, then decide whether the relation is local or remote. "
            + "Output ONLY the final JSON with keys: "
            "\"interactive_element\" (specific touched part, 1-4 lowercase words), "
            "\"parent_object\" (object node linked to the interactive element, 1-4 lowercase words), "
            "\"relation_scope\" ('local' or 'remote'), "
            "\"confidence\" (0.0-1.0), and "
            "\"reasoning\" (short string explaining the functional relationship). "
            "Return exactly one JSON object and do not wrap it in prose."
        )
        max_tokens = bundle.config.cot_max_tokens
        enable_reasoning = True
    else:
        prompt_text = (
            "A hand is touching or manipulating something in these egocentric observations. "
            f"These inputs were prepared with the '{mode_name}' view. "
            + inactive_context_note
            + sampled_note
            + shared_rules
            + "Internally decide: what part is touched, what object is manipulated through that part, and whether that pair is local or remote. "
            + "Output ONLY JSON with keys: "
            "\"interactive_element\" (specific touched part, 1-4 lowercase words), "
            "\"parent_object\" (object node linked to the interactive element, 1-4 lowercase words), "
            "\"relation_scope\" ('local' or 'remote'), "
            "\"confidence\" (0.0-1.0), and "
            "\"reasoning\" (short string explaining the functional relationship). "
            "Return exactly one JSON object and do not wrap it in prose."
        )
        max_tokens = bundle.config.qwen_max_tokens
        enable_reasoning = False
    content.append({"type": "text", "text": prompt_text})
    raw = vlm_generate(
        bundle,
        [{"role": "user", "content": content}],
        max_tokens=max_tokens,
        enable_reasoning=enable_reasoning,
    )
    payload = safe_json_loads(raw)
    parent_object = str(
        payload.get(
            "parent_object",
            payload.get("manipulated_object", payload.get("functional_object", "unknown")),
        )
    ).strip().lower()
    relation_scope = str(payload.get("relation_scope", "local")).strip().lower()
    return {
        "interactive_element": str(payload.get("interactive_element", "object")).strip().lower(),
        "parent_object": parent_object,
        "relation_scope": relation_scope if relation_scope in {"local", "remote"} else "local",
        "confidence": float(payload.get("confidence", 0.0)),
        "reasoning": str(payload.get("reasoning", "")).strip(),
        "raw": raw,
        "mode": mode_name,
        "model": vlm_model_name(bundle),
        "video_path": str(video_path),
        "sampled_frame_ids": sampled_frame_ids,
    }


def infer_interaction_hierarchy(
    bundle: QwenBundle,
    frame_rgb: np.ndarray,
    interaction_bbox: Sequence[int],
    inactive_frame_rgb: Optional[np.ndarray] = None,
    later_frame_rgb: Optional[np.ndarray] = None,
    later_interaction_bbox: Optional[Sequence[int]] = None,
    element_hint: str = "",
    object_hint: str = "",
) -> dict:
    """Infer semantic part/object labels from an interaction-centric crop on the first active frame."""
    crop_box = expand_bbox_xyxy(interaction_bbox, frame_rgb.shape, scale=1.8, padding=24)
    crop_x1, crop_y1, crop_x2, crop_y2 = [int(v) for v in crop_box]
    crop_rgb = np.asarray(frame_rgb, dtype=np.uint8)[crop_y1:crop_y2, crop_x1:crop_x2]
    if crop_rgb.size == 0:
        crop_rgb = np.asarray(frame_rgb, dtype=np.uint8)
        crop_box = (0, 0, frame_rgb.shape[1], frame_rgb.shape[0])
    later_crop_box = None
    later_crop_rgb = None
    if later_frame_rgb is not None and later_interaction_bbox is not None:
        later_crop_box = expand_bbox_xyxy(later_interaction_bbox, later_frame_rgb.shape, scale=1.8, padding=24)
        lx1, ly1, lx2, ly2 = [int(v) for v in later_crop_box]
        later_crop_rgb = np.asarray(later_frame_rgb, dtype=np.uint8)[ly1:ly2, lx1:lx2]
        if later_crop_rgb.size == 0:
            later_crop_rgb = np.asarray(later_frame_rgb, dtype=np.uint8)
            later_crop_box = (0, 0, later_frame_rgb.shape[1], later_frame_rgb.shape[0])

    hints = []
    if element_hint:
        hints.append(f"interactive element hint: '{element_hint}'")
    if object_hint:
        hints.append(f"parent object hint: '{object_hint}'")
    hint_text = " ".join(hints) if hints else "No label hints are provided."

    few_shots = [
        {
            "observation": "The crop shows a hand pulling a drawer by its handle.",
            "interactive_element": "drawer handle",
            "parent_object": "drawer",
            "why": "The handle is the touched part, and the manipulated local object instance is the drawer itself.",
        },
        {
            "observation": "The crop shows a hand opening a cabinet by gripping a handle.",
            "interactive_element": "cabinet handle",
            "parent_object": "cabinet door",
            "why": "The touched part is the handle, and the manipulated local object is the moving cabinet door.",
        },
        {
            "observation": "The crop shows a hand grasping a spoon resting in a bowl and lifting it away.",
            "interactive_element": "spoon handle",
            "parent_object": "spoon",
            "why": "The bowl is just context or support; the spoon is the carried manipulated object.",
        },
    ]
    prompt = (
        ("The first image shows the inactive scene immediately before interaction. " if inactive_frame_rgb is not None else "")
        + "The next image is the first active frame. "
        + "The following image is a local crop centered on the touched region of that first active frame. "
        + ("The final two images show a later active state of the same interaction, where the manipulated local object may be more visible. " if later_crop_rgb is not None else "")
        + hint_text
        + " Identify the specific touched part and the specific single manipulated local object instance it belongs to. "
        "This prompt is for local geometry only, not remote functional targets. "
        "Think in OpenFunGraph-style local functional terms: what small part is directly touched, and what single local object is manipulated through that part. "
        "If the hand is grasping a movable tool or utensil taken from storage, choose the tool itself as the parent object. Example: spoon handle -> spoon, not the bowl, drawer, dish rack, or holder around it. "
        "If the touched part is an appliance/cabinet door handle, choose the moving door/drawer, not the whole appliance/cabinet. Example: refrigerator door handle -> refrigerator door. "
        "Do not choose a nearby support or container just because the object starts inside or on it. If the hand lifts the spoon from a bowl, the parent object is the spoon. "
        "Do not describe the whole cabinet bank, wall, counter, or room; choose only the single parent object instance directly attached to or grasped with the touched part. "
        f"Few-shot examples: {json.dumps(few_shots)}. "
        "Output ONLY JSON with keys: "
        "\"interactive_element\" (1-4 lowercase words), "
        "\"parent_object\" (1-4 lowercase words), "
        "\"confidence\" (0.0-1.0), and "
        "\"reasoning\" (short string explaining the local functional relationship)."
    )
    raw = vlm_generate(
        bundle,
        [
            {
                "role": "user",
                "content": (
                    ([{"type": "image", "image": image_to_data_url(prepare_full_frame_for_qwen(inactive_frame_rgb))}] if inactive_frame_rgb is not None else [])
                    + [
                        {"type": "image", "image": image_to_data_url(prepare_full_frame_for_qwen(frame_rgb))},
                        {"type": "image", "image": image_to_data_url(prepare_full_frame_for_qwen(crop_rgb))},
                    ]
                    + (
                        [
                            {"type": "image", "image": image_to_data_url(prepare_full_frame_for_qwen(later_frame_rgb))},
                            {"type": "image", "image": image_to_data_url(prepare_full_frame_for_qwen(later_crop_rgb))},
                        ]
                        if later_crop_rgb is not None and later_frame_rgb is not None
                        else []
                    )
                    + [{"type": "text", "text": prompt}]
                ),
            }
        ],
        max_tokens=bundle.config.qwen_max_tokens,
        enable_reasoning=bool(bundle.config.use_cot and bundle.is_qwen35_vl),
    )
    payload = safe_json_loads(raw)
    return {
        "interactive_element": str(payload.get("interactive_element", element_hint or "interactive element")).strip().lower(),
        "parent_object": str(payload.get("parent_object", object_hint or "object")).strip().lower(),
        "interactive_element_box_xyxy": None,
        "parent_object_box_xyxy": None,
        "interaction_crop_box_xyxy": tuple(int(v) for v in crop_box),
        "later_interaction_crop_box_xyxy": tuple(int(v) for v in later_crop_box) if later_crop_box is not None else None,
        "confidence": float(payload.get("confidence", 0.0)),
        "reasoning": str(payload.get("reasoning", "")).strip(),
        "raw": raw,
    }


def infer_inactive_instance_prompts(
    bundle: QwenBundle,
    inactive_frame_rgb: np.ndarray,
    active_frame_rgb: np.ndarray,
    interaction_bbox: Sequence[int],
    later_frame_rgb: Optional[np.ndarray] = None,
    later_interaction_bbox: Optional[Sequence[int]] = None,
    element_hint: str = "",
    object_hint: str = "",
    shares_rigid_motion: bool = False,
) -> dict:
    """Ask Qwen for inactive-scene semantic reasoning only; no spatial prompts are requested."""
    crop_box = expand_bbox_xyxy(interaction_bbox, active_frame_rgb.shape, scale=1.8, padding=24)
    crop_x1, crop_y1, crop_x2, crop_y2 = [int(v) for v in crop_box]
    crop_rgb = np.asarray(active_frame_rgb, dtype=np.uint8)[crop_y1:crop_y2, crop_x1:crop_x2]
    if crop_rgb.size == 0:
        crop_rgb = np.asarray(active_frame_rgb, dtype=np.uint8)
        crop_box = (0, 0, active_frame_rgb.shape[1], active_frame_rgb.shape[0])

    later_crop_box = None
    later_crop_rgb = None
    if later_frame_rgb is not None and later_interaction_bbox is not None:
        later_crop_box = expand_bbox_xyxy(later_interaction_bbox, later_frame_rgb.shape, scale=1.8, padding=24)
        lx1, ly1, lx2, ly2 = [int(v) for v in later_crop_box]
        later_crop_rgb = np.asarray(later_frame_rgb, dtype=np.uint8)[ly1:ly2, lx1:lx2]
        if later_crop_rgb.size == 0:
            later_crop_rgb = np.asarray(later_frame_rgb, dtype=np.uint8)
            later_crop_box = (0, 0, later_frame_rgb.shape[1], later_frame_rgb.shape[0])

    prompt = (
        "The first image is the beginning of the inactive clip, before the interaction starts. "
        "The second image is the first active frame. "
        "The third image is a local crop around the touched region in that first active frame. "
        + ("The final two images show a later active state where the interacted object may be more visible. " if later_crop_rgb is not None else "")
        + f"Interactive element hint: '{element_hint}'. Parent object hint: '{object_hint}'. "
        + ("The interactive element and parent object likely share the same rigid motion during the interaction. " if shares_rigid_motion else "")
        + "On the FIRST image only, identify the exact inactive instances that correspond to the interaction. "
        "For drawers, choose the single drawer that will be opened, not the whole cabinet bank. "
        "Output ONLY JSON with keys: "
        "\"confidence\" (0.0-1.0), and "
        "\"reasoning\" (short string)."
    )
    raw = vlm_generate(
        bundle,
        [
            {
                "role": "user",
                "content": (
                    [
                        {"type": "image", "image": image_to_data_url(prepare_full_frame_for_qwen(inactive_frame_rgb))},
                        {"type": "image", "image": image_to_data_url(prepare_full_frame_for_qwen(active_frame_rgb))},
                        {"type": "image", "image": image_to_data_url(prepare_full_frame_for_qwen(crop_rgb))},
                    ]
                    + (
                        [
                            {"type": "image", "image": image_to_data_url(prepare_full_frame_for_qwen(later_frame_rgb))},
                            {"type": "image", "image": image_to_data_url(prepare_full_frame_for_qwen(later_crop_rgb))},
                        ]
                        if later_crop_rgb is not None and later_frame_rgb is not None
                        else []
                    )
                    + [{"type": "text", "text": prompt}]
                ),
            }
        ],
        max_tokens=bundle.config.qwen_max_tokens,
        enable_reasoning=bool(bundle.config.use_cot and bundle.is_qwen35_vl),
    )
    payload = safe_json_loads(raw)
    return {
        "inactive_interactive_element_box_xyxy": None,
        "inactive_parent_object_box_xyxy": None,
        "inactive_interactive_element_point_xy": None,
        "inactive_parent_object_point_xy": None,
        "confidence": float(payload.get("confidence", 0.0)),
        "reasoning": str(payload.get("reasoning", "")).strip(),
        "raw": raw,
    }


def infer_relationship(
    bundle: QwenBundle,
    elem_label: str,
    obj_label: str,
    pre_frame: np.ndarray,
    post_frame: np.ndarray,
    existing_node_labels: Sequence[str],
    element_desc: str = "",
    object_desc: str = "",
) -> dict:
    """Infer a functional affordance relation for one element/object pair."""
    desc_prefix = f"Interactive element description: '{element_desc}'. Parent object description: '{object_desc}'. "
    if bundle.config.use_cot and bundle.is_qwen35_vl:
        instructions = (
            f"Interactive element: '{elem_label}'. Parent object: '{obj_label}'. "
            + desc_prefix +
            "Image 1 is before interaction. Image 2 is after interaction. "
            "Infer ONLY a functional affordance relation from the part/control to the object. "
            "Follow the functional-scene-graph definition: local relations are rigid part-to-object relations, and remote relations are control-to-target relations. "
            f"Allowed relations: {json.dumps(list(FUNCTIONAL_RELATIONS))}. "
            "Do not output any spatial or geometric relation such as inside, on, over, attached to, touching, intersecting, or near. "
            "Do not infer storage, support, contact, or containment relations such as spoon in bowl, spoon in rack, cup in holder, or bottle on shelf. "
            "If this is not a clear affordance-style part/object pair, output relationship='unknown'. "
            "Reason step by step before producing the final JSON. "
            "Then output ONLY JSON with keys: \"relationship\" (str), \"confidence\" (float), and \"reasoning\" (short string)."
        )
        max_tokens = bundle.config.cot_max_tokens
        enable_reasoning = True
    else:
        instructions = (
            f"Interactive element: '{elem_label}'. Parent object: '{obj_label}'. "
            + desc_prefix +
            "Image 1 is before interaction. Image 2 is after interaction. "
            "Infer ONLY a functional affordance relation from the part/control to the object. "
            "Follow the functional-scene-graph definition: local relations are rigid part-to-object relations, and remote relations are control-to-target relations. "
            f"Allowed relations: {json.dumps(list(FUNCTIONAL_RELATIONS))}. "
            "Do not output any spatial or geometric relation such as inside, on, over, attached to, touching, intersecting, or near. "
            "Do not infer storage, support, contact, or containment relations such as spoon in bowl, spoon in rack, cup in holder, or bottle on shelf. "
            "If this is not a clear affordance-style part/object pair, output relationship='unknown'. "
            "Output ONLY JSON with keys: \"relationship\" (str), \"confidence\" (float), and \"reasoning\" (short string)."
        )
        max_tokens = bundle.config.qwen_max_tokens
        enable_reasoning = False

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_to_data_url(prepare_full_frame_for_qwen(pre_frame))},
                {"type": "image", "image": image_to_data_url(prepare_full_frame_for_qwen(post_frame))},
                {"type": "text", "text": instructions},
            ],
        }
    ]
    raw = vlm_generate(bundle, messages, max_tokens=max_tokens, enable_reasoning=enable_reasoning)
    payload = safe_json_loads(raw)
    relationship = normalize_functional_relation(str(payload.get("relationship", "unknown")))
    return {
        "relationship": relationship,
        "confidence": float(payload.get("confidence", 0.0)),
        "reasoning": str(payload.get("reasoning", "")).strip() or raw,
        "edge_type": "functional",
        "parent_object": obj_label,
    }


def infer_motion_coupling(
    bundle: QwenBundle,
    elem_label: str,
    obj_label: str,
    pre_frame: np.ndarray,
    post_frame: np.ndarray,
    element_desc: str = "",
    object_desc: str = "",
) -> dict:
    """Infer the global interaction type plus the current-pair motion relation."""
    desc_prefix = f"Interactive element description: '{element_desc}'. Parent object description: '{object_desc}'. "
    few_shots = [
        {
            "interactive_element": "spoon handle",
            "parent_object": "spoon",
            "interaction_type": "rigid_transport",
            "shares_rigid_motion": True,
            "moving_part_label": "spoon",
            "why": "The spoon is a free rigid object being carried. The handle and spoon body move together as one rigid body.",
        },
        {
            "interactive_element": "mug handle",
            "parent_object": "mug",
            "interaction_type": "rigid_transport",
            "shares_rigid_motion": True,
            "moving_part_label": "mug",
            "why": "The mug is a free rigid object during carrying; handle and mug body share one 6-DoF motion.",
        },
        {
            "interactive_element": "cabinet handle",
            "parent_object": "cabinet door",
            "interaction_type": "articulated_object",
            "shares_rigid_motion": True,
            "moving_part_label": "cabinet door",
            "why": "This is an articulated object. The handle is fixed to the cabinet door, and that moving door rotates around a hinge relative to the cabinet frame.",
        },
        {
            "interactive_element": "drawer handle",
            "parent_object": "drawer",
            "interaction_type": "articulated_object",
            "shares_rigid_motion": True,
            "moving_part_label": "drawer",
            "why": "This is an articulated object. The handle and drawer front translate together as one moving part relative to the cabinet body.",
        },
        {
            "interactive_element": "refrigerator door handle",
            "parent_object": "refrigerator",
            "interaction_type": "articulated_object",
            "shares_rigid_motion": False,
            "moving_part_label": "refrigerator door",
            "why": "The overall interaction is articulated, but the current object label names the static appliance body. The moving part is the refrigerator door.",
        },
        {
            "interactive_element": "microwave door handle",
            "parent_object": "microwave",
            "interaction_type": "articulated_object",
            "shares_rigid_motion": False,
            "moving_part_label": "microwave door",
            "why": "The appliance body stays still. The moving articulated part is the microwave door.",
        },
        {
            "interactive_element": "button",
            "parent_object": "microwave",
            "interaction_type": "remote_functional",
            "shares_rigid_motion": False,
            "moving_part_label": None,
            "why": "Pressing the button triggers a function but does not create a shared rigid motion with the microwave body.",
        },
        {
            "interactive_element": "light switch",
            "parent_object": "ceiling light",
            "interaction_type": "remote_functional",
            "shares_rigid_motion": False,
            "moving_part_label": None,
            "why": "The switch functionally controls the light, but the two named entities do not move together as one rigid body.",
        },
    ]
    instructions = (
        f"Interactive element: '{elem_label}'. Parent object: '{obj_label}'. "
        + desc_prefix
        + "Image 1 is before interaction. Image 2 is after interaction. "
        "Classify the OVERALL interaction into exactly one of three cases: "
        "remote_functional, rigid_transport, articulated_object. "
        "DEFINITIONS: "
        "remote_functional = the element functionally affects another object but there is no shared object motion (e.g. button -> light). "
        "rigid_transport = a free rigid object is carried through space as one piece (e.g. spoon, mug, bowl, knife, pot). "
        "articulated_object = a moving part rotates around a hinge or slides along a track relative to a larger static parent body (doors, drawers, lids, panels, covers). "
        "HARD DECISION RULES (apply in order, top wins): "
        "(R1) If the parent_object name contains any of {door, drawer, lid, cover, panel, hood, hatch, flap, gate}, the interaction_type is ALWAYS articulated_object — this overrides any visual ambiguity. "
        "(R2) If the parent_object name names a free-standing small carried item (spoon, fork, knife, mug, cup, bowl, plate, pot, pan, bottle, can, jar, container) and the element is a fixed part of it, the interaction_type is rigid_transport. Such objects are NEVER articulated_object unless the label explicitly names a moving part like door, drawer, lid, cap, or flap. "
        "(R3) If the element is a control (button, switch, knob, dial, lever) on an appliance body and there is no shared bulk motion, the interaction_type is remote_functional. "
        "(R4) Otherwise judge by definitions above. "
        "Next, judge the CURRENT element/object pair only: do THESE TWO NAMED ENTITIES share the same rigid motion? "
        "For articulated_object pairs, ALSO output moving_part_label: the immediate moving rigid part being manipulated. "
        "If the current object label already names the moving part, set moving_part_label equal to it. "
        "If the current object label names the static parent body instead, set moving_part_label to the moving child part (e.g. refrigerator door, microwave door, cabinet door, drawer, oven door). "
        f"Few-shot examples: {json.dumps(few_shots)}. "
        "Output ONLY JSON with keys: \"interaction_type\" (one of remote_functional, rigid_transport, articulated_object), \"shares_rigid_motion\" (bool), \"moving_part_label\" (string or null), \"confidence\" (float), and \"reasoning\" (short string). The reasoning string MUST cite which rule (R1/R2/R3/R4) drove the interaction_type choice."
    )
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_to_data_url(prepare_full_frame_for_qwen(pre_frame))},
                {"type": "image", "image": image_to_data_url(prepare_full_frame_for_qwen(post_frame))},
                {"type": "text", "text": instructions},
            ],
        }
    ]
    raw = vlm_generate(
        bundle,
        messages,
        max_tokens=bundle.config.qwen_max_tokens,
        enable_reasoning=bool(bundle.config.use_cot and bundle.is_qwen35_vl),
    )
    payload = safe_json_loads(raw)
    return apply_motion_coupling_guardrails(payload, elem_label, obj_label, raw)


def infer_sam_seed_endpoint(
    bundle: QwenBundle,
    clip: ClipInfo,
    rgb_list: Sequence[np.ndarray],
    element_label: str,
    object_label: str,
    motion_coupling: Optional[dict] = None,
    previous_inactive_clip: Optional[ClipInfo] = None,
) -> dict:
    """Ask Qwen whether SAM should seed from the beginning, middle, or end of the active clip."""
    default_choice = {
        "seed_choice": "beginning",
        "confidence": 0.0,
        "reasoning": "",
        "raw": "",
    }
    if not rgb_list:
        return dict(default_choice)

    begin_rgb = prepare_full_frame_for_qwen(np.asarray(rgb_list[0], dtype=np.uint8))
    middle_idx = max(0, min(len(rgb_list) - 1, len(rgb_list) // 2))
    middle_rgb = prepare_full_frame_for_qwen(np.asarray(rgb_list[middle_idx], dtype=np.uint8))
    end_rgb = prepare_full_frame_for_qwen(np.asarray(rgb_list[-1], dtype=np.uint8))
    content: List[dict] = []
    inactive_note = ""
    if previous_inactive_clip is not None and previous_inactive_clip.samples:
        inactive_sample = None
        for sample in reversed(previous_inactive_clip.samples):
            if sample.get("rgb_valid", True):
                inactive_sample = sample
                break
        if inactive_sample is None:
            inactive_sample = previous_inactive_clip.samples[-1]
        bgr = cv2.imread(str(inactive_sample["rgb_path"]))
        if bgr is not None:
            inactive_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            content.append({"type": "image", "image": image_to_data_url(prepare_full_frame_for_qwen(inactive_rgb))})
            inactive_note = "Image 1 is the last inactive frame before manipulation. "
    content.extend(
        [
            {"type": "image", "image": image_to_data_url(begin_rgb)},
            {"type": "image", "image": image_to_data_url(middle_rgb)},
            {"type": "image", "image": image_to_data_url(end_rgb)},
        ]
    )
    qwen_joint_type = str((motion_coupling or {}).get("joint_type", "")).strip().lower() or "unknown"
    few_shots = [
        {
            "interactive_element": "drawer handle",
            "moving_object": "drawer",
            "joint_type": "prismatic",
            "choice": "end",
            "why": "A drawer is most fully revealed at the end, so the moving drawer body is easier to segment than near the beginning or middle when it is still partly inside the cabinet.",
        },
        {
            "interactive_element": "cabinet handle",
            "moving_object": "cabinet door",
            "joint_type": "revolute",
            "choice": "beginning",
            "why": "A cabinet door is usually least distorted near the beginning; middle or end can create foreshortening, overlap with neighboring doors, and confusing silhouettes.",
        },
        {
            "interactive_element": "hand",
            "moving_object": "bowl",
            "joint_type": "rigid",
            "choice": "end",
            "why": "There are several identical bowls next to/stacked with each other, so at the beginning the target bowl is ambiguous and SAM3 may seed the WRONG instance. By the end the manipulated bowl has been lifted/carried AWAY from its identical neighbors, so it is spatially separated and unambiguous to segment as the correct single instance.",
        },
    ]
    prompt = (
        f"{inactive_note}"
        "The next image is the BEGINNING of the active clip. "
        "The following image is the MIDDLE of the active clip. "
        "The final image is the END of the active clip. "
        f"We are segmenting the moving object/part '{object_label}' that is manipulated via '{element_label}'. "
        f"Qwen's current motion guess is joint_type='{qwen_joint_type}'. "
        "Choose exactly one SAM seed location: 'beginning', 'middle', or 'end'. "
        "Pick the endpoint where the MOVING OBJECT/PART itself is easiest to segment as one correct instance with the most useful visual information. "
        "Prefer the view where the manipulated part is most visible, least occluded, and least likely to merge with neighboring static structure. "
        "For sliding drawers/trays, the end is often better because the moved body becomes revealed only after translation. "
        "For revolving cabinet/fridge doors, the beginning is often better because later rotation can create foreshortening, overlap, and visual distortion. "
        "IMPORTANT: if there are MULTIPLE SIMILAR/IDENTICAL objects of the same kind near the target (e.g. several bowls, cups, plates, or bottles together), prefer the 'end', because by then the manipulated instance has been carried away and is spatially SEPARATED from its look-alikes, so SAM3 seeds the correct single instance instead of an ambiguous merged/wrong one. "
        "Use 'middle' only if it clearly offers the best balance of visibility and low distortion. "
        f"Few-shot examples: {json.dumps(few_shots)}. "
        "Reason step by step internally, but output ONLY one JSON object with keys: "
        "\"seed_choice\" ('beginning', 'middle', or 'end'), \"confidence\" (0.0-1.0), and \"reasoning\" (short string)."
    )
    content.append({"type": "text", "text": prompt})
    raw = vlm_generate(
        bundle,
        [{"role": "user", "content": content}],
        max_tokens=bundle.config.cot_max_tokens if (bundle.config.use_cot and bundle.is_qwen35_vl) else bundle.config.qwen_max_tokens,
        enable_reasoning=bool(bundle.config.use_cot and bundle.is_qwen35_vl),
    )
    payload = safe_json_loads(raw) or {}
    choice = str(payload.get("seed_choice", "")).strip().lower()
    if choice in {"start", "first", "begin", "beginning"}:
        choice = "beginning"
    elif choice in {"middle", "mid", "center", "centre"}:
        choice = "middle"
    elif choice in {"end", "last", "final"}:
        choice = "end"
    else:
        choice = "beginning"
    try:
        confidence = float(payload.get("confidence", 0.0))
    except Exception:
        confidence = 0.0
    return {
        "seed_choice": choice,
        "confidence": confidence,
        "reasoning": str(payload.get("reasoning", "")).strip() or raw,
        "raw": raw,
    }


def infer_element_seed_frame(
    bundle: QwenBundle,
    active_rgb_list: Sequence[np.ndarray],
    element_label: str,
    object_label: str,
) -> dict:
    """Ask Qwen whether to seed the element (handle) at the FIRST or LAST active frame.

    Like the object seed, the element is now seeded from the ACTIVE clip. Qwen is
    shown the BEGINNING / MIDDLE / END of the active clip but must pick a BINARY
    SAM seed location — 'beginning' (first active frame) or 'end' (last active
    frame) — choosing where the element is the cleanest single instance to segment
    (handle visible/separated, least occluded by the gripping hand). Defaults to
    'beginning'.
    """
    default_choice = {"seed_choice": "beginning", "confidence": 0.0, "reasoning": "", "raw": ""}
    if active_rgb_list is None or len(active_rgb_list) == 0:
        return dict(default_choice)
    n = len(active_rgb_list)
    begin_rgb = prepare_full_frame_for_qwen(np.asarray(active_rgb_list[0], dtype=np.uint8))
    middle_rgb = prepare_full_frame_for_qwen(np.asarray(active_rgb_list[n // 2], dtype=np.uint8))
    end_rgb = prepare_full_frame_for_qwen(np.asarray(active_rgb_list[-1], dtype=np.uint8))
    content = [
        {"type": "image", "image": image_to_data_url(begin_rgb)},
        {"type": "image", "image": image_to_data_url(middle_rgb)},
        {"type": "image", "image": image_to_data_url(end_rgb)},
    ]
    prompt = (
        "These three images are the BEGINNING, MIDDLE, and END of the active manipulation clip "
        f"where a hand manipulates the '{object_label}' via its '{element_label}'. "
        f"We must seed SAM3 on the '{element_label}' at EITHER the first (beginning) OR the last (end) active frame. "
        "Choose the ONE binary location where the "
        f"'{element_label}' is EASIEST to segment as a single correct instance: most visible, largest/most frontal, "
        "least occluded by the gripping hand, and least likely to merge with a neighbouring identical part. "
        "Often the beginning is best because the hand has not yet fully wrapped the handle; but if the part is only "
        "clearly separated from look-alikes (or only revealed) after the motion, choose the end. "
        "Reason step by step internally, but output ONLY one JSON object with keys: "
        "\"seed_choice\" ('beginning' or 'end'), \"confidence\" (0.0-1.0), and \"reasoning\" (short string)."
    )
    content.append({"type": "text", "text": prompt})
    raw = vlm_generate(
        bundle,
        [{"role": "user", "content": content}],
        max_tokens=bundle.config.cot_max_tokens if (bundle.config.use_cot and bundle.is_qwen35_vl) else bundle.config.qwen_max_tokens,
        enable_reasoning=bool(bundle.config.use_cot and bundle.is_qwen35_vl),
    )
    payload = safe_json_loads(raw) or {}
    choice = str(payload.get("seed_choice", "")).strip().lower()
    if choice in {"end", "last", "final"}:
        choice = "end"
    else:
        choice = "beginning"
    try:
        confidence = float(payload.get("confidence", 0.0))
    except Exception:
        confidence = 0.0
    return {
        "seed_choice": choice,
        "confidence": confidence,
        "reasoning": str(payload.get("reasoning", "")).strip() or raw,
        "raw": raw,
    }


def choose_keyframes(clip: ClipInfo, rgb_list: Sequence[np.ndarray]) -> Tuple[List[np.ndarray], List[dict]]:
    """Pick representative keyframes and matching sample metadata for Qwen."""
    if not rgb_list:
        return [], []
    valid_indices = [idx for idx, sample in enumerate(clip.samples) if sample.get("rgb_valid", True)]
    source_indices = valid_indices if valid_indices else list(range(len(rgb_list)))
    positions = sorted({0, len(source_indices) // 4, len(source_indices) // 2, (3 * len(source_indices)) // 4, len(source_indices) - 1})
    indices = sorted({source_indices[pos] for pos in positions})
    return [rgb_list[idx] for idx in indices], [clip.samples[idx] for idx in indices]

# this is useless, but can modified to be useful
def compare_detection_views(
    qwen_bundle: QwenBundle,
    clip: ClipInfo,
    rgb_list: Sequence[np.ndarray],
    out_dir: Path,
    previous_inactive_clip: Optional[ClipInfo] = None,
    K_color: Optional[np.ndarray] = None,
    D_color: Optional[np.ndarray] = None,
) -> Tuple[dict, dict]:
    """Run active-clip video interaction detection with Qwen and save the comparison record."""
    inactive_context_frames: List[np.ndarray] = []
    inactive_context_frame_ids: List[int] = []
    if not QWEN_CONTEXT_FRAMES:
        print(f"[Qwen-Detect][Ablation] clip={int(clip.clip_id)} QWEN_CONTEXT_FRAMES=0 — NOT showing Qwen any inactive context frames.")
    if QWEN_CONTEXT_FRAMES and previous_inactive_clip is not None and K_color is not None and D_color is not None and previous_inactive_clip.samples:
        valid_inactive_indices = [idx for idx, sample in enumerate(previous_inactive_clip.samples) if sample.get("rgb_valid", True)]
        source_indices = valid_inactive_indices if valid_inactive_indices else list(range(len(previous_inactive_clip.samples)))
        inactive_indices = sorted({source_indices[max(0, len(source_indices) // 2)], source_indices[-1]})
        for idx in inactive_indices:
            sample = previous_inactive_clip.samples[idx]
            bgr = cv2.imread(str(sample["rgb_path"]))
            if bgr is None:
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            inactive_context_frames.append(prepare_full_frame_for_qwen(undistort_rgb(rgb, K_color, D_color)))
            inactive_context_frame_ids.append(int(sample.get("frame", idx)))
    active_rgb = sanitize_rgb_sequence_for_perception(rgb_list, clip.samples)
    sampled_indices = sampled_video_frame_indices(len(active_rgb), stride=QWEN_ACTIVE_VIDEO_FRAME_STRIDE, max_frames=QWEN_ACTIVE_VIDEO_MAX_FRAMES)
    sampled_frames = [active_rgb[idx] for idx in sampled_indices]
    sampled_frame_ids = [int(clip.samples[idx].get("frame", idx)) for idx in sampled_indices]
    print(
        f"[Qwen-Detect] clip={int(clip.clip_id)} inactive_context_frames={inactive_context_frame_ids} "
        f"active_video_frame_ids={sampled_frame_ids}"
    )
    video_dir = out_dir / "_qwen_active_videos"
    video_path = video_dir / f"clip_{int(clip.clip_id):03d}_active_stride{QWEN_ACTIVE_VIDEO_FRAME_STRIDE}.mp4"
    qwen_video = detect_interaction_video(
        qwen_bundle,
        sampled_frames,
        mode_name=f"active_video_stride{QWEN_ACTIVE_VIDEO_FRAME_STRIDE}",
        video_path=video_path,
        inactive_context_frames=inactive_context_frames,
        sampled_frame_ids=sampled_frame_ids,
    )
    comparison = {
        "clip_id": clip.clip_id,
        "selected_model": "qwen",
        "selected_mode": qwen_video.get("mode", f"active_video_stride{QWEN_ACTIVE_VIDEO_FRAME_STRIDE}"),
        "inactive_context_frames": inactive_context_frame_ids,
        "active_video_frame_ids": sampled_frame_ids,
        "active_video_path": str(video_path),
        "qwen": {
            "selected": qwen_video,
            qwen_video.get("mode", f"active_video_stride{QWEN_ACTIVE_VIDEO_FRAME_STRIDE}"): qwen_video,
        },
    }
    with open(out_dir / "crop_comparison.json", "w", encoding="utf-8") as handle:
        json.dump([comparison], handle, indent=2)
    return qwen_video, comparison


def select_mask_view_indices(
    masks: Sequence[np.ndarray],
    max_views: int = CAPTION_MAX_VIEWS,
) -> List[int]:
    """Return up to ``max_views`` frame indices best suited for VLM captioning.

    Picks indices spread across the clip whose mask has the largest area
    (clearest view of the object) and skips masks with zero foreground.
    Order of the returned indices is the original temporal order so the
    captions read in narrative order.
    """
    if not masks or int(max_views) <= 0:
        return []
    areas: List[Tuple[int, int]] = []
    for idx, m in enumerate(masks):
        if m is None:
            continue
        arr = np.asarray(m, dtype=bool)
        a = int(arr.sum())
        if a > 0:
            areas.append((idx, a))
    if not areas:
        return []
    # Sort by area descending, then keep the largest-area subset while
    # ensuring temporal spread by picking every ``stride``-th candidate.
    areas.sort(key=lambda p: -p[1])
    k = min(int(max_views), len(areas))
    chosen = sorted(idx for idx, _ in areas[:k])
    return chosen


def caption_entity_views(
    bundle: QwenBundle,
    views: Sequence[np.ndarray],
    entity_kind: str,
    label_hint: str,
    parent_hint: str = "",
) -> List[str]:
    """Ask Qwen-VL to caption each prepared view of one entity.

    ``views`` are RGB image crops (already prepared by
    ``make_multiscale_crop`` / ``make_object_caption_view``). Returns one
    short caption string per view, in the same order as ``views``.
    """
    if not views:
        return []
    role_text = (
        "interactive element (a part being touched/manipulated)"
        if entity_kind == "element"
        else "object (the parent object the interactive element is attached to or part of)"
    )
    parent_clause = (
        f" The parent object is around '{parent_hint}'." if parent_hint else ""
    )
    captions: List[str] = []
    for view in views:
        if view is None or not isinstance(view, np.ndarray) or view.size == 0:
            captions.append("")
            continue
        prompt = (
            f"This image is one view of a single {role_text}."
            f" Label hint: '{label_hint}'."
            + parent_clause
            + " Write ONE concise sentence (max ~25 words) that names the entity"
            " and describes its visible appearance (material, colour, shape, distinguishing features)."
            " Do NOT mention the hand, person, camera, or surrounding scene unless it is essential."
            " Output ONLY the sentence; no JSON, no quotes, no preamble."
        )
        try:
            raw = vlm_generate(
                bundle,
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image_to_data_url(prepare_full_frame_for_qwen(view))},
                            {"type": "text", "text": prompt},
                        ],
                    }
                ],
                max_tokens=bundle.config.qwen_max_tokens,
                enable_reasoning=False,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[Qwen-Caption] view caption failed: {exc}")
            raw = ""
        captions.append(str(raw or "").strip())
    return captions


def refine_entity_caption(
    bundle: QwenBundle,
    captions: Sequence[str],
    entity_kind: str,
    label_hint: str,
    parent_hint: str = "",
) -> Tuple[str, str]:
    """Fuse per-view captions into one canonical name + one short description.

    Returns ``(canonical_name, description)``. Empty inputs fall back to
    ``(label_hint, "")`` so callers always get strings.
    """
    cleaned = [str(c).strip() for c in (captions or []) if str(c).strip()]
    if not cleaned:
        return (str(label_hint or "").strip().lower(), "")
    role_text = (
        "interactive element"
        if entity_kind == "element"
        else "object"
    )
    parent_clause = (
        f" The parent object is around '{parent_hint}'." if parent_hint else ""
    )
    bullets = "\n".join(f"- {c}" for c in cleaned)
    prompt = (
        f"Below are several short captions, each describing the same single {role_text}"
        f" from a different view of the same active interaction."
        f" Label hint: '{label_hint}'."
        + parent_clause
        + " Fuse them into ONE canonical name (1-4 lowercase words, no punctuation)"
        " and ONE short description (1-2 sentences, ~40 words max)."
        " Resolve disagreements by majority across views."
        " Output ONLY JSON with keys: \"canonical_name\" and \"description\"."
        f"\n\nCaptions:\n{bullets}"
    )
    try:
        raw = vlm_generate(
            bundle,
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            max_tokens=bundle.config.qwen_max_tokens,
            enable_reasoning=False,
        )
        payload = safe_json_loads(raw) or {}
    except Exception as exc:  # noqa: BLE001
        print(f"[Qwen-Caption] caption refine failed: {exc}")
        payload = {}
    canonical = str(payload.get("canonical_name", "") or "").strip().lower()
    if not canonical:
        canonical = str(label_hint or "").strip().lower()
    description = str(payload.get("description", "") or "").strip()
    if not description and cleaned:
        # Fallback: use the longest single caption.
        description = max(cleaned, key=len)
    return (canonical, description)


def infer_label_candidates(
    bundle: QwenBundle,
    captions: Sequence[str],
    entity_kind: str,
    label_hint: str,
    description: str = "",
    top_k: int = 5,
) -> List[str]:
    """Ask Qwen for a RANKED list of ``top_k`` alternative labels (for recall@K).

    Returns up to ``top_k`` short lowercase noun-phrase labels, most-likely first,
    with ``label_hint`` guaranteed to appear (deduped, order-preserving). Text-only
    (reuses the per-view captions), so it adds no image pass.
    """
    cleaned = [str(c).strip() for c in (captions or []) if str(c).strip()]
    role_text = "interactive element/part" if entity_kind == "element" else "object"
    context = "\n".join(f"- {c}" for c in cleaned) if cleaned else "(no captions)"
    desc_clause = f" Current description: {description}." if description else ""
    prompt = (
        f"We are labelling a single {role_text} seen in an egocentric interaction."
        f" Best-guess label: '{label_hint}'.{desc_clause}"
        f" Propose the {int(top_k)} MOST LIKELY concrete labels for it, ranked from"
        " most to least likely. Each label must be a short generic noun phrase"
        " (1-3 lowercase words, no punctuation), e.g. 'bowl', 'cabinet door',"
        " 'drawer handle'. Include the best guess. Do NOT invent unrelated objects."
        " Output ONLY JSON: {\"labels\": [\"...\", \"...\"]}."
        f"\n\nCaptions:\n{context}"
    )
    labels: List[str] = []
    try:
        raw = vlm_generate(
            bundle,
            [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
            max_tokens=bundle.config.qwen_max_tokens,
            enable_reasoning=False,
        )
        payload = safe_json_loads(raw) or {}
        for item in (payload.get("labels") or []):
            lab = str(item or "").strip().lower()
            if lab:
                labels.append(lab)
    except Exception as exc:  # noqa: BLE001
        print(f"[Qwen-Labels] candidate-label inference failed: {exc}")
    # Guarantee the best guess is present and dedupe order-preserving.
    hint = str(label_hint or "").strip().lower()
    ordered = ([hint] if hint else []) + labels
    seen: set = set()
    out: List[str] = []
    for lab in ordered:
        if lab and lab not in seen:
            seen.add(lab)
            out.append(lab)
    return out[: int(top_k)]

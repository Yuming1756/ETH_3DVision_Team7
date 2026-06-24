# ======================================================================
# graph.py section: 16_inactive_depth_cache.py
# Inactive label priors, depth caching, global scene saving
#
# NOTE: not a standalone module. Loaded by graph.py into a shared
# namespace via exec(); names from earlier sections are in scope here.
# ======================================================================
from __future__ import annotations
def voxel_downsample_points_colors(
    points: np.ndarray,
    colors: np.ndarray,
    voxel_size: float = SEM_VOXEL_SIZE,
) -> Tuple[np.ndarray, np.ndarray]:
    """Downsample one point cloud while preserving aligned colors."""
    pts = np.asarray(points, dtype=np.float32)
    cols = np.asarray(colors, dtype=np.float32)
    if len(pts) == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(cols if len(cols) == len(pts) else np.full((len(pts), 3), 0.5, dtype=np.float32))
    pcd = pcd.voxel_down_sample(voxel_size=float(voxel_size))
    return np.asarray(pcd.points, dtype=np.float32), np.asarray(pcd.colors, dtype=np.float32)


def _remove_statistical_outliers(
    points: np.ndarray,
    colors: np.ndarray,
    nb_neighbors: int = 30,
    std_ratio: float = 2.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Drop floating/isolated points by k-NN mean-distance outlier filtering."""
    pts = np.asarray(points, dtype=np.float32)
    cols = np.asarray(colors, dtype=np.float32)
    if len(pts) <= nb_neighbors:
        return pts, cols
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(cols if len(cols) == len(pts) else np.full((len(pts), 3), 0.5, dtype=np.float32))
    _filtered, kept = pcd.remove_statistical_outlier(nb_neighbors=int(nb_neighbors), std_ratio=float(std_ratio))
    if len(kept) == 0:
        return pts, cols
    kept_arr = np.asarray(kept, dtype=np.int64)
    return pts[kept_arr], cols[kept_arr] if len(cols) == len(pts) else cols


def _prepare_inactive_prior_clip_data(
    inactive_clip: ClipInfo,
    config: PipelineConfig,
    K_depth: np.ndarray,
    K_color: np.ndarray,
    D_color: np.ndarray,
    T_depth_color: np.ndarray,
    color_w: int,
    color_h: int,
    depth_cache_dir: Optional[Path],
    preferred_hand: Optional[str] = None,
    forced_seed_frame_idx: Optional[int] = None,
) -> Dict[str, Any]:
    """Load and stride-select the inactive clip exactly as the prior builder uses it."""
    rgb_list, depth_raw_list, rgb_T_wc_list, depth_T_wc_list = load_clip_frames_with_dual_poses(inactive_clip, K_color, D_color)
    prior_samples = list(inactive_clip.samples)
    sam_rgb_list = sanitize_rgb_sequence_for_perception(rgb_list, prior_samples)
    if forced_seed_frame_idx is not None and 0 <= int(forced_seed_frame_idx) < len(prior_samples):
        prompt_frame_idx = int(forced_seed_frame_idx)
        print(
            f"  inactive SAM3 seed frame forced to local {int(prompt_frame_idx)} "
            f"(global {int(inactive_clip.frame_start + prompt_frame_idx)})"
        )
    else:
        prompt_frame_idx = choose_best_sam_seed_frame(
            prior_samples,
            preferred_hand=preferred_hand,
            prefer_contact=False,
        )
        print(
            f"  inactive SAM3 element seed frame: local {int(prompt_frame_idx)} "
            f"(global {int(inactive_clip.frame_start + prompt_frame_idx)})"
        )
    inactive_prior_frames = max(0, int(config.inactive_prior_frames))
    requested_stride = max(1, int(config.inactive_stride))
    n_inactive_frames = len(prior_samples)
    coverage_cap = max(1, n_inactive_frames // max(1, MIN_INACTIVE_PRIOR_FRAMES))
    prior_stride = max(1, min(requested_stride, coverage_cap))
    if prior_stride != requested_stride:
        print(
            f"  inactive_stride clamped {requested_stride} -> {prior_stride} "
            f"({n_inactive_frames} inactive frames available; floor at {MIN_INACTIVE_PRIOR_FRAMES} prior frames)"
        )
    original_prompt_frame_idx = int(prompt_frame_idx)
    selected_prior_indices = list(range(0, n_inactive_frames, prior_stride))
    if not selected_prior_indices or selected_prior_indices[-1] != (n_inactive_frames - 1):
        selected_prior_indices.append(n_inactive_frames - 1)
    selected_prior_indices.append(original_prompt_frame_idx)
    selected_prior_indices = sorted(set(int(idx) for idx in selected_prior_indices))
    if inactive_prior_frames > 0 and len(selected_prior_indices) > inactive_prior_frames:
        tail = selected_prior_indices[-inactive_prior_frames:]
        if original_prompt_frame_idx not in tail:
            tail.append(original_prompt_frame_idx)
            tail = sorted(set(tail))
            while len(tail) > inactive_prior_frames:
                for drop_idx, value in enumerate(tail):
                    if int(value) != int(original_prompt_frame_idx):
                        del tail[drop_idx]
                        break
        selected_prior_indices = tail
    if selected_prior_indices != list(range(len(rgb_list))):
        selected_global_frames = [inactive_clip.frame_start + idx for idx in selected_prior_indices]
        frame_span = (
            f"{selected_global_frames[0]}-{selected_global_frames[-1]}"
            if selected_global_frames
            else "none"
        )
        mode_text = (
            f"last {inactive_prior_frames} strided frames (forcing seed-frame inclusion)"
            if inactive_prior_frames > 0
            else "all strided frames across inactive clip"
        )
        print(
            f"  limiting inactive prior SAM3 frames to {len(selected_prior_indices)}/{len(rgb_list)} "
            f"using stride {prior_stride} ({mode_text}, global span {frame_span})"
        )
        rgb_list = [rgb_list[idx] for idx in selected_prior_indices]
        depth_raw_list = [depth_raw_list[idx] for idx in selected_prior_indices]
        rgb_T_wc_list = [rgb_T_wc_list[idx] for idx in selected_prior_indices]
        depth_T_wc_list = [depth_T_wc_list[idx] for idx in selected_prior_indices]
        sam_rgb_list = [sam_rgb_list[idx] for idx in selected_prior_indices]
        prior_samples = [prior_samples[idx] for idx in selected_prior_indices]
        prompt_frame_idx = selected_prior_indices.index(original_prompt_frame_idx)
    geometry_T_wc_list = rgb_T_wc_list if depth_cache_dir is not None else depth_T_wc_list
    if depth_cache_dir is not None:
        cached_clip_depths = load_clip_depth_cache(depth_cache_dir, inactive_clip)
        clip_depths = [cached_clip_depths[idx] for idx in selected_prior_indices]
    else:
        clip_depths = compute_clip_sensor_depths(
            depth_raw_list,
            K_depth,
            K_color,
            D_color,
            T_depth_color,
            color_w,
            color_h,
        )
    return {
        "rgb_list": rgb_list,
        "depth_raw_list": depth_raw_list,
        "rgb_T_wc_list": rgb_T_wc_list,
        "depth_T_wc_list": depth_T_wc_list,
        "geometry_T_wc_list": geometry_T_wc_list,
        "clip_depths": clip_depths,
        "prior_samples": prior_samples,
        "sam_rgb_list": sam_rgb_list,
        "prompt_frame_idx": int(prompt_frame_idx),
        "selected_prior_indices": [int(idx) for idx in selected_prior_indices],
    }


def run_cross_clip_object_session(
    inactive_clip: ClipInfo,
    object_label: str,
    config: PipelineConfig,
    K_depth: np.ndarray,
    K_color: np.ndarray,
    D_color: np.ndarray,
    T_depth_color: np.ndarray,
    color_w: int,
    color_h: int,
    depth_cache_dir: Optional[Path],
    active_sam_rgb_list: Sequence[np.ndarray],
    active_prompt_frame_idx: int,
    inactive_seed_frame_idx: Optional[int] = None,
    debug_dir: Optional[Path] = None,
    sam3_predictor: Optional[Any] = None,
    preferred_hand: Optional[str] = None,
    element_label: Optional[str] = None,
    hand_prompts: Optional[Sequence[Any]] = None,
    element_seed_choice: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Run one SAM3 object session over inactive+active frames with a boundary-aware seed."""
    if not active_sam_rgb_list:
        return None
    prepared = _prepare_inactive_prior_clip_data(
        inactive_clip,
        config,
        K_depth,
        K_color,
        D_color,
        T_depth_color,
        color_w,
        color_h,
        depth_cache_dir,
        preferred_hand=preferred_hand,
        forced_seed_frame_idx=inactive_seed_frame_idx,
    )
    inactive_rgb_list = prepared["rgb_list"]
    if not inactive_rgb_list:
        return None
    cross_rgb_list = list(prepared["sam_rgb_list"]) + [np.asarray(frame, dtype=np.uint8).copy() for frame in active_sam_rgb_list]
    cross_jpeg_dir = config.jpeg_dir / "sam3_crossclip_object"
    save_jpegs(cross_rgb_list, cross_jpeg_dir)
    seed_origin = "active"
    seed_global_frame = int(inactive_clip.frame_end + 1 + active_prompt_frame_idx)
    if inactive_seed_frame_idx is not None:
        inactive_seed_frame_idx = int(inactive_seed_frame_idx)
        selected_prior_indices = [int(idx) for idx in prepared.get("selected_prior_indices", [])]
        if inactive_seed_frame_idx in selected_prior_indices:
            cross_prompt_frame_idx = selected_prior_indices.index(inactive_seed_frame_idx)
            seed_origin = "inactive"
            seed_global_frame = int(inactive_clip.frame_start + inactive_seed_frame_idx)
        else:
            cross_prompt_frame_idx = len(prepared["sam_rgb_list"]) + int(active_prompt_frame_idx)
    else:
        cross_prompt_frame_idx = len(prepared["sam_rgb_list"]) + int(active_prompt_frame_idx)
    if seed_origin == "inactive":
        print(
            f"  cross-clip SAM3 object seed frame: inactive local {int(inactive_seed_frame_idx)} "
            f"(global {seed_global_frame}), combined local {int(cross_prompt_frame_idx)}; "
            f"propagating forward into active for '{object_label}'"
        )
    else:
        print(
            f"  cross-clip SAM3 object seed frame: active local {int(active_prompt_frame_idx)} "
            f"(global {seed_global_frame}), combined local {int(cross_prompt_frame_idx)}; "
            f"propagating backward into inactive and forward through active for '{object_label}'"
        )
    # ONE session, seeded at the SAME cross-clip frame, segmenting the object AND
    # (optionally) the element + hands so every label is seeded exactly once and
    # propagated both ways across inactive+active.
    prompt_specs = [
        {
            "label": object_label,
            "text": object_label,
            "frame_index": int(cross_prompt_frame_idx),
            "disable_bootstrap_boxes": True,
        }
    ]
    seg_element = bool(element_label) and str(element_label) != str(object_label)
    if seg_element:
        # Seed the ELEMENT (handle) from the ACTIVE clip (like the object): Qwen
        # picks a BINARY first-active ('beginning') or last-active ('end') frame
        # where the handle is the cleanest single instance. The combined frame list
        # is [inactive..., active...], so active-local i maps to combined n_inactive+i.
        # The single seed propagates BOTH ways (backward into inactive, forward
        # through active). Defaults to 'beginning'.
        _n_inactive = len(prepared["sam_rgb_list"])
        _n_active = len(active_sam_rgb_list)
        _choice = str(element_seed_choice or "beginning").strip().lower()
        if _n_active <= 0:
            element_seed_combined = int(cross_prompt_frame_idx)
            _choice_label = "fallback object seed"
        elif _choice == "end":
            element_seed_combined = _n_inactive + (_n_active - 1)
            _choice_label = "active END (last active frame)"
        else:
            element_seed_combined = _n_inactive + 0
            _choice_label = "active BEGINNING (first active frame)"
        print(
            f"  cross-clip SAM3 element seed frame: {_choice_label} (Qwen choice='{_choice}'), combined local "
            f"{int(element_seed_combined)} (n_inactive={_n_inactive}, n_active={_n_active}); "
            f"propagating both ways for '{element_label}'."
        )
        prompt_specs.append(
            {
                "label": element_label,
                "text": element_label,
                "frame_index": int(element_seed_combined),
                "disable_bootstrap_boxes": True,
            }
        )
    object_masks_dict, cross_hand_masks, object_masks_metadata = run_sam3_prompt_specs(
        cross_jpeg_dir,
        prompt_specs,
        list(hand_prompts) if hand_prompts is not None else [],
        len(cross_rgb_list),
        color_w,
        color_h,
        predictor=sam3_predictor,
    )
    cross_masks = object_masks_dict[object_label]
    cross_metadata = object_masks_metadata.get(object_label, [])
    n_inactive = len(prepared["sam_rgb_list"])

    def _split(seq):
        seq = list(seq) if seq is not None else []
        return (
            [np.asarray(m, dtype=bool).copy() for m in seq[:n_inactive]],
            [np.asarray(m, dtype=bool).copy() for m in seq[n_inactive:]],
        )

    inactive_masks, active_masks = _split(cross_masks)
    inactive_metadata = list(cross_metadata[:n_inactive])
    active_metadata = list(cross_metadata[n_inactive:])
    inactive_element_masks: List[np.ndarray] = []
    active_element_masks: List[np.ndarray] = []
    inactive_element_metadata: List[Any] = []
    active_element_metadata: List[Any] = []
    if seg_element:
        elem_cross_masks = object_masks_dict.get(element_label, [])
        elem_cross_metadata = object_masks_metadata.get(element_label, [])
        inactive_element_masks, active_element_masks = _split(elem_cross_masks)
        inactive_element_metadata = list(elem_cross_metadata[:n_inactive])
        active_element_metadata = list(elem_cross_metadata[n_inactive:])
    _inactive_hand_masks, active_hand_masks = _split(cross_hand_masks)
    if debug_dir is not None:
        try:
            out_dir = Path(debug_dir) / "cross_clip_object_debug"
            out_dir.mkdir(parents=True, exist_ok=True)
            if seed_origin == "inactive":
                seed_rgb = np.ascontiguousarray(np.asarray(prepared["sam_rgb_list"][int(cross_prompt_frame_idx)], dtype=np.uint8).copy())
                seed_mask = np.asarray(inactive_masks[int(cross_prompt_frame_idx)], dtype=bool) if int(cross_prompt_frame_idx) < len(inactive_masks) else None
                seed_label = f"cross-clip seed: inactive local {int(inactive_seed_frame_idx)}"
            else:
                seed_rgb = np.ascontiguousarray(np.asarray(active_sam_rgb_list[int(active_prompt_frame_idx)], dtype=np.uint8).copy())
                seed_mask = np.asarray(active_masks[int(active_prompt_frame_idx)], dtype=bool) if int(active_prompt_frame_idx) < len(active_masks) else None
                seed_label = f"cross-clip seed: active local {int(active_prompt_frame_idx)}"
            _blend_debug_mask_on_canvas(seed_rgb, seed_mask, color=(255, 80, 80), alpha=0.25)
            cv2.putText(
                seed_rgb,
                seed_label,
                (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.imwrite(str(out_dir / "object_seed_frame.png"), cv2.cvtColor(seed_rgb, cv2.COLOR_RGB2BGR))
        except Exception as exc:  # noqa: BLE001
            print(f"  [debug] cross-clip object seed overlay failed: {exc}")
    return {
        "inactive_prepared": prepared,
        "inactive_object_masks": inactive_masks,
        "inactive_object_metadata": inactive_metadata,
        "active_object_masks": active_masks,
        "active_object_metadata": active_metadata,
        "inactive_element_masks": inactive_element_masks,
        "active_element_masks": active_element_masks,
        "inactive_element_metadata": inactive_element_metadata,
        "active_element_metadata": active_element_metadata,
        "active_hand_masks": active_hand_masks,
        "inactive_hand_masks": _inactive_hand_masks,
        "active_prompt_frame_idx": int(active_prompt_frame_idx),
        "cross_prompt_frame_idx": int(cross_prompt_frame_idx),
        "seed_origin": seed_origin,
        "seed_global_frame": int(seed_global_frame),
        "inactive_seed_frame_idx": int(inactive_seed_frame_idx) if inactive_seed_frame_idx is not None else None,
    }


def build_inactive_label_priors(
    inactive_clip: ClipInfo,
    element_label: str,
    object_label: str,
    config: PipelineConfig,
    K_depth: np.ndarray,
    K_color: np.ndarray,
    D_color: np.ndarray,
    T_depth_color: np.ndarray,
    T_cam_depth: np.ndarray,
    T_cam_color: np.ndarray,
    color_w: int,
    color_h: int,
    depth_cache_dir: Optional[Path],
    element_anchor_box_norm: Optional[Sequence[float]] = None,
    object_anchor_box_norm: Optional[Sequence[float]] = None,
    object_crop_box_norm: Optional[Sequence[float]] = None,
    inactive_element_box_norm: Optional[Sequence[float]] = None,
    inactive_object_box_norm: Optional[Sequence[float]] = None,
    inactive_element_point_norm: Optional[Sequence[float]] = None,
    inactive_object_point_norm: Optional[Sequence[float]] = None,
    debug_dir: Optional[Path] = None,
    sam3_predictor: Optional[Any] = None,
    preferred_hand: Optional[str] = None,
    active_sam_rgb_list: Optional[Sequence[np.ndarray]] = None,
    active_prompt_frame_idx: Optional[int] = None,
    active_object_prompt_box_norm: Optional[Sequence[float]] = None,
    precomputed_object_masks: Optional[Sequence[np.ndarray]] = None,
    precomputed_object_confidence_metadata: Optional[Sequence[dict]] = None,
    precomputed_element_masks: Optional[Sequence[np.ndarray]] = None,
    precomputed_element_confidence_metadata: Optional[Sequence[dict]] = None,
    precomputed_hand_masks: Optional[Sequence[np.ndarray]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """Build dense 3D priors for the active labels from the preceding inactive clip."""
    print(
        f"  building inactive priors from clip {inactive_clip.clip_id} "
        f"for '{element_label}' and '{object_label}'"
    )
    prepared = _prepare_inactive_prior_clip_data(
        inactive_clip,
        config,
        K_depth,
        K_color,
        D_color,
        T_depth_color,
        color_w,
        color_h,
        depth_cache_dir,
        preferred_hand=preferred_hand,
    )
    rgb_list = prepared["rgb_list"]
    depth_raw_list = prepared["depth_raw_list"]
    rgb_T_wc_list = prepared["rgb_T_wc_list"]
    depth_T_wc_list = prepared["depth_T_wc_list"]
    geometry_T_wc_list = prepared["geometry_T_wc_list"]
    clip_depths = prepared["clip_depths"]
    prior_samples = prepared["prior_samples"]
    sam_rgb_list = prepared["sam_rgb_list"]
    prompt_frame_idx = int(prepared["prompt_frame_idx"])

    element_anchor_box_abs = (
        normalized_xywh_to_box_xyxy(inactive_element_box_norm, rgb_list[0].shape)
        if inactive_element_box_norm is not None
        else normalized_xywh_to_box_xyxy(element_anchor_box_norm, rgb_list[0].shape)
        if element_anchor_box_norm is not None
        else None
    )
    element_anchor_point_abs = (
        normalized_xy_to_point_xy(inactive_element_point_norm, rgb_list[0].shape)
        if inactive_element_point_norm is not None
        else None
    )
    object_anchor_point_abs = (
        normalized_xy_to_point_xy(inactive_object_point_norm, rgb_list[0].shape)
        if inactive_object_point_norm is not None
        else None
    )
    object_crop_box = None
    # Inactive priors must come ONLY from the unified cross-clip SAM3 session
    # seeded on an ACTIVE frame. Local inactive re-seeding is intentionally
    # disallowed because it is ambiguous for repeated structures such as
    # neighbouring cabinet handles/doors and can switch identity at the
    # inactive boundary.
    save_jpegs(sam_rgb_list, config.jpeg_dir)
    inactive_seed_frame = max(0, len(rgb_list) - 1)
    object_prompt_frame_idx = int(inactive_seed_frame)
    have_precomputed_element = precomputed_element_masks is not None and len(precomputed_element_masks) == len(rgb_list)
    have_precomputed_object = precomputed_object_masks is not None and len(precomputed_object_masks) == len(rgb_list)
    if not (have_precomputed_element and have_precomputed_object):
        raise RuntimeError(
            "[InactivePrior][Error] cross-clip active-seeded SAM3 masks are required for BOTH "
            f"'{element_label}' and '{object_label}'. Local inactive reseeding is disabled."
        )
    print(
        "  inactive prior: NO seeding — both element + object reuse the unified cross-clip backward propagation."
    )
    print(
        f"  using precomputed cross-clip element masks for inactive '{element_label}' "
        f"({len(precomputed_element_masks)} frame(s), seeded from active local "
        f"{int(active_prompt_frame_idx) if active_prompt_frame_idx is not None else -1})"
    )
    print(
        f"  using precomputed cross-clip object masks for inactive '{object_label}' "
        f"({len(precomputed_object_masks)} frame(s), seeded from active local "
        f"{int(active_prompt_frame_idx) if active_prompt_frame_idx is not None else -1})"
    )
    element_masks = [np.asarray(m, dtype=bool).copy() for m in precomputed_element_masks]
    object_masks_metadata = {element_label: list(precomputed_element_confidence_metadata or [])}
    if precomputed_hand_masks is not None and len(precomputed_hand_masks) == len(rgb_list):
        prompt_hand_masks = [np.asarray(m, dtype=bool).copy() for m in precomputed_hand_masks]
    else:
        prompt_hand_masks = [np.zeros(rgb_list[0].shape[:2], dtype=bool) for _ in range(len(rgb_list))]
    object_only_masks = [np.asarray(mask, dtype=bool).copy() for mask in precomputed_object_masks]
    object_only_masks_metadata = {
        object_label: list(precomputed_object_confidence_metadata or [])
    }
    object_anchor_box_abs = (
        normalized_xywh_to_box_xyxy(inactive_object_box_norm, rgb_list[0].shape)
        if inactive_object_box_norm is not None
        else normalized_xywh_to_box_xyxy(object_anchor_box_norm, rgb_list[0].shape)
        if object_anchor_box_norm is not None
        else None
    )
    object_confidence_metadata = object_only_masks_metadata.get(object_label, [])
    element_confidence_metadata = object_masks_metadata.get(element_label, [])
    if not element_confidence_metadata:
        element_confidence_metadata = compute_sam3_mask_confidence_metadata(element_masks, [float("nan")] * len(element_masks), rgb_list[0].shape[:2])
    if not object_confidence_metadata:
        object_confidence_metadata = compute_sam3_mask_confidence_metadata(object_only_masks, [float("nan")] * len(object_only_masks), rgb_list[0].shape[:2])
    prior_rgb_valids = [bool(sample.get("rgb_valid", True)) for sample in prior_samples]
    element_geometry_valids = sam3_geometry_valid_flags(element_confidence_metadata, prior_rgb_valids)
    object_geometry_valids = sam3_geometry_valid_flags(object_confidence_metadata, prior_rgb_valids)
    print(
        f"  SAM3 confidence kept {sum(element_geometry_valids)}/{len(element_geometry_valids)} element frames "
        f"and {sum(object_geometry_valids)}/{len(object_geometry_valids)} object frames for inactive-prior geometry"
    )
    # Per-prior-frame SAM3 confidence from PROPAGATION. sam3_prob is SAM3's raw
    # out_probs (threaded through the unified cross-clip session for both active and
    # inactive frames); confidence = 0.64*sam3_prob + 0.36*shape_consistency.
    _elem_prob = [round(float(m.get("sam3_prob", float("nan"))), 3) for m in element_confidence_metadata]
    _obj_prob = [round(float(m.get("sam3_prob", float("nan"))), 3) for m in object_confidence_metadata]
    _elem_conf = [round(float(m.get("confidence", 0.0)), 3) for m in element_confidence_metadata]
    _obj_conf = [round(float(m.get("confidence", 0.0)), 3) for m in object_confidence_metadata]
    print(f"  inactive-prior element sam3_prob per frame (prior 0..{len(_elem_prob)-1}): {_elem_prob}")
    print(f"  inactive-prior element confidence (fused) per frame: {_elem_conf}")
    print(f"  inactive-prior object  sam3_prob per frame (prior 0..{len(_obj_prob)-1}): {_obj_prob}")
    print(f"  inactive-prior object  confidence (fused) per frame: {_obj_conf}")
    raw_element_masks = [np.asarray(mask, dtype=bool).copy() for mask in element_masks]
    raw_object_masks = [np.asarray(mask, dtype=bool).copy() for mask in object_only_masks]
    # Closest-to-hand anchor for the inactive prior. Boxes were removed, so the
    # anchor is the HAND position at the boundary (last inactive) frame — nearest
    # the active interaction, where the hand is closest to the part. Used to keep
    # the hand-nearest connected component for BOTH element and object below.
    _boundary_sample = (
        prior_samples[inactive_seed_frame]
        if 0 <= inactive_seed_frame < len(prior_samples)
        else (prior_samples[-1] if prior_samples else None)
    )
    _inactive_hand_anchor_xy = (
        select_primary_hand_anchor_point(_boundary_sample, rgb_list[0].shape)
        if _boundary_sample is not None
        else None
    )
    inactive_anchor_xy = (
        (float(element_anchor_point_abs[0]), float(element_anchor_point_abs[1]))
        if element_anchor_point_abs is not None
        else _box_center(element_anchor_box_abs) if element_anchor_box_abs is not None
        else _inactive_hand_anchor_xy
    )
    inactive_object_anchor_xy = (
        (float(object_anchor_point_abs[0]), float(object_anchor_point_abs[1]))
        if object_anchor_point_abs is not None
        else _box_center(object_anchor_box_abs)
        if object_anchor_box_abs is not None
        else inactive_anchor_xy
    )
    # Stabilize in REVERSE: seed at the LAST inactive frame (nearest the active
    # interaction — cabinet clearly visible, hand closest, anchor aligned) and track the
    # hand-nearest component backward toward the first frame. This keeps the boundary
    # frames correct and lets the instance fade gradually as the viewpoint diverges,
    # instead of frame-0-forward locking onto the wrong (far, drifted) cabinet.
    if inactive_anchor_xy is not None:
        element_masks = stabilize_mask_sequence_near_anchor(element_masks[::-1], inactive_anchor_xy)[::-1]
    if inactive_object_anchor_xy is not None:
        object_only_masks = stabilize_mask_sequence_near_anchor(object_only_masks[::-1], inactive_object_anchor_xy)[::-1]
    # Drop EARLY inactive frames where the backward propagation drifted onto a
    # neighbouring near-identical instance (e.g. BOTH cabinet doors): the object
    # mask balloons past the boundary-frame area. Walk back from the boundary
    # (last frame, nearest the active seed); at the first frame whose object area
    # exceeds the ratio, drop it and ALL earlier frames (element + object) so the
    # prior is built only from the reliable, near-the-seed frames.
    if float(INACTIVE_PRIOR_DRIFT_AREA_RATIO) > 0.0 and len(object_only_masks) >= 2:
        _obj_areas = [int(np.count_nonzero(np.asarray(m, dtype=bool))) for m in object_only_masks]
        _boundary_area = _obj_areas[-1]
        if _boundary_area > 0:
            _drift_cut = None
            for _i in range(len(object_only_masks) - 2, -1, -1):
                if _obj_areas[_i] > float(INACTIVE_PRIOR_DRIFT_AREA_RATIO) * float(_boundary_area):
                    _drift_cut = _i
                    break
            if _drift_cut is not None:
                _empty = np.zeros(rgb_list[0].shape[:2], dtype=bool)
                for _j in range(_drift_cut + 1):
                    object_only_masks[_j] = _empty.copy()
                    element_masks[_j] = _empty.copy()
                print(
                    f"  inactive prior: dropped earliest {_drift_cut + 1} frame(s) (indices 0..{_drift_cut}) "
                    f"where the object mask drifted to ~both-doors "
                    f"(area > {float(INACTIVE_PRIOR_DRIFT_AREA_RATIO):.1f}x boundary={_boundary_area})."
                )
    tracking_object_masks = [np.asarray(mask, dtype=bool).copy() for mask in object_only_masks]
    element_masks = constrain_part_masks_inside_object(element_masks, object_only_masks, inactive_anchor_xy)
    object_body_masks = build_exclusive_object_masks(object_only_masks, element_masks)
    # Prior-creation debug is drawn AFTER stabilization so it shows the FINAL
    # masks actually used for the prior. Keep it mask-only: object/body in red,
    # interactive element in orange. No bounding boxes, no crop box.
    if debug_dir is not None:
        try:
            _prior_dbg_dir = Path(debug_dir) / "inactive_prior_debug"
            _prior_dbg_dir.mkdir(parents=True, exist_ok=True)
            for _fi in range(len(rgb_list)):
                _canvas = np.ascontiguousarray(np.asarray(rgb_list[_fi], dtype=np.uint8).copy())
                _om = np.asarray(object_only_masks[_fi], dtype=bool) if _fi < len(object_only_masks) else None
                if _om is not None and np.any(_om) and _om.shape[:2] == _canvas.shape[:2]:
                    _canvas[_om] = (0.25 * np.array([255, 80, 80]) + 0.75 * _canvas[_om]).astype(np.uint8)
                _em = np.asarray(element_masks[_fi], dtype=bool) if _fi < len(element_masks) else None
                if _em is not None and np.any(_em) and _em.shape[:2] == _canvas.shape[:2]:
                    _canvas[_em] = (0.35 * np.array([255, 170, 60]) + 0.65 * _canvas[_em]).astype(np.uint8)
                cv2.imwrite(str(_prior_dbg_dir / f"prior_{_fi:04d}.png"), cv2.cvtColor(_canvas, cv2.COLOR_RGB2BGR))
        except Exception as exc:  # noqa: BLE001
            print(f"  [debug] inactive prior overlay failed: {exc}")
    inactive_debug_dirs = [config.jpeg_dir / "sam3_inactive_debug"]
    if debug_dir is not None:
        inactive_debug_dirs.append(debug_dir)

    def _export_inactive_debug(out_dir: Path) -> None:
        # Debug-only: per-prompt SAM3 result PNGs, box overlays, and the full
        # per-frame overlay sequences for inactive masks. Gated behind
        # GRAPH_DEBUG_IMAGES so they don't bloat output_final by default.
        if not DEBUG_IMAGES_ENABLED:
            return
        out_dir.mkdir(parents=True, exist_ok=True)
        save_sam3_overlay_png(
            out_dir / "inactive_sam3_raw_result.png",
            rgb_list[prompt_frame_idx],
            raw_element_masks[prompt_frame_idx] if prompt_frame_idx < len(raw_element_masks) else np.zeros(rgb_list[prompt_frame_idx].shape[:2], dtype=bool),
            raw_object_masks[prompt_frame_idx] if prompt_frame_idx < len(raw_object_masks) else np.zeros(rgb_list[prompt_frame_idx].shape[:2], dtype=bool),
            interaction_anchor_box=None,
            element_anchor_box=element_anchor_box_abs,
            object_anchor_box=None,
            object_crop_box=None,
        )
        save_sam3_overlay_png(
            out_dir / "inactive_sam3_result.png",
            rgb_list[prompt_frame_idx],
            raw_element_masks[prompt_frame_idx] if prompt_frame_idx < len(raw_element_masks) else np.zeros(rgb_list[prompt_frame_idx].shape[:2], dtype=bool),
            raw_object_masks[prompt_frame_idx] if prompt_frame_idx < len(raw_object_masks) else np.zeros(rgb_list[prompt_frame_idx].shape[:2], dtype=bool),
            interaction_anchor_box=None,
            element_anchor_box=element_anchor_box_abs,
            object_anchor_box=None,
            object_crop_box=None,
        )
        save_box_overlay_png(
            out_dir / "inactive_prompt_boxes.png",
            rgb_list[prompt_frame_idx],
            [
                (element_anchor_box_abs, (255, 80, 80), "element"),
            ],
            points=[],
        )
        save_sam3_overlay_sequence(
            out_dir / "inactive_sam3_raw_overlays",
            rgb_list,
            raw_element_masks,
            raw_object_masks,
            primary_frame_boxes={},
            secondary_frame_index=prompt_frame_idx,
            secondary_frame_boxes={
                "element_anchor_box": element_anchor_box_abs,
            },
        )
        save_sam3_overlay_sequence(
            out_dir / "inactive_sam3_overlays",
            rgb_list,
            raw_element_masks,
            raw_object_masks,
            primary_frame_boxes={},
            secondary_frame_index=prompt_frame_idx,
            secondary_frame_boxes={
                "element_anchor_box": element_anchor_box_abs,
            },
        )
        save_binary_mask_sequence(out_dir / "inactive_sam3_masks" / "raw_element", raw_element_masks, "element")
        save_binary_mask_sequence(out_dir / "inactive_sam3_masks" / "raw_object", raw_object_masks, "object")
        save_binary_mask_sequence(out_dir / "inactive_sam3_masks" / "clean_element", element_masks, "element")
        save_binary_mask_sequence(out_dir / "inactive_sam3_masks" / "object_body", object_body_masks, "object")
        save_binary_mask_sequence(out_dir / "inactive_sam3_masks" / "tracking_object", tracking_object_masks, "object")
        save_sam3_confidence_json(
            out_dir / "inactive_sam3_masks" / "confidence.json",
            {
                "element_label": element_label,
                "object_label": object_label,
                "element_prompt_frame_index": int(prompt_frame_idx),
                "object_prompt_frame_index": int(object_prompt_frame_idx),
                "object_metadata_coordinate_frame": "object_crop" if object_crop_box is not None else "full_image",
                "object_crop_box_xyxy": [int(v) for v in object_crop_box] if object_crop_box is not None else None,
                "element": element_confidence_metadata,
                "object": object_confidence_metadata,
                "element_geometry_valid_frames": [int(i) for i, keep in enumerate(element_geometry_valids) if keep],
                "object_geometry_valid_frames": [int(i) for i, keep in enumerate(object_geometry_valids) if keep],
            },
        )

    # Extra inactive SAM3 debug (inactive_sam3_result.png, inactive_sam3_overlays,
    # inactive_sam3_masks, inactive_prompt_boxes.png, ...) removed — only the
    # consolidated inactive_prior_debug overlay (written above) is kept.
    _ = (inactive_debug_dirs, _export_inactive_debug)  # intentionally unused now
    detector_hand_masks = [
        np.asarray(prompt_hand_masks[idx], dtype=bool) if idx < len(prompt_hand_masks) else np.zeros(rgb_list[idx].shape[:2], dtype=bool)
        for idx in range(len(rgb_list))
    ]

    use_da3 = bool(config.use_da3)
    depth_grid = None if use_da3 else build_depth_backprojection_grid(
        K_depth, depth_raw_list[0].shape[0], depth_raw_list[0].shape[1],
    )
    elem_per_frame = collect_mask_lifted_geometries(
        element_masks,
        detector_hand_masks,
        element_geometry_valids,
        clip_depths if use_da3 else None,
        None if use_da3 else depth_raw_list,
        rgb_list,
        rgb_T_wc_list if use_da3 else depth_T_wc_list,
        K_color,
        None if use_da3 else K_depth,
        T_cam_color,
        None if use_da3 else T_cam_depth,
        depth_grid,
        use_da3,
    )
    obj_per_frame = collect_mask_lifted_geometries(
        tracking_object_masks,
        detector_hand_masks,
        object_geometry_valids,
        clip_depths if use_da3 else None,
        None if use_da3 else depth_raw_list,
        rgb_list,
        rgb_T_wc_list if use_da3 else depth_T_wc_list,
        K_color,
        None if use_da3 else K_depth,
        T_cam_color,
        None if use_da3 else T_cam_depth,
        depth_grid,
        use_da3,
    )
    valid_elem_indices = [idx for idx, (pts, _) in enumerate(elem_per_frame) if len(pts)]
    valid_obj_indices = [idx for idx, (pts, _) in enumerate(obj_per_frame) if len(pts)]
    elem_points, elem_colors = geometry_from_indices(elem_per_frame, valid_elem_indices)
    obj_points, obj_colors = geometry_from_indices(obj_per_frame, valid_obj_indices)
    elem_points, elem_colors = voxel_downsample_points_colors(elem_points, elem_colors)
    obj_points, obj_colors = voxel_downsample_points_colors(obj_points, obj_colors)
    elem_points, elem_colors = _remove_statistical_outliers(elem_points, elem_colors)
    obj_points, obj_colors = _remove_statistical_outliers(obj_points, obj_colors)

    return elem_points, elem_colors, obj_points, obj_colors, {}


def depth_cache_path(cache_dir: Path, clip: ClipInfo) -> Path:
    """Return the on-disk path used to store one clip's precomputed depths."""
    return cache_dir / f"clip_{clip.clip_id:03d}_depths.npy"


def store_clip_depth_cache(cache_dir: Path, clip: ClipInfo, clip_depths: Sequence[np.ndarray]) -> None:
    """Persist one clip's depth products to disk in float16 to limit cache size."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    stacked = np.stack([depth.astype(np.float16, copy=False) for depth in clip_depths], axis=0)
    np.save(depth_cache_path(cache_dir, clip), stacked, allow_pickle=False)


def load_clip_depth_cache(cache_dir: Path, clip: ClipInfo) -> List[np.ndarray]:
    """Load one clip's cached depths back into memory as float32 arrays."""
    cache_path = depth_cache_path(cache_dir, clip)
    if not cache_path.exists():
        raise FileNotFoundError(f"Depth cache missing for clip {clip.clip_id}: {cache_path}")
    stacked = np.load(cache_path, mmap_mode="r")
    return [np.asarray(stacked[idx], dtype=np.float32) for idx in range(stacked.shape[0])]


def compact_global_pointcloud(global_pcd_pts: List[np.ndarray], global_pcd_cols: List[np.ndarray], voxel_size: float = SEM_VOXEL_SIZE) -> None:
    """Compact the accumulated global point cloud in memory via voxel downsampling."""
    if not global_pcd_pts:
        return
    all_pts = np.concatenate(list(global_pcd_pts), axis=0)
    all_cols = np.concatenate(list(global_pcd_cols), axis=0)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(all_pts)
    pcd.colors = o3d.utility.Vector3dVector(all_cols)
    pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    compact_pts = np.asarray(pcd.points, dtype=np.float32)
    compact_cols = np.asarray(pcd.colors, dtype=np.float32)
    global_pcd_pts[:] = [compact_pts] if compact_pts.size else []
    global_pcd_cols[:] = [compact_cols] if compact_cols.size else []
    gc.collect()


def global_point_count(global_pcd_pts: Sequence[np.ndarray]) -> int:
    """Return the total number of currently accumulated global scene points."""
    return int(sum(chunk.shape[0] for chunk in global_pcd_pts))


def save_global_scene(global_pcd_pts: Sequence[np.ndarray], global_pcd_cols: Sequence[np.ndarray], out_dir: Path) -> None:
    """Voxel-downsample and save the accumulated inactive-scene point cloud."""
    if not global_pcd_pts:
        return
    all_pts = np.concatenate(list(global_pcd_pts), axis=0)
    all_cols = np.concatenate(list(global_pcd_cols), axis=0)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(all_pts)
    pcd.colors = o3d.utility.Vector3dVector(all_cols)
    pcd = pcd.voxel_down_sample(voxel_size=SEM_VOXEL_SIZE)
    o3d.io.write_point_cloud(str(out_dir / "scene_full.ply"), pcd)


def precompute_clip_depths(
    clip: ClipInfo,
    da3_model: Optional[DepthAnything3],
    K_color: np.ndarray,
    D_color: np.ndarray,
    K_depth: np.ndarray,
    T_depth_color: np.ndarray,
    T_cam_color: np.ndarray,
    color_w: int,
    color_h: int,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """Load one clip and return metric depth maps with RGB frames and poses."""
    rgb_list, depth_raw_list, T_wc_list = load_clip_frames(clip, K_color, D_color)
    if da3_model is None:
        sensor_depths = [
            prepare_sensor_depth_for_lifting(
                depth_raw_list[idx],
                K_depth,
                T_depth_color,
                K_color,
                D_color,
                color_w,
                color_h,
            )
            for idx in range(len(rgb_list))
        ]
        del depth_raw_list
        gc.collect()
        return sensor_depths, rgb_list, T_wc_list

    da3_depths = run_da3(da3_model, rgb_list, T_wc_list, K_color, T_cam_color, color_w, color_h)
    del depth_raw_list
    clip_depths = []
    for idx in range(len(rgb_list)):
        depth = np.asarray(da3_depths[idx], dtype=np.float32)
        depth[(depth <= 1e-4) | (depth > DEPTH_MAX)] = 0.0
        clip_depths.append(depth)
    del da3_depths
    torch.cuda.empty_cache()
    gc.collect()
    return clip_depths, rgb_list, T_wc_list


def precompute_depth_cache(
    clips: Sequence[ClipInfo],
    config: PipelineConfig,
    K_color: np.ndarray,
    D_color: np.ndarray,
    K_depth: np.ndarray,
    T_depth_color: np.ndarray,
    T_cam_color: np.ndarray,
    color_w: int,
    color_h: int,
) -> Path:
    """Run the dedicated DA3 pass once and cache per-clip depth products on disk."""
    cache_dir = config.output_dir / "_depth_cache"
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    da3_model = load_da3(config)
    try:
        for clip in clips:
            print(f"\n=== Depth Pass Clip {clip.clip_id} [{'INACTIVE' if clip.is_inactive else 'ACTIVE'}] frames {clip.frame_start}-{clip.frame_end} ===")
            clip_depths, rgb_list, T_wc_list = precompute_clip_depths(
                clip,
                da3_model,
                K_color,
                D_color,
                K_depth,
                T_depth_color,
                T_cam_color,
                color_w,
                color_h,
            )
            store_clip_depth_cache(cache_dir, clip, clip_depths)
            del clip_depths
            del rgb_list
            del T_wc_list
            gc.collect()
    finally:
        del da3_model
        torch.cuda.empty_cache()
        gc.collect()

    return cache_dir

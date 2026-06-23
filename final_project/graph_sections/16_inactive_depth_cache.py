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
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """Build dense 3D priors for the active labels from the preceding inactive clip."""
    print(
        f"  building inactive priors from clip {inactive_clip.clip_id} "
        f"for '{element_label}' and '{object_label}'"
    )
    rgb_list, depth_raw_list, rgb_T_wc_list, depth_T_wc_list = load_clip_frames_with_dual_poses(inactive_clip, K_color, D_color)
    prior_samples = list(inactive_clip.samples)
    sam_rgb_list = sanitize_rgb_sequence_for_perception(rgb_list, prior_samples)
    prompt_frame_idx = max(
        (idx for idx, sample in enumerate(prior_samples) if sample.get("rgb_valid", True)),
        default=max(0, len(prior_samples) - 1),
    )
    inactive_prior_frames = max(0, int(config.inactive_prior_frames))
    requested_stride = max(1, int(config.inactive_stride))
    # Adaptive stride: never honour a stride that would leave fewer than
    # MIN_INACTIVE_PRIOR_FRAMES frames lifted from this inactive clip.
    # Shrinks the requested stride against the actual clip length so short
    # inactive clips (e.g. 307_2200 after the pre-pad migration) still
    # produce a usable prior cloud.
    n_inactive_frames = len(prior_samples)
    coverage_cap = max(1, n_inactive_frames // max(1, MIN_INACTIVE_PRIOR_FRAMES))
    prior_stride = max(1, min(requested_stride, coverage_cap))
    if prior_stride != requested_stride:
        print(
            f"  inactive_stride clamped {requested_stride} -> {prior_stride} "
            f"({n_inactive_frames} inactive frames available; floor at {MIN_INACTIVE_PRIOR_FRAMES} prior frames)"
        )
    original_prompt_frame_idx = int(prompt_frame_idx)
    selected_prior_indices = list(range(0, original_prompt_frame_idx + 1, prior_stride))
    if not selected_prior_indices or selected_prior_indices[-1] != original_prompt_frame_idx:
        selected_prior_indices.append(original_prompt_frame_idx)
    selected_prior_indices = sorted(set(int(idx) for idx in selected_prior_indices))
    if inactive_prior_frames > 0 and len(selected_prior_indices) > inactive_prior_frames:
        selected_prior_indices = selected_prior_indices[-inactive_prior_frames:]
    if selected_prior_indices != list(range(len(rgb_list))):
        selected_global_frames = [inactive_clip.frame_start + idx for idx in selected_prior_indices]
        frame_span = (
            f"{selected_global_frames[0]}-{selected_global_frames[-1]}"
            if selected_global_frames
            else "none"
        )
        mode_text = (
            f"last {inactive_prior_frames} strided frames"
            if inactive_prior_frames > 0
            else "all strided frames up to prompt"
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

    save_jpegs(sam_rgb_list, config.jpeg_dir)
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
    prompt_specs = [
        {
            "label": element_label,
            "text": build_interaction_aware_sam_text(
                element_label,
                interactive_element=element_label,
                parent_object=object_label,
                role="element",
                active=False,
            ),
            "frame_index": prompt_frame_idx,
        },
    ]
    object_masks, prompt_hand_masks, object_masks_metadata = run_sam3_prompt_specs(
        config.jpeg_dir,
        prompt_specs,
        HAND_PROMPTS,
        len(rgb_list),
        color_w,
        color_h,
        predictor=sam3_predictor,
    )
    element_masks = object_masks[element_label]
    # Method A: anchor the inactive object box on the inactive HANDLE mask (the
    # handle is visible at rest), instead of the carried-over ACTIVE-clip box,
    # which is taken at the moved/open position and mismatches the closed-drawer
    # prior (that mismatch dropped the drawer prior to 1/9 object frames). Fall
    # back to the saved boxes only if the inactive handle wasn't segmented.
    inactive_handle_object_box = None
    _prompt_elem_mask = (
        np.asarray(element_masks[prompt_frame_idx], dtype=bool)
        if prompt_frame_idx < len(element_masks)
        else None
    )
    if _prompt_elem_mask is not None and np.any(_prompt_elem_mask):
        inactive_handle_object_box = select_object_anchor_box(
            prior_samples[prompt_frame_idx],
            rgb_list[prompt_frame_idx].shape,
            _prompt_elem_mask,
            fallback_object_box=None,
            hint_object_box=None,
        )
    object_anchor_box_abs = (
        tuple(int(v) for v in inactive_handle_object_box)
        if inactive_handle_object_box is not None
        else normalized_xywh_to_box_xyxy(inactive_object_box_norm, rgb_list[0].shape)
        if inactive_object_box_norm is not None
        else normalized_xywh_to_box_xyxy(object_anchor_box_norm, rgb_list[0].shape)
        if object_anchor_box_norm is not None
        else None
    )
    object_anchor_point_abs = (
        normalized_xy_to_point_xy(inactive_object_point_norm, rgb_list[0].shape)
        if inactive_object_point_norm is not None
        else None
    )
    object_crop_box = None
    _prior_crop_box = None
    object_prompt_specs = [
        {
            "label": object_label,
            "text": build_interaction_aware_sam_text(
                object_label,
                interactive_element=element_label,
                parent_object=object_label,
                role="object",
                active=False,
            ),
            "frame_index": prompt_frame_idx,
            "disable_bootstrap_boxes": True,
            # SAM3-I expects normalized xywh boxes in [0,1]; object_anchor_box_abs
            # is absolute xyxy, so convert it (passing raw xyxy trips SAM3-I's
            # `boxes_xywh <= 1` assertion).
            "bounding_boxes": (
                [box_xyxy_to_normalized_xywh(object_anchor_box_abs, rgb_list[0].shape)]
                if object_anchor_box_abs is not None
                else None
            ),
            "bounding_box_labels": [1] if object_anchor_box_abs is not None else None,
        }
    ]
    # Build the object-centric crop box up front from the ACTIVE handle/object
    # anchors mapped into the inactive frame. The active handle box is a clean,
    # single-handle box on the moving part (the kitchen has several cabinet
    # handles, so we do NOT trust the raw inactive "cabinet handle" grounding to
    # localize the crop). Inactive/active cameras are close enough that this same
    # crop region covers the closed object too.
    _crop_elem_box = element_anchor_box_abs
    _crop_obj_box = (
        normalized_xywh_to_box_xyxy(object_anchor_box_norm, rgb_list[0].shape)
        if object_anchor_box_norm is not None
        else object_anchor_box_abs
    )
    if (
        os.environ.get("GRAPH_OBJECT_CROP", "0") == "1"
        and active_sam_rgb_list is not None
        and active_prompt_frame_idx is not None
    ):
        # Unified pipeline (instance carry): the inactive (closed/static) clip
        # has NO motion cue to disambiguate which of several near-identical
        # doors/drawers is "the one". A crop derived from the active handle's
        # NORMALIZED box does not transfer: the camera moves between the
        # inactive and active clips, so that normalized position lands on a
        # DIFFERENT physical handle in the inactive view (observed: cabinet
        # prior locked onto the wrong, center handle while the active part
        # tracked the right door). The only viewpoint-robust carry is SAM3
        # VIDEO PROPAGATION of the object INSTANCE: seed the object once at the
        # active open frame (anchored by the active object box), then let SAM3
        # track that exact mask backward through the contiguous frames into the
        # inactive clip. No per-frame text re-grounding (which would re-pick an
        # arbitrary door) and no crop (which drifts to the neighbour).
        _n_inact = len(sam_rgb_list)
        _combined_sam_rgb = list(sam_rgb_list) + list(active_sam_rgb_list)
        _combined_jpeg_dir = config.jpeg_dir.parent / f"{config.jpeg_dir.name}_combinedprop"
        save_jpegs(_combined_sam_rgb, _combined_jpeg_dir)
        _prompt_combined = _n_inact + int(active_prompt_frame_idx)
        _prop_specs = [
            {
                "label": object_label,
                "text": build_interaction_aware_sam_text(
                    object_label,
                    interactive_element=element_label,
                    parent_object=object_label,
                    role="object",
                    active=True,
                ),
                "frame_index": _prompt_combined,
                "disable_bootstrap_boxes": True,
                # Seed the propagated instance with the ACTIVE OBJECT MASK's bbox
                # (the real door the active clip segmented), falling back to the
                # handle-anchored object box only if no active mask was provided.
                # The anchor box sits on the cabinet's lower-right corner/wall, so
                # seeding from it drifts to the wrong door; the mask bbox is the
                # actual door region.
                "bounding_boxes": (
                    [list(active_object_prompt_box_norm)] if active_object_prompt_box_norm is not None
                    else [list(object_anchor_box_norm)] if object_anchor_box_norm is not None
                    else None
                ),
                "bounding_box_labels": (
                    [1] if (active_object_prompt_box_norm is not None or object_anchor_box_norm is not None) else None
                ),
            }
        ]
        _seed_src = "active-object-mask" if active_object_prompt_box_norm is not None else "anchor-box"
        print(
            f"  [Method3-prior/instance] (seed={_seed_src}) propagating object '{object_label}' from active frame "
            f"{int(active_prompt_frame_idx)} (combined idx {_prompt_combined}) backward into {_n_inact} inactive frames"
        )
        _prop_dict, _unused_local_hand_masks, object_only_masks_metadata = run_sam3_prompt_specs(
            _combined_jpeg_dir, _prop_specs, [], len(_combined_sam_rgb), color_w, color_h, predictor=sam3_predictor,
        )
        object_only_masks = [np.asarray(m, dtype=bool).copy() for m in _prop_dict[object_label][:_n_inact]]
        object_only_masks_metadata = {}  # confidence recomputed downstream (full image)
    elif (
        os.environ.get("GRAPH_PRIOR_METHOD", "A").upper() == "B"
        and active_sam_rgb_list is not None
        and active_prompt_frame_idx is not None
    ):
        # Method B: segment the inactive object by PROPAGATING from the ACTIVE
        # open frame backward into the (contiguous) inactive frames in ONE SAM3
        # session, instead of prompting the closed-drawer inactive clip directly
        # — which collapses to empty on backward propagation (1/9). The open
        # drawer is a strong anchor that holds through propagation.
        _n_inact = len(sam_rgb_list)
        _combined_sam_rgb = list(sam_rgb_list) + list(active_sam_rgb_list)
        _combined_jpeg_dir = config.jpeg_dir.parent / f"{config.jpeg_dir.name}_combinedB"
        save_jpegs(_combined_sam_rgb, _combined_jpeg_dir)
        _prompt_combined = _n_inact + int(active_prompt_frame_idx)
        # First propagate the HANDLE across the combined sequence to drive a
        # handle-tracking per-frame crop, then segment the OBJECT inside that crop.
        # The crop keeps propagation locked to the moving part each frame, so it
        # can't wander to a different cabinet door or grab the drawer's static
        # lower band.
        _h_specs = [{
            "label": element_label,
            "text": build_interaction_aware_sam_text(
                element_label, interactive_element=element_label,
                parent_object=object_label, role="element", active=True,
            ),
            "frame_index": _prompt_combined,
            # Anchor the handle prompt with the active handle box so backward
            # propagation locks onto the CORRECT handle (the kitchen has several
            # cabinet handles; without this the crop drifts to the wrong cabinet).
            "bounding_boxes": [list(element_anchor_box_norm)] if element_anchor_box_norm is not None else None,
            "bounding_box_labels": [1] if element_anchor_box_norm is not None else None,
        }]
        _h_dict, _unused_hh, _unused_hm = run_sam3_prompt_specs(
            _combined_jpeg_dir, _h_specs, [], len(_combined_sam_rgb), color_w, color_h, predictor=sam3_predictor,
        )
        _combined_handle = _h_dict[element_label]
        _elem_box_abs_b = normalized_xywh_to_box_xyxy(element_anchor_box_norm, rgb_list[0].shape) if element_anchor_box_norm is not None else None
        _obj_box_abs_b = normalized_xywh_to_box_xyxy(object_anchor_box_norm, rgb_list[0].shape) if object_anchor_box_norm is not None else None
        if _elem_box_abs_b is not None:
            _cb = build_object_centric_crop_box(_elem_box_abs_b, _obj_box_abs_b, rgb_list[0].shape)
            _bcrop_w, _bcrop_h = max(1, _cb[2] - _cb[0]), max(1, _cb[3] - _cb[1])
        else:
            _bcrop_w, _bcrop_h = color_w // 2, color_h // 2
        _bcrop_boxes = build_stabilized_crop_boxes(_combined_handle, _bcrop_w, _bcrop_h, rgb_list[0].shape)
        _bcropped = crop_rgb_sequence_per_frame(_combined_sam_rgb, _bcrop_boxes)
        _bcrop_dir = config.jpeg_dir.parent / f"{config.jpeg_dir.name}_combinedBcrop"
        save_jpegs(_bcropped, _bcrop_dir)
        _o_specs = [{
            "label": object_label,
            "text": build_interaction_aware_sam_text(
                object_label, interactive_element=element_label,
                parent_object=object_label, role="object", active=True,
            ),
            "frame_index": _prompt_combined,
            "disable_bootstrap_boxes": True,
        }]
        print(
            f"  [MethodB+crop] handle-tracking crop {_bcrop_w}x{_bcrop_h}; propagating object "
            f"'{object_label}' from active frame {int(active_prompt_frame_idx)} into {_n_inact} inactive frames"
        )
        _o_dict, _unused_local_hand_masks, object_only_masks_metadata = run_sam3_prompt_specs(
            _bcrop_dir, _o_specs, [], len(_combined_sam_rgb), _bcrop_w, _bcrop_h, predictor=sam3_predictor,
        )
        _combined_obj_full = uncrop_mask_sequence_per_frame(_o_dict[object_label], _bcrop_boxes, rgb_list[0].shape[:2])
        object_only_masks = [np.asarray(m, dtype=bool).copy() for m in _combined_obj_full[:_n_inact]]
        object_only_masks_metadata = {}  # confidence recomputed downstream
    else:
        object_only_masks_dict, _unused_local_hand_masks, object_only_masks_metadata = run_sam3_prompt_specs(
            config.jpeg_dir,
            object_prompt_specs,
            [],
            len(rgb_list),
            color_w,
            color_h,
            predictor=sam3_predictor,
        )
        object_only_masks = object_only_masks_dict[object_label]
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
    raw_element_masks = [np.asarray(mask, dtype=bool).copy() for mask in element_masks]
    raw_object_masks = [np.asarray(mask, dtype=bool).copy() for mask in object_only_masks]
    inactive_anchor_xy = (
        (float(element_anchor_point_abs[0]), float(element_anchor_point_abs[1]))
        if element_anchor_point_abs is not None
        else _box_center(element_anchor_box_abs) if element_anchor_box_abs is not None else None
    )
    inactive_object_anchor_xy = (
        _box_center(object_anchor_box_abs)
        if object_anchor_box_abs is not None
        else inactive_anchor_xy
    )
    if inactive_anchor_xy is not None:
        element_masks = stabilize_mask_sequence_near_anchor(element_masks, inactive_anchor_xy)
    if inactive_object_anchor_xy is not None:
        object_only_masks = stabilize_mask_sequence_near_anchor(object_only_masks, inactive_object_anchor_xy)
    tracking_object_masks = [np.asarray(mask, dtype=bool).copy() for mask in object_only_masks]
    element_masks = constrain_part_masks_inside_object(element_masks, object_only_masks, inactive_anchor_xy)
    object_body_masks = build_exclusive_object_masks(object_only_masks, element_masks)
    # Prior-creation debug (drawn AFTER stabilization, so it shows the FINAL masks
    # actually used for the prior): blue = object prompt box (handle-anchored),
    # orange = the single touched-handle (element) mask kept nearest the anchor
    # (NOT the raw multi-handle SAM3 grounding), green = per-frame object-mask
    # bbox, red = the object mask.
    if debug_dir is not None:
        try:
            _prior_dbg_dir = Path(debug_dir) / "inactive_prior_debug"
            _prior_dbg_dir.mkdir(parents=True, exist_ok=True)
            for _fi in range(len(rgb_list)):
                _canvas = np.ascontiguousarray(np.asarray(rgb_list[_fi], dtype=np.uint8).copy())
                _om = np.asarray(object_only_masks[_fi], dtype=bool) if _fi < len(object_only_masks) else None
                if _om is not None and np.any(_om) and _om.shape[:2] == _canvas.shape[:2]:
                    _canvas[_om] = (0.25 * np.array([255, 80, 80]) + 0.75 * _canvas[_om]).astype(np.uint8)
                    _ob = mask_to_bbox(_om)
                    if _ob is not None:
                        gx1, gy1, gx2, gy2 = [int(v) for v in _ob]
                        cv2.rectangle(_canvas, (gx1, gy1), (gx2, gy2), (80, 255, 80), 2)
                        cv2.putText(_canvas, f"obj mask {gx2-gx1}x{gy2-gy1}", (gx1 + 4, max(gy1 - 8, 30)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 255, 80), 2, cv2.LINE_AA)
                # Cyan = object-centric crop window (the SAME crop the active clip
                # uses): SAM3 grounds the object text-only INSIDE this box, so it
                # can only pick the one touched door/drawer, not every cabinet door.
                if _prior_crop_box is not None:
                    cbx1, cby1, cbx2, cby2 = [int(v) for v in _prior_crop_box]
                    cv2.rectangle(_canvas, (cbx1, cby1), (cbx2, cby2), (60, 220, 220), 2)
                    cv2.putText(_canvas, "obj crop", (cbx1 + 4, max(cby1 - 8, 18)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 220, 220), 2, cv2.LINE_AA)
                _hb = mask_to_bbox(np.asarray(element_masks[_fi], dtype=bool)) if _fi < len(element_masks) and np.any(element_masks[_fi]) else None
                if _hb is not None:
                    hx1, hy1, hx2, hy2 = [int(v) for v in _hb]
                    cv2.rectangle(_canvas, (hx1, hy1), (hx2, hy2), (255, 170, 60), 2)
                    cv2.putText(_canvas, element_label, (hx1 + 4, min(hy2 + 20, _canvas.shape[0] - 6)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 170, 60), 2, cv2.LINE_AA)
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
                "object_metadata_coordinate_frame": "object_crop" if object_crop_box is not None else "full_image",
                "object_crop_box_xyxy": [int(v) for v in object_crop_box] if object_crop_box is not None else None,
                "element": element_confidence_metadata,
                "object": object_confidence_metadata,
                "element_geometry_valid_frames": [int(i) for i, keep in enumerate(element_geometry_valids) if keep],
                "object_geometry_valid_frames": [int(i) for i, keep in enumerate(object_geometry_valids) if keep],
            },
        )

    for inactive_debug_dir in inactive_debug_dirs:
        _export_inactive_debug(inactive_debug_dir)
    print(f"  inactive SAM3 debug saved to {inactive_debug_dirs[0]}")
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



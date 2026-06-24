# ======================================================================
# graph.py section: 17_pipeline_main.py
# Active/inactive clip processing, run_pipeline, argparse, main
#
# NOTE: not a standalone module. Loaded by graph.py into a shared
# namespace via exec(); names from earlier sections are in scope here.
# ======================================================================
from __future__ import annotations


class Sam3Manager:
    """Build the SAM3 video predictor ONCE and keep it resident for the whole video.

    SAM3 only needs to co-exist with CoTracker3, and since CoTracker now runs in
    ONLINE/streaming mode (constant ~window-bounded memory, not whole-clip), the two
    fit on the GPU together. So there is no need to release/rebuild SAM3 per clip --
    and doing so was actively harmful: repeated large alloc/free cycles fragmented
    the CUDA caching allocator and crept the baseline up ~1.2 GiB by mid-video,
    which then OOM'd the Qwen detection forward. Build once, free once at the end.

    ``release()`` is therefore a deliberate NO-OP (per-clip callers stay wired but
    don't churn the model); ``shutdown()`` does the single real teardown.
    """

    def __init__(self, config: "PipelineConfig") -> None:
        self._config = config
        self._predictor: Optional[Any] = None

    def get(self) -> Any:
        if self._predictor is None:
            self._predictor = build_configured_sam3_video_predictor(self._config)
        return self._predictor

    def release(self) -> None:
        # NO-OP: keep SAM3 resident (online CoTracker fits alongside). Rebuilding it
        # per clip fragmented GPU memory and caused the Qwen OOM.
        return

    def shutdown(self) -> None:
        if self._predictor is not None:
            try:
                self._predictor.shutdown()
            except Exception as exc:  # noqa: BLE001
                print(f"[Sam3Manager] shutdown raised ({exc}); continuing.")
            self._predictor = None
            torch.cuda.empty_cache()
            gc.collect()


def process_inactive_clip(
    clip: ClipInfo,
    K_depth: np.ndarray,
    K_color: np.ndarray,
    D_color: np.ndarray,
    T_cam_depth: np.ndarray,
    T_cam_color: np.ndarray,
    global_pcd_pts: List[np.ndarray],
    global_pcd_cols: List[np.ndarray],
    jpeg_dir: Path,
    clip_depths: Optional[Sequence[np.ndarray]] = None,
    rgb_list: Optional[Sequence[np.ndarray]] = None,
    T_wc_list: Optional[Sequence[np.ndarray]] = None,
    inactive_stride: int = 1,
    sam3_manager: Optional["Sam3Manager"] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run inactive reconstruction.

    Native sensor-only mode lifts in the depth-camera geometry path. When DA3
    depths are available, we lift directly in the RGB camera frame using the
    DA3 metric depth map instead of routing geometry through the native depth
    rays.
    """
    sam3_predictor = sam3_manager.get() if sam3_manager is not None else None
    print()
    print(f"=== Clip {clip.clip_id} [INACTIVE] frames {clip.frame_start}-{clip.frame_end} ===")
    using_da3_depth = clip_depths is not None and rgb_list is not None and T_wc_list is not None
    if using_da3_depth:
        rgb_frames = list(rgb_list)
        depth_raw_list: List[np.ndarray] = []
        world_poses = list(T_wc_list)
    else:
        rgb_frames, depth_raw_list, world_poses = load_clip_native_frames(clip, K_color, D_color)
    clip_point_blocks: List[np.ndarray] = []
    clip_color_blocks: List[np.ndarray] = []
    skipped_invalid_pose = 0
    skipped_invalid_rgb = 0
    stride = max(1, int(inactive_stride))
    prompt_hand_masks: List[np.ndarray] = []
    sam_rgb_frames = sanitize_rgb_sequence_for_perception(rgb_frames, clip.samples) if rgb_frames else []
    if rgb_frames:
        save_jpegs(sam_rgb_frames, jpeg_dir)
        _, prompt_hand_masks = run_sam3(
            jpeg_dir,
            [],
            HAND_PROMPTS,
            len(rgb_frames),
            rgb_frames[0].shape[1],
            rgb_frames[0].shape[0],
            predictor=sam3_predictor,
        )
    depth_grid = None
    if not using_da3_depth and depth_raw_list:
        depth_grid = build_depth_backprojection_grid(K_depth, depth_raw_list[0].shape[0], depth_raw_list[0].shape[1])
    for idx in tqdm(range(0, len(rgb_frames), stride), desc="  lifting", leave=False):
        if not clip.samples[idx].get("pose_valid", True):
            skipped_invalid_pose += 1
            continue
        if not clip.samples[idx].get("rgb_valid", True):
            skipped_invalid_rgb += 1
            continue
        hand_mask = np.asarray(prompt_hand_masks[idx], dtype=bool) if idx < len(prompt_hand_masks) else np.zeros(rgb_frames[idx].shape[:2], dtype=bool)
        if using_da3_depth:
            full_mask = np.ones(rgb_frames[idx].shape[:2], dtype=bool)
            if hand_mask is not None:
                full_mask &= ~np.asarray(hand_mask, dtype=bool)
            pts, cols = lift_mask_to_world(
                full_mask,
                np.asarray(clip_depths[idx], dtype=np.float32),
                rgb_frames[idx],
                K_color,
                world_poses[idx],
                T_cam_color,
                depth_max=DEPTH_MAX,
            )
        else:
            pts, cols = lift_depth_frame_to_world_with_rgb(
                depth_raw=depth_raw_list[idx],
                rgb=rgb_frames[idx],
                depth_grid=depth_grid,
                K_color=K_color,
                T_wc=world_poses[idx],
                T_cam_depth=T_cam_depth,
                T_cam_color=T_cam_color,
                D_color=None,
                exclusion_mask=hand_mask,
                aligned_color_depth=None,
                pixel_stride=INACTIVE_DEPTH_PIXEL_STRIDE,
                depth_max=DEPTH_MAX,
            )
        if len(pts):
            global_pcd_pts.append(pts)
            global_pcd_cols.append(cols)
            clip_point_blocks.append(pts)
            clip_color_blocks.append(cols)
            if len(global_pcd_pts) >= GLOBAL_COMPACT_CHUNKS or global_point_count(global_pcd_pts) >= GLOBAL_COMPACT_POINT_THRESHOLD:
                compact_global_pointcloud(global_pcd_pts, global_pcd_cols)
            if len(clip_point_blocks) >= GLOBAL_COMPACT_CHUNKS or global_point_count(clip_point_blocks) >= GLOBAL_COMPACT_POINT_THRESHOLD:
                compact_global_pointcloud(clip_point_blocks, clip_color_blocks)
    compact_global_pointcloud(global_pcd_pts, global_pcd_cols)
    compact_global_pointcloud(clip_point_blocks, clip_color_blocks)
    if skipped_invalid_pose:
        print(f"  skipped {skipped_invalid_pose} inactive frames with invalid camera pose")
    if skipped_invalid_rgb:
        print(f"  skipped {skipped_invalid_rgb} inactive frames with invalid color health")
    return stack_point_blocks(clip_point_blocks, clip_color_blocks)


def process_active_clip(
    clip: ClipInfo,
    depth_raw_list: Sequence[np.ndarray],
    clip_depths: Sequence[np.ndarray],
    rgb_list: Sequence[np.ndarray],
    T_wc_list: Sequence[np.ndarray],
    geometry_T_wc_list: Sequence[np.ndarray],
    qwen_bundle: QwenBundle,
    scene_graph: DynamicSceneGraph,
    K_depth: np.ndarray,
    K_color: np.ndarray,
    D_color: np.ndarray,
    T_depth_color: np.ndarray,
    T_cam_depth: np.ndarray,
    T_cam_color: np.ndarray,
    color_w: int,
    color_h: int,
    scene_suppression_boxes: List[Tuple[np.ndarray, np.ndarray]],
    previous_inactive_clip: Optional[ClipInfo],
    depth_cache_dir: Optional[Path],
    config: PipelineConfig,
    sam3_manager: Optional["Sam3Manager"] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run the active-mode graph-only update pipeline from precomputed clip depths."""
    sam3_predictor = sam3_manager.get() if sam3_manager is not None else None
    print()
    print(f"=== Clip {clip.clip_id} [ACTIVE] frames {clip.frame_start}-{clip.frame_end} ===")

    detection, _detection_comparison = compare_detection_views(
        qwen_bundle,
        clip,
        rgb_list,
        config.output_dir,
        previous_inactive_clip=previous_inactive_clip,
        K_color=K_color,
        D_color=D_color,
    )
    elem_label = detection["interactive_element"]
    tracked_obj_label = detection["parent_object"]
    if elem_label in NON_INTERACTABLE:
        elem_label = "interactive element"
    if tracked_obj_label in NON_INTERACTABLE:
        tracked_obj_label = "object"

    sam_rgb_list = sanitize_rgb_sequence_for_perception(rgb_list, clip.samples)
    save_jpegs(sam_rgb_list, config.jpeg_dir)
    active_anchor_idx = next((idx for idx, sample in enumerate(clip.samples) if sample.get("rgb_valid", True)), 0)
    first_sample = clip.samples[active_anchor_idx]
    interaction_anchor_box, detected_object_anchor_box = select_interaction_anchor_boxes(first_sample, rgb_list[active_anchor_idx].shape)
    primary_hand_anchor_xy = select_primary_hand_anchor_point(first_sample, rgb_list[active_anchor_idx].shape)
    support_context_labels = {
        "bowl",
        "plate",
        "dish",
        "rack",
        "holder",
        "table",
        "counter",
        "countertop",
        "shelf",
        "sink",
        "bin",
        "basket",
    }
    tool_like_tokens = {
        "spoon",
        "fork",
        "knife",
        "ladle",
        "spatula",
        "whisk",
        "tongs",
        "mug",
        "cup",
        "bottle",
        "jar",
        "can",
    }
    if (
        tracked_obj_label in support_context_labels
        or any(tok in elem_label for tok in tool_like_tokens)
        or any(tok in tracked_obj_label for tok in tool_like_tokens)
    ):
        inactive_rgb_for_hierarchy = None
        if previous_inactive_clip is not None and previous_inactive_clip.samples:
            for inactive_sample in reversed(previous_inactive_clip.samples):
                if not inactive_sample.get("rgb_valid", True):
                    continue
                bgr = cv2.imread(str(inactive_sample["rgb_path"]))
                if bgr is None:
                    continue
                inactive_rgb_for_hierarchy = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                break
        later_valid_idx = next(
            (idx for idx in range(len(clip.samples) - 1, -1, -1) if clip.samples[idx].get("rgb_valid", True)),
            active_anchor_idx,
        )
        later_sample = clip.samples[later_valid_idx]
        later_box, _ = select_interaction_anchor_boxes(later_sample, rgb_list[later_valid_idx].shape)
        hierarchy_result = infer_interaction_hierarchy(
            qwen_bundle,
            rgb_list[active_anchor_idx],
            interaction_anchor_box,
            inactive_frame_rgb=inactive_rgb_for_hierarchy,
            later_frame_rgb=rgb_list[later_valid_idx] if later_valid_idx != active_anchor_idx else None,
            later_interaction_bbox=later_box if later_valid_idx != active_anchor_idx else None,
            element_hint=elem_label,
            object_hint=tracked_obj_label,
        )
        detection["hierarchy_recheck"] = hierarchy_result
        refined_elem = str(hierarchy_result.get("interactive_element", "")).strip().lower()
        refined_obj = str(hierarchy_result.get("parent_object", "")).strip().lower()
        refined_conf = float(hierarchy_result.get("confidence", 0.0))
        suspicious_support_parent = tracked_obj_label in support_context_labels and any(
            tok in refined_obj for tok in tool_like_tokens
        )
        suspicious_generic_pair = refined_conf >= 0.6 and (
            tracked_obj_label in NON_INTERACTABLE
            or elem_label in NON_INTERACTABLE
        )
        if suspicious_support_parent or suspicious_generic_pair:
            print(
                "  [QwenHierarchyOverride] "
                f"video detector suggested '{elem_label}' -> '{tracked_obj_label}', "
                f"crop hierarchy refined to '{refined_elem}' -> '{refined_obj}' "
                f"(conf={refined_conf:.2f})"
            )
            if refined_elem:
                elem_label = refined_elem
            if refined_obj:
                tracked_obj_label = refined_obj
            detection["interactive_element"] = elem_label
            detection["parent_object"] = tracked_obj_label
            detection["reasoning"] = (
                str(detection.get("reasoning", "")).strip()
                + " | hierarchy_recheck: "
                + str(hierarchy_result.get("reasoning", "")).strip()
            ).strip(" |")
    qwen_parent_body_label = tracked_obj_label
    motion_coupling = infer_motion_coupling(
        qwen_bundle,
        elem_label,
        tracked_obj_label,
        rgb_list[0],
        rgb_list[-1],
    )
    # Only re-route the tracked object label to the moving-part label when
    # Qwen ACTUALLY classified the interaction as articulated. The original
    # logic entered this branch on any non-empty ``moving_part_label`` and
    # then forcibly overwrote ``interaction_type`` to ``"articulated_object"``
    # — which misclassified rigid-transport interactions (e.g. lid + container
    # being carried as one rigid unit) into the articulated path. Qwen's
    # explicit classification wins; ``moving_part_label`` alone is not enough
    # to imply articulated motion.
    qwen_interaction_type = str(motion_coupling.get("interaction_type", "")).strip().lower()
    articulated_child_label: Optional[str] = None
    if qwen_interaction_type == "articulated_object":
        articulated_child_label = str(motion_coupling.get("moving_part_label") or "").strip() or None
        if articulated_child_label is None:
            articulated_child_label = infer_articulated_part_prompt_label(elem_label, tracked_obj_label)
    if articulated_child_label:
        tracked_obj_label = articulated_child_label
        motion_coupling = apply_motion_coupling_guardrails(
            {
                "interaction_type": "articulated_object",
                "shares_rigid_motion": True,
                "moving_part_label": articulated_child_label,
                "confidence": max(float(motion_coupling.get("confidence", 0.0)), 0.9),
                "reasoning": (
                    f"Rerouted Qwen parent '{qwen_parent_body_label}' to moving child part "
                    f"'{articulated_child_label}' for tracking. "
                    + str(motion_coupling.get("reasoning", ""))
                ),
            },
            elem_label,
            tracked_obj_label,
            str(motion_coupling.get("reasoning", "")),
        )
    # Two independent decisions, deliberately decoupled:
    #
    #  (1) use_active_object_sam3 — run the active-SAM3 + inactive-prior +
    #      active-trajectory segmentation with the first/last-active-frame seed
    #      logic. This applies to BOTH articulated objects AND rigid-transport
    #      carries (spoon/mug carried as one rigid body).
    #
    #  (2) is_truly_articulated_motion — fit a 1-DoF ArtiPoint JOINT and use the
    #      ArtiPoint tracker + hand-anchor mask stabilization. This applies ONLY
    #      to genuinely articulated objects. A carried rigid object has no joint,
    #      so it keeps LostFound 6-DoF rigid-pose math and SKIPS the joint fit
    #      (matching the original working behavior). Crucially, the hand-anchor
    #      mask stabilization (which assumes element+object are co-located at the
    #      hand) must NOT run for a rigid pair whose object can be away from the
    #      hand — that is what collapsed the object mask.
    use_active_object_sam3 = bool(
        motion_coupling.get("shares_rigid_motion", False)
        and (
            is_articulated_object_motion(motion_coupling, elem_label, tracked_obj_label)
            or qwen_interaction_type == "rigid_transport"
        )
    )
    is_truly_articulated_motion = bool(
        motion_coupling.get("shares_rigid_motion", False)
        and is_articulated_object_motion(motion_coupling, elem_label, tracked_obj_label)
    )
    # Holds the single unified cross-clip SAM3 session result (one seed per label,
    # propagated both ways). Defined for BOTH the active-SAM3 and the fallback paths
    # so the inactive-prior section can reuse it (or None) without a NameError.
    _unified_cross_result = None
    if use_active_object_sam3:
        if is_truly_articulated_motion:
            print("  using active SAM3 because articulated tracking requires active object masks")
        else:
            print("  using active SAM3 for rigid-transport object (LostFound rigid-pose tracking, no joint fit)")
        # Let Qwen choose whether SAM seeds at the BEGINNING / MIDDLE / END of the
        # active clip (begin/mid/end frames + last inactive frame for context).
        # Prismatic parts (drawer) are clearest at the END once fully revealed;
        # revolute doors (cabinet) at the BEGINNING before rotation foreshortens
        # them. Computed while Qwen is still loaded, just before release. Applies to
        # BOTH articulated and rigid objects.
        try:
            _seed_endpoint = infer_sam_seed_endpoint(
                qwen_bundle,
                clip,
                rgb_list,
                elem_label,
                tracked_obj_label,
                motion_coupling=motion_coupling,
                previous_inactive_clip=previous_inactive_clip,
            )
            sam_seed_choice = str(_seed_endpoint.get("seed_choice", "end")).strip().lower()
            print(
                f"  Qwen SAM seed choice for '{tracked_obj_label}': {sam_seed_choice} "
                f"(conf={float(_seed_endpoint.get('confidence', 0.0)):.2f}) — {_seed_endpoint.get('reasoning', '')}"
            )
        except Exception as exc:  # noqa: BLE001
            sam_seed_choice = "end"
            print(f"  [seed-endpoint] Qwen seed-choice failed ({exc}); defaulting to '{sam_seed_choice}'.")
        if sam_seed_choice not in {"beginning", "middle", "end"}:
            sam_seed_choice = "end"
        # Best INACTIVE frame to seed the ELEMENT (handle), picked by Qwen while it is
        # still loaded. The element is now seeded from the ACTIVE clip (like the
        # object): Qwen picks a BINARY first-active ('beginning') or last-active
        # ('end') frame where the handle is the cleanest single instance.
        element_seed_choice = "beginning"
        if elem_label and str(elem_label) != str(tracked_obj_label):
            try:
                _elem_seed = infer_element_seed_frame(
                    qwen_bundle, rgb_list, elem_label, tracked_obj_label,
                )
                element_seed_choice = str(_elem_seed.get("seed_choice", "beginning")).strip().lower()
                print(
                    f"  Qwen element seed choice for '{elem_label}': {element_seed_choice} (active) "
                    f"(conf={float(_elem_seed.get('confidence', 0.0)):.2f}) — {_elem_seed.get('reasoning', '')}"
                )
            except Exception as exc:  # noqa: BLE001
                element_seed_choice = "beginning"
                print(f"  [element-seed] Qwen element seed-choice failed ({exc}); defaulting to 'beginning'.")
        if element_seed_choice not in {"beginning", "end"}:
            element_seed_choice = "beginning"
        release_qwen_bundle(qwen_bundle)
        print("  released Qwen before active SAM3 segmentation")
        clip_dir = config.output_dir / f"clip_{clip.clip_id:03d}"
        clip_dir.mkdir(parents=True, exist_ok=True)
        # Seed frame from Qwen's begin/middle/end choice mapped DIRECTLY onto the
        # valid active frames (no element-mask dependency, no box). One frame for
        # everything.
        _valid_active = [idx for idx in range(len(rgb_list)) if clip.samples[idx].get("rgb_valid", True)] or list(range(len(rgb_list)))
        if sam_seed_choice == "beginning":
            object_prompt_frame_idx = _valid_active[0]
        elif sam_seed_choice == "middle":
            object_prompt_frame_idx = _valid_active[len(_valid_active) // 2]
        else:  # "end"
            object_prompt_frame_idx = _valid_active[-1]
        object_anchor_box = None
        object_crop_box = None
        object_prompt_box_norm = None
        element_anchor_box = None
        # ===== UNIFIED CROSS-CLIP SAM3 — one seed per label, propagated both ways =====
        # Seed the element + object (+ hands) ONCE at the Qwen-chosen active frame
        # and propagate forward through active AND backward into inactive in a
        # SINGLE SAM3 session. The active portion drives keypoint tracking; the
        # inactive portion (reused in the prior section below) is the prior. Pure
        # text grounding, no boxes, no crop.
        print(
            f"  unified SAM3 (text-only, no box/crop): single seed for '{elem_label}' + "
            f"'{tracked_obj_label}' at active frame {object_prompt_frame_idx} ({sam_seed_choice}); "
            f"propagating both ways across inactive+active."
        )
        unified_cross = None
        if previous_inactive_clip is not None:
            try:
                unified_cross = run_cross_clip_object_session(
                    previous_inactive_clip,
                    tracked_obj_label,
                    config,
                    K_depth,
                    K_color,
                    D_color,
                    T_depth_color,
                    color_w,
                    color_h,
                    depth_cache_dir,
                    sam_rgb_list,
                    object_prompt_frame_idx,
                    debug_dir=clip_dir,
                    sam3_predictor=sam3_predictor,
                    preferred_hand=clip.interaction_hand,
                    element_label=elem_label,
                    hand_prompts=HAND_PROMPTS,
                    element_seed_choice=element_seed_choice,
                )
            except Exception as exc:  # noqa: BLE001
                print(f"  [unified-sam] cross-clip session failed ({exc}); falling back to separate active seeds.")
                unified_cross = None
        _unified_cross_result = None
        if (
            unified_cross is not None
            and unified_cross.get("active_object_masks")
            and unified_cross.get("active_element_masks")
        ):
            _unified_cross_result = unified_cross
            raw_active_elem_masks = [np.asarray(m, dtype=bool).copy() for m in unified_cross["active_element_masks"]]
            obj_masks = [np.asarray(m, dtype=bool).copy() for m in unified_cross["active_object_masks"]]
            _active_hands = unified_cross.get("active_hand_masks", []) or []
            detector_hand_masks = [
                np.asarray(_active_hands[idx], dtype=bool) if idx < len(_active_hands) else np.zeros(rgb_list[idx].shape[:2], dtype=bool)
                for idx in range(len(rgb_list))
            ]
            element_masks_metadata = {elem_label: unified_cross.get("active_element_metadata", [])}
            object_masks_local_metadata = {tracked_obj_label: unified_cross.get("active_object_metadata", [])}
            print(
                f"  unified SAM3 active masks: element {sum(1 for m in raw_active_elem_masks if np.any(m))}/"
                f"{len(raw_active_elem_masks)} non-empty, object "
                f"{sum(1 for m in obj_masks if np.any(m))}/{len(obj_masks)} non-empty."
            )
        else:
            # Fallback: separate active element + object text seeds (no unified session).
            print(
                f"  active SAM3 fallback element seed: active local {int(object_prompt_frame_idx)} "
                f"(global {int(clip.frame_start + object_prompt_frame_idx)}) for '{elem_label}'"
            )
            element_prompt_specs = [
                {
                    "label": elem_label,
                    "text": build_interaction_aware_sam_text(
                        elem_label, interactive_element=elem_label, parent_object=tracked_obj_label,
                        role="element", active=True,
                    ),
                    "frame_index": object_prompt_frame_idx,
                }
            ]
            element_masks_dict, prompt_hand_masks, element_masks_metadata = run_sam3_prompt_specs(
                config.jpeg_dir, element_prompt_specs, HAND_PROMPTS, len(rgb_list), color_w, color_h, predictor=sam3_predictor,
            )
            detector_hand_masks = [
                np.asarray(prompt_hand_masks[idx], dtype=bool) if idx < len(prompt_hand_masks) else np.zeros(rgb_list[idx].shape[:2], dtype=bool)
                for idx in range(len(rgb_list))
            ]
            raw_active_elem_masks = [np.asarray(mask, dtype=bool).copy() for mask in element_masks_dict[elem_label]]
            object_prompt_specs = [
                {
                    "label": tracked_obj_label,
                    "text": build_interaction_aware_sam_text(
                        tracked_obj_label, interactive_element=elem_label, parent_object=tracked_obj_label,
                        role="object", active=True,
                    ),
                    "frame_index": object_prompt_frame_idx,
                    "disable_bootstrap_boxes": True,
                }
            ]
            print(
                f"  active SAM3 fallback object seed: active local {int(object_prompt_frame_idx)} "
                f"(global {int(clip.frame_start + object_prompt_frame_idx)}) for '{tracked_obj_label}'"
            )
            object_masks_local_dict, _unused_object_hand_masks, object_masks_local_metadata = run_sam3_prompt_specs(
                config.jpeg_dir, object_prompt_specs, [], len(rgb_list), color_w, color_h, predictor=sam3_predictor,
            )
            obj_masks = object_masks_local_dict[tracked_obj_label]
        elem_masks = stabilize_mask_sequence_near_anchor(raw_active_elem_masks, primary_hand_anchor_xy)
        active_element_confidence_metadata = element_masks_metadata.get(elem_label, [])
        if not active_element_confidence_metadata:
            active_element_confidence_metadata = compute_sam3_mask_confidence_metadata(
                raw_active_elem_masks,
                [float("nan")] * len(raw_active_elem_masks),
                rgb_list[0].shape[:2],
            )
        active_object_confidence_metadata = object_masks_local_metadata.get(tracked_obj_label, [])
        if not active_object_confidence_metadata:
            active_object_confidence_metadata = compute_sam3_mask_confidence_metadata(
                obj_masks,
                [float("nan")] * len(obj_masks),
                rgb_list[0].shape[:2],
            )
        active_rgb_valids = [bool(sample.get("rgb_valid", True)) for sample in clip.samples]
        element_geometry_valids = sam3_geometry_valid_flags(active_element_confidence_metadata, active_rgb_valids)
        object_geometry_valids = sam3_geometry_valid_flags(active_object_confidence_metadata, active_rgb_valids)
        print(
            f"  SAM3 confidence kept {sum(element_geometry_valids)}/{len(element_geometry_valids)} element frames "
            f"and {sum(object_geometry_valids)}/{len(object_geometry_valids)} object frames for active geometry"
        )
        # Per-active-frame SAM3 propagation probability (sam3_prob = raw out_probs).
        _ae_prob = [round(float(m.get("sam3_prob", float("nan"))), 3) for m in active_element_confidence_metadata]
        _ao_prob = [round(float(m.get("sam3_prob", float("nan"))), 3) for m in active_object_confidence_metadata]
        print(f"  active element sam3_prob per frame (active 0..{len(_ae_prob)-1}): {_ae_prob}")
        print(f"  active object  sam3_prob per frame (active 0..{len(_ao_prob)-1}): {_ao_prob}")
        tracking_obj_masks = [np.asarray(mask, dtype=bool).copy() for mask in obj_masks]
        if is_truly_articulated_motion:
            tracking_obj_masks = stabilize_mask_sequence_near_anchor(tracking_obj_masks, primary_hand_anchor_xy)
            raw_area_stats = [int(np.count_nonzero(np.asarray(mask, dtype=bool))) for mask in tracking_obj_masks]
            print(
                "  articulated active object masks: keeping an anchor-near, area-biased connected "
                f"component per frame (median area={int(np.median(raw_area_stats)) if raw_area_stats else 0})"
            )
            tracking_obj_masks = lightly_shrink_mask_sequence(tracking_obj_masks)
            print("  articulated active object masks: lightly shrinking masks to reduce boundary leakage")
        elem_masks = constrain_part_masks_inside_object(elem_masks, tracking_obj_masks, primary_hand_anchor_xy)
        obj_body_masks = build_exclusive_object_masks(tracking_obj_masks, elem_masks)
        shared_motion_masks = [
            (np.asarray(obj_body_mask, dtype=bool) | np.asarray(elem_mask, dtype=bool))
            for obj_body_mask, elem_mask in zip(obj_body_masks, elem_masks)
        ]
        use_da3 = bool(config.use_da3)
        depth_grid = None if use_da3 else build_depth_backprojection_grid(
            K_depth, depth_raw_list[0].shape[0], depth_raw_list[0].shape[1],
        )
        rgb_T_wc_list_for_lift = T_wc_list if use_da3 else geometry_T_wc_list
        elem_per_frame = collect_mask_lifted_geometries(
            elem_masks,
            detector_hand_masks,
            element_geometry_valids,
            clip_depths if use_da3 else None,
            None if use_da3 else depth_raw_list,
            rgb_list,
            rgb_T_wc_list_for_lift,
            K_color,
            None if use_da3 else K_depth,
            T_cam_color,
            None if use_da3 else T_cam_depth,
            depth_grid,
            use_da3,
        )
        obj_tracking_per_frame = collect_mask_lifted_geometries(
            tracking_obj_masks,
            detector_hand_masks,
            object_geometry_valids,
            clip_depths if use_da3 else None,
            None if use_da3 else depth_raw_list,
            rgb_list,
            rgb_T_wc_list_for_lift,
            K_color,
            None if use_da3 else K_depth,
            T_cam_color,
            None if use_da3 else T_cam_depth,
            depth_grid,
            use_da3,
            pre_eroded_masks=bool(is_truly_articulated_motion),
        )
        obj_semantic_per_frame = collect_mask_lifted_geometries(
            obj_body_masks,
            detector_hand_masks,
            object_geometry_valids,
            clip_depths if use_da3 else None,
            None if use_da3 else depth_raw_list,
            rgb_list,
            rgb_T_wc_list_for_lift,
            K_color,
            None if use_da3 else K_depth,
            T_cam_color,
            None if use_da3 else T_cam_depth,
            depth_grid,
            use_da3,
            pre_eroded_masks=bool(is_truly_articulated_motion),
        )

        elem_track_pts, elem_track_cols = geometry_from_indices(elem_per_frame, [idx for idx in range(len(elem_per_frame)) if len(elem_per_frame[idx][0])])
        obj_track_pts, obj_track_cols = geometry_from_indices(obj_tracking_per_frame, [idx for idx in range(len(obj_tracking_per_frame)) if len(obj_tracking_per_frame[idx][0])])
        # NOTE: these init_/final_ point sets are for the active-clip *visualization*
        # (sam3_mask_reconstruction*.ply and the active_clip_points overlay), NOT
        # for the LostFound tracking prior. The LostFound prior is the
        # inactive-clip geometry from build_inactive_label_priors above; see
        # `run_lostfound_geometry_update` which is fed inactive_obj_pts /
        # inactive_elem_pts.
        init_element_pts, init_element_cols = first_valid_geometry(elem_per_frame)
        init_object_pts, init_object_cols = first_valid_geometry(obj_tracking_per_frame)
        init_object_semantic_pts, init_object_semantic_cols = first_valid_geometry(obj_semantic_per_frame)
        init_shared_motion_pts, init_shared_motion_cols = stack_point_blocks(
            [pts for pts in (init_object_semantic_pts, init_element_pts) if len(pts)],
            [cols for cols in (init_object_semantic_cols, init_element_cols) if len(cols)],
        )
        final_element_pts, _final_element_cols = geometry_from_indices(elem_per_frame, select_final_geometry_indices(elem_per_frame))
        final_object_pts, _final_object_cols = geometry_from_indices(obj_tracking_per_frame, select_final_geometry_indices(obj_tracking_per_frame))
        final_object_semantic_pts, _final_object_semantic_cols = geometry_from_indices(obj_semantic_per_frame, select_final_geometry_indices(obj_semantic_per_frame))
        qwen_bundle = ensure_qwen_bundle_loaded(qwen_bundle, config)
        refined_elem_label, element_desc, element_captions, element_view_indices, _element_description_comparison = build_entity_description(
            qwen_bundle,
            rgb_list,
            elem_masks,
            "element",
            elem_label,
            tracked_obj_label,
        )
        refined_obj_label, object_desc, object_captions, object_view_indices, _object_description_comparison = build_entity_description(
            qwen_bundle,
            rgb_list,
            obj_body_masks,
            "object",
            tracked_obj_label,
            elem_label,
        )
        if elem_label in {"interactive element", "object", "unknown"} and refined_elem_label:
            elem_label = refined_elem_label
        if tracked_obj_label in {"object", "unknown"} and refined_obj_label:
            tracked_obj_label = refined_obj_label

        # Ranked top-5 alternative labels (for recall@K), persisted on the nodes
        # and in event.json. Text-only (reuses the per-view captions).
        element_label_candidates = infer_label_candidates(
            qwen_bundle, element_captions, "element", elem_label, element_desc, top_k=5
        )
        object_label_candidates = infer_label_candidates(
            qwen_bundle, object_captions, "object", tracked_obj_label, object_desc, top_k=5
        )
        print(f"  recall@K candidates — element {element_label_candidates} | object {object_label_candidates}")

        final_obj_label = tracked_obj_label
        if is_affordance_style_pair(elem_label, final_obj_label):
            relationship = infer_relationship(
                qwen_bundle,
                elem_label,
                final_obj_label,
                rgb_list[0],
                rgb_list[-1],
                [],
                element_desc=element_desc,
                object_desc=object_desc,
            )
        else:
            relationship = {
                "relationship": "unknown",
                "confidence": 0.0,
                "reasoning": "Pair is not a clear affordance-style part/object relation.",
                "edge_type": "functional",
                "parent_object": final_obj_label,
            }
        clip_dir = config.output_dir / f"clip_{clip.clip_id:03d}"
        clip_dir.mkdir(parents=True, exist_ok=True)
        # Active SAM3 debug images (sam3_result.png, sam3_active_overlays,
        # sam3_active_masks/*, object_prompt_boxes.png) removed — only the
        # consolidated tracking_debug (artipoint <label>_tracks / <label>_masks)
        # is kept, matching the inactive_prior_debug + tracking_debug pair.
    else:
        print("  non-articulated clip: active SAM3 disabled by design; using inactive priors/LostFound for active geometry")
        final_obj_label = tracked_obj_label
        element_desc, object_desc = "", ""
        if is_affordance_style_pair(elem_label, final_obj_label):
            relationship = infer_relationship(
                qwen_bundle,
                elem_label,
                final_obj_label,
                rgb_list[0],
                rgb_list[-1],
                [],
                element_desc=element_desc,
                object_desc=object_desc,
            )
        else:
            relationship = {
                "relationship": "unknown",
                "confidence": 0.0,
                "reasoning": "Pair is not a clear affordance-style part/object relation.",
                "edge_type": "functional",
                "parent_object": final_obj_label,
            }
        empty_masks = [np.zeros(rgb_list[idx].shape[:2], dtype=bool) for idx in range(len(rgb_list))]
        detector_hand_masks = [mask.copy() for mask in empty_masks]
        raw_active_elem_masks = [mask.copy() for mask in empty_masks]
        elem_masks = [mask.copy() for mask in empty_masks]
        obj_masks = [mask.copy() for mask in empty_masks]
        tracking_obj_masks = [mask.copy() for mask in empty_masks]
        obj_body_masks = [mask.copy() for mask in empty_masks]
        element_anchor_box = interaction_anchor_box
        object_prompt_frame_idx = active_anchor_idx
        object_prompt_sample = first_sample
        object_prompt_interaction_box = interaction_anchor_box
        object_prompt_detected_object_box = detected_object_anchor_box
        prompt_element_anchor_box = element_anchor_box
        object_anchor_box = select_object_anchor_box(
            object_prompt_sample,
            rgb_list[object_prompt_frame_idx].shape,
            None,
            fallback_object_box=object_prompt_detected_object_box or object_prompt_interaction_box,
            hint_object_box=None,
        )
        object_crop_box = None
        active_element_confidence_metadata = []
        active_object_confidence_metadata = []
        elem_per_frame = [(np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)) for _ in rgb_list]
        obj_tracking_per_frame = [(np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)) for _ in rgb_list]
        obj_semantic_per_frame = [(np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)) for _ in rgb_list]
        init_element_pts, init_element_cols = np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)
        init_object_pts, init_object_cols = np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)
        init_object_semantic_pts, init_object_semantic_cols = np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)
        init_shared_motion_pts, init_shared_motion_cols = np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)
        final_element_pts, final_object_pts, final_object_semantic_pts = (
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0, 3), dtype=np.float32),
        )
        refined_elem_label, refined_obj_label = elem_label, tracked_obj_label
        element_captions, object_captions = [], []
        element_view_indices, object_view_indices = [], []
        clip_dir = config.output_dir / f"clip_{clip.clip_id:03d}"
        clip_dir.mkdir(parents=True, exist_ok=True)
    inactive_elem_pts = np.zeros((0, 3), dtype=np.float32)
    inactive_elem_cols = np.zeros((0, 3), dtype=np.float32)
    inactive_obj_pts = np.zeros((0, 3), dtype=np.float32)
    inactive_obj_cols = np.zeros((0, 3), dtype=np.float32)
    if previous_inactive_clip is not None:
        release_qwen_bundle(qwen_bundle)
        print("  released Qwen before inactive-prior SAM3 segmentation")
        # No bounding boxes anywhere: the inactive prior is seeded purely by the
        # cross-clip text propagation below (and bare-label text inside
        # build_inactive_label_priors), never by a box.
        active_object_prompt_box_norm = None
        # Unified cross-clip object SAM3 session: seed ONCE at the active object
        # prompt frame (where the moving part is most distinct) and propagate the
        # instance BACKWARD through the inactive clip. The inactive portion of that
        # single propagation becomes the object prior — so a part that is flush /
        # closed in the inactive frames (a closed drawer is unsegmentable by a
        # fresh bare-label seed, max_area=0) is still recovered, because SAM3 video
        # propagation tracks the same instance back through the near-identical
        # boundary frames. Falls back safely to the inactive-frame seed on failure.
        # Reuse the SINGLE unified cross-clip session already run in the active
        # block (one seed per label, propagated both ways). Its INACTIVE portion is
        # the prior for BOTH element and object — no second SAM3 session here.
        precomputed_inactive_object_masks = None
        precomputed_inactive_object_metadata = None
        precomputed_inactive_element_masks = None
        precomputed_inactive_element_metadata = None
        if _unified_cross_result is not None:
            precomputed_inactive_object_masks = _unified_cross_result.get("inactive_object_masks")
            precomputed_inactive_object_metadata = _unified_cross_result.get("inactive_object_metadata")
            precomputed_inactive_element_masks = _unified_cross_result.get("inactive_element_masks")
            precomputed_inactive_element_metadata = _unified_cross_result.get("inactive_element_metadata")
        inactive_elem_pts, inactive_elem_cols, inactive_obj_pts, inactive_obj_cols, _inactive_prior_metadata = build_inactive_label_priors(
            previous_inactive_clip,
            elem_label,
            final_obj_label,
            config,
            K_depth,
            K_color,
            D_color,
            T_depth_color,
            T_cam_depth,
            T_cam_color,
            color_w,
            color_h,
            depth_cache_dir,
            element_anchor_box_norm=None,
            object_anchor_box_norm=None,
            object_crop_box_norm=None,
            debug_dir=clip_dir,
            sam3_predictor=sam3_predictor,
            preferred_hand=clip.interaction_hand,
            active_sam_rgb_list=sam_rgb_list,
            active_prompt_frame_idx=object_prompt_frame_idx,
            active_object_prompt_box_norm=active_object_prompt_box_norm,
            precomputed_object_masks=precomputed_inactive_object_masks,
            precomputed_object_confidence_metadata=precomputed_inactive_object_metadata,
            precomputed_element_masks=precomputed_inactive_element_masks,
            precomputed_element_confidence_metadata=precomputed_inactive_element_metadata,
            precomputed_hand_masks=(_unified_cross_result.get("inactive_hand_masks") if _unified_cross_result is not None else None),
        )
    active_clip_points, active_clip_colors = stack_semantic_point_blocks(
        [final_element_pts, final_object_semantic_pts],
        [elem_label, final_obj_label],
    )
    if len(active_clip_points) == 0:
        active_clip_points, active_clip_colors = stack_semantic_point_blocks(
            [init_element_pts, init_object_semantic_pts],
            [elem_label, final_obj_label],
        )
    mask_preview_points, _mask_preview_colors = stack_point_blocks(
        [pts for pts in (final_element_pts, final_object_semantic_pts) if len(pts)],
        [constant_point_colors(pts) for pts in (final_element_pts, final_object_semantic_pts) if len(pts)],
    )
    if len(mask_preview_points) == 0:
        mask_preview_points, _mask_preview_colors = stack_point_blocks(
            [pts for pts in (init_element_pts, init_object_semantic_pts) if len(pts)],
            [constant_point_colors(pts) for pts in (init_element_pts, init_object_semantic_pts) if len(pts)],
        )
    write_point_cloud_snapshot(
        clip_dir / "sam3_mask_reconstruction_white.ply",
        mask_preview_points,
        constant_point_colors(mask_preview_points),
    )
    mask_preview_semantic_points, mask_preview_semantic_colors = stack_semantic_point_blocks(
        [pts for pts in (final_element_pts, final_object_semantic_pts) if len(pts)],
        [elem_label, final_obj_label],
    )
    if len(mask_preview_semantic_points) == 0:
        mask_preview_semantic_points, mask_preview_semantic_colors = stack_semantic_point_blocks(
            [pts for pts in (init_element_pts, init_object_semantic_pts) if len(pts)],
            [elem_label, final_obj_label],
        )
    write_point_cloud_snapshot(
        clip_dir / "sam3_mask_reconstruction_maskcolors.ply",
        mask_preview_semantic_points,
        mask_preview_semantic_colors,
    )
    create_functional_edge = relationship["relationship"] != "unknown"

    prev_elem_node, prev_elem_match_debug = scene_graph.match_node_by_geometry(
        elem_label,
        "element",
        inactive_elem_pts,
        centroid_hint=(np.asarray(inactive_elem_pts, dtype=np.float32).mean(axis=0) if len(inactive_elem_pts) else None),
    )
    prev_obj_node, prev_obj_match_debug = scene_graph.match_node_by_geometry(
        final_obj_label,
        "object",
        inactive_obj_pts,
        centroid_hint=(np.asarray(inactive_obj_pts, dtype=np.float32).mean(axis=0) if len(inactive_obj_pts) else None),
    )
    print(f"  scene-graph element match: {summarize_scene_graph_match(prev_elem_match_debug)}")
    print(f"  scene-graph object match: {summarize_scene_graph_match(prev_obj_match_debug)}")

    if previous_inactive_clip is None:
        msg = (
            f"[ActiveClip][Error] clip {clip.clip_id}: no preceding inactive clip; "
            "graph-prior fallback is disabled."
        )
        print(msg)
        raise RuntimeError(msg)

    # (C) Element+object prior union for rigid-transport interactions.
    # When the object label is the same rigid body as the element (e.g.
    # handle + drawer, lid + container), Qwen sets
    # ``shares_rigid_motion=True`` and the two priors describe the SAME
    # 3-D body from different SAM3 prompts. If the object prior came back
    # sparse (e.g. SAM3 couldn't ground "drawer" on a closed drawer, or
    # depth coverage on the parent was too sparse), we concatenate the
    # element prior into the object prior — duplicate/overlapping points
    # are harmless on a single rigid body and downstream voxel downsample
    # dedupes anyway. Matches the existing ``len == 0`` union fallback at
    # the shared-motion branch below. Unblocks 183 drawer / 307_2200
    # container without touching SAM3.
    shares_rigid_motion = bool(motion_coupling.get("shares_rigid_motion", False))
    obj_sparse = len(inactive_obj_pts) < int(MIN_TRACKED_POINTS_FOR_REUSE)
    elem_healthy = len(inactive_elem_pts) >= int(MIN_TRACKED_POINTS_FOR_REUSE)
    if obj_sparse and shares_rigid_motion and elem_healthy:
        merged_obj_pts, merged_obj_cols = stack_point_blocks(
            [pts for pts in (inactive_obj_pts, inactive_elem_pts) if len(pts)],
            [cols for cols in (inactive_obj_cols, inactive_elem_cols) if len(cols)],
        )
        print(
            f"  [PriorFallback] object prior sparse ({len(inactive_obj_pts)} pts) for "
            f"'{final_obj_label}' but shares_rigid_motion=True with '{elem_label}' "
            f"({len(inactive_elem_pts)} pts) — unioning into {len(merged_obj_pts)} pts."
        )
        inactive_obj_pts = merged_obj_pts.astype(np.float32, copy=False)
        inactive_obj_cols = merged_obj_cols.astype(np.float32, copy=False)

    allow_missing_element_prior = bool(
        motion_coupling.get("shares_rigid_motion", False) and len(inactive_obj_pts) > 0 and len(inactive_elem_pts) == 0
    )
    if allow_missing_element_prior:
        print(
            f"  [PriorFallback] element prior for '{elem_label}' is empty but shared-motion object prior "
            f"for '{final_obj_label}' has {len(inactive_obj_pts)} points; continuing with zero-point element geometry."
        )
    if (len(inactive_elem_pts) == 0 and not allow_missing_element_prior) or len(inactive_obj_pts) == 0:
        msg = (
            f"[ActiveClip][Error] inactive clip {previous_inactive_clip.clip_id} produced sparse priors "
            f"({len(inactive_elem_pts)} / {len(inactive_obj_pts)} points) for '{elem_label}' / '{final_obj_label}'."
        )
        print(msg)
        raise RuntimeError(msg)

    if motion_coupling["shares_rigid_motion"]:
        element_reference_pts = np.asarray(inactive_elem_pts, dtype=np.float32)
        element_reference_cols = np.asarray(inactive_elem_cols, dtype=np.float32)
        if len(element_reference_pts) == 0 and prev_elem_node is not None and len(prev_elem_node.points):
            element_reference_pts = prev_elem_node.points.copy().astype(np.float32)
            element_reference_cols = prev_elem_node.colors.copy().astype(np.float32)
        object_reference_pts = np.asarray(inactive_obj_pts, dtype=np.float32)
        object_reference_cols = np.asarray(inactive_obj_cols, dtype=np.float32)
        if len(object_reference_pts) == 0 and prev_obj_node is not None and len(prev_obj_node.points):
            object_reference_pts = prev_obj_node.points.copy().astype(np.float32)
            object_reference_cols = prev_obj_node.colors.copy().astype(np.float32)
        tracking_reference_pts, tracking_reference_cols = stack_point_blocks(
            [pts for pts in (object_reference_pts, element_reference_pts) if len(pts)],
            [cols for cols in (object_reference_cols, element_reference_cols) if len(cols)],
        )
        if len(tracking_reference_pts) == 0 and len(init_shared_motion_pts):
            tracking_reference_pts = np.asarray(init_shared_motion_pts, dtype=np.float32)
            tracking_reference_cols = np.asarray(init_shared_motion_cols, dtype=np.float32)
        if len(tracking_reference_pts) == 0:
            tracking_reference_pts, tracking_reference_cols = stack_point_blocks(
                [pts for pts in (inactive_obj_pts, inactive_elem_pts) if len(pts)],
                [cols for cols in (inactive_obj_cols, inactive_elem_cols) if len(cols)],
            )
        shared_prev_node = prev_obj_node if prev_obj_node is not None and len(prev_obj_node.points) else prev_elem_node
        shared_is_articulated = is_truly_articulated_motion
        shared_prior_points = tracking_reference_pts
        shared_prior_colors = tracking_reference_cols
        # Both articulated parts and rigid carries use the SAME mask-seeded
        # CoTracker3 frontend; only the pose solver differs (joint fit vs PnP).
        shared_pose_estimation = "articulation" if is_truly_articulated_motion else "rigid_pnp"
        if use_active_object_sam3:
            if shared_is_articulated:
                print(
                    f"  using direct Artipoint articulation tracking for shared '{elem_label}' + "
                    f"'{final_obj_label}' motion from {len(shared_prior_points)} prior points"
                )
            else:
                print(
                    f"  using rigid 3D-3D tracking (mask-seeded CoTracker3 + depth-lifted "
                    f"per-frame Kabsch/RANSAC, LostFound geometry, no joint fit) for shared "
                    f"'{elem_label}' + '{final_obj_label}' motion from {len(shared_prior_points)} prior points"
                )

            # Build depth-camera backprojection grid once (used by the native
            # depth lift). Reuse the same K_depth + intrinsics the prior cloud
            # was built with, so both lifts produce consistent world points.
            active_depth_grid = None
            if depth_raw_list and len(depth_raw_list) > 0:
                active_depth_grid = build_depth_backprojection_grid(
                    np.asarray(K_depth, dtype=np.float64),
                    depth_raw_list[0].shape[0],
                    depth_raw_list[0].shape[1],
                )
            # Exclude the hand/arm from the shared motion query masks before
            # ArtiPoint seeds corners. We track one combined element+object
            # motion hypothesis, but keep the element mask separately so its own
            # keypoints can be contributed alongside the object-body keypoints.
            # This avoids the current failure mode where all retained tracks come
            # from the handle alone while still exporting separate geometries.
            hand_dilate_kernel = np.ones((25, 25), dtype=np.uint8)
            artipoint_query_masks: List[np.ndarray] = []
            artipoint_element_masks: List[np.ndarray] = []
            for motion_mask, elem_mask, hand_mask in zip(shared_motion_masks, elem_masks, detector_hand_masks):
                obj_bool = np.asarray(motion_mask, dtype=bool)
                elem_bool = np.asarray(elem_mask, dtype=bool)
                hand_bool = np.asarray(hand_mask, dtype=bool)
                if np.any(hand_bool) and np.any(obj_bool):
                    hand_dilated = cv2.dilate(hand_bool.astype(np.uint8), hand_dilate_kernel, iterations=1) > 0
                    cleaned = obj_bool & ~hand_dilated
                    cleaned_elem = elem_bool & ~hand_dilated
                    if np.any(cleaned) and primary_hand_anchor_xy is not None:
                        cleaned = keep_mask_component_nearest_point(cleaned, primary_hand_anchor_xy)
                    if np.any(cleaned_elem) and primary_hand_anchor_xy is not None:
                        cleaned_elem = keep_mask_component_nearest_point(cleaned_elem, primary_hand_anchor_xy)
                    artipoint_query_masks.append(np.asarray(cleaned, dtype=bool))
                    artipoint_element_masks.append(np.asarray(cleaned_elem, dtype=bool))
                else:
                    artipoint_query_masks.append(obj_bool.copy())
                    artipoint_element_masks.append(elem_bool.copy())
            n_emptied = sum(
                1
                for before, after in zip(shared_motion_masks, artipoint_query_masks)
                if np.any(np.asarray(before, dtype=bool)) and not np.any(after)
            )
            print(
                f"  ArtiPoint query masks: subtracted hand/arm from shared element+object motion masks "
                f"({n_emptied} frame(s) became hand-only and will be skipped)"
            )
            # Per-frame object-anchor box (handle-anchored, recomputed from EACH
            # frame's element mask) for the keypoint debug overlay — so the blue
            # box tracks the handle through the clip instead of showing the single
            # frozen prompt-frame box. None on frames without an element mask.
            per_frame_object_boxes: List[Optional[Tuple[int, int, int, int]]] = []
            for _fi in range(len(rgb_list)):
                _em = np.asarray(elem_masks[_fi], dtype=bool) if _fi < len(elem_masks) else None
                if _em is not None and np.any(_em):
                    per_frame_object_boxes.append(
                        select_object_anchor_box(
                            clip.samples[_fi],
                            rgb_list[_fi].shape,
                            _em,
                            fallback_object_box=None,
                            hint_object_box=None,
                        )
                    )
                else:
                    per_frame_object_boxes.append(None)
            # No temporal striding/downsampling: every available active frame is
            # tracked at full resolution (both rigid_pnp and articulated). CoTracker3
            # memory is kept in check by SHORTENING the active clip at the launcher
            # (earlier --frame-end) rather than dropping frames, so SAM masks and the
            # rigid/articulated trajectory keep their full temporal density.
            _trk_clip, _trk_rgb, _trk_depths, _trk_T_wc = clip, rgb_list, clip_depths, T_wc_list
            _trk_query, _trk_elem, _trk_depth_raw, _trk_boxes = (
                artipoint_query_masks, artipoint_element_masks, depth_raw_list, per_frame_object_boxes,
            )
            # SAM3 stays resident; online CoTracker fits alongside it, so no
            # per-clip SAM3 teardown (which fragmented GPU memory) is needed.
            shared_lostfound, shared_lostfound_debug = run_artipoint_articulation_update(
                _trk_clip,
                final_obj_label,
                shared_prior_points,
                shared_prior_colors,
                _trk_rgb,
                _trk_depths,
                _trk_T_wc,
                K_color,
                T_cam_color,
                active_query_masks=_trk_query,
                depth_raw_list=_trk_depth_raw,
                depth_grid=active_depth_grid,
                K_depth=K_depth,
                T_cam_depth=T_cam_depth,
                D_color=D_color,
                debug_dir=clip_dir / "tracking_debug",
                object_prompt_boxes=_trk_boxes,
                pose_estimation=shared_pose_estimation,
                secondary_query_masks=_trk_elem,
                hand_masks=detector_hand_masks,
            )
            shared_prior_source = (
                "artipoint_inactive_clip" if shared_is_articulated else "rigid_pnp_inactive_clip"
            )
        else:
            shared_lostfound, shared_prior_source, shared_lostfound_debug = run_lostfound_geometry_update(
                clip,
                final_obj_label,
                shared_prev_node,
                shared_prior_points,
                shared_prior_colors,
                rgb_list,
                T_wc_list,
                K_color,
                T_cam_color,
            )
        if shared_lostfound is None:
            tracker_name = "Artipoint" if shared_is_articulated else ("RigidPnP" if use_active_object_sam3 else "LostFound")
            print(f"[Error] Shared-motion {tracker_name} failed for '{elem_label}' + '{final_obj_label}' in clip {clip.clip_id}; skipping active clip.")
            export_lostfound_debug_assets(
                clip,
                config.output_dir,
                final_obj_label,
                shared_prior_points,
                rgb_list,
                T_wc_list,
                K_color,
                T_cam_color,
                asset_prefix="shared",
            )
            save_lostfound_failure_debug(
                clip,
                config.output_dir,
                elem_label,
                final_obj_label,
                shared_lostfound_debug,
                shared_lostfound_debug,
            )
            return active_clip_points, active_clip_colors
        shared_delta = np.asarray(shared_lostfound["pose_delta"], dtype=np.float32)
        element_pts = apply_homogeneous_transform(element_reference_pts, shared_delta)
        element_cols = np.asarray(element_reference_cols, dtype=np.float32)
        object_pts = apply_homogeneous_transform(object_reference_pts, shared_delta)
        object_cols = np.asarray(object_reference_cols, dtype=np.float32)
        export_tracking_debug_assets(
            clip,
            config.output_dir,
            final_obj_label,
            shared_prior_points,
            object_pts,
            debug_payload=shared_lostfound_debug,
            asset_prefix="shared",
        )
        element_lostfound = shared_lostfound
        object_lostfound = shared_lostfound
        element_prior_source = shared_prior_source
        object_prior_source = shared_prior_source
        element_lostfound_debug = dict(shared_lostfound_debug)
        object_lostfound_debug = dict(shared_lostfound_debug)
        element_motion = build_motion_record_from_tracking_reference(
            clip,
            shared_lostfound,
            shared_prior_source,
            element_reference_pts,
            element_reference_cols,
        )
        object_motion = build_motion_record_from_tracking_reference(
            clip,
            shared_lostfound,
            shared_prior_source,
            object_reference_pts,
            object_reference_cols,
        )
        if shared_lostfound.get("tracker") == "artipoint_direct":
            element_motion = attach_articulation_metadata(element_motion, shared_lostfound)
            object_motion = attach_articulation_metadata(object_motion, shared_lostfound)
        else:
            element_motion = apply_articulation_constraint(
                element_motion,
                label=f"shared_element/{elem_label}",
                element_label=elem_label,
                object_label=final_obj_label,
                motion_coupling=motion_coupling,
            )
            object_motion = apply_articulation_constraint(
                object_motion,
                label=f"shared_object/{final_obj_label}",
                element_label=elem_label,
                object_label=final_obj_label,
                motion_coupling=motion_coupling,
            )
    else:
        element_lostfound, element_prior_source, element_lostfound_debug = run_lostfound_geometry_update(
            clip,
            elem_label,
            prev_elem_node,
            inactive_elem_pts,
            inactive_elem_cols,
            rgb_list,
            T_wc_list,
            K_color,
            T_cam_color,
        )
        if element_lostfound is None:
            print(f"[Error] LostFound failed for element {elem_label} in clip {clip.clip_id}; fallback registration is disabled.")
            export_lostfound_debug_assets(
                clip,
                config.output_dir,
                elem_label,
                inactive_elem_pts,
                rgb_list,
                T_wc_list,
                K_color,
                T_cam_color,
                asset_prefix="element",
            )
            save_lostfound_failure_debug(
                clip,
                config.output_dir,
                elem_label,
                final_obj_label,
                element_lostfound_debug,
                None,
            )
            return active_clip_points, active_clip_colors
        else:
            element_pts = np.asarray(element_lostfound["final_points"], dtype=np.float32)
            element_cols = np.asarray(element_lostfound["final_colors"], dtype=np.float32)
            export_tracking_debug_assets(
                clip,
                config.output_dir,
                elem_label,
                inactive_elem_pts,
                element_pts,
                debug_payload=element_lostfound_debug,
                asset_prefix="element",
            )
            element_motion = build_motion_record_from_tracking(clip, element_lostfound, element_prior_source)
            element_motion = apply_articulation_constraint(
                element_motion,
                label=f"element/{elem_label}",
                element_label=elem_label,
                object_label=final_obj_label,
                motion_coupling=motion_coupling,
            )

        static_parent_object = bool(motion_coupling.get("static_parent_body", False))
        if static_parent_object:
            object_prior_source = "static_parent_inactive_clip"
            object_pts = np.asarray(inactive_obj_pts, dtype=np.float32)
            object_cols = np.asarray(inactive_obj_cols, dtype=np.float32)
            if len(object_pts) == 0 and prev_obj_node is not None and len(prev_obj_node.points):
                object_prior_source = "static_parent_scene_graph"
                object_pts = prev_obj_node.points.copy().astype(np.float32)
                object_cols = prev_obj_node.colors.copy().astype(np.float32)
            if len(object_pts) == 0 and len(init_object_semantic_pts):
                object_prior_source = "static_parent_active_initial_mask"
                object_pts = np.asarray(init_object_semantic_pts, dtype=np.float32)
                object_cols = np.asarray(init_object_semantic_cols, dtype=np.float32)
            if len(object_pts) == 0 and len(final_object_semantic_pts):
                object_prior_source = "static_parent_active_final_mask"
                object_pts = np.asarray(final_object_semantic_pts, dtype=np.float32)
                object_cols = constant_point_colors(object_pts)
            object_lostfound = {
                "hand": None,
                "hand_anchor_mode": None,
                "pose_updates": 0,
                "translation_m": 0.0,
                "rotation_deg": 0.0,
                "used_frames": [],
            }
            object_lostfound_debug = {
                "label": final_obj_label,
                "clip_id": int(clip.clip_id),
                "status": "static_parent_prior",
                "prior_source": object_prior_source,
                "point_count": int(len(object_pts)),
                "reason": motion_coupling.get("reasoning", ""),
            }
            object_motion = None
            print(
                f"  keeping parent object '{final_obj_label}' static for articulated/control interaction "
                f"using {object_prior_source} ({len(object_pts)} points)"
            )
            export_tracking_debug_assets(
                clip,
                config.output_dir,
                final_obj_label,
                inactive_obj_pts,
                object_pts,
                debug_payload=object_lostfound_debug,
                asset_prefix="object",
            )
        else:
            object_lostfound, object_prior_source, object_lostfound_debug = run_lostfound_geometry_update(
                clip,
                final_obj_label,
                prev_obj_node,
                inactive_obj_pts,
                inactive_obj_cols,
                rgb_list,
                T_wc_list,
                K_color,
                T_cam_color,
            )
            if object_lostfound is None:
                print(f"[Error] LostFound failed for object '{final_obj_label}' in clip {clip.clip_id}; skipping active clip.")
                export_lostfound_debug_assets(
                    clip,
                    config.output_dir,
                    elem_label,
                    inactive_elem_pts,
                    rgb_list,
                    T_wc_list,
                    K_color,
                    T_cam_color,
                    asset_prefix="element",
                )
                export_lostfound_debug_assets(
                    clip,
                    config.output_dir,
                    final_obj_label,
                    inactive_obj_pts,
                    rgb_list,
                    T_wc_list,
                    K_color,
                    T_cam_color,
                    asset_prefix="object",
                )
                save_lostfound_failure_debug(
                    clip,
                    config.output_dir,
                    elem_label,
                    final_obj_label,
                    element_lostfound_debug,
                    object_lostfound_debug,
                )
                return active_clip_points, active_clip_colors
            object_pts = np.asarray(object_lostfound["final_points"], dtype=np.float32)
            object_cols = np.asarray(object_lostfound["final_colors"], dtype=np.float32)
            export_tracking_debug_assets(
                clip,
                config.output_dir,
                final_obj_label,
                inactive_obj_pts,
                object_pts,
                debug_payload=object_lostfound_debug,
                asset_prefix="object",
            )
            object_motion = build_motion_record_from_tracking(clip, object_lostfound, object_prior_source)
            object_motion = apply_articulation_constraint(
                object_motion,
                label=f"object/{final_obj_label}",
                element_label=elem_label,
                object_label=final_obj_label,
                motion_coupling=motion_coupling,
            )

    semantic_only_element_geometry = False
    update_element_geometry = bool(len(np.asarray(element_pts, dtype=np.float32)))
    record_element_motion = bool(element_motion is not None)
    element_display_centroid = (
        np.asarray(element_pts, dtype=np.float32).mean(axis=0)
        if len(element_pts)
        else np.asarray(object_pts, dtype=np.float32).mean(axis=0)
        if len(object_pts)
        else None
    )
    previous_object_state = (
        str(prev_obj_node.state).strip().lower()
        if prev_obj_node is not None and str(prev_obj_node.state).strip()
        else "unknown"
    )
    qwen_bundle = ensure_qwen_bundle_loaded(qwen_bundle, config)
    object_state_result = infer_object_state_from_video(
        qwen_bundle,
        clip,
        rgb_list,
        config.output_dir,
        previous_inactive_clip,
        K_color,
        D_color,
        elem_label,
        final_obj_label,
        relationship,
        motion_coupling,
        element_desc=element_desc,
        object_desc=object_desc,
        object_motion=object_motion,
        previous_object_state=previous_object_state,
    )

    object_state_before = str(
        object_state_result.get("object_state_before", previous_object_state) or "unknown"
    ).strip().lower()

    object_state_after = str(
        object_state_result.get(
            "object_state_after",
            object_state_result.get("object_state", "unknown"),
        ) or "unknown"
    ).strip().lower()

    object_state_label = str(
        object_state_result.get("object_state", object_state_after) or object_state_after or "unknown"
    ).strip().lower()

    status_changed = bool(object_state_result.get("status_changed", False))

    relationship["object_state_before"] = object_state_before
    relationship["object_state_after"] = object_state_after
    relationship["object_state"] = object_state_label
    relationship["status_changed"] = status_changed
    relationship["status_change_confidence"] = float(object_state_result.get("confidence", 0.0))
    relationship["status_change_reasoning"] = str(object_state_result.get("reasoning", ""))
    relationship["status_video_path"] = str(object_state_result.get("active_video_path", ""))
    relationship["status_change_raw"] = str(object_state_result.get("raw", ""))
    previous_graph = copy.deepcopy(scene_graph)
    elem_node_id, obj_node_id, edge_id = scene_graph.register_interaction(
        element_label=elem_label,
        element_pts=element_pts,
        element_colors=element_cols,
        object_label=final_obj_label,
        object_pts=object_pts,
        object_colors=object_cols,
        relationship=relationship["relationship"] if create_functional_edge else "",
        edge_type="functional",
        confidence=max(float(detection.get("confidence", 0.0)), float(relationship.get("confidence", 0.0)), 0.5),
        clip_id=clip.clip_id,
        frame_start=clip.frame_start,
        frame_end=clip.frame_end,
        cot_reasoning=relationship["reasoning"] if create_functional_edge else "",
        element_state="unknown",
        object_state=object_state_label,
        element_state_reasoning="",
        object_state_reasoning=str(object_state_result.get("reasoning", "")),
        element_desc=element_desc,
        object_desc=object_desc,
        element_label_candidates=element_label_candidates,
        object_label_candidates=object_label_candidates,
        voxel_size=SEM_VOXEL_SIZE,
        replace_element_geometry=True,
        update_element_geometry=update_element_geometry,
        element_centroid_hint=element_display_centroid,
        record_element_motion=record_element_motion,
        replace_object_geometry=True,
        element_motion=element_motion,
        object_motion=object_motion,
        create_edge=create_functional_edge,
        element_node_id=prev_elem_node.node_id if prev_elem_node is not None else None,
        object_node_id=prev_obj_node.node_id if prev_obj_node is not None else None,
        use_label_lookup_for_element=False,
        use_label_lookup_for_object=False,
    )

    export_viser_motion_bundle(
        clip,
        clip_dir,
        final_obj_label,
        object_motion,
        (object_lostfound or element_lostfound or {}).get("hand") if isinstance((object_lostfound or element_lostfound), dict) else None,
        (object_lostfound or element_lostfound or {}).get("hand_anchor_mode") if isinstance((object_lostfound or element_lostfound), dict) else None,
    )

    with open(clip_dir / "event.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "clip_id": clip.clip_id,
                "frame_start": clip.frame_start,
                "frame_end": clip.frame_end,
                "element": elem_label,
                "tracked_object": tracked_obj_label,
                "object": final_obj_label,
                "qwen_parent_body_object": qwen_parent_body_label,
                "parent_object": detection.get("parent_object", tracked_obj_label),
                "relation_scope": detection.get("relation_scope", "local"),
                "detection_reasoning": detection.get("reasoning", ""),
                "qwen_inactive_context_frames": _detection_comparison.get("inactive_context_frames", []),
                "element_description": element_desc,
                "object_description": object_desc,
                "element_captions": element_captions,
                "object_captions": object_captions,
                "element_label_candidates": element_label_candidates,
                "object_label_candidates": object_label_candidates,
                "element_view_indices": element_view_indices,
                "object_view_indices": object_view_indices,
                "primary_hand_anchor_xy": [round(float(primary_hand_anchor_xy[0]), 1), round(float(primary_hand_anchor_xy[1]), 1)],
                "active_interaction_anchor_box_xyxy": [int(v) for v in interaction_anchor_box],
                "active_element_anchor_box_xyxy": [int(v) for v in element_anchor_box] if element_anchor_box is not None else None,
                "active_object_anchor_box_xyxy": [int(v) for v in object_anchor_box] if object_anchor_box is not None else None,
                "active_object_crop_box_xyxy": [int(v) for v in object_crop_box] if object_crop_box is not None else None,
                "active_object_prompt_frame_index": int(object_prompt_frame_idx),
                "functional_relationship": relationship["relationship"] if create_functional_edge else None,
                "functional_confidence": round(float(relationship["confidence"]), 3) if create_functional_edge else None,
                "functional_reasoning": relationship["reasoning"] if create_functional_edge else "",
                "registered_element_node_id": elem_node_id,
                "registered_object_node_id": obj_node_id,
                "registered_edge_id": edge_id,
                "element_node_merged": bool(prev_elem_node is not None and elem_node_id == prev_elem_node.node_id),
                "object_node_merged": bool(prev_obj_node is not None and obj_node_id == prev_obj_node.node_id),
                "element_node_match": prev_elem_match_debug,
                "object_node_match": prev_obj_match_debug,
                "previous_object_state": previous_object_state,

                # graph_frank.py-compatible status-change fields
                "object_state_before": object_state_before,
                "object_state_after": object_state_after,
                "object_state": object_state_label,
                "status_changed": status_changed,
                "status_change_confidence": round(float(relationship.get("status_change_confidence", 0.0)), 3),
                "status_change_reasoning": relationship.get("status_change_reasoning", ""),
                "status_video_path": relationship.get("status_video_path", ""),
                "status_change_raw": relationship.get("status_change_raw", ""),

                # backward-compatible final-state fields
                "object_state_confidence": round(float(object_state_result.get("confidence", 0.0)), 3),
                "object_state_reasoning": object_state_result.get("reasoning", ""),
                "object_state_video_mode": object_state_result.get("mode", ""),
                "object_state_inactive_context_frames": object_state_result.get("inactive_context_frames", []),
                "object_state_active_video_frame_ids": object_state_result.get("active_video_frame_ids", []),
                "object_state_active_video_path": object_state_result.get("active_video_path", ""),
                "shares_rigid_motion": bool(motion_coupling.get("shares_rigid_motion", False)),
                "motion_coupling_confidence": round(float(motion_coupling.get("confidence", 0.0)), 3),
                "motion_coupling_reasoning": motion_coupling.get("reasoning", ""),
                "semantic_only_element_geometry": bool(semantic_only_element_geometry),
                "rigid_tracking_parent_label": None,
                "viser_motion_dir": "viser_motion",
                "element_geometry_update": summarize_lostfound_result(element_lostfound, element_prior_source),
                "object_geometry_update": summarize_lostfound_result(object_lostfound, object_prior_source),
                "element_lostfound_debug": element_lostfound_debug,
                "object_lostfound_debug": object_lostfound_debug,
                "element_motion_frames": [int(sample["frame"]) for sample in (element_motion or {}).get("pose_history", [])],
                "object_motion_frames": [int(sample["frame"]) for sample in (object_motion or {}).get("pose_history", [])],
                "element_articulation": _articulation_summary(element_motion),
                "object_articulation": _articulation_summary(object_motion),
                "graph_diff": scene_graph.diff(previous_graph),
            },
            handle,
            indent=2,
        )

    return active_clip_points, active_clip_colors


def run_pipeline(config: PipelineConfig) -> None:
    """Execute the full two-mode dynamic scene graph pipeline."""
    if config.output_dir.exists():
        shutil.rmtree(config.output_dir)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    depth_scale = load_depth_scale_from_dataset(config.data_dir)
    set_runtime_depth_scale(depth_scale)
    print(f"Using depth scale divisor {depth_scale:.6f} from {config.data_dir / 'camera_depth_unit_of_depth_in_mm'}")
    K_color, D_color, color_w, color_h = load_intrinsic(find_info_json(config.data_dir, "color"))
    K_depth, _D_depth, _depth_w, _depth_h = load_intrinsic(find_info_json(config.data_dir, "depth"))
    T_cam_depth, T_depth_color, T_cam_color = load_static_transform_components(config.resolve_static_tf_path())
    pose_buffer = build_pose_buffer(config.data_dir, T_cam_depth, T_depth_color)

    all_samples = load_all_frames(
        config.data_dir,
        config.frame_start,
        config.frame_end,
        K_color,
        K_depth,
        T_cam_color,
        T_depth_color,
        pose_buffer,
    )
    valid_samples = filter_pose_consistent_samples(all_samples)
    if not valid_samples:
        raise RuntimeError("No frames remain after tf_camera pose gating; reconstruction would be unreliable.")
    clips = segment_clips(valid_samples, config.clip_segmentation_mode)
    with open(config.output_dir / "clips.json", "w", encoding="utf-8") as handle:
        json.dump(
            [
                {
                    "clip_id": clip.clip_id,
                    "frame_start": clip.frame_start,
                    "frame_end": clip.frame_end,
                    "n_frames": len(clip.samples),
                    "type": "inactive" if clip.is_inactive else "active",
                    "hand_ratio": round(float(clip.hand_ratio), 3),
                    "pose_block_id": clip.pose_block_id,
                    "interaction_hand": clip.interaction_hand,
                }
                for clip in clips
            ],
            handle,
            indent=2,
        )
    for clip in clips:
        export_clip_wrist_observation_bundle(clip, config.output_dir)

    global_pcd_pts: List[np.ndarray] = []
    global_pcd_cols: List[np.ndarray] = []
    scene_graph = DynamicSceneGraph()
    scene_suppression_boxes: List[Tuple[np.ndarray, np.ndarray]] = []
    depth_cache_dir: Optional[Path] = None
    qwen_bundle: Optional[QwenBundle] = None
    previous_inactive_clip: Optional[ClipInfo] = None

    if config.use_da3:
        print("DA3 enabled: precomputing clip depths in a dedicated pass.")
        depth_cache_dir = precompute_depth_cache(
            clips,
            config,
            K_color,
            D_color,
            K_depth,
            T_depth_color,
            T_cam_color,
            color_w,
            color_h,
        )
    else:
        print("Using native depth-camera reconstruction for inactive clips and sensor depth for active clips.")

    # Plain SAM3 only grounds on simple noun-phrase prompts; SAM3-I tolerates the
    # elaborate interaction-aware text. Gate the prompt style on the backend.
    _set_sam_prompt_simple(_normalize_sam_backend_name(config.sam_backend) == "sam3")
    print(
        f"SAM text prompt style: {'simple (bare label)' if config.sam_backend == 'sam3' else 'elaborate (interaction-aware)'}"
    )
    print(f"Building shared {config.sam_backend.upper()} video predictor for the whole pipeline...")
    sam3_manager = Sam3Manager(config)
    # SAM3 is built lazily by the manager and RELEASED before each clip's CoTracker
    # pass, so SAM3 and CoTracker never co-reside on the GPU (fixes long-clip OOM).

    for clip in clips:
        depth_raw_list: List[np.ndarray] = []
        clip_depths = None
        rgb_list: List[np.ndarray] = []
        T_wc_list: List[np.ndarray] = []
        geometry_T_wc_list: List[np.ndarray] = []
        if clip.is_inactive:
            # Free Qwen before inactive-clip SAM3 work so both models do not
            # compete for the same GPU memory after an active clip.
            if qwen_bundle is not None:
                release_qwen_bundle(qwen_bundle)
                qwen_bundle = None
            if depth_cache_dir is not None:
                clip_depths = load_clip_depth_cache(depth_cache_dir, clip)
                rgb_list, T_wc_list = load_clip_rgb_pose_frames(clip, K_color, D_color)
            clip_points, clip_colors = process_inactive_clip(
                clip,
                K_depth,
                K_color,
                D_color,
                T_cam_depth,
                T_cam_color,
                global_pcd_pts,
                global_pcd_cols,
                config.jpeg_dir,
                clip_depths=clip_depths,
                rgb_list=rgb_list if clip_depths is not None else None,
                T_wc_list=T_wc_list if clip_depths is not None else None,
                inactive_stride=config.inactive_stride,
                sam3_manager=sam3_manager,
            )
            previous_inactive_clip = clip
        else:
            if depth_cache_dir is not None:
                clip_depths = load_clip_depth_cache(depth_cache_dir, clip)
                rgb_list, depth_raw_list, T_wc_list, _depth_T_wc_list = load_clip_frames_with_dual_poses(clip, K_color, D_color)
                geometry_T_wc_list = list(T_wc_list)
            else:
                rgb_list, depth_raw_list, T_wc_list, geometry_T_wc_list = load_clip_frames_with_dual_poses(
                    clip,
                    K_color,
                    D_color,
                )
                clip_depths = compute_clip_sensor_depths(
                    depth_raw_list,
                    K_depth,
                    K_color,
                    D_color,
                    T_depth_color,
                    color_w,
                    color_h,
                )
            if qwen_bundle is None or qwen_bundle.model is None or qwen_bundle.processor is None:
                qwen_bundle = load_qwen_bundle(config)
            clip_points, clip_colors = process_active_clip(
                clip,
                depth_raw_list,
                clip_depths,
                rgb_list,
                T_wc_list,
                geometry_T_wc_list,
                qwen_bundle,
                scene_graph,
                K_depth,
                K_color,
                D_color,
                T_depth_color,
                T_cam_depth,
                T_cam_color,
                color_w,
                color_h,
                scene_suppression_boxes,
                previous_inactive_clip,
                depth_cache_dir,
                config,
                sam3_manager=sam3_manager,
            )

        scene_points, scene_colors = build_scene_snapshot(global_pcd_pts, global_pcd_cols, scene_graph, scene_suppression_boxes)
        save_clip_snapshot(
            clip,
            clip_points,
            clip_colors,
            scene_points,
            scene_colors,
            scene_graph,
            config.output_dir,
            save_visualization=not clip.is_inactive,
        )
        write_point_cloud_snapshot(config.output_dir / "scene_full.ply", scene_points, scene_colors)
        scene_graph.save(str(config.output_dir / "scene_graph"))
        if not clip.is_inactive:
            save_graph_visualization(scene_graph, scene_points, scene_colors, config.output_dir, clip.clip_id)

        del clip_depths
        del depth_raw_list
        del rgb_list
        del T_wc_list
        del geometry_T_wc_list
        del clip_points
        del clip_colors
        del scene_points
        del scene_colors
        gc.collect()

    if qwen_bundle is not None:
        release_qwen_bundle(qwen_bundle)
        del qwen_bundle
        torch.cuda.empty_cache()
        gc.collect()

    print("Shutting down SAM3 video predictor...")
    sam3_manager.shutdown()
    del sam3_manager
    torch.cuda.empty_cache()
    gc.collect()

    final_scene_points, final_scene_colors = build_scene_snapshot(
        global_pcd_pts,
        global_pcd_cols,
        scene_graph,
        scene_suppression_boxes,
    )
    final_clip_id = clips[-1].clip_id if clips else -1
    save_openfungraph_visualization(
        scene_graph,
        final_scene_points,
        final_scene_colors,
        config.output_dir,
        final_clip_id,
    )

    print(scene_graph.summary())
    print(scene_graph.to_json_for_llm())

def parse_args() -> PipelineConfig:
    """Parse command-line arguments into a PipelineConfig instance."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=PipelineConfig.data_dir)
    parser.add_argument("--da3-model-dir", type=Path, default=PipelineConfig.da3_model_dir)
    parser.add_argument("--output-dir", type=Path, default=PipelineConfig.output_dir)
    parser.add_argument("--jpeg-dir", type=Path, default=PipelineConfig.jpeg_dir)
    parser.add_argument("--frame-start", type=int, default=PipelineConfig.frame_start)
    parser.add_argument("--frame-end", type=int, default=PipelineConfig.frame_end)
    parser.add_argument("--static-tf-path", type=Path, default=None)
    parser.add_argument("--qwen-vl-model-id", default=PipelineConfig.qwen_vl_model_id)
    parser.add_argument("--use-da3", action="store_true", default=PipelineConfig.use_da3)
    parser.add_argument("--debug-alignment-frame", type=int, default=PipelineConfig.debug_alignment_frame)
    parser.add_argument("--hf-cache-dir", default=PipelineConfig.hf_cache_dir)
    parser.add_argument("--sam-backend", choices=["sam3"], default="sam3")
    parser.add_argument("--sam3i-package-root", type=Path, default=PipelineConfig.sam3i_package_root)
    parser.add_argument("--sam3i-checkpoint-path", type=Path, default=PipelineConfig.sam3i_checkpoint_path)
    parser.add_argument("--sam3i-inst-stage", default=PipelineConfig.sam3i_inst_stage)
    parser.add_argument("--sam3i-adapter-dim", type=int, default=PipelineConfig.sam3i_adapter_dim)
    parser.add_argument("--sam3i-adapter-heads", type=int, default=PipelineConfig.sam3i_adapter_heads)
    parser.add_argument("--sam3i-adapter-scale", type=float, default=PipelineConfig.sam3i_adapter_scale)
    parser.add_argument("--local-files-only", action="store_true", default=PipelineConfig.local_files_only)
    parser.add_argument("--allow-downloads", action="store_false", dest="local_files_only")
    parser.add_argument("--use-cot", action="store_true", default=PipelineConfig.use_cot)
    parser.add_argument("--no-cot", action="store_false", dest="use_cot")
    parser.add_argument("--cot-max-tokens", type=int, default=PipelineConfig.cot_max_tokens)
    parser.add_argument("--qwen-max-tokens", type=int, default=PipelineConfig.qwen_max_tokens)
    parser.add_argument("--use-openfungraph-crops", action="store_true", default=PipelineConfig.use_openfungraph_crops)
    parser.add_argument(
        "--clip-segmentation-mode",
        choices=["smoothed", "strict_lostfound", "auto"],
        default=PipelineConfig.clip_segmentation_mode,
    )
    parser.add_argument("--inactive-stride", type=int, default=PipelineConfig.inactive_stride)
    parser.add_argument("--inactive-prior-frames", type=int, default=PipelineConfig.inactive_prior_frames, help="Number of inactive-prior SAM3 frames to sample; use 0 for all strided frames up to the prompt frame.")
    args = parser.parse_args()
    return PipelineConfig(
        data_dir=args.data_dir,
        da3_model_dir=args.da3_model_dir,
        output_dir=args.output_dir,
        jpeg_dir=args.jpeg_dir,
        qwen_vl_model_id=args.qwen_vl_model_id,
        use_da3=args.use_da3,
        debug_alignment_frame=args.debug_alignment_frame,
        use_cot=args.use_cot,
        cot_max_tokens=args.cot_max_tokens,
        qwen_max_tokens=args.qwen_max_tokens,
        local_files_only=args.local_files_only,
        hf_cache_dir=args.hf_cache_dir,
        sam_backend=args.sam_backend,
        sam3i_package_root=args.sam3i_package_root,
        sam3i_checkpoint_path=args.sam3i_checkpoint_path,
        sam3i_inst_stage=args.sam3i_inst_stage,
        sam3i_adapter_dim=int(args.sam3i_adapter_dim),
        sam3i_adapter_heads=int(args.sam3i_adapter_heads),
        sam3i_adapter_scale=float(args.sam3i_adapter_scale),
        use_openfungraph_crops=args.use_openfungraph_crops,
        clip_segmentation_mode=args.clip_segmentation_mode,
        inactive_stride=max(1, int(args.inactive_stride)),
        inactive_prior_frames=max(0, int(args.inactive_prior_frames)),
        frame_start=args.frame_start,
        frame_end=args.frame_end,
        static_tf_path=args.static_tf_path,
    )


def run_alignment_diagnostics(config: PipelineConfig) -> None:
    """Export side-by-side 2D overlays and one-frame point clouds for depth alignment debugging."""
    if config.debug_alignment_frame is None:
        return
    set_runtime_depth_scale(load_depth_scale_from_dataset(config.data_dir))

    frame_idx = int(config.debug_alignment_frame)
    key = f"{frame_idx:06d}"
    data_dir = config.data_dir
    rgb_frames, rgb_timestamps = build_topic_index(data_dir / "camera_color_image", "*.png")
    depth_frames, depth_timestamps = build_topic_index(data_dir / "camera_depth_image", "*.png")
    pose_frames, pose_timestamps = build_topic_index(data_dir / "tf_camera", "*.json")
    pose_health = load_health_by_timestamp(data_dir / "tf_camera_health", missing_valid_default=False)
    rgb_frame_by_key = {frame.key: frame for frame in rgb_frames}
    rgb_frame = rgb_frame_by_key.get(key)
    if rgb_frame is None:
        raise FileNotFoundError(f"Debug RGB frame {frame_idx} not found under {data_dir / 'camera_color_image'}")
    depth_frame = nearest_topic_frame(depth_frames, depth_timestamps, rgb_frame.timestamp_ns)
    pose_frame = nearest_topic_frame(
        pose_frames,
        pose_timestamps,
        depth_frame.timestamp_ns if depth_frame is not None else rgb_frame.timestamp_ns,
    )
    if depth_frame is None or pose_frame is None:
        raise FileNotFoundError(f"Debug frame {frame_idx} could not be matched across RGB, depth, and pose streams")
    if not pose_health.get(pose_frame.timestamp_ns, False):
        raise RuntimeError(f"Debug frame {frame_idx} falls inside an invalid tf_camera pose region")

    out_dir = config.output_dir / "alignment_debug" / key
    out_dir.mkdir(parents=True, exist_ok=True)

    K_color, D_color, color_w, color_h = load_intrinsic(find_info_json(data_dir, "color"))
    K_depth, _D_depth, _depth_w, _depth_h = load_intrinsic(find_info_json(data_dir, "depth"))
    T_cam_depth, T_depth_color, T_cam_color = load_static_transform_components(config.resolve_static_tf_path())
    pose_buffer = build_pose_buffer(data_dir, T_cam_depth, T_depth_color)

    bgr = cv2.imread(str(rgb_frame.path))
    rgb_raw = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb_undist = undistort_rgb(rgb_raw, K_color, D_color)
    sensor_raw = cv2.imread(str(depth_frame.path), cv2.IMREAD_ANYDEPTH)
    T_wc = rt_to_matrix(*pose_buffer.camera_pose(rgb_frame.timestamp_ns))

    cv2.imwrite(str(out_dir / "rgb_raw.png"), cv2.cvtColor(rgb_raw, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(out_dir / "rgb_undistorted.png"), cv2.cvtColor(rgb_undist, cv2.COLOR_RGB2BGR))

    diagnostics = [
        (rgb_raw, "raw"),
        (rgb_undist, "undistorted"),
    ]

    depth_m = prepare_sensor_depth_for_lifting(
        sensor_raw,
        K_depth,
        T_depth_color,
        K_color,
        D_color,
        color_w,
        color_h,
    )

    for rgb_ref, rgb_tag in diagnostics:
        suffix = f"align_color_undistorted__{rgb_tag}"
        depth_color = colorize_depth(depth_m)
        overlay = overlay_depth_points(rgb_ref, depth_m, stride=18, radius=1)
        rgb_edges = rgb_canny_edges(rgb_ref)
        depth_edges = depth_canny_edges(depth_m)
        edge_overlay = overlay_rgb_depth_edges(rgb_ref, rgb_edges, depth_edges)
        cv2.imwrite(str(out_dir / f"depth_{suffix}.png"), cv2.cvtColor(depth_color, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(out_dir / f"overlay_{suffix}.png"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(out_dir / f"rgb_edges_{suffix}.png"), rgb_edges)
        cv2.imwrite(str(out_dir / f"depth_edges_{suffix}.png"), depth_edges)
        cv2.imwrite(str(out_dir / f"edge_overlay_{suffix}.png"), cv2.cvtColor(edge_overlay, cv2.COLOR_RGB2BGR))

        full_mask = (depth_m > 1e-4) & (depth_m < DEPTH_MAX)
        pts, cols = lift_mask_to_world(full_mask, depth_m, rgb_undist, K_color, T_wc, T_cam_color, DEPTH_MAX)
        write_point_cloud_snapshot(out_dir / f"pointcloud_{suffix}.ply", pts, cols)

    metadata = {
        "frame_index": frame_idx,
        "rgb_raw_path": str(rgb_frame.path),
        "depth_raw_path": str(depth_frame.path),
        "pose_path": str(pose_frame.path),
        "depth_modes": [f"align_color_undistorted__{tag}" for _rgb, tag in diagnostics],
    }
    with open(out_dir / "metadata.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    print(f"[Debug] alignment diagnostics saved to {out_dir}")


def main() -> None:
    """Entrypoint for running the dynamic scene graph pipeline from the CLI."""
    config = parse_args()
    if config.debug_alignment_frame is not None:
        run_alignment_diagnostics(config)
        return
    run_pipeline(config)

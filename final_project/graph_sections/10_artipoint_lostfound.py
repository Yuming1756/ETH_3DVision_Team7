# ======================================================================
# graph.py section: 10_artipoint_lostfound.py
# ArtiPoint articulation update + LostFound single/dual-hand tracking
#
# NOTE: not a standalone module. Loaded by graph.py into a shared
# namespace via exec(); names from earlier sections are in scope here.
# ======================================================================
from __future__ import annotations
def _copy_articulation_fields_from_fit(target: dict, fit: dict) -> dict:
    """Attach articulation-fit metadata to a result or motion record."""
    target["articulation_axis_direction"] = fit.get("axis_direction")
    target["articulation_axis_point"] = fit.get("axis_point")
    target["articulation_joint_type"] = fit.get("joint_type")
    target["articulation_thetas"] = fit.get("thetas")
    target["articulation_residual_mean_m"] = fit.get("mean_residual_m")
    target["articulation_residual_max_m"] = fit.get("max_residual_m")
    target["articulation_singular_ratio"] = fit.get("singular_ratio")
    target["articulation_xi"] = fit.get("xi")
    target["articulation_optimizer"] = fit.get("optimizer")
    target["articulation_solver_path"] = fit.get("solver_path")
    target["articulation_fallback_reason"] = fit.get("fallback_reason")
    target["articulation_pivot_candidates"] = fit.get("pivot_candidates")
    target["articulation_pivot_candidate_scores"] = fit.get("pivot_candidate_scores")
    target["articulation_chosen_pivot_source"] = fit.get("chosen_pivot_source")
    if fit.get("pitch") is not None:
        target["articulation_pitch"] = fit.get("pitch")
    if fit.get("axis_prior") is not None:
        target["articulation_axis_prior"] = fit.get("axis_prior")
    if fit.get("articulation_axis_candidates") is not None:
        target["articulation_axis_candidates"] = list(fit.get("articulation_axis_candidates") or [])
    return target


def attach_articulation_metadata(motion_record: Optional[dict], source: Optional[dict]) -> Optional[dict]:
    """Copy articulation metadata from a tracker result into a motion record."""
    if motion_record is None or source is None:
        return motion_record
    copied = dict(motion_record)
    for key in (
        "articulation_axis_direction",
        "articulation_axis_point",
        "articulation_joint_type",
        "articulation_thetas",
        "articulation_residual_mean_m",
        "articulation_residual_max_m",
        "articulation_singular_ratio",
        "articulation_xi",
        "articulation_optimizer",
        "articulation_solver_path",
        "articulation_fallback_reason",
        "articulation_pitch",
        "articulation_axis_prior",
        "articulation_pivot_candidates",
        "articulation_pivot_candidate_scores",
        "articulation_chosen_pivot_source",
        "articulation_axis_candidates",
    ):
        if key in source:
            copied[key] = source[key]
    return copied


def run_artipoint_articulation_update(
    clip: ClipInfo,
    label: str,
    initial_points: np.ndarray,
    initial_colors: np.ndarray,
    rgb_list: Sequence[np.ndarray],
    clip_depths: Sequence[np.ndarray],
    T_wc_list: Sequence[np.ndarray],
    K_color: np.ndarray,
    T_cam_color: np.ndarray,
    active_query_masks: Optional[Sequence[np.ndarray]] = None,
    depth_raw_list: Optional[Sequence[np.ndarray]] = None,
    depth_grid: Optional[np.ndarray] = None,
    K_depth: Optional[np.ndarray] = None,
    T_cam_depth: Optional[np.ndarray] = None,
    D_color: Optional[np.ndarray] = None,
    debug_dir: Optional[Path] = None,
    object_prompt_boxes: Optional[Sequence[Optional[Sequence[float]]]] = None,
) -> Tuple[Optional[dict], dict]:
    """Direct Artipoint-style articulation path for 1-DoF moving parts."""
    points = np.asarray(initial_points, dtype=np.float32)
    colors = np.asarray(initial_colors, dtype=np.float32)
    debug = {
        "label": str(label),
        "clip_id": int(clip.clip_id),
        "tracker": "artipoint_direct",
        "status": "failed",
        "failure_reason": "",
        "initial_point_count": int(len(points)),
        "selected_hand": None,
        "hand_attempts": [],
    }
    if len(points) < ARTIPOINT_MIN_TRACK_POINTS:
        debug["failure_reason"] = "insufficient_initial_points"
        return None, debug
    if CoTracker3Predictor is None or not COTRACKER3_AVAILABLE or not torch.cuda.is_available():
        debug["failure_reason"] = "cotracker3_or_cuda_unavailable"
        return None, debug
    checkpoint_path = _artipoint_cotracker_checkpoint()
    if checkpoint_path is None:
        debug["failure_reason"] = "cotracker3_offline_checkpoint_missing"
        return None, debug
    debug["cotracker_version"] = "cotracker3_offline"
    debug["cotracker_checkpoint"] = str(checkpoint_path)
    if len(rgb_list) == 0 or len(clip_depths) == 0 or len(T_wc_list) == 0:
        debug["failure_reason"] = "missing_rgb_depth_or_pose"
        return None, debug

    # Active object-mask keypoint seeding: use the propagated SAM3 object mask
    # on active frames, then detect Shi-Tomasi corners inside those masks.
    # This restores the active-clip frontend while forcing the query source to
    # always be the object / moving-part mask rather than the element mask.
    def _artipoint_sample_hand_position(sample_index: int) -> Optional[np.ndarray]:
        sample = clip.samples[int(sample_index)]
        preferred = getattr(clip, "interaction_hand", None)
        hand_order = [preferred] if preferred in ("left", "right") else ["right", "left"]
        for hand_name in hand_order:
            for key in (f"{hand_name}_hand_world_wrist", f"{hand_name}_hand_world"):
                value = sample.get(key)
                if value is None:
                    continue
                arr = np.asarray(value, dtype=np.float32).reshape(-1)
                if arr.size == 3 and np.all(np.isfinite(arr)):
                    return arr.astype(np.float32)
        return None

    n_video_frames = int(len(rgb_list))
    if n_video_frames < 2:
        debug["failure_reason"] = "active_clip_too_short"
        return None, debug
    candidate_keyframes = list(range(n_video_frames))

    def _masks_usable(masks_seq) -> bool:
        if masks_seq is None:
            return False
        if len(masks_seq) < max(candidate_keyframes) + 1:
            return False
        return any(int(np.count_nonzero(np.asarray(m, dtype=bool))) >= 16 for m in masks_seq)

    if not _masks_usable(active_query_masks):
        debug["failure_reason"] = "missing_active_query_masks"
        print(
            f"[ArtiPoint-Articulation][Error] no usable active object SAM3 masks for '{label}'; "
            "active object-mask keypoint seeding requires propagated object masks."
        )
        return None, debug

    # Only seed corners on frames whose object mask is actually usable. The
    # propagated SAM3 object mask can be empty/too-small on some active frames
    # (e.g. a drawer body briefly occluded by the hand mid-pull), and
    # detect_corner_queries_in_image deliberately RAISES on a <16px mask
    # (fallback-free contract). Iterating over those frames crashes the whole
    # clip even when most frames are healthy, so we drop the unusable keyframes
    # here and keep the rest. best_start_idx then resolves to the first usable
    # frame, which is also what the active rest-pose surface anchor expects.
    usable_keyframes = [
        kf for kf in candidate_keyframes
        if kf < len(active_query_masks)
        and int(np.count_nonzero(np.asarray(active_query_masks[kf], dtype=bool))) >= 16
    ]
    n_dropped_keyframes = len(candidate_keyframes) - len(usable_keyframes)
    candidate_keyframes = usable_keyframes
    debug["seeding_keyframe_count"] = int(len(candidate_keyframes))
    debug["dropped_empty_mask_keyframes"] = int(n_dropped_keyframes)
    if n_dropped_keyframes:
        print(
            f"[ArtiPoint-Articulation] '{label}': skipping {n_dropped_keyframes} active "
            f"frame(s) with empty/too-small object masks; seeding from "
            f"{len(candidate_keyframes)} usable frame(s)."
        )

    debug["tracking_point_count"] = int(len(points))
    debug["query_mask_frame_count"] = int(len(active_query_masks)) if active_query_masks is not None else 0

    # Build native depth-camera world-position lookups on demand. Materials
    # needed: depth_raw_list + depth_grid + K_depth + T_cam_depth + T_cam_color
    # + K_color + D_color + color_w/h. When ALL are present we use Path 1
    # convention; otherwise we fall back to the color-aligned shortcut.
    color_h, color_w = rgb_list[0].shape[:2]
    can_native_lift = (
        depth_raw_list is not None
        and depth_grid is not None
        and K_depth is not None
        and T_cam_depth is not None
        and len(depth_raw_list) >= len(rgb_list)
    )
    if can_native_lift and depth_grid is None and K_depth is not None and depth_raw_list:
        depth_grid = build_depth_backprojection_grid(
            np.asarray(K_depth, dtype=np.float64),
            depth_raw_list[0].shape[0],
            depth_raw_list[0].shape[1],
        )

    _world_lookup_cache: Dict[int, np.ndarray] = {}

    def _world_lookup_for(frame_idx: int) -> Optional[np.ndarray]:
        if not can_native_lift:
            return None
        if frame_idx in _world_lookup_cache:
            return _world_lookup_cache[frame_idx]
        lookup = build_color_pixel_world_lookup(
            depth_raw=depth_raw_list[frame_idx],
            depth_grid=depth_grid,
            T_wc=np.asarray(T_wc_list[frame_idx]),
            T_cam_depth=np.asarray(T_cam_depth),
            T_cam_color=np.asarray(T_cam_color),
            K_color=np.asarray(K_color),
            D_color=np.asarray(D_color) if D_color is not None else None,
            color_w=color_w,
            color_h=color_h,
        )
        _world_lookup_cache[frame_idx] = lookup
        return lookup

    class _WorldLookupSeq:
        """Lazy proxy: indexing builds-and-caches on demand so we never hold
        all 180 × (1408² × 3 floats) in RAM at once."""
        def __init__(self, n_frames: int):
            self._n = int(n_frames)
        def __len__(self) -> int:
            return self._n
        def __getitem__(self, idx: int):
            return _world_lookup_for(int(idx))

    world_lookups = _WorldLookupSeq(len(rgb_list)) if can_native_lift else None
    using_native_lift = world_lookups is not None
    debug["lift_convention"] = (
        "native_depth_camera[path1]" if using_native_lift else "color_aligned[path2]"
    )

    per_keyframe_query_counts: List[int] = []
    all_queries_2d_list: List[np.ndarray] = []
    all_queries_world_list: List[np.ndarray] = []
    for kf_idx in candidate_keyframes:
        explicit_mask = np.asarray(active_query_masks[kf_idx], dtype=bool)
        world_lookup_kf = world_lookups[kf_idx] if using_native_lift else None
        queries_2d_kf, queries_world_kf = detect_corner_queries_in_image(
            points,
            rgb_list[kf_idx],
            clip_depths[kf_idx],
            T_wc_list[kf_idx],
            K_color,
            T_cam_color,
            max_corners=int(ARTIPOINT_MAX_CORNERS_PER_KEYFRAME),
            hand_world=_artipoint_sample_hand_position(kf_idx),
            explicit_mask=explicit_mask,
            world_lookup=world_lookup_kf,
        )
        if debug_dir is not None:
            try:
                _save_artipoint_keypoints_imagery(
                    debug_dir,
                    kf_idx,
                    rgb_list[kf_idx],
                    queries_2d_kf,
                    label,
                    active_mask=explicit_mask,
                    prompt_box=(
                        object_prompt_boxes[kf_idx]
                        if object_prompt_boxes is not None and kf_idx < len(object_prompt_boxes)
                        else None
                    ),
                )
            except Exception as exc:  # noqa: BLE001
                print(f"[ArtiPoint-Debug] keypoint overlay save failed: {exc}")
        n_kf = int(len(queries_2d_kf))
        per_keyframe_query_counts.append(n_kf)
        if n_kf == 0:
            continue
        queries_2d_kf = np.asarray(queries_2d_kf, dtype=np.float32).copy()
        queries_2d_kf[:, 0] = float(kf_idx)
        all_queries_2d_list.append(queries_2d_kf)
        all_queries_world_list.append(np.asarray(queries_world_kf, dtype=np.float32))

    best_start_idx = int(candidate_keyframes[0]) if candidate_keyframes else 0
    keyframe_indices = list(candidate_keyframes)
    debug["query_source"] = "goodFeaturesToTrack@image[multi_keyframe+active_object_sam3]"
    debug["keyframe_indices"] = [int(k) for k in keyframe_indices]
    debug["per_keyframe_query_counts"] = per_keyframe_query_counts
    if not all_queries_2d_list:
        debug["projected_query_count"] = 0
        debug["failure_reason"] = "too_few_corner_queries"
        print(
            f"[ArtiPoint-Articulation][Error] no valid corners extracted from active object masks for '{label}'."
        )
        return None, debug
    best_queries = np.concatenate(all_queries_2d_list, axis=0).astype(np.float32)
    best_queries_world = np.concatenate(all_queries_world_list, axis=0).astype(np.float32)
    if len(best_queries) > int(ARTIPOINT_MAX_TOTAL_QUERIES):
        keep_idx = np.linspace(0, len(best_queries) - 1, num=int(ARTIPOINT_MAX_TOTAL_QUERIES), dtype=np.int32)
        best_queries = best_queries[keep_idx]
        best_queries_world = best_queries_world[keep_idx]
    debug["projected_query_count"] = int(len(best_queries))
    debug["query_frame_index"] = int(best_start_idx)
    if len(best_queries) < ARTIPOINT_MIN_VISIBLE_POINTS:
        debug["failure_reason"] = "too_few_corner_queries"
        print(
            f"[ArtiPoint-Articulation][Error] only {len(best_queries)} corners survived from active object masks for '{label}' "
            f"(need >= {ARTIPOINT_MIN_VISIBLE_POINTS})."
        )
        return None, debug
    print(
        f"[ArtiPoint-Queries] seeded {len(best_queries)} active object-mask corners across "
        f"{len(keyframe_indices)} active frames "
        f"(max_feat_points={ARTIPOINT_MAX_CORNERS_PER_KEYFRAME}, "
        f"max_queries={ARTIPOINT_MAX_TOTAL_QUERIES})"
    )
    tracker = None
    video = queries_tensor = pred_tracks = pred_visibility = None
    try:
        # Aggressive cleanup: SAM3 video predictor leaves substantial GPU
        # state in the encoder/decoder buffers even after the session ends.
        # On long active clips (e.g. 307 frames 2200-2390 = 180 frames @
        # 1408² RGB ≈ 4 GB just for the input tensor) CoTracker3 offline can
        # otherwise OOM on intermediate activations.
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        free_before, total = torch.cuda.mem_get_info()
        print(
            f"[ArtiPoint-CoTracker] pre-instantiation GPU: free "
            f"{free_before/2**30:.2f} GiB / total {total/2**30:.2f} GiB, "
            f"video frames={len(rgb_list)} @ {rgb_list[0].shape}"
        )
        # Always CoTracker3, always offline (per user requirement). v2=False
        # selects CoTrackerThreeOffline in build_cotracker; offline=True
        # disables the streaming/online window.
        tracker = CoTracker3Predictor(
            checkpoint=str(checkpoint_path),
            v2=False,
            offline=True,
        ).to("cuda")
        video = torch.from_numpy(np.stack(rgb_list).astype(np.float32)).permute(0, 3, 1, 2)[None].to("cuda")
        queries_tensor = torch.from_numpy(best_queries)[None].to("cuda")
        with torch.inference_mode():
            pred_tracks, pred_visibility = tracker(video, queries=queries_tensor, backward_tracking=True)
        tracks_np = pred_tracks.squeeze(0).detach().cpu().numpy().astype(np.float32)
        visibility_np = pred_visibility.squeeze(0).detach().cpu().numpy().astype(bool)
    except Exception as exc:  # noqa: BLE001
        debug["failure_reason"] = f"cotracker_failed:{exc}"
        return None, debug
    finally:
        if tracker is not None:
            del tracker
        del video
        del queries_tensor
        del pred_tracks
        del pred_visibility
        torch.cuda.empty_cache()
        gc.collect()

    tracks_world, valid_tracks = lift_cotracker_tracks_to_world(
        tracks_np,
        visibility_np,
        clip_depths,
        rgb_list,
        T_wc_list,
        K_color,
        T_cam_color,
        world_lookups=world_lookups,
    )
    prefilter_query_points_world = np.asarray(best_queries_world, dtype=np.float32)
    # Anchor the prior-surface filter to the moving part's ACTIVE rest-pose
    # surface (its object mask on the first usable active frame, lifted with the
    # SAME convention as the corner anchors) rather than the inactive-clip
    # reconstruction. For an articulated part the inactive reconstruction is the
    # pose the part moves AWAY from, and it is also offset by inter-clip
    # registration drift — so checking active-frame proximity to it rejects the
    # very motion we are trying to measure (observed: door tracks a median
    # ~17.6 cm from the inactive surface vs a 6 cm gate -> 0/146 kept). The
    # active rest-pose surface keeps the filter's real job (rejecting
    # hand/background tracks) without penalising articulation.
    surface_anchor_points = points
    try:
        surf_frame = int(best_start_idx)
        surf_mask = np.asarray(active_query_masks[surf_frame], dtype=bool)
        active_surface_points: Optional[np.ndarray] = None
        if using_native_lift:
            surf_lut = world_lookups[surf_frame]
            if surf_lut is not None:
                lut_pts = np.asarray(surf_lut, dtype=np.float32)[surf_mask]
                lut_pts = lut_pts[np.all(np.isfinite(lut_pts), axis=1)]
                if len(lut_pts):
                    active_surface_points = lut_pts
        if active_surface_points is None:
            lift_pts, _lift_cols = lift_mask_to_world(
                surf_mask,
                clip_depths[surf_frame],
                rgb_list[surf_frame],
                K_color,
                T_wc_list[surf_frame],
                T_cam_color,
            )
            if len(lift_pts):
                active_surface_points = np.asarray(lift_pts, dtype=np.float32)
        if active_surface_points is not None and len(active_surface_points) >= ARTIPOINT_MIN_VISIBLE_POINTS:
            surface_anchor_points = active_surface_points
            debug["prior_surface_source"] = "active_rest_pose_frame"
            debug["prior_surface_point_count"] = int(len(active_surface_points))
        else:
            debug["prior_surface_source"] = "inactive_prior_fallback"
    except Exception as exc:  # noqa: BLE001
        debug["prior_surface_source"] = f"inactive_prior_fallback:{exc}"
    tracks_world, valid_tracks, kept_track_mask, track_filter_debug = filter_artipoint_tracks(
        tracks_world,
        valid_tracks,
        query_points_world=prefilter_query_points_world,
        prior_surface_points=surface_anchor_points,
    )
    best_queries_world = best_queries_world[kept_track_mask]
    print(
        f"[ArtiPoint-TrackFilter] label='{label}' kept {int(np.count_nonzero(kept_track_mask))}/"
        f"{int(len(kept_track_mask))} tracks after frontend filtering."
    )
    if debug_dir is not None:
        try:
            _save_artipoint_tracks_imagery(
                debug_dir,
                rgb_list,
                tracks_np[:, kept_track_mask, :],
                visibility_np[:, kept_track_mask],
                label,
                active_masks=active_query_masks,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[ArtiPoint-Debug] filtered track overlay save failed: {exc}")
    debug["track_filtering"] = track_filter_debug
    if len(best_queries_world) < ARTIPOINT_MIN_VISIBLE_POINTS:
        debug["failure_reason"] = "too_few_tracks_after_filtering"
        return None, debug
    # ArtiPoint Eq. (1): smooth the lifted 3D point trajectories before
    # per-frame Procrustes. This is the step that was previously missing and
    # caused the RMSE cliff at frame ~24 in event.json (door swung enough that
    # half the tracks lost visibility, RANSAC fit garbage to the remaining
    # slice). Smoothing carries the early clean 3D positions through the noisy
    # late frames, weighted by visibility, so each frame's Kabsch sees a
    # stable target.
    tracks_world_smoothed = smooth_lifted_world_tracks(tracks_world, valid_tracks)
    debug["track_smoothing"] = {
        "lambda_vel": float(ARTIPOINT_TRACK_SMOOTH_LAMBDA_VEL),
        "lambda_jerk": float(ARTIPOINT_TRACK_SMOOTH_LAMBDA_JERK),
        "applied": True,
    }
    reference_points_world = np.asarray(best_queries_world, dtype=np.float32)
    if (
        0 <= int(best_start_idx) < tracks_world_smoothed.shape[0]
        and tracks_world_smoothed.shape[1] == len(best_queries_world)
    ):
        reference_candidate = np.asarray(tracks_world_smoothed[int(best_start_idx)], dtype=np.float32)
        if np.all(np.isfinite(reference_candidate)):
            reference_points_world = reference_candidate
            debug["reference_points_source"] = "smoothed_query_frame"
        else:
            debug["reference_points_source"] = "lifted_query_corners"
    else:
        debug["reference_points_source"] = "lifted_query_corners"
    raw_history, track_debug = pose_history_from_surface_tracks(
        clip,
        reference_points_world,
        tracks_world_smoothed,
        valid_tracks,
    )
    debug.update(track_debug)
    if len(raw_history) < ARTIPOINT_MIN_POSE_UPDATES:
        debug["failure_reason"] = "insufficient_surface_pose_updates"
        debug["pose_updates"] = int(len(raw_history))
        return None, debug

    # Collect wrist positions per pose-history frame for the wrist-derived
    # theta path. The wrist is on the handle, so it traces a circular arc
    # around the door's vertical hinge — a much cleaner signal than the
    # noisy per-frame CoTracker3+depth Procrustes poses on a textureless door.
    sample_by_frame = {
        int(sample.get("prefix", clip.frame_start + idx)): sample
        for idx, sample in enumerate(clip.samples)
    }

    def _frame_wrist(frame: int) -> Optional[np.ndarray]:
        sample = sample_by_frame.get(int(frame))
        if sample is None:
            return None
        for key in ("right_hand_world_wrist", "left_hand_world_wrist", "right_hand_world", "left_hand_world"):
            value = sample.get(key)
            if value is None:
                continue
            arr = np.asarray(value, dtype=np.float64).reshape(-1)
            if arr.size == 3 and np.all(np.isfinite(arr)):
                return arr
        return None

    wrist_positions = [_frame_wrist(int(item["frame"])) for item in raw_history]
    try:
        fit = fit_articulation_model_joint_optimization(
            raw_history,
            label=label,
            object_points=points,
            wrist_positions=wrist_positions,
            tracks_world=tracks_world_smoothed,
            valid_tracks=valid_tracks,
        )
    except Exception as exc:  # noqa: BLE001
        debug["failure_reason"] = f"articulation_fit_failed:{exc}"
        debug["pose_updates"] = int(len(raw_history))
        return None, debug

    fitted_history = list(fit.get("reconstructed_pose_history", []) or [])
    if not fitted_history:
        debug["failure_reason"] = "empty_fitted_pose_history"
        return None, debug
    initial_pose = np.eye(4, dtype=np.float32)
    result_history: List[dict] = []
    for sample in fitted_history:
        pose = np.asarray(sample["pose"], dtype=np.float32)
        if int(sample.get("frame", -1)) == int(clip.frame_start) and np.allclose(pose, initial_pose, atol=1e-5):
            continue
        result_history.append(
            {
                "frame": int(sample["frame"]),
                "pose": pose,
                "source": str(sample.get("source", "artipoint_joint_optimization")),
                "confidence": float(sample.get("confidence", 1.0)),
                "clip_id": int(clip.clip_id),
                "theta": float(sample.get("theta", 0.0)),
            }
        )
    final_pose = np.asarray(fitted_history[-1]["pose"], dtype=np.float32)
    final_points = apply_homogeneous_transform(points, final_pose)
    if len(colors) != len(points):
        colors = np.full((len(points), 3), 0.5, dtype=np.float32)
    debug["status"] = "success"
    debug["failure_reason"] = ""
    debug["pose_updates"] = int(len(result_history))
    debug["used_frames"] = [int(sample["frame"]) for sample in result_history]
    debug["articulation"] = {
        "joint_type": fit.get("joint_type"),
        "axis_direction": fit.get("axis_direction"),
        "axis_point": fit.get("axis_point"),
        "residual_mean_m": fit.get("mean_residual_m"),
        "residual_max_m": fit.get("max_residual_m"),
        "optimizer": fit.get("optimizer"),
        "solver_path": fit.get("solver_path"),
        "fallback_reason": fit.get("fallback_reason"),
        "chosen_pivot_source": fit.get("chosen_pivot_source"),
        "pivot_candidates": fit.get("pivot_candidates"),
        "pivot_candidate_scores": fit.get("pivot_candidate_scores"),
    }
    centroid_translation = float(np.linalg.norm(final_points.mean(axis=0) - points.mean(axis=0))) if len(final_points) else 0.0
    # Per-corner 3-D trajectories for visualisation (ArtiPoint Fig. 2 panel 4):
    # one curve per tracked image corner, after Eq. (1) smoothing. Invalid
    # samples (corner off-screen / occluded / no depth) are flagged so the
    # renderer can draw poly-lines only over the valid segments.
    track_frames_timeline = [
        int(clip.samples[idx].get("prefix", clip.frame_start + idx))
        for idx in range(min(len(clip.samples), tracks_world_smoothed.shape[0]))
    ]
    result = {
        "tracker": "artipoint_direct",
        "hand": None,
        "pose_updates": int(len(result_history)),
        "initial_pose": initial_pose,
        "final_pose": final_pose,
        "pose_delta": final_pose.astype(np.float32),
        "initial_points": points.astype(np.float32),
        "initial_colors": colors.astype(np.float32),
        "final_points": final_points.astype(np.float32),
        "final_colors": colors.astype(np.float32),
        "pose_history": result_history,
        "used_frames": [int(sample["frame"]) for sample in result_history],
        "translation_m": centroid_translation,
        "rotation_deg": rotation_degrees_from_transform(final_pose),
        "translation_disagreement_m": 0.0,
        "hand_anchor_mode": "artipoint_surface_tracks",
        "corner_tracks_world": np.asarray(tracks_world_smoothed, dtype=np.float32),
        "corner_tracks_valid": np.asarray(valid_tracks, dtype=bool),
        "corner_track_frames": track_frames_timeline,
    }
    _copy_articulation_fields_from_fit(result, fit)
    return result, debug


def build_temporary_lostfound_graph(
    label: str,
    points: np.ndarray,
    confidence: float = 1.0,
) -> Optional[Tuple[LFSceneGraph, int]]:
    """Create a minimal LostFound scene graph with one movable node for 6DoF updates."""
    if not LOSTFOUND_AVAILABLE or LFSceneGraph is None:
        return None
    node_points = np.asarray(points, dtype=np.float32)
    if len(node_points) < LOSTFOUND_TRACK_MIN_POINTS:
        return None
    tracking_points = sample_tracking_points(node_points)
    graph = LFSceneGraph()
    graph.add_node(
        np.array([0.5, 0.5, 0.5], dtype=np.float32),
        str(label),
        node_points,
        tracking_points,
        np.ones(len(node_points), dtype=bool),
        float(confidence),
    )
    graph.init_graph()
    graph.tree = KDTree(np.array([graph.nodes[index].centroid for index in graph.ids]))
    return graph, graph.ids[0]


def transform_temporary_lostfound_node(
    temporary_graph: LFSceneGraph,
    node_id: int,
    transform: np.ndarray,
) -> None:
    """Apply a rigid transform to the temporary LostFound node without requiring graph connections."""
    if temporary_graph is None or node_id not in getattr(temporary_graph, "nodes", {}):
        return
    if len(getattr(temporary_graph, "nodes", {})) <= 1:
        temporary_graph.nodes[node_id].transform(np.asarray(transform, dtype=np.float32))
        if getattr(temporary_graph, "ids", None):
            temporary_graph.tree = KDTree(np.array([temporary_graph.nodes[index].centroid for index in temporary_graph.ids]))
        return
    temporary_graph.transform(node_id, np.asarray(transform, dtype=np.float32))




def safe_temporary_graph_nearest_node(temporary_graph: LFSceneGraph, point: Optional[np.ndarray]) -> Tuple[Optional[float], Optional[int]]:
    """Query the nearest visible movable node without assuming the temporary graph has at least four nodes."""
    if point is None or temporary_graph is None or not getattr(temporary_graph, 'ids', None):
        return None, None
    k = min(4, len(temporary_graph.ids))
    _distances, neighbor_indices = temporary_graph.tree.query(point, k=k)
    neighbor_indices = np.atleast_1d(neighbor_indices)
    valid_node_ids: List[int] = []
    for neighbor_idx in neighbor_indices:
        neighbor_idx = int(neighbor_idx)
        if neighbor_idx < 0 or neighbor_idx >= len(temporary_graph.ids):
            continue
        node_id = temporary_graph.ids[neighbor_idx]
        node = temporary_graph.nodes[node_id]
        if node.movable and node.visible:
            valid_node_ids.append(node_id)
    if not valid_node_ids:
        return None, None
    hull_distances = []
    query_point = np.asarray(point, dtype=np.float32)
    for node_id in valid_node_ids:
        node = temporary_graph.nodes[node_id]
        reference_points = np.asarray(getattr(node, "tracking_points", np.zeros((0, 3), dtype=np.float32)), dtype=np.float32)
        if len(reference_points) == 0:
            reference_points = np.asarray(getattr(node, "points", np.zeros((0, 3), dtype=np.float32)), dtype=np.float32)
        if len(reference_points) == 0:
            hull_distances.append(float("inf"))
            continue
        hull_distances.append(float(KDTree(reference_points).query(query_point, k=1)[0]))
    best_index = int(np.argmin(hull_distances))
    return hull_distances[best_index], valid_node_ids[best_index]


def run_lostfound_single_hand_tracking(
    hand_name: str,
    temporary_graph: LFSceneGraph,
    node_id: int,
    clip: ClipInfo,
    rgb_list: Sequence[np.ndarray],
    T_wc_list: Sequence[np.ndarray],
    K_color: np.ndarray,
    T_cam_color: np.ndarray,
    use_hand: bool = True,
) -> Tuple[Optional[dict], dict]:
    """Run a LostFound-style 6DoF tracker for one hand over an active clip."""
    node_label = ""
    if temporary_graph is not None and node_id in temporary_graph.nodes:
        node = temporary_graph.nodes[node_id]
        node_label = str(getattr(node, "label", getattr(node, "sem_label", "")))
    debug = _new_lostfound_hand_debug(
        hand_name,
        clip,
        node_label,
    )
    if (
        not LOSTFOUND_TRACKING_AVAILABLE
        or CoTrackerOnlinePredictor is None
        or LOSTFOUND_ROOT is None
        or not torch.cuda.is_available()
        or len(rgb_list) <= LOOKAHEAD_FRAMES
    ):
        reasons = []
        if not LOSTFOUND_TRACKING_AVAILABLE:
            reasons.append("tracking_unavailable")
        if CoTrackerOnlinePredictor is None:
            reasons.append("cotracker_missing")
        if LOSTFOUND_ROOT is None:
            reasons.append("lostfound_root_missing")
        if not torch.cuda.is_available():
            reasons.append("cuda_unavailable")
        if len(rgb_list) <= LOOKAHEAD_FRAMES:
            reasons.append("clip_too_short")
        debug["status"] = "precondition_failed"
        debug["failure_reason"] = ",".join(reasons)
        return None, debug

    checkpoint_path = _cotracker3_checkpoint(online=True)
    if checkpoint_path is None:
        debug["status"] = "precondition_failed"
        debug["failure_reason"] = "cotracker3_online_checkpoint_missing"
        return None, debug
    debug["cotracker_version"] = "cotracker3_online"
    debug["cotracker_checkpoint"] = str(checkpoint_path)

    hand_obs_index = 4 if hand_name == 'left' else 5
    hand_world_key = f"{hand_name}_hand_world"
    hand_wrist_key = f"{hand_name}_hand_world_wrist"

    def _sample_hand_position(sample_index: int) -> Optional[np.ndarray]:
        sample = clip.samples[int(sample_index)]
        wrist = sample.get(hand_wrist_key)
        if wrist is not None:
            return wrist
        return sample.get(hand_world_key)

    def _observation_hand_position(observation: tuple) -> Optional[np.ndarray]:
        return _sample_hand_position(int(observation[0]))
    observations = [
        (
            idx,
            np.asarray(T_wc_list[idx], dtype=np.float32),
            clip.samples[idx].get('hand_dets'),
            clip.samples[idx].get('obj_dets'),
            clip.samples[idx].get('left_hand_world'),
            clip.samples[idx].get('right_hand_world'),
            int(clip.samples[idx].get('prefix', idx)),
            np.asarray(rgb_list[idx], dtype=np.uint8),
        )
        for idx in range(len(rgb_list))
    ]

    context: deque[tuple] = deque(maxlen=LOOKAHEAD_FRAMES)
    for observation in observations[:LOOKAHEAD_FRAMES]:
        context.append(observation)

    node = temporary_graph.nodes[node_id]
    initial_pose = node.pose.copy()
    initial_points = node.points.copy()
    hand_reference_positions = [observation[hand_obs_index] for observation in observations]
    node.tracking_points = select_hand_local_tracking_points(node.points, hand_reference_positions)
    debug["tracking_point_count"] = int(len(node.tracking_points))
    valid_hand_reference_positions = [np.asarray(hand, dtype=np.float32) for hand in hand_reference_positions if hand is not None]
    if len(node.tracking_points) and valid_hand_reference_positions:
        hand_tree = KDTree(np.stack(valid_hand_reference_positions, axis=0))
        tracking_point_distances, _ = hand_tree.query(np.asarray(node.tracking_points, dtype=np.float32), k=1)
        debug["tracking_point_to_hand_distance_stats_m"] = _distance_stats(
            tracking_point_distances.tolist(),
            threshold=LOSTFOUND_HAND_DISTANCE_THRESHOLD,
        )
    hand_positions: deque[Optional[np.ndarray]] = deque(maxlen=HAND_HISTORY_SIZE)
    tracker = None
    queries_tensor = None
    queries_object = None
    tracker_frames: List[np.ndarray] = []
    pred_tracks = None
    pred_visibility = None
    tracker_is_first_step = True
    tracker_index = 0
    prev_pose_inv = None
    delta_obj = None
    smoothed_rotation_obj_world: Optional[np.ndarray] = None
    pose_updates = 0
    used_frames: List[int] = []
    pose_history: List[dict] = []
    active = False

    def _future_distance(observation: tuple) -> float:
        hand_pos = _observation_hand_position(observation)
        if hand_pos is None:
            return float('inf')
        dist, _ = safe_temporary_graph_nearest_node(temporary_graph, hand_pos)
        return float('inf') if dist is None else float(dist)

    # Dynamic activation threshold: pre-compute wrist-to-prior distance across
    # every frame of this active clip, then set the gate so that the bulk of
    # frames pass it. This rescues case-2 (rigid transport) tracking when the
    # prior was built from frames where the object was on a counter or table
    # — once the user picks it up the wrist drifts well beyond the static
    # 0.20 m gate but stays at a clip-relative "close-approach" range. The
    # `future_far` guard inside the loop still pins activation to the actual
    # closest-approach frame, so a generous gate doesn't admit pre-contact
    # approach frames.
    clip_hand_to_prior_dists: List[float] = []
    for obs in observations:
        hand_p = _sample_hand_position(int(obs[0]))
        if hand_p is None:
            continue
        d, _ = safe_temporary_graph_nearest_node(temporary_graph, hand_p)
        if d is not None and np.isfinite(d):
            clip_hand_to_prior_dists.append(float(d))
    if clip_hand_to_prior_dists:
        # Use the 90th-percentile distance so ~90% of frames pass the gate.
        # Clamp to a sane range: must be at least the default 0.20 m
        # (else stationary clips would over-tighten the gate), and at most
        # 1.50 m so we don't spuriously activate when the hand never gets
        # near the prior at all.
        pct = float(np.percentile(np.asarray(clip_hand_to_prior_dists, dtype=np.float64), 90.0))
        dynamic_activation_distance = float(
            max(float(LOSTFOUND_ACTIVATION_DISTANCE), min(pct, 1.5))
        )
        debug["dynamic_activation_distance_m"] = dynamic_activation_distance
        debug["clip_hand_to_prior_p10_p50_p90_m"] = [
            float(np.percentile(clip_hand_to_prior_dists, 10.0)),
            float(np.percentile(clip_hand_to_prior_dists, 50.0)),
            pct,
        ]
        debug["clip_hand_to_prior_min_m"] = float(min(clip_hand_to_prior_dists))
    else:
        dynamic_activation_distance = float(LOSTFOUND_ACTIVATION_DISTANCE)
        debug["dynamic_activation_distance_m"] = dynamic_activation_distance
    debug["distance_threshold_m"] = float(dynamic_activation_distance)

    try:
        for next_observation in observations[LOOKAHEAD_FRAMES:]:
            current_observation = context[0]
            sample_index, camera_pose, hand_dets, obj_dets, left_palm, right_palm, global_frame_id, image = current_observation
            context.append(next_observation)
            hand_pos = _sample_hand_position(int(sample_index))
            hand_positions.append(hand_pos)
            vel_prior = hand_velocity(list(hand_positions))
            vel_post = hand_velocity([_observation_hand_position(obs) for obs in context])
            threshold = lostfound_contact_threshold(vel_prior, vel_post)
            contact_count = sum(1 for obs in context if check_hand_object_tracker(obs[2], obs[3]))
            has_detection = check_hand_object_tracker(hand_dets, obj_dets)

            if not active:
                debug["activation_attempts"] += 1
                if has_detection and hand_pos is not None:
                    dist, _ = safe_temporary_graph_nearest_node(temporary_graph, hand_pos)
                    if dist is not None and np.isfinite(dist):
                        debug["hand_to_tracking_distances_m"].append(float(dist))
                    active_clip_contact = has_detection and contact_count > 0
                    distance_ok = active_clip_contact or (dist is not None and dist < dynamic_activation_distance)
                    future_far = active_clip_contact or all((_future_distance(obs) > dist) or (not check_hand_object_tracker(obs[2], obs[3])) for obs in context)
                    contact_ok = active_clip_contact or (contact_count > threshold)
                    if distance_ok:
                        if future_far and contact_ok:
                            queries_2d, queries_3d = project_tracking_points_to_image(
                                node.tracking_points,
                                image,
                                camera_pose,
                                K_color,
                                T_cam_color,
                            )
                            debug["projected_query_counts"].append(int(len(queries_2d)))
                            if len(queries_2d) < 8:
                                debug["start_fail_counts"]["too_few_projected_queries"] += 1
                                continue
                            delta_world = hand_pos - node.pose[:3, 3]
                            delta_obj = node.pose[:3, :3].T @ delta_world
                            queries_object = (node.pose[:3, :3].T @ (queries_3d - node.pose[:3, 3]).T).T.astype(np.float32)
                            tracker = CoTrackerOnlinePredictor(checkpoint=str(checkpoint_path)).to('cuda')
                            queries_tensor = torch.from_numpy(queries_2d)[None].to('cuda')
                            tracker_frames = [image]
                            tracker_is_first_step = True
                            tracker_index = 1
                            pred_tracks = None
                            pred_visibility = None
                            for context_observation in list(context):
                                if tracker_index % tracker.step == 0 and tracker_index != 0:
                                    pred_tracks, pred_visibility = lostfound_process_step(tracker, tracker_frames, tracker_is_first_step, queries_tensor)
                                    tracker_is_first_step = False
                                tracker_frames.append(context_observation[7])
                                tracker_index += 1
                            if pred_tracks is None or pred_visibility is None:
                                debug["start_fail_counts"]["tracker_bootstrap_failed"] += 1
                                del tracker
                                tracker = None
                                torch.cuda.empty_cache()
                                continue
                            pred_index = (tracker_index - 1) % tracker.step - LOOKAHEAD_FRAMES
                            pose_cam, pose_debug = compute_lostfound_pose(pred_tracks, pred_visibility, queries_object, pred_index, K_color)
                            debug["visible_point_counts"].append(int(pose_debug.get("visible_points", 0)))
                            if pose_cam is None:
                                debug["start_fail_counts"]["initial_pose_failed"] += 1
                                del tracker
                                tracker = None
                                torch.cuda.empty_cache()
                                continue
                            rotation_obj_cam, t_vec = pose_cam
                            world_from_color = np.asarray(camera_pose, dtype=np.float32) @ np.asarray(T_cam_color, dtype=np.float32)
                            rotation_obj_world = world_from_color[:3, :3] @ rotation_obj_cam
                            # First (activation) frame seeds the SLERP EMA; no smoothing on it.
                            smoothed_rotation_obj_world = rotation_obj_world.astype(np.float32)
                            translation_obj_world, translation_debug = select_lostfound_translation(
                                rotation_obj_world=smoothed_rotation_obj_world,
                                hand_pos=hand_pos,
                                delta_obj=delta_obj,
                                world_from_color=world_from_color,
                                t_vec=t_vec,
                                use_hand=use_hand,
                                tracked_pose_index=0,
                            )
                            if translation_obj_world is None:
                                debug["start_fail_counts"]["missing_wrist_translation"] += 1
                                del tracker
                                tracker = None
                                torch.cuda.empty_cache()
                                continue
                            object_pose = np.eye(4, dtype=np.float32)
                            object_pose[:3, :3] = smoothed_rotation_obj_world
                            object_pose[:3, 3] = translation_obj_world.astype(np.float32)
                            transform_temporary_lostfound_node(
                                temporary_graph,
                                node_id,
                                object_pose @ np.linalg.inv(temporary_graph.nodes[node_id].pose.copy()),
                            )
                            prev_pose_inv = np.linalg.inv(object_pose)
                            debug["hand_anchor_mode"] = "wrist"
                            debug["activation_mode"] = "active_clip_contact" if active_clip_contact else "lostfound_default"
                            debug.setdefault("translation_modes", []).append(str(translation_debug.get("mode", "pnp")))
                            debug.setdefault("translation_blends", []).append(float(translation_debug.get("blend", 0.0)))
                            debug.setdefault("translation_disagreements_m", []).append(float(translation_debug.get("disagreement_m", 0.0)))
                            debug["status"] = "tracking_started"
                            pose_updates = 1
                            used_frames.append(int(global_frame_id))
                            debug["pose_updates"] = int(pose_updates)
                            debug["used_frames"] = [int(frame) for frame in used_frames]
                            pose_history.append(
                                {
                                    "frame": int(global_frame_id),
                                    "pose": temporary_graph.nodes[node_id].pose.copy().astype(np.float32),
                                    "source": "activation",
                                    "confidence": 1.0,
                                }
                            )
                            active = True
                        else:
                            if not future_far:
                                debug["start_fail_counts"]["future_not_far_enough"] += 1
                            if not contact_ok:
                                debug["start_fail_counts"]["contact_threshold_not_met"] += 1
                    else:
                        debug["start_fail_counts"]["nearest_node_too_far_or_missing"] += 1
                else:
                    if not has_detection:
                        debug["start_fail_counts"]["no_detection"] += 1
                    if hand_pos is None:
                        debug["start_fail_counts"]["hand_missing"] += 1
            else:
                if hand_pos is not None and contact_count > threshold and tracker is not None and queries_tensor is not None and queries_object is not None:
                    if tracker_index % tracker.step == 0 and tracker_index != 0:
                        pred_tracks, pred_visibility = lostfound_process_step(tracker, tracker_frames, tracker_is_first_step, queries_tensor)
                        tracker_is_first_step = False
                    tracker_frames.append(context[-1][7])
                    tracker_index += 1
                    if pred_tracks is None or pred_visibility is None:
                        debug["update_fail_counts"]["tracker_prediction_missing"] += 1
                        continue
                    pred_index = (tracker_index - 1) % tracker.step - LOOKAHEAD_FRAMES
                    pose_cam, pose_debug = compute_lostfound_pose(pred_tracks, pred_visibility, queries_object, pred_index, K_color)
                    debug["visible_point_counts"].append(int(pose_debug.get("visible_points", 0)))
                    if pose_cam is None:
                        debug["update_fail_counts"]["pose_estimation_failed"] += 1
                        continue
                    rotation_obj_cam, t_vec = pose_cam
                    world_from_color = np.asarray(camera_pose, dtype=np.float32) @ np.asarray(T_cam_color, dtype=np.float32)
                    rotation_obj_world = world_from_color[:3, :3] @ rotation_obj_cam
                    # SLERP the new PnP rotation toward the EMA-smoothed previous rotation.
                    smoothed_rotation_obj_world = slerp_rotation_step(
                        smoothed_rotation_obj_world,
                        rotation_obj_world,
                        LOSTFOUND_ROTATION_SLERP_ALPHA,
                    )
                    translation_obj_world, translation_debug = select_lostfound_translation(
                        rotation_obj_world=smoothed_rotation_obj_world,
                        hand_pos=hand_pos,
                        delta_obj=delta_obj,
                        world_from_color=world_from_color,
                        t_vec=t_vec,
                        use_hand=use_hand,
                        tracked_pose_index=pose_updates,
                    )
                    if translation_obj_world is None:
                        debug["update_fail_counts"]["missing_wrist_translation"] += 1
                        break
                    object_pose = np.eye(4, dtype=np.float32)
                    object_pose[:3, :3] = smoothed_rotation_obj_world
                    object_pose[:3, 3] = translation_obj_world.astype(np.float32)
                    transform_temporary_lostfound_node(temporary_graph, node_id, object_pose @ prev_pose_inv)
                    prev_pose_inv = np.linalg.inv(object_pose)
                    pose_updates += 1
                    used_frames.append(int(global_frame_id))
                    debug["pose_updates"] = int(pose_updates)
                    debug["used_frames"] = [int(frame) for frame in used_frames]
                    debug.setdefault("translation_modes", []).append(str(translation_debug.get("mode", "pnp")))
                    debug.setdefault("translation_blends", []).append(float(translation_debug.get("blend", 0.0)))
                    debug.setdefault("translation_disagreements_m", []).append(float(translation_debug.get("disagreement_m", 0.0)))
                    pose_history.append(
                        {
                            "frame": int(global_frame_id),
                            "pose": temporary_graph.nodes[node_id].pose.copy().astype(np.float32),
                            "source": "tracked",
                            "confidence": 1.0,
                        }
                    )
                else:
                    debug["update_fail_counts"]["contact_gate_dropped"] += 1
                    break
    finally:
        if tracker is not None:
            del tracker
            torch.cuda.empty_cache()
            gc.collect()

    debug["hand_to_tracking_distance_stats_m"] = _distance_stats(
        debug["hand_to_tracking_distances_m"],
        threshold=LOSTFOUND_HAND_DISTANCE_THRESHOLD,
    )

    if pose_updates < LOSTFOUND_MIN_UPDATES:
        debug["status"] = "failed"
        debug["failure_reason"] = "insufficient_pose_updates"
        return None, debug

    final_pose = temporary_graph.nodes[node_id].pose.copy()
    pose_delta = final_pose @ np.linalg.inv(initial_pose)
    disagreement_values = np.asarray(debug.get("translation_disagreements_m", []), dtype=np.float32)
    translation_disagreement_m = float(np.median(disagreement_values)) if len(disagreement_values) else 1e9
    debug["status"] = "success"
    debug["failure_reason"] = ""
    return {
        'hand': hand_name,
        'pose_updates': pose_updates,
        'initial_pose': initial_pose,
        'final_pose': final_pose,
        'pose_delta': pose_delta.astype(np.float32),
        'initial_points': initial_points.astype(np.float32),
        'final_points': temporary_graph.nodes[node_id].points.copy().astype(np.float32),
        'pose_history': pose_history,
        'used_frames': used_frames,
        'translation_m': float(np.linalg.norm(pose_delta[:3, 3])),
        'rotation_deg': rotation_degrees_from_transform(pose_delta),
        'translation_disagreement_m': translation_disagreement_m,
        'hand_anchor_mode': "wrist",
    }, debug


def run_lostfound_pose_tracking(
    clip: ClipInfo,
    label: str,
    initial_points: np.ndarray,
    initial_colors: np.ndarray,
    rgb_list: Sequence[np.ndarray],
    T_wc_list: Sequence[np.ndarray],
    K_color: np.ndarray,
    T_cam_color: np.ndarray,
) -> Tuple[Optional[dict], dict]:
    """Run the LostFound 6DoF bridge on one active-clip node and keep the best hand trajectory."""
    debug = {
        "label": str(label),
        "clip_id": int(clip.clip_id),
        "initial_point_count": int(len(initial_points)),
        "status": "failed",
        "failure_reason": "",
        "selected_hand": None,
        "hand_attempts": [],
    }
    temporary = build_temporary_lostfound_graph(label, initial_points)
    if temporary is None:
        debug["failure_reason"] = "insufficient_initial_points_for_temporary_graph"
        return None, debug
    temp_graph, node_id = temporary
    preferred_hand = getattr(clip, "interaction_hand", None)
    hand_names = (preferred_hand,) if preferred_hand in ("left", "right") else ("left", "right")
    debug["requested_hand"] = preferred_hand if preferred_hand in ("left", "right") else None
    debug["candidate_hands"] = [str(hand) for hand in hand_names]
    candidates = []
    for hand_name in hand_names:
        hand_result, hand_debug = run_lostfound_single_hand_tracking(
            hand_name,
            copy.deepcopy(temp_graph),
            node_id,
            clip,
            rgb_list,
            T_wc_list,
            K_color,
            T_cam_color,
        )
        debug["hand_attempts"].append(hand_debug)
        if hand_result is not None:
            candidates.append(hand_result)
    if not candidates:
        debug["failure_reason"] = "no_hand_tracker_succeeded"
        return None, debug
    best = max(
        candidates,
        key=lambda item: (
            item['pose_updates'],
            -float(item.get('translation_disagreement_m', 1e9)),
            -float(item.get('rotation_deg', 0.0)),
            -float(item.get('translation_m', 0.0)),
        ),
    )
    debug["status"] = "success"
    debug["failure_reason"] = ""
    debug["selected_hand"] = str(best["hand"])
    best['initial_colors'] = np.asarray(initial_colors, dtype=np.float32)
    final_colors = np.asarray(initial_colors, dtype=np.float32)
    if len(final_colors) != len(best['final_points']):
        fill = np.full((len(best['final_points']), 3), 0.5, dtype=np.float32)
        if len(final_colors) == len(best['initial_points']):
            final_colors = fill
        else:
            final_colors = fill
    best['final_colors'] = final_colors
    return best, debug



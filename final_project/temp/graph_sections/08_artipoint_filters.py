# ======================================================================
# graph.py section: 08_artipoint_filters.py
# ArtiPoint track filtering and smoothing
#
# NOTE: not a standalone module. Loaded by graph.py into a shared
# namespace via exec(); names from earlier sections are in scope here.
# ======================================================================
from __future__ import annotations
def calculate_artipoint_track_variance(points: np.ndarray, visibility: np.ndarray) -> np.ndarray:
    """Direct copy of ArtiPoint's per-track variance calculation, adapted as a helper."""
    variances = []
    for track_idx in range(points.shape[1]):
        visible_points = points[visibility[:, track_idx], track_idx]
        visible_points = visible_points[np.isfinite(visible_points).all(axis=1)]
        if visible_points.shape[0] > 0:
            variances.append(np.var(visible_points, axis=0))
        else:
            variances.append(np.zeros(points.shape[2], dtype=np.float32))
    return np.array(variances)


def artipoint_filter_non_smooth_tracks(
    pred_3d_tracker: np.ndarray,
    pred_visibility: np.ndarray,
    indices: np.ndarray,
    thrsh: float = ARTIPOINT_FILTER_SMOOTHNESS_THRESHOLD,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """ArtiPoint filter_non_smooth_tracks, retaining original track indices."""
    change = np.zeros(pred_3d_tracker.shape[1], dtype=np.float32)
    for i in range(pred_3d_tracker.shape[1]):
        visible_points = pred_3d_tracker[pred_visibility[:, i], i]
        visible_points = visible_points[np.isfinite(visible_points).all(axis=1)]
        change[i] = (
            np.linalg.norm(np.abs(np.diff(visible_points, axis=0)), axis=0).mean()
            if len(visible_points) > 1
            else 0.0
        )
    significant_change = change > float(thrsh)
    keep = ~significant_change
    return (
        pred_3d_tracker[:, keep],
        pred_visibility[:, keep],
        indices[keep],
        {
            "stage": "filter_non_smooth_tracks",
            "threshold": float(thrsh),
            "removed": int(np.count_nonzero(significant_change)),
            "kept": int(np.count_nonzero(keep)),
        },
    )


def artipoint_median_filter(
    pred_3d_tracker: np.ndarray,
    pred_visibility: np.ndarray,
    indices: np.ndarray,
    variance_type: str = ARTIPOINT_FILTER_VARIANCE_TYPE,
    percentile: int = ARTIPOINT_FILTER_PERCENTILE,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """ArtiPoint median_filter: remove low-variance/static tracks."""
    if pred_3d_tracker.shape[1] == 0:
        return pred_3d_tracker, pred_visibility, indices, {"stage": "median_filter", "removed": 0, "kept": 0}
    if variance_type == "2d":
        var = calculate_artipoint_track_variance(pred_3d_tracker[:, :, :2], pred_visibility)
    elif variance_type == "3d":
        var = calculate_artipoint_track_variance(pred_3d_tracker, pred_visibility)
    else:
        raise ValueError("Invalid variance_type. Choose either '2d' or '3d'.")
    points_motion = np.linalg.norm(var, axis=1)
    static_points = points_motion < np.percentile(points_motion, int(percentile))
    keep = ~static_points
    return (
        pred_3d_tracker[:, keep],
        pred_visibility[:, keep],
        indices[keep],
        {
            "stage": "median_filter",
            "variance_type": str(variance_type),
            "percentile": int(percentile),
            "removed_static": int(np.count_nonzero(static_points)),
            "kept_dynamic": int(np.count_nonzero(keep)),
            "median_motion_variance": float(np.median(points_motion)) if len(points_motion) else 0.0,
        },
    )


def artipoint_filter_static_tracks_absolute(
    pred_3d_tracker: np.ndarray,
    pred_visibility: np.ndarray,
    indices: np.ndarray,
    min_motion_m: float = ARTIPOINT_STATIC_TRACK_MIN_MOTION_M,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Remove tracks whose 3D motion extent stays below an absolute floor."""
    if pred_3d_tracker.shape[1] == 0:
        return pred_3d_tracker, pred_visibility, indices, {"stage": "filter_static_tracks_absolute", "removed": 0, "kept": 0}
    motion_extents = np.zeros(pred_3d_tracker.shape[1], dtype=np.float32)
    keep = np.zeros(pred_3d_tracker.shape[1], dtype=bool)
    for track_idx in range(pred_3d_tracker.shape[1]):
        visible = pred_visibility[:, track_idx] & np.all(np.isfinite(pred_3d_tracker[:, track_idx]), axis=1)
        if int(np.count_nonzero(visible)) < 2:
            continue
        visible_points = np.asarray(pred_3d_tracker[visible, track_idx], dtype=np.float32)
        extent = float(np.linalg.norm(visible_points.max(axis=0) - visible_points.min(axis=0)))
        motion_extents[track_idx] = extent
        keep[track_idx] = extent >= float(min_motion_m)
    return (
        pred_3d_tracker[:, keep],
        pred_visibility[:, keep],
        indices[keep],
        {
            "stage": "filter_static_tracks_absolute",
            "min_motion_m": float(min_motion_m),
            "removed_static": int(np.count_nonzero(~keep)),
            "kept_dynamic": int(np.count_nonzero(keep)),
            "median_motion_extent_m": float(np.median(motion_extents)) if len(motion_extents) else 0.0,
            "max_motion_extent_m": float(np.max(motion_extents)) if len(motion_extents) else 0.0,
        },
    )


def artipoint_filter_occluded_tracks(
    pred_3d_tracker: np.ndarray,
    pred_visibility: np.ndarray,
    indices: np.ndarray,
    occlusion_threshold: float = ARTIPOINT_FILTER_OCCLUSION_THRESHOLD,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """ArtiPoint filter_occluded_tracks, retaining original track indices."""
    if pred_visibility.shape[1] == 0:
        return pred_3d_tracker, pred_visibility, indices, {"stage": "filter_occluded_tracks", "removed": 0, "kept": 0}
    num_frames = pred_visibility.shape[0]
    visibility_percentage = np.sum(pred_visibility, axis=0) / max(1, num_frames)
    visible_enough = visibility_percentage >= (1.0 - float(occlusion_threshold))
    return (
        pred_3d_tracker[:, visible_enough],
        pred_visibility[:, visible_enough],
        indices[visible_enough],
        {
            "stage": "filter_occluded_tracks",
            "occlusion_threshold": float(occlusion_threshold),
            "removed": int(np.count_nonzero(~visible_enough)),
            "kept": int(np.count_nonzero(visible_enough)),
            "median_visibility_fraction": float(np.median(visibility_percentage)) if len(visibility_percentage) else 0.0,
        },
    )


def artipoint_filter_outlier_tracks(
    pred_3d_tracker: np.ndarray,
    pred_visibility: np.ndarray,
    indices: np.ndarray,
    eps: float = ARTIPOINT_FILTER_DBSCAN_EPS,
    min_samples: int = ARTIPOINT_FILTER_DBSCAN_MIN_SAMPLES,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """ArtiPoint DBSCAN outlier filter on the frame with most visible tracks."""
    num_frames, num_tracks, _ = pred_3d_tracker.shape
    debug = {"stage": "filter_outlier_tracks", "eps": float(eps), "min_samples": int(min_samples), "removed": 0, "kept": int(num_tracks)}
    if num_tracks == 0:
        return pred_3d_tracker, pred_visibility, indices, debug
    visible_counts = np.sum(pred_visibility, axis=1)
    frame_idx = int(np.argmax(visible_counts))
    visible_tracks = pred_visibility[frame_idx]
    if not np.any(visible_tracks):
        debug["reason"] = "no_visible_points"
        return pred_3d_tracker, pred_visibility, indices, debug
    points_for_clustering = pred_3d_tracker[frame_idx, visible_tracks]
    track_indices = np.where(visible_tracks)[0]
    finite = np.isfinite(points_for_clustering).all(axis=1)
    points_for_clustering = points_for_clustering[finite]
    track_indices = track_indices[finite]
    if len(points_for_clustering) < max(1, int(min_samples)):
        debug["reason"] = "too_few_points_for_dbscan"
        return pred_3d_tracker, pred_visibility, indices, debug
    clustering = DBSCAN(eps=float(eps), min_samples=int(min_samples)).fit(points_for_clustering)
    labels = clustering.labels_
    unique_labels, counts = np.unique(labels[labels != -1], return_counts=True)
    if len(unique_labels) == 0:
        debug["reason"] = "no_clusters"
        return pred_3d_tracker, pred_visibility, indices, debug
    main_cluster = unique_labels[np.argmax(counts)]
    main_cluster_mask = np.zeros(num_tracks, dtype=bool)
    main_cluster_mask[track_indices[labels == main_cluster]] = True
    debug.update(
        {
            "frame_idx": frame_idx,
            "cluster_count": int(len(unique_labels)),
            "kept": int(np.count_nonzero(main_cluster_mask)),
            "removed": int(np.count_nonzero(~main_cluster_mask)),
        }
    )
    return pred_3d_tracker[:, main_cluster_mask], pred_visibility[:, main_cluster_mask], indices[main_cluster_mask], debug


def artipoint_filter_tracks_by_prior_surface(
    pred_3d_tracker: np.ndarray,
    pred_visibility: np.ndarray,
    indices: np.ndarray,
    query_points_world: Optional[np.ndarray],
    prior_surface_points: Optional[np.ndarray],
    max_query_prior_distance_m: float = ARTIPOINT_SURFACE_TRACK_MAX_QUERY_PRIOR_DISTANCE_M,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Keep only tracks whose query-frame anchors lie close to the prior moving surface."""
    num_tracks = int(pred_3d_tracker.shape[1]) if pred_3d_tracker.ndim == 3 else 0
    debug = {
        "stage": "filter_tracks_by_prior_surface",
        "max_query_prior_distance_m": float(max_query_prior_distance_m),
        "removed_off_surface": 0,
        "kept_on_surface": num_tracks,
    }
    if num_tracks == 0:
        return pred_3d_tracker, pred_visibility, indices, debug
    if query_points_world is None or prior_surface_points is None:
        debug["skipped"] = True
        debug["reason"] = "missing_query_points_or_prior_surface"
        return pred_3d_tracker, pred_visibility, indices, debug
    anchors_all = np.asarray(query_points_world, dtype=np.float32)
    prior_all = np.asarray(prior_surface_points, dtype=np.float32)
    if anchors_all.ndim != 2 or anchors_all.shape[1] != 3 or len(anchors_all) == 0:
        debug["skipped"] = True
        debug["reason"] = "invalid_query_points"
        return pred_3d_tracker, pred_visibility, indices, debug
    if prior_all.ndim != 2 or prior_all.shape[1] != 3 or len(prior_all) == 0:
        debug["skipped"] = True
        debug["reason"] = "invalid_prior_surface"
        return pred_3d_tracker, pred_visibility, indices, debug
    if len(indices) == 0 or int(np.max(indices)) >= len(anchors_all):
        debug["skipped"] = True
        debug["reason"] = "query_index_mismatch"
        return pred_3d_tracker, pred_visibility, indices, debug
    finite_prior = np.all(np.isfinite(prior_all), axis=1)
    finite_anchors = np.all(np.isfinite(anchors_all), axis=1)
    prior = prior_all[finite_prior]
    if len(prior) == 0:
        debug["skipped"] = True
        debug["reason"] = "no_finite_prior_surface_points"
        return pred_3d_tracker, pred_visibility, indices, debug
    anchors = anchors_all[indices]
    anchor_finite_mask = finite_anchors[indices] & np.all(np.isfinite(anchors), axis=1)
    distances = np.full(len(indices), np.inf, dtype=np.float32)
    if int(np.count_nonzero(anchor_finite_mask)) > 0:
        tree = KDTree(prior.astype(np.float64, copy=False))
        query_pts = anchors[anchor_finite_mask].astype(np.float64, copy=False)
        nearest, _ = tree.query(query_pts, k=1)
        distances[anchor_finite_mask] = np.asarray(nearest, dtype=np.float32)
    keep = anchor_finite_mask & np.isfinite(distances) & (distances <= float(max_query_prior_distance_m))
    finite_distances = distances[np.isfinite(distances)]
    kept_distances = distances[keep]
    debug.update(
        {
            "removed_off_surface": int(np.count_nonzero(~keep)),
            "kept_on_surface": int(np.count_nonzero(keep)),
            "median_query_prior_distance_m": float(np.median(finite_distances)) if finite_distances.size else 0.0,
            "p95_query_prior_distance_m": float(np.percentile(finite_distances, 95)) if finite_distances.size else 0.0,
            "max_query_prior_distance_m_all": float(np.max(finite_distances)) if finite_distances.size else 0.0,
            "max_query_prior_distance_m_kept": float(np.max(kept_distances)) if kept_distances.size else 0.0,
        }
    )
    return pred_3d_tracker[:, keep], pred_visibility[:, keep], indices[keep], debug


def filter_artipoint_tracks(
    tracks_world: np.ndarray,
    valid_tracks: np.ndarray,
    query_points_world: Optional[np.ndarray] = None,
    prior_surface_points: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Run ArtiPoint's frontend filters in the same order as artipoint.py.

    Original order: filter_non_smooth_tracks -> median_filter ->
    filter_occluded_tracks -> filter_outlier_tracks -> smooth tracks. We keep
    the original track indices so our query/reference points stay aligned.
    """
    tracks = np.asarray(tracks_world, dtype=np.float32)
    valid = np.asarray(valid_tracks, dtype=bool)
    debug = {
        "input_track_count": int(tracks.shape[1]) if tracks.ndim == 3 else 0,
        "kept_track_count": 0,
        "stages": [],
        "source": "artipoint_repo_filters_plus_team_filters",
        "repo_filter_order": [
            "filter_non_smooth_tracks",
            "median_filter",
            "filter_occluded_tracks",
            "filter_outlier_tracks",
            "smooth_lifted_world_tracks",
        ],
        "extra_filters": [
            "filter_tracks_by_prior_surface",
        ],
        "disabled_filters": [],
    }
    if tracks.ndim != 3 or valid.ndim != 2 or tracks.shape[:2] != valid.shape:
        return tracks_world, valid_tracks, np.ones((0,), dtype=bool), debug
    original_count = tracks.shape[1]
    indices = np.arange(original_count)
    if original_count == 0:
        return tracks, valid, np.zeros((0,), dtype=bool), debug

    for filter_fn in (
        artipoint_filter_non_smooth_tracks,
        artipoint_filter_static_tracks_absolute,
        artipoint_median_filter,
        artipoint_filter_occluded_tracks,
        artipoint_filter_outlier_tracks,
    ):
        if (
            filter_fn is not artipoint_filter_static_tracks_absolute
            and tracks.shape[1] < int(ARTIPOINT_MIN_FILTERED_TRACKS)
        ):
            debug["stages"].append({"stage": filter_fn.__name__, "skipped": True, "reason": "too_few_tracks"})
            continue
        tracks, valid, indices, stage_debug = filter_fn(tracks, valid, indices)
        debug["stages"].append(stage_debug)

    tracks, valid, indices, stage_debug = artipoint_filter_tracks_by_prior_surface(
        tracks,
        valid,
        indices,
        query_points_world=query_points_world,
        prior_surface_points=prior_surface_points,
    )
    debug["stages"].append(stage_debug)

    if tracks.shape[1] < int(ARTIPOINT_MIN_FILTERED_TRACKS):
        debug["fallback"] = "disabled"
        debug["failure_reason"] = "too_few_tracks_after_filtering"
        print(
            f"[ArtiPoint-TrackFilter][Error] kept {tracks.shape[1]}/{original_count} tracks after filtering; "
            "top-motion-track fallback is disabled."
        )
    else:
        debug["fallback"] = "none"

    keep_mask = np.zeros(original_count, dtype=bool)
    keep_mask[indices] = True
    debug["kept_track_count"] = int(np.count_nonzero(keep_mask))
    return tracks, valid, keep_mask, debug

def smooth_lifted_world_tracks(
    tracks_world: np.ndarray,
    valid_tracks: np.ndarray,
    lambda_vel: float = ARTIPOINT_TRACK_SMOOTH_LAMBDA_VEL,
    lambda_jerk: float = ARTIPOINT_TRACK_SMOOTH_LAMBDA_JERK,
) -> np.ndarray:
    """Apply ArtiPoint Eq. (1) smoothing to the lifted 3D track tensor.

    ``tracks_world`` has shape (T, N, 3) and may contain NaN at invisible
    samples; ``valid_tracks`` has shape (T, N) and acts as the fidelity weight
    (1 = observed, 0 = ignored). Invisible samples are zeroed before solving so
    the sparse linear system stays finite; the smoother then interpolates the
    gaps using the velocity + jerk priors weighted by the visible neighbours.

    Why we apply this BEFORE per-frame Procrustes: the lifted CoTracker tracks
    on the far end of a door swing get noisy as ~half the tracks fall out of
    the camera view or get occluded by the hand. Without smoothing, the
    per-frame Kabsch RMSE jumps from ~1 cm to >30 cm and the downstream
    articulation fit collapses (we observed this directly in event.json).
    Smoothing propagates the trustworthy early-frame 3D positions through the
    noisy frames, keeping every Procrustes anchor stable.
    """
    tracks = np.asarray(tracks_world, dtype=np.float64)
    valid = np.asarray(valid_tracks, dtype=bool)
    if tracks.ndim != 3 or tracks.shape[2] != 3:
        return tracks_world
    safe_tracks = np.where(np.isfinite(tracks), tracks, 0.0)
    safe_tracks = np.where(valid[..., None], safe_tracks, 0.0)
    visibility = valid.astype(np.float64)
    try:
        smoothed = smooth_trajectory_optimization(
            safe_tracks,
            visibility=visibility,
            lambda_vel=float(lambda_vel),
            lambda_jerk=float(lambda_jerk),
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[smooth_lifted_world_tracks] solver failed: {exc}; keeping raw tracks.")
        return tracks_world
    # Where the original sample was observed (and finite), keep the smoothed
    # value; where it was missing, smoothed already gives the interpolated
    # value. NaN-clean output by construction.
    return smoothed.astype(np.float32)


def pose_history_from_surface_tracks(
    clip: ClipInfo,
    reference_points_world: np.ndarray,
    tracks_world: np.ndarray,
    valid_tracks: np.ndarray,
) -> Tuple[List[dict], dict]:
    """Recover per-frame rigid deltas from lifted Artipoint/CoTracker tracks."""
    reference = np.asarray(reference_points_world, dtype=np.float32)
    tracks = np.asarray(tracks_world, dtype=np.float32)
    valid = np.asarray(valid_tracks, dtype=bool)
    debug = {
        "visible_point_counts": [],
        "inlier_counts": [],
        "rigid_rmse_m": [],
        "used_frames": [],
    }
    if tracks.ndim != 3 or valid.ndim != 2 or len(reference) != tracks.shape[1]:
        return [], debug
    history: List[dict] = []
    n_frames = min(len(clip.samples), tracks.shape[0])
    n_total_tracks = int(tracks.shape[1])
    debug["pose_fails_by_reason"] = {
        "too_few_visible_tracks": 0,
        "ransac_returned_none": 0,
        "too_few_inliers": 0,
    }
    for frame_idx in range(n_frames):
        frame_valid = valid[frame_idx] & np.isfinite(reference).all(axis=1) & np.isfinite(tracks[frame_idx]).all(axis=1)
        visible_count = int(np.count_nonzero(frame_valid))
        debug["visible_point_counts"].append(visible_count)
        if visible_count < ARTIPOINT_MIN_VISIBLE_POINTS:
            debug["pose_fails_by_reason"]["too_few_visible_tracks"] += 1
            print(
                f"[ArtiPoint-Procrustes] frame_idx={frame_idx} FAIL too_few_visible_tracks: "
                f"{visible_count}/{n_total_tracks} above visibility threshold "
                f"(need >= {ARTIPOINT_MIN_VISIBLE_POINTS}, "
                f"COTRACKER3_VIS_THRESHOLD={os.environ.get('COTRACKER3_VIS_THRESHOLD', '0.70')})"
            )
            continue
        transform, inliers, rmse = estimate_rigid_transform_ransac(
            reference[frame_valid],
            tracks[frame_idx, frame_valid],
        )
        if transform is None:
            debug["pose_fails_by_reason"]["ransac_returned_none"] += 1
            print(
                f"[ArtiPoint-Procrustes] frame_idx={frame_idx} FAIL ransac_returned_none: "
                f"had {visible_count}/{n_total_tracks} visible tracks but Kabsch "
                f"could not find a consensus rigid transform."
            )
            continue
        inlier_count = int(np.count_nonzero(inliers))
        if inlier_count < ARTIPOINT_MIN_VISIBLE_POINTS:
            debug["pose_fails_by_reason"]["too_few_inliers"] += 1
            print(
                f"[ArtiPoint-Procrustes] frame_idx={frame_idx} FAIL too_few_inliers: "
                f"{inlier_count}/{visible_count} inliers out of {n_total_tracks} total tracks "
                f"(need >= {ARTIPOINT_MIN_VISIBLE_POINTS}). RMSE={rmse:.4f} m."
            )
            continue
        frame_id = int(clip.samples[frame_idx].get("prefix", clip.frame_start + frame_idx))
        debug["inlier_counts"].append(inlier_count)
        debug["rigid_rmse_m"].append(float(rmse))
        debug["used_frames"].append(frame_id)
        history.append(
            {
                "frame": frame_id,
                "track_frame_idx": int(frame_idx),
                "pose": np.asarray(transform, dtype=np.float32),
                "source": "artipoint_surface_tracks",
                "confidence": 1.0,
                "clip_id": int(clip.clip_id),
                "visible_points": visible_count,
                "inlier_points": inlier_count,
                "rigid_rmse_m": float(rmse),
            }
        )
    return history, debug



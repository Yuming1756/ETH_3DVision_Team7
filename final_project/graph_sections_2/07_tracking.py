# ======================================================================
# graph.py section: 07_tracking.py
# Tracking-point selection, rigid transforms, CoTracker lifting, LostFound stepping
#
# NOTE: not a standalone module. Loaded by graph.py into a shared
# namespace via exec(); names from earlier sections are in scope here.
# ======================================================================
from __future__ import annotations
def infer_articulated_part_prompt_label(element_label: str, object_label: str) -> Optional[str]:
    """Return a text prompt for the moving child part of a larger static parent."""
    element_text = str(element_label or "").strip().lower()
    object_text = str(object_label or "").strip().lower()
    if not element_text or not object_text or not is_articulated_parent_body_pair(element_text, object_text):
        return None
    for part_phrase in ARTICULATED_PART_LABELS:
        if has_label_phrase(element_text, (part_phrase,)):
            if has_label_phrase(object_text, (part_phrase,)):
                return str(object_label).strip()
            return f"{str(object_label).strip()} {part_phrase}".strip()
    return None



def sample_tracking_points(points: np.ndarray, max_points: int = LOSTFOUND_TRACK_MAX_POINTS) -> np.ndarray:
    """Subsample a point cloud to a stable set of tracking points for LostFound pose updates."""
    pts = np.asarray(points, dtype=np.float32)
    if len(pts) <= max_points:
        return pts
    indices = np.linspace(0, len(pts) - 1, num=max_points, dtype=np.int32)
    return pts[indices]


def select_hand_local_tracking_points(
    points: np.ndarray,
    hand_positions: Sequence[Optional[np.ndarray]],
    max_points: int = LOSTFOUND_TRACK_MAX_POINTS,
    candidate_points: int = LOSTFOUND_TRACK_CANDIDATE_POINTS,
    radius: float = LOSTFOUND_TRACK_HAND_RADIUS,
) -> np.ndarray:
    """Select a compact tracking subset that stays close to the acting hand over the clip."""
    pts = np.asarray(points, dtype=np.float32)
    if len(pts) == 0:
        return np.zeros((0, 3), dtype=np.float32)
    valid_hands = [np.asarray(hand, dtype=np.float32) for hand in hand_positions if hand is not None]
    if not valid_hands:
        return sample_tracking_points(pts, max_points=max_points)
    hand_tree = KDTree(np.stack(valid_hands, axis=0))
    distances, _ = hand_tree.query(pts, k=1)
    distances = np.asarray(distances, dtype=np.float32)
    nearby = pts[distances <= float(radius)]
    if len(nearby) >= LOSTFOUND_TRACK_MIN_POINTS:
        candidate = nearby
    else:
        keep = min(len(pts), max(int(candidate_points), int(max_points), LOSTFOUND_TRACK_MIN_POINTS))
        if keep >= len(pts):
            candidate = pts
        else:
            indices = np.argpartition(distances, keep - 1)[:keep]
            candidate = pts[np.sort(indices)]
    return sample_tracking_points(candidate, max_points=max_points)


def first_valid_geometry(per_frame_geometry: Sequence[Tuple[np.ndarray, np.ndarray]]) -> Tuple[np.ndarray, np.ndarray]:
    """Return the earliest non-empty geometry from a per-frame lifted sequence."""
    for points, colors in per_frame_geometry:
        if len(points):
            return np.asarray(points, dtype=np.float32), np.asarray(colors, dtype=np.float32)
    return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)


def tracking_sequence_motion_extent(per_frame_geometry: Sequence[Tuple[np.ndarray, np.ndarray]]) -> float:
    """Estimate how much a tracked geometry moves over time from lifted-mask centroids."""
    centroids = [points.mean(axis=0) for points, _ in per_frame_geometry if len(points) >= MIN_TRACKED_POINTS_FOR_REUSE]
    if len(centroids) < 2:
        return 0.0
    reference = centroids[0]
    return float(max(np.linalg.norm(centroid - reference) for centroid in centroids[1:]))


def apply_homogeneous_transform(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """Apply a 4x4 rigid transform to a point cloud."""
    pts = np.asarray(points, dtype=np.float32)
    if len(pts) == 0:
        return np.zeros((0, 3), dtype=np.float32)
    pts_h = np.concatenate([pts, np.ones((len(pts), 1), dtype=np.float32)], axis=1)
    return (np.asarray(transform, dtype=np.float32) @ pts_h.T).T[:, :3].astype(np.float32)


def rotation_degrees_from_transform(transform: np.ndarray) -> float:
    """Return the angle in degrees represented by the rotational part of a rigid transform."""
    rotation = np.asarray(transform, dtype=np.float64)[:3, :3]
    trace = np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(trace)))


def slerp_rotation_step(
    previous: Optional[np.ndarray],
    current: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """SLERP one rotation step from previous toward current by `alpha` in [0, 1].

    alpha=0 keeps th  (first frame) or alpha is clamped to [0, 1].
    """
    current_arr = np.asarray(current, dtype=np.float32)
    if previous is None:
        return current_arr
    alpha = float(np.clip(alpha, 0.0, 1.0))
    if alpha >= 1.0:
        return current_arr
    if alpha <= 0.0:
        return np.asarray(previous, dtype=np.float32)
    rots = Rotation.from_matrix(np.stack([
        np.asarray(previous, dtype=np.float64),
        current_arr.astype(np.float64),
    ], axis=0))
    slerp = Slerp([0.0, 1.0], rots)
    return slerp(alpha).as_matrix().astype(np.float32)


def select_lostfound_translation(
    rotation_obj_world: np.ndarray,
    hand_pos: Optional[np.ndarray],
    delta_obj: Optional[np.ndarray],
    world_from_color: np.ndarray,
    t_vec: np.ndarray,
    use_hand: bool,
    tracked_pose_index: int,
) -> Tuple[Optional[np.ndarray], dict]:
    """Use strict LostFound-style wrist-anchor translation with no PnP fallback."""
    rotation_obj_world = np.asarray(rotation_obj_world, dtype=np.float32)
    debug = {"mode": "wrist_missing", "blend": 0.0, "disagreement_m": 0.0}
    if not use_hand or hand_pos is None or delta_obj is None:
        return None, debug

    hand_pos = np.asarray(hand_pos, dtype=np.float32)
    delta_obj = np.asarray(delta_obj, dtype=np.float32)
    translation_hand = hand_pos - (rotation_obj_world @ delta_obj)
    world_from_color = np.asarray(world_from_color, dtype=np.float32)
    t_vec = np.asarray(t_vec, dtype=np.float32)
    translation_pnp = (world_from_color[:3, :3] @ t_vec) + world_from_color[:3, 3]
    debug["mode"] = "wrist"
    debug["blend"] = 1.0
    debug["disagreement_m"] = float(np.linalg.norm(translation_hand - translation_pnp))
    return np.asarray(translation_hand, dtype=np.float32), debug


def _coverage_mask_from_projection(
    points_world: np.ndarray,
    image_rgb: np.ndarray,
    camera_pose: np.ndarray,
    K_color: np.ndarray,
    T_cam_color: np.ndarray,
    dilate_px: int = 8,
) -> Optional[np.ndarray]:
    """Project 3-D prior points to image space and return a uint8 mask that covers
    every pixel where the prior is visible (dilated to fill small gaps)."""
    pts = np.asarray(points_world, dtype=np.float32)
    if len(pts) == 0:
        return None
    color_from_world = (
        np.linalg.inv(np.asarray(T_cam_color, dtype=np.float64))
        @ np.linalg.inv(np.asarray(camera_pose, dtype=np.float64))
    )
    homogeneous = np.concatenate([pts.astype(np.float64), np.ones((len(pts), 1), dtype=np.float64)], axis=1)
    cam_pts = (color_from_world @ homogeneous.T).T[:, :3]
    in_front = cam_pts[:, 2] > 1e-4
    if not np.any(in_front):
        return None
    cam_pts = cam_pts[in_front]
    pixels = (np.asarray(K_color, dtype=np.float64) @ cam_pts.T).T
    px = pixels[:, :2] / pixels[:, 2:3]
    height, width = image_rgb.shape[:2]
    inside = (px[:, 0] >= 0) & (px[:, 0] < width) & (px[:, 1] >= 0) & (px[:, 1] < height)
    if not np.any(inside):
        return None
    px = px[inside].astype(np.int32)
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[px[:, 1], px[:, 0]] = 255
    if dilate_px > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate_px + 1, 2 * dilate_px + 1))
        mask = cv2.dilate(mask, kernel)
    return mask


def detect_corner_queries_in_image(
    points_world: np.ndarray,
    image_rgb: np.ndarray,
    depth_m: np.ndarray,
    camera_pose: np.ndarray,
    K_color: np.ndarray,
    T_cam_color: np.ndarray,
    max_corners: int = 512,
    quality_level: float = 0.01,
    min_distance: float = 5.0,
    depth_lift_radius: int = 3,
    hand_world: Optional[np.ndarray] = None,
    explicit_mask: Optional[np.ndarray] = None,
    world_lookup: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Detect high-quality 2-D image corners inside an explicit segmentation mask.

    Returns ``(queries_2d, queries_world)`` where:
      - ``queries_2d`` has shape (M, 3) — CoTracker's (frame, x, y) format,
        with frame set to 0 (filled in by the caller for the actual start frame)
      - ``queries_world`` has shape (M, 3) — 3-D anchor positions used as
        Procrustes references for each query, obtained by lifting the corner
        pixel to world via the per-frame depth map.

    Replaces project_tracking_points_to_image for the ArtiPoint path: instead of
    projecting arbitrary prior 3-D points (which often land in flat textureless
    regions of a door), we run cv2.goodFeaturesToTrack on the original image,
    constrained to the provided segmentation mask, so every query pixel is
    guaranteed to be a high-texture corner. The resulting tracks are
    dramatically cleaner on metal door handles, drawer pulls, knobs — exactly
    the things CoTracker3 locks onto in the ArtiPoint paper's examples.
    """
    # Fallback-free mode: corner queries must come from an explicit mask rather
    # than a projected prior silhouette.
    if explicit_mask is None:
        raise RuntimeError(
            "[ArtiPoint][Error] detect_corner_queries_in_image requires an explicit segmentation mask; "
            "projected-prior mask fallback is disabled."
        )
    em = np.asarray(explicit_mask, dtype=bool)
    if em.shape[:2] != image_rgb.shape[:2] or int(np.count_nonzero(em)) < 16:
        raise RuntimeError(
            "[ArtiPoint][Error] detect_corner_queries_in_image received an invalid or too-small segmentation mask."
        )
    mask = (em.astype(np.uint8) * 255)
    height, width = image_rgb.shape[:2]
    hand_arr = None
    if hand_world is not None:
        hand_arr = np.asarray(hand_world, dtype=np.float64).reshape(-1)
        if hand_arr.size != 3 or not np.all(np.isfinite(hand_arr)):
            hand_arr = None
    if hand_arr is not None:
        color_from_world = (
            np.linalg.inv(np.asarray(T_cam_color, dtype=np.float64))
            @ np.linalg.inv(np.asarray(camera_pose, dtype=np.float64))
        )
        hand_h = np.concatenate([hand_arr[:3], np.ones(1, dtype=np.float64)], axis=0)
        hand_cam = (color_from_world @ hand_h)[:3]
        if float(hand_cam[2]) > 1e-4:
            hand_px_h = np.asarray(K_color, dtype=np.float64) @ hand_cam
            hand_px = hand_px_h[:2] / max(float(hand_px_h[2]), 1e-9)
            hx = int(round(float(hand_px[0])))
            hy = int(round(float(hand_px[1])))
            if 0 <= hx < width and 0 <= hy < height:
                component_mask = mask.copy()
                num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
                if num_labels > 1:
                    best_label = None
                    best_score = None
                    hand_xy = np.array([float(hx), float(hy)], dtype=np.float32)
                    for label_idx in range(1, num_labels):
                        area = int(stats[label_idx, cv2.CC_STAT_AREA])
                        if area <= 0:
                            continue
                        centroid = np.asarray(centroids[label_idx], dtype=np.float32)
                        score = (float(np.linalg.norm(centroid - hand_xy)), -area)
                        if best_score is None or score < best_score:
                            best_score = score
                            best_label = label_idx
                    if best_label is not None:
                        component_mask = ((labels == best_label).astype(np.uint8) * 255)
                if int(np.count_nonzero(component_mask)) >= 16:
                    mask = component_mask
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY) if image_rgb.ndim == 3 else image_rgb
    corners = cv2.goodFeaturesToTrack(
        gray,
        maxCorners=int(max_corners),
        qualityLevel=float(quality_level),
        minDistance=float(min_distance),
        mask=mask,
        useHarrisDetector=False,
        blockSize=5,
    )
    if corners is None or len(corners) == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)
    queries_2d_list: List[List[float]] = []
    queries_world_list: List[np.ndarray] = []
    # Prefer the precomputed native depth-camera world lookup (Path 1 lift
    # convention) when the caller provided one. Falls back to the legacy
    # color-aligned shortcut (Path 2 lift) otherwise.
    use_lookup = (
        world_lookup is not None
        and isinstance(world_lookup, np.ndarray)
        and world_lookup.shape == (height, width, 3)
    )
    if not use_lookup:
        depth_arr = np.asarray(depth_m, dtype=np.float32)
        if depth_arr.shape[:2] != (height, width):
            depth_arr = cv2.resize(depth_arr, (width, height), interpolation=cv2.INTER_NEAREST)
    for corner in corners.reshape(-1, 2):
        x, y = float(corner[0]), float(corner[1])
        if use_lookup:
            ix, iy = int(round(x)), int(round(y))
            if not (0 <= ix < width and 0 <= iy < height):
                continue
            # Search a small neighbourhood for the closest valid native-lift
            # world position — GFTT corners can sit ON an edge where the
            # exact pixel has no depth coverage (z-buffer winner was filtered
            # earlier), but a neighbour does.
            ws = world_lookup[
                max(0, iy - depth_lift_radius):min(height, iy + depth_lift_radius + 1),
                max(0, ix - depth_lift_radius):min(width, ix + depth_lift_radius + 1),
            ].reshape(-1, 3)
            ws = ws[np.isfinite(ws).all(axis=1)]
            if ws.shape[0] == 0:
                continue
            world = ws.mean(axis=0)
        else:
            z = _depth_patch_median(depth_arr, int(round(x)), int(round(y)), radius=depth_lift_radius)
            if z is None:
                continue
            world = _pixel_to_world(x, y, z, K_color, np.asarray(camera_pose), T_cam_color)
        if not np.all(np.isfinite(world)):
            continue
        queries_2d_list.append([0.0, x, y])
        queries_world_list.append(np.asarray(world, dtype=np.float32))
    if not queries_2d_list:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)
    return (
        np.asarray(queries_2d_list, dtype=np.float32),
        np.stack(queries_world_list, axis=0).astype(np.float32),
    )


def project_tracking_points_to_image(
    tracking_points_world: np.ndarray,
    image_rgb: np.ndarray,
    camera_pose: np.ndarray,
    K_color: np.ndarray,
    T_cam_color: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Project world-space tracking points into the current color image using the active camera intrinsics."""
    points_world = np.asarray(tracking_points_world, dtype=np.float32)
    if len(points_world) == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)

    color_from_world = np.linalg.inv(np.asarray(T_cam_color, dtype=np.float64)) @ np.linalg.inv(np.asarray(camera_pose, dtype=np.float64))
    points_h = np.concatenate([points_world.astype(np.float64), np.ones((len(points_world), 1), dtype=np.float64)], axis=1)
    points_color = (color_from_world @ points_h.T).T[:, :3]
    valid = points_color[:, 2] > 1e-4
    if not np.any(valid):
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)

    points_color = points_color[valid]
    points_world = points_world[valid]
    pixels_h = (np.asarray(K_color, dtype=np.float64) @ points_color.T).T
    pixels = pixels_h[:, :2] / pixels_h[:, 2:3]
    height, width = image_rgb.shape[:2]
    inside = (
        (pixels[:, 0] >= 0.0)
        & (pixels[:, 0] < width)
        & (pixels[:, 1] >= 0.0)
        & (pixels[:, 1] < height)
    )
    if not np.any(inside):
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)

    pixels = pixels[inside]
    points_world = points_world[inside]
    queries = np.concatenate([np.zeros((len(pixels), 1), dtype=np.float32), pixels.astype(np.float32)], axis=1)
    rounded = np.round(queries[:, 1:]).astype(np.int32)
    _, unique_indices = np.unique(rounded, axis=0, return_index=True)
    keep = np.sort(unique_indices)
    return queries[keep].astype(np.float32), points_world[keep].astype(np.float32)


def lostfound_process_step(
    cotracker: CoTrackerOnlinePredictor,
    window_frames: Sequence[np.ndarray],
    is_first_step: bool,
    queries: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run one LostFound-style online CoTracker update over the current frame window."""
    video_chunk = (
        torch.tensor(np.stack(window_frames[-cotracker.step * 2 :]), device='cuda')
        .float()
        .permute(0, 3, 1, 2)[None]
    )
    return cotracker(video_chunk, is_first_step=is_first_step, queries=queries)


def _new_lostfound_hand_debug(hand_name: str, clip: ClipInfo, label: str) -> dict:
    """Create a compact debug record for one LostFound hand attempt."""
    return {
        "label": str(label),
        "clip_id": int(clip.clip_id),
        "frame_start": int(clip.frame_start),
        "frame_end": int(clip.frame_end),
        "hand": hand_name,
        "status": "not_started",
        "failure_reason": "",
        "activation_attempts": 0,
        "pose_updates": 0,
        "used_frames": [],
        "start_fail_counts": {
            "no_detection": 0,
            "hand_missing": 0,
            "nearest_node_too_far_or_missing": 0,
            "future_not_far_enough": 0,
            "contact_threshold_not_met": 0,
            "too_few_projected_queries": 0,
            "tracker_bootstrap_failed": 0,
            "initial_pose_failed": 0,
        },
        "update_fail_counts": {
            "tracker_prediction_missing": 0,
            "pose_estimation_failed": 0,
            "contact_gate_dropped": 0,
        },
        "distance_threshold_m": float(LOSTFOUND_HAND_DISTANCE_THRESHOLD),
        "tracking_point_selection_radius_m": float(LOSTFOUND_TRACK_HAND_RADIUS),
        "hand_anchor_mode": None,
        "hand_to_tracking_distances_m": [],
        "hand_to_tracking_distance_stats_m": {},
        "tracking_point_to_hand_distance_stats_m": {},
        "projected_query_counts": [],
        "visible_point_counts": [],
    }


def _distance_stats(values: Sequence[float], threshold: Optional[float] = None) -> dict:
    """Summarize a list of distances for lightweight LostFound debugging."""
    if not values:
        return {}
    arr = np.asarray(values, dtype=np.float32)
    stats = {
        "count": int(arr.size),
        "min": float(arr.min()),
        "median": float(np.median(arr)),
        "max": float(arr.max()),
    }
    if threshold is not None:
        stats["within_threshold_count"] = int(np.count_nonzero(arr <= float(threshold)))
    return stats


def compute_lostfound_pose(
    pred_tracks: torch.Tensor,
    pred_visibility: torch.Tensor,
    queries_object: np.ndarray,
    frame_index: int,
    K_color: np.ndarray,
) -> Tuple[Optional[Tuple[np.ndarray, np.ndarray]], dict]:
    """Solve the LostFound PnP pose update for one tracked frame index and report why it failed."""
    current_tracks = pred_tracks[:, frame_index, :, :].squeeze(0).detach().cpu().numpy().astype(np.float32)
    current_visibility = pred_visibility[:, frame_index, :].squeeze(0).detach().cpu().numpy().astype(bool)
    visible_3d = np.asarray(queries_object, dtype=np.float32)[current_visibility]
    visible_2d = current_tracks[current_visibility]
    total_tracks = int(current_visibility.shape[0])
    visible_count = int(len(visible_3d))
    debug = {
        "frame_index": int(frame_index),
        "visible_points": visible_count,
        "total_tracks": total_tracks,
        "visibility_ratio": float(visible_count) / float(max(total_tracks, 1)),
        "pnp_success": False,
        "failure_reason": "",
    }
    if visible_count < LOSTFOUND_POSE_MIN_VISIBLE_POINTS:
        debug["failure_reason"] = "too_few_visible_points"
        print(
            f"[LostFound-PnP] frame_idx={frame_index} FAIL too_few_visible_points: "
            f"{visible_count}/{total_tracks} above visibility threshold "
            f"(need >= {LOSTFOUND_POSE_MIN_VISIBLE_POINTS}, "
            f"COTRACKER3_VIS_THRESHOLD={os.environ.get('COTRACKER3_VIS_THRESHOLD', '0.70')})"
        )
        return None, debug
    success, rvec_obj_cam, tvec_obj_cam, _ = cv2.solvePnPRansac(
        visible_3d,
        visible_2d,
        np.asarray(K_color, dtype=np.float64),
        np.zeros((5, 1), dtype=np.float64),
    )
    if not success:
        debug["failure_reason"] = "solvepnp_failed"
        print(
            f"[LostFound-PnP] frame_idx={frame_index} FAIL solvepnp_failed: "
            f"had {visible_count}/{total_tracks} visible tracks but RANSAC could not "
            f"find a consensus pose."
        )
        return None, debug
    debug["pnp_success"] = True
    rotation_obj_cam, _ = cv2.Rodrigues(rvec_obj_cam)
    return (rotation_obj_cam.astype(np.float32), tvec_obj_cam.flatten().astype(np.float32)), debug



def _cotracker3_checkpoint(online: bool = False) -> Optional[Path]:
    """Return the configured CoTracker3 checkpoint, if present."""
    checkpoint_path = COTRACKER3_ONLINE_CHECKPOINT_PATH if online else COTRACKER3_OFFLINE_CHECKPOINT_PATH
    return checkpoint_path if checkpoint_path.exists() else None


def _artipoint_cotracker_checkpoint() -> Optional[Path]:
    """Return the CoTracker3 OFFLINE checkpoint, or None.

    We never fall back to CoTracker2: user requirement is CoTracker3 + offline
    only. If the scaled_offline.pth weights are missing, the ArtiPoint path
    refuses to run rather than silently downgrading the tracker quality.
    """
    return _cotracker3_checkpoint(online=False)


def _normalize3(vector: np.ndarray, fallback: Sequence[float] = (0.0, 0.0, 1.0)) -> np.ndarray:
    """Normalize a 3-vector with a deterministic fallback for degenerate cases."""
    vec = np.asarray(vector, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(vec))
    if norm < 1e-10 or not np.all(np.isfinite(vec)):
        return np.asarray(fallback, dtype=np.float64)
    return vec / norm


def _artipoint_axis_prior(label: str) -> Optional[np.ndarray]:
    """Weak world-axis prior for common side-hinged doors in egocentric kitchens."""
    text = str(label or "").strip().lower()
    if not text or "door" not in text:
        return None
    # Fridge/cabinet/closet doors are usually side-hinged around gravity. Do not
    # force this for oven/microwave/dishwasher doors, which may hinge downward.
    downward = ("microwave", "oven", "dishwasher", "washer", "dryer")
    if any(word in text for word in downward):
        return None
    vertical = ("refrigerator", "fridge", "cabinet", "cupboard", "closet", "door")
    if any(word in text for word in vertical):
        return np.array([0.0, 0.0, 1.0], dtype=np.float64)
    return None


def estimate_rigid_transform_svd(source_points: np.ndarray, target_points: np.ndarray) -> Optional[np.ndarray]:
    """Estimate the rigid transform mapping source_points to target_points."""
    src = np.asarray(source_points, dtype=np.float64)
    dst = np.asarray(target_points, dtype=np.float64)
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 3 or len(src) < 3:
        return None
    src_centroid = src.mean(axis=0)
    dst_centroid = dst.mean(axis=0)
    src_centered = src - src_centroid
    dst_centered = dst - dst_centroid
    H = src_centered.T @ dst_centered
    try:
        U, _S, Vt = np.linalg.svd(H)
    except np.linalg.LinAlgError:
        return None
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0.0:
        Vt[-1, :] *= -1.0
        R = Vt.T @ U.T
    t = dst_centroid - R @ src_centroid
    transform = np.eye(4, dtype=np.float32)
    transform[:3, :3] = R.astype(np.float32)
    transform[:3, 3] = t.astype(np.float32)
    return transform


def estimate_rigid_transform_ransac(
    source_points: np.ndarray,
    target_points: np.ndarray,
    min_inliers: int = ARTIPOINT_MIN_VISIBLE_POINTS,
    threshold_m: float = ARTIPOINT_RANSAC_INLIER_M,
    max_iterations: int = ARTIPOINT_RANSAC_ITERS,
) -> Tuple[Optional[np.ndarray], np.ndarray, float]:
    """Robust 3D-3D Kabsch using a small deterministic RANSAC loop."""
    src = np.asarray(source_points, dtype=np.float32)
    dst = np.asarray(target_points, dtype=np.float32)
    empty = np.zeros((0,), dtype=bool)
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 3 or len(src) < 3:
        return None, empty, float("inf")
    finite = np.isfinite(src).all(axis=1) & np.isfinite(dst).all(axis=1)
    src = src[finite]
    dst = dst[finite]
    if len(src) < max(3, min_inliers):
        return None, np.zeros(len(src), dtype=bool), float("inf")

    rng = np.random.default_rng(13)
    best_inliers = np.zeros(len(src), dtype=bool)
    best_score = -1
    best_error = float("inf")
    for _ in range(int(max_iterations)):
        sample_indices = rng.choice(len(src), size=3, replace=False)
        candidate = estimate_rigid_transform_svd(src[sample_indices], dst[sample_indices])
        if candidate is None:
            continue
        pred = apply_homogeneous_transform(src, candidate)
        errors = np.linalg.norm(pred - dst, axis=1)
        inliers = errors <= float(threshold_m)
        score = int(np.count_nonzero(inliers))
        if score == 0:
            continue
        mean_error = float(np.mean(errors[inliers]))
        if score > best_score or (score == best_score and mean_error < best_error):
            best_score = score
            best_error = mean_error
            best_inliers = inliers

    if int(np.count_nonzero(best_inliers)) < max(3, min_inliers):
        # If the cloud is coherent but just noisier than the threshold, use all
        # points rather than silently dropping the articulated path.
        transform = estimate_rigid_transform_svd(src, dst)
        if transform is None:
            return None, best_inliers, float("inf")
        pred = apply_homogeneous_transform(src, transform)
        errors = np.linalg.norm(pred - dst, axis=1)
        return transform, np.ones(len(src), dtype=bool), float(np.mean(errors))

    refined = estimate_rigid_transform_svd(src[best_inliers], dst[best_inliers])
    if refined is None:
        return None, best_inliers, float("inf")
    pred = apply_homogeneous_transform(src[best_inliers], refined)
    rmse = float(np.sqrt(np.mean(np.sum((pred - dst[best_inliers]) ** 2, axis=1))))
    return refined, best_inliers, rmse


def lift_cotracker_tracks_to_world(
    tracks_2d: np.ndarray,
    visibility: np.ndarray,
    clip_depths: Sequence[np.ndarray],
    rgb_list: Sequence[np.ndarray],
    T_wc_list: Sequence[np.ndarray],
    K_color: np.ndarray,
    T_cam_color: np.ndarray,
    world_lookups: Optional[Sequence[np.ndarray]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Lift CoTracker RGB tracks into world-space 3D tracks.

    When ``world_lookups[frame_idx]`` is provided (shape ``(H, W, 3)``), we
    sample the native depth-camera lift result (Path 1 convention) — same
    convention used by the prior-cloud builder. Otherwise we fall back to the
    legacy color-aligned shortcut (Path 2 convention).
    """
    tracks = np.asarray(tracks_2d, dtype=np.float32)
    vis = np.asarray(visibility)
    if vis.dtype != bool:
        vis = vis > 0.5
    if tracks.ndim != 3 or tracks.shape[-1] != 2:
        return np.zeros((0, 0, 3), dtype=np.float32), np.zeros((0, 0), dtype=bool)
    n_frames = min(len(rgb_list), len(clip_depths), len(T_wc_list), tracks.shape[0], vis.shape[0])
    n_tracks = tracks.shape[1]
    world_tracks = np.full((n_frames, n_tracks, 3), np.nan, dtype=np.float32)
    valid = np.zeros((n_frames, n_tracks), dtype=bool)
    use_lookup = world_lookups is not None and len(world_lookups) >= n_frames
    for frame_idx in range(n_frames):
        rgb = np.asarray(rgb_list[frame_idx])
        height, width = rgb.shape[:2]
        frame_lookup = world_lookups[frame_idx] if use_lookup else None
        frame_lookup_ok = (
            frame_lookup is not None
            and isinstance(frame_lookup, np.ndarray)
            and frame_lookup.shape == (height, width, 3)
        )
        if not frame_lookup_ok:
            depth_m = np.asarray(clip_depths[frame_idx], dtype=np.float32)
            if depth_m.shape[:2] != (height, width):
                depth_m = cv2.resize(depth_m, (width, height), interpolation=cv2.INTER_NEAREST)
        for track_idx in range(n_tracks):
            if not bool(vis[frame_idx, track_idx]):
                continue
            x, y = tracks[frame_idx, track_idx]
            if not np.isfinite(x) or not np.isfinite(y):
                continue
            if x < 0.0 or x >= width or y < 0.0 or y >= height:
                continue
            if frame_lookup_ok:
                ix, iy = int(round(float(x))), int(round(float(y)))
                ws = frame_lookup[
                    max(0, iy - 3):min(height, iy + 4),
                    max(0, ix - 3):min(width, ix + 4),
                ].reshape(-1, 3)
                ws = ws[np.isfinite(ws).all(axis=1)]
                if ws.shape[0] == 0:
                    continue
                world = ws.mean(axis=0)
            else:
                depth = _depth_patch_median(depth_m, int(round(float(x))), int(round(float(y))), radius=3)
                if depth is None:
                    continue
                world = _pixel_to_world(float(x), float(y), depth, K_color, np.asarray(T_wc_list[frame_idx]), T_cam_color)
            if not np.all(np.isfinite(world)):
                continue
            world_tracks[frame_idx, track_idx] = world
            valid[frame_idx, track_idx] = True
    return world_tracks, valid




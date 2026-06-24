# ======================================================================
# graph.py section: 03_frames_pose_hands.py
# Topic indexing, pose buffer, health gating, hand/contact/interaction inference
#
# NOTE: not a standalone module. Loaded by graph.py into a shared
# namespace via exec(); names from earlier sections are in scope here.
# ======================================================================
from __future__ import annotations
def parse_topic_frame(path: Path) -> TopicFrame:
    """Parse the frame index key and nanosecond timestamp encoded in an extracted filename."""
    stem = path.stem
    key, _, ts_str = stem.partition("_")
    timestamp_ns = int(ts_str) if ts_str else 0
    return TopicFrame(key=key, timestamp_ns=timestamp_ns, path=path)


def build_topic_index(topic_dir: Path, pattern: str) -> Tuple[List[TopicFrame], np.ndarray]:
    """Load and sort one extracted topic directory by timestamp."""
    frames = [parse_topic_frame(path) for path in topic_dir.glob(pattern)] if topic_dir.exists() else []
    frames.sort(key=lambda item: item.timestamp_ns)
    timestamps = np.asarray([frame.timestamp_ns for frame in frames], dtype=np.int64)
    return frames, timestamps


def nearest_topic_frame(
    frames: Sequence[TopicFrame],
    timestamps_ns: np.ndarray,
    query_timestamp_ns: int,
) -> Optional[TopicFrame]:
    """Return the topic frame whose timestamp is closest to the requested time."""
    if not frames:
        return None
    idx = bisect_left(timestamps_ns.tolist(), int(query_timestamp_ns))
    candidates = []
    if idx < len(frames):
        candidates.append(frames[idx])
    if idx > 0:
        candidates.append(frames[idx - 1])
    if not candidates:
        return None
    return min(candidates, key=lambda frame: abs(int(frame.timestamp_ns) - int(query_timestamp_ns)))


def build_color_pixel_world_lookup(
    depth_raw: np.ndarray,
    depth_grid: np.ndarray,
    T_wc: np.ndarray,
    T_cam_depth: np.ndarray,
    T_cam_color: np.ndarray,
    K_color: np.ndarray,
    D_color: Optional[np.ndarray],
    color_w: int,
    color_h: int,
    depth_scale: Optional[float] = None,
    depth_max: float = DEPTH_MAX,
) -> np.ndarray:
    """Native depth-camera world lookup keyed by color image pixel.

    Returns ``(color_h, color_w, 3)`` of world XYZ positions for each color
    pixel — NaN where no depth covers that pixel. Uses the SAME lift convention
    as ``lift_depth_frame_to_world_with_rgb`` (the prior-cloud builder):

      1. Every native depth-camera pixel is backprojected via ``K_depth``-derived
         rays.
      2. Transformed into the color camera frame via the static extrinsic.
      3. Projected into the (undistorted) color image to find which color pixel
         it lands on.
      4. The closest such depth-camera 3-D point wins per color pixel
         (z-buffer), and we record the corresponding world point computed via
         ``T_wc @ T_cam_depth``.

    Downstream consumers (corner-query lift, CoTracker track lift) can then
    just index ``world_lookup[round(y), round(x)]`` to get a world position
    that's consistent with how the prior cloud was constructed.
    """
    depth_scale = resolve_depth_scale(depth_scale)
    depth_m = depth_raw.astype(np.float32) / depth_scale
    out = np.full((color_h, color_w, 3), np.nan, dtype=np.float32)
    out_depth = np.full((color_h, color_w), np.inf, dtype=np.float32)
    valid = (depth_m > 1e-4) & (depth_m < depth_max)
    if not np.any(valid):
        return out
    rays = depth_grid
    valid_idx = np.where(valid)
    rays_valid = rays[valid_idx]
    depth_values = depth_m[valid_idx].astype(np.float32)
    # Transform depth-camera points into color-camera frame for image-grid
    # association (ArtiPoint-style: T_depth_color = inv(T_cam_color) @ T_cam_depth).
    T_depth_color = np.asarray(np.linalg.inv(T_cam_color) @ T_cam_depth, dtype=np.float32)
    R_depth_color = T_depth_color[:3, :3]
    t_depth_color = T_depth_color[:3, 3]
    pts_depth_sensor = rays_valid * depth_values[:, None]
    pts_color_sensor = (R_depth_color @ pts_depth_sensor.T).T + t_depth_color[None, :]
    mask_color = pts_color_sensor[:, 2] > 1e-4
    if not np.any(mask_color):
        return out
    rays_valid = rays_valid[mask_color]
    depth_values = depth_values[mask_color]
    pts_depth_sensor = pts_depth_sensor[mask_color]
    pts_color_sensor = pts_color_sensor[mask_color]
    pixels_h = (np.asarray(K_color, dtype=np.float32) @ pts_color_sensor.T).T
    pixels_raw = pixels_h[:, :2] / pixels_h[:, 2:3]
    if D_color is not None and np.any(np.abs(np.asarray(D_color, dtype=np.float64)) > 1e-12):
        pixels_color = distorted_pixels_to_undistorted(
            pixels_raw.astype(np.float32), K_color, np.asarray(D_color, dtype=np.float64)
        )
    else:
        pixels_color = pixels_raw.astype(np.float32)
    px = np.floor(pixels_color).astype(np.int32)
    inside = (px[:, 0] >= 0) & (px[:, 0] < color_w) & (px[:, 1] >= 0) & (px[:, 1] < color_h)
    if not np.any(inside):
        return out
    px = px[inside]
    pts_depth_sensor = pts_depth_sensor[inside]
    pts_color_z = pts_color_sensor[inside, 2]
    # Transform all valid depth-sensor 3D points to world space (via T_cam_depth,
    # NOT T_cam_color — this is the key correctness step ArtiPoint paths skip).
    world_from_depth = np.asarray(T_wc @ T_cam_depth, dtype=np.float32)
    pts_world = apply_homogeneous_transform(pts_depth_sensor, world_from_depth).astype(np.float32)
    # Z-buffer: per color pixel, keep the closest depth-sensor sample.
    flat = px[:, 1] * color_w + px[:, 0]
    order = np.argsort(pts_color_z)
    flat_sorted = flat[order]
    pts_world_sorted = pts_world[order]
    pts_color_z_sorted = pts_color_z[order]
    unique_flat, first_idx = np.unique(flat_sorted, return_index=True)
    out_flat = out.reshape(-1, 3)
    out_flat[unique_flat] = pts_world_sorted[first_idx]
    out_depth_flat = out_depth.reshape(-1)
    out_depth_flat[unique_flat] = pts_color_z_sorted[first_idx]
    return out


def build_pose_buffer(data_dir: Path, T_cam_depth: np.ndarray, T_depth_color: np.ndarray) -> PoseBuffer:
    """Build an interpolated pose buffer from extracted valid camera poses."""
    pose_frames, _pose_timestamps = build_topic_index(data_dir / "tf_camera", "*.json")
    pose_health = load_health_by_timestamp(data_dir / "tf_camera_health", missing_valid_default=False)
    valid_pose_frames = [frame for frame in pose_frames if pose_health.get(frame.timestamp_ns, False)]
    if not valid_pose_frames:
        raise RuntimeError(f"No valid camera poses found under {data_dir / 'tf_camera'}")

    timestamps_s = np.asarray([frame.timestamp_ns * 1e-9 for frame in valid_pose_frames], dtype=np.float64)
    camera_to_world = [matrix_to_rt(load_extrinsic(frame.path)) for frame in valid_pose_frames]
    depth_to_camera = matrix_to_rt(np.asarray(T_cam_depth, dtype=np.float64))
    rgb_to_depth = matrix_to_rt(np.asarray(T_depth_color, dtype=np.float64))
    return PoseBuffer(timestamps_s, camera_to_world, depth_to_camera, rgb_to_depth)


def load_health(health_dir: Path, missing_valid_default: Optional[bool] = False) -> Dict[str, bool]:
    """Load per-frame camera health flags if present."""
    health = {}
    if not health_dir.exists():
        return health
    for path in health_dir.glob("*.json"):
        key = path.stem.split("_", 1)[0]
        try:
            with open(path, "r", encoding="utf-8") as handle:
                data = json.load(handle).get("data", {})
                if "valid" in data:
                    health[key] = bool(data.get("valid", False))
                elif missing_valid_default is not None:
                    health[key] = bool(missing_valid_default)
                else:
                    health[key] = False
        except Exception:
            health[key] = False
    return health


def load_health_by_timestamp(health_dir: Path, missing_valid_default: Optional[bool] = False) -> Dict[int, bool]:
    """Load per-message health flags keyed by the message timestamp."""
    health = {}
    if not health_dir.exists():
        return health
    for path in health_dir.glob("*.json"):
        try:
            with open(path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            timestamp_ns = int(payload.get("timestamp_ns", 0))
            data = payload.get("data", {})
            if "valid" in data:
                health[timestamp_ns] = bool(data.get("valid", False))
            elif missing_valid_default is not None:
                health[timestamp_ns] = bool(missing_valid_default)
            else:
                health[timestamp_ns] = False
        except Exception:
            continue
    return health


def annotate_pose_blocks(samples: Sequence[dict]) -> None:
    """Assign contiguous valid-pose block ids to each sample."""
    next_block_id = 0
    current_block_id: Optional[int] = None
    prev_frame_idx: Optional[int] = None
    prev_valid = False
    for sample in samples:
        frame_idx = int(sample["prefix"])
        pose_valid = bool(sample.get("pose_valid", False))
        if pose_valid:
            contiguous = prev_valid and prev_frame_idx is not None and frame_idx == (prev_frame_idx + 1)
            if not contiguous:
                current_block_id = next_block_id
                next_block_id += 1
            sample["pose_block_id"] = current_block_id
        else:
            current_block_id = None
            sample["pose_block_id"] = None
        prev_valid = pose_valid
        prev_frame_idx = frame_idx


def filter_pose_consistent_samples(samples: Sequence[dict]) -> List[dict]:
    """Keep only samples inside valid tf_camera pose blocks."""
    valid_samples = [sample for sample in samples if sample.get("pose_valid", False) and sample.get("pose_block_id") is not None]
    dropped = len(samples) - len(valid_samples)
    if dropped:
        first_valid = valid_samples[0]["prefix"] if valid_samples else "none"
        print(f"Dropping {dropped} frames outside valid tf_camera pose blocks; first valid frame: {first_valid}")
    if valid_samples:
        block_ids = sorted({int(sample["pose_block_id"]) for sample in valid_samples if sample.get("pose_block_id") is not None})
        print(f"Using {len(valid_samples)} pose-consistent frames across {len(block_ids)} valid pose block(s): {block_ids}")
    return valid_samples


def check_hand_object_tracker(hand_dets: Optional[np.ndarray], obj_dets: Optional[np.ndarray], thresh: float = DETECTION_THRESH) -> bool:
    """Return whether there is a confident hand-object contact event in one frame."""
    if hand_dets is None or obj_dets is None:
        return False
    if hand_dets.shape[0] == 0 or obj_dets.shape[0] == 0:
        return False
    hand_confident = any(hand_dets[i, 4] > thresh for i in range(hand_dets.shape[0])) # threshold following lostofund, can be modified
    obj_confident = any(obj_dets[i, 4] > thresh for i in range(obj_dets.shape[0]))
    hand_in_contact = any(hand_dets[i, 5] > 0 for i in range(hand_dets.shape[0]))
    return hand_confident and obj_confident and hand_in_contact


def hand_object_contact_breakdown(
    hand_dets: Optional[np.ndarray],
    obj_dets: Optional[np.ndarray],
    thresh: float = DETECTION_THRESH,
) -> dict:
    """Return a per-frame diagnostic showing which contact prerequisite failed.

    Mirrors the boolean checks in ``check_hand_object_tracker`` but returns the
    individual signals so we can see at which frame each component dropped
    (hand detector confidence, object detector confidence, or the per-hand
    contact-state flag). Useful for understanding why an active interaction
    ended earlier than expected (e.g. drawer fully extended -> RCNN loses
    object detection -> hand_contact flips to False).
    """
    if hand_dets is None or obj_dets is None:
        # One detector is missing. Still surface the hand-side signals when the
        # hand IS present, so we can tell whether the hand was confidently in
        # contact while only the OBJECT detector dropped (e.g. fridge once the
        # door swings open) — otherwise "missing_dets" hides which side failed.
        hand_present = hand_dets is not None and getattr(hand_dets, "shape", [0])[0] > 0
        hand_scores_m = [float(hand_dets[i, 4]) for i in range(hand_dets.shape[0])] if hand_present else []
        hand_conf_m = any(s > thresh for s in hand_scores_m)
        hand_contact_m = any(float(hand_dets[i, 5]) > 0 for i in range(hand_dets.shape[0])) if hand_present else False
        return {
            "hand_dets_present": hand_dets is not None,
            "obj_dets_present": obj_dets is not None,
            "hand_confident": hand_conf_m,
            "obj_confident": False,
            "hand_in_contact": hand_contact_m,
            "best_hand_score": float(max(hand_scores_m)) if hand_scores_m else 0.0,
            "best_obj_score": 0.0,
            "contact": False,
            "first_failure": "missing_dets",
        }
    hand_scores = (
        [float(hand_dets[i, 4]) for i in range(hand_dets.shape[0])]
        if hand_dets.shape[0] else []
    )
    obj_scores = (
        [float(obj_dets[i, 4]) for i in range(obj_dets.shape[0])]
        if obj_dets.shape[0] else []
    )
    hand_confident = any(s > thresh for s in hand_scores)
    obj_confident = any(s > thresh for s in obj_scores)
    hand_in_contact = (
        any(float(hand_dets[i, 5]) > 0 for i in range(hand_dets.shape[0]))
        if hand_dets.shape[0] else False
    )
    contact = hand_confident and obj_confident and hand_in_contact
    if contact:
        first_failure = None
    elif not hand_confident:
        first_failure = "hand_detector_low_conf"
    elif not obj_confident:
        first_failure = "object_detector_low_conf"
    else:
        first_failure = "rcnn_no_in_contact_flag"
    return {
        "hand_dets_present": True,
        "obj_dets_present": True,
        "hand_confident": hand_confident,
        "obj_confident": obj_confident,
        "hand_in_contact": hand_in_contact,
        "best_hand_score": float(max(hand_scores)) if hand_scores else 0.0,
        "best_obj_score": float(max(obj_scores)) if obj_scores else 0.0,
        "contact": contact,
        "first_failure": first_failure,
    }


def hand_mask_from_detections(
    hand_dets: Optional[np.ndarray],
    image_h: int,
    image_w: int,
    K_color: Optional[np.ndarray] = None,
    D_color: Optional[np.ndarray] = None,
    thresh: float = DETECTION_THRESH,
    padding: int = 12,
) -> np.ndarray:
    """Build a conservative binary hand mask from detector boxes for reconstruction cleanup."""
    mask = np.zeros((image_h, image_w), dtype=bool)
    if hand_dets is None or len(hand_dets) == 0:
        return mask
    for row in hand_dets:
        if len(row) < 5 or row[4] < thresh:
            continue
        corners = np.array(
            [
                [row[0], row[1]],
                [row[2], row[1]],
                [row[0], row[3]],
                [row[2], row[3]],
            ],
            dtype=np.float32,
        )
        if K_color is not None and D_color is not None:
            corners = distorted_pixels_to_undistorted(corners, K_color, np.asarray(D_color, dtype=np.float64))
        x1 = int(np.floor(corners[:, 0].min()))
        y1 = int(np.floor(corners[:, 1].min()))
        x2 = int(np.ceil(corners[:, 0].max()))
        y2 = int(np.ceil(corners[:, 1].max()))
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(image_w, x2 + padding)
        y2 = min(image_h, y2 + padding)
        if x2 <= x1 or y2 <= y1:
            continue
        mask[y1:y2, x1:x2] = True
    return mask


def merge_human_exclusion_masks(detector_mask: np.ndarray, prompt_mask: Optional[np.ndarray]) -> np.ndarray:
    """Union detector boxes and SAM3 human masks into one reconstruction exclusion mask."""
    merged = np.asarray(detector_mask, dtype=bool)
    if prompt_mask is None:
        return merged
    return merged | np.asarray(prompt_mask, dtype=bool)


def load_hand_world_wrist_positions(
    tf_hands_path: Optional[Path],
    T_wc: np.ndarray,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Load left/right wrist anchors in world coordinates from a tf_hands frame."""
    if tf_hands_path is None or not tf_hands_path.exists():
        return None, None

    with open(tf_hands_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    left_wrist_world = None
    right_wrist_world = None
    for transform in payload.get("data", {}).get("transforms", []):
        child = transform.get("child_frame_id", "")
        if child not in ("left_wrist", "right_wrist"):
            continue
        translation = transform.get("translation", {})
        rotation = transform.get("rotation", {})
        T_camera_hand = np.eye(4, dtype=np.float64)
        T_camera_hand[:3, :3] = quaternion_to_R(
            rotation.get("x", 0.0),
            rotation.get("y", 0.0),
            rotation.get("z", 0.0),
            rotation.get("w", 1.0),
        )
        T_camera_hand[:3, 3] = [
            translation.get("x", 0.0),
            translation.get("y", 0.0),
            translation.get("z", 0.0),
        ]
        wrist_world = (T_wc @ T_camera_hand)[:3, 3].astype(np.float32)
        if not np.all(np.isfinite(wrist_world)):
            continue
        if child == "left_wrist":
            left_wrist_world = wrist_world
        else:
            right_wrist_world = wrist_world
    return left_wrist_world, right_wrist_world


def load_hand_world_positions(
    tf_hands_path: Optional[Path],
    T_wc: np.ndarray,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Load left and right wrist anchor positions in world coordinates for one frame."""
    return load_hand_world_wrist_positions(tf_hands_path, T_wc)


def _depth_patch_median(depth_m: np.ndarray, x: int, y: int, radius: int = 2) -> Optional[float]:
    """Return the median valid depth around one pixel or None if unavailable."""
    x1 = max(0, x - radius)
    x2 = min(depth_m.shape[1], x + radius + 1)
    y1 = max(0, y - radius)
    y2 = min(depth_m.shape[0], y + radius + 1)
    patch = depth_m[y1:y2, x1:x2]
    valid = patch[(patch > 1e-4) & (patch < DEPTH_MAX)]
    if valid.size == 0:
        return None
    return float(np.median(valid))


def _pixel_to_world(x: float, y: float, depth: float, K_color: np.ndarray, T_wc: np.ndarray, T_cam_color: np.ndarray) -> np.ndarray:
    """Project a color pixel with metric depth into world coordinates."""
    fx, fy, cx, cy = K_color[0, 0], K_color[1, 1], K_color[0, 2], K_color[1, 2]
    point_color = np.array([(x - cx) * depth / fx, (y - cy) * depth / fy, depth, 1.0], dtype=np.float32)
    return (T_wc @ T_cam_color @ point_color)[:3].astype(np.float32)


def estimate_nearest_detected_object_distance(
    hand_world: Optional[np.ndarray],
    obj_dets: Optional[np.ndarray],
    depth_raw: np.ndarray,
    image_shape: Sequence[int],
    K_color: np.ndarray,
    K_depth: np.ndarray,
    T_depth_color: np.ndarray,
    T_wc: np.ndarray,
    T_cam_color: np.ndarray,
) -> Optional[float]:
    """Estimate the nearest 3D distance between one hand and the detected object boxes in a frame."""
    if hand_world is None or obj_dets is None or obj_dets.shape[0] == 0:
        return None

    image_h, image_w = int(image_shape[0]), int(image_shape[1])
    depth_m = align_depth_to_color(depth_raw, K_depth, T_depth_color, K_color, image_w, image_h)

    min_distance = None
    for row in obj_dets:
        if len(row) < 5 or row[4] < DETECTION_THRESH:
            continue
        x1, y1, x2, y2 = [int(round(v)) for v in row[:4]]
        x1 = int(np.clip(x1, 0, image_w - 1))
        y1 = int(np.clip(y1, 0, image_h - 1))
        x2 = int(np.clip(max(x2, x1 + 1), 1, image_w))
        y2 = int(np.clip(max(y2, y1 + 1), 1, image_h))
        xs = np.linspace(x1, x2 - 1, num=3)
        ys = np.linspace(y1, y2 - 1, num=3)
        for sx in xs:
            for sy in ys:
                depth = _depth_patch_median(depth_m, int(round(sx)), int(round(sy)))
                if depth is None:
                    continue
                point_world = _pixel_to_world(float(sx), float(sy), depth, K_color, T_wc, T_cam_color)
                distance = float(np.linalg.norm(point_world - hand_world))
                if min_distance is None or distance < min_distance:
                    min_distance = distance
    return min_distance


def hand_velocity(hand_observations: Sequence[Optional[np.ndarray]]) -> float:
    """Compute the average displacement magnitude for a sequence of hand positions."""
    displacements = [
        float(np.linalg.norm(hand_observations[idx + 1] - hand_observations[idx]))
        for idx in range(len(hand_observations) - 1)
        if hand_observations[idx] is not None and hand_observations[idx + 1] is not None
    ]
    if not displacements:
        return 0.0
    return float(sum(displacements) / len(displacements))


def hand_travel_distance(hand_observations: Sequence[Optional[np.ndarray]]) -> float:
    """Compute cumulative wrist travel across one observation sequence."""
    travel = 0.0
    prev = None
    for obs in hand_observations:
        if obs is None:
            continue
        current = np.asarray(obs, dtype=np.float32).reshape(-1)
        if current.size != 3 or not np.all(np.isfinite(current)):
            continue
        if prev is not None:
            travel += float(np.linalg.norm(current - prev))
        prev = current
    return float(travel)


def lostfound_contact_threshold(vel_prior: float, vel_post: float) -> int:
    """Return the LostFound contact threshold adjusted by velocity change."""
    return LOSTFOUND_BASE_THRESHOLD + LOSTFOUND_HIGH_THRESHOLD_BONUS * int(abs(vel_prior - vel_post) > HAND_VELOCITY_DIFF_THRESHOLD)


def infer_interaction_flags(samples: Sequence[dict]) -> None:
    """Infer per-frame interaction activity using a LostFound-style hand/contact state machine."""
    left_history: deque[Optional[np.ndarray]] = deque(maxlen=HAND_HISTORY_SIZE)
    right_history: deque[Optional[np.ndarray]] = deque(maxlen=HAND_HISTORY_SIZE)
    left_active = False
    right_active = False

    for idx, sample in enumerate(samples):
        future_context = list(samples[idx + 1 : idx + 1 + LOOKAHEAD_FRAMES])
        left_history.append(sample.get("left_hand_world"))
        right_history.append(sample.get("right_hand_world"))
        future_positive = sum(1 for future in future_context if future.get("hand_contact", False))

        for prefix, history, current_active in (
            ("left", left_history, left_active),
            ("right", right_history, right_active),
        ):
            hand_key = f"{prefix}_hand_world"
            dist_key = f"{prefix}_nearest_object_distance"
            hand_pos = sample.get(hand_key)
            dist = sample.get(dist_key)
            vel_prior = hand_velocity(list(history))
            vel_post = hand_velocity([future.get(hand_key) for future in future_context])
            threshold = lostfound_contact_threshold(vel_prior, vel_post)
            # Activation = contact + proximity ONLY (no wrist-travel / velocity /
            # dwell / future-far gate). The interaction starts the moment the RCNN
            # hand-object detector reports contact this frame AND the hand is within
            # HAND_DISTANCE_THRESHOLD of the prior object, so a grasp that rests
            # motionless on the object before moving it still starts on time
            # (fixes the cabinet's late start + bad inactive prior). Isolated
            # single-frame false contacts are rejected downstream by the radius-3
            # smoothing in infer_smoothed_contact_flags, not by a motion gate.
            proximity_contact = bool(
                sample.get("hand_contact")
                and hand_pos is not None
                and dist is not None
                and dist < HAND_DISTANCE_THRESHOLD
            )
            sample[f"{prefix}_proximity_contact"] = bool(proximity_contact)

            if not current_active:
                if proximity_contact:
                    current_active = True
                    sample[f"{prefix}_activation_mode"] = "proximity_contact"
                else:
                    sample[f"{prefix}_activation_mode"] = ""
            else:
                # End once the look-ahead no longer shows sustained contact.
                if hand_pos is not None:
                    if future_positive <= threshold:
                        current_active = False
                elif not sample.get("hand_contact") and future_positive == 0:
                    current_active = False
                sample[f"{prefix}_activation_mode"] = str(sample.get(f"{prefix}_activation_mode", ""))

            sample[f"{prefix}_active"] = current_active
            if prefix == "left":
                left_active = current_active
            else:
                right_active = current_active

        sample["proximity_contact"] = bool(
            sample.get("left_proximity_contact") or sample.get("right_proximity_contact")
        )
        sample["interaction_active"] = bool(sample.get("left_active") or sample.get("right_active"))

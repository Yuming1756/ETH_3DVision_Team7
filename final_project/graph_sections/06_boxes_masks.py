# ======================================================================
# graph.py section: 06_boxes_masks.py
# Box/crop/mask geometry, SAM3 prompt specs, mask confidence metadata
#
# NOTE: not a standalone module. Loaded by graph.py into a shared
# namespace via exec(); names from earlier sections are in scope here.
# ======================================================================
from __future__ import annotations
def box_xyxy_to_normalized_xywh(
    bbox: Sequence[float],
    image_shape: Sequence[int],
) -> List[float]:
    """Convert an absolute xyxy box into SAM3 normalized xywh format."""
    height, width = int(image_shape[0]), int(image_shape[1])
    x1, y1, x2, y2 = [float(v) for v in bbox]
    x1 = float(np.clip(x1, 0, max(width - 1, 0)))
    y1 = float(np.clip(y1, 0, max(height - 1, 0)))
    x2 = float(np.clip(x2, x1 + 1.0, max(width, 1)))
    y2 = float(np.clip(y2, y1 + 1.0, max(height, 1)))
    return [
        x1 / max(width, 1),
        y1 / max(height, 1),
        (x2 - x1) / max(width, 1),
        (y2 - y1) / max(height, 1),
    ]


def normalized_xywh_to_box_xyxy(
    bbox_xywh: Sequence[float],
    image_shape: Sequence[int],
) -> Tuple[int, int, int, int]:
    """Convert a SAM3 normalized xywh box back into absolute xyxy pixel coordinates."""
    height, width = int(image_shape[0]), int(image_shape[1])
    x, y, w, h = [float(v) for v in bbox_xywh]
    x1 = int(round(np.clip(x * width, 0, max(width - 1, 0))))
    y1 = int(round(np.clip(y * height, 0, max(height - 1, 0))))
    x2 = int(round(np.clip((x + w) * width, x1 + 1, max(width, 1))))
    y2 = int(round(np.clip((y + h) * height, y1 + 1, max(height, 1))))
    return x1, y1, x2, y2


def normalized_xyxy_to_box_xyxy(
    bbox_xyxy: Sequence[float],
    image_shape: Sequence[int],
) -> Tuple[int, int, int, int]:
    """Convert a normalized xyxy box into absolute xyxy pixel coordinates."""
    height, width = int(image_shape[0]), int(image_shape[1])
    x1, y1, x2, y2 = [float(v) for v in bbox_xyxy]
    x1 = int(round(np.clip(x1 * width, 0, max(width - 1, 0))))
    y1 = int(round(np.clip(y1 * height, 0, max(height - 1, 0))))
    x2 = int(round(np.clip(x2 * width, x1 + 1, max(width, 1))))
    y2 = int(round(np.clip(y2 * height, y1 + 1, max(height, 1))))
    return x1, y1, x2, y2


def normalized_xy_to_point_xy(
    point_xy: Sequence[float],
    image_shape: Sequence[int],
) -> Tuple[float, float]:
    """Convert a normalized [x, y] point into absolute pixel coordinates."""
    height, width = int(image_shape[0]), int(image_shape[1])
    x, y = [float(v) for v in point_xy]
    return (
        float(np.clip(x * width, 0.0, max(width - 1, 0))),
        float(np.clip(y * height, 0.0, max(height - 1, 0))),
    )


def expand_bbox_xyxy(
    bbox: Sequence[float],
    image_shape: Sequence[int],
    scale: float = 1.5,
    padding: int = 0,
) -> Tuple[int, int, int, int]:
    """Expand an xyxy box about its center."""
    h, w = int(image_shape[0]), int(image_shape[1])
    x1, y1, x2, y2 = [float(v) for v in bbox]
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    bw = max(1.0, (x2 - x1) * float(scale) + 2.0 * float(padding))
    bh = max(1.0, (y2 - y1) * float(scale) + 2.0 * float(padding))
    nx1 = int(np.clip(round(cx - 0.5 * bw), 0, max(w - 1, 0)))
    ny1 = int(np.clip(round(cy - 0.5 * bh), 0, max(h - 1, 0)))
    nx2 = int(np.clip(round(cx + 0.5 * bw), nx1 + 1, max(w, 1)))
    ny2 = int(np.clip(round(cy + 0.5 * bh), ny1 + 1, max(h, 1)))
    return nx1, ny1, nx2, ny2


def _box_area(bbox: Sequence[float]) -> float:
    """Return the area of an xyxy box."""
    x1, y1, x2, y2 = [float(v) for v in bbox]
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _intersect_boxes_xyxy(
    a: Sequence[float],
    b: Sequence[float],
) -> Optional[Tuple[int, int, int, int]]:
    """Return the xyxy intersection of two boxes, or None if they barely overlap."""
    ax1, ay1, ax2, ay2 = [float(v) for v in a]
    bx1, by1, bx2, by2 = [float(v) for v in b]
    x1 = max(ax1, bx1)
    y1 = max(ay1, by1)
    x2 = min(ax2, bx2)
    y2 = min(ay2, by2)
    if (x2 - x1) < 1.0 or (y2 - y1) < 1.0:
        return None
    return int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))


def _confident_detection_boxes(
    detections: Optional[np.ndarray],
    thresh: float = DETECTION_THRESH,
) -> List[Tuple[Tuple[int, int, int, int], float]]:
    """Return confident xyxy detection boxes with scores."""
    boxes: List[Tuple[Tuple[int, int, int, int], float]] = []
    if detections is None:
        return boxes
    for row in detections:
        if len(row) < 5 or float(row[4]) < float(thresh):
            continue
        x1, y1, x2, y2 = [int(round(v)) for v in row[:4]]
        if x2 <= x1 or y2 <= y1:
            continue
        boxes.append(((x1, y1, x2, y2), float(row[4])))
    return boxes


def _box_center(bbox: Sequence[float]) -> Tuple[float, float]:
    """Return the center of an xyxy box."""
    x1, y1, x2, y2 = [float(v) for v in bbox]
    return 0.5 * (x1 + x2), 0.5 * (y1 + y2)


def _point_to_box_distance(point_xy: Tuple[float, float], bbox: Sequence[float]) -> float:
    """Return the Euclidean distance from a point to the nearest point inside an xyxy box."""
    px, py = float(point_xy[0]), float(point_xy[1])
    x1, y1, x2, y2 = [float(v) for v in bbox]
    dx = max(x1 - px, 0.0, px - x2)
    dy = max(y1 - py, 0.0, py - y2)
    return float(np.hypot(dx, dy))


def _sanitize_normalized_xyxy_box(value: object) -> Optional[List[float]]:
    """Validate and clamp a normalized xyxy box in [0, 1]."""
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        x1, y1, x2, y2 = [float(v) for v in value]
    except Exception:
        return None
    if not np.all(np.isfinite([x1, y1, x2, y2])):
        return None
    x1 = float(np.clip(x1, 0.0, 1.0))
    y1 = float(np.clip(y1, 0.0, 1.0))
    x2 = float(np.clip(x2, x1 + 1e-3, 1.0))
    y2 = float(np.clip(y2, y1 + 1e-3, 1.0))
    return [x1, y1, x2, y2]


def _sanitize_normalized_xy_point(value: object) -> Optional[List[float]]:
    """Validate and clamp a normalized point in [0, 1]^2."""
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    try:
        x, y = [float(v) for v in value]
    except Exception:
        return None
    if not np.all(np.isfinite([x, y])):
        return None
    return [float(np.clip(x, 0.0, 1.0)), float(np.clip(y, 0.0, 1.0))]


def crop_relative_normalized_xyxy_to_full_box(
    bbox_xyxy: Sequence[float],
    crop_box: Sequence[int],
    image_shape: Sequence[int],
) -> Tuple[int, int, int, int]:
    """Map a normalized xyxy box inside a crop back to full-frame xyxy coordinates."""
    crop_x1, crop_y1, crop_x2, crop_y2 = [int(v) for v in crop_box]
    crop_h = max(1, crop_y2 - crop_y1)
    crop_w = max(1, crop_x2 - crop_x1)
    local_box = normalized_xyxy_to_box_xyxy(bbox_xyxy, (crop_h, crop_w))
    full_box = (
        crop_x1 + local_box[0],
        crop_y1 + local_box[1],
        crop_x1 + local_box[2],
        crop_y1 + local_box[3],
    )
    return expand_bbox_xyxy(full_box, image_shape, scale=1.0, padding=0)


def box_xyxy_to_crop_normalized_xywh(
    bbox: Sequence[float],
    crop_box: Sequence[int],
) -> List[float]:
    """Convert a full-frame xyxy box into crop-relative normalized xywh coordinates."""
    crop_x1, crop_y1, crop_x2, crop_y2 = [int(v) for v in crop_box]
    crop_h = max(1, crop_y2 - crop_y1)
    crop_w = max(1, crop_x2 - crop_x1)
    local_box = (
        float(bbox[0]) - float(crop_x1),
        float(bbox[1]) - float(crop_y1),
        float(bbox[2]) - float(crop_x1),
        float(bbox[3]) - float(crop_y1),
    )
    return box_xyxy_to_normalized_xywh(local_box, (crop_h, crop_w))


def crop_rgb_sequence(
    rgb_list: Sequence[np.ndarray],
    crop_box: Sequence[int],
) -> List[np.ndarray]:
    """Crop one RGB frame sequence with a shared xyxy box."""
    x1, y1, x2, y2 = [int(v) for v in crop_box]
    cropped: List[np.ndarray] = []
    for rgb in rgb_list:
        crop = np.asarray(rgb, dtype=np.uint8)[y1:y2, x1:x2]
        if crop.size == 0:
            crop = np.asarray(rgb, dtype=np.uint8)
        cropped.append(crop.copy())
    return cropped


def uncrop_mask_sequence(
    masks: Sequence[np.ndarray],
    crop_box: Sequence[int],
    full_image_shape: Sequence[int],
) -> List[np.ndarray]:
    """Paste crop-local masks back into full-frame masks."""
    full_h, full_w = int(full_image_shape[0]), int(full_image_shape[1])
    x1, y1, x2, y2 = [int(v) for v in crop_box]
    output: List[np.ndarray] = []
    for mask in masks:
        canvas = np.zeros((full_h, full_w), dtype=bool)
        local_mask = np.asarray(mask, dtype=bool)
        if local_mask.shape != (max(1, y2 - y1), max(1, x2 - x1)):
            local_mask = cv2.resize(
                local_mask.astype(np.uint8),
                (max(1, x2 - x1), max(1, y2 - y1)),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
        canvas[y1:y2, x1:x2] = local_mask
        output.append(canvas)
    return output


def build_stabilized_crop_boxes(
    element_masks: Sequence[np.ndarray],
    crop_w: int,
    crop_h: int,
    image_shape: Sequence[int],
) -> List[Tuple[int, int, int, int]]:
    """Per-frame fixed-size crop boxes that TRACK the per-frame element (handle)
    center. Cropping every frame to a window centered on the handle stabilizes the
    moving part in the middle of the crop, so SAM3 only ever segments the part
    around the handle — excluding far static regions (the drawer's lower band) and
    other instances (the wrong cabinet door). Carries the last known center over
    frames where the handle mask is empty."""
    h, w = int(image_shape[0]), int(image_shape[1])
    cw, ch = max(1, min(int(crop_w), w)), max(1, min(int(crop_h), h))
    boxes: List[Tuple[int, int, int, int]] = []
    last_center = (w / 2.0, h / 2.0)
    for em in element_masks:
        e = np.asarray(em, dtype=bool)
        if np.any(e):
            bb = mask_to_bbox(e)
            if bb is not None:
                last_center = _box_center(bb)
        cx, cy = last_center
        x1 = int(np.clip(round(cx - cw / 2.0), 0, max(w - cw, 0)))
        y1 = int(np.clip(round(cy - ch / 2.0), 0, max(h - ch, 0)))
        boxes.append((x1, y1, x1 + cw, y1 + ch))
    return boxes


def crop_rgb_sequence_per_frame(
    rgb_list: Sequence[np.ndarray],
    crop_boxes: Sequence[Sequence[int]],
) -> List[np.ndarray]:
    """Crop each RGB frame with its OWN xyxy box (all boxes share a size)."""
    out: List[np.ndarray] = []
    for rgb, box in zip(rgb_list, crop_boxes):
        x1, y1, x2, y2 = [int(v) for v in box]
        crop = np.asarray(rgb, dtype=np.uint8)[y1:y2, x1:x2]
        out.append(crop.copy() if crop.size else np.asarray(rgb, dtype=np.uint8).copy())
    return out


def uncrop_mask_sequence_per_frame(
    masks: Sequence[np.ndarray],
    crop_boxes: Sequence[Sequence[int]],
    image_shape: Sequence[int],
) -> List[np.ndarray]:
    """Paste each per-frame crop-local mask back at its own crop offset."""
    full_h, full_w = int(image_shape[0]), int(image_shape[1])
    out: List[np.ndarray] = []
    for mask, box in zip(masks, crop_boxes):
        x1, y1, x2, y2 = [int(v) for v in box]
        canvas = np.zeros((full_h, full_w), dtype=bool)
        local = np.asarray(mask, dtype=bool)
        th, tw = max(1, y2 - y1), max(1, x2 - x1)
        if local.shape != (th, tw):
            local = cv2.resize(local.astype(np.uint8), (tw, th), interpolation=cv2.INTER_NEAREST).astype(bool)
        canvas[y1:y2, x1:x2] = local
        out.append(canvas)
    return out


def build_object_centric_crop_box(
    element_bbox: Sequence[int],
    object_bbox: Optional[Sequence[int]],
    image_shape: Sequence[int],
) -> Tuple[int, int, int, int]:
    """Create a local crop around the touched part and its parent-object candidate."""
    base_element = tuple(int(v) for v in element_bbox)
    if object_bbox is None:
        return expand_bbox_xyxy(base_element, image_shape, scale=4.0, padding=32)

    image_h, image_w = int(image_shape[0]), int(image_shape[1])
    image_area = float(max(1, image_h * image_w))
    object_area = _box_area(object_bbox)
    element_area = max(_box_area(base_element), 1.0)
    if object_area > 0.45 * image_area or object_area > 10.0 * element_area:
        union_box = expand_bbox_xyxy(base_element, image_shape, scale=4.0, padding=32)
    else:
        union_box = (
            min(int(base_element[0]), int(object_bbox[0])),
            min(int(base_element[1]), int(object_bbox[1])),
            max(int(base_element[2]), int(object_bbox[2])),
            max(int(base_element[3]), int(object_bbox[3])),
        )
    return expand_bbox_xyxy(union_box, image_shape, scale=1.6, padding=24)


def select_interaction_anchor_boxes(
    sample: dict,
    image_shape: Sequence[int],
) -> Tuple[Tuple[int, int, int, int], Optional[Tuple[int, int, int, int]]]:
    """Pick a hand-local interaction box and the nearest detected object box on the first active frame."""
    h, w = int(image_shape[0]), int(image_shape[1])
    hand_boxes = _confident_detection_boxes(sample.get("hand_dets"))
    object_boxes = _confident_detection_boxes(sample.get("obj_dets"))
    if hand_boxes and object_boxes:
        best_pair = None
        best_distance = None
        for hand_box, _hand_score in hand_boxes:
            hand_center = _box_center(hand_box)
            for object_box, obj_score in object_boxes:
                distance = _point_to_box_distance(hand_center, object_box)
                score = (distance, -obj_score)
                if best_distance is None or score < best_distance:
                    best_distance = score
                    best_pair = (hand_box, object_box)
        if best_pair is not None:
            hand_box, object_box = best_pair
            union_box = (
                min(hand_box[0], object_box[0]),
                min(hand_box[1], object_box[1]),
                max(hand_box[2], object_box[2]),
                max(hand_box[3], object_box[3]),
            )
            return expand_bbox_xyxy(union_box, (h, w), scale=1.1, padding=12), object_box
    if object_boxes:
        object_box = max(object_boxes, key=lambda item: item[1])[0]
        return expand_bbox_xyxy(object_box, (h, w), scale=1.1, padding=12), object_box
    return detect_interaction_bbox(sample, image_shape), None


def select_primary_hand_anchor_point(
    sample: dict,
    image_shape: Sequence[int],
) -> Tuple[float, float]:
    """Return the center of the hand box most relevant to the current interaction."""
    h, w = int(image_shape[0]), int(image_shape[1])
    hand_boxes = _confident_detection_boxes(sample.get("hand_dets"))
    object_boxes = _confident_detection_boxes(sample.get("obj_dets"))
    if hand_boxes and object_boxes:
        best_hand = None
        best_score = None
        for hand_box, _hand_score in hand_boxes:
            hand_center = _box_center(hand_box)
            nearest_distance = min(_point_to_box_distance(hand_center, object_box) for object_box, _score in object_boxes)
            score = (nearest_distance, -_box_area(hand_box))
            if best_score is None or score < best_score:
                best_score = score
                best_hand = hand_box
        if best_hand is not None:
            return _box_center(best_hand)
    if hand_boxes:
        return _box_center(max(hand_boxes, key=lambda item: item[1])[0])
    if object_boxes:
        return _box_center(max(object_boxes, key=lambda item: item[1])[0])
    return 0.5 * w, 0.5 * h


def choose_best_sam_seed_frame(
    samples: Sequence[dict],
    *,
    preferred_hand: Optional[str] = None,
    prefer_contact: bool = False,
    prefer_early: bool = False,
) -> int:
    """Pick the best clip frame to seed SAM3 propagation from.

    We keep the propagation behavior from the last commit (single prompt frame,
    then propagate both directions across the whole clip), but replace the old
    fixed "first/last valid frame" choice with a scored selection:

    - active clips prefer confident hand+object contact frames,
    - inactive clips naturally fall back to object visibility + recency,
    - both use the same logic.
    """
    valid_indices = [
        idx for idx, sample in enumerate(samples)
        if bool(sample.get("rgb_valid", True))
    ]
    if not valid_indices:
        return 0

    preferred_hand = str(preferred_hand or "").strip().lower()
    n = max(len(samples) - 1, 1)
    if prefer_early:
        # Seed at the EARLIEST clear frame (hand just on the object, before it moves):
        # first valid frame with hand contact or a confident detection; else earliest.
        for idx in valid_indices:
            sample = samples[idx]
            hand_contact = bool(sample.get("hand_contact", False))
            if preferred_hand in {"left", "right"}:
                hand_contact = hand_contact or bool(sample.get(f"{preferred_hand}_active", False))
            object_boxes = _confident_detection_boxes(sample.get("obj_dets"))
            obj_score = max((float(s) for _b, s in object_boxes), default=0.0)
            if hand_contact or obj_score > 0.0:
                return int(idx)
        return int(valid_indices[0])
    best_idx = valid_indices[-1]
    best_score = None

    for idx in valid_indices:
        sample = samples[idx]
        object_boxes = _confident_detection_boxes(sample.get("obj_dets"))
        hand_boxes = _confident_detection_boxes(sample.get("hand_dets"))
        obj_score = max((float(score) for _box, score in object_boxes), default=0.0)
        hand_score = max((float(score) for _box, score in hand_boxes), default=0.0)
        hand_contact = bool(sample.get("hand_contact", False))
        if preferred_hand in {"left", "right"}:
            hand_contact = hand_contact or bool(sample.get(f"{preferred_hand}_active", False))
        nearest_dist = None
        dist_keys: List[str] = []
        if preferred_hand in {"left", "right"}:
            dist_keys.append(f"{preferred_hand}_nearest_object_distance")
        dist_keys.extend(["left_nearest_object_distance", "right_nearest_object_distance"])
        seen_keys = set()
        for key in dist_keys:
            if key in seen_keys:
                continue
            seen_keys.add(key)
            value = sample.get(key)
            if value is None:
                continue
            try:
                value_f = float(value)
            except Exception:
                continue
            if np.isfinite(value_f):
                nearest_dist = value_f if nearest_dist is None else min(nearest_dist, value_f)
        dist_bonus = 0.0
        if nearest_dist is not None:
            dist_bonus = max(0.0, 1.0 - nearest_dist / max(float(HAND_DISTANCE_THRESHOLD), 1e-6))
        contact_bonus = 2.0 if hand_contact else 0.0
        if prefer_contact and hand_contact:
            contact_bonus += 2.0
        # Tiebreak by recency by default; for active clips prefer EARLY frames where
        # the touched object is clear (door closed, not yet swung over a neighbour).
        frame_bonus = (1.0 - float(idx) / float(n)) if prefer_early else (float(idx) / float(n))
        score = (
            float(obj_score + 0.35 * hand_score + contact_bonus + 0.5 * dist_bonus),
            frame_bonus,
        )
        if best_score is None or score > best_score:
            best_score = score
            best_idx = int(idx)
    return int(best_idx)


def choose_boundary_inactive_seed_frame(
    samples: Sequence[dict],
    *,
    preferred_hand: Optional[str] = None,
    tail_frames: int = 12,
) -> int:
    """Pick a grasp-near seed from the tail of the inactive clip.

    This is used for late-start revolute clips: even if the active segment only
    begins once the door is already partly open, the best boundary seed often
    lives in the last few inactive frames where the hand is already on the
    handle but the object is still in the closed/rest pose.
    """
    valid_indices = [
        idx for idx, sample in enumerate(samples)
        if bool(sample.get("rgb_valid", True))
    ]
    if not valid_indices:
        return 0
    tail_count = max(1, int(tail_frames))
    candidate_indices = valid_indices[-tail_count:]
    preferred_hand = str(preferred_hand or "").strip().lower()
    n = max(len(samples) - 1, 1)
    best_idx = candidate_indices[-1]
    best_score = None

    for idx in candidate_indices:
        sample = samples[idx]
        object_boxes = _confident_detection_boxes(sample.get("obj_dets"))
        hand_boxes = _confident_detection_boxes(sample.get("hand_dets"))
        obj_score = max((float(score) for _box, score in object_boxes), default=0.0)
        hand_score = max((float(score) for _box, score in hand_boxes), default=0.0)
        hand_contact = bool(sample.get("hand_contact", False))
        nearest_dist = None
        dist_keys: List[str] = []
        if preferred_hand in {"left", "right"}:
            dist_keys.append(f"{preferred_hand}_nearest_object_distance")
        dist_keys.extend(["left_nearest_object_distance", "right_nearest_object_distance"])
        seen_keys = set()
        for key in dist_keys:
            if key in seen_keys:
                continue
            seen_keys.add(key)
            value = sample.get(key)
            if value is None:
                continue
            try:
                value_f = float(value)
            except Exception:
                continue
            if np.isfinite(value_f):
                nearest_dist = value_f if nearest_dist is None else min(nearest_dist, value_f)
        dist_bonus = 0.0
        if nearest_dist is not None:
            dist_bonus = max(0.0, 1.0 - nearest_dist / max(float(HAND_DISTANCE_THRESHOLD), 1e-6))
        contact_bonus = 2.0 if hand_contact else 0.0
        recency_bonus = float(idx) / float(n)
        score = (
            float(obj_score + 0.35 * hand_score + contact_bonus + 0.5 * dist_bonus),
            recency_bonus,
        )
        if best_score is None or score > best_score:
            best_idx = idx
            best_score = score
    return int(best_idx)


def mask_to_bbox(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """Return the tight bounding box of a binary mask or None if empty."""
    mask_bool = np.asarray(mask, dtype=bool)
    ys, xs = np.where(mask_bool)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def select_object_anchor_box(
    sample: dict,
    image_shape: Sequence[int],
    element_mask: Optional[np.ndarray],
    fallback_object_box: Optional[Sequence[float]] = None,
    hint_object_box: Optional[Sequence[float]] = None,
) -> Tuple[int, int, int, int]:
    """Choose the specific parent-object instance box that corresponds to the touched element."""
    if element_mask is not None and np.any(element_mask):
        element_bbox = mask_to_bbox(np.asarray(element_mask, dtype=bool))
    else:
        element_bbox = None
    object_boxes = _confident_detection_boxes(sample.get("obj_dets"))

    # Preferred: anchor the object box on the interactive element (handle),
    # expanded so it covers the part of the moving body around the handle. The
    # element is physically ON the moving part, so this is a far more reliable
    # spatial anchor than (a) a raw object-detector box — for a kitchen unit that
    # is the WHOLE cabinet/drawer-front row, which makes SAM3 fill the entire
    # band — or (b) the full-frame fallback, which makes SAM3 lock onto the hand.
    # When a confident object detection actually contains the element, we clip the
    # expanded box to that detection so we never spill past the real object.
    if element_bbox is not None and hint_object_box is None:
        element_anchored = expand_bbox_xyxy(
            element_bbox,
            image_shape,
            scale=OBJECT_ANCHOR_FROM_ELEMENT_SCALE,
            padding=OBJECT_ANCHOR_FROM_ELEMENT_PADDING,
        )
        element_center = _box_center(element_bbox)
        containing = [box for box, _score in object_boxes if _point_to_box_distance(element_center, box) <= 1e-3]
        if containing:
            host = min(containing, key=_box_area)
            clipped = _intersect_boxes_xyxy(element_anchored, host)
            if clipped is not None:
                return clipped
        return element_anchored

    if hint_object_box is not None and object_boxes:
        hint_center = _box_center(hint_object_box)
        overlapping = [
            (box, score)
            for box, score in object_boxes
            if _point_to_box_distance(hint_center, box) <= 1e-3
        ]
        if overlapping:
            return min(
                overlapping,
                key=lambda item: (_box_area(item[0]), -item[1]),
            )[0]
        return min(
            object_boxes,
            key=lambda item: (
                _point_to_box_distance(hint_center, item[0]),
                _box_area(item[0]),
                -item[1],
            ),
        )[0]
    if element_bbox is not None and object_boxes:
        center = _box_center(element_bbox)
        containing = [(box, score) for box, score in object_boxes if _point_to_box_distance(center, box) <= 1e-3]
        if containing:
            return min(
                containing,
                key=lambda item: (_box_area(item[0]), -item[1]),
            )[0]
        nearest = min(
            object_boxes,
            key=lambda item: (
                _point_to_box_distance(center, item[0]),
                _box_area(item[0]),
                -item[1],
            ),
        )[0]
        return nearest
    if hint_object_box is not None:
        return tuple(int(v) for v in hint_object_box)
    if fallback_object_box is not None:
        return tuple(int(v) for v in fallback_object_box)
    if element_bbox is not None:
        return expand_bbox_xyxy(element_bbox, image_shape, scale=4.0, padding=24)
    return detect_interaction_bbox(sample, image_shape)


def keep_mask_component_nearest_point(
    mask: np.ndarray,
    point_xy: Tuple[float, float],
    bbox_distance_tolerance_px: float = 64.0,
    min_relative_area: float = 0.12,
) -> np.ndarray:
    """Keep the anchor-near connected component while resisting tiny sliver fragments.

    A pure centroid-distance rule is too brittle for articulated masks: when
    SAM returns a large, fragmented door/drawer mask plus a tiny shard near the
    handle, the tiny shard wins and ArtiPoint later receives an unusably small
    query mask. We therefore:
      1. rank components by point-to-bbox distance first,
      2. keep all components within a small tolerance of the best bbox distance,
      3. prefer components that are not tiny relative to the largest eligible one.
    """
    candidate = np.asarray(mask, dtype=np.uint8)
    if candidate.sum() == 0:
        return candidate.astype(bool)
    num_labels, labels, _stats, centroids = cv2.connectedComponentsWithStats(candidate, connectivity=8)
    if num_labels <= 2:
        return candidate.astype(bool)
    anchor = np.asarray(point_xy, dtype=np.float32)
    components = []
    for label_idx in range(1, num_labels):
        component_mask = labels == label_idx
        if not np.any(component_mask):
            continue
        centroid = np.asarray(centroids[label_idx], dtype=np.float32)
        area = int(component_mask.sum())
        bbox = mask_to_bbox(component_mask)
        if bbox is None:
            continue
        x1, y1, x2, y2 = [float(v) for v in bbox]
        dx = max(x1 - float(anchor[0]), 0.0, float(anchor[0]) - x2)
        dy = max(y1 - float(anchor[1]), 0.0, float(anchor[1]) - y2)
        bbox_dist = float(np.hypot(dx, dy))
        centroid_dist = float(np.linalg.norm(centroid - anchor))
        components.append(
            {
                "label": int(label_idx),
                "area": area,
                "bbox_dist": bbox_dist,
                "centroid_dist": centroid_dist,
            }
        )
    if not components:
        return candidate.astype(bool)
    best_bbox_dist = min(float(comp["bbox_dist"]) for comp in components)
    tol = max(0.0, float(bbox_distance_tolerance_px))
    eligible = [comp for comp in components if float(comp["bbox_dist"]) <= best_bbox_dist + tol]
    if not eligible:
        eligible = components
    largest_eligible_area = max(int(comp["area"]) for comp in eligible)
    rel_floor = max(16, int(round(largest_eligible_area * max(0.0, float(min_relative_area)))))
    sizable = [comp for comp in eligible if int(comp["area"]) >= rel_floor]
    if sizable:
        eligible = sizable
    best_label = None
    best_score = None
    for comp in eligible:
        score = (float(comp["bbox_dist"]), -int(comp["area"]), float(comp["centroid_dist"]))
        if best_score is None or score < best_score:
            best_score = score
            best_label = int(comp["label"])
    if best_label is None:
        return candidate.astype(bool)
    return (labels == best_label)


def choose_best_object_prompt_frame(
    samples: Sequence[dict],
    element_masks: Sequence[np.ndarray],
    *,
    fallback_index: int = 0,
    prefer_early: bool = False,
) -> int:
    """Choose the best frame to seed object SAM3 from the propagated element mask.

    For both active and inactive clips we want the same rule:
    pick a frame where the element mask is present and not obviously bloated,
    then prefer stronger detections and slightly later frames.
    """
    valid_indices = [
        idx
        for idx, mask in enumerate(element_masks)
        if idx < len(samples)
        and bool(samples[idx].get("rgb_valid", True))
        and np.any(np.asarray(mask, dtype=bool))
    ]
    if not valid_indices:
        return int(fallback_index)

    elem_areas = {
        idx: int(np.count_nonzero(np.asarray(element_masks[idx], dtype=bool)))
        for idx in valid_indices
    }
    median_area = float(np.median([elem_areas[idx] for idx in valid_indices])) if valid_indices else 0.0
    clean_indices = [
        idx
        for idx in valid_indices
        if elem_areas[idx] >= 64
        and elem_areas[idx] <= max(2.0 * median_area, 64.0)
    ]
    pool = clean_indices if clean_indices else valid_indices

    n = max(len(samples) - 1, 1)
    if prefer_early:
        # Seed at the earliest frame where the object is sufficiently VISIBLE, not just
        # the earliest detected. A door is strongly detected even when closed -> seeds
        # early (avoiding the swing over a near-identical neighbour); a drawer is only
        # weakly detected closed -> it waits until pulled open and clearly visible.
        # Gate on the detection being a real fraction of the clip's best object score.
        pool_obj_scores = {
            idx: max(
                (float(score) for _box, score in _confident_detection_boxes(samples[idx].get("obj_dets"))),
                default=0.0,
            )
            for idx in pool
        }
        max_obj_score = max(pool_obj_scores.values(), default=0.0)
        visibility_floor = 0.6 * max_obj_score
        if max_obj_score > 0.0:
            for idx in pool:
                if pool_obj_scores[idx] >= visibility_floor:
                    return int(idx)
        # No usable object detection anywhere: fall back to the earliest hand-contact
        # frame, else the earliest pooled frame.
        for idx in pool:
            if bool(samples[idx].get("hand_contact", False)):
                return int(idx)
        return int(pool[0])
    best_idx = int(pool[-1])
    best_score = None
    for idx in pool:
        sample = samples[idx]
        object_boxes = _confident_detection_boxes(sample.get("obj_dets"))
        obj_score = max((float(score) for _box, score in object_boxes), default=0.0)
        hand_contact = bool(sample.get("hand_contact", False))
        recency_bonus = float(idx) / float(n)
        score = (
            float(obj_score + (0.5 if hand_contact else 0.0)),
            recency_bonus,
        )
        if best_score is None or score > best_score:
            best_score = score
            best_idx = int(idx)
    return int(best_idx)


def stabilize_mask_sequence_near_anchor(
    masks: Sequence[np.ndarray],
    anchor_point_xy: Tuple[float, float],
) -> List[np.ndarray]:
    """Track one connected component across time by repeatedly selecting the nearest component."""
    stabilized: List[np.ndarray] = []
    reference_point = (float(anchor_point_xy[0]), float(anchor_point_xy[1]))
    for mask in masks:
        filtered = keep_mask_component_nearest_point(mask, reference_point)
        stabilized.append(np.asarray(filtered, dtype=bool))
        bbox = mask_to_bbox(filtered)
        if bbox is not None:
            reference_point = _box_center(bbox)
    return stabilized


def lightly_shrink_mask_sequence(
    masks: Sequence[np.ndarray],
    kernel_size: int = 3,
    iterations: int = 1,
) -> List[np.ndarray]:
    """Lightly erode each mask while falling back when thin silhouettes would disappear."""
    kernel_size = max(1, int(kernel_size))
    if kernel_size % 2 == 0:
        kernel_size += 1
    iterations = max(1, int(iterations))
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    shrunk: List[np.ndarray] = []
    for mask in masks:
        mask_bool = np.asarray(mask, dtype=bool)
        if not np.any(mask_bool):
            shrunk.append(mask_bool.copy())
            continue
        eroded = cv2.erode(mask_bool.astype(np.uint8), kernel, iterations=iterations).astype(bool)
        shrunk.append(eroded if np.any(eroded) else mask_bool.copy())
    return shrunk


def constrain_part_masks_inside_object(
    part_masks: Sequence[np.ndarray],
    object_masks: Sequence[np.ndarray],
    anchor_point_xy: Optional[Tuple[float, float]] = None,
) -> List[np.ndarray]:
    """Keep the part mask inside the parent-object mask and retain one anchor-nearest component."""
    constrained_masks: List[np.ndarray] = []
    reference_point = (
        (float(anchor_point_xy[0]), float(anchor_point_xy[1]))
        if anchor_point_xy is not None
        else None
    )
    for part_mask, object_mask in zip(part_masks, object_masks):
        part = np.asarray(part_mask, dtype=bool)
        obj = np.asarray(object_mask, dtype=bool)
        if np.any(part) and np.any(obj):
            candidate = (part & obj)
        elif np.any(part) and not np.any(obj):
            candidate = part.copy()
        else:
            candidate = np.zeros_like(part, dtype=bool)
        if reference_point is None:
            bbox = mask_to_bbox(candidate)
            if bbox is None and np.any(obj):
                bbox = mask_to_bbox(obj)
            if bbox is not None:
                reference_point = _box_center(bbox)
        if reference_point is not None and np.any(candidate):
            candidate = keep_mask_component_nearest_point(candidate, reference_point)
            bbox = mask_to_bbox(candidate)
            if bbox is not None:
                reference_point = _box_center(bbox)
        constrained_masks.append(np.asarray(candidate, dtype=bool))
    return constrained_masks


def expand_bbox_by_fraction(
    bbox: Sequence[int],
    image_shape: Sequence[int],
    margin_fraction: float,
) -> Tuple[int, int, int, int]:
    """Expand an absolute xyxy bbox by a fraction of its own size."""
    height, width = int(image_shape[0]), int(image_shape[1])
    x1, y1, x2, y2 = [float(v) for v in bbox]
    margin_x = max(1.0, (x2 - x1) * float(margin_fraction))
    margin_y = max(1.0, (y2 - y1) * float(margin_fraction))
    return (
        int(round(np.clip(x1 - margin_x, 0, max(width - 1, 0)))),
        int(round(np.clip(y1 - margin_y, 0, max(height - 1, 0)))),
        int(round(np.clip(x2 + margin_x, 1, max(width, 1)))),
        int(round(np.clip(y2 + margin_y, 1, max(height, 1)))),
    )


def select_bootstrap_box_from_masks(
    masks: Sequence[np.ndarray],
    preferred_frame_idx: int,
    image_shape: Sequence[int],
) -> Tuple[Optional[int], Optional[Tuple[int, int, int, int]]]:
    """Choose a SAM-derived prompt box from the best available first-pass mask."""
    n_frames = len(masks)
    if n_frames == 0:
        return None, None
    preferred_frame_idx = int(np.clip(preferred_frame_idx, 0, n_frames - 1))
    frame_order = [preferred_frame_idx]
    frame_order.extend(idx for idx in range(n_frames) if idx != preferred_frame_idx)
    best: Optional[Tuple[int, Tuple[int, int, int, int], int]] = None
    for idx in frame_order:
        mask = np.asarray(masks[idx], dtype=bool)
        area = int(mask.sum())
        if area < SAM3_BOOTSTRAP_BOX_MIN_PIXELS:
            continue
        bbox = mask_to_bbox(mask)
        if bbox is None:
            continue
        # Prefer the requested prompt frame, but avoid choosing tiny accidental blobs.
        score = (2 if idx == preferred_frame_idx else 1) * area
        if best is None or score > best[0]:
            best = (score, bbox, idx)
    if best is None:
        return None, None
    return int(best[2]), expand_bbox_by_fraction(best[1], image_shape, SAM3_BOOTSTRAP_BOX_MARGIN)


def connected_component_quality(mask: np.ndarray) -> float:
    """Score whether a mask is dominated by one connected component."""
    mask_uint8 = np.asarray(mask, dtype=bool).astype(np.uint8)
    area = int(mask_uint8.sum())
    if area <= 0:
        return 0.0
    num_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(mask_uint8, connectivity=8)
    if num_labels <= 1:
        return 0.0
    largest = int(stats[1:, cv2.CC_STAT_AREA].max()) if num_labels > 1 else 0
    return float(np.clip(largest / max(area, 1), 0.0, 1.0))


def compute_sam3_mask_confidence_metadata(
    masks: Sequence[np.ndarray],
    probs: Sequence[float],
    image_shape: Sequence[int],
    native_boxes: Optional[Sequence[Optional[Sequence[float]]]] = None,
) -> List[dict]:
    """Fuse SAM3 probability with temporal mask sanity checks into per-frame confidence."""
    height, width = int(image_shape[0]), int(image_shape[1])
    diag = float(np.hypot(max(width, 1), max(height, 1)))
    areas = np.asarray([int(np.asarray(mask, dtype=bool).sum()) for mask in masks], dtype=np.float32)
    positive_areas = areas[areas > 0]
    median_area = float(np.median(positive_areas)) if len(positive_areas) else 0.0
    bboxes: List[Optional[Tuple[int, int, int, int]]] = [mask_to_bbox(np.asarray(mask, dtype=bool)) for mask in masks]
    centers = []
    for bbox in bboxes:
        if bbox is None:
            centers.append(None)
        else:
            centers.append(np.asarray(_box_center(bbox), dtype=np.float32))
    valid_centers = np.stack([center for center in centers if center is not None], axis=0) if any(center is not None for center in centers) else None
    median_center = np.median(valid_centers, axis=0) if valid_centers is not None else None
    native_boxes = list(native_boxes or [])

    metadata: List[dict] = []
    for idx, mask in enumerate(masks):
        area = float(areas[idx])
        raw_prob = float(probs[idx]) if idx < len(probs) and np.isfinite(float(probs[idx])) else (0.5 if area > 0 else 0.0)
        raw_prob = float(np.clip(raw_prob, 0.0, 1.0))
        if area <= 0 or median_area <= 0:
            area_score = 0.0
        else:
            ratio = max(area / median_area, median_area / max(area, 1.0))
            area_score = float(np.clip(1.0 - (ratio - 1.0) / max(SAM3_AREA_OUTLIER_RATIO - 1.0, 1e-6), 0.0, 1.0))
        bbox = bboxes[idx]
        center = centers[idx]
        if center is None or median_center is None:
            box_score = 0.0
        else:
            dist = float(np.linalg.norm(center - median_center))
            box_score = float(np.clip(1.0 - dist / max(diag * SAM3_BOX_CENTER_OUTLIER_RATIO, 1e-6), 0.0, 1.0))
        component_score = connected_component_quality(mask)
        # Confidence is now SAM3's genuine per-frame propagation score alone
        # (out_sam2_probs, captured upstream as raw_prob). The previous blend with
        # mask-consistency heuristics (area/box/component) was a PSEUDO confidence
        # that (a) punished the legitimately moving active object and (b) papered
        # over the fact that the raw prob we were reading (out_probs) was a
        # constant first-frame detection score. With a real per-frame score we
        # trust it directly; area/box/component are still recorded for inspection
        # but no longer alter the confidence.
        confidence = float(raw_prob)
        native_box = native_boxes[idx] if idx < len(native_boxes) else None
        native_box_xywh = None
        if native_box is not None:
            native_box_arr = np.asarray(native_box, dtype=np.float32).reshape(-1)
            if native_box_arr.shape[0] >= 4 and np.all(np.isfinite(native_box_arr[:4])):
                native_box_xywh = [round(float(v), 4) for v in native_box_arr[:4]]
        metadata.append(
            {
                "frame_index": int(idx),
                "sam3_prob": round(raw_prob, 4),
                "sam3_box_xywh": native_box_xywh,
                "area_pixels": int(area),
                "bbox_xyxy": [int(v) for v in bbox] if bbox is not None else None,
                "area_consistency": round(area_score, 4),
                "box_consistency": round(box_score, 4),
                "component_consistency": round(component_score, 4),
                "confidence": round(confidence, 4),
                "valid_for_geometry": bool(area > 0 and confidence >= SAM3_CONFIDENCE_MIN),
            }
        )
    return metadata


def sam3_geometry_valid_flags(metadata: Sequence[dict], base_valids: Sequence[bool]) -> List[bool]:
    """Return frame-valid flags that reject low-confidence SAM3 mask outliers."""
    flags: List[bool] = []
    for idx, item in enumerate(metadata):
        base_valid = bool(base_valids[idx]) if idx < len(base_valids) else True
        flags.append(base_valid and bool(item.get("valid_for_geometry", False)))
    if sum(flags) >= SAM3_CONFIDENCE_MIN_VALID_FRAMES:
        return flags
    candidates = []
    for idx, item in enumerate(metadata):
        base_valid = bool(base_valids[idx]) if idx < len(base_valids) else True
        if base_valid and int(item.get("area_pixels", 0)) > 0:
            candidates.append((float(item.get("confidence", 0.0)), idx))
    if not candidates:
        return flags
    candidates.sort(reverse=True)
    for _score, idx in candidates[:SAM3_CONFIDENCE_MIN_VALID_FRAMES]:
        flags[idx] = True
    return flags


def save_sam3_confidence_json(path: Path, payload: dict) -> None:
    """Persist SAM3 confidence diagnostics for mask debugging."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _interaction_target_phrase(target_label: str, interactive_element: Optional[str], role: str) -> str:
    """Return a more explicit manipulated-body phrase for SAM prompts."""
    target = str(target_label or '').strip().lower()
    element = str(interactive_element or '').strip().lower()
    role = str(role or 'object').strip().lower()
    if not target:
        return ''
    if role == 'object' and element and element != target:
        if target == 'drawer' or (target.endswith('drawer') and 'front' not in target and 'door' not in target):
            return 'drawer front or drawer door'
        if 'drawer front' in target or 'drawer door' in target:
            return target
        if target == 'cabinet door':
            return 'cabinet door panel'
        if target == 'refrigerator door':
            return 'refrigerator door panel'
        if target == 'microwave door':
            return 'microwave door panel'
    return target


def _interaction_focus_clause(target_label: str, interactive_element: Optional[str], role: str = 'object') -> str:
    """Return a short clause that nudges SAM toward the manipulated body, not just the contact part."""
    target = str(target_label or '').strip().lower()
    element = str(interactive_element or '').strip().lower()
    role = str(role or 'object').strip().lower()
    if not target:
        return ''
    target_phrase = _interaction_target_phrase(target, element, role)
    if element and element != target:
        if 'drawer' in target:
            return f"include the whole {target_phrase}, not only the {element}"
        if 'door' in target:
            return f"include the whole {target_phrase}, not only the {element}"
        if any(tok in target for tok in ('lid', 'cover', 'cap', 'flap', 'hatch', 'hood', 'gate', 'panel')):
            return f"include the whole {target_phrase}, not only the {element}"
    return ''


# When True, SAM text prompts are reduced to the bare noun-phrase label. Plain
# SAM3 only grounds on simple noun phrases — it returns EMPTY masks for the
# elaborate multi-clause prompts below (verified: "the drawer handle being
# touched and used to manipulate the drawer; ..." -> 0 px, while "drawer handle"
# -> clean stable masks). SAM3-I tolerates the elaborate text, so this is gated
# on the backend and set once at pipeline start.
_SAM_PROMPT_SIMPLE = False


def _set_sam_prompt_simple(flag: bool) -> None:
    global _SAM_PROMPT_SIMPLE
    _SAM_PROMPT_SIMPLE = bool(flag)


def build_interaction_aware_sam_text(
    target_label: str,
    *,
    interactive_element: Optional[str] = None,
    parent_object: Optional[str] = None,
    role: str = 'object',
    active: bool = False,
) -> str:
    """Return the last-commit SAM prompt style: the bare noun-phrase label."""
    target = str(target_label or '').strip().lower()
    if not target:
        return ''
    return target


def run_sam3_prompt_specs(
    jpeg_dir: Path,
    prompt_specs: Sequence[dict],
    hand_prompts: Sequence[str],
    n_frames: int,
    color_w: int,
    color_h: int,
    predictor: Optional[Any] = None,
) -> Tuple[Dict[str, List[np.ndarray]], List[np.ndarray], Dict[str, List[dict]]]:
    """Track objects and hands with SAM3, returning masks plus per-frame confidence metadata.

    If ``predictor`` is provided, it is reused (caller owns its lifecycle). Otherwise
    a fresh predictor is built and shut down inside this call (legacy behavior).
    """
    H, W = color_h, color_w
    owns_predictor = predictor is None
    if owns_predictor:
        predictor = build_configured_sam3_video_predictor()
    object_masks = {
        str(spec["label"]): [np.zeros((H, W), dtype=bool) for _ in range(n_frames)]
        for spec in prompt_specs
    }
    object_metadata: Dict[str, List[dict]] = {
        str(spec["label"]): []
        for spec in prompt_specs
    }
    hand_masks = [np.zeros((H, W), dtype=bool) for _ in range(n_frames)]

    def _normalize_mask(mask_data) -> Optional[np.ndarray]:
        if isinstance(mask_data, torch.Tensor):
            mask = mask_data.float().squeeze().cpu().numpy() > 0.5
        elif isinstance(mask_data, np.ndarray):
            mask = mask_data.squeeze() > 0.5
        else:
            return None
        mask = np.asarray(mask)
        if mask.size == 0:
            return None
        while mask.ndim > 2:
            if mask.shape[0] == 0:
                return None
            mask = mask[0]
        if mask.ndim != 2 or mask.shape[0] == 0 or mask.shape[1] == 0:
            return None
        if mask.shape != (H, W):
            mask = cv2.resize(mask.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
        return np.asarray(mask, dtype=bool)

    # One session per clip — frames load + GPU encoding happen once and are
    # reused across every prompt / bootstrap pass / hand prompt. add_prompt
    # internally resets the per-prompt state, so prompts are processed
    # sequentially within the shared session.
    response = predictor.handle_request({"type": "start_session", "resource_path": str(jpeg_dir)})
    session_id = response["session_id"]

    def _consume_session(
        store: List[np.ndarray],
        prob_store: List[float],
        add_request: dict,
        prompt_frame_idx: int,
        box_store: Optional[List[Optional[List[float]]]] = None,
    ) -> None:
        predictor.handle_request({"type": "reset_session", "session_id": session_id})
        add_payload = {"type": "add_prompt", "session_id": session_id, "frame_index": int(prompt_frame_idx)}
        add_payload.update(add_request)
        predictor.handle_request(add_payload)
        for response in predictor.handle_stream_request(
            {
                "type": "propagate_in_video",
                "session_id": session_id,
                "start_frame_idx": int(prompt_frame_idx),
                "propagation_direction": "both",
            }
        ):
            frame_index = response.get("frame_index", -1)
            outputs = response.get("outputs")
            if outputs is None or frame_index < 0 or frame_index >= n_frames:
                continue
            if isinstance(outputs, dict) and "out_binary_masks" in outputs:
                masks_arr = outputs.get("out_binary_masks")
                probs_arr = outputs.get("out_probs", None)
                # Per-frame SAM2 propagation score (genuine per-frame mask
                # quality). out_probs is only the constant first-frame DETECTION
                # score; out_sam2_probs varies frame-to-frame and is the real
                # propagation confidence (0 on the seed frame, where we fall back
                # to out_probs).
                sam2_probs_arr = outputs.get("out_sam2_probs", None)
                boxes_arr = outputs.get("out_boxes_xywh", None)
                if isinstance(masks_arr, torch.Tensor):
                    masks_arr = masks_arr.detach().cpu().numpy()
                if isinstance(probs_arr, torch.Tensor):
                    probs_arr = probs_arr.detach().cpu().numpy()
                if isinstance(sam2_probs_arr, torch.Tensor):
                    sam2_probs_arr = sam2_probs_arr.detach().cpu().numpy()
                if isinstance(boxes_arr, torch.Tensor):
                    boxes_arr = boxes_arr.detach().cpu().numpy()
                boxes_np = None
                if boxes_arr is not None:
                    boxes_np = np.asarray(boxes_arr, dtype=np.float32)
                    if boxes_np.ndim == 1 and boxes_np.size >= 4:
                        boxes_np = boxes_np.reshape(1, -1)
                    elif boxes_np.ndim > 2 and boxes_np.shape[-1] >= 4:
                        boxes_np = boxes_np.reshape(-1, boxes_np.shape[-1])
                masks_arr = np.asarray(masks_arr)
                if masks_arr.ndim == 2:
                    masks_arr = masks_arr[None, ...]
                for obj_idx in range(masks_arr.shape[0]):
                    mask = _normalize_mask(masks_arr[obj_idx])
                    if mask is None:
                        continue
                    # Prefer the per-frame SAM2 propagation score; fall back to
                    # the (constant) first-frame detection score on the seed
                    # frame, where the SAM2 score is 0/absent.
                    det_prob = np.nan
                    if probs_arr is not None:
                        probs_np = np.asarray(probs_arr).reshape(-1)
                        if obj_idx < len(probs_np):
                            det_prob = float(probs_np[obj_idx])
                    sam2_prob = np.nan
                    if sam2_probs_arr is not None:
                        sam2_np = np.asarray(sam2_probs_arr).reshape(-1)
                        if obj_idx < len(sam2_np):
                            sam2_prob = float(sam2_np[obj_idx])
                    prob = sam2_prob if (np.isfinite(sam2_prob) and sam2_prob > 0.0) else det_prob
                    # Hard score gate: when SAM3's own probability is below the
                    # threshold, DROP this frame's mask (absent/ambiguous object ->
                    # empty, not "random stuff"). NaN (no score) is kept as legacy.
                    if SAM3_RAW_SCORE_MIN > 0.0 and np.isfinite(prob) and prob < float(SAM3_RAW_SCORE_MIN):
                        continue
                    store[frame_index] |= mask
                    should_update_box = (
                        box_store is not None
                        and boxes_np is not None
                        and obj_idx < len(boxes_np)
                        and (
                            box_store[frame_index] is None
                            or not np.isfinite(prob_store[frame_index])
                            or (np.isfinite(prob) and prob >= float(prob_store[frame_index]))
                        )
                    )
                    if np.isfinite(prob):
                        old_prob = prob_store[frame_index]
                        prob_store[frame_index] = max(float(old_prob), prob) if np.isfinite(old_prob) else prob
                    if should_update_box:
                        box_values = np.asarray(boxes_np[obj_idx], dtype=np.float32).reshape(-1)
                        if box_values.shape[0] >= 4 and np.all(np.isfinite(box_values[:4])):
                            box_store[frame_index] = [float(v) for v in box_values[:4]]
                continue

            # Fallback for older response formats, without native confidence.
            for key, mask_data in outputs.items() if isinstance(outputs, dict) else []:
                if str(key).startswith("out_") or str(key) == "frame_stats":
                    continue
                if isinstance(mask_data, dict):
                    mask_data = mask_data.get("mask", mask_data.get("masks"))
                if mask_data is None:
                    continue
                mask = _normalize_mask(mask_data)
                if mask is None:
                    continue
                store[frame_index] |= mask

    try:
        for spec in tqdm(prompt_specs, desc="  SAM3", leave=False):
            label = str(spec["label"])
            prompt_frame_idx = int(spec.get("frame_index", 0))
            text_prompt = str(spec.get("text", "")) if spec.get("text") else ""
            first_pass_masks = [np.zeros((H, W), dtype=bool) for _ in range(n_frames)]
            first_pass_probs = [float("nan") for _ in range(n_frames)]
            first_pass_boxes: List[Optional[List[float]]] = [None for _ in range(n_frames)]
            first_request: dict = {}
            if text_prompt:
                first_request["text"] = text_prompt
            if spec.get("bounding_boxes") is not None:
                first_request["bounding_boxes"] = spec["bounding_boxes"]
                first_request["bounding_box_labels"] = spec.get("bounding_box_labels", [1] * len(spec["bounding_boxes"]))
            _consume_session(first_pass_masks, first_pass_probs, first_request, prompt_frame_idx, first_pass_boxes)
            first_pass_areas = [int(np.asarray(mask, dtype=bool).sum()) for mask in first_pass_masks]
            max_first_pass_area = max(first_pass_areas) if first_pass_areas else 0

            if text_prompt and spec.get("bounding_boxes") is None:
                print(
                    f"[SAM3] label='{label}' prompt_frame={int(prompt_frame_idx)} "
                    f"using first-pass text masks only (bootstrap disabled, max_area={int(max_first_pass_area)})"
                )
            object_masks[label] = first_pass_masks
            object_metadata[label] = compute_sam3_mask_confidence_metadata(
                first_pass_masks,
                first_pass_probs,
                (H, W),
                first_pass_boxes,
            )

        for prompt in tqdm(hand_prompts, desc="  SAM3-hands", leave=False):
            _consume_session(hand_masks, [float("nan") for _ in range(n_frames)], {"text": str(prompt)}, 0)
    finally:
        predictor.handle_request({"type": "close_session", "session_id": session_id})

    if owns_predictor:
        predictor.shutdown()
        del predictor
    torch.cuda.empty_cache()
    gc.collect()
    return object_masks, hand_masks, object_metadata



def build_exclusive_object_masks(
    object_masks: Sequence[np.ndarray],
    element_masks: Sequence[np.ndarray],
    dilation_pixels: int = 8,
) -> List[np.ndarray]:
    """Remove the interactive element region from the parent-object masks."""
    def _nearest_component(mask: np.ndarray, reference_mask: np.ndarray) -> np.ndarray:
        candidate = np.asarray(mask, dtype=np.uint8)
        if candidate.sum() == 0:
            return candidate.astype(bool)
        num_labels, labels, _stats, centroids = cv2.connectedComponentsWithStats(candidate, connectivity=8)
        if num_labels <= 2:
            return candidate.astype(bool)
        ref_bbox = mask_to_bbox(np.asarray(reference_mask, dtype=bool))
        if ref_bbox is None:
            return candidate.astype(bool)
        ref_center = np.asarray(_box_center(ref_bbox), dtype=np.float32)
        best_label = None
        best_score = None
        for label_idx in range(1, num_labels):
            component_mask = labels == label_idx
            if not np.any(component_mask):
                continue
            centroid = np.asarray(centroids[label_idx], dtype=np.float32)
            area = int(component_mask.sum())
            score = (float(np.linalg.norm(centroid - ref_center)), -area)
            if best_score is None or score < best_score:
                best_score = score
                best_label = label_idx
        if best_label is None:
            return candidate.astype(bool)
        return (labels == best_label)

    exclusive_masks: List[np.ndarray] = []
    kernel_size = max(1, int(dilation_pixels) * 2 + 1)
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    for object_mask, element_mask in zip(object_masks, element_masks):
        obj = np.asarray(object_mask, dtype=bool)
        elem = np.asarray(element_mask, dtype=bool)
        if not np.any(obj) or not np.any(elem):
            exclusive_masks.append(obj.copy())
            continue
        dilated_elem = cv2.dilate(elem.astype(np.uint8), kernel, iterations=1) > 0
        exclusive = obj & ~dilated_elem
        if not np.any(exclusive):
            exclusive = obj & ~elem
        if np.any(exclusive):
            exclusive = _nearest_component(exclusive.astype(np.uint8), elem)
        exclusive_masks.append(exclusive.astype(bool))
    return exclusive_masks

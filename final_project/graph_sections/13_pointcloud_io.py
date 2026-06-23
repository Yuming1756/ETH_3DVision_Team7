# ======================================================================
# graph.py section: 13_pointcloud_io.py
# Point-cloud bounds/colors/snapshots and overlay PNG export
#
# NOTE: not a standalone module. Loaded by graph.py into a shared
# namespace via exec(); names from earlier sections are in scope here.
# ======================================================================
from __future__ import annotations
def stack_point_blocks(point_blocks: Sequence[np.ndarray], color_blocks: Sequence[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """Concatenate several point-cloud chunks into one array pair."""
    if not point_blocks:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)
    return np.concatenate(list(point_blocks), axis=0), np.concatenate(list(color_blocks), axis=0)


def point_cloud_bounds(points: np.ndarray, margin: float = 0.0) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Return an expanded axis-aligned bounding box for a point cloud."""
    if points is None or len(points) == 0:
        return None
    pts = np.asarray(points, dtype=np.float32)
    margin_vec = np.full(3, float(margin), dtype=np.float32)
    return pts.min(axis=0) - margin_vec, pts.max(axis=0) + margin_vec


def append_suppression_box(
    suppression_boxes: List[Tuple[np.ndarray, np.ndarray]],
    points: np.ndarray,
    margin: float = SUPPRESSION_BOX_MARGIN,
) -> None:
    """Append one suppression box used to erase stale geometry from the static scene."""
    bounds = point_cloud_bounds(points, margin=margin)
    if bounds is not None:
        suppression_boxes.append(bounds)


def stable_semantic_color(label: str) -> np.ndarray:
    """Return one deterministic bright color in [0, 1] for a semantic label."""
    digest = hashlib.md5(str(label).strip().lower().encode("utf-8")).digest()
    base = np.frombuffer(digest[:3], dtype=np.uint8).astype(np.float32) / 255.0
    return 0.35 + 0.65 * base


def semantic_point_colors(points: np.ndarray, label: str) -> np.ndarray:
    """Paint a point cloud with one deterministic semantic color."""
    pts = np.asarray(points, dtype=np.float32)
    if len(pts) == 0:
        return np.zeros((0, 3), dtype=np.float32)
    color = stable_semantic_color(label).astype(np.float32)
    return np.repeat(color[None, :], len(pts), axis=0).astype(np.float32)


def stack_semantic_point_blocks(point_blocks: Sequence[np.ndarray], labels: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
    """Stack multiple semantic point blocks and assign one color per semantic label."""
    stacked_points: List[np.ndarray] = []
    stacked_colors: List[np.ndarray] = []
    for points, label in zip(point_blocks, labels):
        pts = np.asarray(points, dtype=np.float32)
        if len(pts) == 0:
            continue
        stacked_points.append(pts)
        stacked_colors.append(semantic_point_colors(pts, label))
    if not stacked_points:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)
    return stack_point_blocks(stacked_points, stacked_colors)


def remove_points_in_boxes(
    points: np.ndarray,
    colors: np.ndarray,
    suppression_boxes: Sequence[Tuple[np.ndarray, np.ndarray]],
) -> Tuple[np.ndarray, np.ndarray]:
    """Remove points that fall inside any suppression box."""
    if points is None or len(points) == 0 or not suppression_boxes:
        return points, colors
    keep = np.ones(len(points), dtype=bool)
    for bbox_min, bbox_max in suppression_boxes:
        inside = np.all((points >= bbox_min) & (points <= bbox_max), axis=1)
        keep &= ~inside
    return points[keep], colors[keep]


def build_scene_snapshot(
    global_pcd_pts: Sequence[np.ndarray],
    global_pcd_cols: Sequence[np.ndarray],
    scene_graph: DynamicSceneGraph,
    suppression_boxes: Sequence[Tuple[np.ndarray, np.ndarray]],
) -> Tuple[np.ndarray, np.ndarray]:
    """Merge the static reconstruction with current graph-node geometry."""
    if global_pcd_pts:
        static_pts = np.concatenate(list(global_pcd_pts), axis=0)
        static_cols = np.concatenate(list(global_pcd_cols), axis=0)
    else:
        static_pts = np.zeros((0, 3), dtype=np.float32)
        static_cols = np.zeros((0, 3), dtype=np.float32)

    node_pts = [node.points for node in scene_graph.nodes.values() if len(node.points)]
    node_cols = [node.colors for node in scene_graph.nodes.values() if len(node.points)]
    if node_pts:
        node_pts_arr, node_cols_arr = stack_point_blocks(node_pts, node_cols)
        merged_pts = np.concatenate([static_pts, node_pts_arr], axis=0) if len(static_pts) else node_pts_arr
        merged_cols = np.concatenate([static_cols, node_cols_arr], axis=0) if len(static_cols) else node_cols_arr
    else:
        merged_pts, merged_cols = static_pts, static_cols

    if len(merged_pts) == 0:
        return merged_pts, merged_cols

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(merged_pts)
    pcd.colors = o3d.utility.Vector3dVector(merged_cols)
    pcd = pcd.voxel_down_sample(voxel_size=SEM_VOXEL_SIZE)
    return np.asarray(pcd.points, dtype=np.float32), np.asarray(pcd.colors, dtype=np.float32)


def write_point_cloud_snapshot(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    """Write one downsampled point-cloud snapshot to disk if it contains any points."""
    if points is None or len(points) == 0:
        return
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float32))
    pcd.colors = o3d.utility.Vector3dVector(np.asarray(colors, dtype=np.float32))
    pcd = pcd.voxel_down_sample(voxel_size=SEM_VOXEL_SIZE)
    o3d.io.write_point_cloud(str(path), pcd)


def constant_point_colors(points: np.ndarray, color: Sequence[float] = (1.0, 1.0, 1.0)) -> np.ndarray:
    """Return one constant color block for a point cloud."""
    pts = np.asarray(points, dtype=np.float32)
    if len(pts) == 0:
        return np.zeros((0, 3), dtype=np.float32)
    base = np.asarray(color, dtype=np.float32).reshape(1, 3)
    return np.repeat(base, len(pts), axis=0)


def remove_horizontal_plane_leakage(
    points: np.ndarray,
    colors: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Remove a dominant horizontal support plane accidentally included in a small object mask."""
    pts = np.asarray(points, dtype=np.float32)
    cols = np.asarray(colors, dtype=np.float32)
    if len(pts) < MASK_LIFT_TABLE_PLANE_MIN_POINTS:
        return pts, cols
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    try:
        plane_model, inlier_indices = pcd.segment_plane(
            distance_threshold=float(MASK_LIFT_TABLE_PLANE_DISTANCE_M),
            ransac_n=3,
            num_iterations=100,
        )
    except RuntimeError:
        return pts, cols
    if not inlier_indices:
        return pts, cols
    normal = np.asarray(plane_model[:3], dtype=np.float32)
    norm = float(np.linalg.norm(normal))
    if norm <= 1e-6:
        return pts, cols
    normal /= norm
    if abs(float(normal[2])) < float(MASK_LIFT_TABLE_PLANE_NORMAL_Z):
        return pts, cols
    inlier_mask = np.zeros(len(pts), dtype=bool)
    inlier_mask[np.asarray(inlier_indices, dtype=np.int32)] = True
    inlier_ratio = float(inlier_mask.mean())
    if inlier_ratio < float(MASK_LIFT_TABLE_PLANE_MIN_RATIO):
        return pts, cols
    keep = ~inlier_mask
    if int(keep.sum()) < int(MASK_LIFT_TABLE_PLANE_MIN_REMAINING):
        return pts, cols
    if len(cols) == len(pts):
        return pts[keep], cols[keep]
    return pts[keep], cols


def save_sam3_overlay_png(
    path: Path,
    image_rgb: np.ndarray,
    element_mask: np.ndarray,
    object_mask: np.ndarray,
    interaction_anchor_box: Optional[Sequence[int]] = None,
    element_anchor_box: Optional[Sequence[int]] = None,
    object_anchor_box: Optional[Sequence[int]] = None,
    object_crop_box: Optional[Sequence[int]] = None,
) -> None:
    """Save one RGB overlay visualizing the active SAM3 element/object masks (debug-only)."""
    if not DEBUG_IMAGES_ENABLED:
        return
    canvas = np.asarray(image_rgb, dtype=np.uint8).copy()

    def _blend_mask(mask: np.ndarray, color: Tuple[int, int, int], alpha: float = 0.35) -> None:
        mask_bool = np.asarray(mask, dtype=bool)
        if not np.any(mask_bool):
            return
        color_arr = np.asarray(color, dtype=np.float32)
        canvas[mask_bool] = np.clip(
            (1.0 - alpha) * canvas[mask_bool].astype(np.float32) + alpha * color_arr[None, :],
            0.0,
            255.0,
        ).astype(np.uint8)
        contours, _ = cv2.findContours(mask_bool.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(canvas, contours, -1, color, 2)

    def _draw_box(bbox: Optional[Sequence[int]], color: Tuple[int, int, int], label: str) -> None:
        if bbox is None:
            return
        x1, y1, x2, y2 = [int(v) for v in bbox]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        cv2.putText(canvas, label, (x1 + 4, max(14, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    _blend_mask(object_mask, (80, 220, 120), alpha=0.28)
    _blend_mask(element_mask, (255, 80, 80), alpha=0.45)
    _draw_box(interaction_anchor_box, (255, 255, 255), "interaction")
    _draw_box(element_anchor_box, (255, 80, 80), "element")
    # Avoid visualizing whole-object/crop boxes in mask debug views; they make
    # the grounding look box-driven. Keep only the element box cue above.
    cv2.imwrite(str(path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))


def save_box_overlay_png(
    path: Path,
    image_rgb: np.ndarray,
    boxes: Sequence[Tuple[Optional[Sequence[int]], Tuple[int, int, int], str]],
    points: Optional[Sequence[Tuple[Optional[Sequence[float]], Tuple[int, int, int], str]]] = None,
) -> None:
    """Save one RGB image with labeled debugging boxes (debug-only)."""
    if not DEBUG_IMAGES_ENABLED:
        return
    canvas = np.asarray(image_rgb, dtype=np.uint8).copy()
    for bbox, color, label in boxes:
        if bbox is None:
            continue
        x1, y1, x2, y2 = [int(v) for v in bbox]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        cv2.putText(canvas, label, (x1 + 4, max(14, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    for point, color, label in list(points or []):
        if point is None:
            continue
        x, y = int(round(float(point[0]))), int(round(float(point[1])))
        cv2.circle(canvas, (x, y), 6, color, thickness=2)
        cv2.putText(canvas, label, (x + 6, max(14, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    cv2.imwrite(str(path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))


def save_binary_mask_sequence(
    out_dir: Path,
    masks: Sequence[np.ndarray],
    prefix: str = "mask",
) -> None:
    """Save binary SAM masks as grayscale PNGs (debug-only; gated by GRAPH_DEBUG_IMAGES)."""
    if not DEBUG_IMAGES_ENABLED:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    for idx, mask in enumerate(masks):
        mask_uint8 = (np.asarray(mask, dtype=bool).astype(np.uint8) * 255)
        cv2.imwrite(str(out_dir / f"{prefix}_{idx:04d}.png"), mask_uint8)


def save_sam3_overlay_sequence(
    out_dir: Path,
    rgb_list: Sequence[np.ndarray],
    element_masks: Sequence[np.ndarray],
    object_masks: Sequence[np.ndarray],
    primary_frame_boxes: Optional[dict] = None,
    secondary_frame_index: Optional[int] = None,
    secondary_frame_boxes: Optional[dict] = None,
) -> None:
    """Save per-frame SAM overlays for quick visual inspection (debug-only; gated by GRAPH_DEBUG_IMAGES)."""
    if not DEBUG_IMAGES_ENABLED:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    primary_frame_boxes = dict(primary_frame_boxes or {})
    secondary_frame_boxes = dict(secondary_frame_boxes or {})
    for idx, rgb in enumerate(rgb_list):
        frame_boxes = {}
        if idx == 0:
            frame_boxes.update(primary_frame_boxes)
        if secondary_frame_index is not None and idx == int(secondary_frame_index):
            frame_boxes.update(secondary_frame_boxes)
        save_sam3_overlay_png(
            out_dir / f"{idx:04d}.png",
            rgb,
            element_masks[idx] if idx < len(element_masks) else np.zeros(rgb.shape[:2], dtype=bool),
            object_masks[idx] if idx < len(object_masks) else np.zeros(rgb.shape[:2], dtype=bool),
            interaction_anchor_box=frame_boxes.get("interaction_anchor_box"),
            element_anchor_box=frame_boxes.get("element_anchor_box"),
            object_anchor_box=frame_boxes.get("object_anchor_box"),
            object_crop_box=frame_boxes.get("object_crop_box"),
        )


def _blend_debug_mask_on_canvas(
    canvas: np.ndarray,
    mask: Optional[np.ndarray],
    color: Tuple[int, int, int] = (255, 80, 80),
    alpha: float = 0.22,
) -> None:
    """Blend one binary mask onto a debug RGB canvas in-place."""
    if mask is None:
        return
    mask_bool = np.asarray(mask, dtype=bool)
    if mask_bool.shape[:2] != canvas.shape[:2] or not np.any(mask_bool):
        return
    color_arr = np.asarray(color, dtype=np.float32)
    canvas[mask_bool] = np.clip(
        (1.0 - alpha) * canvas[mask_bool].astype(np.float32) + alpha * color_arr[None, :],
        0.0,
        255.0,
    ).astype(np.uint8)
    contours, _ = cv2.findContours(mask_bool.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(canvas, contours, -1, color, 2)


def _save_artipoint_keypoints_imagery(
    debug_dir: Path,
    frame_idx: int,
    image_rgb: np.ndarray,
    queries_2d: np.ndarray,
    label: str,
    active_mask: Optional[np.ndarray] = None,
    prompt_box: Optional[Sequence[float]] = None,
) -> None:
    """Save one keyframe image showing seeded ArtiPoint corners and the local mask.

    Unlike the bulk SAM/LostFound debug exports, ArtiPoint corner overlays are
    always written when this helper is called so articulated-track debugging is
    available on normal runs without requiring GRAPH_DEBUG_IMAGES=1.
    """
    out_dir = Path(debug_dir) / f"{_slugify_label(label)}_keypoints"
    out_dir.mkdir(parents=True, exist_ok=True)
    canvas = np.asarray(image_rgb, dtype=np.uint8).copy()
    _blend_debug_mask_on_canvas(canvas, active_mask, color=(255, 80, 80), alpha=0.22)
    # Draw the SAM3 object PROMPT box (blue) — the fixed seed region the mask was
    # propagated from. Constant across frames, so when the green per-frame mask
    # box drifts far from this blue box you are seeing SAM3 propagation drift
    # (over/under-segmentation) rather than a bad prompt box.
    if prompt_box is not None:
        try:
            px1, py1, px2, py2 = [int(v) for v in prompt_box]
            cv2.rectangle(canvas, (px1, py1), (px2, py2), (80, 160, 255), 2)
            cv2.putText(
                canvas, "prompt box", (px1 + 4, min(py2 + 22, canvas.shape[0] - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 160, 255), 2, cv2.LINE_AA,
            )
        except Exception:  # noqa: BLE001
            pass
    # Draw the per-frame mask bounding box (green) so the segmentation extent is
    # visible at a glance — makes over-segmentation on individual frames obvious.
    if active_mask is not None:
        mask_bool = np.asarray(active_mask, dtype=bool)
        if np.any(mask_bool):
            bbox = mask_to_bbox(mask_bool)
            if bbox is not None:
                bx1, by1, bx2, by2 = [int(v) for v in bbox]
                cv2.rectangle(canvas, (bx1, by1), (bx2, by2), (80, 255, 80), 2)
                cv2.putText(
                    canvas,
                    f"mask bbox {bx2 - bx1}x{by2 - by1} area={int(np.count_nonzero(mask_bool))}",
                    (bx1 + 4, max(by1 - 8, 38)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (80, 255, 80),
                    2,
                    cv2.LINE_AA,
                )
    queries = np.asarray(queries_2d, dtype=np.float32)
    if queries.ndim == 2 and queries.shape[1] >= 2:
        xy = queries[:, -2:]
        height, width = canvas.shape[:2]
        for x_f, y_f in xy.tolist():
            x = int(round(float(x_f)))
            y = int(round(float(y_f)))
            if 0 <= x < width and 0 <= y < height:
                cv2.circle(canvas, (x, y), 3, (80, 255, 255), thickness=-1)
    cv2.putText(
        canvas,
        f"kf={int(frame_idx)} queries={int(len(queries))}",
        (10, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.imwrite(str(out_dir / f"keypoints_{int(frame_idx):04d}.png"), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
    if active_mask is not None:
        mask_canvas = np.zeros_like(canvas)
        _blend_debug_mask_on_canvas(mask_canvas, active_mask, color=(255, 80, 80), alpha=1.0)
        cv2.imwrite(str(out_dir / f"mask_{int(frame_idx):04d}.png"), cv2.cvtColor(mask_canvas, cv2.COLOR_RGB2BGR))


def _save_artipoint_mask_sequence(
    out_dir: Path,
    rgb_list: Sequence[np.ndarray],
    active_masks: Optional[Sequence[np.ndarray]],
) -> None:
    """Save raw per-frame active masks for ArtiPoint debugging."""
    if active_masks is None or len(rgb_list) == 0:
        return
    mask_dir = Path(out_dir) / "active_query_masks"
    mask_dir.mkdir(parents=True, exist_ok=True)
    image_shape = np.asarray(rgb_list[0], dtype=np.uint8).shape[:2]
    for frame_idx, mask in enumerate(active_masks[: len(rgb_list)]):
        mask_arr = np.asarray(mask, dtype=bool)
        if mask_arr.shape[:2] != image_shape:
            continue
        mask_canvas = np.zeros((*image_shape, 3), dtype=np.uint8)
        _blend_debug_mask_on_canvas(mask_canvas, mask_arr, color=(255, 80, 80), alpha=1.0)
        cv2.imwrite(str(mask_dir / f"mask_{int(frame_idx):04d}.png"), cv2.cvtColor(mask_canvas, cv2.COLOR_RGB2BGR))


def _save_artipoint_tracks_imagery(
    debug_dir: Path,
    rgb_list: Sequence[np.ndarray],
    tracks_2d: np.ndarray,
    visibility: np.ndarray,
    label: str,
    active_masks: Optional[Sequence[np.ndarray]] = None,
    trail_frames: int = 12,
) -> None:
    """Save per-frame ArtiPoint track overlays, optionally with live SAM masks.

    For each frame, draws:
      - The (optional) live SAM mask, blended translucent red.
      - A trailing trajectory line for every visible track covering the last
        ``trail_frames`` frames, coloured deterministically by track index so
        the same track keeps the same colour across the clip.
      - The current 2-D position as a small filled dot (green if currently
        visible, red if invisible — invisible points are still drawn so a
        track suddenly going red flags that CoTracker lost confidence
        rather than that the point left the frame).
      - Header text with frame index and per-frame visible count.

    Additionally writes one ``trajectory_summary.png`` showing every track's
    full path across the entire clip overlaid on the first RGB frame. Use it
    to spot whether the dominant motion in the clip matches the expected
    articulation, or whether a large fraction of tracks are drifting because
    they were seeded on background pixels.
    """
    tracks = np.asarray(tracks_2d, dtype=np.float32)
    vis = np.asarray(visibility)
    if vis.dtype != bool:
        vis = vis > 0.5
    if tracks.ndim != 3 or tracks.shape[-1] != 2 or vis.ndim != 2:
        return
    out_dir = Path(debug_dir) / f"{_slugify_label(label)}_tracks"
    out_dir.mkdir(parents=True, exist_ok=True)
    n_frames = min(len(rgb_list), tracks.shape[0], vis.shape[0])
    if n_frames == 0:
        return
    n_tracks = int(tracks.shape[1])
    # Deterministic colour per track index so a given track stays the same
    # colour across both the per-frame overlays and the trajectory summary.
    palette_rng = np.random.default_rng(seed=0)
    palette = palette_rng.integers(40, 230, size=(max(n_tracks, 1), 3), dtype=np.int32)
    height, width = np.asarray(rgb_list[0]).shape[:2]
    trail = max(0, int(trail_frames))
    visible_counts_by_frame: List[int] = []
    mask_pixels_by_frame: List[int] = []

    _save_artipoint_mask_sequence(out_dir, rgb_list, active_masks)

    for frame_idx in range(n_frames):
        canvas = np.asarray(rgb_list[frame_idx], dtype=np.uint8).copy()
        mask = active_masks[frame_idx] if active_masks is not None and frame_idx < len(active_masks) else None
        mask_arr = np.asarray(mask, dtype=bool) if mask is not None else None
        _blend_debug_mask_on_canvas(canvas, mask, color=(255, 80, 80), alpha=0.18)
        visible_count = 0
        for t in range(n_tracks):
            colour = tuple(int(c) for c in palette[t])
            # Trailing polyline across the most recent ``trail`` frames.
            if trail > 0:
                start_idx = max(0, frame_idx - trail)
                trail_pts: List[Tuple[int, int]] = []
                for k in range(start_idx, frame_idx + 1):
                    if not bool(vis[k, t]):
                        # Break the polyline on invisibility so we don't draw
                        # straight lines across hidden segments.
                        if len(trail_pts) >= 2:
                            cv2.polylines(canvas, [np.asarray(trail_pts, dtype=np.int32)], False, colour, 1, cv2.LINE_AA)
                        trail_pts = []
                        continue
                    xy = tracks[k, t]
                    if not (np.isfinite(xy[0]) and np.isfinite(xy[1])):
                        if len(trail_pts) >= 2:
                            cv2.polylines(canvas, [np.asarray(trail_pts, dtype=np.int32)], False, colour, 1, cv2.LINE_AA)
                        trail_pts = []
                        continue
                    trail_pts.append((int(round(float(xy[0]))), int(round(float(xy[1])))))
                if len(trail_pts) >= 2:
                    cv2.polylines(canvas, [np.asarray(trail_pts, dtype=np.int32)], False, colour, 1, cv2.LINE_AA)
            # Current position dot.
            xy_now = tracks[frame_idx, t]
            if not (np.isfinite(xy_now[0]) and np.isfinite(xy_now[1])):
                continue
            x = int(round(float(xy_now[0])))
            y = int(round(float(xy_now[1])))
            if not (0 <= x < width and 0 <= y < height):
                continue
            if bool(vis[frame_idx, t]):
                cv2.circle(canvas, (x, y), 3, (80, 255, 80), thickness=-1, lineType=cv2.LINE_AA)
                visible_count += 1
            else:
                cv2.circle(canvas, (x, y), 2, (0, 0, 220), thickness=-1, lineType=cv2.LINE_AA)
        visible_counts_by_frame.append(int(visible_count))
        if mask_arr is not None and mask_arr.shape[:2] == (height, width):
            mask_pixels_by_frame.append(int(np.count_nonzero(mask_arr)))
        else:
            mask_pixels_by_frame.append(0)
        header = f"frame={int(frame_idx)} visible={int(visible_count)}/{n_tracks} trail={trail}"
        cv2.putText(canvas, header, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(canvas, header, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imwrite(str(out_dir / f"tracks_{int(frame_idx):04d}.png"), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))

    # Trajectory summary: all tracks' visible-segment polylines overlaid on
    # the first frame. Quick way to eyeball the dominant motion across the
    # whole clip without scrubbing every per-frame image.
    summary = np.asarray(rgb_list[0], dtype=np.uint8).copy()
    if active_masks is not None:
        try:
            mask_union = np.any(
                np.stack([
                    np.asarray(mask, dtype=bool)
                    for mask in active_masks[:n_frames]
                    if np.asarray(mask).shape[:2] == summary.shape[:2]
                ], axis=0),
                axis=0,
            )
            _blend_debug_mask_on_canvas(summary, mask_union, color=(255, 80, 80), alpha=0.15)
        except Exception:
            mask_union = None
    else:
        mask_union = None
    summary_height, summary_width = summary.shape[:2]
    drawn_tracks = 0
    for t in range(n_tracks):
        colour = tuple(int(c) for c in palette[t])
        seg: List[Tuple[int, int]] = []
        any_drawn = False
        for k in range(n_frames):
            if not bool(vis[k, t]):
                if len(seg) >= 2:
                    cv2.polylines(summary, [np.asarray(seg, dtype=np.int32)], False, colour, 1, cv2.LINE_AA)
                    any_drawn = True
                seg = []
                continue
            xy = tracks[k, t]
            if not (np.isfinite(xy[0]) and np.isfinite(xy[1])):
                if len(seg) >= 2:
                    cv2.polylines(summary, [np.asarray(seg, dtype=np.int32)], False, colour, 1, cv2.LINE_AA)
                    any_drawn = True
                seg = []
                continue
            x = int(round(float(xy[0])))
            y = int(round(float(xy[1])))
            if 0 <= x < summary_width and 0 <= y < summary_height:
                seg.append((x, y))
        if len(seg) >= 2:
            cv2.polylines(summary, [np.asarray(seg, dtype=np.int32)], False, colour, 1, cv2.LINE_AA)
            any_drawn = True
        if any_drawn:
            drawn_tracks += 1
    header = f"trajectory summary: {drawn_tracks}/{n_tracks} tracks, {n_frames} frames"
    cv2.putText(summary, header, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(summary, header, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(str(out_dir / "trajectory_summary.png"), cv2.cvtColor(summary, cv2.COLOR_RGB2BGR))
    if mask_union is not None:
        mask_canvas = np.zeros_like(summary)
        _blend_debug_mask_on_canvas(mask_canvas, mask_union, color=(255, 80, 80), alpha=1.0)
        cv2.imwrite(str(out_dir / "mask_union.png"), cv2.cvtColor(mask_canvas, cv2.COLOR_RGB2BGR))
    with open(out_dir / "metadata.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "label": str(label),
                "n_frames": int(n_frames),
                "n_tracks": int(n_tracks),
                "trail_frames": int(trail),
                "visible_counts_by_frame": visible_counts_by_frame,
                "mask_pixels_by_frame": mask_pixels_by_frame,
                "has_active_masks": bool(active_masks is not None),
            },
            handle,
            indent=2,
        )




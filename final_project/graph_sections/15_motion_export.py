# ======================================================================
# graph.py section: 15_motion_export.py
# Motion records, viser/wrist export, clip frame IO
#
# NOTE: not a standalone module. Loaded by graph.py into a shared
# namespace via exec(); names from earlier sections are in scope here.
# ======================================================================
from __future__ import annotations
def build_motion_record_from_tracking(
    clip: ClipInfo,
    result: Optional[dict],
    prior_source: str,
) -> Optional[dict]:
    """Convert one LostFound result into an exact-frame motion segment for the persistent graph."""
    if result is None or len(result.get("initial_points", [])) == 0:
        return None
    pose_history = [
        {
            "frame": int(clip.frame_start),
            "pose": np.asarray(result["initial_pose"], dtype=np.float32),
            "source": "reference",
            "confidence": 1.0,
            "clip_id": int(clip.clip_id),
        }
    ]
    for sample in result.get("pose_history", []) or []:
        pose_history.append(
            {
                "frame": int(sample["frame"]),
                "pose": np.asarray(sample["pose"], dtype=np.float32),
                "source": str(sample.get("source", "tracked")),
                "confidence": float(sample.get("confidence", 1.0)),
                "clip_id": int(clip.clip_id),
            }
        )
    record = {
        "reference_points": np.asarray(result["initial_points"], dtype=np.float32),
        "reference_colors": np.asarray(result.get("initial_colors", np.zeros((0, 3), dtype=np.float32)), dtype=np.float32),
        "reference_pose": np.asarray(result["initial_pose"], dtype=np.float32),
        "pose_history": pose_history,
        "prior_source": prior_source,
    }
    # Pass per-corner trajectories (ArtiPoint-style) through when present so
    # they ultimately land on the NodeMotionSegment and become available to
    # the viser viewer for per-keypoint trajectory visualisation.
    if "corner_tracks_world" in result:
        record["corner_tracks_world"] = np.asarray(result["corner_tracks_world"], dtype=np.float32)
    if "corner_tracks_valid" in result:
        record["corner_tracks_valid"] = np.asarray(result["corner_tracks_valid"], dtype=bool)
    if "corner_track_frames" in result:
        record["corner_track_frames"] = list(result["corner_track_frames"])
    if "articulation_axis_candidates" in result:
        record["articulation_axis_candidates"] = list(result["articulation_axis_candidates"] or [])
    return record


def build_motion_record_from_tracking_reference(
    clip: ClipInfo,
    result: Optional[dict],
    prior_source: str,
    reference_points: np.ndarray,
    reference_colors: np.ndarray,
) -> Optional[dict]:
    """Create a motion segment from a shared LostFound track but with custom reference geometry."""
    if result is None or reference_points is None or len(reference_points) == 0:
        return None
    pose_history = [
        {
            "frame": int(clip.frame_start),
            "pose": np.asarray(result["initial_pose"], dtype=np.float32),
            "source": "reference",
            "confidence": 1.0,
            "clip_id": int(clip.clip_id),
        }
    ]
    for sample in result.get("pose_history", []) or []:
        pose_history.append(
            {
                "frame": int(sample["frame"]),
                "pose": np.asarray(sample["pose"], dtype=np.float32),
                "source": str(sample.get("source", "tracked")),
                "confidence": float(sample.get("confidence", 1.0)),
                "clip_id": int(clip.clip_id),
            }
        )
    record = {
        "reference_points": np.asarray(reference_points, dtype=np.float32),
        "reference_colors": np.asarray(reference_colors, dtype=np.float32),
        "reference_pose": np.asarray(result["initial_pose"], dtype=np.float32),
        "pose_history": pose_history,
        "prior_source": prior_source,
    }
    # Per-corner trajectories were missing on the shared-element articulated
    # path (cabinet hits this), so viser had nothing to render for the
    # "Keypoint trajectories" toggle. Mirror what
    # ``build_motion_record_from_tracking`` does so the shared path also
    # forwards CoTracker3 arcs through to NodeMotionSegment.
    if "corner_tracks_world" in result:
        record["corner_tracks_world"] = np.asarray(result["corner_tracks_world"], dtype=np.float32)
    if "corner_tracks_valid" in result:
        record["corner_tracks_valid"] = np.asarray(result["corner_tracks_valid"], dtype=bool)
    if "corner_track_frames" in result:
        record["corner_track_frames"] = list(result["corner_track_frames"])
    if "articulation_axis_candidates" in result:
        record["articulation_axis_candidates"] = list(result["articulation_axis_candidates"] or [])
    return record


def build_observed_final_motion_record(
    frame: int,
    points: np.ndarray,
    colors: np.ndarray,
) -> Optional[dict]:
    """Create an exact one-frame fallback record when LostFound cannot track the object."""
    pts = np.asarray(points, dtype=np.float32) if points is not None else np.zeros((0, 3), dtype=np.float32)
    if len(pts) == 0:
        return None
    cols = np.asarray(colors, dtype=np.float32) if colors is not None and len(colors) == len(pts) else np.full((len(pts), 3), 0.5, dtype=np.float32)
    identity = np.eye(4, dtype=np.float32)
    return {
        "reference_points": pts,
        "reference_colors": cols,
        "reference_pose": identity,
        "pose_history": [
            {
                "frame": int(frame),
                "pose": identity,
                "source": "observed_final",
                "confidence": 1.0,
            }
        ],
        "prior_source": "observed_final",
    }


def _slugify_label(label: str) -> str:
    """Create a filesystem-safe label fragment for debug artifact names."""
    slug = re.sub(r"[^a-z0-9]+", "_", str(label).strip().lower())
    slug = slug.strip("_")
    return slug or "object"


def _constant_point_colors(points: np.ndarray, color: Sequence[float]) -> np.ndarray:
    """Create one constant RGB color block for a point cloud."""
    pts = np.asarray(points, dtype=np.float32)
    if len(pts) == 0:
        return np.zeros((0, 3), dtype=np.float32)
    color_arr = np.asarray(color, dtype=np.float32).reshape(1, 3)
    return np.repeat(color_arr, len(pts), axis=0)


def project_world_points_to_pixels(
    points_world: np.ndarray,
    image_rgb: np.ndarray,
    camera_pose: np.ndarray,
    K_color: np.ndarray,
    T_cam_color: np.ndarray,
) -> np.ndarray:
    """Project world-space points into one RGB frame and return visible pixel coordinates."""
    points_world = np.asarray(points_world, dtype=np.float32)
    if len(points_world) == 0:
        return np.zeros((0, 2), dtype=np.int32)

    color_from_world = np.linalg.inv(np.asarray(T_cam_color, dtype=np.float64)) @ np.linalg.inv(np.asarray(camera_pose, dtype=np.float64))
    points_h = np.concatenate([points_world.astype(np.float64), np.ones((len(points_world), 1), dtype=np.float64)], axis=1)
    points_color = (color_from_world @ points_h.T).T[:, :3]
    valid = points_color[:, 2] > 1e-4
    if not np.any(valid):
        return np.zeros((0, 2), dtype=np.int32)

    points_color = points_color[valid]
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
        return np.zeros((0, 2), dtype=np.int32)
    return np.round(pixels[inside]).astype(np.int32)


def write_hand_trajectory_snapshot(
    path: Path,
    left_positions: Sequence[Optional[np.ndarray]],
    right_positions: Sequence[Optional[np.ndarray]],
) -> None:
    """Write a combined left/right hand trajectory point cloud for debugging."""
    point_blocks: List[np.ndarray] = []
    color_blocks: List[np.ndarray] = []
    left_pts = np.asarray([hand for hand in left_positions if hand is not None], dtype=np.float32)
    right_pts = np.asarray([hand for hand in right_positions if hand is not None], dtype=np.float32)
    if len(left_pts):
        point_blocks.append(left_pts)
        color_blocks.append(_constant_point_colors(left_pts, (1.0, 0.2, 0.2)))
    if len(right_pts):
        point_blocks.append(right_pts)
        color_blocks.append(_constant_point_colors(right_pts, (0.2, 0.4, 1.0)))
    if not point_blocks:
        return
    points, colors = stack_point_blocks(point_blocks, color_blocks)
    write_point_cloud_snapshot(path, points, colors)


def export_clip_wrist_observation_bundle(clip: ClipInfo, output_dir: Path) -> None:
    """Export per-frame wrist observations for timeline playback, including inactive clips."""
    viser_dir = output_dir / f"clip_{clip.clip_id:03d}" / "viser_wrist"
    viser_dir.mkdir(parents=True, exist_ok=True)
    hands: Dict[str, List[dict]] = {}
    for hand in ("left", "right"):
        records: List[dict] = []
        wrist_key = f"{hand}_hand_world_wrist"
        estimated_key = f"{hand}_hand_world_wrist_estimated"
        valid_key = f"{hand}_hand_valid"
        for sample in clip.samples:
            position = sample.get(wrist_key)
            if position is None:
                continue
            position_arr = np.asarray(position, dtype=np.float32).reshape(-1)
            if position_arr.shape[0] != 3 or not np.all(np.isfinite(position_arr)):
                continue
            frame = int(sample.get("prefix", sample.get("frame", -1)))
            if frame < 0:
                continue
            records.append(
                {
                    "frame": frame,
                    "position": [float(v) for v in position_arr.tolist()],
                    "estimated": bool(sample.get(estimated_key, False)),
                    "valid": bool(sample.get(valid_key, False)),
                }
            )
        hands[hand] = records
    frame_assets: List[dict] = []
    for idx, sample in enumerate(clip.samples):
        frame = int(sample.get("prefix", sample.get("frame", -1)))
        if frame < 0:
            continue
        rgb_path = sample.get("rgb_path")
        depth_path = sample.get("depth_path")
        frame_assets.append(
            {
                "frame": frame,
                "clip_index": int(idx),
                "rgb_path": str(rgb_path) if rgb_path is not None else "",
                "depth_path": str(depth_path) if depth_path is not None else "",
            }
        )
    with open(viser_dir / "metadata.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "clip_id": int(clip.clip_id),
                "frame_start": int(clip.frame_start),
                "frame_end": int(clip.frame_end),
                "type": "inactive" if clip.is_inactive else "active",
                "interaction_hand": clip.interaction_hand,
                "hands": hands,
                "frames": frame_assets,
            },
            handle,
            indent=2,
        )


def render_motion_record_geometry_at_frame(
    motion_record: Optional[dict],
    frame: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Render one motion record's reference geometry at an exact global frame."""
    if motion_record is None:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)
    reference_points = np.asarray(motion_record.get("reference_points", np.zeros((0, 3), dtype=np.float32)), dtype=np.float32)
    if len(reference_points) == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)
    reference_colors = np.asarray(motion_record.get("reference_colors", np.zeros((0, 3), dtype=np.float32)), dtype=np.float32)
    if len(reference_colors) != len(reference_points):
        reference_colors = np.full((len(reference_points), 3), 0.5, dtype=np.float32)
    reference_pose = np.asarray(motion_record.get("reference_pose", np.eye(4, dtype=np.float32)), dtype=np.float32)
    pose_sample = None
    for sample in motion_record.get("pose_history", []) or []:
        if int(sample.get("frame", -1)) == int(frame):
            pose_sample = sample
            break
    if pose_sample is None:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)
    pose = np.asarray(pose_sample.get("pose", np.eye(4, dtype=np.float32)), dtype=np.float32)
    delta = pose @ np.linalg.inv(reference_pose)
    return apply_homogeneous_transform(reference_points, delta), reference_colors


def make_wrist_marker_points(
    center: np.ndarray,
    radius: float = 0.02,
    n_theta: int = 12,
    n_phi: int = 6,
) -> np.ndarray:
    """Create a small visible point marker around one wrist position for Viser."""
    center = np.asarray(center, dtype=np.float32).reshape(3)
    points = [center]
    for phi in np.linspace(0.0, np.pi, num=max(2, int(n_phi)), endpoint=True):
        sin_phi = float(np.sin(phi))
        cos_phi = float(np.cos(phi))
        for theta in np.linspace(0.0, 2.0 * np.pi, num=max(4, int(n_theta)), endpoint=False):
            points.append(
                center
                + float(radius) * np.array(
                    [sin_phi * np.cos(theta), sin_phi * np.sin(theta), cos_phi],
                    dtype=np.float32,
                )
            )
    return np.asarray(points, dtype=np.float32)


def export_viser_motion_bundle(
    clip: ClipInfo,
    clip_dir: Path,
    object_label: str,
    motion_record: Optional[dict],
    tracked_hand: Optional[str],
    hand_anchor_mode: Optional[str],
) -> None:
    """Export per-frame object geometry and wrist markers for direct Viser inspection."""
    # Disabled: viser_motion is an extra per-frame debug bundle. The live viser
    # viewer (viser_io.py) only needs scene_graph/ + viser_wrist/, which are
    # still written, so this folder is removed to keep the debug output minimal.
    return
    if motion_record is None:
        return
    pose_history = list(motion_record.get("pose_history", []) or [])
    if not pose_history:
        return

    viser_dir = clip_dir / "viser_motion"
    object_dir = viser_dir / "object_frames"
    wrist_dir = viser_dir / "wrist_frames"
    combined_dir = viser_dir / "combined_frames"
    for directory in (viser_dir, object_dir, wrist_dir, combined_dir):
        directory.mkdir(parents=True, exist_ok=True)

    sample_by_frame = {
        int(sample.get("frame", sample.get("prefix", -1))): sample
        for sample in clip.samples
    }
    exported_frames: List[int] = []
    wrist_positions: List[np.ndarray] = []
    object_centroids: List[np.ndarray] = []
    for pose_sample in pose_history:
        frame = int(pose_sample.get("frame", -1))
        if frame < 0:
            continue
        object_points, object_colors = render_motion_record_geometry_at_frame(motion_record, frame)
        if len(object_points) == 0:
            continue
        exported_frames.append(frame)
        write_point_cloud_snapshot(
            object_dir / f"{object_label.replace(' ', '_')}_frame_{frame:06d}.ply",
            object_points,
            object_colors,
        )
        object_centroids.append(np.asarray(object_points, dtype=np.float32).mean(axis=0))

        wrist_points = np.zeros((0, 3), dtype=np.float32)
        wrist_colors = np.zeros((0, 3), dtype=np.float32)
        sample = sample_by_frame.get(frame)
        wrist_position = None
        if sample is not None and tracked_hand in {"left", "right"}:
            wrist_position = sample.get(f"{tracked_hand}_hand_world_wrist")
            if wrist_position is None:
                wrist_position = sample.get(f"{tracked_hand}_hand_world")
        if wrist_position is not None:
            wrist_position = np.asarray(wrist_position, dtype=np.float32)
            wrist_positions.append(wrist_position)
            wrist_points = make_wrist_marker_points(wrist_position)
            wrist_colors = constant_point_colors(wrist_points, (1.0, 0.1, 0.1) if tracked_hand == "left" else (0.15, 0.45, 1.0))
            write_point_cloud_snapshot(
                wrist_dir / f"{tracked_hand or 'hand'}_wrist_frame_{frame:06d}.ply",
                wrist_points,
                wrist_colors,
            )

        combined_points, combined_colors = stack_point_blocks(
            [pts for pts in (object_points, wrist_points) if len(pts)],
            [cols for cols in (object_colors, wrist_colors) if len(cols)],
        )
        write_point_cloud_snapshot(
            combined_dir / f"{object_label.replace(' ', '_')}_with_wrist_frame_{frame:06d}.ply",
            combined_points,
            combined_colors,
        )

    if object_centroids:
        object_traj = np.asarray(object_centroids, dtype=np.float32)
        write_point_cloud_snapshot(
            viser_dir / "object_trajectory.ply",
            object_traj,
            constant_point_colors(object_traj, (1.0, 0.65, 0.15)),
        )

    if wrist_positions:
        wrist_traj = np.asarray(wrist_positions, dtype=np.float32)
        write_point_cloud_snapshot(
            viser_dir / "wrist_trajectory.ply",
            wrist_traj,
            constant_point_colors(wrist_traj, (1.0, 0.1, 0.1) if tracked_hand == "left" else (0.15, 0.45, 1.0)),
        )

    with open(viser_dir / "metadata.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "clip_id": int(clip.clip_id),
                "frame_start": int(clip.frame_start),
                "frame_end": int(clip.frame_end),
                "object_label": str(object_label),
                "tracked_hand": tracked_hand,
                "hand_anchor_mode": hand_anchor_mode,
                "exported_frames": exported_frames,
            },
            handle,
            indent=2,
        )

    viewer_cmd = (
        f'venv/bin/python viser_io.py --output-dir {clip_dir.parent} --focus-label "{object_label}" --host 127.0.0.1 --port 8080'
    )
    username = getpass.getuser()
    hostname = socket.gethostname()
    instructions = (
        "This folder augments the global viser_io.py viewer with wrist/object motion data.\n\n"
        "1. Start the main viewer on the cluster machine where the output is stored:\n"
        f"   {viewer_cmd}\n\n"
        "2. From your laptop, tunnel the port to that same machine:\n"
        f"   ssh -L 8080:127.0.0.1:8080 {username}@{hostname}\n\n"
        "3. Open the viewer locally in your browser:\n"
        "   http://127.0.0.1:8080\n\n"
        "Inside viser_io.py, use the wrist toggles to inspect the selected hand pose and trajectory.\n"
        "If the viewer is running on a compute node that you can only reach through a login node,\n"
        "use SSH ProxyJump (-J) through the login node instead.\n"
    )
    with open(viser_dir / "README.txt", "w", encoding="utf-8") as handle:
        handle.write(instructions)

    print(f"  [Viser] clip={clip.clip_id} -> {viser_dir}/")
    print(f"  [Viser] wrist/object motion bundle exported for viser_io.py: {viewer_cmd}")


def export_lostfound_debug_assets(
    clip: ClipInfo,
    out_dir: Path,
    label: str,
    prior_points: np.ndarray,
    rgb_list: Sequence[np.ndarray],
    T_wc_list: Sequence[np.ndarray],
    K_color: np.ndarray,
    T_cam_color: np.ndarray,
    asset_prefix: str,
) -> None:
    """Export per-clip LostFound overlays and point clouds (debug-only; gated by GRAPH_DEBUG_IMAGES)."""
    # Disabled: tracking_debug is kept minimal (only the artipoint overlays).
    return
    clip_dir = out_dir / f"clip_{clip.clip_id:03d}"
    asset_dir = clip_dir / "tracking_debug" / f"{asset_prefix}_{_slugify_label(label)}"
    asset_dir.mkdir(parents=True, exist_ok=True)

    prior_points = np.asarray(prior_points, dtype=np.float32)
    hand_positions_left = [sample.get("left_hand_world") for sample in clip.samples]
    hand_positions_right = [sample.get("right_hand_world") for sample in clip.samples]
    hand_reference_positions = [hand for hand in (list(hand_positions_left) + list(hand_positions_right)) if hand is not None]
    tracking_points = select_hand_local_tracking_points(prior_points, hand_reference_positions)

    if len(prior_points):
        write_point_cloud_snapshot(
            asset_dir / "prior_points.ply",
            prior_points,
            _constant_point_colors(prior_points, (0.75, 0.75, 0.75)),
        )
    if len(tracking_points):
        write_point_cloud_snapshot(
            asset_dir / "tracking_points.ply",
            tracking_points,
            _constant_point_colors(tracking_points, (0.1, 1.0, 0.2)),
        )
    write_hand_trajectory_snapshot(asset_dir / "hand_trajectory.ply", hand_positions_left, hand_positions_right)

    overlay_dir = asset_dir / "overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    prior_visual_points = sample_tracking_points(prior_points, max_points=1024)
    tracking_visual_points = sample_tracking_points(tracking_points, max_points=256)
    for idx, (rgb, T_wc) in enumerate(zip(rgb_list, T_wc_list)):
        overlay = np.asarray(rgb, dtype=np.uint8).copy()
        for pixel in project_world_points_to_pixels(prior_visual_points, overlay, T_wc, K_color, T_cam_color):
            cv2.circle(overlay, tuple(int(v) for v in pixel), 1, (180, 180, 180), thickness=-1)
        for pixel in project_world_points_to_pixels(tracking_visual_points, overlay, T_wc, K_color, T_cam_color):
            cv2.circle(overlay, tuple(int(v) for v in pixel), 2, (0, 255, 0), thickness=-1)
        left_hand = clip.samples[idx].get("left_hand_world")
        right_hand = clip.samples[idx].get("right_hand_world")
        for point, color, name in (
            (left_hand, (255, 64, 64), "L"),
            (right_hand, (64, 128, 255), "R"),
        ):
            if point is None:
                continue
            pixels = project_world_points_to_pixels(np.asarray([point], dtype=np.float32), overlay, T_wc, K_color, T_cam_color)
            if len(pixels) == 0:
                continue
            x, y = [int(v) for v in pixels[0]]
            cv2.circle(overlay, (x, y), 7, color, thickness=2)
            cv2.putText(overlay, name, (x + 6, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
        frame_key = str(clip.samples[idx].get("prefix", idx))
        cv2.imwrite(str(overlay_dir / f"{frame_key}.png"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))


def save_clip_snapshot(
    clip: ClipInfo,
    clip_points: np.ndarray,
    clip_colors: np.ndarray,
    scene_points: np.ndarray,
    scene_colors: np.ndarray,
    scene_graph: DynamicSceneGraph,
    out_dir: Path,
    save_visualization: bool = False,
) -> None:
    """Persist per-clip reconstruction and graph snapshots."""
    clip_dir = out_dir / f"clip_{clip.clip_id:03d}"
    clip_dir.mkdir(parents=True, exist_ok=True)
    write_point_cloud_snapshot(clip_dir / "clip_reconstruction.ply", clip_points, clip_colors)
    write_point_cloud_snapshot(clip_dir / "scene_after_clip.ply", scene_points, scene_colors)
    scene_graph.save(str(clip_dir / "scene_graph"))
    if save_visualization:
        save_graph_visualization(scene_graph, scene_points, scene_colors, clip_dir, clip.clip_id)


def load_clip_frames(clip: ClipInfo, K_color: np.ndarray, D_color: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """Load RGB, raw depth, and poses for all frames in a clip."""
    rgb_list, depth_raw_list, T_wc_list = [], [], []
    for sample in tqdm(clip.samples, desc="  loading", leave=False):
        bgr = cv2.imread(str(sample["rgb_path"]))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb_list.append(undistort_rgb(rgb, K_color, D_color))
        depth_raw_list.append(cv2.imread(str(sample["depth_path"]), cv2.IMREAD_ANYDEPTH))
        T_wc_list.append(np.asarray(sample.get("rgb_camera_pose_matrix", load_extrinsic(sample["pose_path"])), dtype=np.float32))
    return rgb_list, depth_raw_list, T_wc_list


def load_clip_frames_with_dual_poses(
    clip: ClipInfo,
    K_color: np.ndarray,
    D_color: np.ndarray,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """Load RGB, raw depth, and both RGB-time and depth-time camera poses for a clip."""
    rgb_list, depth_raw_list, rgb_T_wc_list, depth_T_wc_list = [], [], [], []
    for sample in tqdm(clip.samples, desc="  loading", leave=False):
        bgr = cv2.imread(str(sample["rgb_path"]))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb_list.append(undistort_rgb(rgb, K_color, D_color))
        depth_raw_list.append(cv2.imread(str(sample["depth_path"]), cv2.IMREAD_ANYDEPTH))
        rgb_T_wc_list.append(np.asarray(sample.get("rgb_camera_pose_matrix", load_extrinsic(sample["pose_path"])), dtype=np.float32))
        depth_T_wc_list.append(np.asarray(sample.get("depth_camera_pose_matrix", load_extrinsic(sample["pose_path"])), dtype=np.float32))
    return rgb_list, depth_raw_list, rgb_T_wc_list, depth_T_wc_list


def load_clip_rgb_pose_frames(clip: ClipInfo, K_color: np.ndarray, D_color: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Load only RGB frames and poses for replay after depth precomputation."""
    rgb_list, T_wc_list = [], []
    for sample in tqdm(clip.samples, desc="  loading", leave=False):
        bgr = cv2.imread(str(sample["rgb_path"]))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb_list.append(undistort_rgb(rgb, K_color, D_color))
        T_wc_list.append(np.asarray(sample.get("rgb_camera_pose_matrix", load_extrinsic(sample["pose_path"])), dtype=np.float32))
    return rgb_list, T_wc_list


def load_clip_depth_raw_frames(clip: ClipInfo) -> List[np.ndarray]:
    """Load only raw depth frames for one clip."""
    return [cv2.imread(str(sample["depth_path"]), cv2.IMREAD_ANYDEPTH) for sample in clip.samples]


def load_clip_native_frames(
    clip: ClipInfo,
    K_color: Optional[np.ndarray] = None,
    D_color: Optional[np.ndarray] = None,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """Load RGB, raw depth, and camera poses for native depth-camera reconstruction.

    When intrinsics are provided, RGB is undistorted into the same pinhole color
    grid used by native depth-to-color projection and SAM masks.
    """
    rgb_list, depth_raw_list, T_wc_list = [], [], []
    for sample in tqdm(clip.samples, desc="  loading", leave=False):
        bgr = cv2.imread(str(sample["rgb_path"]))
        if bgr is None:
            raise FileNotFoundError(f"Could not read RGB frame: {sample['rgb_path']}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if K_color is not None and D_color is not None:
            rgb = undistort_rgb(rgb, K_color, D_color)
        rgb_list.append(rgb)
        depth_raw = cv2.imread(str(sample["depth_path"]), cv2.IMREAD_ANYDEPTH)
        if depth_raw is None:
            raise FileNotFoundError(f"Could not read depth frame: {sample['depth_path']}")
        depth_raw_list.append(depth_raw)
        T_wc_list.append(np.asarray(sample.get("depth_camera_pose_matrix", load_extrinsic(sample["pose_path"])), dtype=np.float32))
    return rgb_list, depth_raw_list, T_wc_list


def compute_clip_sensor_depths(
    depth_raw_list: Sequence[np.ndarray],
    K_depth: np.ndarray,
    K_color: np.ndarray,
    D_color: np.ndarray,
    T_depth_color: np.ndarray,
    color_w: int,
    color_h: int,
) -> List[np.ndarray]:
    """Prepare one aligned metric depth map per frame from the raw depth images."""
    return [
        prepare_sensor_depth_for_lifting(
            depth_raw,
            K_depth,
            T_depth_color,
            K_color,
            D_color,
            color_w,
            color_h,
        )
        for depth_raw in depth_raw_list
    ]


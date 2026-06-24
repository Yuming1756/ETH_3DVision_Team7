# ======================================================================
# graph.py section: 14_geometry_update.py
# Geometry updates, SE3 utilities, trajectory smoothing, articulation application
#
# NOTE: not a standalone module. Loaded by graph.py into a shared
# namespace via exec(); names from earlier sections are in scope here.
# ======================================================================
from __future__ import annotations
def collect_mask_lifted_geometries(
    masks: Sequence[np.ndarray],
    hand_masks: Sequence[np.ndarray],
    frame_valids: Optional[Sequence[bool]],
    clip_depths: Optional[Sequence[np.ndarray]],
    depth_raw_list: Optional[Sequence[np.ndarray]],
    rgb_list: Sequence[np.ndarray],
    rgb_T_wc_list: Sequence[np.ndarray],
    K_color: np.ndarray,
    K_depth: Optional[np.ndarray],
    T_cam_color: np.ndarray,
    T_cam_depth: Optional[np.ndarray],
    depth_grid: Optional[np.ndarray],
    use_da3: bool,
    point_color: Optional[Sequence[float]] = None,
    pre_eroded_masks: bool = False,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Lift one tracked mask sequence into per-frame world-space point clouds.

    Two depth sources are supported and lift through different cameras:

    - **DA3 (``use_da3=True``)**: depth is estimated directly from the color
      image, lives at color resolution. For each masked color pixel we read
      ``clip_depths[idx]`` and back-project through ``K_color``, transforming
      to world via ``rgb_T_wc @ T_cam_color``.
    - **Native (``use_da3=False``)**: depth comes from the dedicated depth
      sensor, lives at the depth resolution in its own frame. We walk depth
      pixels, back-project through ``K_depth``, transform to world via
      ``rgb_T_wc @ T_cam_depth``, and use the color SAM mask only to test
      membership (a depth pixel is kept iff its projection into color falls
      inside the mask). This keeps the spoon points at native depth-sensor
      precision instead of round-tripping through ``align_depth_to_color``.

    Either ``clip_depths`` or ``depth_raw_list`` + ``depth_grid`` must be
    provided depending on ``use_da3``.
    """
    # Erode the silhouette by a small band before lifting to drop boundary
    # depth-bleed pixels (sensor "mixed pixels" at the edge return averaged
    # foreground/background depths -> hanging points in 3D). 3 px is enough to
    # peel off the bleed without losing meaningful surface coverage.
    erode_kernel = np.ones((3, 3), np.uint8)
    erode_iterations = 1
    # Dilate the hand mask before subtracting it from the object mask. SAM3's
    # hand silhouette is tight against the visible hand; the door/hand boundary
    # is exactly where pixels can be misattributed in either direction. A small
    # buffer absorbs that error so arm/finger pixels at hand depth do not get
    # lifted as door geometry.
    hand_dilate_kernel = np.ones((3, 3), np.uint8)
    hand_dilate_iterations = 2  # ~5 px buffer around the hand silhouette

    per_frame = []
    for idx in range(len(masks)):
        if frame_valids is not None and idx < len(frame_valids) and not frame_valids[idx]:
            per_frame.append((np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)))
            continue
        hand_mask_arr = np.asarray(hand_masks[idx], dtype=np.uint8)
        if np.any(hand_mask_arr):
            hand_mask_arr = cv2.dilate(hand_mask_arr, hand_dilate_kernel, iterations=hand_dilate_iterations)
        mask = masks[idx].copy() & ~hand_mask_arr.astype(bool)
        if not np.any(mask):
            per_frame.append((np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)))
            continue
        if not pre_eroded_masks:
            eroded = cv2.erode(mask.astype(np.uint8), erode_kernel, iterations=erode_iterations).astype(bool)
            # Fall back to the un-eroded mask if erosion wipes out the silhouette
            # (e.g. thin masks like handles).
            if np.any(eroded):
                mask = eroded
        if use_da3:
            assert clip_depths is not None, "DA3 lifting requires per-frame color-grid depth maps."
            pts, cols = lift_mask_to_world(
                mask,
                np.asarray(clip_depths[idx], dtype=np.float32),
                rgb_list[idx],
                K_color,
                rgb_T_wc_list[idx],
                T_cam_color,
                depth_max=DEPTH_MAX,
            )
        else:
            assert depth_raw_list is not None and depth_grid is not None and K_depth is not None and T_cam_depth is not None, (
                "Native lifting requires raw depth, depth_grid, K_depth, and T_cam_depth."
            )
            pts, cols = lift_depth_frame_to_world_with_rgb(
                depth_raw=depth_raw_list[idx],
                rgb=rgb_list[idx],
                depth_grid=depth_grid,
                K_color=K_color,
                T_wc=rgb_T_wc_list[idx],
                T_cam_depth=T_cam_depth,
                T_cam_color=T_cam_color,
                D_color=None,
                exclusion_mask=~mask,
                aligned_color_depth=None,
                pixel_stride=1,
                depth_max=DEPTH_MAX,
            )
        if point_color is not None and len(pts):
            cols = constant_point_colors(pts, point_color)
        per_frame.append((pts, cols))
    return per_frame


def select_final_geometry_indices(per_frame_geometry: Sequence[Tuple[np.ndarray, np.ndarray]], max_frames: int = FINAL_GEOMETRY_MAX_FRAMES) -> List[int]:
    """Keep only the last few valid frames so moved objects end up only in their final pose."""
    valid_indices = [idx for idx, (pts, _) in enumerate(per_frame_geometry) if len(pts)]
    return valid_indices[-max_frames:]


def select_initial_geometry_indices(per_frame_geometry: Sequence[Tuple[np.ndarray, np.ndarray]], max_frames: int = INITIAL_GEOMETRY_MAX_FRAMES) -> List[int]:
    """Keep only the first few valid frames so LostFound starts from the clip's initial pose."""
    valid_indices = [idx for idx, (pts, _) in enumerate(per_frame_geometry) if len(pts)]
    return valid_indices[:max_frames]


def geometry_from_indices(
    per_frame_geometry: Sequence[Tuple[np.ndarray, np.ndarray]],
    indices: Sequence[int],
) -> Tuple[np.ndarray, np.ndarray]:
    """Merge selected per-frame geometries into one point cloud."""
    point_blocks = [per_frame_geometry[idx][0] for idx in indices if len(per_frame_geometry[idx][0])]
    color_blocks = [per_frame_geometry[idx][1] for idx in indices if len(per_frame_geometry[idx][1])]
    return stack_point_blocks(point_blocks, color_blocks)


def run_lostfound_geometry_update(
    clip: ClipInfo,
    label: str,
    previous_node,
    inactive_prior_points: np.ndarray,
    inactive_prior_colors: np.ndarray,
    rgb_list: Sequence[np.ndarray],
    T_wc_list: Sequence[np.ndarray],
    K_color: np.ndarray,
    T_cam_color: np.ndarray,
) -> Tuple[Optional[dict], str, dict]:
    """Run LostFound from the inactive-clip prior only; scene-graph fallback is disabled."""
    debug = {
        "label": str(label),
        "clip_id": int(clip.clip_id),
        "status": "failed",
        "failure_reason": "no_prior_candidates",
        "prior_attempts": [],
        "selected_prior_source": "",
        "inactive_prior_point_count": int(len(inactive_prior_points)) if inactive_prior_points is not None else 0,
        "scene_graph_prior_point_count": int(len(previous_node.points)) if previous_node is not None and len(previous_node.points) else 0,
    }
    candidates: List[Tuple[str, np.ndarray, np.ndarray]] = []
    if inactive_prior_points is not None and len(inactive_prior_points):
        candidates.append(
            (
                "inactive_clip",
                np.asarray(inactive_prior_points, dtype=np.float32),
                np.asarray(inactive_prior_colors, dtype=np.float32),
            )
        )
    elif previous_node is not None and len(previous_node.points):
        debug["failure_reason"] = "inactive_prior_missing_scene_graph_fallback_disabled"
        print(
            f"[LostFound][Error] clip={clip.clip_id} label='{label}': inactive prior missing; "
            "scene-graph prior fallback is disabled."
        )

    for prior_source, prior_points, prior_colors in candidates:
        result, attempt_debug = run_lostfound_pose_tracking(
            clip,
            label,
            prior_points,
            prior_colors,
            rgb_list,
            T_wc_list,
            K_color,
            T_cam_color,
        )
        attempt_debug["prior_source"] = prior_source
        debug["prior_attempts"].append(attempt_debug)
        if result is not None:
            debug["status"] = "success"
            debug["failure_reason"] = ""
            debug["selected_prior_source"] = prior_source
            return result, prior_source, debug
    return None, "", debug


def save_lostfound_failure_debug(
    clip: ClipInfo,
    out_dir: Path,
    element_label: str,
    object_label: str,
    element_debug: dict,
    object_debug: Optional[dict] = None,
) -> None:
    """Persist a compact JSON artifact when LostFound blocks an active clip."""
    clip_dir = out_dir / f"clip_{clip.clip_id:03d}"
    clip_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "clip_id": int(clip.clip_id),
        "frame_start": int(clip.frame_start),
        "frame_end": int(clip.frame_end),
        "element": str(element_label),
        "object": str(object_label),
        "element_tracking_debug": element_debug,
        "object_tracking_debug": object_debug,
        "element_lostfound_debug": element_debug,
        "object_lostfound_debug": object_debug,
    }
    with open(clip_dir / "tracking_failure.json", "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    with open(clip_dir / "lostfound_failure.json", "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(f"[Tracking] saved failure debug to {clip_dir / 'tracking_failure.json'}")


def export_tracking_debug_assets(
    clip: ClipInfo,
    out_dir: Path,
    label: str,
    prior_points: np.ndarray,
    final_points: Optional[np.ndarray] = None,
    debug_payload: Optional[dict] = None,
    asset_prefix: str = "object",
) -> None:
    """Persist lightweight tracking debug for both rigid and articulated paths."""
    # Disabled: tracking_debug is kept minimal (only the artipoint
    # <label>_tracks / <label>_masks overlays written by run_artipoint).
    return
    clip_dir = out_dir / f"clip_{clip.clip_id:03d}"
    safe_label = re.sub(r"[^a-z0-9]+", "_", str(label).strip().lower()).strip("_") or "item"
    asset_dir = clip_dir / "tracking_debug" / f"{asset_prefix}_{safe_label}"
    asset_dir.mkdir(parents=True, exist_ok=True)

    prior_arr = np.asarray(prior_points, dtype=np.float32)
    if len(prior_arr):
        write_point_cloud_snapshot(
            asset_dir / "prior_points.ply",
            prior_arr,
            constant_point_colors(prior_arr, (0.75, 0.75, 0.75)),
        )

    if final_points is not None:
        final_arr = np.asarray(final_points, dtype=np.float32)
        if len(final_arr):
            write_point_cloud_snapshot(
                asset_dir / "final_points.ply",
                final_arr,
                constant_point_colors(final_arr, (0.10, 0.90, 0.20)),
            )

    left_pts = [
        np.asarray(sample.get("left_hand_world"), dtype=np.float32)
        for sample in clip.samples
        if sample.get("left_hand_world") is not None
    ]
    right_pts = [
        np.asarray(sample.get("right_hand_world"), dtype=np.float32)
        for sample in clip.samples
        if sample.get("right_hand_world") is not None
    ]
    if left_pts:
        left_arr = np.stack(left_pts).astype(np.float32)
        write_point_cloud_snapshot(
            asset_dir / "left_wrist_trajectory.ply",
            left_arr,
            constant_point_colors(left_arr, (1.0, 0.35, 0.35)),
        )
    if right_pts:
        right_arr = np.stack(right_pts).astype(np.float32)
        write_point_cloud_snapshot(
            asset_dir / "right_wrist_trajectory.ply",
            right_arr,
            constant_point_colors(right_arr, (0.35, 0.55, 1.0)),
        )

    summary = {
        "clip_id": int(clip.clip_id),
        "frame_start": int(clip.frame_start),
        "frame_end": int(clip.frame_end),
        "label": str(label),
        "asset_prefix": str(asset_prefix),
        "prior_point_count": int(len(prior_arr)),
        "final_point_count": int(len(final_points)) if final_points is not None else 0,
        "debug_payload": debug_payload or {},
    }
    with open(asset_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)


def summarize_lostfound_result(result: Optional[dict], prior_source: str) -> Optional[dict]:
    """Serialize the key LostFound update stats for debugging and event logs."""
    if result is None:
        return None
    return {
        "prior_source": prior_source,
        "hand": result["hand"],
        "hand_anchor_mode": result.get("hand_anchor_mode"),
        "pose_updates": int(result["pose_updates"]),
        "translation_m": round(float(result["translation_m"]), 4),
        "rotation_deg": round(float(result["rotation_deg"]), 2),
        "used_frames": [int(frame_id) for frame_id in result["used_frames"]],
    }


def _skew3(v: np.ndarray) -> np.ndarray:
    """Return the 3x3 skew-symmetric matrix corresponding to a 3-vector."""
    return np.array(
        [
            [0.0, -float(v[2]), float(v[1])],
            [float(v[2]), 0.0, -float(v[0])],
            [-float(v[1]), float(v[0]), 0.0],
        ],
        dtype=np.float64,
    )


def se3_log(T: np.ndarray) -> np.ndarray:
    """SE(3) matrix logarithm. Returns 6-vector [omega; v]."""
    T = np.asarray(T, dtype=np.float64)
    R = T[:3, :3]
    t = T[:3, 3]
    cos_theta = float(np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0))
    theta = float(np.arccos(cos_theta))
    if theta < 1e-8:
        omega = 0.5 * np.array(
            [R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]],
            dtype=np.float64,
        )
        v = t.astype(np.float64)
        return np.concatenate([omega, v])
    if abs(theta - np.pi) < 1e-3:
        # rotation near pi: extract axis from symmetric part
        M = (R + np.eye(3)) * 0.5
        idx = int(np.argmax(np.diag(M)))
        axis = M[:, idx]
        axis = axis / max(np.linalg.norm(axis), 1e-12)
        omega = theta * axis
    else:
        omega = (theta / (2.0 * np.sin(theta))) * np.array(
            [R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]],
            dtype=np.float64,
        )
    K = _skew3(omega)
    A = np.sin(theta) / theta
    B = (1.0 - np.cos(theta)) / (theta * theta)
    J_inv = np.eye(3) - 0.5 * K + (1.0 / (theta * theta)) * (1.0 - A / (2.0 * B)) * (K @ K)
    v = J_inv @ t
    return np.concatenate([omega, v])


def se3_exp(xi: np.ndarray) -> np.ndarray:
    """SE(3) matrix exponential. Takes 6-vector [omega; v] and returns a 4x4 matrix."""
    xi = np.asarray(xi, dtype=np.float64)
    omega = xi[:3]
    v = xi[3:]
    theta = float(np.linalg.norm(omega))
    T = np.eye(4, dtype=np.float64)
    if theta < 1e-8:
        T[:3, :3] = np.eye(3) + _skew3(omega)
        T[:3, 3] = v
        return T
    K = _skew3(omega)
    K2 = K @ K
    R = np.eye(3) + (np.sin(theta) / theta) * K + ((1.0 - np.cos(theta)) / (theta * theta)) * K2
    J = (
        np.eye(3)
        + ((1.0 - np.cos(theta)) / (theta * theta)) * K
        + ((theta - np.sin(theta)) / (theta ** 3)) * K2
    )
    T[:3, :3] = R
    T[:3, 3] = J @ v
    return T


def _build_difference_matrix(size: int, order: int):
    """Sparse n-th order difference matrix on a length-``size`` sequence.

    Ported from ArtiPoint (artipoint/track/arti_estimator.py).
    Order 1 = velocity, 2 = acceleration, 3 = jerk.
    """
    if order == 1:
        return sp.diags(
            [-np.ones(size - 1), np.ones(size - 1)],
            [0, 1],
            shape=(size - 1, size),
            format="csr",
        )
    if order == 2:
        return sp.diags(
            [np.ones(size - 2), -2 * np.ones(size - 2), np.ones(size - 2)],
            [0, 1, 2],
            shape=(size - 2, size),
            format="csr",
        )
    if order == 3:
        if size < 4:
            return sp.csr_matrix((0, size))
        return sp.diags(
            [-np.ones(size - 3), 3 * np.ones(size - 3), -3 * np.ones(size - 3), np.ones(size - 3)],
            [0, 1, 2, 3],
            shape=(size - 3, size),
            format="csr",
        )
    raise ValueError("Order must be 1, 2, or 3")


def smooth_trajectory_optimization(
    points: np.ndarray,
    visibility: Optional[np.ndarray] = None,
    lambda_vel: float = 0.1,
    lambda_jerk: float = 1.0,
) -> np.ndarray:
    """Smooth 3D point trajectories with velocity + jerk regularization.

    Ported from ArtiPoint (CoRL 2025) — their Eq. (1) data-fidelity +
    velocity + jerk penalty solved as a sparse linear system per dimension.

    ``points`` has shape (T, N, 3). ``visibility`` shape (T, N) acts as a
    per-sample fidelity weight (1 = observed, 0 = ignore). If omitted, every
    sample is fully trusted.
    """
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 3 or points.shape[2] != 3:
        raise ValueError("points must have shape (T, N, 3)")
    T_frames, N_points, _ = points.shape
    if visibility is None:
        visibility = np.ones((T_frames, N_points), dtype=np.float64)
    visibility = np.asarray(visibility, dtype=np.float64)
    if visibility.shape != (T_frames, N_points):
        raise ValueError("visibility shape must match points")

    if lambda_jerk > 0 and T_frames < 4:
        lambda_jerk = 0.0
    if lambda_vel > 0 and T_frames < 2:
        lambda_vel = 0.0

    points_flat = np.transpose(points, (1, 0, 2)).reshape(-1, 3)
    visibility_flat = np.transpose(visibility).flatten()

    NT = N_points * T_frames
    V = sp.diags(visibility_flat, format="csc")
    I_N = sp.identity(N_points, format="csc")

    L_vel = sp.csc_matrix((NT, NT))
    if lambda_vel > 0 and T_frames >= 2:
        D1 = _build_difference_matrix(T_frames, 1)
        L_vel = sp.kron(I_N, D1.T @ D1, format="csc")

    L_jerk = sp.csc_matrix((NT, NT))
    if lambda_jerk > 0 and T_frames >= 4:
        D3 = _build_difference_matrix(T_frames, 3)
        L_jerk = sp.kron(I_N, D3.T @ D3, format="csc")

    A = (V + lambda_vel * L_vel + lambda_jerk * L_jerk).tocsc()

    smoothed = np.zeros_like(points_flat)
    for dim in range(3):
        b = V @ points_flat[:, dim]
        try:
            smoothed[:, dim] = spsolve(A, b)
        except Exception as exc:  # noqa: BLE001
            print(f"[smooth_trajectory_optimization] dim {dim} failed: {exc}")
            smoothed[:, dim] = points_flat[:, dim]

    return np.transpose(smoothed.reshape(N_points, T_frames, 3), (1, 0, 2))


def smooth_pose_history_translations(
    pose_history: Sequence[dict],
    lambda_vel: float = 0.1,
    lambda_jerk: float = 1.0,
) -> List[dict]:
    """Pose-history smoother: applies the trajectory smoother to the translation
    part of every pose (rotations are left untouched). Returns a new pose
    history list with smoothed translations.
    """
    history = list(pose_history)
    if len(history) < 4:
        return [dict(s) for s in history]
    translations = np.array([np.asarray(s["pose"], dtype=np.float64)[:3, 3] for s in history], dtype=np.float64)
    # Shape (T, 1, 3) — single point trajectory.
    pts = translations.reshape(translations.shape[0], 1, 3)
    smoothed = smooth_trajectory_optimization(pts, lambda_vel=lambda_vel, lambda_jerk=lambda_jerk)
    smoothed = smoothed.reshape(-1, 3)
    out: List[dict] = []
    for sample, t_new in zip(history, smoothed):
        pose = np.asarray(sample["pose"], dtype=np.float64).copy()
        pose[:3, 3] = t_new
        new_sample = dict(sample)
        new_sample["pose"] = pose.astype(np.float32)
        out.append(new_sample)
    return out


def _so3_log_axis_angle(R: np.ndarray) -> np.ndarray:
    """SO(3) log: 3x3 rotation -> 3-vector (axis * angle)."""
    R = np.asarray(R, dtype=np.float64)
    cos_theta = float(np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0))
    theta = float(np.arccos(cos_theta))
    if theta < 1e-8:
        return 0.5 * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]], dtype=np.float64)
    if abs(theta - np.pi) < 1e-3:
        M = (R + np.eye(3)) * 0.5
        idx = int(np.argmax(np.diag(M)))
        axis = M[:, idx]
        axis = axis / max(np.linalg.norm(axis), 1e-12)
        return theta * axis
    return (theta / (2.0 * np.sin(theta))) * np.array(
        [R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]],
        dtype=np.float64,
    )


def _so3_exp(omega: np.ndarray) -> np.ndarray:
    """SO(3) exp: axis * angle vector -> 3x3 rotation."""
    omega = np.asarray(omega, dtype=np.float64)
    theta = float(np.linalg.norm(omega))
    if theta < 1e-8:
        return np.eye(3) + _skew3(omega)
    K = _skew3(omega)
    return np.eye(3) + (np.sin(theta) / theta) * K + ((1.0 - np.cos(theta)) / (theta * theta)) * (K @ K)


def fit_articulation_model(pose_history: Sequence[dict]) -> dict:
    """Fit a 1-DoF articulation (revolute / prismatic) to a SE(3) pose trajectory.

    Decouples rotation-axis estimation from translation noise, which is
    important when the hinge is offset from the pose origin (the door
    centroid translates a lot even for a pure revolute, and a naive 6-vector
    SVD picks up that translation pattern as the dominant direction).

    Algorithm:
      1) Per-frame deltas ``D_m = T_m T_0^{-1}`` (world-frame).
      2) Rotation-only axis: SVD of the stacked ``omega_m = SO(3)_log(R_m)``
         to get a unit axis ``n`` and signed per-frame angles
         ``theta_m = omega_m . n``.
      3) Hinge point ``p_h`` via least-squares: stack the system
         ``(I - R_m) p_h = t_m`` for all frames and solve in world coords.
      4) Reconstruct pure-revolute deltas:
         ``R_m = exp(theta_m * n)``, ``t_m = (I - R_m) p_h``.
      5) Joint type: revolute by default; falls back to prismatic if the
         rotations are all near identity (i.e. motion is pure translation).

    Returns world-frame ``axis_direction`` (= n) and ``axis_point`` (= p_h)
    plus per-frame angles and the reconstructed pose history. Raises
    ``ValueError`` on degenerate input.
    """
    if pose_history is None or len(pose_history) < 3:
        raise ValueError(
            f"fit_articulation_model needs at least 3 pose samples; got {0 if pose_history is None else len(pose_history)}"
        )

    sorted_history = sorted(pose_history, key=lambda s: int(s.get("frame", 0)))
    poses = [np.asarray(s["pose"], dtype=np.float64) for s in sorted_history]
    frames = [int(s["frame"]) for s in sorted_history]
    clip_ids = [int(s.get("clip_id", -1)) for s in sorted_history]

    reference_pose = poses[0]
    reference_pose_inv = np.linalg.inv(reference_pose)

    omegas: List[np.ndarray] = []
    rotations: List[np.ndarray] = []
    translations: List[np.ndarray] = []
    for pose in poses:
        delta = pose @ reference_pose_inv
        R = delta[:3, :3]
        t = delta[:3, 3]
        omegas.append(_so3_log_axis_angle(R))
        rotations.append(R)
        translations.append(t)
    omegas_arr = np.stack(omegas, axis=0)  # M x 3
    rotations_arr = np.stack(rotations, axis=0)  # M x 3 x 3
    translations_arr = np.stack(translations, axis=0)  # M x 3

    rotation_magnitudes = np.linalg.norm(omegas_arr, axis=1)
    max_rot = float(np.max(rotation_magnitudes))

    if max_rot < 1e-3:
        # Pure-translation case: prismatic joint.
        # SVD on translations to find the dominant direction.
        U, S_t, Vt_t = np.linalg.svd(translations_arr, full_matrices=False)
        axis_direction = Vt_t[0].astype(np.float64)
        axis_direction = axis_direction / max(np.linalg.norm(axis_direction), 1e-12)
        thetas = translations_arr @ axis_direction
        if float(np.sum(thetas)) < 0.0:
            axis_direction = -axis_direction
            thetas = -thetas
        joint_type = "prismatic"
        axis_point = np.zeros(3, dtype=np.float64)
        singular_ratio = float(S_t[1] / S_t[0]) if len(S_t) > 1 and S_t[0] > 1e-12 else 0.0
    else:
        # PCA on rotation axes (treating omega_m as a 3D vector). The dominant
        # principal component is the hinge axis n in world frame.
        U, S_r, Vt_r = np.linalg.svd(omegas_arr, full_matrices=False)
        axis_direction = Vt_r[0].astype(np.float64)
        axis_direction = axis_direction / max(np.linalg.norm(axis_direction), 1e-12)
        thetas = omegas_arr @ axis_direction
        if float(np.sum(thetas)) < 0.0:
            axis_direction = -axis_direction
            thetas = -thetas

        # Solve for the hinge point: t_m = (I - R_m) p_h, least-squares over m.
        A = np.zeros((3, 3), dtype=np.float64)
        b = np.zeros(3, dtype=np.float64)
        for R, t in zip(rotations_arr, translations_arr):
            IR = np.eye(3) - R
            A += IR.T @ IR
            b += IR.T @ t
        try:
            axis_point, _residuals, _rank, _sv = np.linalg.lstsq(A, b, rcond=None)
        except np.linalg.LinAlgError:
            axis_point = np.zeros(3, dtype=np.float64)
        # axis_point is in the reference (= initial pose) coordinate frame's
        # contribution to the world translation: it IS the hinge in world
        # frame because our deltas are world-frame and (I - R_m) p_h = t_m has
        # p_h naturally in world coords.

        joint_type = "revolute"
        singular_ratio = float(S_r[1] / S_r[0]) if len(S_r) > 1 and S_r[0] > 1e-12 else 0.0

    # Reconstruct deltas as pure 1-DoF motion around (axis_direction, axis_point)
    reconstructed_pose_history: List[dict] = []
    residuals_t: List[float] = []
    for frame, clip_id, observed, theta in zip(frames, clip_ids, poses, thetas):
        if joint_type == "prismatic":
            delta_recon = np.eye(4, dtype=np.float64)
            delta_recon[:3, 3] = axis_direction * float(theta)
        else:
            R_recon = _so3_exp(axis_direction * float(theta))
            t_recon = (np.eye(3) - R_recon) @ axis_point
            delta_recon = np.eye(4, dtype=np.float64)
            delta_recon[:3, :3] = R_recon
            delta_recon[:3, 3] = t_recon
        pose_recon = delta_recon @ reference_pose
        reconstructed_pose_history.append(
            {
                "frame": int(frame),
                "pose": pose_recon.astype(np.float32),
                "source": "articulation_fit",
                "confidence": 1.0,
                "clip_id": int(clip_id),
                "theta": float(theta),
            }
        )
        residuals_t.append(float(np.linalg.norm(observed[:3, 3] - pose_recon[:3, 3])))

    return {
        "xi": (axis_direction.tolist() + axis_point.tolist()) if joint_type == "revolute" else (axis_direction.tolist() + [0.0, 0.0, 0.0]),
        "thetas": [float(t) for t in thetas],
        "axis_direction": axis_direction.astype(np.float32).tolist(),
        "axis_point": axis_point.astype(np.float32).tolist(),
        "joint_type": joint_type,
        "reconstructed_pose_history": reconstructed_pose_history,
        "mean_residual_m": float(np.mean(residuals_t)),
        "max_residual_m": float(np.max(residuals_t)),
        "n_frames": len(sorted_history),
        "singular_ratio": singular_ratio,
    }


def is_articulated_object_motion(
    motion_coupling: Optional[dict],
    element_label: str,
    object_label: str,
) -> bool:
    """Decide whether this interaction should use the 1-DoF articulated path."""
    if motion_coupling is None:
        raise RuntimeError(
            f"[MotionCoupling][Error] missing motion_coupling for '{element_label}' / '{object_label}'; "
            "heuristic articulated-motion fallback is disabled."
        )
    interaction_type = str(motion_coupling.get("interaction_type", "")).strip().lower()
    if not interaction_type:
        raise RuntimeError(
            f"[MotionCoupling][Error] empty interaction_type for '{element_label}' / '{object_label}'; "
            "heuristic articulated-motion fallback is disabled."
        )
    return interaction_type == "articulated_object"


def apply_articulation_constraint(
    motion_record: Optional[dict],
    label: str = "",
    element_label: str = "",
    object_label: str = "",
    motion_coupling: Optional[dict] = None,
) -> Optional[dict]:
    """Replace a motion record's pose history with the articulation-constrained fit.

    Only runs the ArtiPoint-style 1-DoF fit when Qwen's coupling output (or the
    label fallback) says the pair is articulated. For free 6-DoF motion the
    motion record is returned unchanged.

    On articulated motion with a failing fit (insufficient frames, degenerate
    motion, numerical issues) prints an error and returns ``None`` — by user
    request we deliberately do NOT fall back to the raw LostFound output.
    """
    if motion_record is None:
        return None
    if not is_articulated_object_motion(motion_coupling, element_label, object_label):
        print(
            f"[ArticulationFit] '{label}': skipping fit — '{element_label}' / '{object_label}' "
            f"not classified as articulated; keeping raw LostFound pose history."
        )
        return motion_record
    history = list(motion_record.get("pose_history", []) or [])
    if len(history) < 3:
        print(
            f"[ArticulationFit][Error] '{label}': need >=3 pose samples; got {len(history)}. Dropping motion record."
        )
        return None
    # Wrist-anchored translation (translation_hand = wrist - R · delta_obj) means
    # the first 1-3 frames can encode the wrist still snapping onto the handle
    # rather than real articulation. Detect those frames as having centroid jumps
    # much larger than the steady-state median, and drop them.
    trim_history = list(history)
    if len(trim_history) >= 6:
        centroids = np.stack(
            [np.asarray(item["pose"], dtype=np.float64)[:3, 3] for item in trim_history],
            axis=0,
        )
        jumps = np.linalg.norm(np.diff(centroids, axis=0), axis=1)
        if jumps.size >= 4:
            steady_median = float(np.median(jumps[max(2, len(jumps) // 3):]))
            jump_cap = max(steady_median * 3.0, 0.04)
            drop_n = 0
            for jump_val in jumps:
                if jump_val > jump_cap and drop_n < 4:
                    drop_n += 1
                else:
                    break
            if drop_n > 0:
                print(
                    f"[ArticulationFit] '{label}': trimming {drop_n} leading frame(s) "
                    f"with centroid jump >{jump_cap*100:.1f} cm (wrist-snap artifact); "
                    f"steady median jump {steady_median*100:.2f} cm."
                )
                trim_history = trim_history[drop_n:]
    if len(trim_history) < 3:
        print(
            f"[ArticulationFit][Error] '{label}': after trimming wrist-snap leading frames "
            f"only {len(trim_history)} remain. Dropping motion record."
        )
        return None
    history = trim_history
    # ArtiPoint Eq. (1) translation smoother: velocity + jerk penalties on the
    # per-frame translations. Stabilises the fit against LostFound noise that
    # grows over the clip.
    try:
        smoothed_history = smooth_pose_history_translations(history)
    except Exception as exc:  # noqa: BLE001
        print(f"[ArticulationFit][Error] '{label}': smoothing failed ({exc}). Dropping motion record.")
        return None
    try:
        fit = fit_articulation_model(smoothed_history)
    except Exception as exc:  # noqa: BLE001 - log and propagate as None
        print(f"[ArticulationFit][Error] '{label}': {exc}. Dropping motion record.")
        return None
    # Prior-based joint-type override. ArtiPoint classifies type from the screw PITCH,
    # which is ~0 for BOTH a hinge (revolute) and a sideways-sliding drawer (prismatic),
    # so a short, slightly-noisy drawer slide is fit as a small-angle arc and mislabelled
    # revolute. When the moving-part label is clearly prismatic (drawer/tray/slide) and
    # the fit came back revolute with near-zero pitch, trust the prior: relabel prismatic
    # and set the axis to the net translation direction (axis_point is free for prismatic).
    _label_l = str(label or "").lower()
    _predicted_prismatic = any(k in _label_l for k in ("drawer", "tray", "pull-out", "pullout", "slide"))
    if _predicted_prismatic and str(fit.get("joint_type", "")).lower() == "revolute" and abs(float(fit.get("pitch", 0.0))) < 0.2:
        _poses = fit.get("reconstructed_pose_history") or []
        if len(_poses) >= 2:
            _p0 = np.asarray(_poses[0], dtype=np.float64)[:3, 3]
            _p1 = np.asarray(_poses[-1], dtype=np.float64)[:3, 3]
            _disp = _p1 - _p0
            _dn = float(np.linalg.norm(_disp))
            if _dn > 1e-6:
                fit["joint_type"] = "prismatic"
                fit["axis_direction"] = (_disp / _dn).astype(np.float32).tolist()
                fit["axis_point"] = [0.0, 0.0, 0.0]
                print(
                    f"[ArticulationFit] '{label}': prismatic-prior override revolute->prismatic "
                    f"(pitch={abs(float(fit.get('pitch', 0.0))):.3f}, net_disp={_dn*100:.1f}cm)"
                )
    print(
        f"[ArticulationFit] '{label}': joint={fit['joint_type']} "
        f"axis_dir={np.round(fit['axis_direction'], 3).tolist()} "
        f"axis_pt={np.round(fit['axis_point'], 3).tolist()} "
        f"residual_mean={fit['mean_residual_m']*100:.2f}cm "
        f"residual_max={fit['max_residual_m']*100:.2f}cm "
        f"ratio={fit['singular_ratio']:.4f} n={fit['n_frames']}"
    )
    fitted = dict(motion_record)
    fitted["pose_history"] = fit["reconstructed_pose_history"]
    fitted["articulation_axis_direction"] = fit["axis_direction"]
    fitted["articulation_axis_point"] = fit["axis_point"]
    fitted["articulation_joint_type"] = fit["joint_type"]
    fitted["articulation_thetas"] = fit["thetas"]
    fitted["articulation_residual_mean_m"] = fit["mean_residual_m"]
    fitted["articulation_residual_max_m"] = fit["max_residual_m"]
    fitted["articulation_singular_ratio"] = fit["singular_ratio"]
    fitted["articulation_xi"] = fit["xi"]
    return fitted


def _articulation_summary(motion_record: Optional[dict]) -> Optional[dict]:
    """Compact articulation report suitable for event.json."""
    if motion_record is None:
        return None
    keys = (
        "articulation_joint_type",
        "articulation_axis_direction",
        "articulation_axis_point",
        "articulation_thetas",
        "articulation_residual_mean_m",
        "articulation_residual_max_m",
        "articulation_singular_ratio",
        "articulation_xi",
        "articulation_optimizer",
        "articulation_axis_prior",
        "articulation_pivot_candidates",
        "articulation_pivot_candidate_scores",
        "articulation_chosen_pivot_source",
    )
    if not any(k in motion_record for k in keys):
        return None
    joint_type = motion_record.get("articulation_joint_type")
    axis_dir = motion_record.get("articulation_axis_direction")
    thetas = motion_record.get("articulation_thetas") or []

    # Human-readable description (debug only — does not feed any downstream logic).
    description = None
    if joint_type and isinstance(axis_dir, (list, tuple)) and len(axis_dir) == 3:
        ax = np.asarray(axis_dir, dtype=np.float64)
        norm = float(np.linalg.norm(ax))
        if norm > 1e-6:
            ax = ax / norm
        # Name the dominant world axis direction.
        components = {"+X": ax[0], "-X": -ax[0], "+Y": ax[1], "-Y": -ax[1], "+Z (up)": ax[2], "-Z (down)": -ax[2]}
        dominant = max(components.items(), key=lambda kv: kv[1])
        dominance = float(dominant[1])
        axis_name = (
            dominant[0]
            if dominance > 0.92
            else f"{dominant[0]}-leaning ({dominance:.2f})"
        )
        if str(joint_type).lower() == "revolute":
            sweep_deg = (
                f"{float(np.degrees(min(thetas))):+.1f}° ... {float(np.degrees(max(thetas))):+.1f}°"
                if thetas else "no thetas"
            )
            description = f"revolute hinge around {axis_name} axis, sweep {sweep_deg}"
        elif str(joint_type).lower() == "prismatic":
            sweep_m = (
                f"{float(min(thetas)):+.3f} m ... {float(max(thetas)):+.3f} m"
                if thetas else "no thetas"
            )
            description = f"prismatic slide along {axis_name} axis, sweep {sweep_m}"
        else:
            description = f"{joint_type} along {axis_name}"

    return {
        "joint_type": joint_type,
        "description": description,
        "axis_direction": axis_dir,
        "axis_point": motion_record.get("articulation_axis_point"),
        "thetas": thetas,
        "residual_mean_m": motion_record.get("articulation_residual_mean_m"),
        "residual_max_m": motion_record.get("articulation_residual_max_m"),
        "singular_ratio": motion_record.get("articulation_singular_ratio"),
        "xi": motion_record.get("articulation_xi"),
        "optimizer": motion_record.get("articulation_optimizer"),
        "solver_path": motion_record.get("articulation_solver_path"),
        "fallback_reason": motion_record.get("articulation_fallback_reason"),
        "axis_prior": motion_record.get("articulation_axis_prior"),
        "chosen_pivot_source": motion_record.get("articulation_chosen_pivot_source"),
        "pivot_candidates": motion_record.get("articulation_pivot_candidates"),
        "pivot_candidate_scores": motion_record.get("articulation_pivot_candidate_scores"),
        "n_frames": len(motion_record.get("pose_history", []) or []),
    }


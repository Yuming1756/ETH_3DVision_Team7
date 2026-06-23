# ======================================================================
# graph.py section: 09_articulation_fit.py
# Plane/circle/revolute fitting, GTSAM + joint-optimization articulation fit
#
# NOTE: not a standalone module. Loaded by graph.py into a shared
# namespace via exec(); names from earlier sections are in scope here.
# ======================================================================
from __future__ import annotations
def _fit_plane_3d(positions: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray, float]]:
    """Fit a 3-D plane to N >= 3 points via SVD.

    Returns ``(normal_unit, centroid, planarity_score)`` where ``planarity_score``
    is the ratio of the smallest to the second-smallest singular value of the
    centred point matrix — values << 1 mean the points lie close to a plane,
    values ~ 1 mean the points fill 3-D space (a poor plane fit).
    """
    pts = np.asarray(positions, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3 or pts.shape[0] < 3:
        return None
    finite = np.all(np.isfinite(pts), axis=1)
    pts = pts[finite]
    if pts.shape[0] < 3:
        return None
    centroid = pts.mean(axis=0)
    centered = pts - centroid
    try:
        _, sv, vt = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError:
        return None
    if sv.size < 3:
        return None
    normal = vt[-1]
    n_norm = float(np.linalg.norm(normal))
    if n_norm < 1e-9:
        return None
    normal = normal / n_norm
    planarity = float(sv[-1] / max(sv[-2], 1e-9))
    return normal.astype(np.float64), centroid.astype(np.float64), planarity


def _fit_circle_in_plane(
    positions: np.ndarray,
    plane_normal: np.ndarray,
    plane_point: np.ndarray,
) -> Optional[Tuple[np.ndarray, float, np.ndarray, np.ndarray]]:
    """2-D circle fit inside an arbitrary 3-D plane.

    Returns ``(center_3d, radius, e_u, e_v)`` where ``e_u``, ``e_v`` are an
    orthonormal basis spanning the plane (so positions can be expressed in
    plane-local coords as ``(p - center_3d) . e_u``, ``... e_v``).
    """
    pts = np.asarray(positions, dtype=np.float64)
    n = np.asarray(plane_normal, dtype=np.float64)
    p0 = np.asarray(plane_point, dtype=np.float64)
    finite = np.all(np.isfinite(pts), axis=1)
    pts = pts[finite]
    if pts.shape[0] < 3:
        return None
    n = n / max(float(np.linalg.norm(n)), 1e-9)
    # Orthonormal plane basis. Pick the world axis least aligned with the
    # normal so the cross product stays well-conditioned.
    seed = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e_u = np.cross(n, seed)
    e_u_norm = float(np.linalg.norm(e_u))
    if e_u_norm < 1e-9:
        return None
    e_u = e_u / e_u_norm
    e_v = np.cross(n, e_u)
    # Project to plane-local 2-D
    rel = pts - p0
    u = rel @ e_u
    v = rel @ e_v
    circle = _fit_circle_xy_least_squares(np.stack([u, v], axis=1))
    if circle is None:
        return None
    center_2d, radius = circle
    center_3d = p0 + center_2d[0] * e_u + center_2d[1] * e_v
    return center_3d.astype(np.float64), float(radius), e_u.astype(np.float64), e_v.astype(np.float64)


def _angles_from_plane_projection(
    positions: Sequence[Optional[np.ndarray]],
    center_3d: np.ndarray,
    e_u: np.ndarray,
    e_v: np.ndarray,
) -> Optional[np.ndarray]:
    """Per-frame unwrapped angle of ``positions`` around ``center_3d`` in the (e_u, e_v) plane."""
    angles_raw: List[Optional[float]] = []
    for w in positions:
        if w is None:
            angles_raw.append(None)
            continue
        rel = np.asarray(w, dtype=np.float64).reshape(-1) - center_3d
        if rel.size != 3 or not np.all(np.isfinite(rel)):
            angles_raw.append(None)
            continue
        u_val = float(rel @ e_u)
        v_val = float(rel @ e_v)
        angles_raw.append(float(np.arctan2(v_val, u_val)))
    valid = np.array([a is not None for a in angles_raw], dtype=bool)
    if int(np.count_nonzero(valid)) < 3:
        return None
    unwrapped = np.zeros(len(angles_raw), dtype=np.float64)
    last_un: Optional[float] = None
    last_raw: Optional[float] = None
    for i, ok in enumerate(valid):
        if not ok:
            unwrapped[i] = last_un if last_un is not None else 0.0
            continue
        ang = float(angles_raw[i])
        if last_un is None:
            unwrapped[i] = ang
        else:
            delta = ang - last_raw
            while delta > np.pi:
                delta -= 2.0 * np.pi
            while delta < -np.pi:
                delta += 2.0 * np.pi
            unwrapped[i] = last_un + delta
        last_un = unwrapped[i]
        last_raw = ang
    if not np.all(valid):
        valid_idx = np.where(valid)[0]
        unwrapped = np.interp(np.arange(len(angles_raw)), valid_idx, unwrapped[valid_idx])
    unwrapped -= unwrapped[0]
    return unwrapped


# FunRec/SceneFun3D-style per-track articulation hypothesis (and HDBSCAN cluster).
# Each surviving CoTracker3 corner trajectory in 3-D gets its own (axis, pivot)
# from a plane + 2-D circle fit. Doors with bilateral cloud symmetry break
# whole-cloud per-frame Procrustes (the 180° twin lands axis_point on the door
# centroid), but a single tracked point's 3-D arc has no symmetric twin.
# We then cluster track-level hypotheses; the dominant cluster's median seeds
# the GTSAM twist refinement.
ARTIPOINT_PER_TRACK_MIN_DISPLACEMENT_M = 0.05
ARTIPOINT_PER_TRACK_MIN_VISIBLE_FRAMES = 6
ARTIPOINT_PER_TRACK_MAX_RMSE_M = 0.04
ARTIPOINT_PER_TRACK_MIN_RADIUS_M = 0.02
ARTIPOINT_PER_TRACK_MAX_RADIUS_M = 2.5
ARTIPOINT_PER_TRACK_MAX_PLANARITY = 0.25
# Cluster acceptance gates. The per-track override is only safer than the
# Procrustes-condensed closed-form seed when the per-track signal is *clearly*
# dominant: many hypotheses surviving, many of them clustering together, and
# the cluster members having tightly aligned axes. When any gate fails we fall
# back to the previous (closed-form / wrist-plane) seed path. This prevents
# the cabinet failure where 4 noisy hypotheses formed a fake cluster pointing
# diagonally and pulled GTSAM into a 14 cm-residual local minimum.
ARTIPOINT_CLUSTER_MIN_SIZE = 6
ARTIPOINT_CLUSTER_MIN_HYPOTHESES = 10
ARTIPOINT_CLUSTER_MAX_AXIS_SPREAD_RAD = float(np.deg2rad(20.0))


def per_track_revolute_hypotheses(
    tracks_world: np.ndarray,
    valid_tracks: np.ndarray,
) -> List[dict]:
    """For each track, fit a 3-D plane + 2-D circle and return one
    ``(axis_dir, pivot, radius, rmse, thetas)`` hypothesis per surviving track.
    """
    tracks = np.asarray(tracks_world, dtype=np.float64)
    valid = np.asarray(valid_tracks, dtype=bool)
    if tracks.ndim != 3 or tracks.shape[-1] != 3:
        return []
    n_frames, n_tracks, _ = tracks.shape
    if n_frames < ARTIPOINT_PER_TRACK_MIN_VISIBLE_FRAMES or n_tracks == 0:
        return []
    hypotheses: List[dict] = []
    for j in range(n_tracks):
        col_valid = valid[:, j] & np.all(np.isfinite(tracks[:, j, :]), axis=1)
        n_valid = int(np.count_nonzero(col_valid))
        if n_valid < ARTIPOINT_PER_TRACK_MIN_VISIBLE_FRAMES:
            continue
        positions = tracks[:, j, :]
        positions_valid = positions[col_valid]
        # Reject static / micro-motion tracks: bounding-sphere radius around
        # the track centroid must exceed the displacement floor.
        track_centroid = positions_valid.mean(axis=0)
        span = float(np.max(np.linalg.norm(positions_valid - track_centroid, axis=1)))
        if span < ARTIPOINT_PER_TRACK_MIN_DISPLACEMENT_M:
            continue
        plane = _fit_plane_3d(positions_valid)
        if plane is None:
            continue
        normal, centroid, planarity = plane
        if not np.isfinite(planarity) or planarity > ARTIPOINT_PER_TRACK_MAX_PLANARITY:
            continue
        circle = _fit_circle_in_plane(positions_valid, normal, centroid)
        if circle is None:
            continue
        center_3d, radius, e_u, e_v = circle
        if not (ARTIPOINT_PER_TRACK_MIN_RADIUS_M <= radius <= ARTIPOINT_PER_TRACK_MAX_RADIUS_M):
            continue
        # Residual = mean |‖proj‖ − r| in the fitted plane
        rel = positions_valid - center_3d
        u_proj = rel @ e_u
        v_proj = rel @ e_v
        residual = float(np.mean(np.abs(np.sqrt(u_proj * u_proj + v_proj * v_proj) - radius)))
        if not np.isfinite(residual) or residual > ARTIPOINT_PER_TRACK_MAX_RMSE_M:
            continue
        positions_seq: List[Optional[np.ndarray]] = [
            positions[t] if col_valid[t] else None for t in range(n_frames)
        ]
        thetas = _angles_from_plane_projection(positions_seq, center_3d, e_u, e_v)
        if thetas is None:
            continue
        # Sign-canonicalise the axis so clustering does not split +n / -n into
        # two clusters. Force the largest-magnitude component positive.
        dominant = int(np.argmax(np.abs(normal)))
        if normal[dominant] < 0:
            normal = -normal
        hypotheses.append(
            {
                "axis_dir": normal.astype(np.float64),
                "pivot": center_3d.astype(np.float64),
                "radius_m": float(radius),
                "rmse_m": residual,
                "n_valid_frames": n_valid,
                "track_index": int(j),
                "thetas": thetas.astype(np.float64),
                "span_m": span,
            }
        )
    return hypotheses


def cluster_revolute_hypotheses(
    hypotheses: List[dict],
    clip_diameter_m: float,
    min_cluster_size: int = ARTIPOINT_CLUSTER_MIN_SIZE,
) -> Optional[dict]:
    """HDBSCAN-cluster the per-track hypotheses by (axis_dir, pivot / diameter).

    Returns the dominant cluster's median axis + pivot, the inlier mask, and
    the number of inliers/total — or ``None`` if no cluster is found.
    """
    if not hypotheses or not HDBSCAN_AVAILABLE or _HDBSCAN is None:
        return None
    diam = max(float(clip_diameter_m), 1e-3)
    feats = np.stack(
        [
            np.concatenate(
                [
                    np.asarray(h["axis_dir"], dtype=np.float64),
                    np.asarray(h["pivot"], dtype=np.float64) / diam,
                ]
            )
            for h in hypotheses
        ],
        axis=0,
    )
    try:
        labels = _HDBSCAN(
            min_cluster_size=int(min_cluster_size),
            min_samples=2,
            cluster_selection_method="eom",
        ).fit_predict(feats)
    except Exception:  # noqa: BLE001
        return None
    valid_mask = labels >= 0
    if int(np.count_nonzero(valid_mask)) < min_cluster_size:
        return None
    unique, counts = np.unique(labels[valid_mask], return_counts=True)
    dominant_label = int(unique[int(np.argmax(counts))])
    inlier_mask = labels == dominant_label
    inlier_axes = np.stack(
        [np.asarray(h["axis_dir"]) for h, m in zip(hypotheses, inlier_mask) if m], axis=0
    )
    inlier_pivots = np.stack(
        [np.asarray(h["pivot"]) for h, m in zip(hypotheses, inlier_mask) if m], axis=0
    )
    median_axis = np.median(inlier_axes, axis=0)
    axis_norm = float(np.linalg.norm(median_axis))
    if axis_norm < 1e-6:
        return None
    median_axis = median_axis / axis_norm
    # Axis-coherence gate: members must agree on the axis direction to within
    # ARTIPOINT_CLUSTER_MAX_AXIS_SPREAD_RAD. A "cluster" of 4-6 noisy short-arc
    # hypotheses with arbitrary plane normals can pass HDBSCAN by happening to
    # sit near each other in (pivot)-space, but their axes disagree wildly. We
    # measure spread as the worst-case angle between any inlier axis and the
    # median axis. If it exceeds the cap, reject the cluster.
    axis_dots = np.clip(inlier_axes @ median_axis, -1.0, 1.0)
    axis_angles = np.arccos(np.abs(axis_dots))  # |dot|: treat +n and -n as same line
    axis_spread = float(np.max(axis_angles)) if axis_angles.size else 0.0
    if axis_spread > float(ARTIPOINT_CLUSTER_MAX_AXIS_SPREAD_RAD):
        return {
            "axis_dir": median_axis.astype(np.float64),
            "pivot": np.median(inlier_pivots, axis=0).astype(np.float64),
            "n_inliers": int(np.count_nonzero(inlier_mask)),
            "n_total": int(len(hypotheses)),
            "dominant_label": dominant_label,
            "inlier_track_indices": [int(h["track_index"]) for h, m in zip(hypotheses, inlier_mask) if m],
            "axis_spread_rad": axis_spread,
            "axis_coherent": False,
        }
    median_pivot = np.median(inlier_pivots, axis=0)
    inlier_track_indices = [int(h["track_index"]) for h, m in zip(hypotheses, inlier_mask) if m]
    return {
        "axis_dir": median_axis.astype(np.float64),
        "pivot": median_pivot.astype(np.float64),
        "n_inliers": int(np.count_nonzero(inlier_mask)),
        "n_total": int(len(hypotheses)),
        "dominant_label": dominant_label,
        "inlier_track_indices": inlier_track_indices,
        "axis_spread_rad": axis_spread,
        "axis_coherent": True,
    }


def _fit_circle_xy_least_squares(positions_xy: np.ndarray) -> Optional[Tuple[np.ndarray, float]]:
    """Algebraic 2-D circle fit. Returns (center_xy, radius) or None."""
    pts = np.asarray(positions_xy, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2 or pts.shape[0] < 3:
        return None
    finite = np.all(np.isfinite(pts), axis=1)
    pts = pts[finite]
    if pts.shape[0] < 3:
        return None
    A = np.column_stack([pts[:, 0], pts[:, 1], np.ones(pts.shape[0])])
    b = -(pts[:, 0] ** 2 + pts[:, 1] ** 2)
    try:
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError:
        return None
    a_p, b_p, c_p = sol
    cx, cy = -a_p / 2.0, -b_p / 2.0
    r_sq = (a_p * a_p + b_p * b_p) / 4.0 - c_p
    if not np.isfinite(r_sq) or r_sq <= 0.0:
        return None
    return np.array([cx, cy], dtype=np.float64), float(np.sqrt(r_sq))


def _estimate_motion_point_bisectors(
    transformations: Sequence[np.ndarray],
    axis: np.ndarray,
) -> Optional[np.ndarray]:
    """Estimate a revolute center from pose translations via perpendicular bisectors.

    This mirrors ArtiPoint's `estimate_motion_point_bisectors(...)` frontend idea:
    for a set of points moving on a circular arc around `axis`, consecutive translation
    pairs define perpendicular bisectors whose common intersection is the rotation
    center projected into the plane normal to `axis`.
    """
    if transformations is None:
        return None
    translations = []
    for transform in transformations:
        arr = np.asarray(transform, dtype=np.float64)
        if arr.shape != (4, 4) or not np.all(np.isfinite(arr)):
            continue
        translations.append(arr[:3, 3])
    if len(translations) < 3:
        return None
    axis_arr = _normalize3(np.asarray(axis, dtype=np.float64).reshape(3))
    if not np.all(np.isfinite(axis_arr)) or float(np.linalg.norm(axis_arr)) < 1e-9:
        return None
    pts = np.asarray(translations, dtype=np.float64)
    centroid = np.mean(pts, axis=0)
    seed = np.array([1.0, 0.0, 0.0], dtype=np.float64) if abs(float(axis_arr[0])) < 0.9 else np.array([0.0, 1.0, 0.0], dtype=np.float64)
    e_u = np.cross(axis_arr, seed)
    e_u_norm = float(np.linalg.norm(e_u))
    if e_u_norm < 1e-9:
        return None
    e_u = e_u / e_u_norm
    e_v = np.cross(axis_arr, e_u)
    rel = pts - centroid
    pts_2d = np.stack([rel @ e_u, rel @ e_v], axis=1)

    A_rows = []
    b_rows = []
    for p_a, p_b in zip(pts_2d[:-1], pts_2d[1:]):
        direction = np.asarray(p_b - p_a, dtype=np.float64)
        if float(np.linalg.norm(direction)) < 1e-6:
            continue
        midpoint = 0.5 * (p_a + p_b)
        A_rows.append(direction)
        b_rows.append(float(direction @ midpoint))
    if len(A_rows) < 2:
        return None
    A = np.asarray(A_rows, dtype=np.float64)
    b = np.asarray(b_rows, dtype=np.float64)
    try:
        center_2d, *_ = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError:
        return None
    if center_2d.shape[0] != 2 or not np.all(np.isfinite(center_2d)):
        return None
    center_3d = centroid + center_2d[0] * e_u + center_2d[1] * e_v
    return center_3d.astype(np.float64)


def _wrist_derived_thetas(
    wrist_positions: Sequence[Optional[np.ndarray]],
    axis_point_xy: np.ndarray,
) -> Optional[np.ndarray]:
    """Unwrap wrist angular position around axis_point.xy into a per-frame theta.

    Frames with no valid wrist are linearly interpolated from their neighbours.
    """
    angles: List[Optional[float]] = []
    for w in wrist_positions:
        if w is None:
            angles.append(None)
            continue
        offset = np.asarray(w, dtype=np.float64).reshape(-1)[:2] - axis_point_xy
        if not np.all(np.isfinite(offset)):
            angles.append(None)
            continue
        angles.append(float(np.arctan2(offset[1], offset[0])))
    valid_mask = np.array([a is not None for a in angles], dtype=bool)
    if int(np.count_nonzero(valid_mask)) < 3:
        return None
    raw = np.array([a if a is not None else 0.0 for a in angles], dtype=np.float64)
    # Unwrap only across valid frames; carry forward the previous unwrapped value through gaps.
    unwrapped = np.zeros_like(raw)
    last_unwrapped: Optional[float] = None
    last_raw: Optional[float] = None
    for i, ok in enumerate(valid_mask):
        if not ok:
            unwrapped[i] = last_unwrapped if last_unwrapped is not None else 0.0
            continue
        if last_unwrapped is None:
            unwrapped[i] = raw[i]
        else:
            delta = raw[i] - last_raw
            while delta > np.pi:
                delta -= 2.0 * np.pi
            while delta < -np.pi:
                delta += 2.0 * np.pi
            unwrapped[i] = last_unwrapped + delta
        last_unwrapped = unwrapped[i]
        last_raw = raw[i]
    # Interpolate gaps so theta_m is defined everywhere
    if not np.all(valid_mask):
        valid_idx = np.where(valid_mask)[0]
        unwrapped = np.interp(np.arange(len(raw)), valid_idx, unwrapped[valid_idx])
    # Zero-anchor at frame 0
    unwrapped -= unwrapped[0]
    return unwrapped


def _build_consecutive_point_pairs_from_tracks(
    tracks_world: np.ndarray,
    valid_tracks: np.ndarray,
) -> List[Optional[np.ndarray]]:
    """Mirror ArtiPoint's ``create_points_pairs`` for our (T, N, 3) tracks tensor.

    Matches their exact algorithm: for each frame ``i``, scan forward through
    frames ``j > i`` until we find one with at least one co-visible track, then
    record the pair ``(i, j)`` and ``break``. Crucially, ``j`` is NOT forced to
    be ``i+1``. When the door is mostly occluded for several early frames so
    that frame 0 and frame 1 have no shared visible track, ArtiPoint still
    yields a single pair anchored at frame 0 by pairing it with frame 5 (or
    whichever is the first usable forward frame). Without this skip-ahead, we
    end up with ``None`` placeholders that leave the early theta variables
    unconstrained, and the optimizer drifts them into garbage.

    Returns a list of length ``T - 1`` (one entry per starting frame ``i``).
    Each entry is an ``(M_i, 2, 3)`` array of co-visible point pairs at frame
    ``i`` and the chosen forward frame ``j``, or ``None`` if no such ``j``
    exists for that ``i`` (i.e. nothing visible at ``i`` is ever visible again).
    """
    tracks = np.asarray(tracks_world, dtype=np.float64)
    valid = np.asarray(valid_tracks, dtype=bool)
    if tracks.ndim != 3 or tracks.shape[2] != 3:
        return []
    T, N, _ = tracks.shape
    pairs: List[Optional[np.ndarray]] = []
    for i in range(T - 1):
        v_i = valid[i] & np.all(np.isfinite(tracks[i]), axis=1)
        pair_found: Optional[np.ndarray] = None
        for j in range(i + 1, T):
            v_j = valid[j] & np.all(np.isfinite(tracks[j]), axis=1)
            common = v_i & v_j
            if int(np.count_nonzero(common)) > 0:
                pair_found = np.stack(
                    [tracks[i, common], tracks[j, common]], axis=1
                ).astype(np.float64)  # (M, 2, 3)
                break
        pairs.append(pair_found)
    return pairs


def artipoint_native_fit_from_tracks(
    tracks_world: np.ndarray,
    valid_tracks: np.ndarray,
    theta_prior_deg: Optional[float] = None,   # paper config: motion_estimation.theta_prior: null
) -> Optional[dict]:
    """Run ArtiPoint's vendored ``PoseEstFactorGraph`` directly on raw 3-D tracks.

    Bypasses our reimplementation entirely. Returns a dict shaped like the rest
    of the pipeline expects (axis_direction, axis_point, joint_type, thetas,
    xi, reconstructed_pose_history, residuals, optimizer label) or ``None`` if
    the solver cannot make progress or ArtiPoint isn't importable.
    """
    if not ARTIPOINT_AVAILABLE or ArtiPointPoseEstFactorGraph is None:
        return None
    pairs = _build_consecutive_point_pairs_from_tracks(tracks_world, valid_tracks)
    pairs_usable = [p for p in pairs if p is not None]
    n_pairs_total = len(pairs)
    n_pairs_ok = len(pairs_usable)
    if n_pairs_ok < ARTIPOINT_MIN_POSE_UPDATES:
        # Distribution of co-visible track counts per pair, for context.
        per_pair_counts = [int(p.shape[0]) for p in pairs_usable]
        max_count = max(per_pair_counts) if per_pair_counts else 0
        print(
            f"[ArtiPoint-FactorGraph] FAIL too_few_usable_pairs: "
            f"{n_pairs_ok}/{n_pairs_total} consecutive frame pairs had >= 3 co-visible "
            f"tracks (need >= {ARTIPOINT_MIN_POSE_UPDATES}, max co-visible count seen: "
            f"{max_count}, "
            f"COTRACKER3_VIS_THRESHOLD={os.environ.get('COTRACKER3_VIS_THRESHOLD', '0.70')})"
        )
        return None
    if any(pair is None for pair in pairs):
        print(
            f"[ArtiPoint-FactorGraph][Error] missing consecutive point pairs "
            f"({n_pairs_ok}/{n_pairs_total} usable); zero-pair fallback is disabled."
        )
        return None
    safe_pairs: List[np.ndarray] = [np.asarray(pair, dtype=np.float64) for pair in pairs]  # type: ignore[arg-type]

    graph = ArtiPointPoseEstFactorGraph(num_frames=len(safe_pairs))
    for i, pair in enumerate(safe_pairs):
        graph.add_point_pair_factor(pair, i)
        graph.add_articulation_factor(i, theta_prior=theta_prior_deg)
        graph.add_world_frame_factor(i)
    try:
        ok = graph.solve_factor_graph()
    except Exception as exc:  # noqa: BLE001
        print(
            f"[ArtiPoint-FactorGraph] FAIL solver_raised: {exc} "
            f"(had {n_pairs_ok}/{n_pairs_total} usable pairs, "
            f"COTRACKER3_VIS_THRESHOLD={os.environ.get('COTRACKER3_VIS_THRESHOLD', '0.70')})"
        )
        return None
    if not ok:
        per_pair_counts = [int(p.shape[0]) for p in pairs_usable]
        avg_count = float(np.mean(per_pair_counts)) if per_pair_counts else 0.0
        print(
            f"[ArtiPoint-FactorGraph] FAIL solver_returned_false: "
            f"{n_pairs_ok}/{n_pairs_total} pairs fed in (avg {avg_count:.1f} co-visible "
            f"tracks per pair); LM did not converge. "
            f"COTRACKER3_VIS_THRESHOLD={os.environ.get('COTRACKER3_VIS_THRESHOLD', '0.70')}"
        )
        return None
    try:
        params = graph.estimate_joint_type_and_parameters()
    except Exception as exc:  # noqa: BLE001
        print(f"[ArtiPoint-FactorGraph] FAIL parameter_extraction_raised: {exc}")
        return None

    # ArtiPoint maintains TWO trajectories at render time
    # (track/artipoint.py:885 free_traj). The "free" trajectory chains the
    # per-frame OPTIMIZED ``T(i)`` matrices — each T(i) was fit directly to
    # its point-pair factor, so it reflects what the data actually shows.
    # The "twist" trajectory chains Exp(ξ·θ_i) — clean but smoothed, can
    # drift when early frames are unconstrained. For the rendered cloud we
    # want the free trajectory (observation-faithful, matches the keypoints).
    # The twist+thetas+axis+pivot still describe the articulation joint.
    try:
        optimized_transformations = graph.get_optimized_transformations()
        free_T_per_frame = [
            np.asarray(T.matrix(), dtype=np.float64) for T in optimized_transformations
        ]
    except Exception as exc:  # noqa: BLE001
        print(f"[ArtiPoint-FactorGraph][Error] free_T_per_frame extraction failed: {exc}")
        return None

    joint_type = params.get("type", "revolute")
    axis_unit = np.asarray(params.get("axis"), dtype=np.float64).reshape(3)
    axis_unit = axis_unit / max(float(np.linalg.norm(axis_unit)), 1e-12)
    pivot = np.asarray(params.get("q", params.get("center", np.zeros(3))), dtype=np.float64).reshape(3)
    twist = np.asarray(params.get("twist"), dtype=np.float64).reshape(6)
    thetas_per_frame = np.asarray(params.get("thetas", []), dtype=np.float64).reshape(-1)
    pitch = float(params.get("pitch", 0.0))

    # ArtiPoint's twist is unnormalised; cumulative angle at frame k is the
    # running sum of ``thetas_per_frame[:k]`` scaled by ``‖ω‖`` for revolute or
    # ``‖v‖`` for prismatic. We canonicalise by absorbing the scale into the
    # reported per-frame thetas, then re-emit a unit-axis twist.
    omega = twist[:3]
    v_lin = twist[3:]
    omega_norm = float(np.linalg.norm(omega))
    v_norm = float(np.linalg.norm(v_lin))
    if joint_type == "revolute" and omega_norm > 1e-9:
        scale = omega_norm
    elif joint_type == "prismatic" and v_norm > 1e-9:
        scale = v_norm
    else:
        scale = 1.0
    thetas_scaled = thetas_per_frame * scale
    if joint_type == "revolute" and omega_norm > 1e-9:
        xi_canonical = np.concatenate([omega / omega_norm, v_lin / omega_norm], axis=0)
    elif joint_type == "prismatic" and v_norm > 1e-9:
        xi_canonical = np.concatenate([omega / v_norm, v_lin / v_norm], axis=0)
    else:
        xi_canonical = twist.copy()

    # Cumulative thetas for the pipeline's downstream consumers (description
    # strings, viser slider, residual reporting).
    cumulative = np.concatenate([[0.0], np.cumsum(thetas_scaled)], axis=0)
    n_frames = cumulative.shape[0]

    return {
        "joint_type": joint_type,
        "axis_direction": axis_unit.astype(np.float32).tolist(),
        "axis_point": pivot.astype(np.float32).tolist(),
        "xi": xi_canonical.astype(np.float32).tolist(),
        "thetas": [float(v) for v in cumulative],
        "pitch": pitch,
        "optimizer": "artipoint_native_pose_est",
        "n_frames": int(n_frames),
        # Per-frame OPTIMIZED T(i) for free-trajectory rendering — observation
        # faithful, matches ArtiPoint paper's free_traj branch.
        "free_T_per_frame": [m.tolist() for m in free_T_per_frame],
    }


def _fit_articulation_gtsam(
    rotations: Sequence[np.ndarray],
    translations: Sequence[np.ndarray],
    axis0: np.ndarray,
    point0: np.ndarray,
    theta0: np.ndarray,
    trans_scale: float,
    vel_w: float,
    jerk_w: float,
    mono_w: float,
    prior_axis: Optional[np.ndarray],
    prior_axis_w: float,
    axis_locked: bool,
    point_pairs: Optional[Sequence[Tuple[np.ndarray, np.ndarray]]] = None,
    huber_k: float = 1.345,
    pose_sigma: float = 0.1,
    point_sigma: float = 0.1,
    max_iters: int = 200,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str, np.ndarray, float]:
    """ArtiPoint Eq.(4) factor-graph fit using the **twist** parameterization,
    mirroring ``artipoint/factor_graph/pose_est.py``.

    Variables:
      * ``X(0)`` = Vector6 twist ξ = (ω, v) ∈ se(3), shared across all frames
      * ``T(i)`` = Pose3 per-frame transform from reference frame to frame i (FREE)
      * ``H(i)`` = scalar (Python double) articulation angle θ_i

    Factors (when ``point_pairs`` is supplied — the ArtiPoint-faithful path):
      * **Point-pair factor** (per frame): error = ``T(i)·p_array − q_array``
        — raw 3D track observations, NOT Procrustes-condensed. This sidesteps
        the symmetric-cloud 180°-twin ambiguity that locks Pose3 priors to
        random per-frame solutions.
      * **Articulation factor** (per frame): error = ``Logmap(Expmap(ξ·θ_i).between(T_i))``
        in se(3), Huber-robust. Uses analytical Pose3 Expmap/Logmap Jacobians.
      * **Theta velocity / jerk / monotonicity** smoothness factors — our addition
        on top of the paper, useful for noisy egocentric data.
      * **Axis prior** factor — only on the rotational part of ξ when a label
        prior is available; never used as a hard lock.

    Factors (legacy path — when ``point_pairs`` is ``None``):
      * **Pose3 prior** on each ``T(i)`` anchoring it to the observed RANSAC-Kabsch
        pose, wrapped in Huber-robust noise. Used as fallback when the caller
        does not have access to the raw 3D track pairs.

    After optimization we extract joint parameters using ArtiPoint's convention:
      * ω = ξ[:3], v = ξ[3:]
      * pitch = (ω·v) / ‖ω‖²
      * Joint type by pitch: |pitch| > 0.2 → prismatic, else revolute
      * For revolute: axis = ω̂, axis_point from motion-bisector geometry,
        with the Plücker closest point (ω × v) / ‖ω‖² as fallback
      * For prismatic: axis = v̂, axis_point = 0 (any point on the line)

    Returns
    -------
    axis : (3,) unit world-space axis direction
    point : (3,) world-space point on the axis
    thetas : (N,) per-frame articulation angle (θ[0] = 0)
    joint_type : "revolute" or "prismatic"
    twist : (6,) optimized se(3) twist
    pitch : (ω·v)/‖ω‖² — useful for screw-joint reporting
    """
    n_frames = len(rotations)
    n_theta_vars = n_frames - 1
    if n_theta_vars < 1:
        raise ValueError("Need at least 2 pose samples for the GTSAM fit")

    gt = _gtsam
    if gt is None:
        raise RuntimeError("gtsam unavailable; caller should fall back to scipy")

    sym = gt.symbol_shorthand
    X_key = sym.X(0)         # shared twist ξ (Vector6)
    T_keys = [sym.T(i) for i in range(n_theta_vars)]   # observed pose per frame
    H_keys = [sym.H(i) for i in range(n_theta_vars)]   # scalar θ_i per frame

    # ----- Initial values -------------------------------------------------
    # Mode switch: when ``point_pairs`` is provided we run ArtiPoint's
    # *exact* setup (T(i)=I, ξ=zeros, θ=1e-6, weak θ prior, no smoothness).
    # Otherwise we keep the warm-start path used by the legacy Pose3-prior
    # fallback (axis0/point0/theta0 from upstream closed-form).
    artipoint_native = (
        point_pairs is not None
        and len(point_pairs) == n_theta_vars
        and any(p is not None and len(p[0]) > 0 for p in point_pairs)
    )

    rotations_arr = [np.asarray(R, dtype=np.float64).reshape(3, 3) for R in rotations]
    translations_arr = [np.asarray(t, dtype=np.float64).reshape(3) for t in translations]
    obs_poses = [
        gt.Pose3(gt.Rot3(rotations_arr[i + 1]), translations_arr[i + 1])
        for i in range(n_theta_vars)
    ]

    values = gt.Values()
    if artipoint_native:
        # ArtiPoint init (pose_est.py:142-187): twist=zeros(6), T(i)=identity,
        # theta=1e-6. No warm-start from Procrustes/closed-form. The optimizer
        # discovers T(i), ξ, θ purely from point-pair gradients.
        values.insert(X_key, np.zeros(6, dtype=np.float64))
        for i in range(n_theta_vars):
            values.insert(T_keys[i], gt.Pose3())
            values.insert(H_keys[i], 1e-6)
    else:
        # Legacy warm-start (Pose3-prior fallback path).
        omega_init = np.asarray(axis0, dtype=np.float64).reshape(3)
        omega_norm_init = float(np.linalg.norm(omega_init))
        if omega_norm_init > 1e-9:
            omega_init = omega_init / omega_norm_init
        point_init = np.asarray(point0, dtype=np.float64).reshape(3)
        v_init = np.cross(point_init, omega_init)
        twist_init = np.concatenate([omega_init, v_init], axis=0).astype(np.float64)
        values.insert(X_key, twist_init)
        for i in range(n_theta_vars):
            values.insert(T_keys[i], obs_poses[i])
            th_init = float(theta0[i + 1]) if (i + 1) < len(theta0) else 0.0
            if not np.isfinite(th_init):
                th_init = 0.0
            values.insert(H_keys[i], th_init)

    graph = gt.NonlinearFactorGraph()

    # ----- Noise models ---------------------------------------------------
    # 6-D Huber-robust pose noise (matches ArtiPoint's huber_pose_noise_6D
    # with k = 1.345). pose_sigma scales the underlying Gaussian; the Huber
    # threshold is applied in whitened-residual space, so 1.345 is the
    # statistical default (95% efficiency on Gaussian, robust to outliers).
    pose_sigmas = np.array([pose_sigma] * 6)
    pose_noise = gt.noiseModel.Diagonal.Sigmas(pose_sigmas)
    huber_pose_6d = gt.noiseModel.Robust.Create(
        gt.noiseModel.mEstimator.Huber.Create(float(huber_k)),
        pose_noise,
    )

    # ----- Data-fit factors -----------------------------------------------
    # Preferred (ArtiPoint-faithful): per-frame point-pair custom factors on
    # raw 3D tracks. T(i) is FREE; the factor pulls it toward consistency with
    # observed correspondences. This avoids the per-pair RANSAC-Kabsch step
    # collapsing the symmetric-cloud ambiguity into one of two 180°-twin
    # solutions before the optimizer sees the data.
    # Fallback (legacy): Pose3 prior on each T(i) from RANSAC-Kabsch. Used
    # when point_pairs is not supplied.
    if artipoint_native:
        def _make_point_pair_err(p_arr_local: np.ndarray, q_arr_local: np.ndarray):
            def _err(this, vals, jacobians):
                keys = this.keys()
                T_i = vals.atPose3(keys[0])
                R = T_i.rotation().matrix()
                t = T_i.translation()
                p_transformed = p_arr_local @ R.T + t
                residual = (p_transformed - q_arr_local).reshape(-1)
                if jacobians is not None and len(jacobians) > 0:
                    n_pts = p_arr_local.shape[0]
                    J = np.zeros((n_pts * 3, 6), dtype=np.float64)
                    for k in range(n_pts):
                        p = p_arr_local[k]
                        skew_p = np.array(
                            [[0.0, -p[2], p[1]],
                             [p[2], 0.0, -p[0]],
                             [-p[1], p[0], 0.0]],
                            dtype=np.float64,
                        )
                        J[k * 3:(k + 1) * 3, :3] = -R @ skew_p
                        J[k * 3:(k + 1) * 3, 3:] = np.eye(3)
                    jacobians[0] = J
                return residual
            return _err

        for i in range(n_theta_vars):
            pair = point_pairs[i]
            if pair is None:
                raise RuntimeError(
                    f"[ArticulationFit][Error] missing point-pair data at frame {i}; Pose3-prior fallback is disabled."
                )
            p_arr, q_arr = pair
            p_arr = np.asarray(p_arr, dtype=np.float64)
            q_arr = np.asarray(q_arr, dtype=np.float64)
            if p_arr.ndim != 2 or p_arr.shape[1] != 3 or p_arr.shape != q_arr.shape:
                raise RuntimeError(
                    f"[ArticulationFit][Error] invalid point-pair shapes at frame {i}; Pose3-prior fallback is disabled."
                )
            n_pts = int(p_arr.shape[0])
            if n_pts < 3:
                raise RuntimeError(
                    f"[ArticulationFit][Error] only {n_pts} point pairs at frame {i}; Pose3-prior fallback is disabled."
                )
            point_sigmas = np.array([point_sigma] * 3 * n_pts)
            huber_point = gt.noiseModel.Robust.Create(
                gt.noiseModel.mEstimator.Huber.Create(float(huber_k)),
                gt.noiseModel.Diagonal.Sigmas(point_sigmas),
            )
            graph.add(
                gt.CustomFactor(huber_point, [T_keys[i]], _make_point_pair_err(p_arr, q_arr))
            )
    else:
        for i, T_obs in enumerate(obs_poses):
            graph.add(gt.PriorFactorPose3(T_keys[i], T_obs, huber_pose_6d))

    # ----- Articulation factor: Logmap(Expmap(ξ·θ).between(T)) ------------
    # Mirrors ArtiPoint's ``_articulation_factor`` with the proper chain-rule
    # Jacobians via Pose3.Expmap / Pose3.Logmap / Pose3.between's H-buffers.
    def _articulation_err(this, vals, jacobians):
        keys = this.keys()
        xi = vals.atVector(keys[0])           # Vector6 twist
        T_i = vals.atPose3(keys[1])           # Pose3 observed
        th_i = vals.atDouble(keys[2])         # double angle
        H_xi_th = np.zeros((6, 6), order="F")
        H_T_twist = np.zeros((6, 6), order="F")
        H_T_i = np.zeros((6, 6), order="F")
        H_err = np.zeros((6, 6), order="F")
        T_twist = gt.Pose3.Expmap(xi * float(th_i), H_xi_th)
        T_diff = T_twist.between(T_i, H_T_twist, H_T_i)
        error = gt.Pose3.Logmap(T_diff, H_err)
        if jacobians is not None and len(jacobians) > 0:
            jacobians[0] = H_err @ H_T_twist @ H_xi_th @ (float(th_i) * np.eye(6))  # ∂err/∂ξ
            jacobians[1] = H_err @ H_T_i                                              # ∂err/∂T_i
            jacobians[2] = (H_err @ H_T_twist @ H_xi_th @ xi.reshape(6, 1))           # ∂err/∂θ_i
        return error

    for i in range(n_theta_vars):
        graph.add(gt.CustomFactor(huber_pose_6d, [X_key, T_keys[i], H_keys[i]], _articulation_err))

    # ----- ArtiPoint theta prior (per-frame Gaussian pull to ~0) ----------
    # Mirrors PriorFactorDouble(th(i), theta_prior, σ=25°) in
    # pose_est.py:178-183. Pulls each per-frame angle weakly toward 1e-6,
    # absorbing the global scale ambiguity (twist+theta vs scaled twist).
    # Only active in the ArtiPoint-native path.
    if artipoint_native:
        theta_prior_noise = gt.noiseModel.Diagonal.Sigmas(
            np.array([float(np.deg2rad(25.0))], dtype=np.float64)
        )
        for h_key in H_keys:
            graph.add(gt.PriorFactorDouble(h_key, 1e-6, theta_prior_noise))

    # ----- Theta-velocity smoothness (between consecutive doubles) --------
    # Our addition; absent from ArtiPoint. Disabled in ArtiPoint-native mode
    # so the factor graph matches their setup exactly.
    if not artipoint_native and vel_w > 0.0 and len(H_keys) >= 2:
        vel_noise = gt.noiseModel.Isotropic.Sigma(1, 1.0 / max(float(vel_w), 1e-6))
        for a, b in zip(H_keys[:-1], H_keys[1:]):
            graph.add(gt.BetweenFactorDouble(a, b, 0.0, vel_noise))

    # ----- Theta-jerk smoothness (3-term second difference) ---------------
    if not artipoint_native and jerk_w > 0.0 and len(H_keys) >= 3:
        jerk_noise = gt.noiseModel.Isotropic.Sigma(1, 1.0 / max(float(jerk_w), 1e-6))
        for i in range(1, len(H_keys) - 1):
            ka, kb, kc = H_keys[i - 1], H_keys[i], H_keys[i + 1]
            def jerk_err(this, vals, jacobians, ka=ka, kb=kb, kc=kc):
                ta = vals.atDouble(ka)
                tb = vals.atDouble(kb)
                tc = vals.atDouble(kc)
                r = np.array([ta - 2.0 * tb + tc], dtype=np.float64)
                if jacobians is not None and len(jacobians) > 0:
                    jacobians[0] = np.array([[1.0]])
                    jacobians[1] = np.array([[-2.0]])
                    jacobians[2] = np.array([[1.0]])
                return r
            graph.add(gt.CustomFactor(jerk_noise, [ka, kb, kc], jerk_err))

    # ----- Theta monotonicity (hinge factor) ------------------------------
    monotonic_sign = float(np.sign(float(np.sum(theta0)))) or 1.0
    if not artipoint_native and mono_w > 0.0 and len(H_keys) >= 2:
        mono_noise = gt.noiseModel.Isotropic.Sigma(1, 1.0 / max(float(mono_w), 1e-6))
        for a, b in zip(H_keys[:-1], H_keys[1:]):
            def mono_err(this, vals, jacobians, ka=a, kb=b, sgn=monotonic_sign):
                ta = vals.atDouble(ka)
                tb = vals.atDouble(kb)
                d = float(tb) - float(ta)
                backward = max(0.0, -sgn * d)
                r = np.array([backward])
                if jacobians is not None and len(jacobians) > 0:
                    if backward > 0.0:
                        jacobians[0] = np.array([[sgn]])
                        jacobians[1] = np.array([[-sgn]])
                    else:
                        jacobians[0] = np.array([[0.0]])
                        jacobians[1] = np.array([[0.0]])
                return r
            graph.add(gt.CustomFactor(mono_noise, [a, b], mono_err))

    # ----- Optional axis-direction soft prior (on ω = ξ[:3]) -------------
    # Our addition; absent from ArtiPoint. Disabled in ArtiPoint-native mode.
    if not artipoint_native and prior_axis is not None and prior_axis_w > 0.0:
        prior_axis_arr = np.asarray(prior_axis, dtype=np.float64).reshape(3)
        prior_axis_arr = prior_axis_arr / max(float(np.linalg.norm(prior_axis_arr)), 1e-9)
        ap_noise = gt.noiseModel.Isotropic.Sigma(3, 1.0 / max(float(prior_axis_w), 1e-6))
        def axis_prior_err(this, vals, jacobians, prior=prior_axis_arr):
            xi = vals.atVector(this.keys()[0])
            omega = xi[:3]
            n = max(float(np.linalg.norm(omega)), 1e-9)
            omega_n = omega / n
            r = np.cross(omega_n, prior)
            if jacobians is not None and len(jacobians) > 0:
                # Numerical Jacobian limited to the ω rows (cols 0..2). The v
                # rows (cols 3..5) are zero since the prior touches direction only.
                J = np.zeros((3, 6), dtype=np.float64)
                eps = 1e-7
                for k in range(3):
                    d = np.zeros(6); d[k] = eps
                    f_plus = np.cross((omega + d[:3]) / max(float(np.linalg.norm(omega + d[:3])), 1e-9), prior)
                    f_minus = np.cross((omega - d[:3]) / max(float(np.linalg.norm(omega - d[:3])), 1e-9), prior)
                    J[:, k] = (f_plus - f_minus) / (2.0 * eps)
                jacobians[0] = J
            return r
        graph.add(gt.CustomFactor(ap_noise, [X_key], axis_prior_err))

    # ----- Solve ----------------------------------------------------------
    params = gt.LevenbergMarquardtParams()
    params.setMaxIterations(int(max_iters))
    params.setAbsoluteErrorTol(1e-8)
    params.setRelativeErrorTol(1e-8)
    optimizer = gt.LevenbergMarquardtOptimizer(graph, values, params)
    result = optimizer.optimize()

    # ----- Extract twist, joint type, axis, point (ArtiPoint Plücker) -----
    xi_opt = np.asarray(result.atVector(X_key), dtype=np.float64).reshape(6)
    omega = xi_opt[:3]
    v_lin = xi_opt[3:]
    omega_norm = float(np.linalg.norm(omega))
    v_norm = float(np.linalg.norm(v_lin))
    pitch = float(np.dot(omega, v_lin) / max(omega_norm * omega_norm, 1e-12))
    if abs(pitch) > 0.2:
        joint_type = "prismatic"
        axis_out = v_lin / max(v_norm, 1e-9)
        # On a pure prismatic line the "axis point" is degenerate; use a
        # plausible anchor by chaining the per-frame increments and taking the
        # first frame's accumulated translation.
        cum_pris = np.eye(4, dtype=np.float64)
        if n_theta_vars >= 1:
            th0 = float(result.atDouble(H_keys[0]))
            cum_pris = np.asarray(gt.Pose3.Expmap(xi_opt * th0).matrix(), dtype=np.float64)
        point_out = cum_pris[:3, 3]
    else:
        joint_type = "revolute"
        axis_out = omega / max(omega_norm, 1e-9)
        # Chain per-frame increments to recover the cumulative body trajectory
        # (mirrors ArtiPoint's `traj_twist` construction in
        # estimate_joint_type_and_parameters). Motion bisectors then operate on
        # the cumulative translations, which trace the circular arc whose
        # center is the actual hinge point.
        free_transforms: List[np.ndarray] = [np.eye(4, dtype=np.float64)]
        for i in range(n_theta_vars):
            th_i = float(result.atDouble(H_keys[i]))
            inc = np.asarray(gt.Pose3.Expmap(xi_opt * th_i).matrix(), dtype=np.float64)
            free_transforms.append(inc @ free_transforms[-1])
        point_out = _estimate_motion_point_bisectors(free_transforms, axis_out)
        if point_out is None:
            # Plücker closest-point fallback: q = (ω × v) / ‖ω‖²
            point_out = np.cross(omega, v_lin) / max(omega_norm * omega_norm, 1e-12)

    # Convert per-frame increments back to a CUMULATIVE theta series for
    # downstream consumers (the "sweep θ_min ... θ_max" descriptions, the
    # visualisation slider, the event.json export, etc.). All of those expect
    # ``thetas_out[i]`` = total angle traversed up to and including frame i.
    thetas_out = np.zeros(n_frames, dtype=np.float64)
    cumulative = 0.0
    for i in range(n_theta_vars):
        cumulative += float(result.atDouble(H_keys[i]))
        thetas_out[i + 1] = cumulative

    # GTSAM has a scale ambiguity between (ω, v, θ): scaling all three by
    # (α, α, 1/α) leaves every per-frame SE(3) invariant. Canonicalise by
    # rescaling ω to unit length and absorbing the factor into θ. After this
    # rescale, ``thetas_out`` is in radians of actual rotation, matching how
    # downstream code (and the "sweep ... ° .. ...°" description) interprets them.
    # Note: ``axis_out`` and ``point_out`` are already invariant under this
    # rescaling (axis is the normalized ω; point comes from the Plücker formula
    # or bisector geometry which both cancel the magnitude of ω).
    if joint_type == "revolute" and omega_norm > 1e-9:
        thetas_out = thetas_out * omega_norm
        # Renormalise the twist for the returned ``xi`` to be consistent.
        xi_opt = np.concatenate([omega / omega_norm, v_lin / omega_norm], axis=0)
        pitch = float(np.dot(xi_opt[:3], xi_opt[3:]))
    elif joint_type == "prismatic" and v_norm > 1e-9:
        # For pure prismatic, ‖ω‖≈0, so the natural scale is ‖v‖. Rescale so
        # ‖v‖ = 1 and θ carries the translation magnitude (in metres).
        thetas_out = thetas_out * v_norm
        xi_opt = np.concatenate([omega / v_norm, v_lin / v_norm], axis=0)
        pitch = 0.0

    return axis_out, point_out, thetas_out, joint_type, xi_opt, pitch


def _solve_revolute_axis_point_fixed_axis_thetas(
    axis: np.ndarray,
    thetas: Sequence[float],
    observed_translations: np.ndarray,
) -> Optional[np.ndarray]:
    """Solve ``(I - R(theta_i)) p = t_i`` for a fixed revolute axis and cumulative theta series."""
    axis_arr = _normalize3(np.asarray(axis, dtype=np.float64).reshape(3))
    theta_arr = np.asarray(thetas, dtype=np.float64).reshape(-1)
    translations_arr = np.asarray(observed_translations, dtype=np.float64)
    if translations_arr.ndim != 2 or translations_arr.shape[1] != 3 or theta_arr.shape[0] != translations_arr.shape[0]:
        return None
    if not np.all(np.isfinite(axis_arr)) or not np.all(np.isfinite(theta_arr)) or not np.all(np.isfinite(translations_arr)):
        return None
    A = np.zeros((3, 3), dtype=np.float64)
    b = np.zeros(3, dtype=np.float64)
    for theta, translation in zip(theta_arr, translations_arr):
        R_theta = _so3_exp(axis_arr * float(theta))
        I_minus_R = np.eye(3, dtype=np.float64) - R_theta
        A += I_minus_R.T @ I_minus_R
        b += I_minus_R.T @ np.asarray(translation, dtype=np.float64)
    try:
        point, *_ = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError:
        return None
    if point.shape != (3,) or not np.all(np.isfinite(point)):
        return None
    return point.astype(np.float64)


def _score_revolute_pivot_candidate(
    axis: np.ndarray,
    point: np.ndarray,
    thetas: Sequence[float],
    observed_translations: np.ndarray,
) -> Optional[dict]:
    """Score one revolute pivot candidate against observed cumulative translations."""
    axis_arr = _normalize3(np.asarray(axis, dtype=np.float64).reshape(3))
    point_arr = np.asarray(point, dtype=np.float64).reshape(3)
    theta_arr = np.asarray(thetas, dtype=np.float64).reshape(-1)
    translations_arr = np.asarray(observed_translations, dtype=np.float64)
    if translations_arr.ndim != 2 or translations_arr.shape[1] != 3 or theta_arr.shape[0] != translations_arr.shape[0]:
        return None
    if not np.all(np.isfinite(axis_arr)) or not np.all(np.isfinite(point_arr)) or not np.all(np.isfinite(theta_arr)) or not np.all(np.isfinite(translations_arr)):
        return None
    residuals = []
    for theta, observed_t in zip(theta_arr, translations_arr):
        R_theta = _so3_exp(axis_arr * float(theta))
        predicted_t = (np.eye(3, dtype=np.float64) - R_theta) @ point_arr
        residuals.append(float(np.linalg.norm(np.asarray(observed_t, dtype=np.float64) - predicted_t)))
    if not residuals:
        return None
    residuals_arr = np.asarray(residuals, dtype=np.float64)
    return {
        "translation_rmse_m": float(np.sqrt(np.mean(residuals_arr * residuals_arr))),
        "translation_mean_m": float(np.mean(residuals_arr)),
        "translation_max_m": float(np.max(residuals_arr)),
    }


def _build_articulation_axis_candidates(
    axis_direction: np.ndarray,
    pivot_candidates: Dict[str, Sequence[float]],
    pivot_candidate_scores: Dict[str, dict],
    chosen_pivot_source: Optional[str],
) -> List[dict]:
    """Format pivot candidates for DynamicSceneGraph / Viser debug rendering."""
    axis_arr = _normalize3(np.asarray(axis_direction, dtype=np.float64).reshape(3))
    if axis_arr.size != 3 or not np.all(np.isfinite(axis_arr)):
        return []
    color_map = {
        "point_bisector_free_traj": [230, 40, 40],
        "point_ls_fixed_axis_theta": [255, 170, 40],
        "point_track_cluster": [70, 210, 120],
        "point_seed": [90, 160, 255],
    }
    pretty_name = {
        "point_bisector_free_traj": "artipoint_bisector",
        "point_ls_fixed_axis_theta": "fixed_axis_ls",
        "point_track_cluster": "track_cluster",
        "point_seed": "seed",
    }
    items: List[dict] = []
    for name, pivot in pivot_candidates.items():
        pivot_arr = np.asarray(pivot, dtype=np.float64).reshape(-1)
        if pivot_arr.size != 3 or not np.all(np.isfinite(pivot_arr)):
            continue
        score = pivot_candidate_scores.get(name, {})
        rmse = score.get("translation_rmse_m")
        label = pretty_name.get(name, name)
        if rmse is not None and np.isfinite(float(rmse)):
            label = f"{label} ({float(rmse):.3f}m)"
        items.append(
            {
                "name": label,
                "axis_direction": axis_arr.astype(np.float32).tolist(),
                "axis_point": pivot_arr.astype(np.float32).tolist(),
                "color": list(color_map.get(name, [200, 200, 200])),
                "chosen_for_rendering": bool(name == chosen_pivot_source),
            }
        )
    return items


def fit_articulation_model_joint_optimization(
    pose_history: Sequence[dict],
    label: str = "",
    object_points: Optional[np.ndarray] = None,
    wrist_positions: Optional[Sequence[Optional[np.ndarray]]] = None,
    tracks_world: Optional[np.ndarray] = None,
    valid_tracks: Optional[np.ndarray] = None,
) -> dict:
    """Fit a robust 1-DoF articulation with a shared joint over all frames.

    This is the direct Artipoint-style replacement for the old pipeline stage:
    we fit the shared revolute/prismatic model from surface-track poses instead
    of first building a wrist-anchored LostFound trajectory and only then trying
    to clean it up.
    """
    if pose_history is None or len(pose_history) < ARTIPOINT_MIN_POSE_UPDATES:
        raise ValueError(
            f"Artipoint fit needs at least {ARTIPOINT_MIN_POSE_UPDATES} pose samples; "
            f"got {0 if pose_history is None else len(pose_history)}"
        )
    sorted_history = sorted(pose_history, key=lambda s: int(s.get("frame", 0)))

    # Preferred path: vendored ArtiPoint PoseEstFactorGraph. This is now the
    # DEFAULT articulated solver. Our custom closed-form / wrist / custom-GTSAM
    # path is retained strictly as a fallback when the native ArtiPoint solve
    # truly fails (unavailable import, unusable point-pair graph, empty
    # reconstruction), not because we prefer our own articulation math.
    native_fallback_reason: Optional[str] = None
    native_requested = tracks_world is not None and valid_tracks is not None
    if native_requested:
        if not ARTIPOINT_AVAILABLE or ArtiPointPoseEstFactorGraph is None:
            native_fallback_reason = "native_artipoint_unavailable"
            print(
                f"[ArticulationFit] '{label}': native ArtiPoint unavailable; "
                "using custom fallback articulation path."
            )
        else:
            native = artipoint_native_fit_from_tracks(tracks_world, valid_tracks)
            if native is None:
                native_fallback_reason = "native_artipoint_failed"
                print(
                    f"[ArticulationFit] '{label}': native ArtiPoint failed; "
                    "using custom fallback articulation path."
                )
            else:
                print(
                    f"[ArticulationFit] '{label}': using native ArtiPoint factor-graph path as the primary articulated solver."
                )
                print(
                    f"[ArtiPointNative] '{label}': joint={native['joint_type']}, "
                    f"pitch={native['pitch']:+.3f}, "
                    f"axis=({native['axis_direction'][0]:+.2f},"
                    f"{native['axis_direction'][1]:+.2f},"
                    f"{native['axis_direction'][2]:+.2f}), "
                    f"pivot=({native['axis_point'][0]:+.2f},"
                    f"{native['axis_point'][1]:+.2f},"
                    f"{native['axis_point'][2]:+.2f})."
                )
                reference_pose = np.asarray(sorted_history[0]["pose"], dtype=np.float64)
                reference_pose_inv = np.linalg.inv(reference_pose)
                xi_canonical = np.asarray(native["xi"], dtype=np.float64)
                cumulative_thetas = np.asarray(native["thetas"], dtype=np.float64)
                free_T_per_frame = [
                    np.asarray(m, dtype=np.float64) for m in native.get("free_T_per_frame", [])
                ]
                free_trajectory: List[np.ndarray] = [reference_pose.copy()]
                for T_i in free_T_per_frame:
                    free_trajectory.append(np.asarray(T_i, dtype=np.float64) @ free_trajectory[-1])
                use_free = len(free_T_per_frame) > 0
                if use_free:
                    print(
                        f"[ArtiPointNative] '{label}': rendering cloud via free trajectory "
                        f"({len(free_T_per_frame)} per-frame T(i) matrices); "
                        f"twist used only for joint description."
                    )
                else:
                    print(
                        f"[ArtiPointNative] '{label}': free trajectory unavailable; "
                        f"falling back to twist-reconstructed cloud poses."
                    )
                recon_history: List[dict] = []
                observed_used: List[np.ndarray] = []
                theta_history: List[float] = []
                for sample in sorted_history:
                    tf_idx = int(sample.get("track_frame_idx", -1))
                    if tf_idx < 0 or tf_idx >= cumulative_thetas.shape[0]:
                        continue
                    theta_at_frame = float(cumulative_thetas[tf_idx])
                    if use_free and tf_idx < len(free_trajectory):
                        pose_recon = free_trajectory[tf_idx]
                    elif _gtsam is not None:
                        delta = np.asarray(
                            _gtsam.Pose3.Expmap(xi_canonical * theta_at_frame).matrix(),
                            dtype=np.float64,
                        )
                        pose_recon = delta @ reference_pose
                    else:
                        pose_recon = reference_pose.copy()
                    recon_history.append(
                        {
                            "frame": int(sample.get("frame", 0)),
                            "pose": pose_recon.astype(np.float32),
                            "source": "artipoint_native_pose_est_free_traj" if use_free else "artipoint_native_pose_est_twist",
                            "confidence": 1.0,
                            "clip_id": int(sample.get("clip_id", -1)),
                            "theta": theta_at_frame,
                        }
                    )
                    observed_used.append(np.asarray(sample["pose"], dtype=np.float64))
                    theta_history.append(theta_at_frame)
                if not recon_history:
                    native_fallback_reason = "native_artipoint_empty_reconstruction"
                    print(
                        f"[ArticulationFit] '{label}': native ArtiPoint produced no aligned reconstructed poses; "
                        "using custom fallback articulation path."
                    )
                else:
                    residuals_t = []
                    residuals_r = []
                    for entry, observed in zip(recon_history, observed_used):
                        recon = np.asarray(entry["pose"], dtype=np.float64)
                        residuals_t.append(float(np.linalg.norm(observed[:3, 3] - recon[:3, 3])))
                        residuals_r.append(
                            float(
                                np.linalg.norm(
                                    _so3_log_axis_angle(recon[:3, :3].T @ observed[:3, :3])
                                )
                            )
                        )
                    native["reconstructed_pose_history"] = recon_history
                    native["mean_residual_m"] = float(np.mean(residuals_t)) if residuals_t else 0.0
                    native["max_residual_m"] = float(np.max(residuals_t)) if residuals_t else 0.0
                    native["mean_rotation_residual_rad"] = float(np.mean(residuals_r)) if residuals_r else 0.0
                    native["singular_ratio"] = 0.0
                    native["optimizer"] = "artipoint_native_pose_est_primary"
                    native["solver_path"] = "native_artipoint_primary"
                    native["fallback_reason"] = None
                    if (
                        not np.isfinite(float(native["mean_residual_m"]))
                        or float(native["mean_residual_m"]) > float(ARTIPOINT_NATIVE_MAX_MEAN_RESIDUAL_M)
                    ):
                        print(
                            f"[ArtiPointNative][Warn] '{label}': mean residual "
                            f"{float(native['mean_residual_m']):.3f} m exceeds the old warning cap "
                            f"{float(ARTIPOINT_NATIVE_MAX_MEAN_RESIDUAL_M):.3f} m, but we keep the native fit "
                            "because native ArtiPoint is now the primary articulated solver."
                        )

                    pivot_candidates: Dict[str, List[float]] = {}
                    pivot_candidate_scores: Dict[str, dict] = {}
                    chosen_pivot_source: Optional[str] = None
                    articulation_axis_candidates: List[dict] = []
                    joint_type_native = str(native.get("joint_type", "revolute")).lower()
                    axis_native = _normalize3(np.asarray(native.get("axis_direction", [0.0, 0.0, 1.0]), dtype=np.float64))
                    theta_history_arr = np.asarray(theta_history, dtype=np.float64)
                    if joint_type_native == "revolute" and observed_used:
                        observed_cumulative_translations = np.stack(
                            [(observed @ reference_pose_inv)[:3, 3] for observed in observed_used],
                            axis=0,
                        )
                        candidate_priority = [
                            "point_ls_fixed_axis_theta",
                            "point_track_cluster",
                            "point_seed",
                            "point_bisector_free_traj",
                        ]
                        candidate_priority_idx = {name: idx for idx, name in enumerate(candidate_priority)}

                        def _register_native_pivot_candidate(name: str, candidate_point: Optional[np.ndarray]) -> None:
                            nonlocal pivot_candidates, pivot_candidate_scores
                            if candidate_point is None:
                                return
                            candidate_arr = np.asarray(candidate_point, dtype=np.float64).reshape(-1)
                            if candidate_arr.size != 3 or not np.all(np.isfinite(candidate_arr)):
                                return
                            score = _score_revolute_pivot_candidate(
                                axis_native,
                                candidate_arr,
                                theta_history_arr,
                                observed_cumulative_translations,
                            )
                            if score is None:
                                return
                            pivot_candidates[name] = candidate_arr.astype(np.float32).tolist()
                            pivot_candidate_scores[name] = score

                        _register_native_pivot_candidate("point_bisector_free_traj", native.get("axis_point"))
                        point_ls_fixed_axis_theta = _solve_revolute_axis_point_fixed_axis_thetas(
                            axis_native,
                            theta_history_arr,
                            observed_cumulative_translations,
                        )
                        _register_native_pivot_candidate("point_ls_fixed_axis_theta", point_ls_fixed_axis_theta)
                        hypotheses = per_track_revolute_hypotheses(
                            np.asarray(tracks_world, dtype=np.float64),
                            np.asarray(valid_tracks, dtype=bool),
                        )
                        clip_diameter_m = 0.0
                        if object_points is not None:
                            object_pts_arr = np.asarray(object_points, dtype=np.float64)
                            finite_object_pts = object_pts_arr[np.all(np.isfinite(object_pts_arr), axis=1)]
                            if finite_object_pts.size:
                                clip_diameter_m = float(
                                    np.linalg.norm(finite_object_pts.max(axis=0) - finite_object_pts.min(axis=0))
                                )
                        if clip_diameter_m <= 1e-6 and hypotheses:
                            pivots_arr = np.stack([np.asarray(item["pivot"], dtype=np.float64) for item in hypotheses], axis=0)
                            clip_diameter_m = float(np.linalg.norm(pivots_arr.max(axis=0) - pivots_arr.min(axis=0)))
                        cluster = None
                        if len(hypotheses) >= int(ARTIPOINT_CLUSTER_MIN_HYPOTHESES) and clip_diameter_m > 1e-6:
                            cluster = cluster_revolute_hypotheses(hypotheses, clip_diameter_m)
                        if (
                            cluster is not None
                            and bool(cluster.get("axis_coherent", False))
                            and int(cluster.get("n_inliers", 0)) >= int(ARTIPOINT_CLUSTER_MIN_SIZE)
                            and int(cluster.get("n_total", 0)) >= int(ARTIPOINT_CLUSTER_MIN_HYPOTHESES)
                        ):
                            _register_native_pivot_candidate("point_track_cluster", cluster.get("pivot"))

                        if pivot_candidate_scores:
                            chosen_pivot_source = min(
                                pivot_candidate_scores,
                                key=lambda name: (
                                    float(pivot_candidate_scores[name].get("translation_rmse_m", float("inf"))),
                                    int(candidate_priority_idx.get(name, len(candidate_priority))),
                                ),
                            )
                            native["axis_point"] = np.asarray(pivot_candidates[chosen_pivot_source], dtype=np.float32).tolist()
                            articulation_axis_candidates = _build_articulation_axis_candidates(
                                axis_native,
                                pivot_candidates,
                                pivot_candidate_scores,
                                chosen_pivot_source,
                            )
                            print(
                                f"[ArticulationFit] '{label}': native ArtiPoint chose pivot {chosen_pivot_source} with "
                                f"translation RMSE {float(pivot_candidate_scores[chosen_pivot_source]['translation_rmse_m']):.4f} m."
                            )
                            if "point_bisector_free_traj" in pivot_candidate_scores or "point_ls_fixed_axis_theta" in pivot_candidate_scores:
                                print(
                                    f"[ArticulationFit] '{label}': pivot score summary "
                                    f"bisector={pivot_candidate_scores.get('point_bisector_free_traj', {}).get('translation_rmse_m', float('nan')):.4f} m, "
                                    f"fixed_axis_ls={pivot_candidate_scores.get('point_ls_fixed_axis_theta', {}).get('translation_rmse_m', float('nan')):.4f} m"
                                )

                    native["pivot_candidates"] = pivot_candidates
                    native["pivot_candidate_scores"] = pivot_candidate_scores
                    native["chosen_pivot_source"] = chosen_pivot_source
                    native["articulation_axis_candidates"] = articulation_axis_candidates
                    return native
    else:
        native_fallback_reason = "missing_tracks_world_or_valid_tracks"

    # ArtiPoint-native mode: tracks_world / valid_tracks are plumbed in by the
    # caller. We skip the Procrustes-based defensive filters (RMSE outlier +
    # continuity gate) because ArtiPoint's factor graph operates on raw 3D
    # point pairs with Huber-robust loss — defends against bad data at the
    # factor level rather than the frame level.
    going_artipoint_native = tracks_world is not None and valid_tracks is not None
    # Reject pose samples whose RANSAC-Kabsch residual is way above the median —
    # they're the source of the +2 rad theta jumps we saw in event.json.
    # Disabled in ArtiPoint-native mode.
    if not going_artipoint_native and len(sorted_history) >= 8:
        rmses = [
            float(sample.get("rigid_rmse_m", 0.0)) for sample in sorted_history
        ]
        finite = [r for r in rmses if np.isfinite(r) and r > 0.0]
        if finite:
            median_rmse = float(np.median(finite))
            cap = max(
                float(ARTIPOINT_POSE_RMSE_OUTLIER_MULT) * median_rmse,
                float(ARTIPOINT_POSE_RMSE_OUTLIER_MIN_M),
            )
            kept = [
                sample
                for sample, rmse in zip(sorted_history, rmses)
                if rmse <= cap or not np.isfinite(rmse)
            ]
            if len(kept) >= ARTIPOINT_MIN_POSE_UPDATES:
                if len(kept) < len(sorted_history):
                    print(
                        f"[ArticulationFit] rejecting {len(sorted_history) - len(kept)} "
                        f"outlier pose frame(s) with RMSE > {cap*100:.2f} cm "
                        f"(median {median_rmse*100:.2f} cm)."
                    )
                sorted_history = kept
    # Continuity gate: walk the chain and drop any frame whose rotation from
    # the previous *accepted* frame is implausibly large. These are the
    # Procrustes 180°-twin samples that drag axis_point toward the centroid.
    # Disabled in ArtiPoint-native mode (no Procrustes warm-start, so no
    # 180°-twin risk to defend against).
    if not going_artipoint_native and len(sorted_history) >= 4:
        raw_poses_seq = [np.asarray(s["pose"], dtype=np.float64) for s in sorted_history]
        kept_continuity = [sorted_history[0]]
        prev_pose = raw_poses_seq[0]
        rejected_flipped = 0
        for sample, pose in zip(sorted_history[1:], raw_poses_seq[1:]):
            try:
                rel = pose @ np.linalg.inv(prev_pose)
                omega = _so3_log_axis_angle(rel[:3, :3])
                jump = float(np.linalg.norm(omega))
            except Exception:
                jump = 0.0
            if np.isfinite(jump) and jump > float(ARTIPOINT_CONTINUITY_MAX_JUMP_RAD):
                rejected_flipped += 1
                continue
            kept_continuity.append(sample)
            prev_pose = pose
        if rejected_flipped > 0 and len(kept_continuity) >= ARTIPOINT_MIN_POSE_UPDATES:
            print(
                f"[ArticulationFit] continuity gate rejected {rejected_flipped} "
                f"frame(s) with |Δrotation| > {np.degrees(float(ARTIPOINT_CONTINUITY_MAX_JUMP_RAD)):.0f}° "
                f"(suspected Procrustes 180°-twins)."
            )
            sorted_history = kept_continuity
    poses = [np.asarray(sample["pose"], dtype=np.float64) for sample in sorted_history]
    frames = [int(sample["frame"]) for sample in sorted_history]
    clip_ids = [int(sample.get("clip_id", -1)) for sample in sorted_history]
    # Reference (= first kept frame's world pose) — used downstream to map the
    # reconstructed cumulative deltas back into world space, and to express the
    # observed cumulative delta of each frame for residual reporting.
    reference_pose = poses[0]
    reference_pose_inv = np.linalg.inv(reference_pose)
    # PER-FRAME-INCREMENT CONVENTION (matches ArtiPoint pose_est.py).
    # Procrustes upstream produced cumulative poses "reference → tracks[i]" for
    # each kept frame i. We now collapse those into consecutive-pair increments
    # ``inc[i] = pose[i+1] · pose[i]^{-1}`` so that the GTSAM ``T(i)`` variable
    # represents the small motion from sorted_history[i] to sorted_history[i+1].
    # That matches ArtiPoint's ``add_point_pair_factor`` which feeds raw 3D
    # point pairs ``(tracks[i], tracks[i+1])`` — and keeps the per-factor
    # residual bounded regardless of how long the clip swings. ``rotations[0]``
    # / ``translations[0]`` stay as identity placeholders so the downstream
    # closed-form code (which iterates from index 1) keeps working.
    increments_se3 = [
        poses[i + 1] @ np.linalg.inv(poses[i]) for i in range(len(poses) - 1)
    ]
    rotations = [np.eye(3, dtype=np.float64)] + [inc[:3, :3] for inc in increments_se3]
    translations = [np.zeros(3, dtype=np.float64)] + [inc[:3, 3] for inc in increments_se3]
    omegas = np.stack([_so3_log_axis_angle(rotation) for rotation in rotations], axis=0)
    max_rot = float(np.max(np.linalg.norm(omegas, axis=1)))

    closed = fit_articulation_model(sorted_history)
    if closed.get("joint_type") == "prismatic" or max_rot < 1e-3:
        closed["optimizer"] = "closed_form_prismatic"
        return closed

    axis0 = _normalize3(np.asarray(closed.get("axis_direction", [0.0, 0.0, 1.0]), dtype=np.float64))
    point0 = np.asarray(closed.get("axis_point", [0.0, 0.0, 0.0]), dtype=np.float64).reshape(3)
    # theta0 is initialised as PER-FRAME-INCREMENT angles to match the per-frame
    # T(i) convention. ``omegas`` are already per-frame (computed from per-frame
    # increments above), so ``omegas @ axis0`` gives the per-frame angles. We
    # do NOT use ``closed["thetas"]`` here because the closed-form computes
    # cumulative thetas internally and would seed the optimiser with the wrong
    # scale.
    theta0 = omegas @ axis0

    # --- Axis estimation: data-driven only. No label-based axis lock. ---
    # Preference order:
    #   (A) Keypoint Procrustes if the per-frame fit is clean (RMSE median low
    #       AND closed-form rotation-PCA has a dominant principal direction):
    #       trust the PCA axis from the per-frame omega vectors.
    #   (B) Wrist 3-D plane fit: the wrist is rigidly attached to the moving
    #       part during the interaction, so its trajectory lies in the plane
    #       perpendicular to the hinge axis. Plane normal = axis direction.
    #   (C) Free joint optimisation (fallback when both signals are bad).
    keypoint_rmses = [
        float(sample.get("rigid_rmse_m", float("inf")))
        for sample in sorted_history
        if np.isfinite(float(sample.get("rigid_rmse_m", float("inf"))))
        and float(sample.get("rigid_rmse_m", 0.0)) > 0.0
    ]
    keypoint_rmse_median = float(np.median(keypoint_rmses)) if keypoint_rmses else float("inf")
    closed_singular_ratio = float(closed.get("singular_ratio", 1.0))
    keypoints_clean = keypoint_rmse_median <= 0.05 and closed_singular_ratio <= 0.30

    # Wrist 3-D plane+circle fit (used both for axis when keypoints fail AND
    # as the source of axis_point + thetas in either case — the wrist is the
    # cleanest geometric anchor we have on the moving part).
    wrist_pts_finite: List[np.ndarray] = []
    if wrist_positions is not None:
        for w in wrist_positions:
            if w is None:
                continue
            arr = np.asarray(w, dtype=np.float64).reshape(-1)
            if arr.size == 3 and np.all(np.isfinite(arr)):
                wrist_pts_finite.append(arr)
    wrist_plane = (
        _fit_plane_3d(np.asarray(wrist_pts_finite, dtype=np.float64))
        if len(wrist_pts_finite) >= 3
        else None
    )

    axis_locked = False
    wrist_geometry_path = False
    axis_source = "free"
    if keypoints_clean:
        # (A) Keypoint axis: rotation-PCA over per-frame omegas (already in `closed`).
        axis0 = _normalize3(np.asarray(closed.get("axis_direction", axis0), dtype=np.float64))
        axis_locked = True
        axis_source = "keypoints"
    elif wrist_plane is not None and wrist_plane[2] < 0.30:
        # (B) Wrist plane axis: data-derived, no label assumption.
        plane_normal, _plane_pt, planarity = wrist_plane
        axis0 = _normalize3(plane_normal)
        # Orient so the recovered theta integrates to a positive value (right-hand rule
        # around axis0). We'll do this AFTER fitting thetas below.
        axis_locked = True
        axis_source = f"wrist_plane(planarity={planarity:.3f})"
    # else: keypoints noisy AND wrist plane fit either missing or non-planar
    #   -> fall through to the free optimiser starting from `closed`'s guess.

    # If axis is locked AND we have a wrist trajectory, fit the circle in the
    # plane perpendicular to the recovered axis. That gives axis_point and
    # per-frame thetas in one shot, regardless of where the axis came from.
    if axis_locked and len(wrist_pts_finite) >= 3:
        wrist_pts_arr = np.asarray(wrist_pts_finite, dtype=np.float64)
        plane_centroid_for_circle = wrist_pts_arr.mean(axis=0)
        circle_3d = _fit_circle_in_plane(wrist_pts_arr, axis0, plane_centroid_for_circle)
        if circle_3d is not None:
            center_3d, radius_3d, e_u, e_v = circle_3d
            # Plausibility check on the radius: a real handle-to-hinge distance
            # is roughly 0.05–1.5 m. Outside that, the wrist trajectory is more
            # likely jitter than a true arc, so we abandon the wrist path.
            if 0.05 <= radius_3d <= 1.5:
                wrist_thetas = _angles_from_plane_projection(
                    wrist_positions, center_3d, e_u, e_v,
                )
                if wrist_thetas is not None and wrist_thetas.shape[0] == len(sorted_history):
                    # Orient axis so the integrated swing is positive (arrow
                    # tip points in the swing direction).
                    if float(np.sum(wrist_thetas)) < 0.0:
                        axis0 = -axis0
                        # Recompute thetas with the flipped plane orientation
                        # (just negate; the projection is anti-symmetric).
                        wrist_thetas = -wrist_thetas
                    point0 = center_3d
                    # Convert cumulative wrist-derived thetas to per-frame
                    # increments to match the per-frame T(i) convention. ``theta0[0]``
                    # is the dummy "before frame 0" slot and stays at 0.
                    theta0 = np.zeros_like(wrist_thetas)
                    theta0[1:] = np.diff(wrist_thetas)
                    wrist_geometry_path = True
                    print(
                        f"[ArticulationFit] '{label}': data-driven articulation. "
                        f"axis_source={axis_source}, "
                        f"axis_dir=({axis0[0]:.2f}, {axis0[1]:.2f}, {axis0[2]:.2f}), "
                        f"axis_point=({center_3d[0]:.2f}, {center_3d[1]:.2f}, {center_3d[2]:.2f}), "
                        f"radius={radius_3d:.3f} m, sweep "
                        f"{float(np.degrees(float(wrist_thetas.min()))):.1f}° .. "
                        f"{float(np.degrees(float(wrist_thetas.max()))):.1f}°"
                    )

    # Soft prior axis (purely for logging / future tiebreakers; not enforced).
    prior_axis = _artipoint_axis_prior(label)

    if axis_locked:
        x0 = np.concatenate([point0, theta0[1:]], axis=0)
    else:
        x0 = np.concatenate([axis0, point0, theta0[1:]], axis=0)
    trans_scale = max(float(ARTIPOINT_TRANSLATION_RESIDUAL_SCALE_M), 1e-4)

    monotonic_sign = float(np.sign(float(np.sum(theta0)))) or 1.0

    def residual(params: np.ndarray) -> np.ndarray:
        if axis_locked:
            axis = axis0
            point = np.asarray(params[:3], dtype=np.float64)
            theta_vars = np.concatenate([[0.0], np.asarray(params[3:], dtype=np.float64)], axis=0)
        else:
            axis = _normalize3(params[:3], axis0)
            point = np.asarray(params[3:6], dtype=np.float64)
            theta_vars = np.concatenate([[0.0], np.asarray(params[6:], dtype=np.float64)], axis=0)
        parts: List[np.ndarray] = []
        for R_obs, t_obs, theta in zip(rotations[1:], translations[1:], theta_vars[1:]):
            R_pred = _so3_exp(axis * float(theta))
            t_pred = (np.eye(3) - R_pred) @ point
            rot_res = _so3_log_axis_angle(R_pred.T @ R_obs)
            trans_res = (t_pred - t_obs) / trans_scale
            parts.extend([rot_res, trans_res])
        # Temporal smoothness on theta (velocity): keep adjacent thetas close.
        vel_w = float(ARTIPOINT_THETA_VELOCITY_WEIGHT)
        if vel_w > 0.0 and len(theta_vars) >= 2:
            theta_velocity = np.diff(theta_vars)
            parts.append(theta_velocity * vel_w)
        # Jerk smoothness on theta (second difference): pushes against sign flips.
        jerk_w = float(ARTIPOINT_THETA_JERK_WEIGHT)
        if jerk_w > 0.0 and len(theta_vars) >= 3:
            theta_jerk = np.diff(theta_vars, n=2)
            parts.append(theta_jerk * jerk_w)
        # Monotonicity: a door swing is one-directional. Penalize backward steps
        # (theta_velocity opposing the global sign of motion). Use a soft hinge.
        mono_w = float(ARTIPOINT_THETA_MONOTONIC_WEIGHT)
        if mono_w > 0.0 and len(theta_vars) >= 2:
            theta_velocity = np.diff(theta_vars)
            backward = np.maximum(0.0, -monotonic_sign * theta_velocity)
            parts.append(backward * mono_w)
        # Only emit the soft prior penalty when axis is FREE; when locked, the
        # axis is the prior by construction, so the cross product is zero.
        if prior_axis is not None and not axis_locked:
            parts.append(np.cross(axis, prior_axis) * float(ARTIPOINT_DOOR_AXIS_PRIOR_WEIGHT))
        return np.concatenate(parts, axis=0) if parts else np.zeros((0,), dtype=np.float64)

    optimizer_label = "custom_gtsam_fallback_unoptimized"
    gtsam_joint_type: Optional[str] = None
    gtsam_twist: Optional[np.ndarray] = None
    gtsam_pitch: Optional[float] = None
    # Build per-frame point pairs for the ArtiPoint-faithful data-fit path.
    # PER-FRAME-INCREMENT convention: GTSAM's ``T(i)`` variable represents the
    # motion ``sorted_history[i] → sorted_history[i+1]``. The point-pair factor
    # therefore needs ``(tracks[sorted_history[i].tf_idx], tracks[sorted_history[i+1].tf_idx])``
    # — consecutive-pair tracks, matching ArtiPoint's ``create_points_pairs``.
    point_pairs_per_frame: Optional[List[Optional[Tuple[np.ndarray, np.ndarray]]]] = None
    n_point_pair_frames = 0
    if tracks_world is not None and valid_tracks is not None and len(sorted_history) >= 2:
        tracks_arr = np.asarray(tracks_world, dtype=np.float64)
        valid_arr = np.asarray(valid_tracks, dtype=bool)
        if (
            tracks_arr.ndim == 3
            and tracks_arr.shape[2] == 3
            and valid_arr.shape[:2] == tracks_arr.shape[:2]
        ):
            point_pairs_per_frame = []
            for i in range(len(sorted_history) - 1):
                tf_a = int(sorted_history[i].get("track_frame_idx", -1))
                tf_b = int(sorted_history[i + 1].get("track_frame_idx", -1))
                if (
                    tf_a < 0 or tf_a >= tracks_arr.shape[0]
                    or tf_b < 0 or tf_b >= tracks_arr.shape[0]
                ):
                    point_pairs_per_frame.append(None)
                    continue
                pair_valid = (
                    valid_arr[tf_a]
                    & valid_arr[tf_b]
                    & np.all(np.isfinite(tracks_arr[tf_a]), axis=1)
                    & np.all(np.isfinite(tracks_arr[tf_b]), axis=1)
                )
                if int(np.count_nonzero(pair_valid)) < 3:
                    point_pairs_per_frame.append(None)
                    continue
                p_arr = tracks_arr[tf_a, pair_valid]
                q_arr = tracks_arr[tf_b, pair_valid]
                point_pairs_per_frame.append((p_arr, q_arr))
                n_point_pair_frames += 1
    if point_pairs_per_frame is not None:
        print(
            f"[ArticulationFit] '{label}': using point-pair factors on "
            f"{n_point_pair_frames}/{len(sorted_history) - 1} frames "
            f"(ArtiPoint-faithful path)."
        )
    if not GTSAM_AVAILABLE:
        raise RuntimeError(
            f"[ArticulationFit][Error] '{label}': gtsam unavailable and scipy fallback is disabled."
        )
    try:
        axis_gt, point_gt, thetas_gt, joint_type_gt, twist_gt, pitch_gt = _fit_articulation_gtsam(
            rotations=rotations,
            translations=translations,
            axis0=axis0,
            point0=point0,
            theta0=theta0,
            trans_scale=trans_scale,
            vel_w=float(ARTIPOINT_THETA_VELOCITY_WEIGHT),
            jerk_w=float(ARTIPOINT_THETA_JERK_WEIGHT),
            mono_w=float(ARTIPOINT_THETA_MONOTONIC_WEIGHT),
            prior_axis=prior_axis,
            prior_axis_w=float(ARTIPOINT_DOOR_AXIS_PRIOR_WEIGHT),
            axis_locked=axis_locked,
            point_pairs=point_pairs_per_frame,
        )
        axis = axis_gt
        point = point_gt
        thetas = thetas_gt
        opt = None
        optimizer_label = "custom_gtsam_factor_graph_huber_fallback"
        gtsam_joint_type = joint_type_gt
        gtsam_twist = twist_gt
        gtsam_pitch = pitch_gt
        print(
            f"[ArticulationFit] '{label}': GTSAM twist fit. "
            f"joint_type={joint_type_gt}, pitch={pitch_gt:+.3f}, "
            f"axis=({axis[0]:.2f}, {axis[1]:.2f}, {axis[2]:.2f}), "
            f"point=({point[0]:.2f}, {point[1]:.2f}, {point[2]:.2f})"
        )
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"[ArticulationFit][Error] '{label}': GTSAM fit failed and scipy fallback is disabled: {exc}"
        ) from exc
    if float(np.sum(thetas)) < 0.0:
        axis = -axis
        thetas = -thetas
        if gtsam_twist is not None:
            gtsam_twist = -gtsam_twist

    reported_thetas = np.asarray(thetas, dtype=np.float64)
    if gtsam_twist is not None and gtsam_joint_type is not None:
        omega_mag = float(np.linalg.norm(gtsam_twist[:3]))
        v_mag = float(np.linalg.norm(gtsam_twist[3:]))
        theta_scale = omega_mag if gtsam_joint_type == "revolute" else v_mag
        reported_thetas = reported_thetas * max(theta_scale, 1e-9)
        if gtsam_joint_type == "revolute":
            print(
                f"[ArticulationFit] '{label}': reported revolute range "
                f"{float(np.degrees(np.min(reported_thetas))):.1f}° .. "
                f"{float(np.degrees(np.max(reported_thetas))):.1f}°"
            )
        else:
            print(
                f"[ArticulationFit] '{label}': reported prismatic range "
                f"{float(np.min(reported_thetas)):.3f} m .. "
                f"{float(np.max(reported_thetas)):.3f} m"
            )

    pivot_candidates: Dict[str, List[float]] = {}
    pivot_candidate_scores: Dict[str, dict] = {}
    chosen_pivot_source: Optional[str] = None
    observed_cumulative_translations = np.stack(
        [(np.asarray(observed, dtype=np.float64) @ reference_pose_inv)[:3, 3] for observed in poses],
        axis=0,
    )
    if (gtsam_joint_type or "revolute") == "revolute":
        candidate_priority = [
            "point_ls_fixed_axis_theta",
            "point_track_cluster",
            "point_seed",
            "point_bisector_free_traj",
        ]
        candidate_priority_idx = {name: idx for idx, name in enumerate(candidate_priority)}

        def _register_pivot_candidate(name: str, candidate_point: Optional[np.ndarray]) -> None:
            nonlocal pivot_candidates, pivot_candidate_scores
            if candidate_point is None:
                return
            candidate_arr = np.asarray(candidate_point, dtype=np.float64).reshape(-1)
            if candidate_arr.size != 3 or not np.all(np.isfinite(candidate_arr)):
                return
            score = _score_revolute_pivot_candidate(
                axis,
                candidate_arr,
                reported_thetas,
                observed_cumulative_translations,
            )
            if score is None:
                return
            pivot_candidates[name] = candidate_arr.astype(np.float32).tolist()
            pivot_candidate_scores[name] = score

        _register_pivot_candidate("point_seed", point0)
        _register_pivot_candidate("point_bisector_free_traj", point_gt)
        point_ls_fixed_axis_theta = _solve_revolute_axis_point_fixed_axis_thetas(
            axis,
            reported_thetas,
            observed_cumulative_translations,
        )
        _register_pivot_candidate("point_ls_fixed_axis_theta", point_ls_fixed_axis_theta)
        if tracks_world is not None and valid_tracks is not None:
            hypotheses = per_track_revolute_hypotheses(
                np.asarray(tracks_world, dtype=np.float64),
                np.asarray(valid_tracks, dtype=bool),
            )
            clip_diameter_m = 0.0
            if object_points is not None:
                object_pts_arr = np.asarray(object_points, dtype=np.float64)
                finite_object_pts = object_pts_arr[np.all(np.isfinite(object_pts_arr), axis=1)]
                if finite_object_pts.size:
                    clip_diameter_m = float(np.linalg.norm(finite_object_pts.max(axis=0) - finite_object_pts.min(axis=0)))
            if clip_diameter_m <= 1e-6 and hypotheses:
                pivots_arr = np.stack([np.asarray(item["pivot"], dtype=np.float64) for item in hypotheses], axis=0)
                clip_diameter_m = float(np.linalg.norm(pivots_arr.max(axis=0) - pivots_arr.min(axis=0)))
            cluster = None
            if len(hypotheses) >= int(ARTIPOINT_CLUSTER_MIN_HYPOTHESES) and clip_diameter_m > 1e-6:
                cluster = cluster_revolute_hypotheses(hypotheses, clip_diameter_m)
            if (
                cluster is not None
                and bool(cluster.get("axis_coherent", False))
                and int(cluster.get("n_inliers", 0)) >= int(ARTIPOINT_CLUSTER_MIN_SIZE)
                and int(cluster.get("n_total", 0)) >= int(ARTIPOINT_CLUSTER_MIN_HYPOTHESES)
            ):
                _register_pivot_candidate("point_track_cluster", cluster.get("pivot"))

        if pivot_candidate_scores:
            chosen_pivot_source = min(
                pivot_candidate_scores,
                key=lambda name: (
                    float(pivot_candidate_scores[name].get("translation_rmse_m", float("inf"))),
                    int(candidate_priority_idx.get(name, len(candidate_priority))),
                ),
            )
            point = np.asarray(pivot_candidates[chosen_pivot_source], dtype=np.float64)
            print(
                f"[ArticulationFit] '{label}': chose pivot {chosen_pivot_source} with translation RMSE "
                f"{float(pivot_candidate_scores[chosen_pivot_source]['translation_rmse_m']):.4f} m."
            )
            if "point_bisector_free_traj" in pivot_candidate_scores or "point_ls_fixed_axis_theta" in pivot_candidate_scores:
                print(
                    f"[ArticulationFit] '{label}': pivot score summary "
                    f"bisector={pivot_candidate_scores.get('point_bisector_free_traj', {}).get('translation_rmse_m', float('nan')):.4f} m, "
                    f"fixed_axis_ls={pivot_candidate_scores.get('point_ls_fixed_axis_theta', {}).get('translation_rmse_m', float('nan')):.4f} m"
                )

    reconstructed_pose_history: List[dict] = []
    residuals_t: List[float] = []
    residuals_r: List[float] = []
    for frame, clip_id, observed, theta_raw, theta_value in zip(frames, clip_ids, poses, thetas, reported_thetas):
        if (gtsam_joint_type or "revolute") == "revolute":
            R_recon = _so3_exp(axis * float(theta_raw))
            t_recon = (np.eye(3) - R_recon) @ point
            delta_recon = np.eye(4, dtype=np.float64)
            delta_recon[:3, :3] = R_recon
            delta_recon[:3, 3] = t_recon
        elif gtsam_twist is not None:
            delta_recon = np.asarray(_gtsam.Pose3.Expmap(gtsam_twist * float(theta_raw)).matrix(), dtype=np.float64)
            R_recon = delta_recon[:3, :3]
            t_recon = delta_recon[:3, 3]
        else:
            R_recon = _so3_exp(axis * float(theta_raw))
            t_recon = (np.eye(3) - R_recon) @ point
            delta_recon = np.eye(4, dtype=np.float64)
            delta_recon[:3, :3] = R_recon
            delta_recon[:3, 3] = t_recon
        pose_recon = delta_recon @ reference_pose
        reconstructed_pose_history.append(
            {
                "frame": int(frame),
                "pose": pose_recon.astype(np.float32),
                "source": "artipoint_custom_fallback_joint_optimization",
                "confidence": 1.0,
                "clip_id": int(clip_id),
                "theta": float(theta_value),
            }
        )
        observed_delta = observed @ reference_pose_inv
        residuals_t.append(float(np.linalg.norm(observed_delta[:3, 3] - t_recon)))
        residuals_r.append(float(np.linalg.norm(_so3_log_axis_angle(R_recon.T @ observed_delta[:3, :3]))))

    if (gtsam_joint_type or "revolute") == "revolute":
        xi_linear = -np.cross(axis, point)
        xi_out = np.concatenate([axis, xi_linear], axis=0)
    elif gtsam_twist is not None:
        xi_out = np.asarray(gtsam_twist, dtype=np.float64)
    else:
        xi_out = np.concatenate([axis.astype(np.float64), point.astype(np.float64)], axis=0)

    articulation_axis_candidates = _build_articulation_axis_candidates(
        axis,
        pivot_candidates,
        pivot_candidate_scores,
        chosen_pivot_source,
    )
    return {
        "xi": np.asarray(xi_out, dtype=np.float32).tolist(),
        "thetas": [float(theta) for theta in reported_thetas],
        "axis_direction": axis.astype(np.float32).tolist(),
        "axis_point": point.astype(np.float32).tolist(),
        "joint_type": gtsam_joint_type if gtsam_joint_type is not None else "revolute",
        "reconstructed_pose_history": reconstructed_pose_history,
        "mean_residual_m": float(np.mean(residuals_t)),
        "max_residual_m": float(np.max(residuals_t)),
        "mean_rotation_residual_rad": float(np.mean(residuals_r)),
        "n_frames": len(sorted_history),
        "singular_ratio": float(closed.get("singular_ratio", 0.0)),
        "optimizer": optimizer_label,
        "solver_path": "custom_gtsam_fallback",
        "fallback_reason": native_fallback_reason,
        "optimization_cost": float(opt.cost) if opt is not None else 0.0,
        "axis_prior": prior_axis.astype(np.float32).tolist() if prior_axis is not None else None,
        "pitch": float(gtsam_pitch) if gtsam_pitch is not None else None,
        "pivot_candidates": pivot_candidates,
        "pivot_candidate_scores": pivot_candidate_scores,
        "chosen_pivot_source": chosen_pivot_source,
        "articulation_axis_candidates": articulation_axis_candidates,
    }



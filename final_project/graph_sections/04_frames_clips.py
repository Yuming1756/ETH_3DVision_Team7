# ======================================================================
# graph.py section: 04_frames_clips.py
# Frame loading, clip segmentation, jpeg export, perception input prep
#
# NOTE: not a standalone module. Loaded by graph.py into a shared
# namespace via exec(); names from earlier sections are in scope here.
# ======================================================================
from __future__ import annotations
def load_all_frames(
    data_dir: Path,
    frame_start: int,
    frame_end: int,
    K_color: np.ndarray,
    K_depth: np.ndarray,
    T_cam_color: np.ndarray,
    T_depth_color: np.ndarray,
    pose_buffer: PoseBuffer,
) -> List[dict]:
    """Load frame paths, detections, hand poses, and LostFound-style interaction cues."""
    if not LOSTFOUND_AVAILABLE or hand_object_detection is None or load_faster_rcnn is None:
        raise RuntimeError("LostFound detection_utils is required for clip segmentation but is not available.")

    rgb_frames, rgb_timestamps = build_topic_index(data_dir / "camera_color_image", "*.png")
    depth_frames, depth_timestamps = build_topic_index(data_dir / "camera_depth_image", "*.png")
    pose_frames, pose_timestamps = build_topic_index(data_dir / "tf_camera", "*.json")
    tf_hands_frames, tf_hands_timestamps = build_topic_index(data_dir / "tf_hands", "*.json")
    left_hand_frames, left_hand_timestamps = build_topic_index(data_dir / "hands_left", "*.json")
    right_hand_frames, right_hand_timestamps = build_topic_index(data_dir / "hands_right", "*.json")
    rgb_frame_by_key = {frame.key: frame for frame in rgb_frames}
    pose_health = load_health_by_timestamp(data_dir / "tf_camera_health", missing_valid_default=False)
    if not pose_health:
        raise FileNotFoundError(f"tf_camera_health is required for pose gating under {data_dir / 'tf_camera_health'}")
    color_health = load_health_by_timestamp(data_dir / "camera_color_health")
    left_hand_health = load_health_by_timestamp(data_dir / "hands_left_health")
    right_hand_health = load_health_by_timestamp(data_dir / "hands_right_health")

    print("Running hand-object detector...")
    detector_cwd = os.getcwd()
    if LOSTFOUND_ROOT is not None:
        os.chdir(str(LOSTFOUND_ROOT))
    try:
        detector = load_faster_rcnn()
    finally:
        os.chdir(detector_cwd)

    samples: List[dict] = []
    for frame_idx in tqdm(range(frame_start, frame_end + 1), desc="  detecting"):
        key = f"{frame_idx:06d}"
        rgb_frame = rgb_frame_by_key.get(key)
        if rgb_frame is None:
            continue
        depth_frame = nearest_topic_frame(depth_frames, depth_timestamps, rgb_frame.timestamp_ns)
        if depth_frame is None:
            continue
        pose_frame = nearest_topic_frame(pose_frames, pose_timestamps, depth_frame.timestamp_ns)
        if pose_frame is None:
            continue
        tf_hands_frame = nearest_topic_frame(tf_hands_frames, tf_hands_timestamps, rgb_frame.timestamp_ns)
        left_hand_frame = nearest_topic_frame(left_hand_frames, left_hand_timestamps, rgb_frame.timestamp_ns)
        right_hand_frame = nearest_topic_frame(right_hand_frames, right_hand_timestamps, rgb_frame.timestamp_ns)

        bgr = cv2.imread(str(rgb_frame.path))
        depth_raw = cv2.imread(str(depth_frame.path), cv2.IMREAD_ANYDEPTH)
        if bgr is None or depth_raw is None:
            continue
        T_wc_rgb = rt_to_matrix(*pose_buffer.camera_pose(rgb_frame.timestamp_ns))
        T_wc_depth = rt_to_matrix(*pose_buffer.camera_pose(depth_frame.timestamp_ns))
        T_wc_hands = rt_to_matrix(*pose_buffer.camera_pose(tf_hands_frame.timestamp_ns)) if tf_hands_frame is not None else T_wc_rgb
        pose_valid = bool(pose_health.get(pose_frame.timestamp_ns, False))
        rgb_valid = color_health.get(rgb_frame.timestamp_ns, True) if not color_health else color_health.get(rgb_frame.timestamp_ns, False)
        left_hand_valid = (
            left_hand_health.get(left_hand_frame.timestamp_ns, True) if left_hand_frame is not None and not left_hand_health else
            left_hand_health.get(left_hand_frame.timestamp_ns, False) if left_hand_frame is not None else False
        )
        right_hand_valid = (
            right_hand_health.get(right_hand_frame.timestamp_ns, True) if right_hand_frame is not None and not right_hand_health else
            right_hand_health.get(right_hand_frame.timestamp_ns, False) if right_hand_frame is not None else False
        )
        if rgb_valid:
            hand_dets, obj_dets = hand_object_detection(bgr, detector)
            contact_breakdown = hand_object_contact_breakdown(hand_dets, obj_dets)
            contact = bool(contact_breakdown["contact"])
        else:
            hand_dets, obj_dets = None, None
            contact_breakdown = {
                "hand_dets_present": False,
                "obj_dets_present": False,
                "hand_confident": False,
                "obj_confident": False,
                "hand_in_contact": False,
                "best_hand_score": 0.0,
                "best_obj_score": 0.0,
                "contact": False,
                "first_failure": "rgb_invalid",
            }
            contact = False
        left_hand_wrist_world, right_hand_wrist_world = load_hand_world_wrist_positions(
            tf_hands_frame.path if tf_hands_frame is not None else None,
            T_wc_hands,
        )
        left_hand_world = left_hand_wrist_world
        right_hand_world = right_hand_wrist_world
        if not left_hand_valid:
            left_hand_world = None
            left_hand_wrist_world = None
        if not right_hand_valid:
            right_hand_world = None
            right_hand_wrist_world = None
        left_nearest_distance = estimate_nearest_detected_object_distance(
            left_hand_world,
            obj_dets,
            depth_raw,
            bgr.shape,
            K_color,
            K_depth,
            T_depth_color,
            T_wc_rgb,
            T_cam_color,
        )
        right_nearest_distance = estimate_nearest_detected_object_distance(
            right_hand_world,
            obj_dets,
            depth_raw,
            bgr.shape,
            K_color,
            K_depth,
            T_depth_color,
            T_wc_rgb,
            T_cam_color,
        )
        samples.append(
            {
                "prefix": key,
                "frame": int(frame_idx),
                "rgb_path": rgb_frame.path,
                "depth_path": depth_frame.path,
                "pose_path": pose_frame.path,
                "rgb_timestamp_ns": int(rgb_frame.timestamp_ns),
                "depth_timestamp_ns": int(depth_frame.timestamp_ns),
                "pose_timestamp_ns": int(pose_frame.timestamp_ns),
                "pose_valid": bool(pose_valid),
                "rgb_camera_pose_matrix": np.asarray(T_wc_rgb, dtype=np.float32),
                "depth_camera_pose_matrix": np.asarray(T_wc_depth, dtype=np.float32),
                "rgb_valid": bool(rgb_valid),
                "left_hand_valid": bool(left_hand_valid),
                "right_hand_valid": bool(right_hand_valid),
                "hand_contact": contact,
                "hand_contact_breakdown": contact_breakdown,
                "hand_dets": hand_dets,
                "obj_dets": obj_dets,
                "left_hand_world": left_hand_world,
                "right_hand_world": right_hand_world,
                "left_hand_world_wrist": left_hand_wrist_world,
                "right_hand_world_wrist": right_hand_wrist_world,
                "left_hand_world_wrist_estimated": False,
                "right_hand_world_wrist_estimated": False,
                "left_nearest_object_distance": left_nearest_distance,
                "right_nearest_object_distance": right_nearest_distance,
            }
        )

    del detector
    torch.cuda.empty_cache()
    gc.collect()
    annotate_pose_blocks(samples)
    print(f"Loaded {len(samples)} frames ({frame_start}-{frame_end}), {sum(1 for s in samples if s['hand_contact'])} with contact")
    # Contact-failure breakdown: count how each prerequisite was the FIRST to
    # fail across non-contact frames. Tells you whether the hand detector, the
    # object detector, or the per-hand "in-contact" flag is the bottleneck.
    failure_counts: Dict[str, int] = {}
    for sample in samples:
        bd = sample.get("hand_contact_breakdown") or {}
        if bd.get("first_failure"):
            failure_counts[bd["first_failure"]] = failure_counts.get(bd["first_failure"], 0) + 1
    if failure_counts:
        breakdown_str = ", ".join(f"{k}={v}" for k, v in sorted(failure_counts.items(), key=lambda kv: -kv[1]))
        print(f"  contact-failure breakdown (per frame): {breakdown_str}")
    # Per-frame detail for frames where a HAND was detected but contact still
    # failed (i.e. NOT a total "no detection"). Pinpoints which of the three
    # prerequisites dropped — object detector missing/low-conf, or the per-hand
    # "in-contact" flag off — so a still-touching interaction whose contact flag
    # vanished (e.g. fridge once the door swings open) can be explained exactly
    # instead of lumped into "missing_dets".
    detail_rows: List[str] = []
    for sample in samples:
        bd = sample.get("hand_contact_breakdown") or {}
        if bd.get("contact") or not bd.get("hand_dets_present"):
            continue
        first_failure = bd.get("first_failure")
        reason = first_failure
        if first_failure == "missing_dets":
            reason = "obj_dets_missing" if not bd.get("obj_dets_present") else "hand_dets_missing"
        detail_rows.append(
            f"    frame {int(sample.get('prefix', -1))}: {reason} "
            f"(hand@{float(bd.get('best_hand_score', 0.0)):.2f}, "
            f"obj@{float(bd.get('best_obj_score', 0.0)):.2f}, "
            f"in_contact={bool(bd.get('hand_in_contact'))})"
        )
    if detail_rows:
        print(f"  contact-failure detail (hand detected, no contact), {len(detail_rows)} frame(s):")
        for row in detail_rows:
            print(row)
    return samples


def _sample_requires_new_clip(previous_sample: dict, current_sample: dict) -> bool:
    """Return whether clip construction should split before the current sample."""
    prev_frame_idx = int(previous_sample["prefix"])
    curr_frame_idx = int(current_sample["prefix"])
    if curr_frame_idx != (prev_frame_idx + 1):
        return True
    if previous_sample.get("pose_block_id") != current_sample.get("pose_block_id"):
        return True
    return False


def _build_clips_from_flag(samples: Sequence[dict], flag_key: str) -> List[ClipInfo]:
    """Split samples into alternating inactive and active clips using a boolean per-frame flag."""
    def _make_clip(clip_id: int, clip_samples: Sequence[dict], is_interaction: bool) -> ClipInfo:
        return ClipInfo(
            clip_id=clip_id,
            frame_start=int(clip_samples[0]["prefix"]),
            frame_end=int(clip_samples[-1]["prefix"]),
            is_inactive=not bool(is_interaction),
            hand_ratio=sum(1 for item in clip_samples if item.get("hand_contact")) / len(clip_samples),
            pose_block_id=clip_samples[0].get("pose_block_id"),
            samples=list(clip_samples),
        )

    clips: List[ClipInfo] = []
    if not samples:
        return clips

    raw_runs: List[dict] = []
    current_samples: List[dict] = []
    current_is_interaction: Optional[bool] = None
    for sample in samples:
        contact = bool(sample.get(flag_key, False))
        if not current_samples:
            current_samples = [sample]
            current_is_interaction = contact
            continue

        if _sample_requires_new_clip(current_samples[-1], sample) or contact != current_is_interaction:
            raw_runs.append(
                {
                    "samples": list(current_samples),
                    "is_interaction": bool(current_is_interaction),
                }
            )
            current_samples = [sample]
            current_is_interaction = contact
        else:
            current_samples.append(sample)

    if current_samples:
        raw_runs.append(
            {
                "samples": list(current_samples),
                "is_interaction": bool(current_is_interaction),
            }
        )

    idx = 0
    while idx < len(raw_runs):
        run = raw_runs[idx]
        if len(run["samples"]) >= MIN_CLIP_FRAMES:
            idx += 1
            continue
        prev_run = raw_runs[idx - 1] if idx > 0 else None
        next_run = raw_runs[idx + 1] if idx + 1 < len(raw_runs) else None
        if prev_run is not None and next_run is not None and prev_run["is_interaction"] == next_run["is_interaction"]:
            prev_run["samples"] = list(prev_run["samples"]) + list(run["samples"]) + list(next_run["samples"])
            raw_runs.pop(idx + 1)
            raw_runs.pop(idx)
            idx = max(idx - 1, 0)
            continue
        if prev_run is not None:
            prev_run["samples"] = list(prev_run["samples"]) + list(run["samples"])
            raw_runs.pop(idx)
            idx = max(idx - 1, 0)
            continue
        if next_run is not None:
            next_run["samples"] = list(run["samples"]) + list(next_run["samples"])
            raw_runs.pop(idx)
            continue
        idx += 1

    for clip_id, run in enumerate(raw_runs):
        clips.append(_make_clip(clip_id, run["samples"], run["is_interaction"]))
    return clips


def choose_clip_interaction_hand(clip_samples: Sequence[dict]) -> Optional[str]:
    """Choose one hand to own an active clip, without switching mid-clip."""
    if not clip_samples:
        return None

    def _active_indices(hand: str) -> List[int]:
        return [idx for idx, sample in enumerate(clip_samples) if sample.get(f"{hand}_active", False)]

    def _support_count(hand: str) -> int:
        dist_key = f"{hand}_nearest_object_distance"
        hand_key = f"{hand}_hand_world"
        return sum(
            1
            for sample in clip_samples
            if sample.get("hand_contact", False)
            and sample.get(hand_key) is not None
            and sample.get(dist_key) is not None
            and float(sample.get(dist_key)) < LOSTFOUND_HAND_DISTANCE_THRESHOLD
        )

    def _valid_count(hand: str) -> int:
        return sum(1 for sample in clip_samples if sample.get(f"{hand}_hand_world") is not None)

    def _median_distance(hand: str) -> float:
        values = [
            float(sample[f"{hand}_nearest_object_distance"])
            for sample in clip_samples
            if sample.get(f"{hand}_nearest_object_distance") is not None
        ]
        if not values:
            return float("inf")
        return float(np.median(np.asarray(values, dtype=np.float32)))

    active_by_hand = {hand: _active_indices(hand) for hand in ("left", "right")}
    earliest = [
        (indices[0], hand)
        for hand, indices in active_by_hand.items()
        if indices
    ]
    if earliest:
        first_index = min(index for index, _hand in earliest)
        candidates = [hand for index, hand in earliest if index == first_index]
        return max(
            candidates,
            key=lambda hand: (_support_count(hand), _valid_count(hand), -_median_distance(hand)),
        )

    scored = [
        (_support_count(hand), _valid_count(hand), -_median_distance(hand), hand)
        for hand in ("left", "right")
    ]
    best_support, _best_valid, _best_distance, best_hand = max(scored)
    if best_support <= 0:
        return None
    return str(best_hand)


def interpolate_missing_wrist_positions(clip_samples: Sequence[dict]) -> None:
    """Fill missing wrist positions inside one clip via linear interpolation between neighbours.

    Sets the `<hand>_hand_world_wrist` field on the sample and flips
    `<hand>_hand_world_wrist_estimated` to True. Only inner gaps are filled
    (leading/trailing missing frames stay None — no extrapolation).
    Also overwrites `<hand>_hand_world` so downstream consumers see the
    interpolated anchor when no raw wrist was available.
    """
    if not clip_samples:
        return
    for hand in ("left", "right"):
        wrist_key = f"{hand}_hand_world_wrist"
        world_key = f"{hand}_hand_world"
        estimated_key = f"{hand}_hand_world_wrist_estimated"
        indices_with_value = [
            idx for idx, sample in enumerate(clip_samples)
            if sample.get(wrist_key) is not None
        ]
        if len(indices_with_value) < 2:
            continue
        first_known = indices_with_value[0]
        last_known = indices_with_value[-1]
        prev_idx = first_known
        for next_idx in indices_with_value[1:]:
            if next_idx > prev_idx + 1:
                prev_pos = np.asarray(clip_samples[prev_idx][wrist_key], dtype=np.float32)
                next_pos = np.asarray(clip_samples[next_idx][wrist_key], dtype=np.float32)
                span = float(next_idx - prev_idx)
                for fill_idx in range(prev_idx + 1, next_idx):
                    alpha = float(fill_idx - prev_idx) / span
                    estimated = ((1.0 - alpha) * prev_pos + alpha * next_pos).astype(np.float32)
                    sample = clip_samples[fill_idx]
                    sample[wrist_key] = estimated
                    sample[estimated_key] = True
                    if sample.get(world_key) is None:
                        sample[world_key] = estimated
            prev_idx = next_idx
        _ = (first_known, last_known)  # leading/trailing gaps intentionally left as None


def infer_smoothed_contact_flags(samples: Sequence[dict], radius: int = SMOOTH_FALLBACK_RADIUS) -> None:
    """Populate a smoothed interaction-active signal for clip segmentation."""
    for idx, sample in enumerate(samples):
        window = samples[max(0, idx - radius) : min(len(samples), idx + radius + 1)]
        # Smooth the RAW per-frame contact, NOT the latched ``interaction_active``.
        # ``interaction_active`` is a state machine that latches ON at the first
        # contact frame and is held across gaps by the look-ahead, so a single
        # spurious RCNN contact (e.g. the hand reaching over an object one frame
        # before the real grasp) would pin the active clip's onset there and eat
        # the pre-grasp rest frames -> no inactive clip -> no prior. Smoothing the
        # raw ``hand_contact`` lets the radius-N majority vote actually reject such
        # isolated single-frame contacts, so the active clip starts at the first
        # sustained grasp run and the rest frames before it form the inactive clip.
        positives = sum(1 for item in window if item.get("hand_contact", False))
        sample["smoothed_contact_active"] = positives > (len(window) // 2)


def apply_active_flag_preroll(
    samples: Sequence[dict],
    flag_key: str,
    preroll_frames: int = ACTIVE_CLIP_PREROLL_FRAMES,
) -> int:
    """Back-extend active flags so the active clip starts slightly before the first strict trigger."""
    if preroll_frames <= 0 or not samples:
        return 0
    original = [bool(sample.get(flag_key, False)) for sample in samples]
    expanded = list(original)
    for idx, is_active in enumerate(original):
        if not is_active:
            continue
        for back_idx in range(idx - 1, max(-1, idx - preroll_frames - 1), -1):
            if _sample_requires_new_clip(samples[back_idx], samples[back_idx + 1]):
                break
            expanded[back_idx] = True
    changed = 0
    for sample, new_value in zip(samples, expanded):
        if bool(sample.get(flag_key, False)) != bool(new_value):
            changed += 1
        sample[flag_key] = bool(new_value)
    return changed


def apply_active_flag_postroll(
    samples: Sequence[dict],
    flag_key: str,
    postroll_frames: int = ACTIVE_CLIP_POSTROLL_FRAMES,
) -> int:
    """Forward-extend active flags so the active clip continues through the tail
    of an interaction where the RCNN hand/object detector drops out but the hand
    stream is still present.

    Mirrors ``apply_active_flag_preroll`` in the forward direction. The strict
    contact signal can collapse near the end of a manipulation when the hand
    grips a handle close to the image edge and the detector stops firing
    (``missing_dets``), even though the part is still being moved (e.g. the 183
    drawer sliding fully open). We carry the active flag forward up to
    ``postroll_frames`` while the hand stream is still healthy/present, stopping
    at clip boundaries (frame discontinuities / pose-block changes).
    """
    if postroll_frames <= 0 or not samples:
        return 0

    def _hand_present(sample: dict) -> bool:
        if sample.get("left_hand_valid") or sample.get("right_hand_valid"):
            return True
        hand_dets = sample.get("hand_dets")
        return bool(hand_dets is not None and len(hand_dets) > 0)

    original = [bool(sample.get(flag_key, False)) for sample in samples]
    expanded = list(original)
    n = len(samples)
    for idx in range(n):
        if not original[idx]:
            continue
        extended = 0
        for fwd in range(idx + 1, n):
            if extended >= postroll_frames:
                break
            if _sample_requires_new_clip(samples[fwd - 1], samples[fwd]):
                break
            if not _hand_present(samples[fwd]):
                break
            expanded[fwd] = True
            extended += 1
    changed = 0
    for sample, new_value in zip(samples, expanded):
        if bool(sample.get(flag_key, False)) != bool(new_value):
            changed += 1
        sample[flag_key] = bool(new_value)
    return changed


def bridge_active_flag_gaps(
    samples: Sequence[dict],
    flag_key: str,
    max_gap_frames: int = ACTIVE_CLIP_BRIDGE_GAP_FRAMES,
) -> int:
    """Fill SHORT inactive holes inside a single continuous interaction.

    A brief mid-manipulation detection dropout (hand+object occluded for a few
    frames before reappearing in the SAME carry/motion) flips the active flag to
    False for that hole, which otherwise splits one interaction into two clips and
    strands the tracker (it cannot re-acquire across the gap). We re-activate any
    inactive run of length <= ``max_gap_frames`` that is flanked by active frames
    on BOTH sides. Runs at a frame/pose-block boundary (``_sample_requires_new_clip``)
    are NOT bridged, so genuine clip splits and longer (real) inter-interaction
    gaps are preserved.
    """
    if max_gap_frames <= 0 or not samples:
        return 0
    original = [bool(sample.get(flag_key, False)) for sample in samples]
    expanded = list(original)
    n = len(samples)
    idx = 0
    while idx < n:
        if original[idx]:
            idx += 1
            continue
        # [start, end) is a maximal inactive run
        start = idx
        while idx < n and not original[idx]:
            idx += 1
        end = idx
        # Must be flanked by active frames on both sides (not a leading/trailing run)
        if start == 0 or end >= n:
            continue
        gap_len = end - start
        if gap_len > max_gap_frames:
            continue
        # Do not bridge across a frame/pose-block discontinuity anywhere in the run
        # (including the two flanking transitions).
        discontinuous = False
        for j in range(start, end + 1):
            if _sample_requires_new_clip(samples[j - 1], samples[j]):
                discontinuous = True
                break
        if discontinuous:
            continue
        for j in range(start, end):
            expanded[j] = True
    changed = 0
    for sample, new_value in zip(samples, expanded):
        if bool(sample.get(flag_key, False)) != bool(new_value):
            changed += 1
        sample[flag_key] = bool(new_value)
    return changed


def summarize_scene_graph_match(match_debug: dict) -> str:
    """Format one geometry-aware node-match decision for logs."""
    if not match_debug:
        return "no match debug available"
    if not match_debug.get("matched", False):
        return (
            f"no match for '{match_debug.get('query_label', '')}' "
            f"among {int(match_debug.get('candidate_count', 0))} compatible candidates"
        )
    return (
        f"matched node {match_debug.get('selected_node_id')} '{match_debug.get('selected_label')}' "
        f"(overlap={match_debug.get('selected_overlap_ratio')}, "
        f"centroid={match_debug.get('selected_centroid_distance_m')}m, "
        f"median_nn={match_debug.get('selected_median_nn_distance_m')}m)"
    )


def segment_clips(samples: Sequence[dict], mode: str = "smoothed") -> List[ClipInfo]:
    """Split samples into alternating inactive and active clips using the requested segmentation mode."""
    infer_interaction_flags(samples)
    infer_smoothed_contact_flags(samples)
    def _g(flag):
        return [int(s.get("prefix", -1)) for s in samples if s.get(flag)]
    print(f"[SegDebug] RAW strict interaction_active frames: {_g('interaction_active')}")
    print(f"[SegDebug] RAW smoothed_contact_active frames: {_g('smoothed_contact_active')}")
    print(f"[SegDebug] proximity_contact frames: {_g('proximity_contact')}")
    print(f"[SegDebug] hand_contact frames: {_g('hand_contact')}")
    strict_preroll_changed = apply_active_flag_preroll(samples, "interaction_active")
    smoothed_preroll_changed = apply_active_flag_preroll(samples, "smoothed_contact_active")
    if strict_preroll_changed or smoothed_preroll_changed:
        print(
            f"[ClipSegmentation] applied active preroll={ACTIVE_CLIP_PREROLL_FRAMES} "
            f"(strict +{strict_preroll_changed} frames, smoothed +{smoothed_preroll_changed} frames)"
        )
    strict_postroll_changed = apply_active_flag_postroll(samples, "interaction_active")
    smoothed_postroll_changed = apply_active_flag_postroll(samples, "smoothed_contact_active")
    if strict_postroll_changed or smoothed_postroll_changed:
        print(
            f"[ClipSegmentation] applied active postroll={ACTIVE_CLIP_POSTROLL_FRAMES} "
            f"(strict +{strict_postroll_changed} frames, smoothed +{smoothed_postroll_changed} frames)"
        )
    strict_bridge_changed = bridge_active_flag_gaps(samples, "interaction_active")
    smoothed_bridge_changed = bridge_active_flag_gaps(samples, "smoothed_contact_active")
    if strict_bridge_changed or smoothed_bridge_changed:
        print(
            f"[ClipSegmentation] bridged short contact dropouts (<= {ACTIVE_CLIP_BRIDGE_GAP_FRAMES} frames): "
            f"strict +{strict_bridge_changed} frames, smoothed +{smoothed_bridge_changed} frames"
        )
    print(f"[SegDebug] POST-ROLL strict interaction_active frames: {_g('interaction_active')}")
    print(f"[SegDebug] POST-ROLL smoothed_contact_active frames: {_g('smoothed_contact_active')}")
    if mode == "strict_lostfound":
        clips = _build_clips_from_flag(samples, "interaction_active")
    elif mode == "auto":
        clips = _build_clips_from_flag(samples, "interaction_active")
        n_active = sum(1 for clip in clips if not clip.is_inactive)
        if n_active == 0 and any(sample.get("hand_contact", False) for sample in samples):
            msg = (
                "[ClipSegmentation][Error] strict LostFound interval logic produced no active clip; "
                "smoothed-contact fallback is disabled."
            )
            print(msg)
            raise RuntimeError(msg)
    else:
        clips = _build_clips_from_flag(samples, "smoothed_contact_active")
        n_active = sum(1 for clip in clips if not clip.is_inactive)
        if n_active == 0 and any(sample.get("proximity_contact", False) for sample in samples):
            for sample in samples:
                sample["contact_fallback_active"] = bool(sample.get("proximity_contact", False))
            cf_preroll_changed = apply_active_flag_preroll(samples, "contact_fallback_active")
            cf_postroll_changed = apply_active_flag_postroll(samples, "contact_fallback_active")
            print(
                "[ClipSegmentation] smoothed-contact produced 0 active clips; "
                "falling back to raw proximity contact."
            )
            print(
                f"[ClipSegmentation] proximity-contact fallback expanded by preroll +{cf_preroll_changed} "
                f"and postroll +{cf_postroll_changed} frame(s)"
            )
            print(f"[SegDebug] contact_fallback_active frames: {_g('contact_fallback_active')}")
            clips = _build_clips_from_flag(samples, "contact_fallback_active")
    for clip in clips:
        if not clip.is_inactive:
            clip.interaction_hand = choose_clip_interaction_hand(clip.samples)
        interpolate_missing_wrist_positions(clip.samples)
    print(
        f"Segmented into {len(clips)} clips: "
        f"{sum(1 for c in clips if c.is_inactive)} inactive, "
        f"{sum(1 for c in clips if not c.is_inactive)} active"
    )
    return clips

def find_info_json(data_dir: Path, camera: str = "color") -> Path:
    """Return the first camera info JSON path for the requested stream."""
    jsons = sorted((data_dir / f"camera_{camera}_info").glob("*.json"))
    if not jsons:
        raise FileNotFoundError(f"No camera info JSON found for {camera}")
    return jsons[0]


def save_jpegs(rgb_list: Sequence[np.ndarray], out_dir: Path) -> None:
    """Write a sequence of RGB frames as JPEGs for SAM3 video tracking."""
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    printed_resize = False
    max_side = int(SAM3_MAX_IMAGE_SIDE)
    for idx, rgb in enumerate(rgb_list):
        frame = np.asarray(rgb, dtype=np.uint8)
        if max_side > 0 and frame.size:
            height, width = frame.shape[:2]
            scale = min(1.0, float(max_side) / float(max(height, width)))
            if scale < 1.0:
                new_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
                if not printed_resize:
                    print(f"  SAM3 JPEG resize: {width}x{height} -> {new_size[0]}x{new_size[1]}")
                    printed_resize = True
                frame = cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)
        cv2.imwrite(
            str(out_dir / f"{idx}.jpg"),
            cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),
            [cv2.IMWRITE_JPEG_QUALITY, 92],
        )


def sanitize_rgb_sequence_for_perception(
    rgb_list: Sequence[np.ndarray],
    samples: Sequence[dict],
) -> List[np.ndarray]:
    """Replace color-invalid frames with the nearest valid RGB frame for sequence models."""
    if not rgb_list:
        return []
    valid_indices = [idx for idx, sample in enumerate(samples) if idx < len(rgb_list) and sample.get("rgb_valid", True)]
    if not valid_indices:
        return [np.asarray(rgb, dtype=np.uint8).copy() for rgb in rgb_list]
    sanitized: List[np.ndarray] = []
    for idx, rgb in enumerate(rgb_list):
        if idx < len(samples) and samples[idx].get("rgb_valid", True):
            sanitized.append(np.asarray(rgb, dtype=np.uint8).copy())
            continue
        replacement_idx = min(valid_indices, key=lambda valid_idx: abs(valid_idx - idx))
        sanitized.append(np.asarray(rgb_list[replacement_idx], dtype=np.uint8).copy())
    return sanitized

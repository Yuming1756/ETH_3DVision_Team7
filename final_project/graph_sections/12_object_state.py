# ======================================================================
# graph.py section: 12_object_state.py
# Object-state inference, caption/description building
#
# NOTE: not a standalone module. Loaded by graph.py into a shared
# namespace via exec(); names from earlier sections are in scope here.
# ======================================================================
from __future__ import annotations
def load_qwen_inactive_context_frames(
    previous_inactive_clip: Optional[ClipInfo],
    K_color: Optional[np.ndarray],
    D_color: Optional[np.ndarray],
    max_frames: int = 2,
) -> Tuple[List[np.ndarray], List[int]]:
    """Load a small set of inactive full-frame images as Qwen before-context."""
    inactive_context_frames: List[np.ndarray] = []
    inactive_context_frame_ids: List[int] = []
    if previous_inactive_clip is None or K_color is None or D_color is None or not previous_inactive_clip.samples:
        return inactive_context_frames, inactive_context_frame_ids
    valid_inactive_indices = [
        idx for idx, sample in enumerate(previous_inactive_clip.samples) if sample.get("rgb_valid", True)
    ]
    source_indices = valid_inactive_indices if valid_inactive_indices else list(range(len(previous_inactive_clip.samples)))
    if not source_indices:
        return inactive_context_frames, inactive_context_frame_ids
    candidate_positions = [max(0, len(source_indices) // 2), len(source_indices) - 1]
    inactive_indices = []
    for pos in candidate_positions:
        idx = source_indices[int(np.clip(pos, 0, len(source_indices) - 1))]
        if idx not in inactive_indices:
            inactive_indices.append(idx)
    for idx in inactive_indices[: max(1, int(max_frames))]:
        sample = previous_inactive_clip.samples[idx]
        bgr = cv2.imread(str(sample["rgb_path"]))
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        inactive_context_frames.append(prepare_full_frame_for_qwen(undistort_rgb(rgb, K_color, D_color)))
        inactive_context_frame_ids.append(int(sample.get("frame", idx)))
    return inactive_context_frames, inactive_context_frame_ids


def normalize_semantic_state(raw_state: str) -> str:
    """Normalize free-form Qwen state text into a short stable phrase."""
    text = str(raw_state or "").strip().lower()
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return "unknown"

    exact_map = {
        "opened": "open",
        "open": "open",
        "partially opened": "partially open",
        "partially open": "partially open",
        "slightly open": "partially open",
        "ajar": "partially open",
        "closed": "closed",
        "shut": "closed",
        "turned on": "on",
        "powered on": "on",
        "switched on": "on",
        "on": "on",
        "turned off": "off",
        "powered off": "off",
        "switched off": "off",
        "off": "off",
        "locked": "locked",
        "unlocked": "unlocked",
        "activated": "activated",
        "deactivated": "deactivated",
        "started": "activated",
        "running": "activated",
        "stopped": "deactivated",
        "picked up": "picked up",
        "lifted": "picked up",
        "placed down": "placed down",
        "put down": "placed down",
        "set down": "placed down",
        "removed": "removed",
        "unknown": "unknown",
        "unclear": "unknown",
        "cannot tell": "unknown",
        "cant tell": "unknown",
        "not visible": "unknown",
        "no clear change": "unknown",
        "unchanged": "unknown",
        "no change": "unknown",
    }
    if text in exact_map:
        return exact_map[text]

    contains_map = [
        ("partially open", "partially open"),
        ("turned on", "on"),
        ("powered on", "on"),
        ("switched on", "on"),
        ("turned off", "off"),
        ("powered off", "off"),
        ("switched off", "off"),
        ("picked up", "picked up"),
        ("placed down", "placed down"),
        ("put down", "placed down"),
        ("set down", "placed down"),
    ]
    for needle, normalized in contains_map:
        if needle in text:
            return normalized

    words = text.split()
    if len(words) > 4:
        text = " ".join(words[:4])
    return text or "unknown"


def summarize_object_state_evidence(
    relationship: dict,
    motion_coupling: dict,
    object_motion: Optional[dict],
    previous_object_state: str,
) -> str:
    """Summarize semantic evidence for object-state inference without hinge-specific geometry jargon."""
    pieces: List[str] = []
    prev_state = str(previous_object_state or "").strip().lower()
    if prev_state and prev_state != "unknown":
        pieces.append(f"Previously recorded state before this clip: '{prev_state}'.")
    relation_name = normalize_functional_relation(str((relationship or {}).get("relationship", "unknown")))
    if relation_name != "unknown":
        pieces.append(f"Functional relation inferred for this pair: '{relation_name}'.")
    interaction_type = str((motion_coupling or {}).get("interaction_type", "")).strip().lower()
    if interaction_type == "articulated_object":
        joint_type = str((object_motion or {}).get("articulation_joint_type", "")).strip().lower()
        if joint_type == "revolute":
            pieces.append("The named object is the moving articulated part and visibly swings to a new resting pose during the clip.")
        elif joint_type == "prismatic":
            pieces.append("The named object is the moving articulated part and visibly slides to a new resting pose during the clip.")
        else:
            pieces.append("The named object is the moving articulated part and ends the clip in a new physical configuration.")
    elif interaction_type == "rigid_transport":
        pieces.append("The named object behaves like one rigid carried item. Infer a semantic state only if the video shows a clear persistent outcome such as being picked up or placed down.")
    elif interaction_type == "remote_functional":
        pieces.append("This is a remote/control interaction. Infer the object's resulting state from the visible effect in the video, not from shared bulk motion.")
    pose_history = list((object_motion or {}).get("pose_history", []) or [])
    if pose_history:
        def _frame_of(sample: Any) -> int:
            if isinstance(sample, dict):
                return int(sample.get("frame", -1))
            return int(getattr(sample, "frame", -1))

        first_frame = _frame_of(pose_history[0])
        last_frame = _frame_of(pose_history[-1])
        pieces.append(
            f"Tracked object-motion evidence spans frames {first_frame} to {last_frame} with {len(pose_history)} pose updates."
        )
    return " ".join(piece for piece in pieces if piece).strip()


def infer_object_state_from_video(
    bundle: QwenBundle,
    clip: ClipInfo,
    rgb_list: Sequence[np.ndarray],
    out_dir: Path,
    previous_inactive_clip: Optional[ClipInfo],
    K_color: Optional[np.ndarray],
    D_color: Optional[np.ndarray],
    element_label: str,
    object_label: str,
    relationship: dict,
    motion_coupling: dict,
    element_desc: str = "",
    object_desc: str = "",
    object_motion: Optional[dict] = None,
    previous_object_state: str = "unknown",
) -> dict:
    """Infer the object's resulting semantic state from inactive context plus active video."""
    default_result = {
        "mode": f"state_video_stride{QWEN_ACTIVE_VIDEO_FRAME_STRIDE}",
        "object_state_before": "unknown",
        "object_state_after": "unknown",
        "object_state": "unknown",
        "status_changed": False,
        "confidence": 0.0,
        "reasoning": "",
        "inactive_context_frames": [],
        "active_video_frame_ids": [],
        "active_video_path": "",
        "raw": "",
    }
    if not rgb_list:
        return dict(default_result)

    inactive_context_frames, inactive_context_frame_ids = load_qwen_inactive_context_frames(
        previous_inactive_clip,
        K_color,
        D_color,
    )
    active_rgb = sanitize_rgb_sequence_for_perception(rgb_list, clip.samples)
    sampled_indices = sampled_video_frame_indices(
        len(active_rgb),
        stride=QWEN_ACTIVE_VIDEO_FRAME_STRIDE,
        max_frames=QWEN_ACTIVE_VIDEO_MAX_FRAMES,
    )
    sampled_frames = [active_rgb[idx] for idx in sampled_indices]
    sampled_frame_ids = [int(clip.samples[idx].get("frame", idx)) for idx in sampled_indices]
    if not sampled_frames:
        return dict(default_result)

    prepared_active_frames = [prepare_full_frame_for_qwen(frame) for frame in sampled_frames]
    video_dir = out_dir / "_qwen_active_videos"
    video_path = video_dir / f"clip_{int(clip.clip_id):03d}_state_stride{QWEN_ACTIVE_VIDEO_FRAME_STRIDE}.mp4"
    video_path = save_rgb_clip_as_mp4(prepared_active_frames, video_path, fps=QWEN_ACTIVE_VIDEO_FPS)
    print(
        f"[Qwen-State] clip={int(clip.clip_id)} object='{object_label}' previous_state='{previous_object_state}' "
        f"inactive_context_frames={inactive_context_frame_ids} active_video_frame_ids={sampled_frame_ids} "
        f"saved_video={video_path}"
    )

    content = [{"type": "image", "image": image_to_data_url(frame)} for frame in inactive_context_frames[:2]]
    video_payload = {"type": "video", "video": str(video_path)}
    requested = int(len(prepared_active_frames))
    actual_total = requested
    try:
        cap = cv2.VideoCapture(str(video_path))
        if cap.isOpened():
            probed = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            if probed > 0:
                actual_total = probed
        cap.release()
    except Exception:
        pass
    nframes = min(requested, actual_total)
    if nframes >= 2:
        nframes = (nframes // 2) * 2
    if nframes >= 2:
        video_payload["nframes"] = int(nframes)
    content.append(video_payload)

    desc_prefix = f"Interactive element description: '{element_desc}'. Object description: '{object_desc}'. "
    previous_state_clause = (
        f"Previous stored state for this object before the clip: '{previous_object_state}'. "
        if str(previous_object_state or "").strip().lower() != "unknown"
        else "No reliable previous semantic state is stored for this object. "
    )
    inactive_context_note = (
        "The first image(s) show the same scene immediately before the interaction. "
        if inactive_context_frames
        else ""
    )
    sampled_note = (
        f"The video shows the active interaction sampled every {QWEN_ACTIVE_VIDEO_FRAME_STRIDE} frame(s); sampled active frame ids are {sampled_frame_ids}. "
        if sampled_frame_ids
        else f"The video shows the active interaction sampled every {QWEN_ACTIVE_VIDEO_FRAME_STRIDE} frame(s). "
    )
    motion_summary = summarize_object_state_evidence(
        relationship=relationship,
        motion_coupling=motion_coupling,
        object_motion=object_motion,
        previous_object_state=previous_object_state,
    )
    few_shots = [
        {
            "interactive_element": "drawer handle",
            "object": "drawer",
            "object_state_before": "closed",
            "object_state_after": "open",
            "object_state": "open",
            "status_changed": True,
            "why": "Before the interaction, the drawer is closed inside the cabinet. After the hand pulls the handle, the drawer remains pulled outward and open.",
        },
        {
            "interactive_element": "cabinet handle",
            "object": "cabinet door",
            "object_state_before": "closed",
            "object_state_after": "open",
            "object_state": "open",
            "status_changed": True,
            "why": "Before the interaction, the cabinet door is closed. After the hand pulls the handle, the cabinet door swings away from the frame and remains open.",
        },
        {
            "interactive_element": "cabinet handle",
            "object": "cabinet door",
            "object_state_before": "closed",
            "object_state_after": "partially open",
            "object_state": "partially open",
            "status_changed": True,
            "why": "Before the interaction, the cabinet door is closed. After the hand pulls the handle, the door is ajar but not fully open.",
        },
        {
            "interactive_element": "light switch",
            "object": "ceiling light",
            "object_state_before": "off",
            "object_state_after": "on",
            "object_state": "on",
            "status_changed": True,
            "why": "Before the switch is pressed, the light is not illuminated. After the interaction, the controlled light is visibly on.",
        },
        {
            "interactive_element": "microwave button",
            "object": "microwave",
            "object_state_before": "off",
            "object_state_after": "activated",
            "object_state": "activated",
            "status_changed": True,
            "why": "Before the button press, the microwave is inactive. After the interaction, the appliance starts operating or shows an activated state.",
        },
        {
            "interactive_element": "spoon handle",
            "object": "spoon",
            "object_state_before": "resting",
            "object_state_after": "picked up",
            "object_state": "picked up",
            "status_changed": True,
            "why": "Before the interaction, the spoon is resting on or inside another object. After the hand grasps it, the spoon is lifted and remains in the hand.",
        },
        {
            "interactive_element": "container rim",
            "object": "container",
            "object_state_before": "unknown",
            "object_state_after": "unknown",
            "object_state": "unknown",
            "status_changed": False,
            "why": "The container is moved, but the video does not show a clear persistent semantic state such as open, closed, on, off, picked up, or placed down.",
        },
    ]
    if bundle.config.use_cot and bundle.is_qwen35_vl:
        prompt = (
            f"Interactive element: '{element_label}'. Object: '{object_label}'. "
            + desc_prefix
            + previous_state_clause
            + inactive_context_note
            + sampled_note
            + (motion_summary + " " if motion_summary else "")
            + "Infer the object's semantic state BEFORE and AFTER the interaction. "
            "Use the inactive image(s) and the beginning of the video to infer object_state_before. "
            "Use the end of the active video to infer object_state_after and object_state. "
            "If the previous stored state is reliable and visually consistent, you may use it as a prior for object_state_before, but visual evidence has priority. "
            "Prefer stable final semantic states such as 'open', 'partially open', 'closed', 'on', 'off', 'locked', 'unlocked', 'activated', 'deactivated', 'picked up', 'placed down', 'resting', or 'unknown'. "
            "Do not simply restate the action as an in-progress verb like 'opening', 'closing', or 'moving' if the final result is better described as 'open', 'closed', 'picked up', or 'placed down'. "
            "Only predict a concrete state when the clip shows a meaningful persistent state or configuration. "
            "If a before or after state is not visible enough, use 'unknown' for that field. "
            "status_changed should be true only when the visible before and after states are meaningfully different. "
            "Do not describe geometry, axes, hinges, or numeric motion. "
            f"Few-shot examples: {json.dumps(few_shots)}. "
            "Reason carefully from the temporal evidence, then output ONLY JSON with keys: \"object_state_before\", \"object_state_after\", \"object_state\", \"status_changed\", \"confidence\", and \"reasoning\"."
        )
        max_tokens = bundle.config.cot_max_tokens
        enable_reasoning = True
    else:
        prompt = (
            f"Interactive element: '{element_label}'. Object: '{object_label}'. "
            + desc_prefix
            + previous_state_clause
            + inactive_context_note
            + sampled_note
            + (motion_summary + " " if motion_summary else "")
            + "Infer the object's resulting semantic state AFTER the interaction. "
            "Use the inactive image(s) as before-context and the video as the active interaction evidence. "
            "Prefer short natural lowercase state phrases such as 'open', 'partially open', 'closed', 'on', 'off', 'locked', 'unlocked', 'activated', 'deactivated', 'picked up', 'placed down', or 'unknown'. "
            "Be flexible: choose the most semantically faithful short phrase; it does NOT need to be a single word. "
            "Only predict a concrete state when the clip shows a meaningful persistent state or configuration by the end of the clip. "
            "If the clip only shows generic motion/manipulation without a clear persistent semantic outcome, output 'unknown'. "
            "Do not describe geometry, axes, hinges, or numeric motion. Do not simply restate the action as a verb phrase like 'opening' if the resulting state is better expressed as 'open' or 'partially open'. "
            f"Few-shot examples: {json.dumps(few_shots)}. "
            "Output ONLY JSON with keys: "
            "\"object_state_before\", \"object_state_after\", \"object_state\", "
            "\"status_changed\", \"confidence\", and \"reasoning\"."
        )
        max_tokens = bundle.config.qwen_max_tokens
        enable_reasoning = False

    raw = vlm_generate(
        bundle,
        [{"role": "user", "content": content + [{"type": "text", "text": prompt}]}],
        max_tokens=max_tokens,
        enable_reasoning=enable_reasoning,
    )
    payload = safe_json_loads(raw)
    if not isinstance(payload, dict):
        payload = {}

    previous_state_norm = normalize_semantic_state(previous_object_state)

    before_raw = str(payload.get("object_state_before", "")).strip()
    after_raw = str(payload.get("object_state_after", payload.get("object_state", ""))).strip()
    final_raw = str(payload.get("object_state", after_raw)).strip()

    object_state_before = normalize_semantic_state(before_raw) if before_raw else previous_state_norm
    object_state_after = normalize_semantic_state(after_raw) if after_raw else "unknown"
    object_state = normalize_semantic_state(final_raw) if final_raw else object_state_after

    try:
        confidence = float(payload.get("confidence", 0.0))
    except Exception:
        confidence = 0.0

    reasoning = str(payload.get("reasoning", "")).strip() or str(raw or "").strip()

    status_changed_raw = payload.get("status_changed", None)
    if isinstance(status_changed_raw, bool):
        status_changed = status_changed_raw
    else:
        status_changed = (
            object_state_before != "unknown"
            and object_state_after != "unknown"
            and object_state_before != object_state_after
        )

    if object_state_before == "unknown" or object_state_after == "unknown":
        status_changed = False

    print(
        f"[Qwen-State] clip={int(clip.clip_id)} object='{object_label}' "
        f"before='{object_state_before}' after='{object_state_after}' final='{object_state}' "
        f"changed={status_changed} confidence={confidence:.3f}"
    )
    if reasoning:
        print(f"[Qwen-State] reasoning: {reasoning}")

    return {
        "mode": f"state_video_stride{QWEN_ACTIVE_VIDEO_FRAME_STRIDE}",
        "object_state_before": object_state_before,
        "object_state_after": object_state_after,
        "object_state": object_state,
        "status_changed": bool(status_changed),
        "confidence": float(confidence),
        "reasoning": reasoning,
        "inactive_context_frames": inactive_context_frame_ids,
        "active_video_frame_ids": sampled_frame_ids,
        "active_video_path": str(video_path),
        "raw": str(raw or ""),
    }


def build_entity_description(
    qwen_bundle: QwenBundle,
    rgb_list: Sequence[np.ndarray],
    masks: Sequence[np.ndarray],
    entity_kind: str,
    label_hint: str,
    parent_hint: str = "",
) -> Tuple[str, str, List[str], List[int], dict]:
    """Create refined multi-view descriptions for one node with Qwen."""
    prepared_views: List[np.ndarray] = []
    kept_indices: List[int] = []
    for idx in select_mask_view_indices(masks):
        bbox = mask_to_bbox(masks[idx])
        if bbox is None:
            continue
        if entity_kind == "element":
            prepared_views.append(make_multiscale_crop(rgb_list[idx], bbox, mask=masks[idx], outline=True))
        else:
            prepared_views.append(make_object_caption_view(rgb_list[idx], bbox, mask=masks[idx]))
        kept_indices.append(idx)

    qwen_captions = caption_entity_views(qwen_bundle, prepared_views, entity_kind, label_hint, parent_hint)
    qwen_name, qwen_description = refine_entity_caption(qwen_bundle, qwen_captions, entity_kind, label_hint, parent_hint)
    qwen_result = {
        "model": "qwen",
        "canonical_name": qwen_name,
        "description": qwen_description,
        "captions": qwen_captions,
        "view_indices": kept_indices,
    }
    comparison = qwen_only_comparison(qwen_result)
    return (
        str(qwen_result.get("canonical_name", label_hint)),
        str(qwen_result.get("description", "")),
        list(qwen_result.get("captions", [])),
        kept_indices,
        comparison,
    )



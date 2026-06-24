# ======================================================================
# graph.py section: 01_labels_relations.py
# Label / affordance / functional-relation classification helpers
#
# NOTE: not a standalone module. Loaded by graph.py into a shared
# namespace via exec(); names from earlier sections are in scope here.
# ======================================================================
from __future__ import annotations
def has_label_phrase(label: str, phrases: Sequence[str]) -> bool:
    """Return whether a normalized label contains one of the given phrases."""
    label_norm = f" {str(label or '').strip().lower()} "
    for phrase in phrases:
        phrase_norm = str(phrase or "").strip().lower()
        if phrase_norm and f" {phrase_norm} " in label_norm:
            return True
    return False


def is_affordance_source_label(label: str) -> bool:
    """Return whether a label looks like a manipulable part or control."""
    return has_label_phrase(label, AFFORDANCE_SOURCE_LABELS)


def is_affordance_target_label(label: str) -> bool:
    """Return whether a label looks like an object affected by a manipulable part."""
    return has_label_phrase(label, FUNCTIONAL_TARGET_LABELS)


def is_affordance_style_pair(element_label: str, object_label: str) -> bool:
    """Return whether the element/object pair looks like a functional part-to-object relation."""
    element_label = str(element_label or "").strip().lower()
    object_label = str(object_label or "").strip().lower()
    if not element_label or not object_label or element_label == object_label:
        return False
    if not is_affordance_source_label(element_label):
        return False
    if object_label in element_label:
        return True
    return is_affordance_target_label(object_label)


def label_contains_any(label: str, phrases: Sequence[str]) -> bool:
    """Return whether a normalized label contains any normalized phrase."""
    return has_label_phrase(label, phrases)


def is_rigid_handle_object_pair(element_label: str, object_label: str) -> bool:
    """Return whether a part and object should usually share one rigid motion."""
    element_label = str(element_label or "").strip().lower()
    object_label = str(object_label or "").strip().lower()
    if not element_label or not object_label:
        return False
    if "handle" not in element_label:
        return False
    if label_contains_any(element_label, ARTICULATED_PART_LABELS):
        return False
    return label_contains_any(object_label, RIGID_HANDLE_OBJECT_LABELS)


def is_handle_on_same_articulated_part(element_label: str, object_label: str) -> bool:
    """Return whether a handle/control and object label name the same moving articulated part."""
    element_label = str(element_label or "").strip().lower()
    object_label = str(object_label or "").strip().lower()
    if not element_label or not object_label:
        return False
    if "handle" not in element_label and not label_contains_any(element_label, STATIC_CONTROL_LABELS):
        return False
    return any(has_label_phrase(element_label, (part,)) and has_label_phrase(object_label, (part,)) for part in ARTICULATED_PART_LABELS)


def is_articulated_parent_body_pair(element_label: str, object_label: str) -> bool:
    """Return whether the element likely moves relative to a larger static parent body."""
    element_label = str(element_label or "").strip().lower()
    object_label = str(object_label or "").strip().lower()
    if not element_label or not object_label:
        return False
    if is_handle_on_same_articulated_part(element_label, object_label):
        return False
    parent_is_body = label_contains_any(object_label, ARTICULATED_PARENT_BODY_LABELS)
    parent_is_movable_part = label_contains_any(object_label, ARTICULATED_PART_LABELS)
    element_mentions_part = label_contains_any(element_label, ARTICULATED_PART_LABELS)
    control_on_body = label_contains_any(element_label, STATIC_CONTROL_LABELS) and parent_is_body
    attachment_on_child_of_body = (
        label_contains_any(element_label, MOVABLE_ATTACHMENT_LABELS)
        and element_mentions_part
        and parent_is_body
        and not parent_is_movable_part
    )
    return bool(attachment_on_child_of_body or control_on_body)


def apply_motion_coupling_guardrails(
    payload: dict,
    elem_label: str,
    obj_label: str,
    raw: str,
) -> dict:
    """Normalize Qwen motion-coupling output without overriding it by label heuristics."""
    interaction_type = str(payload.get("interaction_type", "")).strip().lower()
    if interaction_type not in {"remote_functional", "rigid_transport", "articulated_object"}:
        interaction_type = ""
    shares = bool(payload.get("shares_rigid_motion", False))
    try:
        confidence = float(payload.get("confidence", 0.0))
    except Exception:
        confidence = 0.0
    reasoning = str(payload.get("reasoning", "")).strip() or raw
    moving_part_label = payload.get("moving_part_label")
    if is_rigid_handle_object_pair(elem_label, obj_label) and interaction_type == "articulated_object":
        interaction_type = "rigid_transport"
        shares = True
        if moving_part_label is None:
            moving_part_label = obj_label
        reasoning = (
            "Sanity-corrected impossible articulated classification for a free rigid carried object. "
            + reasoning
        )
    if interaction_type == "remote_functional":
        shares = False
    static_parent_body = bool(interaction_type == "articulated_object")
    if moving_part_label is not None:
        moving_part_label = str(moving_part_label).strip()
        if not moving_part_label:
            moving_part_label = None
    # Qwen joint-type guess (revolute vs prismatic), used to pick the SAM3 object
    # seed frame before the ArtiPoint fit knows the real joint type. Only meaningful
    # for articulated objects; blank otherwise.
    joint_type = str(payload.get("joint_type", "")).strip().lower()
    if joint_type not in {"revolute", "prismatic"} or interaction_type != "articulated_object":
        joint_type = ""
    return {
        "interaction_type": interaction_type,
        "shares_rigid_motion": shares,
        "confidence": confidence,
        "reasoning": reasoning,
        "static_parent_body": static_parent_body,
        "moving_part_label": moving_part_label,
        "joint_type": joint_type,
    }


def normalize_functional_relation(raw_relation: str) -> str:
    """Map free-form Qwen relation text into the fixed functional relation vocabulary."""
    text = str(raw_relation or "").strip().lower()
    normalized = re.sub(r"[^a-z0-9 ]+", " ", text)
    normalized = re.sub(r"\s+", " ", normalized).strip().replace(" ", "_")
    if normalized in FUNCTIONAL_RELATIONS:
        return normalized
    if not text:
        return "unknown"
    if "unlock" in text:
        return "can_unlock"
    if "lock" in text:
        return "can_lock"
    if "turn on" in text or "switch on" in text or "power on" in text:
        return "can_turn_on"
    if "turn off" in text or "switch off" in text or "power off" in text:
        return "can_turn_off"
    if "open" in text:
        return "can_open"
    if "close" in text or "shut" in text:
        return "can_close"
    if "activate" in text or "press" in text or "trigger" in text or "start" in text:
        return "can_activate"
    if "move" in text or "pull" in text or "push" in text or "slide" in text:
        return "can_move"
    if "part" in text or "component" in text or "required" in text:
        return "part_required_for_use"
    return "unknown"



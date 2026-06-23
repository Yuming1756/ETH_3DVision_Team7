import json
import re
# Modify safe_json_extract by the following two functions: Frank
def strip_code_fences(raw: str) -> str:
    """Remove markdown code fences from a model response."""
    text = (raw or "").strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def safe_json_value(raw: str):
    """
    Parse the first JSON value from a model response.

    Returns either:
    - dict
    - list
    - {}
    """
    text = strip_code_fences(raw)
    decoder = json.JSONDecoder()

    starts = [idx for idx, ch in enumerate(text) if ch in "{["]

    for start in starts:
        try:
            obj, _end = decoder.raw_decode(text[start:])
            return obj
        except json.JSONDecodeError:
            continue

    print("[Warn] Could not parse JSON from model output:")
    print(text[:1000])
    return {}


def safe_json_loads(raw: str) -> dict:
    """
    Robustly parse model JSON.

    If the model returns a list of detections, keep it under key 'detections'
    instead of silently taking the first item.
    """
    obj = safe_json_value(raw)

    if isinstance(obj, dict):
        return obj

    if isinstance(obj, list):
        return {"detections": obj}

    return {}


def safe_json_extract(raw: str) -> str:
    """Backward-compatible wrapper returning a JSON string."""
    return json.dumps(safe_json_loads(raw))
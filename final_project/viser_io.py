"""Interactive Viser viewer for dynamic scene-graph outputs.

Loads ``scene_graph/graph.json`` plus saved point clouds, motion segments, and
per-clip wrist observations from an output directory. The viewer shows the full
RGB scene by default; per-object/element masks fully overwrite the underlying
RGB so each tracked node visually pops against the scene.
"""

from __future__ import annotations

import argparse
import colorsys
import getpass
import hashlib
import json
import socket
import sys
import threading
import time
from pathlib import Path
from typing import Iterable, Optional

import cv2
import numpy as np
import open3d as o3d
import viser

PROJECT_ROOT = Path(__file__).resolve().parent
TEMP_ROOT = PROJECT_ROOT / "temp"
if TEMP_ROOT.exists() and str(TEMP_ROOT) not in sys.path:
    sys.path.insert(0, str(TEMP_ROOT))

from dynamic_scene_graph import DynamicSceneGraph  # noqa: E402


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def sanitize_name(text: str) -> str:
    """Make a safe Viser scene path component."""
    out = "".join(ch if ch.isalnum() else "_" for ch in str(text).strip())
    return out.strip("_") or "item"


def label_matches(label: str, focus: Optional[str]) -> bool:
    """Return whether a label should be included for the optional focus filter."""
    if not focus:
        return True
    return str(focus).strip().lower() in str(label).strip().lower()


def mask_color(node_id: str, node_type: str, label: str) -> np.ndarray:
    """Vivid, stable mask color per node. Warm hues for elements, cool for objects."""
    digest = hashlib.sha1(f"{node_id}:{label}".encode("utf-8")).digest()
    hue_jitter = digest[0] / 255.0
    if str(node_type).lower() == "element":
        warm_bands = [(0.95, 1.05), (0.0, 0.16), (0.16, 0.22)]
        band = warm_bands[digest[1] % len(warm_bands)]
        hue = (band[0] + (band[1] - band[0]) * hue_jitter) % 1.0
        sat, val = 0.85, 0.98
    else:
        cool_bands = [(0.45, 0.55), (0.55, 0.70), (0.70, 0.82)]
        band = cool_bands[digest[1] % len(cool_bands)]
        hue = band[0] + (band[1] - band[0]) * hue_jitter
        sat, val = 0.78, 0.96
    r, g, b = colorsys.hsv_to_rgb(hue, sat, val)
    return np.clip(np.array([r, g, b]) * 255.0, 0, 255).astype(np.uint8)


def wrist_color(hand: str) -> np.ndarray:
    """Color wrists by handedness."""
    if str(hand).lower().startswith("left"):
        return np.array([255, 190, 55], dtype=np.uint8)
    return np.array([50, 220, 255], dtype=np.uint8)


def normalize_float_colors(
    colors: np.ndarray,
    n_points: int,
    fallback_rgb: np.ndarray | tuple[float, float, float] = (0.6, 0.6, 0.6),
) -> np.ndarray:
    """Return per-point RGB colors in [0, 1], preserving saved PLY/node colors."""
    if n_points <= 0:
        return np.zeros((0, 3), dtype=np.float32)
    cols = np.asarray(colors, dtype=np.float32)
    if cols.shape != (n_points, 3):
        fallback = np.asarray(fallback_rgb, dtype=np.float32).reshape(1, 3)
        if np.max(fallback) > 1.5:
            fallback = fallback / 255.0
        return np.repeat(np.clip(fallback, 0.0, 1.0), n_points, axis=0).astype(np.float32)
    if cols.size and float(np.nanmax(cols)) > 1.5:
        cols = cols / 255.0
    cols = np.nan_to_num(cols, nan=0.6, posinf=1.0, neginf=0.0)
    return np.clip(cols, 0.0, 1.0).astype(np.float32)


def load_ply(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load one point cloud, returning empty arrays if absent."""
    if not path.exists():
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)
    pcd = o3d.io.read_point_cloud(str(path))
    pts = np.asarray(pcd.points, dtype=np.float32)
    if pcd.has_colors():
        cols = np.asarray(pcd.colors, dtype=np.float32)
    else:
        cols = np.full((len(pts), 3), 0.6, dtype=np.float32)
    return pts, cols


def downsample(points: np.ndarray, colors: np.ndarray, voxel_size: float) -> tuple[np.ndarray, np.ndarray]:
    """Voxel-downsample points/colors for browser performance."""
    pts = np.asarray(points, dtype=np.float32)
    cols = np.asarray(colors, dtype=np.float32)
    if len(pts) == 0 or voxel_size <= 0:
        return pts, cols
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    if len(cols) == len(pts):
        pcd.colors = o3d.utility.Vector3dVector(cols)
    pcd = pcd.voxel_down_sample(voxel_size)
    out_pts = np.asarray(pcd.points, dtype=np.float32)
    out_cols = np.asarray(pcd.colors, dtype=np.float32) if pcd.has_colors() else np.full((len(out_pts), 3), 0.6, dtype=np.float32)
    return out_pts, out_cols


def _voxelize(points: np.ndarray, voxel_size: float) -> np.ndarray:
    """Quantize 3D points to integer voxel coordinates."""
    if voxel_size <= 0 or len(points) == 0:
        return np.zeros((0, 3), dtype=np.int64)
    return np.floor(np.asarray(points, dtype=np.float64) / float(voxel_size)).astype(np.int64)


def scene_mask_against_nodes(
    scene_pts: np.ndarray,
    node_points_list: Iterable[np.ndarray],
    voxel_size: float,
    dilation: int = 1,
) -> np.ndarray:
    """Boolean keep-mask: drop scene points whose voxel is covered by any node point."""
    keep = np.ones(len(scene_pts), dtype=bool)
    if len(scene_pts) == 0 or voxel_size <= 0:
        return keep
    occupied: set[tuple[int, int, int]] = set()
    for chunk in node_points_list:
        if chunk is None or len(chunk) == 0:
            continue
        voxels = _voxelize(np.asarray(chunk, dtype=np.float32), voxel_size)
        if dilation <= 0:
            occupied.update(map(tuple, voxels))
            continue
        offsets = range(-dilation, dilation + 1)
        for dx in offsets:
            for dy in offsets:
                for dz in offsets:
                    shifted = voxels + np.array([dx, dy, dz], dtype=np.int64)
                    occupied.update(map(tuple, shifted))
    if not occupied:
        return keep
    scene_voxels = _voxelize(scene_pts, voxel_size)
    for idx, voxel in enumerate(map(tuple, scene_voxels)):
        if voxel in occupied:
            keep[idx] = False
    return keep


def load_graph(output_dir: Path) -> DynamicSceneGraph:
    """Load the saved graph if available."""
    graph_dir = output_dir / "scene_graph"
    if not (graph_dir / "graph.json").exists():
        print(f"[Viser] Missing {graph_dir / 'graph.json'}; showing scene/wrist data only.")
        return DynamicSceneGraph()
    return DynamicSceneGraph.load(str(graph_dir))


def load_clip_events(output_dir: Path) -> list[dict]:
    """Load saved per-clip event summaries exported by graph.py."""
    events: list[dict] = []
    for path in sorted(output_dir.glob("clip_*/event.json")):
        try:
            payload = json.loads(path.read_text())
        except Exception:
            continue
        payload["_path"] = str(path)
        events.append(payload)
    events.sort(key=lambda item: (int(item.get("clip_id", -1)), int(item.get("frame_start", -1))))
    return events


def _fmt_conf(value: object) -> str:
    try:
        return f"{float(value):.2f}"
    except Exception:
        return "-"


def format_clip_event_md(event: dict) -> str:
    """Render one clip event summary as a markdown block for the Viser sidebar."""
    clip_id = int(event.get("clip_id", -1))
    frame_start = int(event.get("frame_start", -1))
    frame_end = int(event.get("frame_end", -1))
    element = str(event.get("element") or "-")
    tracked_object = str(event.get("object") or event.get("tracked_object") or "-")
    relation = str(event.get("functional_relationship") or "-")
    relation_conf = _fmt_conf(event.get("functional_confidence"))
    before = str(event.get("object_state_before") or "unknown")
    after = str(event.get("object_state_after") or event.get("object_state") or "unknown")
    changed = event.get("status_changed")
    changed_text = "true" if changed is True else "false" if changed is False else "-"
    changed_conf = _fmt_conf(event.get("status_change_confidence"))
    shares_motion = event.get("shares_rigid_motion")
    shared_text = "true" if shares_motion is True else "false" if shares_motion is False else "-"
    graph_diff = event.get("graph_diff") or {}
    new_nodes = len(graph_diff.get("new_nodes") or [])
    new_edges = len(graph_diff.get("new_edges") or [])
    removed_nodes = len(graph_diff.get("removed_nodes") or [])
    state_changes = len(graph_diff.get("state_changes") or [])
    lines = [
        f"**Clip** · `{clip_id}` (`{frame_start}`–`{frame_end}`)",
        f"**Element** · {element}",
        f"**Object** · {tracked_object}",
        f"**Function** · `{relation}` (conf {relation_conf})",
        f"**State** · `{before}` → `{after}`",
        f"**Changed** · `{changed_text}` (conf {changed_conf})",
        f"**Shared motion tracking** · `{shared_text}`",
        f"**Graph diff** · new nodes `{new_nodes}`, new edges `{new_edges}`, removed `{removed_nodes}`, state changes `{state_changes}`",
    ]
    detection_reasoning = str(event.get("detection_reasoning") or "").strip()
    functional_reasoning = str(event.get("functional_reasoning") or "").strip()
    status_reasoning = str(event.get("status_change_reasoning") or "").strip()
    motion_reasoning = str(event.get("motion_coupling_reasoning") or "").strip()
    if detection_reasoning:
        lines.extend(["", "**Detection reasoning**", detection_reasoning])
    if functional_reasoning:
        lines.extend(["", "**Functional reasoning**", functional_reasoning])
    if status_reasoning:
        lines.extend(["", "**Status reasoning**", status_reasoning])
    if motion_reasoning:
        lines.extend(["", "**Motion coupling**", motion_reasoning])
    return "\n".join(lines)


def build_clip_bounds_lookup(clip_events: Iterable[dict]) -> dict[int, tuple[int, int]]:
    """Map clip id -> (frame_start, frame_end) for time-aware semantic replay."""
    lookup: dict[int, tuple[int, int]] = {}
    for event in clip_events:
        try:
            clip_id = int(event.get("clip_id", -1))
            frame_start = int(event.get("frame_start", -1))
            frame_end = int(event.get("frame_end", -1))
        except Exception:
            continue
        if clip_id < 0 or frame_start < 0 or frame_end < frame_start:
            continue
        lookup[clip_id] = (frame_start, frame_end)
    return lookup


def build_openfungraph_ids(graph: DynamicSceneGraph, node_ids: Iterable[str]) -> dict[str, str]:
    """Assign stable OpenFunGraph-style display ids (`O*` objects, `P*` parts)."""
    object_ids = sorted(
        [nid for nid in node_ids if str(graph.nodes[nid].node_type).lower() != "element"],
        key=lambda nid: (str(graph.nodes[nid].label).lower(), nid),
    )
    part_ids = sorted(
        [nid for nid in node_ids if str(graph.nodes[nid].node_type).lower() == "element"],
        key=lambda nid: (str(graph.nodes[nid].label).lower(), nid),
    )
    display: dict[str, str] = {}
    for idx, node_id in enumerate(object_ids):
        display[node_id] = f"O{idx}"
    for idx, node_id in enumerate(part_ids):
        display[node_id] = f"P{idx}"
    return display


def node_state_at_frame(node, frame: int, clip_bounds_by_id: dict[int, tuple[int, int]]) -> str:
    """Replay one node's semantic state up to a timeline frame."""
    history = list(getattr(node, "state_history", []) or [])
    if not history:
        return str(getattr(node, "state", "unknown") or "unknown")

    def _history_key(item: dict) -> tuple[int, int]:
        try:
            clip_id = int(item.get("clip_id", -1))
        except Exception:
            clip_id = -1
        bounds = clip_bounds_by_id.get(clip_id)
        clip_end = bounds[1] if bounds is not None else 10**9
        return clip_end, clip_id

    history.sort(key=_history_key)
    first_from = str(history[0].get("from", "")).strip()
    current_state = first_from or "unknown"
    for transition in history:
        try:
            clip_id = int(transition.get("clip_id", -1))
        except Exception:
            clip_id = -1
        bounds = clip_bounds_by_id.get(clip_id)
        clip_end = bounds[1] if bounds is not None else 10**9
        if clip_end <= int(frame):
            next_state = str(transition.get("to", "")).strip()
            if next_state:
                current_state = next_state
        else:
            break
    return current_state or "unknown"


def format_openfungraph_node_label(
    node,
    display_id: str,
    *,
    frame: Optional[int] = None,
    state: Optional[str] = None,
) -> str:
    """Compact label text shared by the scene, GUI, and exports."""
    label = f"{display_id}: {node.label}"
    if state and state != "unknown":
        label += f" [{state}]"
    if frame is not None:
        label += f" @ {int(frame)}"
    return label


def format_live_semantics_md(
    graph: DynamicSceneGraph,
    node_ids: Iterable[str],
    display_ids: dict[str, str],
    frame: int,
    clip_events: list[dict],
    clip_bounds_by_id: dict[int, tuple[int, int]],
) -> str:
    """Summarize what the graph semantically believes at one master frame."""
    active_events = [
        event for event in clip_events
        if int(event.get("frame_start", -1)) <= int(frame) <= int(event.get("frame_end", -1))
    ]
    completed_events = [
        event for event in clip_events
        if int(event.get("frame_end", -1)) <= int(frame)
    ]
    lines = [f"**Master frame** · `{int(frame)}`"]
    if active_events:
        event = sorted(active_events, key=lambda item: int(item.get("frame_start", -1)))[-1]
        clip_id = int(event.get("clip_id", -1))
        element = str(event.get("element") or "-")
        tracked_object = str(event.get("object") or event.get("tracked_object") or "-")
        relation = str(event.get("functional_relationship") or "-")
        before = str(event.get("object_state_before") or "unknown")
        after = str(event.get("object_state_after") or event.get("object_state") or "unknown")
        lines.extend(
            [
                "",
                f"**Active interaction** · `clip_{clip_id:03d}`",
                f"`{element}` → `{tracked_object}`",
                f"Function: `{relation}`",
                f"State transition: `{before}` → `{after}`",
            ]
        )
    elif completed_events:
        event = sorted(completed_events, key=lambda item: int(item.get("frame_end", -1)))[-1]
        clip_id = int(event.get("clip_id", -1))
        relation = str(event.get("functional_relationship") or "-")
        lines.extend(
            [
                "",
                f"**Last completed update** · `clip_{clip_id:03d}`",
                f"Function: `{relation}`",
            ]
        )
    else:
        lines.extend(["", "_No completed semantic updates yet._"])

    state_rows: list[tuple[str, str, str]] = []
    for node_id in node_ids:
        node = graph.nodes.get(node_id)
        if node is None:
            continue
        state = node_state_at_frame(node, int(frame), clip_bounds_by_id)
        if state == "unknown" and not getattr(node, "state_history", None):
            continue
        state_rows.append((display_ids.get(node_id, node_id[:6]), str(node.label), state))
    if state_rows:
        lines.extend(["", "**Known node states**"])
        for display_id, label, state in sorted(state_rows, key=lambda item: item[0]):
            lines.append(f"- `{display_id}` {label} · `{state}`")
    else:
        lines.extend(["", "_All node states are still unknown at this frame._"])
    return "\n".join(lines)


def bbox_segments(points: np.ndarray) -> np.ndarray:
    """Axis-aligned 3D bounding-box wireframe as line segments."""
    pts = np.asarray(points, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[0] == 0 or pts.shape[1] != 3:
        return np.zeros((0, 2, 3), dtype=np.float32)
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    corners = np.asarray(
        [
            [mins[0], mins[1], mins[2]],
            [maxs[0], mins[1], mins[2]],
            [mins[0], maxs[1], mins[2]],
            [maxs[0], maxs[1], mins[2]],
            [mins[0], mins[1], maxs[2]],
            [maxs[0], mins[1], maxs[2]],
            [mins[0], maxs[1], maxs[2]],
            [maxs[0], maxs[1], maxs[2]],
        ],
        dtype=np.float32,
    )
    edges = [
        (0, 1), (1, 3), (3, 2), (2, 0),
        (4, 5), (5, 7), (7, 6), (6, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]
    return np.stack([np.stack([corners[a], corners[b]], axis=0) for a, b in edges], axis=0)


def collect_display_nodes(graph: DynamicSceneGraph, focus_label: Optional[str]) -> list[str]:
    """Return graph nodes with static geometry, motion, or a usable static centroid.

    Semantic-only nodes (e.g. a handle that shares its parent's rigid motion and so
    carries no geometry of its own) have neither points nor motion segments, but they
    still have a graph centroid — keep them so they appear as graph nodes / edge endpoints.
    """
    node_ids: list[str] = []
    for node_id, node in graph.nodes.items():
        has_static = len(getattr(node, "points", [])) > 0
        has_motion = bool(getattr(node, "motion_segments", []))
        cen = getattr(node, "centroid", None)
        has_centroid = (
            cen is not None
            and np.asarray(cen, dtype=np.float32).reshape(-1).size >= 3
            and bool(np.all(np.isfinite(np.asarray(cen, dtype=np.float32).reshape(-1)[:3])))
        )
        if not (has_static or has_motion or has_centroid):
            continue
        if label_matches(node.label, focus_label):
            node_ids.append(node_id)
    return sorted(node_ids, key=lambda item: (graph.nodes[item].label.lower(), item))


def collect_relationship_nodes(graph: DynamicSceneGraph, node_ids: Iterable[str]) -> set[str]:
    """Return the subset of visible node ids that participate in at least one edge."""
    allowed = set(str(nid) for nid in node_ids)
    related: set[str] = set()
    for edge in graph.edges.values():
        src_id = str(getattr(edge, "src_id", ""))
        dst_id = str(getattr(edge, "dst_id", ""))
        if src_id in allowed and dst_id in allowed:
            related.add(src_id)
            related.add(dst_id)
    return related


def expand_with_functional_neighbors(graph: DynamicSceneGraph, node_ids: Iterable[str]) -> list[str]:
    """Expand a focused node set with its 1-hop functional neighbors.

    This keeps focus-label behavior useful in graph view: asking for
    ``cabinet door`` should also show the connected ``cabinet handle`` node
    and edge, not just the single matched node.
    """
    selected = set(str(nid) for nid in node_ids)
    if not selected:
        return []
    expanded = set(selected)
    changed = True
    while changed:
        changed = False
        for edge in graph.edges.values():
            src_id = str(getattr(edge, "src_id", ""))
            dst_id = str(getattr(edge, "dst_id", ""))
            if not src_id or not dst_id:
                continue
            if src_id in selected and dst_id not in expanded and dst_id in graph.nodes:
                expanded.add(dst_id)
                changed = True
            elif dst_id in selected and src_id not in expanded and src_id in graph.nodes:
                expanded.add(src_id)
                changed = True
    return sorted(expanded, key=lambda item: (graph.nodes[item].label.lower(), item))


def graph_frames(graph: DynamicSceneGraph, node_ids: Iterable[str]) -> list[int]:
    """Collect exact motion frames and node active-frame metadata."""
    frames: list[int] = []
    for node_id in node_ids:
        node = graph.nodes[node_id]
        frames.extend(int(frame) for frame in graph.get_exact_motion_frames(node_id))
        if int(getattr(node, "frame_start", -1)) >= 0:
            frames.append(int(node.frame_start))
        if int(getattr(node, "frame_end", -1)) >= 0:
            frames.append(int(node.frame_end))
    return frames


def load_wrist_bundles(output_dir: Path) -> list[dict]:
    """Load raw per-frame wrist observations exported by graph.py."""
    bundles: list[dict] = []
    for metadata_path in sorted(output_dir.glob("clip_*/viser_wrist/metadata.json")):
        try:
            payload = json.loads(metadata_path.read_text())
        except Exception:
            continue
        clip_id = int(payload.get("clip_id", -1))
        clip_type = str(payload.get("type", "clip"))
        for hand, samples in payload.get("hands", {}).items():
            frame_positions: dict[int, np.ndarray] = {}
            estimated_frames: set[int] = set()
            valid_frames: set[int] = set()
            for sample in samples or []:
                try:
                    frame = int(sample.get("frame"))
                    pos = np.asarray(sample.get("position", []), dtype=np.float32)
                except Exception:
                    continue
                if pos.shape != (3,):
                    continue
                frame_positions[frame] = pos
                if bool(sample.get("estimated", False)):
                    estimated_frames.add(frame)
                if bool(sample.get("valid", True)):
                    valid_frames.add(frame)
            if frame_positions:
                bundles.append(
                    {
                        "clip_id": clip_id,
                        "clip_type": clip_type,
                        "hand": str(hand),
                        "frame_start": int(payload.get("frame_start", min(frame_positions))),
                        "frame_end": int(payload.get("frame_end", max(frame_positions))),
                        "frame_positions": frame_positions,
                        "estimated_frames": estimated_frames,
                        "valid_frames": valid_frames,
                        "color": wrist_color(hand),
                    }
                )
    return bundles


def wrist_frames(bundles: Iterable[dict]) -> list[int]:
    """Collect wrist frame indices."""
    frames: list[int] = []
    for bundle in bundles:
        frames.extend(int(frame) for frame in bundle.get("frame_positions", {}).keys())
        frames.append(int(bundle.get("frame_start", -1)))
        frames.append(int(bundle.get("frame_end", -1)))
    return [frame for frame in frames if frame >= 0]


def load_rgbd_frame_assets(output_dir: Path) -> dict[int, dict]:
    """Load frame-indexed RGB/depth source paths exported by graph.py."""
    assets: dict[int, dict] = {}
    for metadata_path in sorted(output_dir.glob("clip_*/viser_wrist/metadata.json")):
        try:
            payload = json.loads(metadata_path.read_text())
        except Exception:
            continue
        clip_id = int(payload.get("clip_id", -1))
        clip_type = str(payload.get("type", "clip"))
        for frame_payload in payload.get("frames", []) or []:
            try:
                frame = int(frame_payload.get("frame", -1))
            except Exception:
                continue
            if frame < 0:
                continue
            rgb_path = Path(str(frame_payload.get("rgb_path", "")))
            depth_path = Path(str(frame_payload.get("depth_path", "")))
            assets[frame] = {
                "clip_id": clip_id,
                "clip_type": clip_type,
                "clip_index": int(frame_payload.get("clip_index", frame - int(payload.get("frame_start", frame)))),
                "rgb_path": rgb_path,
                "depth_path": depth_path,
            }
    return assets


def index_topic_images(topic_dir: Path) -> dict[int, Path]:
    """Index extracted MicroAGI image files by numeric frame key."""
    index: dict[int, Path] = {}
    if not topic_dir.exists():
        return index
    for path in sorted(topic_dir.glob("*.png")):
        key_text = path.stem.split("_")[0]
        try:
            frame = int(key_text)
        except Exception:
            continue
        index.setdefault(frame, path)
    return index


def load_rgbd_assets_from_data_dir(output_dir: Path, data_dir: Optional[Path]) -> dict[int, dict]:
    """Fallback RGB-D lookup from clips.json and an extracted MicroAGI directory."""
    if data_dir is None or not data_dir.exists():
        return {}
    clips_path = output_dir / "clips.json"
    if not clips_path.exists():
        return {}
    try:
        clips = json.loads(clips_path.read_text())
    except Exception:
        return {}
    rgb_index = index_topic_images(data_dir / "camera_color_image")
    depth_index = index_topic_images(data_dir / "camera_depth_image")
    assets: dict[int, dict] = {}
    for clip in clips:
        try:
            clip_id = int(clip.get("clip_id", -1))
            frame_start = int(clip.get("frame_start", -1))
            frame_end = int(clip.get("frame_end", -1))
        except Exception:
            continue
        if frame_start < 0 or frame_end < frame_start:
            continue
        clip_type = str(clip.get("type", "clip"))
        for frame in range(frame_start, frame_end + 1):
            rgb_path = rgb_index.get(frame)
            depth_path = depth_index.get(frame)
            if rgb_path is None and depth_path is None:
                continue
            assets[frame] = {
                "clip_id": clip_id,
                "clip_type": clip_type,
                "clip_index": int(frame - frame_start),
                "rgb_path": rgb_path or Path(""),
                "depth_path": depth_path or Path(""),
            }
    return assets


def colorize_raw_depth_for_view(depth: np.ndarray) -> np.ndarray:
    """Colorize raw uint depth for a compact Viser GUI preview."""
    depth_arr = np.asarray(depth, dtype=np.float32)
    valid = depth_arr > 0
    if not np.any(valid):
        return np.zeros((*depth_arr.shape[:2], 3), dtype=np.uint8)
    vmax = float(np.percentile(depth_arr[valid], 95))
    if vmax <= 1e-4:
        vmax = float(depth_arr[valid].max())
    normalized = np.clip(depth_arr / max(vmax, 1e-4), 0.0, 1.0)
    colored = cv2.applyColorMap((normalized * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    colored[~valid] = 0
    return cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)


def colorize_mask_for_view(mask: np.ndarray) -> np.ndarray:
    """Render a binary mask as a high-contrast GUI preview."""
    mask_bool = np.asarray(mask) > 0
    out = np.zeros((*mask_bool.shape[:2], 3), dtype=np.uint8)
    out[mask_bool] = np.array([255, 80, 80], dtype=np.uint8)
    contours, _ = cv2.findContours(mask_bool.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, (255, 230, 120), 2)
    ys, xs = np.where(mask_bool)
    if len(xs) and len(ys):
        x1, x2 = int(xs.min()), int(xs.max())
        y1, y2 = int(ys.min()), int(ys.max())
        cv2.rectangle(out, (x1, y1), (x2, y2), (255, 255, 255), 2)
    return out


def find_panel_mask_path(output_dir: Path, asset: Optional[dict]) -> Optional[Path]:
    """Find the saved element mask corresponding to one timeline frame, if present."""
    if asset is None:
        return None
    try:
        clip_id = int(asset.get("clip_id", -1))
        clip_index = int(asset.get("clip_index", -1))
    except Exception:
        return None
    if clip_id < 0 or clip_index < 0:
        return None
    clip_dir = output_dir / f"clip_{clip_id:03d}"
    candidates = [
        ("sam3_active_masks/element", "element"),
        ("sam3_active_masks/element_raw", "element"),
        ("inactive_sam3_masks/clean_element", "element"),
        ("inactive_sam3_masks/raw_element", "element"),
        ("sam3_active_masks/object_body", "object"),
        ("inactive_sam3_masks/object_body", "object"),
        ("sam3_active_masks/object_raw", "object"),
        ("inactive_sam3_masks/raw_object", "object"),
    ]
    for relative_dir, prefix in candidates:
        path = clip_dir / relative_dir / f"{prefix}_{clip_index:04d}.png"
        if path.exists():
            return path
    return None


def make_rgbd_panel(output_dir: Path, asset: Optional[dict], max_width: int = 720) -> np.ndarray:
    """Build a side-by-side SAM mask/raw-depth preview image for the GUI."""
    if asset is None:
        return np.zeros((120, 360, 3), dtype=np.uint8)
    mask_preview = None
    depth_color = None
    mask_path = find_panel_mask_path(output_dir, asset)
    depth_path = Path(asset.get("depth_path", ""))
    if mask_path is not None and mask_path.exists():
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is not None:
            mask_preview = colorize_mask_for_view(mask)
    if depth_path.exists():
        depth = cv2.imread(str(depth_path), cv2.IMREAD_ANYDEPTH)
        if depth is not None:
            depth_color = colorize_raw_depth_for_view(depth)
    if mask_preview is None and depth_color is None:
        return np.zeros((120, 360, 3), dtype=np.uint8)
    if mask_preview is None:
        mask_preview = np.zeros_like(depth_color)
    if depth_color is None:
        depth_color = np.zeros_like(mask_preview)
    target_h = min(240, max(80, mask_preview.shape[0] // 4))
    mask_w = max(1, int(round(mask_preview.shape[1] * target_h / mask_preview.shape[0])))
    depth_w = max(1, int(round(depth_color.shape[1] * target_h / depth_color.shape[0])))
    mask_small = cv2.resize(mask_preview, (mask_w, target_h), interpolation=cv2.INTER_NEAREST)
    depth_small = cv2.resize(depth_color, (depth_w, target_h), interpolation=cv2.INTER_AREA)
    panel = np.concatenate([mask_small, depth_small], axis=1)
    if panel.shape[1] > max_width:
        out_h = max(1, int(round(panel.shape[0] * max_width / panel.shape[1])))
        panel = cv2.resize(panel, (max_width, out_h), interpolation=cv2.INTER_AREA)
    return panel.astype(np.uint8)


def clip_frames(output_dir: Path) -> list[int]:
    """Read clip ranges for timeline bounds."""
    path = output_dir / "clips.json"
    if not path.exists():
        return []
    try:
        clips = json.loads(path.read_text())
    except Exception:
        return []
    frames: list[int] = []
    for clip in clips:
        if "frame_start" in clip:
            frames.append(int(clip["frame_start"]))
        if "frame_end" in clip:
            frames.append(int(clip["frame_end"]))
    return frames


def node_geometry(graph: DynamicSceneGraph, node_id: str, frame: int) -> tuple[np.ndarray, np.ndarray]:
    """Render moving geometry when possible, else use static node geometry."""
    node = graph.nodes[node_id]
    if getattr(node, "motion_segments", []):
        rendered = graph.get_node_geometry_at_frame(node_id, int(frame), exact=False)
        if rendered is not None and len(rendered[0]):
            return np.asarray(rendered[0], dtype=np.float32), np.asarray(rendered[1], dtype=np.float32)
    pts = np.asarray(getattr(node, "points", np.zeros((0, 3))), dtype=np.float32)
    cols = np.asarray(getattr(node, "colors", np.zeros((0, 3))), dtype=np.float32)
    return pts, cols


def trajectory_points(graph: DynamicSceneGraph, node_id: str) -> np.ndarray:
    """Centroid trajectory from exact motion samples."""
    points: list[np.ndarray] = []
    for frame in graph.get_exact_motion_frames(node_id):
        rendered = graph.get_node_geometry_at_frame(node_id, int(frame), exact=True)
        if rendered is not None and len(rendered[0]):
            points.append(np.asarray(rendered[0], dtype=np.float32).mean(axis=0))
    if not points:
        return np.zeros((0, 3), dtype=np.float32)
    return np.stack(points, axis=0).astype(np.float32)


def format_node_history_md(node) -> str:
    """Render a node's state_history as a compact markdown block."""
    state = str(getattr(node, "state", "unknown"))
    lines = [f"**Current state:** `{state}`"]
    history = list(getattr(node, "state_history", []) or [])
    if not history:
        lines.append("\n_No transitions recorded._")
        return "\n".join(lines)
    lines.append("\n**Transitions:**")
    for i, transition in enumerate(history, 1):
        clip_id = transition.get("clip_id", -1)
        from_state = str(transition.get("from", "")) or "—"
        to_state = str(transition.get("to", "")) or "—"
        lines.append(
            f"{i}. clip `{clip_id}` &nbsp;·&nbsp; `{from_state}` → `{to_state}`"
        )
        reasoning = str(transition.get("reasoning", "")).strip()
        if reasoning:
            short = reasoning if len(reasoning) <= 180 else reasoning[:177] + "..."
            lines.append(f"   _{short}_")
    return "\n".join(lines)


def format_global_history_md(graph: DynamicSceneGraph, node_ids: Iterable[str]) -> str:
    """Render every node's transitions chronologically (by clip_id then node label)."""
    rows: list[tuple[int, str, str, str, str]] = []
    for nid in node_ids:
        node = graph.nodes.get(nid)
        if node is None:
            continue
        label = str(getattr(node, "label", nid))
        for transition in getattr(node, "state_history", []) or []:
            try:
                clip_id = int(transition.get("clip_id", -1))
            except Exception:
                clip_id = -1
            rows.append(
                (
                    clip_id,
                    label,
                    str(transition.get("from", "")) or "—",
                    str(transition.get("to", "")) or "—",
                    str(transition.get("reasoning", "")).strip(),
                )
            )
    if not rows:
        return "_No state transitions across the graph._"
    rows.sort(key=lambda r: (r[0], r[1].lower()))
    lines = ["**State events** (sorted by clip)"]
    for clip_id, label, from_state, to_state, reasoning in rows:
        lines.append(
            f"- clip `{clip_id}` &nbsp;·&nbsp; **{label}** &nbsp;·&nbsp; "
            f"`{from_state}` → `{to_state}`"
        )
        if reasoning:
            short = reasoning if len(reasoning) <= 160 else reasoning[:157] + "..."
            lines.append(f"  &nbsp;&nbsp;&nbsp;_{short}_")
    return "\n".join(lines)


def wrist_trail(bundle: dict, frame: int, trail_frames: int) -> tuple[Optional[int], Optional[np.ndarray], np.ndarray]:
    """Latest wrist sample and recent trail at or before frame."""
    positions = bundle.get("frame_positions", {})
    frames = sorted(int(item) for item in positions.keys() if int(item) <= int(frame))
    if not frames:
        return None, None, np.zeros((0, 3), dtype=np.float32)
    if trail_frames > 0:
        frames = frames[-trail_frames:]
    pts = np.stack([np.asarray(positions[item], dtype=np.float32) for item in frames], axis=0)
    latest_frame = frames[-1]
    return latest_frame, pts[-1], pts

def gt_bbox_segments(
    center: np.ndarray,
    dims: np.ndarray,
) -> np.ndarray:
    mins = center - dims / 2.0
    maxs = center + dims / 2.0

    corners = np.asarray(
        [
            [mins[0], mins[1], mins[2]],
            [maxs[0], mins[1], mins[2]],
            [mins[0], maxs[1], mins[2]],
            [maxs[0], maxs[1], mins[2]],
            [mins[0], mins[1], maxs[2]],
            [maxs[0], mins[1], maxs[2]],
            [mins[0], maxs[1], maxs[2]],
            [maxs[0], maxs[1], maxs[2]],
        ],
        dtype=np.float32,
    )

    edges = [
        (0, 1), (1, 3), (3, 2), (2, 0),
        (4, 5), (5, 7), (7, 6), (6, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]

    return np.stack(
        [np.stack([corners[a], corners[b]], axis=0) for a, b in edges],
        axis=0,
    )


def load_gt_bbox(
    gt_dir: Path,
    frame_id: int,
    object_name: str,
):
    gt_file = gt_dir / f"{frame_id:06d}.json"
    print(gt_file)
    if not gt_file.exists():
        return None

    try:
        with open(gt_file, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return None

    gt_ann = data["data"][next(iter(data["data"]))]

    center = np.asarray(
        gt_ann["bbox_center_default"],
        dtype=np.float32,
    )

    dims = np.asarray(
        [
            gt_ann["bbox_width_default"],
            gt_ann["bbox_height_default"],
            gt_ann["bbox_length_default"],
        ],
        dtype=np.float32,
    )

    return center, dims
    
# ---------------------------------------------------------------------------
# Main viewer
# ---------------------------------------------------------------------------



def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("output_dynamic_graph"))
    parser.add_argument("--focus-label", default=None, help="Optional initial node filter; omit to show all nodes.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--point-size", type=float, default=0.018)
    parser.add_argument(
        "--static-voxel", type=float, default=0.0,
        help="Voxel size for static scene downsampling; 0 = dense, full resolution.",
    )
    parser.add_argument(
        "--motion-voxel", type=float, default=0.0,
        help="Voxel size for per-node point cloud downsampling; 0 = dense, full resolution.",
    )
    parser.add_argument(
        "--mask-voxel", type=float, default=0.03,
        help="Voxel size used to subtract scene RGB where node masks live.",
    )
    parser.add_argument("--fps", type=float, default=6.0)
    parser.add_argument(
        "--graph-only", action="store_true",
        help="Start in graph-only mode: each node is a sphere at its centroid + functional "
             "edges between them, animated over the master frame. No dense geometry or static scene.",
    )
    parser.add_argument(
        "--functional-only",
        action="store_true",
        help="Start in functional-relationship mode: only show nodes that participate in graph edges, "
             "with an OpenFunGraph-style nodes-and-edges view.",
    )
    parser.add_argument(
        "--style",
        choices=["fungraph", "classic"],
        default="fungraph",
        help="Visualization styling preset. `fungraph` uses OpenFunGraph-like ids, labels, and relation styling.",
    )
    parser.add_argument(
        "--groundtruth-dir",
        type=Path,
        default=None,
        help="Directory containing per-frame GT JSON annotations.",
    )

    parser.add_argument(
        "--groundtruth-object",
        type=str,
        default=None,
        help="Object label to visualize from the GT annotations.",
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    style_fungraph = str(args.style).lower() == "fungraph"
    graph = load_graph(output_dir)
    node_ids = collect_display_nodes(graph, args.focus_label)
    if args.focus_label and not node_ids:
        print(f"[Viser] No graph nodes matched focus '{args.focus_label}'. Showing all graph nodes instead.")
        node_ids = collect_display_nodes(graph, None)
    elif args.focus_label and node_ids:
        node_ids = expand_with_functional_neighbors(graph, node_ids)
    if not node_ids:
        print("[Viser] No graph nodes with geometry/motion found; wrist/static scene may still be shown.")

    def _sort_key(nid: str) -> tuple[int, str, str]:
        node = graph.nodes[nid]
        type_order = 0 if str(node.node_type).lower() == "element" else 1
        return (type_order, str(node.label).lower(), nid)

    node_ids = sorted(node_ids, key=_sort_key)
    display_ids = build_openfungraph_ids(graph, node_ids)
    relationship_node_ids = collect_relationship_nodes(graph, node_ids)
    clip_events = load_clip_events(output_dir)
    clip_bounds_by_id = build_clip_bounds_lookup(clip_events)

    wrist_bundles = load_wrist_bundles(output_dir)
    rgbd_assets = load_rgbd_frame_assets(output_dir)
    frames = (
        graph_frames(graph, node_ids)
        + wrist_frames(wrist_bundles)
        + clip_frames(output_dir)
        + list(rgbd_assets.keys())
    )
    frame_min = min(frames) if frames else 0
    frame_max = max(frames) if frames else 0
    rgbd_asset_frames = sorted(int(frame) for frame in rgbd_assets.keys())
    rgbd_cache: dict[int, np.ndarray] = {}

    def nearest_rgbd_frame(frame: int) -> Optional[int]:
        if not rgbd_asset_frames:
            return None
        frame = int(frame)
        if frame in rgbd_assets:
            return frame
        idx = int(np.searchsorted(rgbd_asset_frames, frame))
        candidates = []
        if idx > 0:
            candidates.append(rgbd_asset_frames[idx - 1])
        if idx < len(rgbd_asset_frames):
            candidates.append(rgbd_asset_frames[idx])
        if not candidates:
            return None
        return min(candidates, key=lambda item: abs(int(item) - frame))

    scene_pts_raw, scene_cols_raw = load_ply(output_dir / "scene_full.ply")
    if len(scene_pts_raw):
        scene_cols_raw = normalize_float_colors(scene_cols_raw, len(scene_pts_raw), (0.39, 0.39, 0.39))
        scene_pts_raw, scene_cols_raw = downsample(scene_pts_raw, scene_cols_raw, args.static_voxel)

    # Collect the union of node footprints across time so that mask colors
    # visibly replace the underlying RGB instead of stacking on top of it.
    node_footprint_chunks: list[np.ndarray] = []
    for node_id in node_ids:
        node = graph.nodes[node_id]
        static_pts = np.asarray(getattr(node, "points", np.zeros((0, 3))), dtype=np.float32)
        if len(static_pts):
            node_footprint_chunks.append(static_pts)
        for sample_frame in graph.get_exact_motion_frames(node_id):
            rendered = graph.get_node_geometry_at_frame(node_id, int(sample_frame), exact=True)
            if rendered is not None and len(rendered[0]):
                node_footprint_chunks.append(np.asarray(rendered[0], dtype=np.float32))

    if len(scene_pts_raw) and node_footprint_chunks:
        scene_keep = scene_mask_against_nodes(scene_pts_raw, node_footprint_chunks, args.mask_voxel, dilation=1)
        scene_pts_carved = scene_pts_raw[scene_keep]
        scene_cols_carved = scene_cols_raw[scene_keep]
        carved = 1.0 - (len(scene_pts_carved) / max(1, len(scene_pts_raw)))
        print(f"[Viser] Carved {carved*100:.1f}% of scene RGB to make room for node masks.")
    else:
        scene_pts_carved, scene_cols_carved = scene_pts_raw, scene_cols_raw
    if style_fungraph and len(scene_cols_carved):
        scene_cols_carved = np.clip(scene_cols_carved * 0.50 + 0.10, 0.0, 1.0)

    # ------------------------------------------------------------------
    # Server + scene environment
    # ------------------------------------------------------------------
    server = viser.ViserServer(host=args.host, port=args.port, verbose=True)
    server.scene.set_up_direction("+z")
    camera_points = scene_pts_carved if len(scene_pts_carved) else scene_pts_raw
    if len(camera_points):
        cam_min = camera_points.min(axis=0)
        cam_max = camera_points.max(axis=0)
        cam_center = ((cam_min + cam_max) * 0.5).astype(np.float64)
        cam_extent = np.maximum(cam_max - cam_min, 1e-3)
        cam_radius = float(max(np.linalg.norm(cam_extent) * 0.5, 0.75))
        server.initial_camera.look_at = tuple(float(v) for v in cam_center)
        server.initial_camera.position = (
            float(cam_center[0] + cam_radius * 0.8),
            float(cam_center[1] - cam_radius * 2.2),
            float(cam_center[2] + cam_radius * 0.9),
        )
    server.scene.add_grid(
        "/world/grid",
        width=8.0,
        height=8.0,
        width_segments=16,
        height_segments=16,
        plane="xy",
    )
    server.scene.add_light_ambient("/world/light/ambient", intensity=1.4)
    server.scene.add_light_directional(
        "/world/light/key",
        color=(255, 255, 255),
        intensity=1.1,
        position=(3.0, 2.0, 5.0),
    )
    server.scene.add_light_directional(
        "/world/light/fill",
        color=(220, 230, 255),
        intensity=0.5,
        position=(-2.5, -1.5, 2.5),
    )

    static_scene = None
    if len(scene_pts_carved):
        static_scene = server.scene.add_point_cloud(
            "/Background" if style_fungraph else "/world/static_scene",
            scene_pts_carved,
            np.clip(255.0 * scene_cols_carved, 0, 255).astype(np.uint8),
            point_size=args.point_size * 0.65,
            point_shape="rounded",
            point_shading="gradient",
            visible=True,
        )

    manual_points = np.array([
        [0.6269235735189533, 0.6429969482863402, 1.789532607676211],
        [0.17260481351601087, 0.5528427421469349, 0.9956597953687604],
        [-0.3, 1.2, 0.8],
    ], dtype=np.float32)

    manual_colors = np.array([
        [255, 0, 0],     # red
        [0, 255, 0],     # green
        [0, 120, 255],   # blue
    ], dtype=np.uint8)

    server.scene.add_point_cloud(
        "/world/manual_points",
        manual_points,
        manual_colors,
        point_size=0.04,       # bump this up if they're hard to spot
        point_shape="rounded",
    )

    # ------------------------------------------------------------------
    # GUI: single sidebar — Display + Bulk + per-object folders + Wrists + Mask/Depth
    # ------------------------------------------------------------------
    n_elements = sum(1 for nid in node_ids if str(graph.nodes[nid].node_type).lower() == "element")
    n_objects = len(node_ids) - n_elements
    header_title = "## Dynamic FunGraph Viewer" if style_fungraph else "## Scene Graph Viewer"
    header_lines = [header_title]
    if style_fungraph:
        header_lines.append(
            f"**Graph ids** &nbsp;·&nbsp; `{n_objects}` objects as `O*` &nbsp;·&nbsp; `{n_elements}` parts as `P*`"
        )
        header_lines.append(
            f"**Timeline** &nbsp;·&nbsp; frames `{frame_min}` – `{frame_max}` &nbsp;·&nbsp; "
            f"`{len(clip_events)}` interaction clips"
        )
        header_lines.append(
            "_OpenFunGraph-style graph view with live geometric motion and frame-aware semantic state replay._"
        )
    else:
        header_lines.extend(
            [
                f"**Nodes** &nbsp;·&nbsp; {len(node_ids)} total &nbsp;·&nbsp; {n_elements} elements &nbsp;·&nbsp; {n_objects} objects",
                f"**Frames** &nbsp;·&nbsp; `{frame_min}` – `{frame_max}`",
            ]
        )
    server.gui.add_markdown("\n".join(header_lines))

    with server.gui.add_folder("Frame semantics", expand_by_default=True):
        frame_semantics_md = server.gui.add_markdown(
            format_live_semantics_md(graph, node_ids, display_ids, frame_min, clip_events, clip_bounds_by_id)
        )
    if clip_events:
        with server.gui.add_folder("Interaction clips", expand_by_default=False):
            for event in clip_events:
                clip_id = int(event.get("clip_id", -1))
                frame_start = int(event.get("frame_start", -1))
                frame_end = int(event.get("frame_end", -1))
                element = str(event.get("element") or "-")
                tracked_object = str(event.get("object") or event.get("tracked_object") or "-")
                title = f"clip_{clip_id:03d} {frame_start}-{frame_end} · {element} → {tracked_object}"
                with server.gui.add_folder(title, expand_by_default=False):
                    server.gui.add_markdown(format_clip_event_md(event))

    # Global chronological summary of every state transition across the graph.
    # Computed once at load — state_history is immutable post-replay.
    with server.gui.add_folder("Graph state history", expand_by_default=False):
        server.gui.add_markdown(format_global_history_md(graph, node_ids))

    # Per-node motion frames + centroid trajectory (computed once).
    node_motion_frames: dict[str, list[int]] = {}
    node_trajectory_positions: dict[str, np.ndarray] = {}
    for node_id in node_ids:
        frames_for_node = sorted(int(f) for f in graph.get_exact_motion_frames(node_id))
        node_motion_frames[node_id] = frames_for_node
        positions: list[np.ndarray] = []
        for f in frames_for_node:
            rendered = graph.get_node_geometry_at_frame(node_id, int(f), exact=True)
            if rendered is not None and len(rendered[0]):
                positions.append(np.asarray(rendered[0], dtype=np.float32).mean(axis=0))
        node_trajectory_positions[node_id] = (
            np.stack(positions, axis=0).astype(np.float32) if positions else np.zeros((0, 3), dtype=np.float32)
        )

    # ---- Display folder (global controls) ----
    with server.gui.add_folder("Display", expand_by_default=True):
        point_size_slider = server.gui.add_slider(
            "Point size", min=0.002, max=0.05, step=0.001, initial_value=float(args.point_size),
        )
        show_static = server.gui.add_checkbox("Show static scene", initial_value=True)
        global_autoplay = server.gui.add_checkbox("Autoplay", initial_value=False)
        global_speed = server.gui.add_slider(
            "Speed (fps)", min=0.5, max=30.0, step=0.5, initial_value=6.0,
        )
        show_traj = server.gui.add_checkbox("Trajectories", initial_value=True)
        show_centroid = server.gui.add_checkbox("Centroids", initial_value=False)
        show_instances = server.gui.add_checkbox(
            "Instance markers", initial_value=False,
            hint="Draw one colored centroid sphere per node, like OpenFunGraph's InsO / InsP markers.",
        )
        show_bboxes = server.gui.add_checkbox(
            "Bounding boxes", initial_value=False,
            hint="Axis-aligned 3D debug boxes around the currently rendered node geometry.",
        )
        semantic_colors = server.gui.add_checkbox(
            "Semantic node colors", initial_value=True,
            hint="Color each node cloud by graph identity (mask color) instead of saved RGB colors.",
        )
        show_labels = server.gui.add_checkbox("Labels", initial_value=True)
        show_relationships = server.gui.add_checkbox(
            "Show relationships", initial_value=False,
            hint="Draw functional graph edges between the currently rendered node centroids, with one label per relationship.",
        )
        graph_only = server.gui.add_checkbox(
            "Graph only (nodes+edges)", initial_value=bool(args.graph_only),
            hint="Hide all dense geometry and the static scene; show each node as a sphere at its "
                 "centroid plus functional edges, animated over the master frame slider.",
        )
        functional_only = server.gui.add_checkbox(
            "Functional relationship mode",
            initial_value=bool(args.functional_only),
            hint="Limit visibility to nodes that actually participate in graph edges and switch to "
                 "a cleaner OpenFunGraph-style nodes-and-edges view.",
        )
        show_all_axes = server.gui.add_checkbox("Show all axes", initial_value=False)
        show_articulation_axes = server.gui.add_checkbox(
            "Articulation axes (red)", initial_value=True,
            hint="Draw the fitted 1-DoF articulation axis as a red arrow (ArtiPoint Fig. 2 style) for every node where the fit succeeded.",
        )
        show_axis_candidates = server.gui.add_checkbox(
            "Hinge candidates (debug)", initial_value=False,
            hint="Overlay every alternative hinge candidate from different estimators (native=red, Plücker=orange, closed-form=green, wrist arc=blue, per-track cluster=magenta). Each candidate uses its own colour so you can eyeball which method recovered the physical hinge correctly.",
        )
        show_keypoint_trajectories = server.gui.add_checkbox(
            "Keypoint trajectories", initial_value=True,
            hint="Draw per-corner 3-D tracks (CoTracker3 + Eq.(1) smoothed) as a bundle of curves — ArtiPoint Fig. 2 panel 4 style.",
        )
        keypoint_trajectory_max = server.gui.add_slider(
            "Max keypoint curves", min=5, max=200, step=5, initial_value=40,
            hint="Hard cap to keep the viewport readable when many corners were tracked.",
        )
        show_wrist = server.gui.add_checkbox("Show wrists", initial_value=True)
        master_frame_slider = server.gui.add_slider(
            "Master frame", min=frame_min, max=frame_max, step=1, initial_value=frame_min,
            hint=(
                "Single timeline that drives wrists, RGBD preview, and every object's geometry. "
                "Per-object 'Pin frame' overrides this for a single node so you can hold it "
                "while everything else animates."
            ),
        )

    # ---- Bulk controls (collapsed) ----
    with server.gui.add_folder("Bulk controls", expand_by_default=False):
        filter_text = server.gui.add_text(
            "Filter (label)", initial_value="",
            hint="Substring filter — only matching nodes stay visible. Empty = restore all.",
        )
        btn_show_all = server.gui.add_button("Show all")
        btn_hide_all = server.gui.add_button("Hide all")
        btn_elements_only = server.gui.add_button("Elements only")
        btn_objects_only = server.gui.add_button("Objects only")
        btn_functional_graph = server.gui.add_button("Functional graph")
        btn_full_scene = server.gui.add_button("Full scene")
        btn_reset_frames = server.gui.add_button("Reset frames to start")

    # ---- Per-object folders ----
    # An element that is articulated/rigidly bound to its parent object is one
    # physical body, so it shares a SINGLE folder/panel with that object. Only
    # genuinely remote controls (controls/activates/...) keep a separate panel.
    SEPARATE_RELATIONS = {
        "controls", "activates", "operates", "switches", "toggles",
        "turns_on", "turns_off", "remote_controls",
    }
    node_id_set = set(node_ids)
    element_to_object: dict[str, str] = {}
    for edge in graph.edges.values():
        rel = str(getattr(edge, "relationship", "") or "").strip().lower()
        if rel in SEPARATE_RELATIONS:
            continue
        src_id = str(getattr(edge, "src_id", ""))
        dst_id = str(getattr(edge, "dst_id", ""))
        if src_id not in graph.nodes or dst_id not in graph.nodes:
            continue
        s_elem = str(graph.nodes[src_id].node_type).lower() == "element"
        d_elem = str(graph.nodes[dst_id].node_type).lower() == "element"
        elem_id = src_id if s_elem else (dst_id if d_elem else None)
        obj_id = dst_id if not d_elem else (src_id if not s_elem else None)
        if (elem_id and obj_id and elem_id != obj_id
                and elem_id in node_id_set and obj_id in node_id_set):
            element_to_object[elem_id] = obj_id

    # Group node ids into panels: an object panel absorbs its bound element(s).
    consumed: set[str] = set()
    panels: list[list[str]] = []
    for node_id in node_ids:
        if node_id in consumed or str(graph.nodes[node_id].node_type).lower() == "element":
            continue
        members = [node_id] + [
            eid for eid, oid in element_to_object.items()
            if oid == node_id and eid not in consumed
        ]
        consumed.update(members)
        panels.append(members)
    for node_id in node_ids:  # leftover elements not bound to a shown object
        if node_id not in consumed:
            panels.append([node_id])
            consumed.add(node_id)

    # ---- Focus & playback: camera-nav + solo one object + object-motion play ----
    # One focus target per functional unit (object + its bound element(s)), so
    # picking "fridge" frames the door+handle together. Object-motion playback
    # steps the master clock through the focused object's OWN motion samples
    # (idle frames skipped), so you watch IT move instead of a global counter.
    focus_choices: list[str] = ["(all)"]
    focus_label_to_members: dict[str, list[str]] = {}
    for _members in panels:
        _head = graph.nodes[_members[0]]
        _disp = f"{display_ids.get(_members[0], _members[0][:6])}: {_head.label or _members[0][:6]}"
        focus_choices.append(_disp)
        focus_label_to_members[_disp] = list(_members)
    with server.gui.add_folder("Focus & playback", expand_by_default=True):
        focus_dropdown = server.gui.add_dropdown(
            "Focus unit", options=focus_choices, initial_value="(all)",
            hint="Pick one functional unit (object + its parts). Solo isolates it; "
                 "'Fly camera to focus' frames it.",
        )
        solo_focus = server.gui.add_checkbox(
            "Solo (hide others)", initial_value=False,
            hint="Show only the focused unit; hide all other nodes. Off restores the scene.",
        )
        btn_fly_to = server.gui.add_button("Fly camera to focus")
        play_choices = ["Master clock"] + [c for c in focus_choices if c != "(all)"]
        play_dropdown = server.gui.add_dropdown(
            "Play through", options=play_choices, initial_value="Master clock",
            hint="Master clock advances one frame at a time. Pick an object to step the "
                 "timeline through that object's own motion samples (idle gaps skipped) "
                 "so you watch it move. Toggle Autoplay in Display to run.",
        )

    node_ui: dict[str, dict] = {}

    def _add_node_controls(node_id: str, prefix: str = "") -> None:
        node = graph.nodes[node_id]
        frames_for_node = node_motion_frames[node_id]
        visible_checkbox = server.gui.add_checkbox(f"{prefix}Visible", initial_value=True)
        if frames_for_node:
            pin_checkbox = server.gui.add_checkbox(
                f"{prefix}Pin frame", initial_value=False,
                hint=(
                    "When off (default), this node follows the Master frame slider. "
                    "When on, the State slider below pins this node at a fixed frame."
                ),
            )
            state_slider = server.gui.add_slider(
                f"{prefix}State (frame)",
                min=min(frames_for_node),
                max=max(frames_for_node),
                step=1,
                initial_value=min(frames_for_node),
            )
        else:
            pin_checkbox = None
            state_slider = None
            server.gui.add_markdown(
                f"_{display_ids.get(node_id, node_id[:6])}: {node.label or node_id[:6]} — static node, no motion samples._"
            )
        history_md = server.gui.add_markdown(format_node_history_md(node))
        axis_checkbox = server.gui.add_checkbox(f"{prefix}Show axis", initial_value=False)
        node_ui[node_id] = dict(
            visible=visible_checkbox,
            pin=pin_checkbox,
            state_slider=state_slider,
            history_md=history_md,
            axis=axis_checkbox,
            frames=frames_for_node,
        )

    server.gui.add_markdown(f"### Functional units ({len(panels)})")
    for members in panels:
        obj_node = graph.nodes[members[0]]
        head_id = display_ids.get(members[0], members[0][:6])
        title = f"{head_id}: {obj_node.label}" if obj_node.label else f"{head_id}"
        n_transitions = len(getattr(obj_node, "state_history", []) or [])
        if n_transitions:
            title = f"{title}  ·  {n_transitions}△"  # triangle = transitions count
        if len(members) > 1:
            title = f"{title}  (+{len(members) - 1} part)"
        merged = len(members) > 1
        with server.gui.add_folder(title, expand_by_default=False):
            for member_id in members:
                member = graph.nodes[member_id]
                if merged:
                    server.gui.add_markdown(
                        f"**{display_ids.get(member_id, member_id[:6])}: {member.label}** · _{member.node_type}_"
                    )
                prefix = f"{display_ids.get(member_id, member_id[:6])}: {member.label}: " if merged else ""
                _add_node_controls(member_id, prefix=prefix)

    # ---- Wrists folder (collapsed) ----
    with server.gui.add_folder("Hand trajectories", expand_by_default=False):
        show_wrist_trail = server.gui.add_checkbox("Show trail", initial_value=True)
        wrist_trail_frames = server.gui.add_slider(
            "Trail length (frames)", min=1, max=180, step=1, initial_value=30,
        )

    # ---- Mask / Depth panel folder (collapsed) ----
    with server.gui.add_folder("2D grounding", expand_by_default=False):
        show_rgbd = server.gui.add_checkbox(
            "Enable preview", initial_value=False,
            hint="Show the 2D SAM mask + raw depth preview alongside the 3D view.",
        )
        rgbd_frame_text = server.gui.add_text(
            "Showing frame", initial_value=str(frame_min), disabled=True,
        )
        rgbd_image = server.gui.add_image(
            make_rgbd_panel(output_dir, None), label="Mask | raw depth",
        )

    # ---- Scene state bookkeeping ----
    state: dict[str, dict[str, object]] = {
        name: {}
        for name in [
            "nodes", "instances", "centroids", "labels", "bboxes", "traj", "waypoints", "axes",
            "articulation_axes", "articulation_axis_tips", "articulation_axis_bases",
            "axis_candidates", "axis_candidate_tips", "axis_candidate_bases", "axis_candidate_labels",
            "keypoint_tracks", "relation_edges", "relation_labels",
            "wrist", "wrist_trail", "wrist_labels",
        ]
    }
    lock = threading.Lock()

    def clear_group(name: str) -> None:
        group = state[name]
        for handle in list(group.values()):
            try:
                handle.remove()
            except Exception:
                pass
        group.clear()

    def nearest_motion_frame(node_id: str, target_frame: int) -> Optional[int]:
        frames_for_node = node_motion_frames.get(node_id, [])
        if not frames_for_node:
            return None
        return min(frames_for_node, key=lambda f: abs(int(f) - int(target_frame)))

    def node_articulation_axis(node_id: str) -> Optional[tuple[np.ndarray, np.ndarray, str]]:
        """Return (axis_direction_unit, axis_point, joint_type) for an articulated node, else None."""
        node = graph.nodes.get(node_id)
        if node is None:
            return None
        for segment in getattr(node, "motion_segments", []) or []:
            direction = np.asarray(getattr(segment, "articulation_axis_direction", []), dtype=np.float32).reshape(-1)
            point = np.asarray(getattr(segment, "articulation_axis_point", []), dtype=np.float32).reshape(-1)
            if direction.size != 3 or point.size != 3:
                continue
            norm = float(np.linalg.norm(direction))
            if norm < 1e-6 or not np.all(np.isfinite(direction)) or not np.all(np.isfinite(point)):
                continue
            return direction / norm, point, str(getattr(segment, "articulation_joint_type", ""))
        return None

    def node_sampled_frame(node_id: str, master_frame: int) -> int:
        """Resolve which frame one node should display at under pin/master controls."""
        ui = node_ui[node_id]
        frames_for_node = ui["frames"]
        if frames_for_node:
            pinned = bool(ui.get("pin") is not None and ui["pin"].value)
            if pinned and ui["state_slider"] is not None:
                target_frame = int(ui["state_slider"].value)
            else:
                target_frame = int(master_frame)
            return nearest_motion_frame(node_id, target_frame) or target_frame
        return int(master_frame)

    def static_node_centroid(node_id: str) -> Optional[np.ndarray]:
        """Fallback position from the node's static graph centroid (semantic-only nodes
        like a handle that shares its parent's motion carry no per-frame geometry)."""
        cen = getattr(graph.nodes.get(node_id), "centroid", None)
        if cen is None:
            return None
        arr = np.asarray(cen, dtype=np.float32).reshape(-1)
        if arr.size < 3 or not np.all(np.isfinite(arr[:3])):
            return None
        return arr[:3]

    def _geom_centroid_at(node_id: str, frame: int) -> Optional[np.ndarray]:
        pts, _ = node_geometry(graph, node_id, int(frame))
        if len(pts) == 0:
            return None
        return np.asarray(pts, dtype=np.float32).mean(axis=0)

    def followed_centroid(node_id: str, master_frame: int) -> Optional[np.ndarray]:
        """Position for a semantic-only node, rigidly carried by its parent object's
        per-frame motion: static_centroid + (parent_centroid(now) - parent_centroid(ref))."""
        static = static_node_centroid(node_id)
        if static is None:
            return None
        parent = element_to_object.get(node_id)
        if not parent:
            return static
        pframes = node_motion_frames.get(parent) or []
        if not pframes:
            return static
        ref = min(pframes)
        now = nearest_motion_frame(parent, master_frame) or ref
        c_ref = _geom_centroid_at(parent, ref)
        c_now = _geom_centroid_at(parent, now)
        if c_ref is None or c_now is None:
            return static
        return (static + (c_now - c_ref)).astype(np.float32)

    def node_display_handle(node_id: str) -> str:
        return display_ids.get(node_id, node_id[:8])

    def node_scene_base(node_id: str) -> str:
        handle = node_display_handle(node_id)
        return sanitize_name(f"{handle}_{graph.nodes[node_id].label}")

    def node_scene_label(node_id: str, frame: int, *, include_frame: bool = True) -> str:
        node = graph.nodes[node_id]
        state_now = node_state_at_frame(node, int(frame), clip_bounds_by_id)
        return format_openfungraph_node_label(
            node,
            node_display_handle(node_id),
            frame=int(frame) if include_frame else None,
            state=state_now,
        )

    def node_display_centroid(node_id: str, master_frame: int) -> Optional[np.ndarray]:
        """Return the rendered centroid for one node at the active/pinned frame."""
        ui = node_ui[node_id]
        if not bool(ui["visible"].value):
            return None
        sampled_frame = node_sampled_frame(node_id, master_frame)
        pts, _node_cols = node_geometry(graph, node_id, int(sampled_frame))
        if len(pts) == 0:
            return followed_centroid(node_id, master_frame)
        pts = np.asarray(pts, dtype=np.float32)
        if args.motion_voxel > 0:
            pts, _dummy = downsample(pts, np.zeros((len(pts), 3), dtype=np.float32), args.motion_voxel)
        if len(pts) == 0:
            return followed_centroid(node_id, master_frame)
        return pts.mean(axis=0).astype(np.float32)

    def render_relationships(master_frame: int) -> None:
        """Draw current functional relationships as 3D line segments between nodes."""
        centroid_cache: dict[str, Optional[np.ndarray]] = {}
        for edge_id, edge in graph.edges.items():
            src_id = str(getattr(edge, "src_id", ""))
            dst_id = str(getattr(edge, "dst_id", ""))
            if not src_id or not dst_id:
                continue
            if src_id not in graph.nodes or dst_id not in graph.nodes:
                continue
            src_centroid = centroid_cache.get(src_id)
            if src_centroid is None and src_id not in centroid_cache:
                src_centroid = node_display_centroid(src_id, master_frame)
                centroid_cache[src_id] = src_centroid
            dst_centroid = centroid_cache.get(dst_id)
            if dst_centroid is None and dst_id not in centroid_cache:
                dst_centroid = node_display_centroid(dst_id, master_frame)
                centroid_cache[dst_id] = dst_centroid
            if src_centroid is None or dst_centroid is None:
                continue
            edge_type = str(getattr(edge, "edge_type", "functional")).lower()
            if style_fungraph:
                color = (100, 100, 100)
            else:
                color = (255, 85, 85) if edge_type == "functional" else (255, 200, 80)
            segs = np.stack([src_centroid, dst_centroid], axis=0).reshape(1, 2, 3).astype(np.float32)
            base = sanitize_name(
                f"{node_display_handle(src_id)}_{node_display_handle(dst_id)}_{edge.relationship}_{edge_id[:8]}"
            )
            state["relation_edges"][edge_id] = server.scene.add_line_segments(
                f"/Edges/{base}" if style_fungraph else f"/graph_edges/{base}",
                segs,
                color,
                line_width=2.5 if style_fungraph else 3.0,
            )
            if show_labels.value:
                mid = ((src_centroid + dst_centroid) * 0.5 + np.array([0.0, 0.0, 0.04], dtype=np.float32)).astype(np.float32)
                label_text = str(edge.relationship or edge.edge_type or "edge")
                state["relation_labels"][edge_id] = server.scene.add_label(
                    f"/EdgeLabels/{base}" if style_fungraph else f"/graph_edges/{base}_label",
                    text=label_text,
                    position=tuple(float(v) for v in mid),
                    font_size_mode="scene",
                    font_scene_height=0.028 if style_fungraph else 0.03,
                    anchor="bottom-center",
                )

    def apply_functional_visibility() -> None:
        """Restrict the viewport to an OpenFunGraph-style geometry + graph overlay."""
        visible_ids = relationship_node_ids if relationship_node_ids else set(node_ids)
        for nid in node_ids:
            node_ui[nid]["visible"].value = nid in visible_ids
        graph_only.value = False
        show_static.value = True
        show_relationships.value = True
        show_labels.value = True
        show_instances.value = True
        show_traj.value = False
        show_centroid.value = False
        show_bboxes.value = False
        semantic_colors.value = False
        show_wrist.value = False
        show_wrist_trail.value = False
        show_all_axes.value = False
        show_articulation_axes.value = False
        show_axis_candidates.value = False
        show_keypoint_trajectories.value = False

    def apply_full_scene_visibility() -> None:
        """Restore the richer geometry-first scene view."""
        for nid in node_ids:
            node_ui[nid]["visible"].value = True
        graph_only.value = False
        show_static.value = True
        show_relationships.value = style_fungraph
        show_labels.value = True
        show_instances.value = style_fungraph

    def render_node(node_id: str, *, base_point_size: float, master_frame: int) -> None:
        ui = node_ui[node_id]
        if not bool(ui["visible"].value):
            return
        node = graph.nodes[node_id]
        frames_for_node = ui["frames"]
        sampled_frame = node_sampled_frame(node_id, master_frame)
        if frames_for_node and ui["state_slider"] is not None and not (ui.get("pin") is not None and ui["pin"].value):
            if int(ui["state_slider"].value) != int(master_frame):
                try:
                    ui["state_slider"].value = int(master_frame)
                except Exception:
                    pass
        pts, node_cols = node_geometry(graph, node_id, int(sampled_frame))
        if len(pts) == 0:
            # Semantic-only node (e.g. a handle that shares its parent's motion): no
            # per-frame geometry, but it still has a static graph centroid. In graph-only
            # or functional-overlay mode draw it there so handle + door both appear and
            # the edge connects.
            if bool(graph_only.value or functional_only.value):
                centroid = followed_centroid(node_id, master_frame)
                if centroid is not None:
                    color = mask_color(node_id, node.node_type, node.label)
                    base = node_scene_base(node_id)
                    state["nodes"][node_id] = server.scene.add_icosphere(
                        f"/{node_display_handle(node_id)}/node" if style_fungraph else f"/graph/{base}/node",
                        radius=float(max(0.03, base_point_size * 3.5)),
                        color=tuple(int(c) for c in color.reshape(-1)[:3]),
                        position=tuple(float(v) for v in centroid),
                        subdivisions=3,
                        material="standard",
                    )
                    if show_labels.value:
                        state["labels"][node_id] = server.scene.add_label(
                            f"/Label{node_display_handle(node_id)}" if style_fungraph else f"/graph/{base}/label",
                            text=node_scene_label(node_id, sampled_frame, include_frame=False),
                            position=centroid + np.array([0.0, 0.0, 0.06], dtype=np.float32),
                            font_size_mode="scene",
                            font_scene_height=0.04,
                            anchor="bottom-center",
                        )
            return
        pts = np.asarray(pts, dtype=np.float32)
        color = mask_color(node_id, node.node_type, node.label)
        if semantic_colors.value:
            rgb_f = np.repeat((color.astype(np.float32) / 255.0).reshape(1, 3), len(pts), axis=0)
        else:
            rgb_f = normalize_float_colors(node_cols, len(pts), color.astype(np.float32) / 255.0)
        if args.motion_voxel > 0:
            pts, rgb_f = downsample(pts, rgb_f, args.motion_voxel)
        if len(pts) == 0:
            return
        cols = np.clip(rgb_f * 255.0, 0, 255).astype(np.uint8)
        base = node_scene_base(node_id)
        centroid = pts.mean(axis=0)

        if bool(graph_only.value):
            # Graph-only: draw the node as a single sphere at its centroid (colored
            # by node type) plus its label. Edges are drawn by render_relationships.
            # Everything else (dense cloud, axes, keypoints) is skipped.
            state["nodes"][node_id] = server.scene.add_icosphere(
                f"/{node_display_handle(node_id)}/node" if style_fungraph else f"/graph/{base}/node",
                radius=float(max(0.03, base_point_size * 3.5)),
                color=tuple(int(c) for c in color.reshape(-1)[:3]),
                position=tuple(float(v) for v in centroid),
                subdivisions=3,
                material="standard",
            )
            if show_labels.value:
                state["labels"][node_id] = server.scene.add_label(
                    f"/Label{node_display_handle(node_id)}" if style_fungraph else f"/graph/{base}/label",
                    text=node_scene_label(node_id, sampled_frame),
                    position=centroid + np.array([0.0, 0.0, 0.06], dtype=np.float32),
                    font_size_mode="scene",
                    font_scene_height=0.04,
                    anchor="bottom-center",
                )
            return

        state["nodes"][node_id] = server.scene.add_point_cloud(
            f"/{node_display_handle(node_id)}" if style_fungraph else f"/graph/{base}/points",
            pts,
            cols,
            point_size=base_point_size,
            point_shape="rounded",
            point_shading="gradient",
        )
        if show_instances.value:
            state["instances"][node_id] = server.scene.add_icosphere(
                f"/Ins{node_display_handle(node_id)}" if style_fungraph else f"/graph/{base}/instance",
                radius=float(max(0.028, base_point_size * (3.0 if str(node.node_type).lower() == "element" else 3.6))),
                color=tuple(int(c) for c in color.reshape(-1)[:3]),
                position=tuple(float(v) for v in centroid),
                subdivisions=3,
                material="standard",
            )
        if show_centroid.value:
            state["centroids"][node_id] = server.scene.add_point_cloud(
                f"/graph/{base}/centroid",
                centroid.reshape(1, 3),
                color.reshape(1, 3),
                point_size=base_point_size * 4.0,
                point_shape="circle",
                point_shading="flat",
            )
        if show_labels.value:
            state["labels"][node_id] = server.scene.add_label(
                f"/Label{node_display_handle(node_id)}" if style_fungraph else f"/graph/{base}/label",
                text=node_scene_label(node_id, sampled_frame),
                position=centroid + np.array([0.0, 0.0, 0.06], dtype=np.float32),
                font_size_mode="scene",
                font_scene_height=0.04,
                anchor="bottom-center",
            )
        if show_bboxes.value:
            segs = bbox_segments(pts)
            if len(segs):
                state["bboxes"][node_id] = server.scene.add_line_segments(
                    f"/BBox{node_display_handle(node_id)}" if style_fungraph else f"/graph/{base}/bbox",
                    segs,
                    tuple(int(v) for v in color),
                    line_width=2.0,
                )
            if args.groundtruth_dir is not None and args.groundtruth_object is not None:
                print("hi")
                gt = load_gt_bbox(
                    args.groundtruth_dir,
                    master_frame,
                    args.groundtruth_object,
                )

                print(gt)

                if gt is not None:
                    center, dims = gt

                    segs = gt_bbox_segments(center, dims)

                    colors = np.tile(
                        np.array([[255, 255, 0]], dtype=np.uint8),
                        (len(segs), 1),
                    )

                    server.scene.add_line_segments(
                        "/GroundTruthBBox",
                        points=segs,
                        colors=np.array([255, 255, 0], dtype=np.uint8),
                        line_width=4.0,
                    )
        if bool(ui["axis"].value) or bool(show_all_axes.value):
            state["axes"][node_id] = server.scene.add_frame(
                f"/graph/{base}/axis",
                position=tuple(float(v) for v in centroid),
                wxyz=(1.0, 0.0, 0.0, 0.0),
                axes_length=0.18,
                axes_radius=0.006,
            )
        # ArtiPoint-style red articulation axis (Fig. 2 of the paper). Drawn as
        # a line segment through the fitted axis_point along axis_direction,
        # length scaled to the node's bbox so the arrow spans the moving part.
        if show_articulation_axes.value:
            axis_info = node_articulation_axis(node_id)
            if axis_info is not None:
                direction, point_on_axis, _joint_type = axis_info
                pts_extent = np.linalg.norm(pts.max(axis=0) - pts.min(axis=0)) if len(pts) else 0.0
                axis_length = float(max(0.30, min(2.0, pts_extent * 1.1)))
                half = axis_length * 0.5
                # The Plücker formula returns the closest point on the screw
                # line to the world origin — for a vertical hinge through a
                # door, that lands at z=0 (floor). Slide the visualisation
                # point along the axis line to land at the node's centroid
                # so the arrow renders through the door, not under the floor.
                # Mathematically identical screw axis; just a nicer pivot.
                if len(pts):
                    node_centroid = pts.mean(axis=0)
                    offset = node_centroid - point_on_axis
                    slide = float(direction @ offset)
                    point_on_axis = point_on_axis + direction * slide
                base_pt = (point_on_axis - direction * half).astype(np.float32)
                tip_pt = (point_on_axis + direction * half).astype(np.float32)
                red = (220, 20, 30)
                shaft = np.stack([base_pt, tip_pt], axis=0).reshape(1, 2, 3).astype(np.float32)
                state["articulation_axes"][node_id] = server.scene.add_line_segments(
                    f"/graph/{base}/articulation_axis",
                    shaft,
                    red,
                    line_width=6.0,
                )
                state["articulation_axis_tips"][node_id] = server.scene.add_icosphere(
                    f"/graph/{base}/articulation_axis_tip",
                    radius=base_point_size * 3.0,
                    color=red,
                    position=tuple(float(v) for v in tip_pt),
                    subdivisions=3,
                    material="standard",
                )
                state["articulation_axis_bases"][node_id] = server.scene.add_icosphere(
                    f"/graph/{base}/articulation_axis_base",
                    radius=base_point_size * 1.6,
                    color=red,
                    position=tuple(float(v) for v in base_pt),
                    subdivisions=2,
                    material="standard",
                )
        # Debug-only: every alternative hinge candidate from different
        # estimators, each in its own colour. Lets you see at a glance which
        # method (native factor graph / Plücker / closed-form / wrist arc /
        # per-track cluster) recovered the physical hinge correctly.
        if show_axis_candidates.value:
            seen_candidates: List[dict] = []
            seen_candidate_keys = set()
            for segment in getattr(node, "motion_segments", []) or []:
                cands = list(getattr(segment, "articulation_axis_candidates", []) or [])
                for cand in cands:
                    try:
                        cand_dir = tuple(round(float(v), 5) for v in np.asarray(cand.get("axis_direction"), dtype=np.float32).reshape(3))
                        cand_pt = tuple(round(float(v), 5) for v in np.asarray(cand.get("axis_point"), dtype=np.float32).reshape(3))
                    except Exception:
                        continue
                    cand_key = (str(cand.get("name", "")), cand_dir, cand_pt)
                    if cand_key in seen_candidate_keys:
                        continue
                    seen_candidate_keys.add(cand_key)
                    seen_candidates.append(cand)
            if seen_candidates and len(pts):
                cand_pts_extent = float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0)))
                cand_axis_length = float(max(0.30, min(2.0, cand_pts_extent * 1.1)))
                cand_half = cand_axis_length * 0.5
                node_centroid_cand = pts.mean(axis=0)
                for i_cand, cand in enumerate(seen_candidates):
                    try:
                        cand_dir = np.asarray(cand.get("axis_direction"), dtype=np.float32).reshape(3)
                        cand_pt = np.asarray(cand.get("axis_point"), dtype=np.float32).reshape(3)
                    except Exception:
                        continue
                    cnorm = float(np.linalg.norm(cand_dir))
                    if cnorm < 1e-6 or not np.all(np.isfinite(cand_dir)) or not np.all(np.isfinite(cand_pt)):
                        continue
                    cand_dir = cand_dir / cnorm
                    # Slide pivot along axis to nearest point from cloud centroid (visualisation only).
                    offset_cand = node_centroid_cand - cand_pt
                    slide_cand = float(cand_dir @ offset_cand)
                    cand_pt_vis = cand_pt + cand_dir * slide_cand
                    cand_base_pt = (cand_pt_vis - cand_dir * cand_half).astype(np.float32)
                    cand_tip_pt = (cand_pt_vis + cand_dir * cand_half).astype(np.float32)
                    cand_color_raw = cand.get("color", [200, 200, 200])
                    try:
                        cand_color = tuple(int(c) for c in cand_color_raw[:3])
                    except Exception:
                        cand_color = (200, 200, 200)
                    cand_name = str(cand.get("name", f"cand_{i_cand}"))
                    is_chosen = bool(cand.get("chosen_for_rendering", False))
                    if is_chosen:
                        cand_name = f"{cand_name}*"  # asterisk = "this is what the cloud rotates around"
                    cand_shaft = np.stack([cand_base_pt, cand_tip_pt], axis=0).reshape(1, 2, 3).astype(np.float32)
                    handle_key = f"{node_id}::{i_cand}"
                    state["axis_candidates"][handle_key] = server.scene.add_line_segments(
                        f"/graph/{base}/axis_candidate_{i_cand}_{sanitize_name(cand_name)}",
                        cand_shaft,
                        cand_color,
                        line_width=8.0 if is_chosen else 4.5,
                    )
                    state["axis_candidate_tips"][handle_key] = server.scene.add_icosphere(
                        f"/graph/{base}/axis_candidate_{i_cand}_tip",
                        radius=base_point_size * 2.5,
                        color=cand_color,
                        position=tuple(float(v) for v in cand_tip_pt),
                        subdivisions=3,
                        material="standard",
                    )
                    state["axis_candidate_bases"][handle_key] = server.scene.add_icosphere(
                        f"/graph/{base}/axis_candidate_{i_cand}_base",
                        radius=base_point_size * 1.4,
                        color=cand_color,
                        position=tuple(float(v) for v in cand_base_pt),
                        subdivisions=2,
                        material="standard",
                    )
                    state["axis_candidate_labels"][handle_key] = server.scene.add_label(
                        f"/graph/{base}/axis_candidate_{i_cand}_label",
                        text=cand_name,
                        position=tuple(float(v) for v in (cand_tip_pt + np.array([0.0, 0.0, 0.03], dtype=np.float32))),
                        font_size_mode="scene",
                        font_scene_height=0.03,
                        anchor="bottom-center",
                    )
        # Per-corner 3-D keypoint trajectories (ArtiPoint Fig. 2 panel 4): a
        # bundle of curves, one per tracked image corner, across the whole
        # clip. Driven by `corner_tracks_world` on the motion segment.
        if show_keypoint_trajectories.value:
            for segment in getattr(node, "motion_segments", []) or []:
                tracks = np.asarray(getattr(segment, "corner_tracks_world", []), dtype=np.float32)
                valid = np.asarray(getattr(segment, "corner_tracks_valid", []), dtype=bool)
                if tracks.ndim != 3 or tracks.shape[0] < 2 or tracks.shape[1] == 0:
                    continue
                if valid.shape != tracks.shape[:2]:
                    valid = np.all(np.isfinite(tracks), axis=2)
                n_tracks_total = tracks.shape[1]
                max_curves = int(keypoint_trajectory_max.value)
                if n_tracks_total > max_curves:
                    # Subsample uniformly so the bundle stays representative
                    # without dominating the viewport.
                    chosen = np.linspace(0, n_tracks_total - 1, num=max_curves, dtype=np.int32)
                else:
                    chosen = np.arange(n_tracks_total, dtype=np.int32)
                segments_list: list[np.ndarray] = []
                for track_idx in chosen:
                    column = tracks[:, int(track_idx), :]
                    column_valid = valid[:, int(track_idx)]
                    # Build polyline by stitching consecutive valid samples.
                    for t in range(column.shape[0] - 1):
                        if not (column_valid[t] and column_valid[t + 1]):
                            continue
                        if not (np.all(np.isfinite(column[t])) and np.all(np.isfinite(column[t + 1]))):
                            continue
                        segments_list.append(np.stack([column[t], column[t + 1]], axis=0))
                if not segments_list:
                    continue
                segs = np.stack(segments_list, axis=0).astype(np.float32)
                key = f"{node_id}_{segment.segment_id}"
                state["keypoint_tracks"][key] = server.scene.add_line_segments(
                    f"/graph/{base}/keypoint_tracks_{segment.segment_id[:6]}",
                    segs,
                    tuple(int(v) for v in color),
                    line_width=1.2,
                )
        traj = node_trajectory_positions.get(node_id, np.zeros((0, 3), dtype=np.float32))
        if show_traj.value and len(traj) >= 2:
            segs = np.stack([traj[:-1], traj[1:]], axis=1).astype(np.float32)
            state["traj"][node_id] = server.scene.add_line_segments(
                f"/graph/{base}/trajectory",
                segs,
                tuple(int(v) for v in color),
                line_width=3.0,
            )
        # Clickable waypoints (small spheres on each motion sample).
        if show_traj.value and ui["state_slider"] is not None and len(traj) >= 1:
            for i, (pos, f) in enumerate(zip(traj, frames_for_node)):
                handle = server.scene.add_icosphere(
                    f"/graph/{base}/waypoint_{i:03d}",
                    radius=base_point_size * 1.3,
                    color=tuple(int(v) for v in color),
                    position=tuple(float(v) for v in pos),
                    subdivisions=2,
                    material="standard",
                )

                def _on_waypoint_click(event, frame_value=int(f), nid=node_id) -> None:
                    # Move the master timeline; unpinned nodes follow automatically.
                    # If the user already pinned this node, also bump its slider.
                    master_frame_slider.value = int(frame_value)
                    ui_local = node_ui[nid]
                    if (
                        ui_local.get("pin") is not None
                        and ui_local["pin"].value
                        and ui_local["state_slider"] is not None
                    ):
                        ui_local["state_slider"].value = int(frame_value)

                handle.on_click(_on_waypoint_click)
                state["waypoints"][f"{node_id}_{i}"] = handle

    def render_wrists(frame: int) -> None:
        if not show_wrist.value:
            return
        for i, bundle in enumerate(wrist_bundles):
            latest_frame, latest_pos, trail = wrist_trail(
                bundle,
                int(frame),
                int(wrist_trail_frames.value) if show_wrist_trail.value else 1,
            )
            if latest_pos is None:
                continue
            hand = str(bundle.get("hand", "hand"))
            clip_id = int(bundle.get("clip_id", -1))
            color = np.asarray(bundle.get("color", wrist_color(hand)), dtype=np.uint8)
            key = sanitize_name(f"clip{clip_id}_{hand}_{i}")
            if show_wrist_trail.value and len(trail) >= 2:
                segs = np.stack([trail[:-1], trail[1:]], axis=1).astype(np.float32)
                state["wrist_trail"][key] = server.scene.add_line_segments(
                    f"/wrist/{key}/trail",
                    segs,
                    tuple(int(v) for v in color),
                    line_width=2.5,
                )
            state["wrist"][key] = server.scene.add_icosphere(
                f"/wrist/{key}/current",
                radius=float(point_size_slider.value) * 2.2,
                color=tuple(int(v) for v in color),
                position=tuple(float(v) for v in latest_pos),
                subdivisions=3,
                material="standard",
            )
            if show_labels.value:
                state["wrist_labels"][key] = server.scene.add_label(
                    f"/wrist/{key}/label",
                    text=f"{hand} | clip {clip_id} | frame {latest_frame}",
                    position=latest_pos + np.array([0.0, 0.0, 0.06], dtype=np.float32),
                    font_size_mode="scene",
                    font_scene_height=0.032,
                    anchor="bottom-center",
                )

    def render_rgbd_panel(frame: int) -> None:
        rgbd_frame = nearest_rgbd_frame(int(frame))
        if show_rgbd.value and rgbd_frame is not None:
            if rgbd_frame not in rgbd_cache:
                rgbd_cache[rgbd_frame] = make_rgbd_panel(output_dir, rgbd_assets.get(rgbd_frame))
            rgbd_image.image = rgbd_cache[rgbd_frame]
            rgbd_frame_text.value = str(int(rgbd_frame))
        else:
            rgbd_image.image = make_rgbd_panel(output_dir, None)
            rgbd_frame_text.value = "—"

    def rerender(_=None) -> None:
        with lock:
            for group_name in state:
                clear_group(group_name)
            if static_scene is not None:
                static_scene.visible = bool(show_static.value) and not bool(graph_only.value)
            base_point_size = float(point_size_slider.value)
            master_frame = int(master_frame_slider.value)
            frame_semantics_md.content = format_live_semantics_md(
                graph,
                node_ids,
                display_ids,
                master_frame,
                clip_events,
                clip_bounds_by_id,
            )
            for node_id in node_ids:
                render_node(node_id, base_point_size=base_point_size, master_frame=master_frame)
            if show_relationships.value or bool(graph_only.value):
                render_relationships(master_frame)
            render_wrists(master_frame)
            render_rgbd_panel(master_frame)

    # ---- Wire up update callbacks ----
    for widget in [
        show_static, point_size_slider, show_relationships, graph_only, functional_only, show_all_axes, show_articulation_axes,
        show_axis_candidates,
        show_keypoint_trajectories, keypoint_trajectory_max,
        show_instances, show_bboxes, semantic_colors,
        show_traj, show_centroid, show_labels, show_wrist, show_wrist_trail,
        wrist_trail_frames, show_rgbd, master_frame_slider,
    ]:
        widget.on_update(rerender)
    for node_id in node_ids:
        ui = node_ui[node_id]
        # Only pinned per-node sliders re-trigger rerender (otherwise we'd
        # re-render twice when the master also bumps them).
        if ui["state_slider"] is not None:
            ui["state_slider"].on_update(
                lambda _e, nid=node_id: rerender() if (node_ui[nid].get("pin") is not None and node_ui[nid]["pin"].value) else None
            )
        if ui.get("pin") is not None:
            ui["pin"].on_update(rerender)
        ui["axis"].on_update(rerender)
        ui["visible"].on_update(rerender)

    # ---- Bulk control wiring ----
    @btn_show_all.on_click
    def _(_event) -> None:
        for nid in node_ids:
            node_ui[nid]["visible"].value = True

    @btn_hide_all.on_click
    def _(_event) -> None:
        for nid in node_ids:
            node_ui[nid]["visible"].value = False

    @btn_elements_only.on_click
    def _(_event) -> None:
        for nid in node_ids:
            node_ui[nid]["visible"].value = (
                str(graph.nodes[nid].node_type).lower() == "element"
            )

    @btn_objects_only.on_click
    def _(_event) -> None:
        for nid in node_ids:
            node_ui[nid]["visible"].value = (
                str(graph.nodes[nid].node_type).lower() != "element"
            )

    @btn_functional_graph.on_click
    def _(_event) -> None:
        functional_only.value = True
        apply_functional_visibility()

    @btn_full_scene.on_click
    def _(_event) -> None:
        functional_only.value = False
        apply_full_scene_visibility()

    @btn_reset_frames.on_click
    def _(_event) -> None:
        for nid in node_ids:
            ui = node_ui[nid]
            frames_for_node = ui["frames"]
            if ui["state_slider"] is not None and frames_for_node:
                ui["state_slider"].value = int(min(frames_for_node))
            if ui.get("pin") is not None:
                ui["pin"].value = False
        master_frame_slider.value = int(frame_min)

    @filter_text.on_update
    def _(_event) -> None:
        needle = str(filter_text.value).strip().lower()
        for nid in node_ids:
            if not needle:
                node_ui[nid]["visible"].value = (
                    nid in relationship_node_ids if functional_only.value and relationship_node_ids else True
                )
            else:
                label = str(graph.nodes[nid].label).lower()
                node_ui[nid]["visible"].value = needle in label

    @functional_only.on_update
    def _(_event) -> None:
        if functional_only.value:
            apply_functional_visibility()
        else:
            apply_full_scene_visibility()

    # ---- Focus & playback handlers ----
    def _focus_members() -> list[str]:
        return focus_label_to_members.get(str(focus_dropdown.value), [])

    def _apply_solo() -> None:
        members = set(_focus_members())
        if bool(solo_focus.value) and members:
            for nid in node_ids:
                node_ui[nid]["visible"].value = nid in members
        elif functional_only.value:
            apply_functional_visibility()
        else:
            apply_full_scene_visibility()

    @solo_focus.on_update
    def _(_event) -> None:
        _apply_solo()

    @focus_dropdown.on_update
    def _(_event) -> None:
        if bool(solo_focus.value):
            _apply_solo()

    def _focus_center_radius(master_frame: int):
        all_pts: list[np.ndarray] = []
        for nid in _focus_members():
            sampled = node_sampled_frame(nid, master_frame)
            pts, _cols = node_geometry(graph, nid, int(sampled))
            if len(pts):
                all_pts.append(np.asarray(pts, dtype=np.float32))
        if all_pts:
            stacked = np.concatenate(all_pts, axis=0)
            center = stacked.mean(axis=0).astype(np.float64)
            radius = float(max(np.linalg.norm(stacked.max(axis=0) - stacked.min(axis=0)) * 0.5, 0.3))
            return center, radius
        for nid in _focus_members():
            fallback = followed_centroid(nid, master_frame) or static_node_centroid(nid)
            if fallback is not None:
                return np.asarray(fallback, dtype=np.float64), 0.6
        return None, None

    @btn_fly_to.on_click
    def _(_event) -> None:
        if not _focus_members():
            return
        if bool(solo_focus.value):
            _apply_solo()
        center, radius = _focus_center_radius(int(master_frame_slider.value))
        if center is None:
            return
        position = (
            float(center[0] + radius * 0.8),
            float(center[1] - radius * 2.4),
            float(center[2] + radius * 1.0),
        )
        for client in server.get_clients().values():
            client.camera.look_at = (float(center[0]), float(center[1]), float(center[2]))
            client.camera.position = position

    def play_loop() -> None:
        # Single master clock — render_node snaps each unpinned node to the
        # nearest motion sample, so we only need to advance one slider here.
        last_t = time.monotonic()
        while True:
            now = time.monotonic()
            if bool(global_autoplay.value):
                fps = max(0.1, float(global_speed.value))
                if (now - last_t) >= (1.0 / fps):
                    last_t = now
                    current = int(master_frame_slider.value)
                    play_members = focus_label_to_members.get(str(play_dropdown.value), None)
                    if str(play_dropdown.value) != "Master clock" and play_members:
                        # Step through the focused object's OWN motion samples so the
                        # viewer watches it move (idle frames between samples skipped).
                        motion = sorted({
                            int(f) for nid in play_members
                            for f in node_motion_frames.get(nid, [])
                        })
                        if motion:
                            next_frame = next((f for f in motion if f > current), motion[0])
                        else:
                            next_frame = current + 1 if current < frame_max else frame_min
                    else:
                        next_frame = current + 1
                        if next_frame > frame_max:
                            next_frame = frame_min
                    master_frame_slider.value = int(next_frame)
            time.sleep(0.02)

    threading.Thread(target=play_loop, daemon=True).start()
    if args.functional_only:
        apply_functional_visibility()
    rerender()

    # Detect the actual login-node hostname so the tunnel hint points where
    # this script is running. ETH student cluster has multiple login nodes
    # (student-cluster1, student-cluster2, ...) and the user has to tunnel
    # specifically to the one viser is bound on.
    short_host = socket.gethostname()
    if "." in short_host:
        short_host = short_host.split(".", 1)[0]
    if short_host.startswith("student-cluster"):
        login_host = f"{short_host}.inf.ethz.ch"
    else:
        # Fallback for unrecognised hosts (compute nodes, dev VMs, etc.)
        try:
            login_host = socket.getfqdn()
        except Exception:
            login_host = short_host or "<this-host>"
    print(f"viser viewer ready at http://{args.host}:{args.port}")
    print("From your laptop, tunnel with:")
    print(f"  ssh -L {args.port}:127.0.0.1:{args.port} {getpass.getuser()}@{login_host}")
    print("Then open locally:")
    print(f"  http://127.0.0.1:{args.port}")
    server.sleep_forever()


if __name__ == "__main__":
    main()

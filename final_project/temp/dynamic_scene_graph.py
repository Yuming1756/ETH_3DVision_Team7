"""Dynamic functional scene graph data structures and persistence helpers."""

from __future__ import annotations

import json
import re
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import open3d as o3d
from scipy.spatial import KDTree


def _normalize_label(label: str) -> str:
    """Normalize a label for graph indexing and matching."""
    return str(label).strip().lower()


NODE_MATCH_MAX_CENTROID_DISTANCE_M = 0.35
NODE_MATCH_OVERLAP_RADIUS_M = 0.08
NODE_MATCH_MIN_OVERLAP_RATIO = 0.12
NODE_MATCH_MAX_MEDIAN_NN_DISTANCE_M = 0.10

_LABEL_TOKEN_ALIASES = {
    "refrigerator": "fridge",
    "fridge": "fridge",
    "cupboard": "cabinet",
    "television": "tv",
    "mug": "cup",
}


def _canonicalize_label_token(token: str) -> str:
    """Normalize one label token for alias-aware matching."""
    token = str(token).strip().lower()
    if not token:
        return ""
    if len(token) > 4 and token.endswith("ies"):
        token = token[:-3] + "y"
    elif len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        token = token[:-1]
    return _LABEL_TOKEN_ALIASES.get(token, token)


def _label_token_set(label: str) -> set[str]:
    """Return a canonical token set for approximate label matching."""
    return {
        canonical
        for canonical in (
            _canonicalize_label_token(token)
            for token in re.findall(r"[a-z0-9]+", _normalize_label(label))
        )
        if canonical
    }


def _labels_are_compatible(query_label: str, candidate_label: str) -> bool:
    """Return whether two labels are similar enough to be geometric-merge candidates."""
    query_norm = _normalize_label(query_label)
    candidate_norm = _normalize_label(candidate_label)
    if not query_norm or not candidate_norm:
        return False
    if query_norm == candidate_norm:
        return True
    query_tokens = _label_token_set(query_norm)
    candidate_tokens = _label_token_set(candidate_norm)
    if not query_tokens or not candidate_tokens:
        return False
    if query_tokens == candidate_tokens:
        return True
    if query_tokens.issubset(candidate_tokens) or candidate_tokens.issubset(query_tokens):
        return True
    shared = query_tokens & candidate_tokens
    union = query_tokens | candidate_tokens
    return bool(shared) and (len(shared) >= 2 or (len(union) > 0 and (len(shared) / len(union)) >= (2.0 / 3.0)))


def _point_cloud_overlap_metrics(
    query_points: np.ndarray,
    reference_points: np.ndarray,
    radius_m: float = NODE_MATCH_OVERLAP_RADIUS_M,
) -> dict:
    """Measure how well one point cloud overlaps another with nearest-neighbor distances."""
    query = np.asarray(query_points, dtype=np.float32)
    reference = np.asarray(reference_points, dtype=np.float32)
    if len(query) == 0 or len(reference) == 0:
        return {
            "overlap_ratio": 0.0,
            "median_nn_distance_m": float("inf"),
            "p90_nn_distance_m": float("inf"),
            "max_nn_distance_m": float("inf"),
        }
    tree = KDTree(reference.astype(np.float64))
    distances, _ = tree.query(query.astype(np.float64), k=1)
    distances = np.asarray(distances, dtype=np.float32).reshape(-1)
    if distances.size == 0:
        return {
            "overlap_ratio": 0.0,
            "median_nn_distance_m": float("inf"),
            "p90_nn_distance_m": float("inf"),
            "max_nn_distance_m": float("inf"),
        }
    return {
        "overlap_ratio": float(np.mean(distances <= float(radius_m))),
        "median_nn_distance_m": float(np.median(distances)),
        "p90_nn_distance_m": float(np.percentile(distances, 90)),
        "max_nn_distance_m": float(np.max(distances)),
    }


def _empty_points() -> np.ndarray:
    """Return an empty point array with the expected dtype."""
    return np.zeros((0, 3), dtype=np.float32)


def _identity_pose() -> np.ndarray:
    """Return a 4x4 identity transform with the expected dtype."""
    return np.eye(4, dtype=np.float32)


def _as_pose_matrix(pose: Optional[np.ndarray]) -> np.ndarray:
    """Normalize a pose-like value into a 4x4 float32 transform."""
    if pose is None:
        return _identity_pose()
    matrix = np.asarray(pose, dtype=np.float32)
    if matrix.shape != (4, 4):
        raise ValueError(f"Expected pose shape (4, 4), got {matrix.shape}")
    return matrix


def _apply_homogeneous_transform(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """Apply a 4x4 rigid transform to a point cloud."""
    pts = np.asarray(points, dtype=np.float32)
    if len(pts) == 0:
        return _empty_points()
    pts_h = np.concatenate([pts, np.ones((len(pts), 1), dtype=np.float32)], axis=1)
    return (np.asarray(transform, dtype=np.float32) @ pts_h.T).T[:, :3].astype(np.float32)


@dataclass
class MotionPoseSample:
    """One exact pose sample for a node at a specific frame."""

    frame: int
    pose: np.ndarray
    source: str = "tracked"
    confidence: float = 1.0
    clip_id: int = -1

    def to_dict(self) -> dict:
        """Serialize a motion sample for JSON persistence."""
        return {
            "frame": int(self.frame),
            "pose": _as_pose_matrix(self.pose).tolist(),
            "source": self.source,
            "confidence": float(self.confidence),
            "clip_id": int(self.clip_id),
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "MotionPoseSample":
        """Create a motion sample from JSON metadata."""
        return cls(
            frame=int(payload.get("frame", -1)),
            pose=_as_pose_matrix(payload.get("pose")),
            source=str(payload.get("source", "tracked")),
            confidence=float(payload.get("confidence", 1.0)),
            clip_id=int(payload.get("clip_id", -1)),
        )


@dataclass
class NodeMotionSegment:
    """A transform history segment for one node during one interaction interval."""

    segment_id: str
    clip_id: int
    frame_start: int
    frame_end: int
    reference_pose: np.ndarray
    reference_points: np.ndarray
    reference_colors: np.ndarray
    prior_source: str = ""
    reference_pointcloud_file: str = ""
    pose_history: List[MotionPoseSample] = field(default_factory=list)
    # ArtiPoint-style 1-DoF articulation, populated only for case 3 (handle on
    # an articulated part). `joint_type` is "revolute" or "prismatic"; axis
    # direction is a unit world-space vector; axis_point is a world-space
    # point on the joint. Empty arrays signal "no articulation fit".
    articulation_joint_type: str = ""
    articulation_axis_direction: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    articulation_axis_point: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    # Per-corner 3-D trajectories from CoTracker3 + Eq.(1) smoothing, used by
    # the viser viewer to draw the ArtiPoint Fig.2-style "spaghetti" of
    # tracked keypoints across the clip. ``corner_tracks_world`` has shape
    # (T_frames, N_tracks, 3); ``corner_tracks_valid`` is (T_frames, N_tracks);
    # ``corner_track_frames`` lists the absolute clip frame index per row.
    corner_tracks_world: np.ndarray = field(default_factory=lambda: np.zeros((0, 0, 3), dtype=np.float32))
    corner_tracks_valid: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=bool))
    corner_track_frames: List[int] = field(default_factory=list)
    # Debug-only alternative hinge candidates from different estimators
    # (native factor graph, Plücker closest-point, closed-form, wrist arc,
    # per-track cluster). Each entry: {name, axis_direction[3], axis_point[3],
    # color[3]}. Viser draws each as a distinct-coloured arrow so you can
    # eyeball which method recovered the physical hinge correctly.
    articulation_axis_candidates: List[dict] = field(default_factory=list)

    def get_pose_at_frame(self, frame: int, exact: bool = True) -> Optional[np.ndarray]:
        """Return the pose sample for a frame, optionally allowing carry-forward lookup."""
        samples = sorted(self.pose_history, key=lambda sample: sample.frame)
        if exact:
            for sample in samples:
                if int(sample.frame) == int(frame):
                    return _as_pose_matrix(sample.pose)
            return None
        eligible = [sample for sample in samples if int(sample.frame) <= int(frame)]
        if not eligible:
            return None
        return _as_pose_matrix(eligible[-1].pose)

    def render_geometry_at_frame(self, frame: int, exact: bool = True) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Render this segment's reference geometry at the requested frame."""
        pose = self.get_pose_at_frame(frame, exact=exact)
        if pose is None or len(self.reference_points) == 0:
            return None
        delta = np.asarray(pose, dtype=np.float32) @ np.linalg.inv(_as_pose_matrix(self.reference_pose))
        points = _apply_homogeneous_transform(self.reference_points, delta)
        colors = np.asarray(self.reference_colors, dtype=np.float32)
        if len(colors) != len(points):
            colors = np.full((len(points), 3), 0.5, dtype=np.float32)
        return points, colors

    def to_dict(self) -> dict:
        """Serialize motion metadata for JSON persistence."""
        payload = {
            "segment_id": self.segment_id,
            "clip_id": int(self.clip_id),
            "frame_start": int(self.frame_start),
            "frame_end": int(self.frame_end),
            "reference_pose": _as_pose_matrix(self.reference_pose).tolist(),
            "n_reference_points": int(len(self.reference_points)),
            "prior_source": self.prior_source,
            "reference_pointcloud_file": self.reference_pointcloud_file,
            "pose_history": [sample.to_dict() for sample in sorted(self.pose_history, key=lambda item: item.frame)],
        }
        if self.articulation_joint_type:
            payload["articulation_joint_type"] = str(self.articulation_joint_type)
        if isinstance(self.articulation_axis_direction, np.ndarray) and self.articulation_axis_direction.size == 3:
            payload["articulation_axis_direction"] = self.articulation_axis_direction.astype(np.float32).tolist()
        if isinstance(self.articulation_axis_point, np.ndarray) and self.articulation_axis_point.size == 3:
            payload["articulation_axis_point"] = self.articulation_axis_point.astype(np.float32).tolist()
        # Per-corner trajectories are too large for JSON; serialise inline as
        # nested lists with a NaN sentinel for invalid samples. Renderer
        # treats NaN as a polyline break.
        if (
            isinstance(self.corner_tracks_world, np.ndarray)
            and self.corner_tracks_world.ndim == 3
            and self.corner_tracks_world.size > 0
        ):
            tracks = self.corner_tracks_world.astype(np.float32).copy()
            if (
                isinstance(self.corner_tracks_valid, np.ndarray)
                and self.corner_tracks_valid.shape == tracks.shape[:2]
            ):
                tracks[~self.corner_tracks_valid] = np.nan
            payload["corner_tracks_world"] = tracks.tolist()
            payload["corner_track_frames"] = [int(f) for f in self.corner_track_frames]
        if self.articulation_axis_candidates:
            payload["articulation_axis_candidates"] = list(self.articulation_axis_candidates)
        return payload

    @classmethod
    def from_dict(cls, payload: dict) -> "NodeMotionSegment":
        """Create a motion segment from JSON metadata."""
        axis_dir_raw = payload.get("articulation_axis_direction")
        axis_pt_raw = payload.get("articulation_axis_point")
        axis_dir = (
            np.asarray(axis_dir_raw, dtype=np.float32).reshape(3)
            if axis_dir_raw is not None and len(axis_dir_raw) == 3
            else np.zeros(0, dtype=np.float32)
        )
        axis_pt = (
            np.asarray(axis_pt_raw, dtype=np.float32).reshape(3)
            if axis_pt_raw is not None and len(axis_pt_raw) == 3
            else np.zeros(0, dtype=np.float32)
        )
        # Per-corner trajectories (NaN = invalid/missing sample)
        tracks_raw = payload.get("corner_tracks_world")
        if tracks_raw is not None and isinstance(tracks_raw, list) and len(tracks_raw) > 0:
            tracks_arr = np.asarray(tracks_raw, dtype=np.float32)
            if tracks_arr.ndim == 3 and tracks_arr.shape[2] == 3:
                valid_arr = np.all(np.isfinite(tracks_arr), axis=2)
            else:
                tracks_arr = np.zeros((0, 0, 3), dtype=np.float32)
                valid_arr = np.zeros((0, 0), dtype=bool)
        else:
            tracks_arr = np.zeros((0, 0, 3), dtype=np.float32)
            valid_arr = np.zeros((0, 0), dtype=bool)
        return cls(
            segment_id=str(payload.get("segment_id", str(uuid.uuid4()))),
            clip_id=int(payload.get("clip_id", -1)),
            frame_start=int(payload.get("frame_start", -1)),
            frame_end=int(payload.get("frame_end", -1)),
            reference_pose=_as_pose_matrix(payload.get("reference_pose")),
            reference_points=_empty_points(),
            reference_colors=_empty_points(),
            prior_source=str(payload.get("prior_source", "")),
            reference_pointcloud_file=str(payload.get("reference_pointcloud_file", "")),
            pose_history=[MotionPoseSample.from_dict(item) for item in payload.get("pose_history", [])],
            articulation_joint_type=str(payload.get("articulation_joint_type", "")),
            articulation_axis_direction=axis_dir,
            articulation_axis_point=axis_pt,
            corner_tracks_world=tracks_arr,
            corner_tracks_valid=valid_arr,
            corner_track_frames=[int(f) for f in payload.get("corner_track_frames", [])],
            articulation_axis_candidates=list(payload.get("articulation_axis_candidates", []) or []),
        )


@dataclass
class GraphNode:
    """One functional graph node representing an interactive element or object."""

    node_id: str
    label: str
    node_type: str
    centroid: np.ndarray
    bbox_min: np.ndarray
    bbox_max: np.ndarray
    points: np.ndarray
    colors: np.ndarray
    state: str = "unknown"
    description: str = ""
    confidence: float = 1.0
    clip_id: int = -1
    frame_start: int = -1
    frame_end: int = -1
    pointcloud_file: str = ""
    state_history: List[dict] = field(default_factory=list)
    motion_segments: List[NodeMotionSegment] = field(default_factory=list)

    def update_state(self, new_state: str, clip_id: int, reasoning: str = "") -> None:
        """Record a state transition on the node."""
        if new_state == self.state:
            return
        self.state_history.append(
            {
                "from": self.state,
                "to": new_state,
                "clip_id": clip_id,
                "reasoning": reasoning,
            }
        )
        self.state = new_state

    def set_points(
        self,
        new_pts: np.ndarray,
        new_colors: Optional[np.ndarray] = None,
        voxel_size: float = 0.02,
    ) -> None:
        """Replace the node geometry with a new point cloud and update centroid and bounds."""
        pts = np.asarray(new_pts, dtype=np.float32) if new_pts is not None else _empty_points()
        if len(pts) == 0:
            self.points = _empty_points()
            self.colors = _empty_points()
            self.centroid = np.zeros(3, dtype=np.float32)
            self.bbox_min = np.zeros(3, dtype=np.float32)
            self.bbox_max = np.zeros(3, dtype=np.float32)
            return

        if new_colors is not None and len(new_colors) == len(pts):
            cols = np.asarray(new_colors, dtype=np.float32)
        else:
            cols = np.full((len(pts), 3), 0.5, dtype=np.float32)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        pcd.colors = o3d.utility.Vector3dVector(cols)
        pcd = pcd.voxel_down_sample(voxel_size)

        self.points = np.asarray(pcd.points, dtype=np.float32)
        self.colors = np.asarray(pcd.colors, dtype=np.float32) if pcd.has_colors() else _empty_points()
        if len(self.points):
            self.centroid = self.points.mean(axis=0)
            self.bbox_min = self.points.min(axis=0)
            self.bbox_max = self.points.max(axis=0)
        else:
            self.centroid = np.zeros(3, dtype=np.float32)
            self.bbox_min = np.zeros(3, dtype=np.float32)
            self.bbox_max = np.zeros(3, dtype=np.float32)

    def append_points(
        self,
        new_pts: np.ndarray,
        new_colors: Optional[np.ndarray] = None,
        voxel_size: float = 0.02,
    ) -> None:
        """Merge new geometry into the node and update centroid and bounds."""
        if new_pts is None or len(new_pts) == 0:
            return

        all_pts = np.concatenate([self.points, np.asarray(new_pts, dtype=np.float32)], axis=0)
        if new_colors is not None and len(new_colors) == len(new_pts):
            all_cols = np.concatenate([self.colors, np.asarray(new_colors, dtype=np.float32)], axis=0)
        else:
            fill = np.full((len(new_pts), 3), 0.5, dtype=np.float32)
            all_cols = np.concatenate([self.colors, fill], axis=0)

        self.set_points(all_pts, all_cols, voxel_size)

    def to_dict(self) -> dict:
        """Serialize node metadata for JSON persistence."""
        return {
            "node_id": self.node_id,
            "label": self.label,
            "node_type": self.node_type,
            "centroid": self.centroid.tolist(),
            "bbox_min": self.bbox_min.tolist(),
            "bbox_max": self.bbox_max.tolist(),
            "n_points": int(len(self.points)),
            "state": self.state,
            "description": self.description,
            "confidence": float(self.confidence),
            "clip_id": int(self.clip_id),
            "frame_start": int(self.frame_start),
            "frame_end": int(self.frame_end),
            "pointcloud_file": self.pointcloud_file,
            "state_history": self.state_history,
            "motion_segments": [segment.to_dict() for segment in self.motion_segments],
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "GraphNode":
        """Create a node from serialized JSON metadata."""
        return cls(
            node_id=payload["node_id"],
            label=payload["label"],
            node_type=payload["node_type"],
            centroid=np.asarray(payload.get("centroid", [0.0, 0.0, 0.0]), dtype=np.float32),
            bbox_min=np.asarray(payload.get("bbox_min", [0.0, 0.0, 0.0]), dtype=np.float32),
            bbox_max=np.asarray(payload.get("bbox_max", [0.0, 0.0, 0.0]), dtype=np.float32),
            points=_empty_points(),
            colors=_empty_points(),
            state=payload.get("state", "unknown"),
            description=payload.get("description", ""),
            confidence=float(payload.get("confidence", 1.0)),
            clip_id=int(payload.get("clip_id", -1)),
            frame_start=int(payload.get("frame_start", -1)),
            frame_end=int(payload.get("frame_end", -1)),
            pointcloud_file=payload.get("pointcloud_file", ""),
            state_history=payload.get("state_history", []),
            motion_segments=[NodeMotionSegment.from_dict(item) for item in payload.get("motion_segments", [])],
        )


@dataclass
class GraphEdge:
    """Directed functional relationship between two nodes."""

    edge_id: str
    src_id: str
    dst_id: str
    edge_type: str
    relationship: str
    confidence: float = 1.0
    clip_id: int = -1
    cot_reasoning: str = ""

    def to_dict(self) -> dict:
        """Serialize edge metadata for JSON persistence."""
        return {
            "edge_id": self.edge_id,
            "src_id": self.src_id,
            "dst_id": self.dst_id,
            "edge_type": self.edge_type,
            "relationship": self.relationship,
            "confidence": float(self.confidence),
            "clip_id": int(self.clip_id),
            "cot_reasoning": self.cot_reasoning,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "GraphEdge":
        """Create an edge from serialized JSON metadata."""
        return cls(**payload)


class DynamicSceneGraph:
    """Dynamic functional graph that grows only when interactions are observed."""

    def __init__(self) -> None:
        self.nodes: Dict[str, GraphNode] = {}
        self.edges: Dict[str, GraphEdge] = {}
        self._label_index: Dict[str, List[str]] = defaultdict(list)

    def add_node(
        self,
        label: str,
        node_type: str,
        points: np.ndarray,
        colors: Optional[np.ndarray] = None,
        state: str = "unknown",
        description: str = "",
        confidence: float = 1.0,
        clip_id: int = -1,
        frame_start: int = -1,
        frame_end: int = -1,
        voxel_size: float = 0.02,
    ) -> str:
        """Create a node, voxelize its geometry, and return the node id."""
        label = _normalize_label(label)
        pts = np.asarray(points, dtype=np.float32) if points is not None else _empty_points()
        cols = np.asarray(colors, dtype=np.float32) if colors is not None else _empty_points()

        if len(pts):
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pts)
            if len(cols) == len(pts):
                pcd.colors = o3d.utility.Vector3dVector(cols)
            pcd = pcd.voxel_down_sample(voxel_size)
            pts = np.asarray(pcd.points, dtype=np.float32)
            cols = np.asarray(pcd.colors, dtype=np.float32) if pcd.has_colors() else np.full(
                (len(pts), 3), 0.5, dtype=np.float32
            )
            centroid = pts.mean(axis=0)
            bbox_min = pts.min(axis=0)
            bbox_max = pts.max(axis=0)
        else:
            pts = _empty_points()
            cols = _empty_points()
            centroid = np.zeros(3, dtype=np.float32)
            bbox_min = np.zeros(3, dtype=np.float32)
            bbox_max = np.zeros(3, dtype=np.float32)

        node_id = str(uuid.uuid4())
        self.nodes[node_id] = GraphNode(
            node_id=node_id,
            label=label,
            node_type=node_type,
            centroid=centroid,
            bbox_min=bbox_min,
            bbox_max=bbox_max,
            points=pts,
            colors=cols,
            state=state,
            description=description,
            confidence=confidence,
            clip_id=clip_id,
            frame_start=frame_start,
            frame_end=frame_end,
        )
        self._label_index[label].append(node_id)
        return node_id

    def find_node_by_label(self, label: str) -> Optional[GraphNode]:
        """Return the most recent node with the given normalized label."""
        ids = self._label_index.get(_normalize_label(label), [])
        return self.nodes[ids[-1]] if ids else None

    def match_node_by_geometry(
        self,
        label: str,
        node_type: Optional[str],
        points: Optional[np.ndarray],
        centroid_hint: Optional[np.ndarray] = None,
        max_centroid_distance_m: float = NODE_MATCH_MAX_CENTROID_DISTANCE_M,
        min_overlap_ratio: float = NODE_MATCH_MIN_OVERLAP_RATIO,
        max_median_nn_distance_m: float = NODE_MATCH_MAX_MEDIAN_NN_DISTANCE_M,
    ) -> Tuple[Optional[GraphNode], dict]:
        """Return the best existing node for a label if it is also geometrically consistent."""
        query_points = np.asarray(points, dtype=np.float32) if points is not None else _empty_points()
        if len(query_points):
            query_centroid = query_points.mean(axis=0).astype(np.float32)
        elif centroid_hint is not None and np.asarray(centroid_hint).size == 3:
            query_centroid = np.asarray(centroid_hint, dtype=np.float32).reshape(3)
        else:
            query_centroid = None

        candidates: List[dict] = []
        best_payload: Optional[dict] = None
        for node in self.nodes.values():
            if node_type and str(node.node_type) != str(node_type):
                continue
            if not _labels_are_compatible(label, node.label):
                continue
            centroid_distance = float("inf")
            if query_centroid is not None and np.asarray(node.centroid).size == 3:
                centroid_distance = float(np.linalg.norm(np.asarray(node.centroid, dtype=np.float32) - query_centroid))
            overlap_metrics = _point_cloud_overlap_metrics(query_points, node.points)
            geometry_available = bool(len(query_points) and len(node.points))
            if geometry_available:
                spatially_consistent = (
                    centroid_distance <= float(max_centroid_distance_m)
                    and (
                        overlap_metrics["overlap_ratio"] >= float(min_overlap_ratio)
                        or overlap_metrics["median_nn_distance_m"] <= float(max_median_nn_distance_m)
                    )
                )
            elif query_centroid is not None:
                spatially_consistent = centroid_distance <= float(max_centroid_distance_m)
            else:
                spatially_consistent = _normalize_label(node.label) == _normalize_label(label)
            candidate_payload = {
                "node_id": node.node_id,
                "label": node.label,
                "clip_id": int(node.clip_id),
                "frame_end": int(node.frame_end),
                "centroid_distance_m": None if not np.isfinite(centroid_distance) else round(float(centroid_distance), 4),
                "overlap_ratio": round(float(overlap_metrics["overlap_ratio"]), 4),
                "median_nn_distance_m": None if not np.isfinite(overlap_metrics["median_nn_distance_m"]) else round(float(overlap_metrics["median_nn_distance_m"]), 4),
                "p90_nn_distance_m": None if not np.isfinite(overlap_metrics["p90_nn_distance_m"]) else round(float(overlap_metrics["p90_nn_distance_m"]), 4),
                "geometry_available": bool(geometry_available),
                "spatially_consistent": bool(spatially_consistent),
            }
            candidates.append(candidate_payload)
            if not spatially_consistent:
                continue
            ranking_key = (
                float(overlap_metrics["overlap_ratio"]),
                -float(overlap_metrics["median_nn_distance_m"]),
                -float(centroid_distance),
                int(node.clip_id),
                int(node.frame_end),
            )
            if best_payload is None or ranking_key > best_payload["ranking_key"]:
                best_payload = {
                    "node": node,
                    "ranking_key": ranking_key,
                    "centroid_distance_m": centroid_distance,
                    "overlap_ratio": float(overlap_metrics["overlap_ratio"]),
                    "median_nn_distance_m": float(overlap_metrics["median_nn_distance_m"]),
                    "p90_nn_distance_m": float(overlap_metrics["p90_nn_distance_m"]),
                }

        candidates.sort(
            key=lambda item: (
                int(bool(item["spatially_consistent"])),
                float(item["overlap_ratio"]),
                -float(item["median_nn_distance_m"] if item["median_nn_distance_m"] is not None else 1e9),
                -float(item["centroid_distance_m"] if item["centroid_distance_m"] is not None else 1e9),
                int(item["clip_id"]),
                int(item["frame_end"]),
            ),
            reverse=True,
        )
        debug = {
            "query_label": _normalize_label(label),
            "node_type": str(node_type or ""),
            "query_point_count": int(len(query_points)),
            "matched": bool(best_payload is not None),
            "selected_node_id": best_payload["node"].node_id if best_payload is not None else None,
            "selected_label": best_payload["node"].label if best_payload is not None else None,
            "selected_clip_id": int(best_payload["node"].clip_id) if best_payload is not None else None,
            "selected_centroid_distance_m": round(float(best_payload["centroid_distance_m"]), 4) if best_payload is not None and np.isfinite(best_payload["centroid_distance_m"]) else None,
            "selected_overlap_ratio": round(float(best_payload["overlap_ratio"]), 4) if best_payload is not None else None,
            "selected_median_nn_distance_m": round(float(best_payload["median_nn_distance_m"]), 4) if best_payload is not None and np.isfinite(best_payload["median_nn_distance_m"]) else None,
            "candidate_count": int(len(candidates)),
            "candidates": candidates[:8],
        }
        return (best_payload["node"] if best_payload is not None else None), debug

    def set_node_geometry(
        self,
        node_id: str,
        new_pts: np.ndarray,
        new_colors: Optional[np.ndarray] = None,
        voxel_size: float = 0.02,
    ) -> None:
        """Replace geometry on an existing node."""
        if node_id in self.nodes:
            self.nodes[node_id].set_points(new_pts, new_colors, voxel_size)

    def append_points_to_node(
        self,
        node_id: str,
        new_pts: np.ndarray,
        new_colors: Optional[np.ndarray] = None,
        voxel_size: float = 0.02,
    ) -> None:
        """Append geometry to an existing node."""
        if node_id in self.nodes:
            self.nodes[node_id].append_points(new_pts, new_colors, voxel_size)

    def update_node_state(self, node_id: str, new_state: str, clip_id: int = -1, reasoning: str = "") -> None:
        """Update the semantic state of an existing node."""
        if node_id in self.nodes and new_state:
            self.nodes[node_id].update_state(new_state, clip_id, reasoning)

    def record_node_motion(
        self,
        node_id: str,
        clip_id: int,
        frame_start: int,
        frame_end: int,
        reference_points: np.ndarray,
        reference_colors: Optional[np.ndarray],
        reference_pose: np.ndarray,
        pose_history: Sequence[dict | MotionPoseSample],
        prior_source: str = "",
        articulation_joint_type: str = "",
        articulation_axis_direction: Optional[np.ndarray] = None,
        articulation_axis_point: Optional[np.ndarray] = None,
        corner_tracks_world: Optional[np.ndarray] = None,
        corner_tracks_valid: Optional[np.ndarray] = None,
        corner_track_frames: Optional[Sequence[int]] = None,
        articulation_axis_candidates: Optional[Sequence[dict]] = None,
    ) -> Optional[str]:
        """Store an exact-frame transform history segment for one node."""
        node = self.nodes.get(node_id)
        if node is None:
            return None

        ref_points = np.asarray(reference_points, dtype=np.float32) if reference_points is not None else _empty_points()
        if len(ref_points) == 0:
            return None
        if reference_colors is not None and len(reference_colors) == len(ref_points):
            ref_colors = np.asarray(reference_colors, dtype=np.float32)
        else:
            ref_colors = np.full((len(ref_points), 3), 0.5, dtype=np.float32)

        normalized_samples: Dict[int, MotionPoseSample] = {}
        for item in pose_history:
            if isinstance(item, MotionPoseSample):
                sample = item
            else:
                sample = MotionPoseSample.from_dict(item)
            if sample.frame < 0:
                continue
            normalized_samples[int(sample.frame)] = MotionPoseSample(
                frame=int(sample.frame),
                pose=_as_pose_matrix(sample.pose),
                source=sample.source,
                confidence=float(sample.confidence),
                clip_id=int(sample.clip_id if sample.clip_id >= 0 else clip_id),
            )
        if not normalized_samples:
            return None

        axis_dir_arr = (
            np.asarray(articulation_axis_direction, dtype=np.float32).reshape(3)
            if articulation_axis_direction is not None and np.asarray(articulation_axis_direction).size == 3
            else np.zeros(0, dtype=np.float32)
        )
        axis_pt_arr = (
            np.asarray(articulation_axis_point, dtype=np.float32).reshape(3)
            if articulation_axis_point is not None and np.asarray(articulation_axis_point).size == 3
            else np.zeros(0, dtype=np.float32)
        )
        if corner_tracks_world is not None and np.asarray(corner_tracks_world).size > 0:
            tracks_arr = np.asarray(corner_tracks_world, dtype=np.float32)
        else:
            tracks_arr = np.zeros((0, 0, 3), dtype=np.float32)
        if corner_tracks_valid is not None and np.asarray(corner_tracks_valid).size > 0:
            valid_arr = np.asarray(corner_tracks_valid, dtype=bool)
        elif tracks_arr.size > 0:
            valid_arr = np.all(np.isfinite(tracks_arr), axis=2)
        else:
            valid_arr = np.zeros((0, 0), dtype=bool)
        frames_list = [int(f) for f in (corner_track_frames or [])]
        segment = NodeMotionSegment(
            segment_id=str(uuid.uuid4()),
            clip_id=int(clip_id),
            frame_start=int(frame_start),
            frame_end=int(frame_end),
            reference_pose=_as_pose_matrix(reference_pose),
            reference_points=ref_points,
            reference_colors=ref_colors,
            prior_source=str(prior_source or ""),
            pose_history=[normalized_samples[frame] for frame in sorted(normalized_samples)],
            articulation_joint_type=str(articulation_joint_type or ""),
            articulation_axis_direction=axis_dir_arr,
            articulation_axis_point=axis_pt_arr,
            corner_tracks_world=tracks_arr,
            corner_tracks_valid=valid_arr,
            corner_track_frames=frames_list,
            articulation_axis_candidates=list(articulation_axis_candidates or []),
        )
        node.motion_segments = [existing for existing in node.motion_segments if int(existing.clip_id) != int(clip_id)]
        node.motion_segments.append(segment)
        node.motion_segments.sort(key=lambda item: (int(item.frame_start), int(item.clip_id), item.segment_id))
        return segment.segment_id

    def add_edge(
        self,
        src_id: str,
        dst_id: str,
        edge_type: str,
        relationship: str,
        confidence: float = 1.0,
        clip_id: int = -1,
        cot_reasoning: str = "",
    ) -> str:
        """Create a new directed relationship edge."""
        edge_id = str(uuid.uuid4())
        self.edges[edge_id] = GraphEdge(
            edge_id=edge_id,
            src_id=src_id,
            dst_id=dst_id,
            edge_type=edge_type,
            relationship=relationship,
            confidence=confidence,
            clip_id=clip_id,
            cot_reasoning=cot_reasoning,
        )
        return edge_id

    def get_edges_from(self, node_id: str) -> List[GraphEdge]:
        """Return outgoing edges from a node."""
        return [edge for edge in self.edges.values() if edge.src_id == node_id]

    def get_edges_to(self, node_id: str) -> List[GraphEdge]:
        """Return incoming edges to a node."""
        return [edge for edge in self.edges.values() if edge.dst_id == node_id]

    def get_node_geometry_at_frame(
        self,
        node_id: str,
        frame: int,
        exact: bool = True,
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Render one node's geometry at a requested frame from stored motion history."""
        node = self.nodes.get(node_id)
        if node is None:
            return None
        candidates: List[Tuple[NodeMotionSegment, np.ndarray]] = []
        for segment in node.motion_segments:
            pose = segment.get_pose_at_frame(frame, exact=exact)
            if pose is not None:
                candidates.append((segment, pose))
        if not candidates:
            return None
        segment, _pose = max(candidates, key=lambda item: (int(item[0].clip_id), int(item[0].frame_end)))
        return segment.render_geometry_at_frame(frame, exact=exact)

    def get_node_geometry_by_label_at_frame(
        self,
        label: str,
        frame: int,
        exact: bool = True,
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Render the latest node matching a label at an exact frame."""
        node = self.find_node_by_label(label)
        if node is None:
            return None
        return self.get_node_geometry_at_frame(node.node_id, frame, exact=exact)

    def get_exact_motion_frames(self, node_id: str) -> List[int]:
        """Return the exact frames for which a node has stored motion samples."""
        node = self.nodes.get(node_id)
        if node is None:
            return []
        frames = sorted({int(sample.frame) for segment in node.motion_segments for sample in segment.pose_history})
        return frames

    def render_nodes_at_frame(
        self,
        frame: int,
        exact: bool = True,
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Render all nodes that have an exact pose sample at a requested frame."""
        rendered: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        for node_id in self.nodes:
            geometry = self.get_node_geometry_at_frame(node_id, frame, exact=exact)
            if geometry is not None:
                rendered[node_id] = geometry
        return rendered

    def export_node_motion_sequence(
        self,
        node_id: str,
        out_dir: str | Path,
        exact: bool = True,
    ) -> List[str]:
        """Write one point cloud per stored motion frame for a node and return the file paths."""
        node = self.nodes.get(node_id)
        if node is None:
            return []
        output_dir = Path(out_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        written_paths: List[str] = []
        for frame in self.get_exact_motion_frames(node_id):
            geometry = self.get_node_geometry_at_frame(node_id, frame, exact=exact)
            if geometry is None:
                continue
            points, colors = geometry
            if len(points) == 0:
                continue
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            if len(colors) == len(points):
                pcd.colors = o3d.utility.Vector3dVector(colors)
            output_path = output_dir / f"{node.label.replace(' ', '_')}_frame_{int(frame):06d}.ply"
            o3d.io.write_point_cloud(str(output_path), pcd)
            written_paths.append(str(output_path))
        return written_paths

    def register_interaction(
        self,
        element_label: str,
        element_pts: np.ndarray,
        element_colors: Optional[np.ndarray],
        object_label: str,
        object_pts: np.ndarray,
        object_colors: Optional[np.ndarray],
        relationship: str,
        edge_type: str = "local",
        confidence: float = 1.0,
        clip_id: int = -1,
        frame_start: int = -1,
        frame_end: int = -1,
        cot_reasoning: str = "",
        element_state: str = "unknown",
        object_state: str = "unknown",
        element_state_reasoning: str = "",
        object_state_reasoning: str = "",
        element_desc: str = "",
        object_desc: str = "",
        voxel_size: float = 0.02,
        replace_element_geometry: bool = False,
        update_element_geometry: bool = True,
        element_centroid_hint: Optional[np.ndarray] = None,
        record_element_motion: bool = True,
        replace_object_geometry: bool = False,
        element_motion: Optional[dict] = None,
        object_motion: Optional[dict] = None,
        create_edge: bool = True,
        element_node_id: Optional[str] = None,
        object_node_id: Optional[str] = None,
        use_label_lookup_for_element: bool = True,
        use_label_lookup_for_object: bool = True,
    ) -> Tuple[str, str, str]:
        """Create or update the element/object nodes and optionally their edge."""
        element_label = _normalize_label(element_label)
        object_label = _normalize_label(object_label)

        elem_node = self.nodes.get(str(element_node_id)) if element_node_id is not None else None
        if elem_node is None and use_label_lookup_for_element:
            elem_node = self.find_node_by_label(element_label)
        semantic_element_points = np.asarray(element_pts, dtype=np.float32) if update_element_geometry else _empty_points()
        semantic_element_colors = np.asarray(element_colors, dtype=np.float32) if (update_element_geometry and element_colors is not None) else _empty_points()
        centroid_hint = np.asarray(element_centroid_hint, dtype=np.float32) if element_centroid_hint is not None else None
        if elem_node is None:
            elem_id = self.add_node(
                label=element_label,
                node_type="element",
                points=semantic_element_points,
                colors=semantic_element_colors,
                state=element_state,
                description=element_desc,
                confidence=confidence,
                clip_id=clip_id,
                frame_start=frame_start,
                frame_end=frame_end,
                voxel_size=voxel_size,
            )
            if not update_element_geometry and centroid_hint is not None:
                self.nodes[elem_id].centroid = centroid_hint
                self.nodes[elem_id].bbox_min = centroid_hint.copy()
                self.nodes[elem_id].bbox_max = centroid_hint.copy()
        else:
            elem_id = elem_node.node_id
            if update_element_geometry:
                if replace_element_geometry:
                    self.set_node_geometry(elem_id, element_pts, element_colors, voxel_size)
                else:
                    self.append_points_to_node(elem_id, element_pts, element_colors, voxel_size)
            else:
                self.set_node_geometry(elem_id, _empty_points(), _empty_points(), voxel_size)
                if centroid_hint is not None:
                    self.nodes[elem_id].centroid = centroid_hint
                    self.nodes[elem_id].bbox_min = centroid_hint.copy()
                    self.nodes[elem_id].bbox_max = centroid_hint.copy()
            if element_desc:
                self.nodes[elem_id].description = element_desc
            self.nodes[elem_id].confidence = max(self.nodes[elem_id].confidence, confidence)
            self.nodes[elem_id].clip_id = clip_id
            self.nodes[elem_id].frame_start = frame_start
            self.nodes[elem_id].frame_end = frame_end
            if element_state != "unknown":
                self.update_node_state(elem_id, element_state, clip_id, element_state_reasoning or cot_reasoning)
        if element_motion and record_element_motion:
            self.record_node_motion(
                elem_id,
                clip_id=clip_id,
                frame_start=frame_start,
                frame_end=frame_end,
                reference_points=element_motion.get("reference_points"),
                reference_colors=element_motion.get("reference_colors"),
                reference_pose=element_motion.get("reference_pose"),
                pose_history=element_motion.get("pose_history", []),
                prior_source=element_motion.get("prior_source", ""),
                articulation_joint_type=str(element_motion.get("articulation_joint_type", "")),
                articulation_axis_direction=element_motion.get("articulation_axis_direction"),
                articulation_axis_point=element_motion.get("articulation_axis_point"),
                corner_tracks_world=element_motion.get("corner_tracks_world"),
                corner_tracks_valid=element_motion.get("corner_tracks_valid"),
                corner_track_frames=element_motion.get("corner_track_frames"),
                articulation_axis_candidates=element_motion.get("articulation_axis_candidates"),
            )
        elif not record_element_motion and elem_id in self.nodes:
            self.nodes[elem_id].motion_segments = [
                segment for segment in self.nodes[elem_id].motion_segments if int(segment.clip_id) != int(clip_id)
            ]

        obj_node = self.nodes.get(str(object_node_id)) if object_node_id is not None else None
        if obj_node is None and use_label_lookup_for_object:
            obj_node = self.find_node_by_label(object_label)
        if obj_node is None:
            obj_id = self.add_node(
                label=object_label,
                node_type="object",
                points=object_pts,
                colors=object_colors,
                state=object_state,
                description=object_desc,
                confidence=confidence,
                clip_id=clip_id,
                frame_start=frame_start,
                frame_end=frame_end,
                voxel_size=voxel_size,
            )
        else:
            obj_id = obj_node.node_id
            if replace_object_geometry:
                self.set_node_geometry(obj_id, object_pts, object_colors, voxel_size)
            else:
                self.append_points_to_node(obj_id, object_pts, object_colors, voxel_size)
            if object_desc:
                self.nodes[obj_id].description = object_desc
            self.nodes[obj_id].confidence = max(self.nodes[obj_id].confidence, confidence)
            self.nodes[obj_id].clip_id = clip_id
            self.nodes[obj_id].frame_start = frame_start
            self.nodes[obj_id].frame_end = frame_end
            if object_state != "unknown":
                self.update_node_state(obj_id, object_state, clip_id, object_state_reasoning or cot_reasoning)
        if object_motion:
            self.record_node_motion(
                obj_id,
                clip_id=clip_id,
                frame_start=frame_start,
                frame_end=frame_end,
                reference_points=object_motion.get("reference_points"),
                reference_colors=object_motion.get("reference_colors"),
                reference_pose=object_motion.get("reference_pose"),
                pose_history=object_motion.get("pose_history", []),
                prior_source=object_motion.get("prior_source", ""),
                articulation_joint_type=str(object_motion.get("articulation_joint_type", "")),
                articulation_axis_direction=object_motion.get("articulation_axis_direction"),
                articulation_axis_point=object_motion.get("articulation_axis_point"),
                corner_tracks_world=object_motion.get("corner_tracks_world"),
                corner_tracks_valid=object_motion.get("corner_tracks_valid"),
                corner_track_frames=object_motion.get("corner_track_frames"),
                articulation_axis_candidates=object_motion.get("articulation_axis_candidates"),
            )

        edge_id = ""
        if create_edge and relationship:
            existing = [
                edge
                for edge in self.edges.values()
                if edge.src_id == elem_id and edge.dst_id == obj_id and edge.relationship == relationship
            ]
            if existing:
                edge = existing[0]
                edge.confidence = max(edge.confidence, confidence)
                edge.clip_id = clip_id
                if cot_reasoning:
                    edge.cot_reasoning = cot_reasoning
                edge_id = edge.edge_id
            else:
                edge_id = self.add_edge(
                    src_id=elem_id,
                    dst_id=obj_id,
                    edge_type=edge_type,
                    relationship=relationship,
                    confidence=confidence,
                    clip_id=clip_id,
                    cot_reasoning=cot_reasoning,
                )
        return elem_id, obj_id, edge_id

    def save(self, path: str, save_pointclouds: bool = True) -> None:
        """Persist the graph to JSON and optionally save per-node point clouds."""
        root = Path(path)
        root.mkdir(parents=True, exist_ok=True)
        pcd_dir = root / "pointclouds"
        motion_pcd_dir = root / "motion_pointclouds"
        if save_pointclouds:
            pcd_dir.mkdir(exist_ok=True)
            motion_pcd_dir.mkdir(exist_ok=True)

        for node in self.nodes.values():
            if save_pointclouds and len(node.points):
                filename = f"{node.label.replace(' ', '_')}_{node.node_id[:8]}.ply"
                node.pointcloud_file = f"pointclouds/{filename}"
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(node.points)
                if len(node.colors) == len(node.points):
                    pcd.colors = o3d.utility.Vector3dVector(node.colors)
                o3d.io.write_point_cloud(str(root / node.pointcloud_file), pcd)
            for segment in node.motion_segments:
                if save_pointclouds and len(segment.reference_points):
                    filename = (
                        f"{node.label.replace(' ', '_')}_{node.node_id[:8]}_clip{int(segment.clip_id):03d}_"
                        f"{segment.segment_id[:8]}.ply"
                    )
                    segment.reference_pointcloud_file = f"motion_pointclouds/{filename}"
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(segment.reference_points)
                    if len(segment.reference_colors) == len(segment.reference_points):
                        pcd.colors = o3d.utility.Vector3dVector(segment.reference_colors)
                    o3d.io.write_point_cloud(str(root / segment.reference_pointcloud_file), pcd)

        payload = {
            "nodes": {node_id: node.to_dict() for node_id, node in self.nodes.items()},
            "edges": {edge_id: edge.to_dict() for edge_id, edge in self.edges.items()},
        }
        with open(root / "graph.json", "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)

    @classmethod
    def load(cls, path: str) -> "DynamicSceneGraph":
        """Load a graph and any saved per-node point clouds from disk."""
        root = Path(path)
        with open(root / "graph.json", "r", encoding="utf-8") as handle:
            payload = json.load(handle)

        graph = cls()
        for node_id, node_payload in payload.get("nodes", {}).items():
            node = GraphNode.from_dict(node_payload)
            if node.pointcloud_file:
                pcd_path = root / node.pointcloud_file
                if pcd_path.exists():
                    pcd = o3d.io.read_point_cloud(str(pcd_path))
                    node.points = np.asarray(pcd.points, dtype=np.float32)
                    node.colors = np.asarray(pcd.colors, dtype=np.float32) if pcd.has_colors() else _empty_points()
            for segment in node.motion_segments:
                if segment.reference_pointcloud_file:
                    pcd_path = root / segment.reference_pointcloud_file
                    if pcd_path.exists():
                        pcd = o3d.io.read_point_cloud(str(pcd_path))
                        segment.reference_points = np.asarray(pcd.points, dtype=np.float32)
                        segment.reference_colors = np.asarray(pcd.colors, dtype=np.float32) if pcd.has_colors() else _empty_points()
            graph.nodes[node_id] = node
            graph._label_index[_normalize_label(node.label)].append(node_id)

        for edge_id, edge_payload in payload.get("edges", {}).items():
            graph.edges[edge_id] = GraphEdge.from_dict(edge_payload)
        return graph

    def to_json_for_llm(self) -> str:
        """Serialize the graph into a compact LLM-friendly JSON string."""
        nodes_payload = []
        for node_id, node in self.nodes.items():
            edges_payload = []
            for edge in self.get_edges_from(node_id):
                target = self.nodes.get(edge.dst_id)
                edges_payload.append(
                    {
                        "target_label": target.label if target else "unknown",
                        "relationship": edge.relationship,
                        "edge_type": edge.edge_type,
                        "confidence": round(float(edge.confidence), 3),
                    }
                )
            nodes_payload.append(
                {
                    "id": node.node_id,
                    "label": node.label,
                    "type": node.node_type,
                    "state": node.state,
                    "centroid": [round(float(v), 4) for v in node.centroid.tolist()],
                    "edges": edges_payload,
                }
            )
        return json.dumps(nodes_payload, separators=(",", ":"))

    def diff(self, other: "DynamicSceneGraph") -> dict:
        """Summarize how this snapshot differs from another graph snapshot."""
        current_ids = set(self.nodes)
        other_ids = set(other.nodes)
        new_nodes = [
            {"id": node_id, "label": self.nodes[node_id].label, "type": self.nodes[node_id].node_type}
            for node_id in sorted(current_ids - other_ids)
        ]
        removed_nodes = [
            {"id": node_id, "label": other.nodes[node_id].label, "type": other.nodes[node_id].node_type}
            for node_id in sorted(other_ids - current_ids)
        ]

        state_changes = []
        for node_id in sorted(current_ids & other_ids):
            new_node = self.nodes[node_id]
            old_node = other.nodes[node_id]
            if new_node.state != old_node.state:
                state_changes.append(
                    {
                        "id": node_id,
                        "label": new_node.label,
                        "from": old_node.state,
                        "to": new_node.state,
                    }
                )

        def _edge_signature(edge: GraphEdge) -> Tuple[str, str, str, str]:
            return (edge.src_id, edge.dst_id, edge.relationship, edge.edge_type)

        current_edge_map = {_edge_signature(edge): edge for edge in self.edges.values()}
        other_edge_map = {_edge_signature(edge): edge for edge in other.edges.values()}
        new_edges = []
        for signature in sorted(set(current_edge_map) - set(other_edge_map)):
            edge = current_edge_map[signature]
            src = self.nodes.get(edge.src_id)
            dst = self.nodes.get(edge.dst_id)
            new_edges.append(
                {
                    "src_label": src.label if src else edge.src_id,
                    "dst_label": dst.label if dst else edge.dst_id,
                    "relationship": edge.relationship,
                    "edge_type": edge.edge_type,
                    "confidence": round(float(edge.confidence), 3),
                }
            )

        return {
            "new_nodes": new_nodes,
            "removed_nodes": removed_nodes,
            "state_changes": state_changes,
            "new_edges": new_edges,
        }

    def summary(self) -> str:
        """Return a compact human-readable graph summary."""
        lines = [f"DynamicSceneGraph: {len(self.nodes)} nodes, {len(self.edges)} edges"]
        for edge in self.edges.values():
            src = self.nodes.get(edge.src_id)
            dst = self.nodes.get(edge.dst_id)
            src_name = src.label if src else edge.src_id
            dst_name = dst.label if dst else edge.dst_id
            lines.append(
                f"  '{src_name}' -[{edge.relationship} | {edge.edge_type} | conf={edge.confidence:.2f}]-> "
                f"'{dst_name}'"
            )
        return "\n".join(lines)

    def __repr__(self) -> str:
        """Return the human-readable summary representation."""
        return self.summary()

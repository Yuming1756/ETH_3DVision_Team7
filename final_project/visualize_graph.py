"""Standalone utilities for writing graph visualizations as PLY files."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import open3d as o3d

from dynamic_scene_graph import DynamicSceneGraph

try:
    import pyviz3d.visualizer as pyviz3d_viz
except ImportError:
    pyviz3d_viz = None

NODE_RADIUS = 0.04
EDGE_RADIUS = 0.012
NODE_COLORS = {
    "element": np.array([1.0, 0.0, 0.0], dtype=np.float64),
    "object": np.array([0.0, 0.35, 1.0], dtype=np.float64),
}
EDGE_COLORS = {
    "local": np.array([0.0, 1.0, 0.0], dtype=np.float64),
    "remote": np.array([1.0, 1.0, 0.0], dtype=np.float64),
    "functional": np.array([1.0, 0.45, 0.0], dtype=np.float64),
}
OPENFUNGRAPH_EDGE_COLOR = np.array([100.0, 100.0, 100.0], dtype=np.float32)
GRAPH_BACKGROUND_COLOR = np.array([0.30, 0.31, 0.35], dtype=np.float32)


def _stable_instance_color(key: str) -> np.ndarray:
    """Return a deterministic bright RGB color in [0, 1] for one node key."""
    digest = hashlib.md5(key.encode("utf-8")).digest()
    base = np.frombuffer(digest[:3], dtype=np.uint8).astype(np.float32) / 255.0
    # Keep the colors bright enough to match OpenFunGraph's instance markers.
    return 0.35 + 0.65 * base


def _scene_arrays(
    global_pcd_pts: Sequence[np.ndarray] | np.ndarray | None,
    global_pcd_cols: Sequence[np.ndarray] | np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert scene point storage into dense arrays."""
    if isinstance(global_pcd_pts, np.ndarray):
        scene_pts = global_pcd_pts
    elif global_pcd_pts:
        scene_pts = np.concatenate(list(global_pcd_pts), axis=0)
    else:
        scene_pts = np.zeros((0, 3), dtype=np.float32)

    if isinstance(global_pcd_cols, np.ndarray):
        scene_cols = global_pcd_cols
    elif global_pcd_cols:
        scene_cols = np.concatenate(list(global_pcd_cols), axis=0)
    else:
        scene_cols = np.zeros((0, 3), dtype=np.float32)

    return scene_pts, scene_cols


def _prepare_viz_points(
    points: np.ndarray,
    colors: np.ndarray | None = None,
    voxel_size: float = 0.01,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Downsample one point set and estimate normals for pyviz3d."""
    if points is None or len(points) == 0:
        return (
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0, 3), dtype=np.float32),
        )

    pcd = o3d.geometry.PointCloud()
    pts = np.asarray(points, dtype=np.float32)
    pcd.points = o3d.utility.Vector3dVector(pts)
    if colors is not None and len(colors) == len(points):
        cols = np.asarray(colors, dtype=np.float32)
    else:
        cols = np.full((len(pts), 3), 0.6, dtype=np.float32)
    pcd.colors = o3d.utility.Vector3dVector(cols)
    if voxel_size > 0.0:
        pcd = pcd.voxel_down_sample(voxel_size)
    if len(pcd.points):
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
        )
    out_pts = np.asarray(pcd.points, dtype=np.float32)
    out_cols = np.asarray(pcd.colors, dtype=np.float32) if pcd.has_colors() else np.full(
        (len(out_pts), 3), 0.6, dtype=np.float32
    )
    out_normals = np.asarray(pcd.normals, dtype=np.float32) if pcd.has_normals() else np.zeros_like(out_pts)
    return out_pts, out_cols, out_normals


def _rotation_from_z(direction: np.ndarray) -> np.ndarray:
    """Return a Rodrigues rotation that maps the z-axis onto a direction."""
    direction = np.asarray(direction, dtype=np.float64)
    norm = np.linalg.norm(direction)
    if norm < 1e-8:
        return np.eye(3, dtype=np.float64)
    direction = direction / norm
    z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    cross = np.cross(z_axis, direction)
    sin_theta = np.linalg.norm(cross)
    cos_theta = float(np.clip(np.dot(z_axis, direction), -1.0, 1.0))
    if sin_theta < 1e-8:
        if cos_theta > 0.0:
            return np.eye(3, dtype=np.float64)
        return np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]], dtype=np.float64)
    axis = cross / sin_theta
    kx, ky, kz = axis
    K = np.array([[0.0, -kz, ky], [kz, 0.0, -kx], [-ky, kx, 0.0]], dtype=np.float64)
    return np.eye(3, dtype=np.float64) + sin_theta * K + (1.0 - cos_theta) * (K @ K)


def _combine_meshes(meshes: Iterable[o3d.geometry.TriangleMesh]) -> o3d.geometry.TriangleMesh:
    """Merge multiple triangle meshes into one mesh object."""
    combined = o3d.geometry.TriangleMesh()
    for mesh in meshes:
        if mesh is not None and len(mesh.vertices):
            combined += mesh
    if len(combined.vertices):
        combined.compute_vertex_normals()
    return combined


def _scene_mesh_from_points(points: np.ndarray, colors: np.ndarray) -> o3d.geometry.TriangleMesh:
    """Encode point vertices into a mesh so graph primitives and scene can be saved together."""
    mesh = o3d.geometry.TriangleMesh()
    if points is None or len(points) == 0:
        return mesh
    mesh.vertices = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
    if colors is not None and len(colors) == len(points):
        mesh.vertex_colors = o3d.utility.Vector3dVector(np.asarray(colors, dtype=np.float64))
    return mesh


def _node_display_color(node_id: str, node_type: str, label: str) -> np.ndarray:
    """Return one stable bright color for a node's semantic geometry."""
    base = _stable_instance_color(f"{node_type}:{label}:{node_id}")
    if node_type == "element":
        tint = np.array([1.0, 0.30, 0.18], dtype=np.float32)
        return np.clip(0.60 * tint + 0.40 * base, 0.0, 1.0)
    tint = np.array([0.12, 0.46, 1.0], dtype=np.float32)
    return np.clip(0.35 * tint + 0.65 * base, 0.0, 1.0)


def _node_geometry_for_frame(
    graph: DynamicSceneGraph,
    node_id: str,
    frame: int | None = None,
    exact: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Return node geometry at one display frame, or the current node geometry."""
    node = graph.nodes[node_id]
    if frame is not None:
        rendered = graph.get_node_geometry_at_frame(node_id, frame, exact=exact)
        if rendered is not None and len(rendered[0]):
            return rendered
    return np.asarray(node.points, dtype=np.float32), np.asarray(node.colors, dtype=np.float32)


def _node_centroid_for_frame(
    graph: DynamicSceneGraph,
    node_id: str,
    frame: int | None = None,
    exact: bool = False,
) -> np.ndarray:
    """Return one node centroid at a display frame, or the current centroid."""
    pts, _cols = _node_geometry_for_frame(graph, node_id, frame=frame, exact=exact)
    if len(pts):
        return np.mean(np.asarray(pts, dtype=np.float64), axis=0)
    node = graph.nodes[node_id]
    return np.asarray(node.centroid, dtype=np.float64)


def _semantic_scene_arrays(
    graph: DynamicSceneGraph,
    scene_points: np.ndarray,
    frame: int | None = None,
    exact: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a graph-focused scene with muted background and semantic node colors."""
    scene_pts = np.asarray(scene_points, dtype=np.float32)
    scene_cols = (
        np.repeat(GRAPH_BACKGROUND_COLOR.reshape(1, 3), len(scene_pts), axis=0)
        if len(scene_pts)
        else np.zeros((0, 3), dtype=np.float32)
    )

    node_pts_blocks: list[np.ndarray] = []
    node_col_blocks: list[np.ndarray] = []
    for node_id, node in graph.nodes.items():
        pts, _cols = _node_geometry_for_frame(graph, node_id, frame=frame, exact=exact)
        if len(pts) == 0:
            continue
        node_pts_blocks.append(np.asarray(pts, dtype=np.float32))
        color = _node_display_color(node.node_id, node.node_type, node.label)
        node_col_blocks.append(np.repeat(color.reshape(1, 3).astype(np.float32), len(pts), axis=0))

    if node_pts_blocks:
        node_pts = np.concatenate(node_pts_blocks, axis=0)
        node_cols = np.concatenate(node_col_blocks, axis=0)
        if len(scene_pts):
            scene_pts = np.concatenate([scene_pts, node_pts], axis=0)
            scene_cols = np.concatenate([scene_cols, node_cols], axis=0)
        else:
            scene_pts, scene_cols = node_pts, node_cols
    return scene_pts, scene_cols


def _make_node_mesh(
    graph: DynamicSceneGraph,
    frame: int | None = None,
    exact: bool = False,
) -> o3d.geometry.TriangleMesh:
    """Create a colored sphere mesh for each graph node."""
    meshes = []
    for node in graph.nodes.values():
        mesh = o3d.geometry.TriangleMesh.create_sphere(radius=NODE_RADIUS)
        mesh.paint_uniform_color(NODE_COLORS.get(node.node_type, NODE_COLORS["object"]))
        mesh.translate(_node_centroid_for_frame(graph, node.node_id, frame=frame, exact=exact))
        meshes.append(mesh)
    return _combine_meshes(meshes)


def _make_edge_mesh(
    graph: DynamicSceneGraph,
    frame: int | None = None,
    exact: bool = False,
) -> o3d.geometry.TriangleMesh:
    """Create a colored cylinder mesh for each graph edge."""
    meshes = []
    for edge in graph.edges.values():
        src = graph.nodes.get(edge.src_id)
        dst = graph.nodes.get(edge.dst_id)
        if src is None or dst is None:
            continue
        start = _node_centroid_for_frame(graph, src.node_id, frame=frame, exact=exact)
        end = _node_centroid_for_frame(graph, dst.node_id, frame=frame, exact=exact)
        direction = end - start
        length = float(np.linalg.norm(direction))
        if length < 1e-8:
            continue
        mesh = o3d.geometry.TriangleMesh.create_cylinder(radius=EDGE_RADIUS, height=length)
        mesh.paint_uniform_color(EDGE_COLORS.get(edge.edge_type, EDGE_COLORS["local"]))
        mesh.rotate(_rotation_from_z(direction), center=np.zeros(3, dtype=np.float64))
        mesh.translate((start + end) * 0.5)
        meshes.append(mesh)
    return _combine_meshes(meshes)


def save_graph_visualization(
    graph: DynamicSceneGraph,
    global_pcd_pts: Sequence[np.ndarray] | np.ndarray | None,
    global_pcd_cols: Sequence[np.ndarray] | np.ndarray | None,
    out_dir: str | Path,
    clip_id: int,
    scene_filename: str = "scene_with_graph.ply",
    frame: int | None = None,
    exact: bool = False,
    write_nodes_edges: bool = True,
) -> None:
    """Write node, edge, and merged scene visualizations after an interaction clip."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    scene_pts, scene_cols = _scene_arrays(global_pcd_pts, global_pcd_cols)
    scene_pts, scene_cols = _semantic_scene_arrays(graph, scene_pts, frame=frame, exact=exact)

    node_mesh = _make_node_mesh(graph, frame=frame, exact=exact)
    edge_mesh = _make_edge_mesh(graph, frame=frame, exact=exact)
    merged_mesh = _scene_mesh_from_points(scene_pts, scene_cols)
    merged_mesh += node_mesh
    merged_mesh += edge_mesh
    if len(merged_mesh.vertices):
        merged_mesh.compute_vertex_normals()

    if write_nodes_edges:
        o3d.io.write_triangle_mesh(str(out_path / "functional_nodes.ply"), node_mesh, write_ascii=False)
        o3d.io.write_triangle_mesh(str(out_path / "functional_edges.ply"), edge_mesh, write_ascii=False)
    o3d.io.write_triangle_mesh(str(out_path / scene_filename), merged_mesh, write_ascii=False)
    if write_nodes_edges:
        print(
            f"  [Vis] clip={clip_id} -> functional_nodes.ply, functional_edges.ply, {scene_filename}"
        )
    else:
        print(f"  [Vis] clip={clip_id} -> {scene_filename}")

def save_openfungraph_visualization(
    graph: DynamicSceneGraph,
    global_pcd_pts: Sequence[np.ndarray] | np.ndarray | None,
    global_pcd_cols: Sequence[np.ndarray] | np.ndarray | None,
    out_dir: str | Path,
    clip_id: int,
) -> None:
    """Write an OpenFunGraph-style interactive viewer using pyviz3d."""
    if pyviz3d_viz is None:
        print("  [Vis] pyviz3d unavailable, skipping OpenFunGraph-style viewer export")
        return

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    viewer_dir = out_path / "fungraph"

    scene_pts, scene_cols = _scene_arrays(global_pcd_pts, global_pcd_cols)
    if len(scene_pts):
        pc_center = np.mean(scene_pts, axis=0).astype(np.float32)
    elif graph.nodes:
        pc_center = np.mean(
            np.stack([np.asarray(node.centroid, dtype=np.float32) for node in graph.nodes.values()], axis=0),
            axis=0,
        ).astype(np.float32)
    else:
        pc_center = np.zeros(3, dtype=np.float32)

    viewer = pyviz3d_viz.Visualizer()
    color_lookup: dict[str, np.ndarray] = {}

    if len(scene_pts):
        bg_pts, bg_cols, bg_normals = _prepare_viz_points(scene_pts, scene_cols, voxel_size=0.0)
        viewer.add_points(
            "Background",
            bg_pts - pc_center,
            bg_cols * 255.0,
            bg_normals,
            point_size=20,
            visible=True,
        )

    sorted_nodes = sorted(
        graph.nodes.values(),
        key=lambda node: (node.node_type != "object", node.label, node.node_id),
    )
    for idx, node in enumerate(sorted_nodes):
        prefix = "O" if node.node_type == "object" else "P"
        handle = f"{prefix}{idx}"
        pts, cols, normals = _prepare_viz_points(node.points, node.colors, voxel_size=0.01)
        instance_color = _stable_instance_color(f"{node.node_type}:{node.label}:{node.node_id}")
        color_lookup[handle] = instance_color

        if len(pts):
            viewer.add_points(
                handle,
                pts - pc_center,
                cols * 255.0,
                normals,
                point_size=20,
                visible=True,
            )

        centroid = np.asarray(node.centroid, dtype=np.float32) - pc_center
        viewer.add_points(
            f"Ins{handle}",
            np.expand_dims(centroid, axis=0),
            np.expand_dims(instance_color * 255.0, axis=0),
            point_size=200 if node.node_type == "object" else 150,
            resolution=15,
            visible=True,
        )

        label_text = f"{handle}: {node.label}"
        if node.state and node.state != "unknown":
            label_text += f" [{node.state}]"
        viewer.add_labels(
            f"Label{handle}",
            [label_text],
            np.expand_dims(centroid, axis=0),
            np.expand_dims(instance_color * 255.0, axis=0),
            visible=True,
        )

    edge_label_idx = 0
    for edge in graph.edges.values():
        src = graph.nodes.get(edge.src_id)
        dst = graph.nodes.get(edge.dst_id)
        if src is None or dst is None:
            continue

        src_center = np.asarray(src.centroid, dtype=np.float32) - pc_center
        dst_center = np.asarray(dst.centroid, dtype=np.float32) - pc_center
        viewer.add_polyline(
            f"Edge_{edge.edge_id[:8]}",
            np.stack([src_center, dst_center], axis=0),
            color=OPENFUNGRAPH_EDGE_COLOR,
            edge_width=0.02,
            alpha=0.5,
        )

        relationship = edge.relationship.strip() if edge.relationship else edge.edge_type
        if relationship:
            viewer.add_labels(
                f"EdgeLabel{edge_label_idx}",
                [relationship],
                np.expand_dims(((src_center + dst_center) * 0.5).astype(np.float32), axis=0),
                np.expand_dims(np.ones(3, dtype=np.float32) * 255.0, axis=0),
                visible=True,
            )
            edge_label_idx += 1

    viewer.save(str(viewer_dir))
    np.save(viewer_dir / "color.npy", color_lookup, allow_pickle=True)
    print(f"  [Vis] clip={clip_id} -> fungraph/ (static pyviz3d export, not a live Viser server)")


def main() -> None:
    """Load a saved graph and scene point cloud, then write graph visualization PLY files."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-graph-dir", default="scene_graph", help="Directory containing graph.json")
    parser.add_argument("--scene-ply", default="scene_full.ply", help="Path to the saved global scene point cloud")
    parser.add_argument("--out-dir", default=".", help="Directory where the visualization PLY files are written")
    parser.add_argument("--clip-id", type=int, default=-1, help="Optional clip id used for logging")
    args = parser.parse_args()
    graph = DynamicSceneGraph.load(args.scene_graph_dir)
    scene_pts = np.zeros((0, 3), dtype=np.float32)
    scene_cols = np.zeros((0, 3), dtype=np.float32)
    scene_path = Path(args.scene_ply)
    if scene_path.exists():
        pcd = o3d.io.read_point_cloud(str(scene_path))
        scene_pts = np.asarray(pcd.points, dtype=np.float32)
        scene_cols = np.asarray(pcd.colors, dtype=np.float32) if pcd.has_colors() else np.zeros((0, 3), dtype=np.float32)
    save_graph_visualization(graph, scene_pts, scene_cols, args.out_dir, args.clip_id)
    save_openfungraph_visualization(graph, scene_pts, scene_cols, args.out_dir, args.clip_id)
if __name__ == "__main__":
    main()

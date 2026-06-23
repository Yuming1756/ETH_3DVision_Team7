from pathlib import Path
import sys
import argparse
import json

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
TEMP_ROOT = PROJECT_ROOT / "temp"

if TEMP_ROOT.exists() and str(TEMP_ROOT) not in sys.path:
    sys.path.insert(0, str(TEMP_ROOT))

from dynamic_scene_graph import DynamicSceneGraph  # noqa: E402


# ============================================================
# Geometry helpers
# ============================================================

def bbox_center(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=float)

    if pts.ndim != 2 or pts.shape[0] == 0:
        return np.zeros(3)

    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)

    return (mins + maxs) * 0.5


def centroid(points: np.ndarray) -> np.ndarray:
    return np.mean(points, axis=0)


def bbox_dimensions(points: np.ndarray) -> np.ndarray:
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    return maxs - mins


def bbox_iou(min1, max1, min2, max2):
    inter_min = np.maximum(min1, min2)
    inter_max = np.minimum(max1, max2)

    inter_dims = np.maximum(inter_max - inter_min, 0)
    inter_vol = np.prod(inter_dims)

    vol1 = np.prod(max1 - min1)
    vol2 = np.prod(max2 - min2)

    union = vol1 + vol2 - inter_vol

    if union <= 0:
        return 0.0

    return inter_vol / union


# ============================================================
# Graph loading
# ============================================================

def load_graph(output_dir: Path) -> DynamicSceneGraph:
    graph_dir = output_dir / "scene_graph"

    if not (graph_dir / "graph.json").exists():
        print(f"Missing {graph_dir / 'graph.json'}")
        return DynamicSceneGraph()

    return DynamicSceneGraph.load(str(graph_dir))


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="Evaluate point cloud motion estimation."
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True
    )

    parser.add_argument(
        "--groundtruth_dir",
        type=str,
        required=True
    )

    parser.add_argument(
        "--label",
        type=str,
        required=True
    )

    args = parser.parse_args()

    graph = load_graph(Path(args.output_dir))

    print(
        f"Loaded graph with "
        f"{len(graph.nodes)} nodes and "
        f"{len(graph.edges)} edges."
    )

    node = None
    node_id = None

    for cur_id, cur_node in graph.nodes.items():

        print(
            f"label={cur_node.label}, "
            f"motion_segments={len(cur_node.motion_segments)}"
        )

        if cur_node.label == args.label:
            node = cur_node
            node_id = cur_id
            break

    if node is None:
        print(f"No node with label '{args.label}' found.")
        return

    # ========================================================
    # Metrics accumulators
    # ========================================================

    translation_errors = []
    centroid_errors = []
    motion_errors = []
    ious = []
    size_errors = []

    gt_positions = []
    est_positions = []

    articulation_direction_errors = []

    initial_gt_pos = None
    initial_est_pos = None

    frame_count = 0

    articulation_directions_gt = None

    gt_file = (
        Path(args.groundtruth_dir)
        / "motion.json"
    )

    if gt_file.exists():
        with open(gt_file, "r", encoding="utf-8") as f:
            articulation_directions_gt = json.load(f)["motion_dir"]

        

    # ========================================================
    # Iterate through frames
    # ========================================================

    for segment in node.motion_segments:

        # ------------------------------------------------
        # articulation direction
        # ------------------------------------------------

        if articulation_directions_gt is not None:
            est_dir = segment.articulation_axis_direction
            articulation_directions_gt /= np.linalg.norm(articulation_directions_gt)
            est_dir /= np.linalg.norm(est_dir)

            cos_sim = np.clip(
                abs(np.dot(articulation_directions_gt, est_dir)),
                -1.0,
                1.0,
            )

            angle_error_deg = np.degrees(
                np.arccos(cos_sim)
            )

            articulation_direction_errors.append(
                angle_error_deg
            )

        for motion_pose in segment.pose_history:

            frame_id = int(motion_pose.frame)

            gt_file = (
                Path(args.groundtruth_dir)
                / f"{frame_id:06d}.json"
            )

            if not gt_file.exists():
                continue

            with open(gt_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            gt_ann = data["data"][next(iter(data["data"]))]

            # ------------------------------------------------
            # GT bbox
            # ------------------------------------------------

            gt_center = np.array(
                gt_ann["bbox_center_default"],
                dtype=float
            )

            gt_dims = np.array(
                [
                    gt_ann["bbox_width_default"],
                    gt_ann["bbox_height_default"],
                    gt_ann["bbox_length_default"],
                ],
                dtype=float,
            )

            gt_min = gt_center - gt_dims / 2.0
            gt_max = gt_center + gt_dims / 2.0

            # ------------------------------------------------
            # Estimated point cloud
            # ------------------------------------------------

            rendered = graph.get_node_geometry_at_frame(
                node_id,
                frame_id,
                exact=True
            )

            if rendered is None:
                continue

            points = np.asarray(rendered[0], dtype=float)

            if len(points) == 0:
                continue

            est_bbox_center = bbox_center(points)
            est_centroid = centroid(points)

            est_min = points.min(axis=0)
            est_max = points.max(axis=0)

            est_dims = est_max - est_min

            # ------------------------------------------------
            # Initialization
            # ------------------------------------------------

            if initial_gt_pos is None:

                initial_gt_pos = gt_center
                initial_est_pos = est_bbox_center

                print("\nInitial positions")
                print("GT :", initial_gt_pos)
                print("EST:", initial_est_pos)

            # ------------------------------------------------
            # Translation Error
            # ------------------------------------------------

            translation_error = np.linalg.norm(
                gt_center - est_bbox_center
            )

            centroid_error = np.linalg.norm(
                gt_center - est_centroid
            )

            # ------------------------------------------------
            # Relative Motion Error
            # ------------------------------------------------

            gt_motion = gt_center - initial_gt_pos
            est_motion = est_bbox_center - initial_est_pos

            motion_error = np.linalg.norm(
                gt_motion - est_motion
            )

            # ------------------------------------------------
            # Bounding Box IoU
            # ------------------------------------------------

            iou = bbox_iou(
                gt_min-initial_gt_pos,
                gt_max-initial_gt_pos,
                est_min-initial_est_pos,
                est_max-initial_est_pos
            )

            # ------------------------------------------------
            # Size Error
            # ------------------------------------------------

            size_error = np.linalg.norm(
                gt_dims - est_dims
            )


            # ------------------------------------------------
            # Save metrics
            # ------------------------------------------------

            translation_errors.append(
                translation_error
            )

            centroid_errors.append(
                centroid_error
            )

            motion_errors.append(
                motion_error
            )

            ious.append(
                iou
            )

            size_errors.append(
                size_error
            )

            gt_positions.append(
                gt_center
            )

            est_positions.append(
                est_bbox_center
            )

            frame_count += 1

            print(
                f"Frame {frame_id:6d} | "
                f"TransErr={translation_error:.4f} m | "
                f"MotionErr={motion_error:.4f} m | "
                f"IoU={iou:.3f} | "
                f"SizeErr={size_error:.4f}"
            )

    # ========================================================
    # Summary
    # ========================================================

    if frame_count == 0:
        print("No valid frames found.")
        return

    gt_positions = np.asarray(gt_positions)
    est_positions = np.asarray(est_positions)

    rmse = np.sqrt(
        np.mean(
            np.sum(
                (gt_positions - est_positions) ** 2,
                axis=1
            )
        )
    )

    print("\n" + "=" * 60)
    print("Evaluation Summary")
    print("=" * 60)

    print(f"Frames evaluated          : {frame_count}")

    print(
        f"Mean Translation Error    : "
        f"{np.mean(translation_errors):.4f} m"
    )

    print(
        f"Median Translation Error  : "
        f"{np.median(translation_errors):.4f} m"
    )

    print(
        f"Trajectory RMSE           : "
        f"{rmse:.4f} m"
    )

    print(
        f"Mean Centroid Error       : "
        f"{np.mean(centroid_errors):.4f} m"
    )

    print(
        f"Mean Relative Motion Err  : "
        f"{np.mean(motion_errors):.4f} m"
    )

    print(
        f"Mean Bounding Box IoU     : "
        f"{np.mean(ious):.4f}"
    )

    print(
        f"Mean Size Error           : "
        f"{np.mean(size_errors):.4f} m"
    )

    print(
        f"Max Translation Error     : "
        f"{np.max(translation_errors):.4f} m"
    )

    print(
        f"Min Translation Error     : "
        f"{np.min(translation_errors):.4f} m"
    )

    if articulation_directions_gt is not None:
        print(
            f"Mean Articulation Dir Err : "
            f"{np.mean(articulation_direction_errors):.4f} deg"
        )

    print("=" * 60)


if __name__ == "__main__":
    main()
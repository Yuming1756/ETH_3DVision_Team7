#!/usr/bin/env python3
"""Main dynamic functional 3D scene graph pipeline for egocentric RGB-D video."""
# ----------------------------------------------------------------------------
# This file was split for readability into temp/graph_sections/NN_*.py.
#
# The pipeline used to live in one ~12.8k-line module. To keep behaviour
# byte-for-byte identical while making the code navigable, the original body is
# stored as contiguous, themed section files and loaded back here into a single
# shared namespace, in order. Because every section executes in THIS module's
# globals, names defined in earlier sections are visible to later ones exactly
# as they were in the monolith — there is no import wiring to get wrong.
#
# To read/edit the pipeline, open the relevant section under graph_sections/:
#   00_imports_config        imports, dataclasses, constants, depth scale
#   01_labels_relations      label / affordance / relation classification
#   02_geometry              intrinsics, depth alignment, mask lifting
#   03_frames_pose_hands     topic indexing, pose buffer, hand/contact cues
#   04_frames_clips          frame loading, clip segmentation
#   05_perception            Depth-Anything-3 + SAM3 wrappers
#   06_boxes_masks           box/crop/mask helpers, SAM3 prompts, confidence
#   07_tracking              tracking points, rigid transforms, CoTracker
#   08_artipoint_filters     ArtiPoint track filtering / smoothing
#   09_articulation_fit      plane/circle/revolute + GTSAM joint-opt fit
#   10_artipoint_lostfound   articulation update + LostFound tracking
#   11_vlm_reasoning         Qwen-VL loading / generation / reasoning
#   12_object_state          object-state inference, captions, descriptions
#   13_pointcloud_io         point-cloud bounds/colors/snapshots, overlays
#   14_geometry_update       SE3 utils, trajectory smoothing, articulation apply
#   15_motion_export         motion records, viser/wrist export, clip IO
#   16_inactive_depth_cache  inactive priors, depth cache, global scene
#   17_pipeline_main         clip processing, run_pipeline, argparse, main
#
# Entry point and CLI are unchanged: `python3 graph.py --data-dir ... etc`.
# ----------------------------------------------------------------------------
from __future__ import annotations

from pathlib import Path as _Path

_SECTIONS_DIR = _Path(__file__).resolve().parent / "graph_sections"
_section_files = sorted(_SECTIONS_DIR.glob("[0-9][0-9]_*.py"))
if not _section_files:
    raise RuntimeError(f"No graph sections found under {_SECTIONS_DIR}")

for _section_path in _section_files:
    _code = compile(_section_path.read_text(encoding="utf-8"), str(_section_path), "exec")
    exec(_code, globals())

del _section_path, _code, _section_files, _SECTIONS_DIR


if __name__ == "__main__":
    main()  # noqa: F821  (defined in 17_pipeline_main.py)

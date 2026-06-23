# Dynamic Functional 3D Scene Graph

`graph.py` builds a **dynamic functional 3D scene graph** from a single egocentric
RGB-D recording. Given RGB-D frames, camera poses, hand tracks, and hand-object
interaction cues, the pipeline reconstructs:

- the static 3D scene geometry,
- the geometry of manipulated objects,
- the **functional relationship** between an interactive *element* such as a
  handle, knob, rim, or switch and its parent or target *object*, and
- the **motion and state change** caused by the interaction, including rigid
  transport and 1-DoF articulated motion.

The pipeline is organized around active and inactive clips:

- **inactive clips** provide static scene reconstruction and pre-interaction
  object priors,
- **active clips** trigger VLM reasoning, object grounding, geometry updates,
  motion estimation, and state updates.

Hands, arms, and the person are masked out and are not reconstructed as scene
geometry.

---

## Dataset and Past Runs

The original MicroAGI01 dataset was taken offline near the end of the project.
For reproducibility, we uploaded one extracted scene ourselves.

Dataset scene:

```text
https://drive.google.com/file/d/1NaD05hQC-PV7V8E6Zkk97Chs7low0ZDz/view?usp=drive_link
```

Extract it into:
```text
data/open-source-183_extracted/
```

Past runs can be found here:
```text
https://drive.google.com/drive/folders/1QIL4kEZyg_k5gTSHQx2qOvDCCQdQC3_s?usp=sharing
```

Store downloaded output folders under:
```text
output/
```

## Expected Submission Structure

The submitted package is expected to have the following structure:

```text
ETH_3DVision_Team7/
├── README.md                         
│
├── final_project/                    # main code snapshot
│   ├── graph.py
│   ├── dynamic_scene_graph.py
│   ├── json_utils.py
│   ├── visualize_graph.py
│   ├── visualize_scene_graph_html_multi_status.py
│   │
│   └── graph_sections/
│       ├── 00_imports_config.py
│       ├── 01_labels_relations.py
│       ├── 02_geometry.py
│       ├── 03_frames_pose_hands.py
│       ├── 04_frames_clips.py
│       ├── 05_perception.py
│       ├── 06_boxes_masks.py
│       ├── 07_tracking.py
│       ├── 08_artipoint_filters.py
│       ├── 09_articulation_fit.py
│       ├── 10_artipoint_lostfound.py
│       ├── 11_vlm_reasoning.py
│       ├── 12_object_state.py
│       ├── 13_pointcloud_io.py
│       ├── 14_geometry_update.py
│       ├── 15_motion_export.py
│       ├── 16_inactive_depth_cache.py
│       └── 17_pipeline_main.py
│
└── team7_env_export/                 # environment and model manifests
    ├── README_REPRODUCE.md
    ├── requirements.txt
    ├── requirements_freeze.txt
    ├── environment_full.yml
    ├── environment_from_history.yml
    ├── conda_list.txt
    ├── conda_explicit.txt
    ├── pip_list.txt
    ├── package_versions.txt
    ├── python_path.txt
    ├── python_version.txt
    ├── nvidia_smi.txt
    ├── nvcc_version.txt
    ├── model_and_sam3_manifest.txt
    ├── da3_qwen_manifest.txt
    ├── hf_model_snapshots.txt
    └── create_hf_model_snapshots.py
```
External model locations and versions are documented in:
```text
model_and_sam3_manifest.txt
da3_qwen_manifest.txt
hf_model_snapshots.txt
```
The MicroAGI scene data and past output runs should be downloaded separately from
the links provided.
After downloading the dataset and optional past runs, the project directory
should look like this:

```text
final_project/
├── graph.py
├── graph_sections/
├── dynamic_scene_graph.py
├── json_utils.py
├── visualize_graph.py
├── visualize_scene_graph_html_multi_status.py
├── eval.py
├── viser_io.py
├── requirements.txt
├── data/
│   └── open-source-183_extracted/
│       ├── camera_color_image/
│       ├── camera_depth_image/
│       ├── camera_color_info/
│       ├── camera_depth_info/
│       ├── camera_depth_unit_of_depth_in_mm/
│       ├── tf_camera/
│       ├── tf_camera_health/
│       ├── tf_static/
│       ├── tf_hands/
│       ├── hands_left/
│       ├── hands_left_health/
│       ├── hands_right/
│       └── hands_right_health/
├── output/
│   └── output_final_183_long/
│       ├── clips.json
│       ├── scene_full.ply
│       ├── scene_with_graph.ply
│       ├── functional_nodes.ply
│       ├── functional_edges.ply
│       ├── scene_graph/
│       │   ├── graph.json
│       │   ├── pointclouds/
│       │   └── motion_pointclouds/
│       └── clip_NNN/
│           └── event.json
└── annotations/
    └── 183/
        ├── bowl/
        ├── spoon/
        ├── cabinet_door/
        └── drawer/
```

## Environment and Reproducibility
The Python package requirements are listed in:
```text
requirements.txt
requirements_freeze.txt
```

A basic setup is:
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

For a closer reproduction of our cluster environment, we also provide:
```text
environment_full.yml
environment_from_history.yml
conda_list.txt
conda_explicit.txt
pip_list.txt
package_versions.txt
nvidia_smi.txt
nvcc_version.txt
```

A conda environment can be recreated with:
```bash
conda env create -f environment_full.yml
conda activate <env_name>
pip install -r requirements_freeze.txt
```

If environment_full.yml is platform-specific, use the lighter history file:
```bash
conda env create -f environment_from_history.yml
conda activate <env_name>
pip install -r requirements_freeze.txt
```

## External Models
Large model weights are not included in the code snapshot. The model
configuration and local cache snapshots are documented in:
```text
model_and_sam3_manifest.txt
da3_qwen_manifest.txt
hf_model_snapshots.txt
```
Main external models/components:

- SAM3 / SAM3-I: open-vocabulary video mask propagation.
- DepthAnything3: metric depth estimation when DA3 is enabled.
- Qwen-VL: element/object detection, functional relation prediction,
object-state inference, seed-frame selection, and recall@K label candidates.
- CoTracker3: 2D point tracking for rigid and articulated motion.
- GTSAM: factor-graph fitting for articulated motion.
- Open3D: point-cloud processing and PLY export.

The HuggingFace models used in the final run were:
```text
Qwen-VL: Qwen/Qwen3-VL-4B-Instruct
DA3: depth-anything/DA3NESTED-GIANT-LARGE-1.1
```

The local snapshot hashes are recorded in **hf_model_snapshots.txt**.

For offline reproduction, copy the HuggingFace cache folders listed in
**hf_model_snapshots.txt** and set:

```bash
export HF_HOME=/path/to/copied/huggingface_cache
export HF_HUB_OFFLINE=1
```

## Code Layout
The main pipeline is split into section files under **graph_sections/**.
**graph.py** loads these files in order into a shared namespace.
| Section | Contents |
|---|---|
| `00_imports_config.py` | Imports, dataclasses, constants, environment/config knobs |
| `01_labels_relations.py` | Label normalization, affordance rules, relation vocabulary |
| `02_geometry.py` | Intrinsics, depth alignment, mask-to-world lifting |
| `03_frames_pose_hands.py` | Topic indexing, pose buffer, hand/contact cues |
| `04_frames_clips.py` | Frame loading, clip segmentation, gap bridging |
| `05_perception.py` | DepthAnything3 and SAM3 wrappers |
| `06_boxes_masks.py` | Box, crop, mask, prompt, and confidence helpers |
| `07_tracking.py` | Tracking points, rigid transforms, CoTracker helpers |
| `08_artipoint_filters.py` | ArtiPoint-style track filtering and smoothing |
| `09_articulation_fit.py` | Revolute/prismatic fitting and GTSAM optimization |
| `10_artipoint_lostfound.py` | Articulated update and LostFound-style rigid tracking |
| `11_vlm_reasoning.py` | Qwen-VL loading, prompting, detection, relation reasoning |
| `12_object_state.py` | Object-state inference, captions, descriptions |
| `13_pointcloud_io.py` | Point-cloud snapshots, colors, overlays |
| `14_geometry_update.py` | SE(3) utilities, trajectory smoothing, articulation application |
| `15_motion_export.py` | Motion records, wrist export, clip I/O |
| `16_inactive_depth_cache.py` | Inactive priors, depth cache, global scene |
| `17_pipeline_main.py` | Active/inactive clip processing, `run_pipeline`, CLI |

Other important files:
| File | Purpose |
|---|---|
| `dynamic_scene_graph.py` | Graph data structure and serialization |
| `json_utils.py` | Robust JSON parsing utilities |
| `visualize_graph.py` | PLY and pyviz3d graph visualization exporter |
| `visualize_scene_graph_html_multi_status.py` | Browser-based HTML viewer for all events and status transitions |
| `eval.py` | Trajectory and articulation evaluation |
| `viser_io.py` | Viser-based graph and motion visualization |

## Expected Input
The input is one extracted MicroAGI / Aria-style recording directory, for example:
```text
data/open-source-183_extracted/
```
Expected topic folders include:
```text
camera_color_image/
camera_depth_image/
camera_color_info/
camera_depth_info/
camera_depth_unit_of_depth_in_mm/
tf_camera/
tf_camera_health/
tf_static/
tf_hands/
hands_left/
hands_left_health/
hands_right/
hands_right_health/
```

RGB and depth frames are named:
```text
NNNNNN_<timestamp_ns>.png
```
The pipeline filters invalid data before reconstruction. Camera poses are only
trusted inside contiguous valid **tf_camera_health** blocks. Invalid RGB frames
are not used for perception, and invalid hand tracks are ignored. This is
important because pose discontinuities corrupt both scene reconstruction and
object-motion estimation.

## How to Run

Example direct invocation:

```bash
python3 graph.py \
  --data-dir data/open-source-183_extracted \
  --frame-start 280 \
  --frame-end 690 \
  --output-dir output/output_final_183_long \
  --inactive-stride 1 \
  --inactive-prior-frames 10 \
  --hf-cache-dir .cache/huggingface \
  --qwen-vl-model-id "Qwen/Qwen3-VL-4B-Instruct" \
  --sam-backend sam3
```
On the cluster, use one of the SLURM launchers in the repository root. These
scripts activate the environment, set HuggingFace cache paths/tokens, start the
required services, and export the relevant environment variables.

Example:
```bash
sbatch graph_183_long_4b.sh
```
**Important: Start on an Inactive Segment**
The frame window should start before the first interaction. The first clip must
be inactive so the pipeline can build the pre-interaction object prior. 

If frame-start lands inside an interaction, the pipeline may fail with:

```text
[ActiveClip][Error] clip 0: no preceding inactive clip; graph-prior fallback is disabled.
```
Fix this by moving **--frame-start** earlier so that at least a few inactive
frames are included before the first active clip.

## Pipeline Summary
### Clip Selection

Inspired by Lost & Found, the video is split into inactive and active clips
using hand-object contact cues. A contact candidate requires:

- a confident hand detection,
- a confident object detection,
- the detector's in-contact flag,
- and a short 3D wrist-to-object distance.

Left- and right-hand active flags are inferred with a temporal state machine.
The raw contact signal is smoothed, short inactive gaps can be bridged, and
active clips can optionally be extended over short detector dropouts.
### Functional Reasoning
For each active clip, Qwen-VL receives a sampled active video and, when
available, context frames from the preceding inactive clip. It predicts:

- the directly touched interactive element,
- the manipulated or controlled object,
- whether the relation is local or remote,
- the functional relation,
- the motion coupling type,
- before/after object states,
- and ranked label candidates.

Few-shot prompts are used to reduce common errors, such as selecting nearby
support objects instead of the object actually manipulated by the hand.

### Segmentation and 3D Grounding

SAM3 segments the predicted element and object labels. The pipeline uses a
cross-clip propagation strategy: object and element masks are seeded from
Qwen-selected active frames and propagated through the concatenated
inactive-active sequence. Backward-propagated inactive masks provide
pre-interaction object priors.

Masks are lifted into 3D using depth and camera poses. Hand masks are removed
before lifting, and low-confidence or invalid mask frames are discarded.

### Motion and Geometry Update

Geometry is updated according to the predicted motion coupling.

- Rigid transport: bowls, spoons, cups, bottles, and similar objects are
modeled as single rigid bodies. Rotation is estimated from world-lifted
CoTracker3 tracks using incremental Kabsch alignment. Translation is anchored
to the acting hand when a reliable wrist trajectory is available.
- Articulated motion: cabinet doors, refrigerator doors, drawers, and
similar objects are modeled as 1-DoF articulations. The system tracks
object-part keypoints, lifts them into 3D, filters and smooths the tracks, and
fits a revolute or prismatic model using an ArtiPoint-style/GTSAM factor graph.

If an object is touched again in a later active clip, the graph attempts to
match it to the existing node and updates the same node instead of creating a
duplicate.

## Key Configuration

Important environment variables and CLI-related parameters:
| Variable / Option | Default | Effect |
|---|---:|---|
| `GRAPH_SAM_BACKEND` | `sam3` | SAM backend |
| `--inactive-prior-frames` | `10` | Number of inactive frames used for object priors |
| `ACTIVE_CLIP_BRIDGE_GAP_FRAMES` | `15` | Bridge short inactive holes inside continuous interactions |
| `RIGID_RESEED_EVERY_FRAME` | `1` | Re-seed rigid tracking points across active frames |
| `RIGID_KABSCH_MIN_PAIRS` | `6` | Minimum shared tracks for Kabsch rotation |
| `QWEN_CONTEXT_FRAMES` | `1` | Number of inactive context frames shown to Qwen |
| `ARTIPOINT_LITERAL_FILTERS` | `1` | Use ArtiPoint-style track filtering |
| `ARTIPOINT_OCCLUSION_FILTER_MODE` | `off` | Occlusion filter mode |
| `SAM3_MAX_SIDE` | `0` | Maximum SAM3 image side; `0` keeps native resolution |

Most constants are defined in:
```text
graph_sections/00_imports_config.py
```

## Outputs

All outputs are written under **--output-dir**.

Important files:
| Output | Description |
|---|---|
| `clips.json` | Active/inactive clip segmentation summary |
| `scene_full.ply` | Accumulated static scene reconstruction |
| `functional_nodes.ply` | Node marker spheres, not object geometry |
| `functional_edges.ply` | Graph edge markers |
| `scene_with_graph.ply` | Scene plus graph markers |
| `scene_graph/graph.json` | Serialized dynamic functional scene graph |
| `scene_graph/pointclouds/*.ply` | Per-node geometry |
| `scene_graph/motion_pointclouds/*.ply` | Motion-segment reference geometry |
| `clip_NNN/event.json` | Per-active-clip reasoning, relation, state update, and debug info |

For evaluation, each motion sample in **graph.json** also stores:
```text
global_position
global_trajectory
```

Use these fields for translation evaluation. Do not use raw **pose[:3,3]** as a
translation trajectory, because it is a delta-pose artifact.

Each node/event may also contain:
```text
label_candidates
```
which are Qwen's ranked top-5 label predictions for recall@K evaluation.

## Evaluation

**eval.py** evaluates output graph trajectories against sparse ground-truth
annotations. It computes distance metrics for node trajectories and, when
available, articulation-axis errors.

Example:

```bash
python3 eval.py \
  --output_dir output/output_final_183_long \
  --groundtruth_dir annotations/183/bowl \
  --label "bowl"
```

Other evaluated labels include:
```text
spoon
bowl
cabinet door
drawer
```

Arguments:


## Visualization

### PLY / PyViz3D Export
**visualize_graph.py** writes graph visualization assets such as:
```text
functional_nodes.ply
functional_edges.ply
scene_with_graph.ply
fungraph/
```

Example:
```text
python3 visualize_graph.py \
  --scene-graph-dir output/output_final_183_long/scene_graph \
  --scene-ply output/output_final_183_long/scene_full.ply \
  --out-dir output/output_final_183_long \
  --clip-id -1
```
### Browser HTML Multi-Status Viewer
**visualize_scene_graph_html_multi_status.py** creates an interactive browser viewer with:

- the reconstructed 3D scene,
- graph nodes and edges,
- all event summaries,
- all status transitions,
- and clickable event-level reasoning.

Example:
```bash
RUN=output/output_final_183_long

python3 visualize_scene_graph_html_multi_status.py \
  --scene-ply "$RUN/scene_full.ply" \
  --graph-json "$RUN/scene_graph/graph.json" \
  --event-jsons $(find "$RUN" -maxdepth 2 -path "$RUN/clip_*/event.json" | sort) \
  --out-html "$RUN/scene_graph_viewer_all_clips.html" \
  --point-size 0.02 \
  --node-radius 0.08
```
Then serve the output folder:
```bash
cd "$RUN"
python3 -m http.server 6018 --bind 127.0.0.1
```

Finally open:
```text
http://localhost:6018/scene_graph_viewer_all_clips.html
```

If running on a remote cluster, forward the port first.


# Dynamic Functional 3D Scene Graph

> **_NOTE:_**  We were unable to confirm in time that the whole pipeline runs as expected. In case of any issues contact thouben@student.ethz.ch or check our repository https://github.com/Yuming1756/ETH_3DVision_Team7/tree/main

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
│   ├── graph.py                      # main pipeline entry point
│   ├── dynamic_scene_graph.py         # dynamic functional scene graph data structure
│   ├── json_utils.py
│   ├── visualize_graph.py             # PLY / OpenFunGraph-style graph visualization
│   ├── visualize_scene_graph_html_multi_status.py
│   ├── requirements.txt               # portable Python dependencies
│   ├── requirements_git.txt           # Git-based dependencies
│   ├── requirements_torch_cuda126.txt # optional CUDA 12.6 PyTorch setup
│   ├── requirements_platform_optional.txt
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
    ├── requirements.txt               # portable dependency file
    ├── requirements_git.txt           # Git-based dependencies
    ├── requirements_torch_cuda126.txt # optional CUDA 12.6 PyTorch setup
    ├── requirements_platform_optional.txt
    ├── requirements_freeze.txt        # exact ETH cluster pip freeze
    ├── requirements_local_success.txt # one tested local installation record
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
    |   ├── bowl/
    |   ├── cabinet_door/
    |   └── drawer/
    └── 307/
        └── spoon/

```

## Environment and Reproducibility
This section describes the recommended procedure for setting up the project environment from a clean machine. The repository separates portable dependencies, Git-based dependencies, CUDA/PyTorch dependencies, and optional platform-specific packages because the exact ETH cluster environment is not fully portable.

### Create a clean Python environment

We recommend Python 3.12.

```bash
conda create -n team7_3dv python=3.12 -y
conda activate team7_3dv

python -m pip install --upgrade pip setuptools wheel
```

### Install portable Python dependencies

The main portable dependency file is:
```text
team7_env_export/requirements.txt
```
Install it with:
```bash
pip install -r team7_env_export/requirements.txt
```
This file is generated from the original cluster environment but removes cluster-local editable paths and highly platform-specific packages.

### Install Git-based dependencies

Some dependencies are installed directly from GitHub and are listed separately in:
```text
team7_env_export/requirements_git.txt
```
Install them with:
```bash
pip install -r team7_env_export/requirements_git.txt
```
If a GitHub clone fails because of a temporary network issue, rerun the command. These dependencies include external tracking/depth packages used by the pipeline.

### Install PyTorch / CUDA dependencies

PyTorch is installed separately because the correct wheel depends on the local NVIDIA driver and CUDA compatibility.

For a CUDA 12.6-compatible Linux GPU setup, use:

pip install -r team7_env_export/requirements_torch_cuda126.txt

This setup was tested locally with:
```text
Python 3.12
PyTorch 2.10.0+cu126
NVIDIA RTX 3090
```
After installation, verify CUDA visibility:
```bash
python - <<'PY'
import torch

print("torch:", torch.__version__)
print("torch cuda version:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())

if torch.cuda.is_available():
    print("device name:", torch.cuda.get_device_name(0))
PY
```
Expected output should include:
```text
cuda available: True
```
If CUDA is not available, the most likely cause is a mismatch between the installed PyTorch CUDA wheel and the local NVIDIA driver. In that case, install the PyTorch wheel matching the target machine instead of using the provided CUDA 12.6 file.

### Install platform-specific packages

They are listed in:
```text
team7_env_export/requirements_platform_optional.txt
```
Install them:
```bash
pip install -r team7_env_export/requirements_platform_optional.txt
```
### Verify the installation

Run a basic import check from the repository root:
```bash
python - <<'PY'
packages = [
    ("numpy", "numpy"),
    ("cv2", "cv2"),
    ("torch", "torch"),
    ("transformers", "transformers"),
    ("accelerate", "accelerate"),
    ("cotracker", "cotracker"),
    ("depth_anything_3", "depth_anything_3"),
]

for name, modname in packages:
    try:
        mod = __import__(modname)
        print(f"{name}: OK", getattr(mod, "__version__", ""))
    except Exception as e:
        print(f"{name}: FAILED -> {e}")
PY
```

### Exact environment records

The following files are included for reproducibility documentation:
```text
team7_env_export/requirements_freeze.txt
team7_env_export/requirements_local_success.txt
team7_env_export/environment_full.yml
team7_env_export/environment_from_history.yml
team7_env_export/conda_list.txt
team7_env_export/pip_list.txt
team7_env_export/package_versions.txt
team7_env_export/nvidia_smi.txt
team7_env_export/nvcc_version.txt
```
requirements_freeze.txt records the exact ETH cluster environment. It should be treated as a reference record, not as the default installation file, because it may contain cluster-local paths, CUDA-specific wheels, and platform-specific package versions.

requirements_local_success.txt records one successfully installed local environment after the portable, Git-based, CUDA, and optional packages were installed.


## External Models and Runtime Assets

Large model weights, HuggingFace cache folders, local source trees, checkpoints,
raw data, and previous output runs are not included in this repository. They
must be restored separately on the target machine.

The external model and checkpoint records are documented in:
```text
team7_env_export/model_and_sam3_manifest.txt
team7_env_export/da3_qwen_manifest.txt
team7_env_export/hf_model_snapshots.txt
```
## External Components Setup

### Required external components

The pipeline depends on three kinds of external resources:

1. **HuggingFace model snapshots**: downloaded into a local HuggingFace cache.
2. **Cluster-restored external code/checkpoints**: copied from the ETH cluster for reproducibility.
3. **Standard Python packages**: installed through `conda` or `pip`.

The main external components used by the pipeline are:

| Component | Used for | Restore source |
|---|---|---|
| Qwen-VL | Interaction reasoning, functional element/object identification, functional relation prediction, object-state inference, seed-frame selection, and label-candidate reasoning. | HuggingFace |
| Depth Anything 3 (DA3) | Metric depth estimation when `--use-da3` is enabled. | HuggingFace |
| SAM3 model weights | Open-vocabulary video mask propagation and object/element segmentation. | HuggingFace |
| SAM3 source code | Local SAM3 Python package required by the project code. | ETH cluster or official SAM3 repository |
| ArtiPoint | Articulated point tracking and articulation fitting. | ETH cluster or official ArtiPoint repository |
| LostFound / hand-object detector | Hand-object contact detection and automatic active/inactive clip segmentation. Required for reproducing the full automatic pipeline from the extracted RGB-D sequence. | ETH cluster |
| CoTracker3 checkpoints | 2D point tracking in rigid and articulated motion estimation. | ETH cluster |
| GTSAM | Factor-graph fitting in articulated motion estimation. | Python package |
| Open3D | Point-cloud processing and PLY export. | Python package |

The external source folders, model caches, and checkpoints should **not** be committed to Git. They should be restored locally under an untracked `external/` directory.

---

### HuggingFace models

The HuggingFace models used in the final setup are:

- Qwen-VL: `Qwen/Qwen3-VL-4B-Instruct`
- Depth Anything 3: `depth-anything/DA3NESTED-GIANT-LARGE-1.1`
- SAM3 model/checkpoint cache: `facebook/sam3`

The Qwen and DA3 snapshot hashes used on the ETH cluster are recorded in:

```text
team7_env_export/hf_model_snapshots.txt
```

For stricter reproduction, the relevant snapshot hashes are:

```text
Qwen/Qwen3-VL-4B-Instruct:
ebb281ec70b05090aa6165b016eac8ec08e71b17

depth-anything/DA3NESTED-GIANT-LARGE-1.1:
b2359bdf726fb44ef62acca04d629dcf158053e7
```

---

### Recommended local layout

On a new machine, restore all external files into a local, untracked `external/` directory:

```text
ETH_3DVision_Team7/
├── final_project/
├── team7_env_export/
├── local_external_env.sh
└── external/
    ├── hf_cache/
    │   └── hub/
    ├── src/
    │   ├── sam3/
    │   ├── artipoint/
    │   └── lostfound/
    └── checkpoints/
        └── cotracker3/
            ├── scaled_online.pth
            └── scaled_offline.pth
```

---

### Restoring HuggingFace models

Create a local HuggingFace cache:

```bash
cd ~/ETH_3DVision_Team7
mkdir -p external/hf_cache

export HF_HOME=$PWD/external/hf_cache
export HF_HUB_CACHE=$HF_HOME/hub
```

If HuggingFace authentication is needed:

```bash
hf auth login
```

Download Qwen-VL, DA3, and SAM3:

```bash
python - <<'PY'
from huggingface_hub import snapshot_download

models = [
    "Qwen/Qwen3-VL-4B-Instruct",
    "depth-anything/DA3NESTED-GIANT-LARGE-1.1",
    "facebook/sam3",
]

for model_id in models:
    print(f"Downloading {model_id} ...")
    path = snapshot_download(repo_id=model_id)
    print(f"{model_id} -> {path}")
PY
```

`facebook/sam3` is a gated HuggingFace repository. If the target account does not have access, copy the pre-downloaded HuggingFace cache folder from the ETH cluster or from another machine where access has already been granted.

For example:

```bash
cd ~/ETH_3DVision_Team7
mkdir -p external/hf_cache/hub

rsync -av --progress \
  ETH_USER@student-cluster2.inf.ethz.ch:/work/courses/3dv/team7/.cache/huggingface/hub/models--facebook--sam3 \
  external/hf_cache/hub/
```

Similarly, Qwen and DA3 can either be downloaded directly from HuggingFace or copied from the cluster cache:

```bash
rsync -av --progress \
  ETH_USER@student-cluster2.inf.ethz.ch:/work/courses/3dv/team7/.cache/huggingface/hub/models--Qwen--Qwen3-VL-4B-Instruct \
  external/hf_cache/hub/

rsync -av --progress \
  ETH_USER@student-cluster2.inf.ethz.ch:/work/courses/3dv/team7/.cache/huggingface/hub/models--depth-anything--DA3NESTED-GIANT-LARGE-1.1 \
  external/hf_cache/hub/
```

After all HuggingFace models have been restored, offline/cache-only execution can be enabled with:

```bash
export HF_HOME=$PWD/external/hf_cache
export HF_HUB_CACHE=$HF_HOME/hub
export HF_HUB_OFFLINE=1
```

---

### Verifying Qwen and DA3

Check the local Qwen snapshot:

```bash
cd ~/ETH_3DVision_Team7

QWEN_DIR=$(find "$PWD/external/hf_cache/hub" -type d \
  -path "*models--Qwen--Qwen3-VL-4B-Instruct/snapshots/*" | head -n 1)

python - <<PY
from transformers import AutoConfig, AutoProcessor

model_path = "$QWEN_DIR"

config = AutoConfig.from_pretrained(
    model_path,
    local_files_only=True,
    trust_remote_code=True,
)

processor = AutoProcessor.from_pretrained(
    model_path,
    local_files_only=True,
    trust_remote_code=True,
)

print("Qwen config loaded:", config.model_type)
print("Qwen processor loaded")
PY
```

Check the local DA3 snapshot:

```bash
DA3_DIR=$(find "$PWD/external/hf_cache/hub" -type d \
  -path "*models--depth-anything--DA3NESTED-GIANT-LARGE-1.1/snapshots/*" | head -n 1)

python - <<PY
from pathlib import Path

da3_dir = Path("$DA3_DIR")
print("DA3 dir:", da3_dir)
print("exists:", da3_dir.exists())
print("files:", [p.name for p in da3_dir.iterdir()])
PY
```

The DA3 snapshot should contain files such as:

```text
config.json
model.safetensors
```

---

### Restoring SAM3 source

SAM3 has two parts:

1. the `facebook/sam3` HuggingFace model/cache, restored above;
2. the local SAM3 source folder, restored here.

For reproducibility, copy the SAM3 source used on the ETH cluster:

```bash
cd ~/ETH_3DVision_Team7
mkdir -p external/src

rsync -av --progress \
  ETH_USER@student-cluster2.inf.ethz.ch:/work/courses/3dv/team7/sam3/ \
  external/src/sam3/
```

Install SAM3 into the active environment:

```bash
conda activate team7_3dv
pip install -e external/src/sam3
```

For compatibility with the project code, create a symlink at the repository root:

```bash
cd ~/ETH_3DVision_Team7
ln -sfn external/src/sam3 sam3
```

Verify SAM3:

```bash
python - <<'PY'
import sam3
from sam3.model_builder import build_sam3_video_predictor

print("sam3:", sam3.__file__)
print("SAM3 video predictor builder: OK")
PY
```

---

### Restoring ArtiPoint

ArtiPoint is used for articulated point tracking and articulation fitting. It is **not** a HuggingFace model. For reproducibility, restore the same ArtiPoint source used on the ETH cluster.

The exact cluster path may depend on the final exported setup. If needed, first locate it on the cluster:

```bash
ssh ETH_USER@student-cluster2.inf.ethz.ch \
  'find /work/courses/3dv/team7 -maxdepth 5 -iname "*artipoint*"'
```

Then copy it locally. For example, if the cluster copy is under `temp/thirdparty/artipoint/`:

```bash
cd ~/ETH_3DVision_Team7
mkdir -p external/src

rsync -av --progress \
  ETH_USER@student-cluster2.inf.ethz.ch:/work/courses/3dv/team7/temp/thirdparty/artipoint/ \
  external/src/artipoint/
```

Install it into the active environment if it contains a Python package setup:

```bash
conda activate team7_3dv
pip install -e external/src/artipoint
```

If the project code expects ArtiPoint under `final_project/thirdparty/`, expose it with a symlink:

```bash
cd ~/ETH_3DVision_Team7
mkdir -p final_project/thirdparty
ln -sfn ../../external/src/artipoint final_project/thirdparty/artipoint
```

Verify that ArtiPoint can be imported:

```bash
python - <<'PY'
from pathlib import Path

root = Path.home() / "ETH_3DVision_Team7" / "external" / "src" / "artipoint"
print("ArtiPoint source exists:", root.exists())
print("ArtiPoint source:", root)
PY
```

---

### Restoring CoTracker3 checkpoints

Copy the CoTracker3 checkpoints from the ETH cluster:

```bash
cd ~/ETH_3DVision_Team7
mkdir -p external/checkpoints/cotracker3

rsync -av --progress \
  ETH_USER@student-cluster2.inf.ethz.ch:/work/courses/3dv/team7/checkpoints/cotracker3/ \
  external/checkpoints/cotracker3/
```

Expected files:

```text
external/checkpoints/cotracker3/scaled_online.pth
external/checkpoints/cotracker3/scaled_offline.pth
```

---

### Restoring LostFound

For full automatic clip segmentation, restore LostFound as a local source folder:

```bash
cd ~/ETH_3DVision_Team7
mkdir -p external/src

rsync -av --progress \
  ETH_USER@student-cluster2.inf.ethz.ch:/work/courses/3dv/team7/temp/thirdparty/lostfound/ \
  external/src/lostfound/
```

Expose it to the project code:

```bash
cd ~/ETH_3DVision_Team7
mkdir -p final_project/thirdparty
ln -sfn ../../external/src/lostfound final_project/thirdparty/lostfound
```

LostFound also requires `projectaria-tools`. If it is not installed yet:

```bash
pip install projectaria-tools==2.1.1
```

Verify that LostFound is available:

```bash
cd ~/ETH_3DVision_Team7/final_project

python - <<'PY'
import graph

print("LOSTFOUND_AVAILABLE:", graph.LOSTFOUND_AVAILABLE)
print("hand_object_detection:", graph.hand_object_detection is not None)
print("load_faster_rcnn:", graph.load_faster_rcnn is not None)
PY
```

Expected output:

```text
LOSTFOUND_AVAILABLE: True
hand_object_detection: True
load_faster_rcnn: True
```

---

### Local external environment script

For convenience, create a local setup script:

```bash
cd ~/ETH_3DVision_Team7

cat > local_external_env.sh <<'EOF'
export PROJECT_ROOT=$HOME/ETH_3DVision_Team7

# HuggingFace cache
export HF_HOME=$PROJECT_ROOT/external/hf_cache
export HF_HUB_CACHE=$HF_HOME/hub
export HF_HUB_OFFLINE=1

# CoTracker3 checkpoints
export COTRACKER3_CHECKPOINT_DIR=$PROJECT_ROOT/external/checkpoints/cotracker3
export COTRACKER3_OFFLINE_CHECKPOINT_PATH=$COTRACKER3_CHECKPOINT_DIR/scaled_offline.pth
export COTRACKER3_ONLINE_CHECKPOINT_PATH=$COTRACKER3_CHECKPOINT_DIR/scaled_online.pth

# Local third-party source folders
export SAM3_ROOT=$PROJECT_ROOT/external/src/sam3
export ARTIPOINT_ROOT=$PROJECT_ROOT/external/src/artipoint
export LOSTFOUND_ROOT=$PROJECT_ROOT/external/src/lostfound

# Make local source folders visible to Python if needed
export PYTHONPATH=$SAM3_ROOT:$ARTIPOINT_ROOT:$LOSTFOUND_ROOT:$PYTHONPATH
EOF
```

Use it before running the pipeline:

```bash
cd ~/ETH_3DVision_Team7/final_project
source ../local_external_env.sh
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

We have prepared the shell script for you to run the codes. 

```bash
bash ETH_3DVision_Team7/graph.sh
```
If you run on the cluster : 
Example:
```bash
sbatch graph.sh
```
You can try different starting frame and ending frame by changing the **frame-start** and **frame-end**. 

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
python eval.py \
  --output_dir output/output_final_183_long \
  --groundtruth_dir final_project/annotations/183/bowl \
  --label "bowl"
```
which will calculate the metrics for the trajectory of the bowl.

The other options for our evaluation are `spoon`(using scene 307), `cabinet door`, `drawer`.

`output_dir` should be a directory of the output of our pipeline

`groundtruth_dir` should be a directory containing ground truth annotations

`label` should be the label of the node for which the metrics are ran



## Visualization

`viser_io.py` provides a way to visualise the functional scene graph. It can be ran using:

```bash
python3 viser_io.py \
  --output-dir output/output_final_183_long \
  --host 127.0.0.1 \
  --port 8003 \
  --groundtruth-dir annotations/183/bowl \
  --groundtruth-object "bowl"
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

python visualize_scene_graph_html_multi_status.py \
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


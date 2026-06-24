# ======================================================================
# graph.py section: 00_imports_config.py
# Module docstring, imports, SAM-backend helpers, dataclasses, global constants, depth-scale helpers
#
# NOTE: not a standalone module. Loaded by graph.py into a shared
# namespace via exec(); names from earlier sections are in scope here.
# ======================================================================
"""Main dynamic functional 3D scene graph pipeline for egocentric RGB-D video."""

from __future__ import annotations

import argparse
import base64
from bisect import bisect_left
import copy
import gc
import getpass
import hashlib
import io
import json
import os
import re
import importlib
import shutil
import sys
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
import socket
from typing import Any, Dict, List, Optional, Sequence, Tuple

from scipy.spatial import KDTree
from scipy.interpolate import interp1d
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation, Slerp
from sklearn.cluster import DBSCAN
try:
    from sklearn.cluster import HDBSCAN as _HDBSCAN  # sklearn >= 1.3
    HDBSCAN_AVAILABLE = True
except Exception:  # noqa: BLE001
    _HDBSCAN = None
    HDBSCAN_AVAILABLE = False
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

# Optional GTSAM dependency for the ArtiPoint Eq.(4) factor-graph fit.
# Verified present in venv (Pose3, Values, NonlinearFactorGraph,
# LevenbergMarquardtOptimizer, noiseModel.Robust + Huber). If the import fails
# we transparently fall back to the scipy soft_l1 path.
try:
    import gtsam as _gtsam
    GTSAM_AVAILABLE = True
except Exception as _gtsam_import_error:  # noqa: BLE001
    _gtsam = None
    GTSAM_AVAILABLE = False
    print(f"[Warn] gtsam unavailable, falling back to scipy soft_l1 fit: {_gtsam_import_error}")

import cv2
import numpy as np
import open3d as o3d
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch
from PIL import Image
from depth_anything_3.api import DepthAnything3
from qwen_vl_utils import process_vision_info
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoProcessor,
    Qwen3VLForConditionalGeneration,
)

from dynamic_scene_graph import DynamicSceneGraph
from visualize_graph import save_graph_visualization, save_openfungraph_visualization
from json_utils import safe_json_loads

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_SAM_BACKEND = os.environ.get("GRAPH_SAM_BACKEND", "sam3").strip().lower()


def _resolve_sam3_package_root() -> Path:
    """Locate the plain-SAM3 package root (the dir whose ``sam3/`` subpackage has
    ``model_builder.py``). The original monolith lived at the repo top level, so
    ``PROJECT_ROOT/sam3`` was correct; after moving graph.py into ``temp/`` the
    package now sits one level up (``temp/../sam3``). Check both so the backend
    works regardless of where graph.py is run from."""
    for candidate in (PROJECT_ROOT / "sam3", PROJECT_ROOT.parent / "sam3"):
        if (candidate / "sam3" / "model_builder.py").exists():
            return candidate
    return PROJECT_ROOT / "sam3"


DEFAULT_SAM3_PACKAGE_ROOT = _resolve_sam3_package_root()
DEFAULT_SAM3I_PACKAGE_ROOT = PROJECT_ROOT / "SAM3-I" / "sam3"
DEFAULT_SAM3I_CHECKPOINT_PATH = PROJECT_ROOT / "SAM3-I" / "checkpoint.pt"
SAM3_BPE_RELATIVE_PATH = Path("sam3") / "assets" / "bpe_simple_vocab_16e6.txt.gz"
_SAM_BACKEND_IMPORT_CACHE: Dict[Tuple[str, str], Tuple[Path, Any]] = {}
_ACTIVE_PIPELINE_SAM_CONFIG: Optional["PipelineConfig"] = None


def set_active_pipeline_sam_config(config: Optional["PipelineConfig"]) -> None:
    """Expose the current pipeline SAM config to lazy helper-side predictor builds."""
    global _ACTIVE_PIPELINE_SAM_CONFIG
    _ACTIVE_PIPELINE_SAM_CONFIG = config


def _normalize_sam_backend_name(name: object) -> str:
    backend = str(name or "sam3").strip().lower()
    if backend not in {"sam3", "sam3i"}:
        raise ValueError(f"Unsupported SAM backend '{backend}'. Expected one of: sam3, sam3i")
    return backend


def _sam_backend_package_root(backend: str, sam3i_package_root: Optional[Path] = None) -> Path:
    backend = _normalize_sam_backend_name(backend)
    if backend == "sam3i":
        return Path(sam3i_package_root or DEFAULT_SAM3I_PACKAGE_ROOT)
    return Path(DEFAULT_SAM3_PACKAGE_ROOT)


def _sam_backend_bpe_path(package_root: Path) -> Optional[Path]:
    bpe_path = Path(package_root) / SAM3_BPE_RELATIVE_PATH
    return bpe_path if bpe_path.exists() else None


def _import_sam_backend_builder(backend: str, package_root: Path):
    cache_key = (_normalize_sam_backend_name(backend), str(package_root.resolve()))
    cached = _SAM_BACKEND_IMPORT_CACHE.get(cache_key)
    if cached is not None:
        return cached
    package_root_str = str(package_root)
    if package_root_str in sys.path:
        sys.path.remove(package_root_str)
    sys.path.insert(0, package_root_str)
    for module_name in list(sys.modules):
        if module_name == "sam3" or module_name.startswith("sam3."):
            del sys.modules[module_name]
    sam3_pkg = importlib.import_module("sam3")
    if getattr(sam3_pkg, "__file__", None) is None:
        sam3_pkg.__file__ = str(package_root / "sam3" / "__init__.py")
    model_builder = importlib.import_module("sam3.model_builder")
    builder = getattr(model_builder, "build_sam3_video_predictor")
    _SAM_BACKEND_IMPORT_CACHE[cache_key] = (package_root, builder)
    return package_root, builder


def build_configured_sam3_video_predictor(config: Optional["PipelineConfig"] = None):
    if config is None:
        config = _ACTIVE_PIPELINE_SAM_CONFIG
    backend = _normalize_sam_backend_name(getattr(config, "sam_backend", DEFAULT_SAM_BACKEND))
    sam3i_package_root = getattr(config, "sam3i_package_root", DEFAULT_SAM3I_PACKAGE_ROOT)
    package_root = _sam_backend_package_root(backend, sam3i_package_root)
    package_root, builder = _import_sam_backend_builder(backend, package_root)
    kwargs: Dict[str, Any] = {}
    bpe_path = _sam_backend_bpe_path(package_root)
    if bpe_path is not None:
        kwargs["bpe_path"] = str(bpe_path)
    if backend == "sam3i":
        checkpoint_path = Path(getattr(config, "sam3i_checkpoint_path", DEFAULT_SAM3I_CHECKPOINT_PATH))
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"SAM3-I backend requested but checkpoint not found at {checkpoint_path}. "
                "Download the stage-3 checkpoint and point --sam3i-checkpoint-path to it."
            )
        # NOTE: Sam3VideoPredictor.__init__ does not take `load_from_HF` (only the
        # image-model builder does). With checkpoint_path set, the video model loads
        # the local checkpoint directly (load_from_HF is irrelevant), so we omit it.
        kwargs.update(
            checkpoint_path=str(checkpoint_path),
            inst_stage=str(getattr(config, "sam3i_inst_stage", "1_2")),
            adapter_config={
                "adapter_dim": int(getattr(config, "sam3i_adapter_dim", 64)),
                "adapter_heads": int(getattr(config, "sam3i_adapter_heads", 4)),
                "adapter_scale": float(getattr(config, "sam3i_adapter_scale", 1.0)),
            },
        )
    return builder(**kwargs)


os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

SEM_VOXEL_SIZE = 0.01
DEPTH_SCALE = 1000.0
CURRENT_DEPTH_SCALE = DEPTH_SCALE
DEPTH_MAX = 3.0
DA3_PROCESS_RES = 504
DA3_CHUNK_SIZE = 25
HAND_PROMPTS = ["human hand", "human arm", "person"]
MIN_CLIP_FRAMES = 5
DETECTION_THRESH = 0.5
FULL_FRAME_QWEN_SIZE = (960, 540)
QWEN_IMAGE_MAX_PIXELS = 360_448
QWEN_VIDEO_MAX_PIXELS = 301_056
QWEN_ACTIVE_VIDEO_FRAME_STRIDE = 5
QWEN_ACTIVE_VIDEO_FPS = 5
# Cap on frames in the Qwen active-clip detection video. Reverted to 48 now that
# SAM3 is built once and kept resident (no per-clip rebuild fragmentation), so the
# Qwen baseline no longer creeps up and the detection forward has its old headroom.
# Env-tunable if a pathologically long clip ever needs trimming.
QWEN_ACTIVE_VIDEO_MAX_FRAMES = int(os.environ.get("QWEN_ACTIVE_VIDEO_MAX_FRAMES", "48"))
# Ablation: whether to show Qwen the inactive "before" context frames in detection
# (the rest-state images that help it tell the moving PART from its static parent,
# e.g. "cabinet door" vs "cabinet"). Set 0 to run the no-context ablation arm:
# Qwen then sees ONLY the active interaction frames.
QWEN_CONTEXT_FRAMES = os.environ.get("QWEN_CONTEXT_FRAMES", "1") == "1"
SAM3_MAX_IMAGE_SIDE = int(os.environ.get("SAM3_MAX_SIDE", "0"))
# When False (default), the debug-image writers — binary mask PNG sequences,
# SAM3 overlay PNGs, LostFound failure overlays — are no-ops. Set GRAPH_DEBUG_IMAGES=1
# to re-enable them when iterating on mask quality or tracker failures.
DEBUG_IMAGES_ENABLED = bool(int(os.environ.get("GRAPH_DEBUG_IMAGES", "0")))
CAPTION_MAX_VIEWS = 4
# Must be >= 2 * CoTracker step or the LostFound bootstrap loop fires
# `lostfound_process_step` only once. That one call has ``is_first_step=True``,
# which the online CoTracker returns ``(None, None)`` from (it's just the init
# call). With CoTracker3's window_len=16 → step=8, this requires >=16.
# Original LostFound used CoTracker2 (window_len=8, step=4) so LOOKAHEAD=8 was
# enough; the implicit invariant broke when we upgraded the tracker.
LOOKAHEAD_FRAMES = 16
HAND_HISTORY_SIZE = 10
HAND_DISTANCE_THRESHOLD = 0.10
# Require at least modest wrist travel before a contact/near-object frame can
# start an active interaction. Filters prolonged static hand-resting on a
# handle/counter while still allowing real manipulations to trigger once the
# wrist actually begins moving.
INTERACTION_WRIST_TRAVEL_THRESHOLD = 0.025
# Activation must sit inside a SUSTAINED contact run, not a spurious isolated
# 1-frame touch. Without this, a single stray contact frame whose 16-frame
# lookahead happens to see the later real grasp (e.g. cabinet frame 289, isolated,
# with sustained contact only from 301) latches the state machine on far too early
# and swallows the whole inactive prefix (0 inactive clips -> no inactive prior).
# Require a majority of the immediate next IMMEDIATE_CONTACT_LOOKAHEAD frames to
# also be in contact before activating.
IMMEDIATE_CONTACT_LOOKAHEAD = 4
IMMEDIATE_CONTACT_MIN = 2
HAND_VELOCITY_DIFF_THRESHOLD = 0.025
LOSTFOUND_HAND_DISTANCE_THRESHOLD = 0.25
# Activation distance for LostFound pose tracking. Tighter than the global
# HAND_DISTANCE_THRESHOLD so we don't activate at every passing approach, but
# loose enough to fire on the actual grasp moment. Calibrated against three
# observed failure modes:
#   - spoon prior on counter, wrist min 13.8 cm from prior
#   - 183 drawer handle, wrist min 17.8 cm from prior
#   - 183 cabinet handle, wrist min 17.1 cm from prior
# 20 cm is just above all observed minima; any lower and case-2 tracking never
# starts. The `future_far` guard inside the activation loop still filters out
# pre-grasp approach frames by requiring the wrist's next 8 positions to be
# farther from the prior than the current one (i.e., we're at the local min
# of the approach), so this looser threshold doesn't re-introduce the
# door-follows-wrist drag we fixed earlier — and ArtiPoint (case 3) doesn't
# go through this gate at all.
LOSTFOUND_ACTIVATION_DISTANCE = 0.20
LOSTFOUND_BASE_THRESHOLD = 3
LOSTFOUND_HIGH_THRESHOLD_BONUS = 2
SMOOTH_FALLBACK_RADIUS = 3
# Hybrid active-onset rule inspired by LostFound plus our articulated-mask needs:
# start either at motion-gated closest-approach OR after a sustained grasp dwell
# on the object. The dwell fallback rescues cabinet-like clips where the hand is
# already stably grasping the handle before visible motion begins.
ACTIVE_CLIP_GRASP_DWELL_MIN = 5
# Keep a small lead-in once activation fires, but shorter than before so drawer
# does not regain a long static pre-motion prefix.
ACTIVE_CLIP_PREROLL_FRAMES = 0
# Forward-extend the active clip past the last strict contact trigger while the
# hand stream is still present. Covers the tail of an interaction where the RCNN
# hand/object detector drops out (hand grips a handle near the image edge and
# stops being detected) but the manipulation is still ongoing — e.g. the 183
# drawer, which keeps sliding fully open across ~15 frames that were wrongly
# marked inactive (hand_ratio 0.067). Stops at clip/pose-block boundaries.
ACTIVE_CLIP_POSTROLL_FRAMES = int(os.environ.get("ACTIVE_CLIP_POSTROLL_FRAMES", "0"))
# Bridge SHORT contact dropouts inside a single continuous interaction. When the
# hand+object briefly leave detection mid-manipulation (e.g. the carried bowl is
# occluded for ~13 frames before reappearing in the same carry), the active flag
# goes False for that hole and the clip is wrongly split in two. Fill False-holes
# up to this many frames that are flanked by active frames on BOTH sides, so the
# carry stays ONE clip. Kept below a real inter-interaction gap (the 183 bowl has
# a 13-frame dropout to bridge and a 30-frame true gap to preserve). Stops at
# clip/pose-block boundaries (never bridges across a frame/pose discontinuity).
ACTIVE_CLIP_BRIDGE_GAP_FRAMES = int(os.environ.get("ACTIVE_CLIP_BRIDGE_GAP_FRAMES", "15"))
# When deriving the active object SAM3 box from the interactive-element (handle)
# mask, expand the element bbox by this factor (about its center) so the box
# covers the part of the moving body around the handle without spilling onto the
# whole cabinet/drawer-front row.
OBJECT_ANCHOR_FROM_ELEMENT_SCALE = 5.0
OBJECT_ANCHOR_FROM_ELEMENT_PADDING = 24
GLOBAL_COMPACT_CHUNKS = 32
GLOBAL_COMPACT_POINT_THRESHOLD = 6_000_000
INACTIVE_DEPTH_PIXEL_STRIDE = 2
# Adaptive stride floor: we always lift at least this many SAM3 frames from the
# inactive clip. The configured `inactive_stride` is shrunk automatically
# (coverage_cap = n_inactive_frames // MIN_INACTIVE_PRIOR_FRAMES) when honouring it
# would leave fewer than this many frames — short inactive clips therefore collapse
# toward stride 1 and use (nearly) all available frames.
# Raised 4 -> 8 (2026-06-09): with only 5 frames the 183 drawer's small 'drawer
# handle' element (~2k px mask) caught none of the clip's sparse pre-opening depth
# and lifted to 0 world points, aborting the active clip. More inactive frames give
# small/thin elements far more chances to hit valid depth for the prior cloud.
MIN_INACTIVE_PRIOR_FRAMES = 8
INITIAL_GEOMETRY_MAX_FRAMES = 1
FINAL_GEOMETRY_MAX_FRAMES = 1
SUPPRESSION_BOX_MARGIN = 0.015
MIN_TRACKED_POINTS_FOR_REUSE = 64
LOSTFOUND_TRACK_MIN_POINTS = 96
LOSTFOUND_TRACK_MAX_POINTS = 256
LOSTFOUND_TRACK_CANDIDATE_POINTS = 1024
LOSTFOUND_TRACK_HAND_RADIUS = 0.18
LOSTFOUND_MIN_MOTION = 0.03
LOSTFOUND_MIN_UPDATES = 2
# Original LostFound (with CoTracker2) defaulted this to 64 because CoTracker2's
# visibility scores were inflated; ~50% of tracked points routinely cleared the
# threshold. CoTracker3 is more honest about uncertainty and emits lower
# visibility on the same input, so a 64 floor becomes unreachable on small/
# handheld/occluded objects (the spoon clip had only 17-23 visible per frame).
# 32 is a balanced compromise: stricter than the temporary 16 floor so PnP
# RANSAC has more correspondences to vote with (less noisy poses on clean
# clips), while still reachable on harder spoon-like clips at vis_threshold
# 0.80.
LOSTFOUND_POSE_MIN_VISIBLE_POINTS = 32
LOSTFOUND_ROTATION_SLERP_ALPHA = 0.4
# Rigid-transport (LostFound) PnP rotation floor: minimum number of VISIBLE tracked
# corners needed to solve PnP for this frame's rotation. LostFound solves PnP on
# whatever points are visible (no hard floor), so a low value keeps a fresh rotation
# almost every frame; carry-forward only fills the rare fully-occluded frame.
RIGID_PNP_MIN_VISIBLE_POINTS = int(os.environ.get("RIGID_PNP_MIN_VISIBLE_POINTS", "6"))
# Re-seed corners on EVERY active frame for the rigid-transport path too (not just
# the first), mirroring the articulated path. The single-seed PnP model froze the
# rotation once the frame-0 corners died (e.g. the carried bowl rotated/occluded
# and no new points were ever added -> orientation stuck at ~4 deg while the hand
# kept translating it). With re-seeding, rotation is driven by INCREMENTAL
# consecutive-frame Kabsch on the world-lifted tracks (robust to corner turnover),
# while translation stays hand-anchored. Set 0 to restore single-seed PnP.
RIGID_RESEED_EVERY_FRAME = os.environ.get("RIGID_RESEED_EVERY_FRAME", "1") == "1"
# Minimum corner pairs shared by two consecutive frames to accept an incremental
# Kabsch rotation step; below this the rotation is carried forward one frame.
RIGID_KABSCH_MIN_PAIRS = int(os.environ.get("RIGID_KABSCH_MIN_PAIRS", "6"))
ARTIPOINT_TRACK_MAX_POINTS = 256
# Active object-mask corner tracking for articulated parts:
# every active frame can contribute up to this many Shi-Tomasi corners from the
# SAM3 object mask, with a global cap before CoTracker3 launch. Raised from 30 to 80
# for denser tracking — small/occluded objects like the drawer otherwise yield too
# few keypoints (sparse, jumpy tracks).
ARTIPOINT_MAX_CORNERS_PER_KEYFRAME = 80
ARTIPOINT_MAX_TOTAL_QUERIES = 2500
# Pixels to dilate the per-frame hand mask before subtracting it from the
# object/handle corner-seeding masks. ArtiPoint must track the moving PART only,
# never the gripping hand (textured skin attracts goodFeaturesToTrack off the
# textureless door). A few-pixel buffer also removes corners on the hand/part
# boundary, which otherwise jitter between hand and part.
ARTIPOINT_HAND_EXCLUSION_DILATE = int(os.environ.get("ARTIPOINT_HAND_EXCLUSION_DILATE", "15"))
# Use ArtiPoint's motion-bisector pivot for the revolute hinge point
# (point_bisector_free_traj = _estimate_motion_point_bisectors), rather than
# picking whichever pivot candidate minimises translation RMSE. This matches the
# ArtiPoint frontend (perpendicular bisectors of the per-frame motion translations
# intersect at the rotation centre). Falls back to the RMSE-best candidate if the
# bisector pivot is unavailable. Env-tunable.
ARTIPOINT_PREFER_BISECTOR_PIVOT = os.environ.get("ARTIPOINT_PREFER_BISECTOR_PIVOT", "1") == "1"
# CoTracker3 offline processes the WHOLE active clip in one forward pass, so a
# long high-res clip (e.g. 151 frames @ 1920x1080 = 313 MPix) OOMs the 16 GB GPU
# on activations (observed: spoon clip, tried to alloc 2.88 GiB with 2.27 free).
# Cap the total tracked pixel volume (frames x H x W of the SCALED video). Above
# this we downscale the video isotropically for tracking only and rescale the 2D
# tracks back to full resolution before lifting/PnP (K_color stays full-res).
# Short clips stay native. Tune via env for tighter/looser GPU budgets.
ARTIPOINT_COTRACKER_MAX_TRACK_PIXELS = int(
    os.environ.get("ARTIPOINT_COTRACKER_MAX_TRACK_PIXELS", str(45_000_000))
)
# For the rigid_pnp path (free carried objects), a long high-res active clip OOMs
# CoTracker3 offline. Instead of downscaling (which the user disabled), we keep
# FULL resolution and temporally STRIDE the clip for tracking — PnP is per-frame,
# so a rigid 6-DoF pose just gets fewer (still dense enough) samples. The stride is
# the LARGEST stride that still leaves AT LEAST this many frames, so CoTracker
# always has enough frames to track reliably. Articulated clips are untouched.
RIGID_PNP_TRACK_MIN_FRAMES = int(os.environ.get("RIGID_PNP_TRACK_MIN_FRAMES", "50"))
# The inactive prior is built by propagating the single active seed BACKWARD
# through the inactive frames. On frames far from the seed (earliest inactive),
# propagation can drift onto a neighbouring near-identical instance (e.g. BOTH
# cabinet doors), ballooning the object mask. Walking back from the boundary
# (frame nearest the active seed), the first frame whose object-mask area exceeds
# this ratio of the boundary-frame area is treated as the drift point: that frame
# and ALL earlier ones are dropped from the prior. Set <=0 to disable.
INACTIVE_PRIOR_DRIFT_AREA_RATIO = float(os.environ.get("INACTIVE_PRIOR_DRIFT_AREA_RATIO", "1.8"))
# Minimum inactive-prior points required to ENTER the direct ArtiPoint path.
# This only gates entry — the actual CoTracker queries are seeded from the
# ACTIVE object-mask Shi-Tomasi corners (hundreds per clip), so a sparse
# inactive reconstruction (e.g. a drawer whose closed-state body lifts to few
# points) should not block tracking when the active masks are healthy. Kept at
# the same floor as ARTIPOINT_MIN_VISIBLE_POINTS so entry and the post-seed
# corner check agree.
ARTIPOINT_MIN_TRACK_POINTS = 12
ARTIPOINT_MIN_VISIBLE_POINTS = 12
ARTIPOINT_MIN_POSE_UPDATES = 3
ARTIPOINT_RANSAC_ITERS = 96
ARTIPOINT_RANSAC_INLIER_M = 0.04
ARTIPOINT_TRANSLATION_RESIDUAL_SCALE_M = 0.05
# Joint-optimization regularisers. Per-frame θ_m otherwise wanders to whatever
# fits a noisy per-frame surface-track pose (sometimes 2 rad jumps between
# adjacent frames), turning the reconstructed centroid trajectory into a
# zigzag. Smoothness + jerk + monotonicity penalties keep the swing coherent.
ARTIPOINT_DOOR_AXIS_PRIOR_WEIGHT = 3.0
# Theta smoothness penalties — kept (we use cumulative θ semantic; ArtiPoint
# uses per-frame increments with a weak PriorFactorDouble(0, σ=25°) instead).
# Weights tuned down from (5.0, 2.0, 1.5) to match ArtiPoint's overall
# regularization strength: their theta prior is essentially noise-floor only.
ARTIPOINT_THETA_VELOCITY_WEIGHT = 0.5
ARTIPOINT_THETA_JERK_WEIGHT = 0.5
ARTIPOINT_THETA_MONOTONIC_WEIGHT = 0.5
ARTIPOINT_POSE_RMSE_OUTLIER_MULT = 3.0
ARTIPOINT_POSE_RMSE_OUTLIER_MIN_M = 0.08
ARTIPOINT_NATIVE_MAX_MEAN_RESIDUAL_M = 0.10
# Continuity gate: a per-frame Procrustes solve on a near-symmetric door cloud
# can land on the 180°-flipped twin of the correct rotation. Adjacent frames
# normally rotate <30° per processed step at our framerates, so any |Δrotation|
# above π/2 between consecutive accepted frames is taken as a flipped-twin and
# the offending frame is dropped before the GTSAM articulation fit. Without this
# the (I-R)·p least-squares averages the two candidate hinges and the axis_point
# collapses to the door centroid.
ARTIPOINT_CONTINUITY_MAX_JUMP_RAD = np.pi / 2.0
# ArtiPoint Eq. (1) — velocity + jerk regulariser on the lifted 3D point tracks
# *before* the per-frame Procrustes pose fit. These are the values reported in the paper implementation details; we
# expose them so the call site can override per clip if needed.
# Trajectory smoothing weights — matched to ArtiPoint's *actual paper config*
# (artipoint/configs/artipoint.yaml: lambda_vel=0.5, lambda_jerk=5.0). The
# function defaults in arti_estimator.py (0.1, 1.0) are NOT what the paper
# uses — they're just function-level defaults overridden by the yaml.
ARTIPOINT_TRACK_SMOOTH_LAMBDA_VEL = 0.5
ARTIPOINT_TRACK_SMOOTH_LAMBDA_JERK = 5.0
ARTIPOINT_TRACK_RELIABILITY_MIN = 0.5
ARTIPOINT_STATIC_TRACK_MIN_MOTION_M = 0.025
ARTIPOINT_SURFACE_TRACK_MAX_QUERY_PRIOR_DISTANCE_M = 0.06
# Lowered 32 -> 16: small/occluded objects (e.g. the drawer) survive segmentation with
# fewer usable frames; 32 was dropping them to zero output. 16 still needs a real cluster
# of tracks for a usable articulation fit, just less aggressively.
ARTIPOINT_MIN_FILTERED_TRACKS = 16
ARTIPOINT_FILTER_SMOOTHNESS_THRESHOLD = 0.2
ARTIPOINT_FILTER_VARIANCE_TYPE = "2d"
ARTIPOINT_FILTER_PERCENTILE = 80
ARTIPOINT_FILTER_OCCLUSION_THRESHOLD = 0.5
# Seed-relative occlusion filter: minimum lifetime (first->last visible frame)
# span a re-seeded track must cover before its visibility fraction is trusted.
# Guards against 1-2 frame blips passing at frac=1.0 once we score tracks over
# their own lifetime instead of the whole clip.
ARTIPOINT_FILTER_OCCLUSION_MIN_LIFETIME_FRAMES = int(
    os.environ.get("ARTIPOINT_FILTER_OCCLUSION_MIN_LIFETIME_FRAMES", "8")
)
# ArtiPoint-literal filtering: ArtiPoint's articulated frontend is ONLY
# non_smooth -> median -> occluded -> outlier -> smooth, then the factor-graph
# fit. We had added two non-ArtiPoint filters (an absolute static-motion floor
# and a prior-surface proximity gate) that together stripped ~half the tracks
# (e.g. 1869 -> 790 -> ... -> 35 on the cabinet door). With this flag set the
# articulated path drops BOTH extras and runs ArtiPoint's chain verbatim.
ARTIPOINT_LITERAL_FILTERS = os.environ.get("ARTIPOINT_LITERAL_FILTERS", "1") == "1"
# Occlusion-filter mode. ArtiPoint's literal filter scores visibility over the
# WHOLE clip (correct for its OFFLINE single-seed tracking). Online causal
# tracking + mask re-seeding needs either "seed_relative" (score each track over
# its own lifetime) or "off" (ArtiPoint gates this via filter_unreliable_tracks).
#   "whole_clip"    -> ArtiPoint literal (offline-faithful; kills online re-seeds)
#   "seed_relative" -> online analog (keeps re-seeds that track cleanly)
#   "off"           -> disable (ArtiPoint filter_unreliable_tracks: false)
ARTIPOINT_OCCLUSION_FILTER_MODE = os.environ.get("ARTIPOINT_OCCLUSION_FILTER_MODE", "off")
ARTIPOINT_FILTER_DBSCAN_EPS = 0.1
ARTIPOINT_FILTER_DBSCAN_MIN_SAMPLES = 5
COTRACKER3_CHECKPOINT_DIR = Path(
    os.environ.get(
        "COTRACKER3_CHECKPOINT_DIR",
        str(PROJECT_ROOT.parent / "checkpoints" / "cotracker3"),
    )
).expanduser()
COTRACKER3_OFFLINE_CHECKPOINT_PATH = Path(
    os.environ.get(
        "COTRACKER3_OFFLINE_CHECKPOINT_PATH",
        str(COTRACKER3_CHECKPOINT_DIR / "scaled_offline.pth"),
    )
).expanduser()
COTRACKER3_ONLINE_CHECKPOINT_PATH = Path(
    os.environ.get(
        "COTRACKER3_ONLINE_CHECKPOINT_PATH",
        str(COTRACKER3_CHECKPOINT_DIR / "scaled_online.pth"),
    )
).expanduser()
MASK_LIFT_TABLE_PLANE_MIN_POINTS = 120
MASK_LIFT_TABLE_PLANE_MIN_RATIO = 0.30
MASK_LIFT_TABLE_PLANE_NORMAL_Z = 0.82
MASK_LIFT_TABLE_PLANE_DISTANCE_M = 0.012
MASK_LIFT_TABLE_PLANE_MIN_REMAINING = 24
ACTIVE_MASK_HAND_KEEP_RATIO = 0.35
ACTIVE_MASK_HAND_KEEP_PIXELS = 64
SAM3_BOOTSTRAP_BOXES_FROM_TEXT = True
SAM3_BOOTSTRAP_BOX_MARGIN = 0.08
SAM3_BOOTSTRAP_BOX_MIN_PIXELS = 64
SAM3_CONFIDENCE_MIN = 0.35
SAM3_CONFIDENCE_MIN_VALID_FRAMES = 1
# Hard gate on SAM3's OWN per-frame probability (out_probs). When SAM3 reports a
# score below this, the mask for that frame is dropped (left empty) at the source,
# so a frame where the prompted object is absent/occluded yields an EMPTY mask
# instead of "random stuff" that then contaminates keypoints/stabilization. An
# empty frame is fine — downstream confidence/visibility handles gaps. Set to 0 to
# disable. Frames where SAM3 returns no score (NaN) are kept (legacy behavior).
SAM3_RAW_SCORE_MIN = float(os.environ.get("SAM3_RAW_SCORE_MIN", "0.5"))
SAM3_AREA_OUTLIER_RATIO = 3.0
SAM3_BOX_CENTER_OUTLIER_RATIO = 0.35

LOSTFOUND_AVAILABLE = False
LOSTFOUND_TRACKING_AVAILABLE = False
LFSceneGraph = None
hand_object_detection = None
load_faster_rcnn = None
CoTrackerOnlinePredictor = None
CoTracker3Predictor = None
COTRACKER3_AVAILABLE = False

# Vendored ArtiPoint factor-graph code (mirrors the LostFound layout).
# We pull in their PoseEstFactorGraph + motion-bisector helpers directly so
# the cabinet/articulated path runs ArtiPoint's *exact* solver instead of our
# reimplementation. See temp/thirdparty/artipoint/artipoint/factor_graph/pose_est.py.
ARTIPOINT_AVAILABLE = False
ArtiPointPoseEstFactorGraph = None
artipoint_estimate_motion_point_bisectors = None
artipoint_estimate_motion_axis_eigen = None
ARTIPOINT_ROOT = None
for _candidate in [
    Path(__file__).resolve().parent / "temp" / "thirdparty" / "artipoint",
    Path(__file__).resolve().parent / "thirdparty" / "artipoint",
]:
    if _candidate.exists():
        ARTIPOINT_ROOT = _candidate
        if str(_candidate) not in sys.path:
            sys.path.insert(0, str(_candidate))
        break
try:
    from artipoint.factor_graph.pose_est import PoseEstFactorGraph as ArtiPointPoseEstFactorGraph
    from artipoint.utils.articulation_helper import (
        estimate_motion_point_bisectors as artipoint_estimate_motion_point_bisectors,
        estimate_motion_axis_eigen as artipoint_estimate_motion_axis_eigen,
    )
    ARTIPOINT_AVAILABLE = True
except Exception as artipoint_import_error:
    print(f"[Warn] ArtiPoint optional import unavailable: {artipoint_import_error}")

# CoTracker resolution order:
#   1. Our local clone at temp/thirdparty/cotracker/ (preferred).
#   2. A pip-installed cotracker in the active env.
# We intentionally keep CoTracker3 as the only tracker family in use.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_OWN_COTRACKER = _PROJECT_ROOT / "temp" / "thirdparty" / "cotracker"
if _OWN_COTRACKER.exists() and str(_OWN_COTRACKER) not in sys.path:
    sys.path.insert(0, str(_OWN_COTRACKER))
try:
    from cotracker.predictor import CoTrackerPredictor as CoTracker3Predictor

    COTRACKER3_AVAILABLE = True
except Exception as cotracker3_import_error:
    print(f"[Warn] CoTracker3 optional import unavailable: {cotracker3_import_error}")
LOSTFOUND_ROOT = None
for _candidate in [
    Path(__file__).resolve().parent / "temp" / "thirdparty" / "lostfound",
    Path(__file__).resolve().parent / "LostFound",
    Path(__file__).resolve().parent / "thirdparty" / "lostfound",
]:
    if _candidate.exists():
        LOSTFOUND_ROOT = _candidate
        # NOTE: do NOT add LostFound's thirdparty/cotracker to sys.path; the
        # local CoTracker3 above must remain the canonical resolution target.
        for _path in (_candidate, _candidate / "src"):
            if _path.exists() and str(_path) not in sys.path:
                sys.path.insert(0, str(_path))
        break
try:
    from src.detection_utils import hand_object_detection, load_faster_rcnn
    from src.scene_graph import SceneGraph as LFSceneGraph
    from cotracker.predictor import CoTrackerOnlinePredictor

    LOSTFOUND_AVAILABLE = True
    LOSTFOUND_TRACKING_AVAILABLE = True
except Exception as lostfound_import_error:
    print(f"[Warn] LostFound optional import unavailable: {lostfound_import_error}")
    try:
        from cotracker.predictor import CoTrackerOnlinePredictor

        LOSTFOUND_TRACKING_AVAILABLE = True
    except Exception as cotracker_import_error:
        print(f"[Warn] CoTracker optional import unavailable: {cotracker_import_error}")


@dataclass
class ClipInfo:
    """Metadata for one contiguous inactive or active clip."""

    clip_id: int
    frame_start: int
    frame_end: int
    is_inactive: bool
    hand_ratio: float
    pose_block_id: Optional[int] = None
    interaction_hand: Optional[str] = None
    samples: List[dict] = field(default_factory=list)


@dataclass(frozen=True)
class TopicFrame:
    """One extracted dataset file with its sequence key and timestamp."""

    key: str
    timestamp_ns: int
    path: Path


class PoseBuffer:
    """Interpolated camera/depth/color pose buffer over valid camera poses."""

    def __init__(
        self,
        timestamps_s: np.ndarray,
        camera_to_world: List[Tuple[Rotation, np.ndarray]],
        depth_to_camera: Tuple[Rotation, np.ndarray],
        rgb_to_depth: Tuple[Rotation, np.ndarray],
    ) -> None:
        if len(timestamps_s) != len(camera_to_world):
            raise ValueError("PoseBuffer timestamps and poses must have the same length")
        if len(timestamps_s) == 0:
            raise ValueError("PoseBuffer requires at least one valid pose")

        self.timestamps_s = np.asarray(timestamps_s, dtype=np.float64)
        self.camera_to_world = list(camera_to_world)
        self.depth_to_camera = depth_to_camera
        self.rgb_to_depth = rgb_to_depth

        if len(self.timestamps_s) < 2:
            self._rotation_interp = None
            self._translation_interp = None
        else:
            self._rotation_interp = Slerp(
                self.timestamps_s,
                Rotation.concatenate([rotation for rotation, _ in self.camera_to_world]),
            )
            self._translation_interp = interp1d(
                self.timestamps_s,
                np.stack([translation for _, translation in self.camera_to_world], axis=0),
                axis=0,
            )

    def _clamp_time_s(self, query_time_ns: int) -> float:
        query_time_s = float(query_time_ns) * 1e-9
        return float(np.clip(query_time_s, self.timestamps_s[0], self.timestamps_s[-1]))

    def camera_pose(self, query_time_ns: int) -> Tuple[Rotation, np.ndarray]:
        """Return interpolated camera-to-world pose at the requested timestamp."""
        query_time_s = self._clamp_time_s(int(query_time_ns))
        if self._rotation_interp is None or self._translation_interp is None:
            rotation, translation = self.camera_to_world[0]
            return rotation, np.asarray(translation, dtype=np.float64)
        return self._rotation_interp(query_time_s), np.asarray(self._translation_interp(query_time_s), dtype=np.float64)

    def depth_pose(self, query_time_ns: int) -> Tuple[Rotation, np.ndarray]:
        """Return interpolated depth-to-world pose at the requested timestamp."""
        rotation, translation = self.camera_pose(query_time_ns)
        depth_rotation, depth_translation = self.depth_to_camera
        return rotation * depth_rotation, rotation.apply(depth_translation) + translation

    def rgb_pose(self, query_time_ns: int) -> Tuple[Rotation, np.ndarray]:
        """Return interpolated color-to-world pose at the requested timestamp."""
        rotation, translation = self.depth_pose(query_time_ns)
        rgb_rotation, rgb_translation = self.rgb_to_depth
        return rotation * rgb_rotation, rotation.apply(rgb_translation) + translation


@dataclass
class PipelineConfig:
    """Runtime configuration for the two-mode scene graph pipeline."""

    data_dir: Path = Path("/work/courses/3dv/team7/data/open-source-307_extracted")
    da3_model_dir: Path = Path("/work/courses/3dv/team7/temp/models/da3_weights")
    output_dir: Path = Path("/work/courses/3dv/team7/output_dynamic_graph")
    jpeg_dir: Path = Path("/work/courses/3dv/team7/temp/frames_jpeg")
    qwen_vl_model_id: str = "Qwen/Qwen3-VL-8B-Instruct"
    use_da3: bool = False
    debug_alignment_frame: Optional[int] = None
    use_cot: bool = True
    cot_max_tokens: int = 512
    qwen_max_tokens: int = 256
    local_files_only: bool = True
    hf_cache_dir: str = "/work/courses/3dv/team7/.cache/huggingface"
    sam_backend: str = DEFAULT_SAM_BACKEND
    sam3i_package_root: Path = DEFAULT_SAM3I_PACKAGE_ROOT
    sam3i_checkpoint_path: Path = DEFAULT_SAM3I_CHECKPOINT_PATH
    sam3i_inst_stage: str = os.environ.get("GRAPH_SAM3I_STAGE", "1_2")
    sam3i_adapter_dim: int = int(os.environ.get("GRAPH_SAM3I_ADAPTER_DIM", "64"))
    sam3i_adapter_heads: int = int(os.environ.get("GRAPH_SAM3I_ADAPTER_HEADS", "4"))
    sam3i_adapter_scale: float = float(os.environ.get("GRAPH_SAM3I_ADAPTER_SCALE", "1.0"))
    use_openfungraph_crops: bool = False
    clip_segmentation_mode: str = "smoothed"
    # Lowered from 1 (every frame) to 5. Every-frame SAM3 propagation on a
    # 30+ frame inactive clip produces hugely redundant priors and burns
    # SAM3 budget. Adaptive logic in build_inactive_label_priors clamps
    # stride down further when the inactive clip is short enough that the
    # configured value would leave fewer than MIN_INACTIVE_PRIOR_FRAMES
    # frames lifted (which previously starved 307_2200's container prior
    # down to 16 world points).
    inactive_stride: int = 5
    # Use the LAST N inactive frames (nearest the active interaction) to build the
    # prior, instead of all of them — the frames closest to the grasp are the most
    # representative rest-state view of the object.
    inactive_prior_frames: int = 10
    frame_start: int = 0
    frame_end: int = 370
    static_tf_path: Optional[Path] = None

    def resolve_static_tf_path(self) -> Path:
        """Return the static transform JSON path for the current dataset."""
        if self.static_tf_path is not None:
            return Path(self.static_tf_path)
        candidates = sorted((self.data_dir / "tf_static").glob("*.json"))
        if not candidates:
            raise FileNotFoundError(f"No static transform JSON found under {self.data_dir / 'tf_static'}")
        return candidates[0]


@dataclass
class QwenBundle:
    """Loaded Qwen model, processor, and dispatch metadata."""

    model: object
    processor: object
    model_type: str
    is_qwen35_vl: bool
    config: PipelineConfig


def set_runtime_depth_scale(depth_scale: float) -> None:
    """Update the runtime depth conversion used by dataset-specific helpers."""
    global CURRENT_DEPTH_SCALE
    CURRENT_DEPTH_SCALE = float(depth_scale)


def resolve_depth_scale(depth_scale: Optional[float] = None) -> float:
    """Return an explicit depth scale or the dataset-specific runtime default."""
    if depth_scale is None:
        return float(CURRENT_DEPTH_SCALE)
    return float(depth_scale)


def load_depth_scale_from_dataset(data_dir: Path, fallback: float = DEPTH_SCALE) -> float:
    """Read the dataset's raw depth unit and convert it into our raw-to-meters divisor."""
    unit_dir = data_dir / "camera_depth_unit_of_depth_in_mm"
    candidates = sorted(unit_dir.glob("*.json"))
    if not candidates:
        return float(fallback)
    try:
        with open(candidates[0], "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        unit_in_mm = float(payload.get("data", {}).get("unit_in_mm", 1.0))
        if unit_in_mm <= 0.0:
            return float(fallback)
        return float(1000.0 / unit_in_mm)
    except Exception:
        return float(fallback)


NON_INTERACTABLE = {
    "floor",
    "wood floor",
    "ceiling",
    "wall",
    "kitchen",
    "room",
    "person",
    "hand",
    "home appliance",
    "appliance",
    "office supply",
    "tool",
    "kitchen counter",
    "counter top",
    "wash",
    "wood",
}


FUNCTIONAL_RELATIONS = (
    "can_open",
    "can_close",
    "can_activate",
    "can_turn_on",
    "can_turn_off",
    "can_lock",
    "can_unlock",
    "can_move",
    "part_required_for_use",
    "unknown",
)


AFFORDANCE_SOURCE_LABELS = (
    "handle",
    "door handle",
    "drawer handle",
    "cabinet handle",
    "knob",
    "door knob",
    "button",
    "switch",
    "lever",
    "latch",
    "pull",
    "slider",
    "control",
    "keypad",
    "key",
    "pedal",
    "trigger",
    "tap handle",
    "faucet handle",
)


FUNCTIONAL_TARGET_LABELS = (
    "door",
    "drawer",
    "cabinet",
    "cupboard",
    "closet",
    "fridge",
    "refrigerator",
    "microwave",
    "oven",
    "dishwasher",
    "washer",
    "window",
    "blind",
    "curtain",
    "machine",
    "appliance",
    "screen",
    "tv",
    "television",
    "computer",
    "fan",
    "light",
    "lamp",
    "lock",
    "faucet",
    "tap",
)


RIGID_HANDLE_OBJECT_LABELS = (
    "spoon",
    "fork",
    "knife",
    "utensil",
    "mug",
    "cup",
    "pan",
    "pot",
    "kettle",
    "bottle",
    "tool",
)


ARTICULATED_PART_LABELS = (
    "door",
    "drawer",
    "lid",
    "cap",
    "window",
    "flap",
)


ARTICULATED_PARENT_BODY_LABELS = (
    "microwave",
    "fridge",
    "refrigerator",
    "oven",
    "dishwasher",
    "washer",
    "cabinet",
    "cupboard",
    "closet",
    "appliance",
    "machine",
)


STATIC_CONTROL_LABELS = (
    "button",
    "switch",
    "keypad",
    "control",
    "knob",
    "dial",
    "lever",
)


MOVABLE_ATTACHMENT_LABELS = (
    "handle",
    "knob",
    "pull",
    "lever",
)

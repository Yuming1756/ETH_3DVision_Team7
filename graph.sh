#!/bin/bash
set -Eeuo pipefail

cd ~/ETH_3DVision_Team7/final_project
source ../local_external_env.sh

export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export GRAPH_SAM_BACKEND=sam3
export GRAPH_PRIOR_METHOD=B
export GRAPH_QWEN_CPU_OFFLOAD=0
export SAM3_MAX_SIDE=0
export INACTIVE_PRIOR_FRAMES=10

# Match graph_183long_100674 behavior
export ACTIVE_CLIP_POSTROLL_FRAMES=0
export ACTIVE_CLIP_BRIDGE_GAP_FRAMES=15

# These are defaults in your current code, but keep them explicit for debugging
export ARTIPOINT_LITERAL_FILTERS=1
export ARTIPOINT_OCCLUSION_FILTER_MODE=off

rm -rf output/output_final_183_long_4b output/tmp_frames_jpeg
mkdir -p output/tmp_frames_jpeg

echo "python=$(which python)"
python -c "import sys; print(sys.executable); import scipy; print('scipy ok', scipy.__version__)"
echo "ACTIVE_CLIP_POSTROLL_FRAMES=$ACTIVE_CLIP_POSTROLL_FRAMES"
echo "ACTIVE_CLIP_BRIDGE_GAP_FRAMES=$ACTIVE_CLIP_BRIDGE_GAP_FRAMES"

python -u graph.py \
  --data-dir data/open-source-183_extracted \
  --frame-start 650 \
  --frame-end 800 \
  --output-dir output/output_final_183_long_4b_650_800 \
  --jpeg-dir output/tmp_frames_jpeg \
  --inactive-stride 1 \
  --inactive-prior-frames "$INACTIVE_PRIOR_FRAMES" \
  --hf-cache-dir "$HF_HOME" \
  --qwen-vl-model-id "Qwen/Qwen3-VL-4B-Instruct" \
  --sam-backend sam3 \
  2>&1 | tee graph_183long_4b_local.log
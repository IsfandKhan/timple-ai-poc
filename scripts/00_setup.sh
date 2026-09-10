#!/usr/bin/env bash
# One-shot RunPod setup. Run from the repo root on the pod.
#   export HF_TOKEN=hf_xxx   # needs FLUX.1-dev license accepted
#   bash scripts/00_setup.sh
set -euo pipefail

: "${HF_TOKEN:?set HF_TOKEN (accept the FLUX.1-dev license on HuggingFace first)}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export COMFY="${COMFY:-/workspace/ComfyUI}"
export HF_HOME="${HF_HOME:-/workspace/hf}"
# keep pip cache on the persistent volume so a pod restart (which wipes /) re-installs fast
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/workspace/pip-cache}"
mkdir -p "$HF_HOME" "$PIP_CACHE_DIR"

# ComfyUI's requirements.txt lists bare torch/vision/audio -> pip will pull a CUDA build
# newer than the host driver (seen: 2.14+cu130 on a 570/cu12.8 driver -> CUDA False).
# Pin the trio and constrain every downstream install. cu128 works with driver >=570;
# ComfyUI HEAD's comfy_kitchen needs torch >= 2.7.
TORCH="torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0"
TORCH_IDX="https://download.pytorch.org/whl/cu128"
CONSTRAINTS=/tmp/constraints.txt
printf 'torch==2.8.0\ntorchvision==0.23.0\ntorchaudio==2.8.0\n' > "$CONSTRAINTS"

echo "== torch =="
pip install -q $TORCH --index-url "$TORCH_IDX"

echo "== ComfyUI =="
[ -d "$COMFY" ] || git clone https://github.com/comfyanonymous/ComfyUI "$COMFY"
cd "$COMFY"
pip install -q -c "$CONSTRAINTS" -r requirements.txt
pip install -q -c "$CONSTRAINTS" huggingface_hub hf_transfer onnxruntime-gpu insightface==0.7.3 \
  opencv-python-headless matplotlib pyyaml einops timm

echo "== custom nodes =="
cd "$COMFY/custom_nodes"
clone() { [ -d "$(basename "$1" .git)" ] || git clone --depth 1 "$1"; }
clone https://github.com/ltdrdata/ComfyUI-Manager
clone https://github.com/cubiq/ComfyUI_IPAdapter_plus
clone https://github.com/cubiq/ComfyUI_InstantID
clone https://github.com/Fannovel16/comfyui_controlnet_aux
clone https://github.com/ltdrdata/ComfyUI-Impact-Pack
clone https://github.com/ltdrdata/ComfyUI-Impact-Subpack
clone https://github.com/ssitu/ComfyUI_UltimateSDUpscale
clone https://github.com/XLabs-AI/x-flux-comfyui
clone https://github.com/kijai/ComfyUI-Florence2
clone https://github.com/rgthree/rgthree-comfy
for d in */; do
  [ -f "$d/requirements.txt" ] && pip install -q -c "$CONSTRAINTS" -r "$d/requirements.txt" || true
done

echo "== torch sanity =="
pip install -q $TORCH --index-url "$TORCH_IDX"   # safety net: a node dep may have moved it
python - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA not available after deps -- torch build vs driver mismatch"
print("cuda ok:", torch.__version__, torch.cuda.get_device_name(0))
PY

echo "== insightface: buffalo_l (eval) =="
python - <<'PY'
from insightface.app import FaceAnalysis
FaceAnalysis(name="buffalo_l")   # -> ~/.insightface/models/buffalo_l (04_eval.py default)
print("ok buffalo_l")
PY
# antelopev2 (InstantID) comes from models.txt -> ComfyUI/models/insightface/models/antelopev2
# (the GitHub release zip extracts to a broken nested path on insightface 0.7.3)

echo "== models =="
# parallel chunked downloads (hf_transfer) -- ~20x faster than single-stream wget
HF_HUB_ENABLE_HF_TRANSFER=1 python "$ROOT/scripts/fetch_models.py"

echo
echo "done. start ComfyUI:"
echo "  cd $COMFY && HF_HOME=$HF_HOME python main.py --listen 0.0.0.0 --port 8188"

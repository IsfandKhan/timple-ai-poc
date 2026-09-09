#!/usr/bin/env bash
# One-shot RunPod setup. Run from the repo root on the pod.
#   export HF_TOKEN=hf_xxx   # needs FLUX.1-dev license accepted
#   bash scripts/00_setup.sh
set -euo pipefail

: "${HF_TOKEN:?set HF_TOKEN (accept the FLUX.1-dev license on HuggingFace first)}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
COMFY="${COMFY:-/workspace/ComfyUI}"
MODELS_TXT="$ROOT/config/models.txt"

echo "== ComfyUI =="
if [ ! -d "$COMFY" ]; then
  git clone https://github.com/comfyanonymous/ComfyUI "$COMFY"
fi
cd "$COMFY"
pip install -q -r requirements.txt
pip install -q "huggingface_hub[cli]" onnxruntime-gpu insightface==0.7.3 \
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
  [ -f "$d/requirements.txt" ] && pip install -q -r "$d/requirements.txt" || true
done

echo "== models =="
# insightface packs for InstantID (antelopev2) + eval (buffalo_l)
python - <<'PY'
from insightface.app import FaceAnalysis
for name in ("antelopev2", "buffalo_l"):
    FaceAnalysis(name=name)  # triggers download to ~/.insightface/models
    print("ok", name)
PY

dl() { # dest_subdir url
  local dest="$COMFY/models/$1"; mkdir -p "$dest"
  local fname; fname="$(basename "${2%%\?*}")"
  if [ -f "$dest/$fname" ]; then echo "  have $fname"; return; fi
  echo "  get $fname -> $1"
  wget -q --header="Authorization: Bearer $HF_TOKEN" -O "$dest/$fname" "$2"
}
grep -vE '^\s*#|^\s*$' "$MODELS_TXT" | while IFS='|' read -r dest url note; do
  dest="$(echo "$dest" | xargs)"; url="$(echo "$url" | xargs)"
  [ -n "$dest" ] && [ -n "$url" ] && dl "$dest" "$url"
done

# a few files need renaming to what the nodes expect
cd "$COMFY/models"
[ -f clip_vision/model.safetensors ] && mv -n clip_vision/model.safetensors clip_vision/CLIP-ViT-H-14.safetensors || true
mkdir -p instantid && [ -f instantid/ip-adapter.bin ] || true

echo
echo "done. start ComfyUI:"
echo "  cd $COMFY && python main.py --listen 0.0.0.0 --port 8188"

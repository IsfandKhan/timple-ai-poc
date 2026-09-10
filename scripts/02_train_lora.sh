#!/usr/bin/env bash
# Flux character LoRA training via ai-toolkit. Run on the pod after the dataset is curated + captioned.
#   bash scripts/02_train_lora.sh
# ai-toolkit gets its OWN venv -- its deps (numpy<2 C-exts, diffusers pin) conflict with
# ComfyUI's system env. Keep them apart.
set -euo pipefail

: "${HF_TOKEN:?set HF_TOKEN}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
AITK="${AITK:-/workspace/ai-toolkit}"
VENV="${AITK_VENV:-/workspace/aitk-venv}"
export HF_HOME="${HF_HOME:-/workspace/hf}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/workspace/pip-cache}"

DS="$ROOT/character/dataset"
n=$(find "$DS" -name '*.png' 2>/dev/null | wc -l | tr -d ' ')
c=$(find "$DS" -name '*.txt' 2>/dev/null | wc -l | tr -d ' ')
[ "$n" -ge 12 ] || { echo "only $n images in $DS -- need >=12 (aim 20-25)"; exit 1; }
[ "$c" -ge "$n" ] || { echo "$c captions for $n images -- run 01_build_dataset.py --stage caption"; exit 1; }
echo "dataset: $n images, $c captions"

[ -d "$AITK" ] || { git clone https://github.com/ostris/ai-toolkit "$AITK"; cd "$AITK" && git submodule update --init --recursive; }

if [ ! -x "$VENV/bin/python" ]; then
  python -m venv "$VENV"
  "$VENV/bin/pip" install -q --upgrade pip
  "$VENV/bin/pip" install -q torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
    --index-url https://download.pytorch.org/whl/cu128
  "$VENV/bin/pip" install -q -r "$AITK/requirements.txt"
fi

cd "$AITK"
export HF_TOKEN HF_HUB_ENABLE_HF_TRANSFER=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # reduce fragmentation OOMs on 24GB
"$VENV/bin/python" -c "import torch; assert torch.cuda.is_available(); print('cuda ok', torch.__version__)"
"$VENV/bin/python" run.py "$ROOT/config/train_lora.yaml"

echo
echo "samples: $ROOT/output/mara_lora/samples/  -- pick the checkpoint with locked identity + prompt response"
echo "then: cp $ROOT/output/mara_lora/mara_lora_000XXXX.safetensors /workspace/ComfyUI/models/loras/mara_lora.safetensors"

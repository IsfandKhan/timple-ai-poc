#!/usr/bin/env bash
# Flux character LoRA training via ai-toolkit. Run on the pod after the dataset is curated + captioned.
#   bash scripts/02_train_lora.sh
set -euo pipefail

: "${HF_TOKEN:?set HF_TOKEN}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
AITK="${AITK:-/workspace/ai-toolkit}"

DS="$ROOT/character/dataset"
n=$(find "$DS" -name '*.png' 2>/dev/null | wc -l | tr -d ' ')
c=$(find "$DS" -name '*.txt' 2>/dev/null | wc -l | tr -d ' ')
[ "$n" -ge 12 ] || { echo "only $n images in $DS -- need >=12 (aim 20-25)"; exit 1; }
[ "$c" -ge "$n" ] || { echo "$c captions for $n images -- run 01_build_dataset.py --stage caption"; exit 1; }
echo "dataset: $n images, $c captions"

if [ ! -d "$AITK" ]; then
  git clone https://github.com/ostris/ai-toolkit "$AITK"
  cd "$AITK" && git submodule update --init --recursive
  pip install -q -r requirements.txt
fi

cd "$AITK"
export HF_TOKEN
python run.py "$ROOT/config/train_lora.yaml"

echo
echo "samples: $ROOT/output/mara_lora/samples/  -- review, pick the checkpoint with locked identity + prompt response"
echo "then: cp \$ROOT/output/mara_lora/mara_lora_000XXXX.safetensors $ROOT/../ComfyUI/models/loras/mara_lora.safetensors"

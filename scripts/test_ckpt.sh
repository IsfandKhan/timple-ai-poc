#!/usr/bin/env bash
# Quick identity check of LoRA checkpoints in ComfyUI's Flux sampler (ai-toolkit's own
# low_vram previews come out as noise -- ignore those).
#   bash scripts/test_ckpt.sh 500 750 1000
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
COMFY="${COMFY:-/workspace/ComfyUI}"
cd "$ROOT"

curl -s -X POST http://127.0.0.1:8188/free -H 'Content-Type: application/json' \
  -d '{"unload_models":true,"free_memory":true}' >/dev/null || true

for step in "$@"; do
  src="$ROOT/output/mara_lora/mara_lora_$(printf '%09d' "$step").safetensors"
  [ -f "$src" ] || { echo "no checkpoint at step $step"; continue; }
  cp "$src" "$COMFY/models/loras/mara_lora.safetensors"
  echo "== step $step =="
  python scripts/03_generate.py --pipeline flux_lora --scenes 9,4
  mkdir -p "reports/ckpt_test/$step"
  mv outputs/flux_lora/scene_09.png "reports/ckpt_test/$step/portrait.png" 2>/dev/null || true
  mv outputs/flux_lora/scene_04.png "reports/ckpt_test/$step/fullbody.png" 2>/dev/null || true
done
echo "done -> reports/ckpt_test/<step>/"

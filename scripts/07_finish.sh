#!/usr/bin/env bash
# Post-training: install the chosen LoRA, render both pipelines, score, compare, contact sheets.
#   bash scripts/07_finish.sh /workspace/svetlana/output/mara_lora/mara_lora_000001500.safetensors
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CKPT="${1:?pass the chosen LoRA checkpoint path}"
COMFY="${COMFY:-/workspace/ComfyUI}"

cp "$CKPT" "$COMFY/models/loras/mara_lora.safetensors"
echo "installed $(basename "$CKPT") -> mara_lora.safetensors"

# free any VRAM ai-toolkit left, let ComfyUI reload on demand
curl -s -X POST http://127.0.0.1:8188/free -H 'Content-Type: application/json' \
  -d '{"unload_models":true,"free_memory":true}' >/dev/null || true

cd "$ROOT"
# preserve the first (reference-heavy) sdxl run before the retuned one overwrites it
[ -d outputs/sdxl_ref ] && rm -rf outputs/sdxl_ref_v1 && mv outputs/sdxl_ref outputs/sdxl_ref_v1 || true

echo "== Pipeline A (flux_lora) =="
python scripts/03_generate.py --pipeline flux_lora --all

echo "== Pipeline B v2 (sdxl_ref, retuned for scene control) =="
python scripts/03_generate.py --pipeline sdxl_ref --all
mv outputs/sdxl_ref outputs/sdxl_ref_v2

echo "== eval =="
python scripts/04_eval.py --dir outputs/flux_lora    --ref character/hero.png --out reports/flux_lora
python scripts/04_eval.py --dir outputs/sdxl_ref_v1  --ref character/hero.png --out reports/sdxl_ref_v1
python scripts/04_eval.py --dir outputs/sdxl_ref_v2  --ref character/hero.png --out reports/sdxl_ref_v2
python scripts/04_eval.py --compare outputs/flux_lora outputs/sdxl_ref_v2 --ref character/hero.png --out reports/compare

echo "== contact sheets =="
python scripts/05_contact_sheet.py --dir outputs/flux_lora
python scripts/05_contact_sheet.py --dir outputs/sdxl_ref_v1
python scripts/05_contact_sheet.py --dir outputs/sdxl_ref_v2
python scripts/05_contact_sheet.py --compare outputs/flux_lora outputs/sdxl_ref_v2

echo
echo "done. reports/ has: flux_lora/, sdxl_ref_v2/, compare/, contact_*.png"

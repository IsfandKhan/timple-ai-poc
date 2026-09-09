# APPROACH — write during/after the build (H30–H42)

This is the document Svetlana reads. Fill each section with real numbers and images
from the run. Keep it honest about what's experimental vs solid.

## 1. What I built

- Two pipelines, end to end, hands-on (list each file, one line each).
- Synthetic character, 12-scene story arc, same seed per scene across pipelines.
- Automated identity scoring on every image.

## 2. The two approaches

| | Flux + trained LoRA | SDXL + IP-Adapter FaceID + InstantID |
|---|---|---|
| identity source | learned weights | one reference image |
| setup cost | ~90 min train, per character | none |
| … | (fill from run) | |

## 3. Results

Paste from `reports/compare/comparison.md`:

| metric | flux_lora | sdxl_ref |
|---|---|---|
| mean pairwise similarity | | |
| min pairwise similarity | | |
| mean similarity to reference | | |
| faces missing | | |

Inline: `reports/*/consistency_heatmap.png`, `reports/contact_compare.png`.
Call out the 2–3 scenes each pipeline struggled with and why (pose extremity, lighting, framing).

## 4. Production recommendation

- LoRA for the recurring hero cast (few characters, generated constantly) — best lock, learns build/hair not just face.
- IP-Adapter + InstantID for long-tail / one-shot characters — no training turnaround.
- Both share one downstream stack: ControlNet pose → FaceDetailer → upscale → eval.
- **Eval as a gate:** every batch scored; anything under threshold is auto-flagged for regen. This is the piece that makes it a pipeline, not a vibe.

## 5. How it evolves (Svetlana's real concern)

- Base model swap: pipelines are staged; `config/pipelines.yaml` points at the checkpoint. New Flux/SDXL successor → change one line + retrain LoRAs. Downstream + eval unchanged.
- Dataset versioning: keep `character/dataset/` under version control per character; retrain is reproducible from `config/train_lora.yaml`.
- Retraining cadence: retrain a character LoRA when (a) base model changes, or (b) eval mean drops below target on new scene types.
- Next capabilities: wardrobe as a separate IP-Adapter/ControlNet input, multi-character composition, temporal consistency for video (AnimateDiff / SVD + per-frame FaceDetailer), pose library expansion.

## 6. Honest limitations

- 20-image synthetic dataset → LoRA is as good as the bootstrap; a real subject with photos would be stronger.
- Face-only identity metric; body/hair consistency judged visually.
- 12 scenes, one character, one ethnicity/age — not a stress test of the full space.
- No temporal/video work in this POC.

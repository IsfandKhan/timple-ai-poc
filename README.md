# Character Consistency POC — Timple.ai

Proof that the same realistic character can be held consistent across many scenes,
poses, and images — as a **repeatable, measured pipeline**, not a one-off.

## What this shows

- One invented photoreal character ("Mara Vance"), 12 story scenes.
- Two pipelines, compared head to head:
  - **A — Flux + trained character LoRA + Flux ControlNet** (identity learned)
  - **B — SDXL + IP-Adapter FaceID + InstantID** (identity from a reference image, no training)
- A face-identity **consistency score on every generated image** (InsightFace / ArcFace).
- A written call on what to ship in production and how the pipeline evolves as models improve.

Covers every tool Svetlana named: Flux, SDXL, ComfyUI, LoRA, IP-Adapter.

## Layout

| Path | What |
|---|---|
| `PLAN.md` | Architecture, both pipelines, metrics, production tradeoffs |
| `RUNBOOK.md` | Exact 48-hour execution plan on a rented RunPod 4090 |
| `character/character_spec.md` | The invented character — fixed appearance, trigger word |
| `character/scenes.yaml` | The 12 story scenes (prompt + pose + framing per scene) |
| `config/models.txt` | Every model/checkpoint to download, with source |
| `scripts/00_setup.sh` | One-shot RunPod setup: ComfyUI + custom nodes + models |
| `scripts/01_build_dataset.py` | Bootstrap ~30 varied training images of the character |
| `scripts/02_train_lora.sh` | Flux character LoRA training (ai-toolkit) |
| `scripts/03_generate.py` | Batch-generate all 12 scenes through both pipelines |
| `scripts/04_eval.py` | Consistency scoring + heatmap + report (runs on the Mac) |
| `scripts/05_contact_sheet.py` | Assemble the final comparison grid |

## Deliverable to Svetlana

1. `outputs/` — Mara in 12 scenes, both pipelines
2. `reports/` — consistency scores, per-scene heatmaps, contact sheets
3. `APPROACH.md` — written during the build: what worked, the tradeoff, production recommendation, evolution plan
4. Short screen-recorded walkthrough

## Budget

RunPod RTX 4090 community ~$0.44/hr. Pod stopped during training waits and write-up.
~30 active GPU-hours + storage ≈ **$15–20**.

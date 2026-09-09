# PLAN — Character Consistency POC

## Goal

Demonstrate **production-grade** character consistency: the same realistic human held
across many scenes/poses, produced by a pipeline that is repeatable, measured, and
maintainable — the opposite of a one-off experiment.

## Why two pipelines

Svetlana named Flux, SDXL, ComfyUI, LoRA, IP-Adapter. The strongest answer is not to
pick one — it's to build both classes of approach and show the tradeoff with numbers.

| | Pipeline A | Pipeline B |
|---|---|---|
| Base | Flux.1-dev (fp8) | SDXL 1.0 |
| Identity method | Trained character **LoRA** (rank 16) | **IP-Adapter FaceID** + **InstantID**, reference image only |
| Training | ~1–2 h once | none |
| Pose control | Flux ControlNet (OpenPose) | SDXL ControlNet (OpenPose) + InstantID keypoints |
| Face cleanup | Impact Pack FaceDetailer | Impact Pack FaceDetailer |
| Upscale | UltimateSDUpscale ×2 (4x-UltraSharp) | same |
| Strengths | Best identity lock, best realism, learns non-face traits (hair, build) | Zero training, instant new characters, mature tooling, faster iteration |
| Weaknesses | Retrain per character, slower gen, LoRA can overfit | Face-only identity, drifts on extreme angles, "IP-Adapter look" |

**Production position to argue:** LoRA for the small set of hero characters that recur
constantly; IP-Adapter/InstantID for long-tail or one-shot characters. Pipeline is
built so both share the same ControlNet + FaceDetailer + upscale + eval stages.

## Stages

```
character_spec ──► 01 build dataset ──► curate + auto-filter ──► 02 train LoRA
                        │                                              │
                        └──────────► hero.png (reference) ──────┐      │
                                                                ▼      ▼
scenes.yaml ──► 03 generate ──► [Pipeline A: Flux+LoRA+CN]  ──► FaceDetailer ──► upscale ──► outputs/flux_lora/
            └─► 03 generate ──► [Pipeline B: SDXL+IPA+InstantID+CN] ──► FaceDetailer ──► upscale ──► outputs/sdxl_ref/
                                                                                              │
                                                          04 eval ◄────────────────────────────┘
                                                             │
                                          reports/ (scores, heatmaps) + 05 contact sheet
```

### 1. Character dataset bootstrap (the hard part)

A synthetic character has no photos. Build them:

1. Generate ~8 hero portraits with Flux from `character_spec.md`, vary seed only.
   Pick the single best as `character/hero.png`. Lock that seed.
2. From `hero.png`, generate ~30 varied shots with **SDXL + InstantID + IP-Adapter FaceID**
   (this is Pipeline B used as a data engine): vary angle, expression, framing
   (close / half / full body), lighting, background. Keep pose neutral-to-moderate.
3. FaceDetailer + ×2 upscale every shot.
4. Auto-filter with `04_eval.py --ref hero.png`: drop any image with cosine
   similarity < 0.60 to the hero.
5. Manual pass: keep the 20–25 cleanest and most identity-true.
6. Caption each with the trigger word `mara_vance` (Florence-2 auto-caption, then
   quick manual fix).

This doubles as evidence: Pipeline B is good enough to *manufacture a training set*,
which is itself a production-useful capability.

### 2. LoRA training

`ai-toolkit`, Flux.1-dev, rank 16, ~2200 steps, lr 1e-4, batch 1, 1024px.
Sample every 250 steps against a fixed validation prompt. Stop at the checkpoint
where identity is locked but flexibility is not yet lost (watch for "same photo
every time" = overfit). ~1–2 h on a 4090.

### 3. Scene generation

`scenes.yaml` defines 12 scenes. Each scene has: `id`, `prompt`, `pose` (reference
image name for OpenPose), `framing`, `seed`. `03_generate.py` runs every scene
through both pipelines via the ComfyUI API, same seed per scene across pipelines so
outputs are comparable.

Pose references: 12 stock OpenPose skeletons (or pose photos) covering standing,
walking, crouching, seated, reaching, over-the-shoulder, etc. Stored in
`character/poses/`.

### 4. Refinement loop

After first full run, `04_eval.py` flags the worst 3–4 scenes per pipeline
(lowest similarity to hero / lowest intra-set similarity). Regenerate those with
seed variation and stronger identity weighting. Re-eval. One or two rounds.

## Metrics

Face embeddings: InsightFace `buffalo_l` (ArcFace), largest face per image,
L2-normalized, cosine similarity.

| Metric | Meaning | Target |
|---|---|---|
| **Intra-set mean pairwise similarity** | how alike the 12 outputs are to each other — the headline consistency number | ≥ 0.75 |
| Intra-set min pairwise | worst drift between any two outputs | ≥ 0.55 |
| Mean similarity to hero reference | anchor drift from the canonical face | ≥ 0.70 |
| Per-scene similarity to hero | which scenes/poses break identity | heatmap |
| Face-detection rate | scenes where no usable face was rendered | 12/12 |

Realism is judged visually + noted per scene (optionally CLIP-IQA). Not the focus —
consistency is.

Report shows both pipelines side by side on every metric, plus a 12×12 similarity
heatmap per pipeline and a cross-pipeline comparison.

## Scope — in / out

**In:** 1 character, 12 scenes, 2 pipelines, full eval, refinement loop, written approach.

**Out (named as next steps in APPROACH.md):** multi-character scenes, video / temporal
consistency, real-person likeness, outfit/wardrobe as a separate control, a web UI,
automated retraining triggers, dataset versioning. All real production work — called
out so Svetlana sees the roadmap, not scoped into 48 h.

## Risks

| Risk | Mitigation |
|---|---|
| Flux.1-dev is gated on HF | Accept license, use HF token in `00_setup.sh`; fp8 fallback mirrors exist |
| LoRA overfits on 20 images | rank 16, moderate steps, sample checkpoints, keep dataset varied |
| Identity drifts on extreme poses | ControlNet weight tuning + FaceDetailer with identity re-inject; keep pose set moderate |
| Bootstrap dataset inherits IP-Adapter artifacts | FaceDetailer + upscale before training; manual curation |
| RunPod 4090 OOM on Flux + ControlNet + detailer | fp8 Flux, sequential (not parallel) node execution, `--lowvram` fallback, or bump to A100 40GB for the gen step |
| 48 h slip | Pipeline B alone is a complete deliverable; LoRA/Pipeline A is the upgrade. Ship B first, add A. |

## Cost

RTX 4090 @ ~$0.44/hr. Active hours: setup 3, dataset 4, gen + refine 12, eval/iter 6,
misc 5 ≈ 30 h → ~$13. Network volume 50 GB ≈ $3–5. **Under $20.**

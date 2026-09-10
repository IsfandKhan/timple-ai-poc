# Character Consistency POC — Mara Vance

**Goal:** hold one invented realistic human identity consistent across a 12-scene
story arc, and *measure* it — as a repeatable pipeline, not hand-picked one-offs.
Built end to end on a rented RTX 4090 (RunPod), driven by the scripts in this repo.

**Delivered:** Pipeline B (SDXL + IP-Adapter FaceID + InstantID, training-free) across
all 12 scenes, scored, in two weight configs. Pipeline A (Flux + trained LoRA) was
built and run but the LoRA did not come out usable in this budget — the honest
account is in §4.

---

## 1. What was built

| file | role |
|---|---|
| `character/character_spec.md` | the fixed identity sheet every prompt references |
| `character/scenes.yaml` | 12 scenes (a dawn-to-night field expedition), fixed seed + pose ref per scene |
| `scripts/00_setup.sh` + `fetch_models.py` | one-shot pod bring-up: ComfyUI + 10 custom nodes, torch pinned, ~48 GB models via parallel download |
| `comfyui/sdxl_ref.api.json` | Pipeline B graph — RealVisXL + InstantID + IP-Adapter FaceID + OpenPose ControlNet |
| `comfyui/flux_lora.api.json` | Pipeline A graph — Flux.1-dev fp8 + LoRA + Flux ControlNet |
| `scripts/01_build_dataset.py` | synthetic bootstrap: hero portrait → 12-pose bank → 32 varied training shots → BLIP captions |
| `scripts/02_train_lora.sh` + `config/train_lora.yaml` | Flux LoRA training via ai-toolkit (isolated venv, low-VRAM quant) |
| `scripts/03_generate.py` | render every scene through either pipeline, same seed, shared prompt + pose |
| `scripts/04_eval.py` | **the scoring** — ArcFace identity metrics, heatmaps, threshold filter, A/B compare |
| `scripts/05_contact_sheet.py` | contact + comparison grids |

No manual image editing anywhere. The character, its training data, the scenes,
and the scores are all pipeline output.

### The synthetic character

"Mara Vance" — 34, field archaeologist, Greek heritage, olive skin, dark wavy hair
in a low bun, small scar through the right eyebrow, silver coin pendant. One Flux
portrait was generated from the spec and chosen as the **identity anchor**
(`character/hero.png`); everything downstream references it.

---

## 2. Results — Pipeline B (SDXL + InstantID + IP-Adapter FaceID)

Two weight configs, same 12 scenes, same seeds:

| config | InstantID | IP-Adapter | pose CN | what it's for |
|---|---|---|---|---|
| **v1** | 0.65 (full) | 0.75 | 0.60 | maximum identity lock |
| **v2** | 0.45, `end_at` 0.45 | 0.55 | 0.85 | identity early, composition + pose late |

### Identity consistency — both configs, all 12 scenes

| metric | v1 (identity-locked) | v2 (retuned) | target |
|---|---|---|---|
| mean pairwise similarity | **0.904** | **0.784** | ≥ 0.75 |
| min pairwise similarity | 0.868 | 0.683 | ≥ 0.55 |
| mean similarity to hero | 0.845 | 0.756 | ≥ 0.70 |
| faces not detected | 0 / 12 | 0 / 12 | 0 |

**Both configs pass every threshold.** `reports/sdxl_ref_v1/`,
`reports/sdxl_ref_v2/`, `reports/compare/comparison.md`, `reports/contact_*.png`.

### The finding — a tunable identity ↔ controllability tradeoff

**v1** (InstantID 0.65, IP-Adapter 0.75) locks identity hard — 0.83–0.86 vs hero on
every scene, σ 0.019 — but at those weights the adapters **dominate composition**.
Every scene collapsed to a centered head-and-shoulders portrait regardless of the
scene prompt and the pose ControlNet; "Ridge hike" and "Departure" (specified
full-body) came back as headshots. This is InstantID's known behaviour: at
identity-lock weights it is a face generator, not a character-in-a-scene tool.

**v2** (InstantID 0.45 with `end_at` 0.45 — guide the first ~45 % of denoising then
let go — IP-Adapter 0.55, pose ControlNet 0.85) buys back **scene detail**: props,
wardrobe and environment read more clearly (coffee mug in "Dawn at camp", soil
sample in "Examining the artifact", radio at the ear in "Radio call", rain +
umbrella in "The storm", market stalls in "Village market"). Identity drops but
stays inside target — mean vs hero 0.756, mean pairwise 0.784. **Framing did not
change** — both configs stay head-and-shoulders. The face-keypoint ControlNet
inside InstantID keeps centering the head regardless of pose-CN strength, so true
full-body shots don't come out of this pipeline at all.

**Read:** you dial the InstantID/IP-Adapter weight and `end_at` to trade identity
lock for scene freedom. There's headroom above the consistency floor to spend on
control — but a hard full-body requirement is where the trained-LoRA path (identity
in the weights, sampler free) would win outright.

---

## 3. The bootstrap loop (B feeds A)

1. Flux generates 8 hero portraits from the spec → one chosen as the anchor.
2. Pipeline B (InstantID + IP-Adapter off the anchor) generates 32 varied shots.
3. `04_eval.py` scores them vs the anchor — **0.899 mean intra-set, 0 rejected** at a
   0.60 threshold, so the whole set was kept.
4. BLIP captions + `mara_vance` trigger → the LoRA training set.

The training-free method is also the *data engine* for the trained method. That
part worked cleanly and is the right pattern regardless of which generator ships.

---

## 4. Pipeline A — what happened

Built and run: ai-toolkit, Flux.1-dev, LoRA rank 16, 2200-step config, samples +
checkpoints every 250. Training itself ran fine — loss tracked normally (~0.5 → 0.4),
checkpoints at 250/500/750/1000.

**But every checkpoint renders as noise** — in ai-toolkit's own previewer *and* in
ComfyUI's Flux sampler, at strengths from 0.15 to 0.95. The weights are finite (no
NaN/Inf) but scale-blown (one tensor max-abs ~166 vs the <10 a healthy LoRA shows),
and saved in diffusers `lora_A/lora_B` format with no `alpha` keys.

**Cause:** ai-toolkit trained the LoRA against a **quantized (fp8) Flux base** with
`low_vram` — the only way Flux LoRA training fits in 24 GB. That path produced a
LoRA that doesn't transfer to a clean fp8 (or full) model. Training on the full
bf16 model would avoid it but needs >24 GB alongside the optimizer state.

**Fix path (didn't fit this budget):**
- Train on a 40–48 GB GPU (A6000 / L40S / A100) against the full bf16 Flux — no quant.
- Or use a training stack that quantizes only for the forward pass and dequantizes
  the LoRA path (kohya-ss `sd-scripts` Flux branch handles this more reliably than
  ai-toolkit's `low_vram` at the moment).
- Then the LoRA loads in the same `comfyui/flux_lora.api.json` graph with no other
  change — the pipeline is staged for it.

The infra to *do* Pipeline A is all here and validated (graph smoke-tested, dataset
built, config written); the missing piece is a training run on hardware with the
headroom to skip quantization.

---

## 5. Production recommendation

- **Trained LoRA for the recurring hero cast** — few characters, generated
  constantly. Identity in the weights frees the sampler to follow pose and scene
  direction, which is exactly where reference-based methods struggle (§2). Worth the
  per-character training turnaround — **on hardware that can train it without
  quantization**.
- **IP-Adapter + InstantID for the long tail** — one-shot / rarely-seen characters.
  Accept the composition fight; keep identity weights moderate with an early
  `end_at`, and lean on ControlNet for pose (the v2 config).
- **One downstream stack** for both: ControlNet pose → FaceDetailer → upscale → eval.
  (FaceDetailer + upscale are specced in `comfyui/NODES.md` and wired as a second
  pass; this run shipped the base graphs + eval and skipped the refinement pass for
  time — it lifts absolute quality, not the consistency comparison.)
- **Eval is the gate.** Every batch scored against the hero; anything below
  threshold auto-flagged for regen (`04_eval.py --filter`). Same 40 lines of ArcFace
  math whether identity came from weights or a reference image — that's what makes
  it a pipeline and not a vibe.

---

## 6. How it evolves

- **Base-model swap.** `config/pipelines.yaml` points at the checkpoint; a Flux/SDXL
  successor is a one-line change plus a LoRA retrain from the same versioned
  `character/dataset/` and `config/train_lora.yaml`. Downstream + eval untouched.
- **Dataset versioning.** Each character's `character/dataset/` (images + captions)
  is the reproducible source of its LoRA — checked in per character.
- **Retraining cadence.** Retrain when the base model changes, or when eval mean
  drops below target on a new scene type.
- **Next capabilities.** Wardrobe as a separate ControlNet/IP-Adapter input;
  multi-character composition (regional prompting + per-region FaceDetailer);
  temporal consistency for video (per-frame FaceDetailer off the hero); larger pose
  library. All bolt onto the staged graph.

---

## 7. Honest limitations

- **Pipeline A not delivered** — see §4. Diagnosed, not solved, within the budget.
- **Synthetic-dataset skew.** InstantID/IP-Adapter pinned the 32 bootstrap shots to
  frontal face closeups — thin on body, angle, wardrobe. A real subject with real
  photos would be a stronger training set (and would sidestep §4 entirely for a
  real person).
- **Face-only metric.** ArcFace scores identity, not hair/build/wardrobe — those are
  judged visually from the contact sheets.
- **One character, one age/ethnicity, 12 scenes.** A demonstration, not a sweep. No
  extreme poses in the scene set.
- **No temporal/video** in this POC.
- **Contended host.** The rented box was heavily oversubscribed; several dependency
  conflicts (torch/torchaudio ABI, numpy 1↔2, Flux-quant VRAM) were worked through
  and captured as fixes in the setup scripts, so a re-run on clean hardware is one
  command.

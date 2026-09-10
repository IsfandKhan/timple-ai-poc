# RUNBOOK — 48 hours

Times are elapsed hours from start. Pod is **stopped** (not terminated) during the
two long unattended waits to save money — network volume keeps all data.

## Prep (before renting the pod — do on the Mac)

- [ ] HuggingFace account, accept license on `black-forest-labs/FLUX.1-dev`. Get a read token.
- [ ] RunPod account, add $25 credit.
- [ ] Local eval env: `cd scripts && python3 -m venv .venv && . .venv/bin/activate && pip install -r ../requirements-mac.txt`
- [ ] `python 04_eval.py --demo` passes.
- [ ] Read `character/character_spec.md` and `character/scenes.yaml`, adjust to taste.

## H0–H3 — Pod up, environment, smoke test

1. RunPod → Deploy → **RTX 4090**, Secure Cloud, template **official "RunPod PyTorch 2.x"**
   (its `/start.sh` handles sshd + injects your account SSH key). Container disk 30 GB,
   **Volume disk 150 GB** at `/workspace`. Add your SSH pubkey under Settings → SSH Public Keys.
2. `rsync` this repo to `/workspace/svetlana/` (incl. `.env` with `HF_TOKEN`). Then:
   ```bash
   cd /workspace/svetlana && export HF_TOKEN=$(grep -o 'hf_[A-Za-z0-9]*' .env)
   bash scripts/00_setup.sh          # in tmux — ~15 min pip + ~8 min models (hf_transfer)
   ```
   Pins torch **2.8.0+cu128** (ComfyUI HEAD needs ≥2.7; cu128 works w/ driver ≥570),
   clones ComfyUI + 10 custom nodes, `fetch_models.py` pulls ~48 GB.
3. Start ComfyUI in `tmux`:
   `cd /workspace/ComfyUI && HF_HOME=/workspace/hf python main.py --listen 127.0.0.1 --port 8188`
   Access the GUI by SSH tunnel: `ssh -L 8188:localhost:8188 root@<ip> -p <port>`.
4. Workflows are **pre-authored** (`comfyui/*.api.json`). Smoke test:
   ```bash
   python scripts/03_generate.py --smoke --pipeline flux_lora --no-lora
   python scripts/03_generate.py --smoke --pipeline sdxl_ref     # needs a character/hero.png; use any face for the smoke
   ```
   One image each in `outputs/_smoke/`. Fix any `node_errors` against `localhost:8188/object_info`.

**Host note:** if the pod lands on a contended host (`cat /proc/loadavg` » vCPU count),
cold model loads are slow but steady-state sampling is normal. If CUDA is dead
(`torch.cuda.is_available()` False while `nvidia-smi` works) → Stop/Start; if still dead → redeploy.

## H3–H7 — Character hero + dataset

1. `python scripts/01_build_dataset.py --stage hero --n 8` → `character/hero_candidates/`.
   Pick best → `cp character/hero_candidates/hero_0X.png character/hero.png`, note seed
   (`1000 + X*137`) into `character/character_spec.md`.
2. `python scripts/01_build_dataset.py --stage poses` → 12 generic pose refs in
   `character/poses/` (DWPose turns them into skeletons downstream).
3. `python scripts/01_build_dataset.py --stage expand --n 32`
   → `character/dataset_raw/` (SDXL + InstantID + IP-Adapter FaceID off `hero.png`).
4. `python scripts/01_build_dataset.py --stage refine`
   → FaceDetailer + upscale → `character/dataset_refined/` (copies through if no
   `detail_only.api.json` — the base sdxl_ref graph already has no detailer, so build that
   file or accept raw).
5. Auto-filter: `python scripts/04_eval.py --dir character/dataset_raw --ref character/hero.png --filter 0.60 --out reports/dataset`
   → moves rejects to `character/dataset_raw_rejected/`.
6. Manual: keep best 20–25 in `character/dataset/`.
7. Caption: `python scripts/01_build_dataset.py --stage caption --trigger mara_vance`
   → writes `.txt` next to each image. Skim and fix.

## H7–H9 — Kick off LoRA training

```bash
bash scripts/02_train_lora.sh
```
Runs `ai-toolkit` with `config/train_lora.yaml`. Samples to
`output/mara_lora/samples/` every 250 steps. ~1.5 h.

While it runs, start Pipeline B scene work (next block). Do **not** stop the pod.

## H9–H13 — Pipeline B scenes (runs during/after training)

1. Prepare pose refs: 12 images in `character/poses/` (stock OpenPose PNGs or pose photos).
   `scenes.yaml` already maps each scene to a pose file.
2. `python scripts/03_generate.py --pipeline sdxl_ref --all`
   → `outputs/sdxl_ref/scene_01.png … scene_12.png`
3. `python scripts/04_eval.py --dir outputs/sdxl_ref --ref character/hero.png --out reports/sdxl_ref`
4. Eyeball the contact sheet. Note weak scenes.

## H13–H15 — Checkpoint the training, pick LoRA

- Review `output/mara_lora/samples/`. Choose the checkpoint with locked identity + still
  responds to prompt changes (usually 1500–2000 steps). Set its path in `config/pipelines.yaml`.
- **Stop the pod now if it's night.** Resume in the morning. (Data persists on the volume.)

## H15–H21 — Pipeline A scenes

1. Restart pod, restart ComfyUI.
2. `python scripts/03_generate.py --pipeline flux_lora --all`
   → `outputs/flux_lora/scene_01.png … scene_12.png`
3. `python scripts/04_eval.py --dir outputs/flux_lora --ref character/hero.png --out reports/flux_lora`

## H21–H27 — Refinement loop

1. `python scripts/04_eval.py --compare outputs/flux_lora outputs/sdxl_ref --out reports/compare`
   → prints worst 3–4 scenes per pipeline.
2. Regenerate those: `python scripts/03_generate.py --pipeline <p> --scenes 3,7,9 --seed-shift`
3. Re-eval. One more round if time.
4. `python scripts/05_contact_sheet.py` → `reports/contact_flux_lora.png`,
   `reports/contact_sdxl_ref.png`, `reports/contact_compare.png`.

## H27–H30 — Pull results down, stop pod

```bash
# on the Mac
scp -r runpod:/workspace/svetlana/outputs ./outputs
scp -r runpod:/workspace/svetlana/reports ./reports
scp    runpod:/workspace/svetlana/output/mara_lora/*.safetensors ./artifacts/
```
Export both ComfyUI graphs (API + UI format) → `comfyui/`.
**Terminate the pod.**

## H30–H42 — Write-up

`APPROACH.md`:
- The two pipelines, what was implemented hands-on (all of it), file by file.
- Results table: both pipelines on every metric. Heatmaps inline.
- The tradeoff and the **production recommendation** (LoRA for hero cast, IP-Adapter for long tail, shared downstream stages).
- Evolution plan: swapping the base model (Flux → next), retraining cadence, dataset
  versioning, adding wardrobe/scene controls, temporal/video, eval as a CI gate.
- Honest limitations.

`README` update with headline numbers.

## H42–H48 — Walkthrough + buffer

- 3–5 min screen recording: run one scene through each pipeline live, show the eval output.
- Buffer for anything that slipped.
- Send: repo link + `APPROACH.md` + contact sheets + recording.

## If time runs short

Priority order to cut from the bottom:
1. Drop the refinement loop's 2nd round.
2. Drop Pipeline A (Flux LoRA), ship Pipeline B + eval + write-up. Still answers the brief.
3. Drop 4 scenes, ship 8.

Never cut: the eval / scoring. The measured-consistency part *is* the differentiator.

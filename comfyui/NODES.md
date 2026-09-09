# Workflow templates — the contract

`03_generate.py` and `01_build_dataset.py` load `comfyui/<name>.api.json` and patch
nodes **by their title** (`_meta.title` in API format — set via right-click → Title in
the ComfyUI editor). Build each graph once in the ComfyUI GUI on the pod during
H0–H3, title the nodes exactly as below, then **Save (API Format)** to the filename shown.

Nothing else in the graph matters to the scripts — wire the rest however works.

---

## `flux_lora.api.json` — Pipeline A

| Node title | Node type | Patched input(s) |
|---|---|---|
| `unet` | UNETLoader (or CheckpointLoaderNF4 / load fp8) | `unet_name` |
| `lora` | LoraLoaderModelOnly | `lora_name`, `strength_model` |
| `positive` | CLIPTextEncode (Flux) | `text` |
| `negative` | CLIPTextEncode | `text` |
| `latent` | EmptyLatentImage / EmptySD3LatentImage | `width`, `height` |
| `sampler` | KSampler | `seed`, `steps`, `cfg`, `sampler_name`, `scheduler` |
| `pose_image` | LoadImage | `image` (uploaded pose ref, subfolder `poses`) |
| `controlnet_loader` | ControlNetLoader (Flux union CN) | `control_net_name` |
| `controlnet_apply` | ControlNetApplyAdvanced / XLabsApplyFluxControlNet | `strength` |

Downstream (not patched, just present): DWPreprocessor on `pose_image` →
`controlnet_apply`; VAEDecode; **FaceDetailer** (Impact Pack); **UltimateSDUpscale**
×2 with `4x-UltraSharp`; SaveImage.

CN mode: set the union ControlNet to `openpose`. If using XLabs Flux ControlNet instead
of InstantX union, adjust `config/pipelines.yaml: flux_lora.controlnet` to its filename.

---

## `sdxl_ref.api.json` — Pipeline B

| Node title | Node type | Patched input(s) |
|---|---|---|
| `checkpoint` | CheckpointLoaderSimple | `ckpt_name` |
| `face_ref` | LoadImage | `image` (uploaded `character/hero.png`, subfolder `refs`) |
| `positive` | CLIPTextEncode | `text` |
| `negative` | CLIPTextEncode | `text` |
| `latent` | EmptyLatentImage | `width`, `height` |
| `sampler` | KSampler | `seed`, `steps`, `cfg`, `sampler_name`, `scheduler` |
| `ipadapter` | IPAdapterFaceID | `weight` |
| `instantid_apply` | ApplyInstantID | `weight`, `start_at`, `end_at` |
| `pose_image` | LoadImage | `image` (subfolder `poses`) |
| `controlnet_pose_apply` | ControlNetApplyAdvanced (OpenPoseXL2) | `strength` |

Wiring order that works: checkpoint → InstantID (`InstantIDModelLoader` +
`InstantIDFaceAnalysis` + `ApplyInstantID` off `face_ref`) → IPAdapter FaceID
(`IPAdapterUnifiedLoaderFaceID` + `IPAdapterFaceID` off `face_ref`) → KSampler.
`face_ref` feeds both InstantID and the IP-Adapter. Pose CN applied to the same
positive/negative conditioning. Then VAEDecode → FaceDetailer → UltimateSDUpscale → SaveImage.

InstantID needs `antelopev2` (auto-downloaded by `00_setup.sh`) and its ControlNet
(`instantid-controlnet.safetensors`).

---

## `detail_only.api.json` — optional, for `01_build_dataset.py --stage refine`

| Node title | Node type | Patched input |
|---|---|---|
| `src_image` | LoadImage | `image` |

FaceDetailer + UltimateSDUpscale + SaveImage. If this file is absent, `--stage refine`
just copies images through (expand output already has detailer + upscale).

---

## Smoke test

```
python scripts/03_generate.py --smoke --pipeline sdxl_ref
python scripts/03_generate.py --smoke --pipeline flux_lora --no-lora
```
One image each in `outputs/_smoke/`. If a `KeyError: no node titled ...` fires, a title
is wrong or missing — fix it in the GUI, re-save API format.

# Workflow templates — the contract

`03_generate.py` and `01_build_dataset.py` load `comfyui/<name>.api.json` and patch
nodes **by their `_meta.title`**. The two templates are **pre-authored** in this repo
(`flux_lora.api.json`, `sdxl_ref.api.json`) — no GUI building. Validate them with the
smoke test below once ComfyUI is up; fix any `node_errors` against the live
`/object_info` schema and re-save.

These are **base graphs**: loaders → (identity) → ControlNet pose → KSampler → VAEDecode
→ SaveImage. FaceDetailer + UltimateSDUpscale are added in a second pass (see bottom)
once base generation is confirmed.

---

## `flux_lora.api.json` — Pipeline A

| Node title | Type | Patched by scripts |
|---|---|---|
| `unet` | UNETLoader | `unet_name` |
| `dualclip` | DualCLIPLoader | — (t5xxl_fp8 + clip_l, type flux) |
| `vae` | VAELoader | — (ae.safetensors) |
| `lora` | LoraLoaderModelOnly | `lora_name`, `strength_model` |
| `positive` | CLIPTextEncode | `text` |
| `negative` | CLIPTextEncode | `text` |
| `flux_guidance` | FluxGuidance | `guidance` |
| `latent` | EmptySD3LatentImage | `width`, `height` |
| `pose_image` | LoadImage | `image` (subfolder `poses`; `_neutral.png` when a scene has no pose ref) |
| `dwpose` | DWPreprocessor | — |
| `controlnet_loader` | ControlNetLoader | `control_net_name` |
| `cn_union_type` | SetUnionControlNetType | — (`openpose`) |
| `controlnet_apply` | ControlNetApplyAdvanced | `strength` |
| `sampler` | KSampler | `seed`, `steps`, `cfg` (=1.0 for Flux), `sampler_name`, `scheduler` |
| `vae_decode` | VAEDecode | — |
| `save` | SaveImage | — |

Flux guidance is the `flux_guidance` node (3.5); KSampler `cfg` stays 1.0.
`01_build_dataset.py --stage hero/poses` sets `lora.strength_model=0` and
`controlnet_apply.strength=0` to get base Flux with no identity / no pose lock.

---

## `sdxl_ref.api.json` — Pipeline B

| Node title | Type | Patched by scripts |
|---|---|---|
| `checkpoint` | CheckpointLoaderSimple | `ckpt_name` |
| `face_ref` | LoadImage | `image` (uploaded `character/hero.png`, subfolder `refs`) |
| `positive` | CLIPTextEncode | `text` |
| `negative` | CLIPTextEncode | `text` |
| `latent` | EmptyLatentImage | `width`, `height` |
| `instantid_loader` | InstantIDModelLoader | — |
| `instantid_faceanalysis` | InstantIDFaceAnalysis | — (`CUDA`) |
| `instantid_cn_loader` | ControlNetLoader | — (`instantid-controlnet.safetensors`) |
| `instantid_apply` | ApplyInstantID | `weight`, `start_at`, `end_at` |
| `ipadapter_loader` | IPAdapterUnifiedLoaderFaceID | — (`FACEID PLUS V2`) |
| `ipadapter` | IPAdapterFaceID | `weight` |
| `pose_image` | LoadImage | `image` (subfolder `poses`) |
| `dwpose` | DWPreprocessor | — |
| `pose_cn_loader` | ControlNetLoader | — (`OpenPoseXL2.safetensors`) |
| `controlnet_pose_apply` | ControlNetApplyAdvanced | `strength` |
| `sampler` | KSampler | `seed`, `steps`, `cfg`, `sampler_name`, `scheduler` |
| `vae_decode` | VAEDecode | — |
| `save` | SaveImage | — |

Chain: checkpoint → `ApplyInstantID` (off `face_ref`) → `IPAdapterFaceID` (off `face_ref`)
→ pose ControlNet on the InstantID-modified conditioning → KSampler.
`--stage expand` sets `controlnet_pose_apply.strength=0` for neutral-pose training data.

---

## Smoke test (validates both templates)

```
cd /workspace/ComfyUI && HF_HOME=/workspace/hf python main.py --listen 0.0.0.0 --port 8188   # in tmux
# then, from /workspace/svetlana:
python scripts/03_generate.py --smoke --pipeline sdxl_ref
python scripts/03_generate.py --smoke --pipeline flux_lora --no-lora
```

One image each in `outputs/_smoke/`. `KeyError: no node titled ...` → a title is wrong.
`node_errors` in the queue response → an input name / combo value is wrong for the
installed node version; check `curl -s localhost:8188/object_info | python -m json.tool`.

---

## Second pass — refinement nodes (add after base works)

Append to each graph, re-save API format:

- **FaceDetailer** (Impact Pack) on the VAEDecode output: `model`/`clip`/`vae`/`positive`/
  `negative` from the same loaders, `bbox_detector` from `UltralyticsDetectorProvider`
  (`bbox/face_yolov8m.pt`), `denoise` 0.35, `steps` 20. Flux: `cfg` 1.0.
- **UltimateSDUpscale** after FaceDetailer: `upscale_model` from `UpscaleModelLoader`
  (`4x-UltraSharp.pth`), `upscale_by` 2.0, `denoise` 0.2.
- SaveImage moves to the UltimateSDUpscale output.

`config/pipelines.yaml: common.face_detailer` / `common.upscale` hold the knobs; wire
them as fixed values first, add script patches only if we tune per-pipeline.

`detail_only.api.json` (optional, for `01_build_dataset.py --stage refine`): just
`src_image` LoadImage (title `src_image`) → FaceDetailer → UltimateSDUpscale → SaveImage.
Absent → `--stage refine` copies images through unchanged.

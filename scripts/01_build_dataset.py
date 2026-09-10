#!/usr/bin/env python3
"""
Bootstrap a training set for a synthetic character.

  python 01_build_dataset.py --stage hero   --n 8      # Flux portraits, pick the best by hand
  python 01_build_dataset.py --stage expand --n 32     # SDXL+InstantID+IPA variations off hero.png
  python 01_build_dataset.py --stage refine            # optional detail/upscale pass on external imgs
  python 01_build_dataset.py --stage caption --trigger mara_vance

Flow: run hero -> copy the best to character/hero.png + record its seed in character_spec.md
      -> run expand -> auto-filter with 04_eval.py -> hand-curate to 20-25 -> caption.
"""
import argparse
import shutil
from pathlib import Path

import yaml

import comfy

ROOT = Path(__file__).parent.parent

ANGLES = ["front view", "3/4 left", "3/4 right", "slight profile left", "looking slightly up",
          "looking slightly down", "head tilted"]
EXPR = ["neutral", "faint smile", "concentrating", "mild fatigue", "alert", "quiet satisfaction",
        "thoughtful", "laughing lightly"]
FRAMING = ["tight head and shoulders portrait", "head and shoulders portrait",
           "waist-up shot", "half body shot", "three-quarter body shot", "full body standing shot"]
LIGHT = ["soft window light", "overcast daylight", "golden hour side light", "cool blue hour light",
         "warm indoor lamp light", "bright midday sun", "open shade"]
BG = ["plain studio backdrop", "blurred outdoor greenery", "blurred canvas tent interior",
      "blurred rocky landscape", "blurred stone wall", "blurred village street"]


def canonical_desc():
    lines = (ROOT / "character" / "character_spec.md").read_text().splitlines()
    h = next(k for k, l in enumerate(lines) if l.startswith("## Canonical description"))
    i = next(k for k in range(h + 1, len(lines)) if lines[k].startswith(">"))
    out = []
    while i < len(lines) and lines[i].startswith(">"):
        out.append(lines[i].lstrip("> ").strip()); i += 1
    return " ".join(out)


def neutral_pose_name():
    """Upload the placeholder pose so the graph's LoadImage resolves (CN strength is 0 here)."""
    return comfy.upload_image(ROOT / "character" / "poses" / "_neutral.png", "poses")


def common_cfg():
    return yaml.safe_load((ROOT / "config" / "pipelines.yaml").read_text())["common"]


def stage_hero(n):
    desc = canonical_desc()
    common = common_cfg()
    style = "photorealistic, 35mm portrait, natural light, realistic skin texture, film grain"
    neg = common["base_negative"]
    out = ROOT / "character" / "hero_candidates"
    pose_name = neutral_pose_name()
    for i in range(n):
        wf = comfy.load_workflow("flux_lora")
        comfy.set_input(wf, "lora", "strength_model", 0.0)          # base Flux, no identity yet
        comfy.set_input(wf, "controlnet_apply", "strength", 0.0)    # no pose control
        comfy.set_input(wf, "pose_image", "image", pose_name)
        comfy.set_input(wf, "positive", "text",
                        f"a photo of {desc}, {FRAMING[1]}, {ANGLES[0]}, {EXPR[0]}, {LIGHT[0]}, "
                        f"{BG[0]}. {style}")
        comfy.set_input(wf, "negative", "text", neg)
        comfy.set_input(wf, "sampler", "seed", 1000 + i * 137)
        comfy.set_input(wf, "latent", "width", 896)
        comfy.set_input(wf, "latent", "height", 1152)
        p = comfy.run(wf, out, f"hero_{i:02d}")
        print("  ", p[0].name, " seed", 1000 + i * 137)
    print(f"\npick the best -> cp {out}/hero_XX.png character/hero.png ; record its seed in character_spec.md")


POSES = {
    "stand_relaxed": "standing relaxed, weight on one leg, arms at sides, facing camera",
    "seated_writing": "sitting on a stool leaning over a low table writing, seen from the side",
    "crouch_reach": "crouching low on the ground, one hand reaching forward toward the ground",
    "walk_lookback": "walking away from camera then turning the head to look back over one shoulder",
    "stand_radio": "standing, one hand raised holding a handheld radio near the head",
    "hunch_shield": "hunched forward against strong wind, one forearm raised to shield the face",
    "seated_talk": "sitting on the ground cross-legged, one hand gesturing while talking",
    "lean_examine": "leaning over a table with both hands flat on it, head down looking closely",
    "portrait_front": "head and shoulders, upright, facing the camera directly",
    "lie_back": "lying on the back on the ground, hands behind the head, looking up",
    "stand_inspect": "standing, holding a small object up at eye level with both hands",
    "stand_pack": "standing, pulling a backpack onto both shoulders",
}


def stage_poses():
    """Generic posed figures for the OpenPose ControlNet refs. Identity-free; the
    workflow's DWPreprocessor turns these into skeletons."""
    out = ROOT / "character" / "poses"
    out.mkdir(exist_ok=True)
    neg = common_cfg()["base_negative"]
    pose_name = neutral_pose_name()
    for i, (name, desc) in enumerate(sorted(POSES.items())):
        wf = comfy.load_workflow("flux_lora")
        comfy.set_input(wf, "lora", "strength_model", 0.0)
        comfy.set_input(wf, "controlnet_apply", "strength", 0.0)
        comfy.set_input(wf, "pose_image", "image", pose_name)
        comfy.set_input(wf, "positive", "text",
                        f"full body photo of a person, {desc}, plain light grey studio "
                        f"background, neutral plain clothing, even lighting, sharp focus")
        comfy.set_input(wf, "negative", "text", neg + ", close-up, cropped limbs")
        comfy.set_input(wf, "sampler", "seed", 7000 + i * 53)
        comfy.set_input(wf, "latent", "width", 896)
        comfy.set_input(wf, "latent", "height", 1152)
        p = comfy.run(wf, out, name)
        # comfy.run names a single output <prefix>.png
        print("  ", p[0].name)
    print(f"\n{len(POSES)} pose refs -> {out}")


def stage_expand(n):
    hero = ROOT / "character" / "hero.png"
    if not hero.exists():
        raise SystemExit("run --stage hero first and create character/hero.png")
    desc = canonical_desc()
    common = common_cfg()
    p = yaml.safe_load((ROOT / "config" / "pipelines.yaml").read_text())["pipelines"]["sdxl_ref"]
    style = "photorealistic, 35mm, natural light, realistic skin texture, film grain, sharp focus on face"
    ref_name = comfy.upload_image(hero, "refs")
    pose_name = neutral_pose_name()
    out = ROOT / "character" / "dataset_raw"
    for i in range(n):
        wf = comfy.load_workflow("sdxl_ref")
        comfy.set_input(wf, "pose_image", "image", pose_name)
        prompt = (f"a photo of {desc}, {FRAMING[i % len(FRAMING)]}, {ANGLES[i % len(ANGLES)]}, "
                  f"{EXPR[i % len(EXPR)]}, {LIGHT[i % len(LIGHT)]}, {BG[i % len(BG)]}. {style}")
        comfy.set_input(wf, "checkpoint", "ckpt_name", p["checkpoint"])
        comfy.set_input(wf, "face_ref", "image", ref_name)
        comfy.set_input(wf, "positive", "text", prompt)
        comfy.set_input(wf, "negative", "text", common["base_negative"])
        comfy.set_input(wf, "sampler", "seed", 5000 + i * 91)
        comfy.set_input(wf, "sampler", "cfg", p["cfg"])
        comfy.set_input(wf, "ipadapter", "weight", p["ipadapter_weight"])
        comfy.set_input(wf, "instantid_apply", "weight", p["instantid_weight"])
        comfy.set_input(wf, "controlnet_pose_apply", "strength", 0.0)   # neutral pose for training data
        comfy.set_input(wf, "latent", "width", 896)
        comfy.set_input(wf, "latent", "height", 1152)
        r = comfy.run(wf, out, f"ds_{i:03d}")
        print("  ", r[0].name)
    print(f"\nnext: python 04_eval.py --dir character/dataset_raw --ref character/hero.png --filter 0.60 --out reports/dataset")
    print("then hand-curate the keepers into character/dataset/")


def stage_refine():
    src = ROOT / "character" / "dataset_raw"
    dst = ROOT / "character" / "dataset_refined"
    dst.mkdir(exist_ok=True)
    tpl = ROOT / "comfyui" / "detail_only.api.json"
    if not tpl.exists():
        print("no comfyui/detail_only.api.json -- expand output already has FaceDetailer+upscale; copying through")
        for f in src.glob("*.png"):
            shutil.copy2(f, dst / f.name)
        return
    for f in sorted(src.glob("*.png")):
        name = comfy.upload_image(f, "refine")
        wf = comfy.load_workflow("detail_only")
        comfy.set_input(wf, "src_image", "image", name)
        comfy.run(wf, dst, f.stem)
        print("  ", f.name)


def stage_caption(trigger):
    d = ROOT / "character" / "dataset"
    if not d.exists() or not any(d.glob("*.png")):
        raise SystemExit("put curated images in character/dataset/ first")
    import torch
    from PIL import Image
    from transformers import BlipForConditionalGeneration, BlipProcessor

    # BLIP-large: small, no remote code, no flash_attn -- unlike Florence-2 whose
    # bundled modeling code breaks on current transformers.
    mid = "Salesforce/blip-image-captioning-large"
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    proc = BlipProcessor.from_pretrained(mid)
    model = BlipForConditionalGeneration.from_pretrained(
        mid, torch_dtype=torch.float16 if dev == "cuda" else torch.float32).to(dev).eval()

    for img_path in sorted(d.glob("*.png")):
        img = Image.open(img_path).convert("RGB")
        inp = proc(img, return_tensors="pt").to(dev, model.dtype)
        ids = model.generate(**inp, max_new_tokens=40, num_beams=3)
        cap = proc.decode(ids[0], skip_special_tokens=True).strip().rstrip(".")
        # BLIP often opens with "a woman ..." / "there is ..." -- trim to the descriptive tail
        for pre in ("there is ", "a close up of ", "an image of ", "a picture of ", "arafed "):
            if cap.lower().startswith(pre):
                cap = cap[len(pre):]
        line = f"{trigger}, {cap.lower()}"
        img_path.with_suffix(".txt").write_text(line + "\n")
        print(f"  {img_path.name}: {line[:90]}")
    print("\nskim the .txt files -- fix wrong hair/eye colour, keep pose/clothing/scene words")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True, choices=["hero", "poses", "expand", "refine", "caption"])
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--trigger", default="mara_vance")
    a = ap.parse_args()
    {"hero": lambda: stage_hero(a.n),
     "poses": stage_poses,
     "expand": lambda: stage_expand(a.n),
     "refine": stage_refine,
     "caption": lambda: stage_caption(a.trigger)}[a.stage]()


if __name__ == "__main__":
    main()

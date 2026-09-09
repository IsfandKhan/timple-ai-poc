#!/usr/bin/env python3
"""
Batch-generate every scene in character/scenes.yaml through one pipeline.

  python 03_generate.py --pipeline sdxl_ref --all
  python 03_generate.py --pipeline flux_lora --all
  python 03_generate.py --pipeline flux_lora --scenes 3,7,9 --seed-shift
  python 03_generate.py --smoke --pipeline sdxl_ref          # one quick test image

Requires ComfyUI running (see RUNBOOK H0-H3) and comfyui/<workflow>.api.json present.
Templates use node TITLES as patch points -- see comfyui/NODES.md.
"""
import argparse
from pathlib import Path

import yaml

import comfy

ROOT = Path(__file__).parent.parent


def load_cfg():
    p = yaml.safe_load((ROOT / "config" / "pipelines.yaml").read_text())
    return p["common"], p["pipelines"]


def load_scenes():
    return yaml.safe_load((ROOT / "character" / "scenes.yaml").read_text())


def build_prompt(scene, spec_desc, trigger, global_style):
    subject = f"{trigger}, {spec_desc}" if trigger else spec_desc
    return f"a photo of {subject}. {scene['prompt'].strip()}. {global_style.strip()}"


def canonical_desc():
    # the ">"-quoted block under "Canonical description" in character_spec.md
    text = (ROOT / "character" / "character_spec.md").read_text()
    lines = text.splitlines()
    i = next(k for k, l in enumerate(lines) if l.startswith("> mara_vance"))
    out = []
    while i < len(lines) and lines[i].startswith(">"):
        out.append(lines[i].lstrip("> ").strip())
        i += 1
    return " ".join(out)


def patch_common(wf, common, pipe, prompt, negative, seed, pose_name):
    S = comfy.set_input
    S(wf, "positive", "text", prompt)
    S(wf, "negative", "text", negative)
    S(wf, "sampler", "seed", seed)
    S(wf, "sampler", "steps", common["steps"])
    S(wf, "sampler", "cfg", pipe["cfg"])
    S(wf, "sampler", "sampler_name", pipe["sampler"])
    S(wf, "sampler", "scheduler", pipe["scheduler"])
    S(wf, "latent", "width", common["width"])
    S(wf, "latent", "height", common["height"])
    if pose_name:
        S(wf, "pose_image", "image", pose_name)


def patch_flux(wf, common, pipe):
    S = comfy.set_input
    S(wf, "unet", "unet_name", pipe["checkpoint_unet"])
    S(wf, "lora", "lora_name", pipe["lora"])
    S(wf, "lora", "strength_model", pipe["lora_strength"])
    S(wf, "controlnet_loader", "control_net_name", pipe["controlnet"])
    S(wf, "controlnet_apply", "strength", pipe["controlnet_strength"])


def patch_sdxl(wf, common, pipe, ref_name):
    S = comfy.set_input
    S(wf, "checkpoint", "ckpt_name", pipe["checkpoint"])
    S(wf, "face_ref", "image", ref_name)
    S(wf, "ipadapter", "weight", pipe["ipadapter_weight"])
    S(wf, "instantid_apply", "weight", pipe["instantid_weight"])
    S(wf, "instantid_apply", "start_at", 0.0)
    S(wf, "instantid_apply", "end_at", 1.0)
    S(wf, "controlnet_pose_apply", "strength", pipe["controlnet_openpose_strength"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pipeline", required=True, choices=["flux_lora", "sdxl_ref"])
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--scenes", help="comma list of scene ids, e.g. 3,7,9")
    ap.add_argument("--smoke", action="store_true", help="one image, scene 09, quick")
    ap.add_argument("--seed-shift", action="store_true", help="+1000 to each seed (refinement reruns)")
    ap.add_argument("--no-lora", action="store_true", help="flux smoke test without a trained LoRA")
    args = ap.parse_args()

    common, pipes = load_cfg()
    pipe = pipes[args.pipeline]
    sc = load_scenes()
    scenes = sc["scenes"]
    desc = canonical_desc()
    trigger = pipe.get("trigger", "") if args.pipeline == "flux_lora" else ""
    negative = common["base_negative"] + ", " + sc.get("negative", "")

    if args.smoke:
        scenes = [next(s for s in scenes if s["id"] == "09")]
    elif args.scenes:
        want = set(args.scenes.split(","))
        scenes = [s for s in scenes if s["id"].lstrip("0") in want or s["id"] in want]
    elif not args.all:
        ap.error("pass --all, --scenes, or --smoke")

    out_dir = ROOT / ("outputs/_smoke" if args.smoke else f"outputs/{args.pipeline}")
    ref_name = None
    if args.pipeline == "sdxl_ref":
        ref_name = comfy.upload_image(ROOT / pipe["reference_image"], "refs")

    for scene in scenes:
        wf = comfy.load_workflow(pipe["workflow"])
        seed = scene["seed"] + (1000 if args.seed_shift else 0)
        prompt = build_prompt(scene, desc, trigger, sc["global_style"])
        pose_name = None
        pose_path = ROOT / "character" / "poses" / scene["pose"]
        if pose_path.exists():
            pose_name = comfy.upload_image(pose_path, "poses")

        patch_common(wf, common, pipe, prompt, negative, seed, pose_name)
        if args.pipeline == "flux_lora":
            patch_flux(wf, common, pipe)
            if args.no_lora:
                comfy.set_input(wf, "lora", "strength_model", 0.0)
        else:
            patch_sdxl(wf, common, pipe, ref_name)

        prefix = f"scene_{scene['id']}"
        print(f"[{args.pipeline}] {prefix}: {scene['title']}  seed={seed}")
        paths = comfy.run(wf, out_dir, prefix)
        print("  ->", ", ".join(p.name for p in paths))

    print(f"\ndone. next: python 04_eval.py --dir {out_dir.relative_to(ROOT)} --ref character/hero.png --out reports/{args.pipeline}")


if __name__ == "__main__":
    main()

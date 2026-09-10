#!/usr/bin/env python3
"""
Character-consistency scoring.

Face identity via InsightFace ArcFace (buffalo_l). Embeddings are L2-normalized,
so cosine similarity is just a dot product.

Usage:
  python 04_eval.py --dir outputs/flux_lora --ref character/hero.png --out reports/flux_lora
  python 04_eval.py --dir character/dataset_refined --ref character/hero.png --filter 0.60
  python 04_eval.py --compare outputs/flux_lora outputs/sdxl_ref --out reports/compare
  python 04_eval.py --demo        # offline self-check, no models needed

Headline metric: intra-set mean pairwise similarity (how alike a set's images are
to each other). See PLAN.md for targets.
"""
import argparse
import shutil
import sys
from pathlib import Path

import numpy as np

IMG_EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


# --------------------------------------------------------------------------- math
# (no heavy imports here so --demo runs anywhere)

def cosine_matrix(embs: np.ndarray) -> np.ndarray:
    """embs: (N, D) L2-normalized -> (N, N) cosine similarity."""
    return embs @ embs.T


def pairwise_stats(sim: np.ndarray) -> dict:
    """Stats over the strict upper triangle (each unordered pair once)."""
    n = sim.shape[0]
    if n < 2:
        return {"mean": float("nan"), "min": float("nan"), "std": float("nan"), "pairs": 0}
    iu = np.triu_indices(n, k=1)
    vals = sim[iu]
    return {
        "mean": float(vals.mean()),
        "min": float(vals.min()),
        "std": float(vals.std()),
        "pairs": int(vals.size),
    }


def ref_similarities(embs: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Cosine sim of each row in embs to the reference embedding."""
    return embs @ ref


# ----------------------------------------------------------------- face embedding

_APP = None


def _get_app():
    global _APP
    if _APP is None:
        from insightface.app import FaceAnalysis
        import onnxruntime as ort
        gpu = "CUDAExecutionProvider" in ort.get_available_providers()
        provs = ["CUDAExecutionProvider", "CPUExecutionProvider"] if gpu else ["CPUExecutionProvider"]
        app = FaceAnalysis(name="buffalo_l", providers=provs)
        app.prepare(ctx_id=0 if gpu else -1, det_size=(640, 640))
        _APP = app
    return _APP


def embed_image(path: Path):
    """Largest detected face -> (D,) normed embedding, or None if no face."""
    import cv2

    img = cv2.imread(str(path))
    if img is None:
        return None
    faces = _get_app().get(img)
    if not faces:
        return None
    faces.sort(key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]), reverse=True)
    return faces[0].normed_embedding.astype(np.float32)


def embed_dir(d: Path):
    """-> (names, embs (N,D), missing[list of names with no face])."""
    paths = sorted(p for p in d.iterdir() if p.suffix.lower() in IMG_EXT)
    if not paths:
        sys.exit(f"no images in {d}")
    names, vecs, missing = [], [], []
    for p in paths:
        e = embed_image(p)
        if e is None:
            missing.append(p.name)
            continue
        names.append(p.name)
        vecs.append(e)
    if not vecs:
        sys.exit(f"no faces detected in any image in {d}")
    return names, np.vstack(vecs), missing


# ----------------------------------------------------------------------- reports

def heatmap(sim: np.ndarray, names, out_png: Path, title: str):
    import matplotlib.pyplot as plt

    n = len(names)
    fig, ax = plt.subplots(figsize=(max(6, n * 0.5), max(5, n * 0.5)))
    im = ax.imshow(sim, vmin=0.0, vmax=1.0, cmap="viridis")
    ax.set_xticks(range(n)); ax.set_xticklabels(names, rotation=90, fontsize=7)
    ax.set_yticks(range(n)); ax.set_yticklabels(names, fontsize=7)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{sim[i, j]:.2f}", ha="center", va="center",
                    color="white" if sim[i, j] < 0.6 else "black", fontsize=6)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="cosine similarity")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


def write_report(out_dir: Path, label: str, names, sim, missing, ref_name, ref_sims):
    out_dir.mkdir(parents=True, exist_ok=True)
    ps = pairwise_stats(sim)
    lines = [
        f"# Consistency report — {label}", "",
        f"- images scored: **{len(names)}**",
        f"- faces not detected: **{len(missing)}**" + (f" ({', '.join(missing)})" if missing else ""),
        "", "## Intra-set (images vs each other)", "",
        f"| metric | value | target |",
        f"|---|---|---|",
        f"| mean pairwise similarity | **{ps['mean']:.3f}** | ≥ 0.75 |",
        f"| min pairwise similarity | {ps['min']:.3f} | ≥ 0.55 |",
        f"| std | {ps['std']:.3f} | — |",
        f"| pairs compared | {ps['pairs']} | — |",
    ]
    if ref_sims is not None:
        order = np.argsort(ref_sims)
        lines += [
            "", f"## Vs reference (`{ref_name}`)", "",
            f"- mean similarity to reference: **{float(ref_sims.mean()):.3f}** (target ≥ 0.70)",
            f"- weakest: " + ", ".join(f"{names[i]} ({ref_sims[i]:.2f})" for i in order[:4]),
            "", "| image | sim to ref |", "|---|---|",
            *[f"| {names[i]} | {ref_sims[i]:.3f} |" for i in range(len(names))],
        ]
    lines += ["", f"![heatmap](consistency_heatmap.png)", ""]
    (out_dir / "report.md").write_text("\n".join(lines))
    heatmap(sim, names, out_dir / "consistency_heatmap.png", f"{label} — face similarity")
    print(f"  wrote {out_dir/'report.md'}")


# ------------------------------------------------------------------------- modes

def run_dir(args):
    d = Path(args.dir)
    names, embs, missing = embed_dir(d)
    sim = cosine_matrix(embs)

    ref_name, ref_sims = None, None
    if args.ref:
        ref_e = embed_image(Path(args.ref))
        if ref_e is None:
            sys.exit(f"no face in reference {args.ref}")
        ref_name = Path(args.ref).name
        ref_sims = ref_similarities(embs, ref_e)

    ps = pairwise_stats(sim)
    print(f"{d.name}: n={len(names)} missing={len(missing)} "
          f"mean_pairwise={ps['mean']:.3f} min={ps['min']:.3f}"
          + (f" mean_vs_ref={ref_sims.mean():.3f}" if ref_sims is not None else ""))

    if args.filter is not None:
        if ref_sims is None:
            sys.exit("--filter needs --ref")
        rej = d.parent / f"{d.name}_rejected"
        rej.mkdir(exist_ok=True)
        moved = 0
        for name, s in zip(names, ref_sims):
            if s < args.filter:
                shutil.move(str(d / name), str(rej / name))
                moved += 1
        for name in missing:  # no face at all -> reject
            shutil.move(str(d / name), str(rej / name))
            moved += 1
        print(f"  filtered {moved} images (< {args.filter} or no face) -> {rej}")

    if args.out:
        write_report(Path(args.out), d.name, names, sim, missing, ref_name, ref_sims)


def run_compare(args):
    a, b = Path(args.compare[0]), Path(args.compare[1])
    ref_e = embed_image(Path(args.ref)) if args.ref else None
    rows = []
    for d in (a, b):
        names, embs, missing = embed_dir(d)
        sim = cosine_matrix(embs)
        ps = pairwise_stats(sim)
        rs = ref_similarities(embs, ref_e) if ref_e is not None else None
        rows.append((d.name, names, sim, missing, ps, rs))
        if args.out:
            write_report(Path(args.out) / d.name, d.name, names, sim, missing,
                         Path(args.ref).name if args.ref else None, rs)

    print(f"\n{'metric':<28}{rows[0][0]:>16}{rows[1][0]:>16}")
    print("-" * 60)
    print(f"{'mean pairwise sim':<28}{rows[0][4]['mean']:>16.3f}{rows[1][4]['mean']:>16.3f}")
    print(f"{'min pairwise sim':<28}{rows[0][4]['min']:>16.3f}{rows[1][4]['min']:>16.3f}")
    if rows[0][5] is not None:
        print(f"{'mean sim to reference':<28}{rows[0][5].mean():>16.3f}{rows[1][5].mean():>16.3f}")
    print(f"{'faces missing':<28}{len(rows[0][3]):>16}{len(rows[1][3]):>16}")

    for name, names, sim, missing, ps, rs in rows:
        score = rs if rs is not None else sim.mean(axis=1)
        worst = np.argsort(score)[:4]
        print(f"\nweakest scenes in {name}: " + ", ".join(f"{names[i]} ({score[i]:.2f})" for i in worst))

    if args.out:
        outp = Path(args.out) / "comparison.md"
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text(
            f"# Pipeline comparison\n\n"
            f"| metric | {rows[0][0]} | {rows[1][0]} |\n|---|---|---|\n"
            f"| mean pairwise similarity | {rows[0][4]['mean']:.3f} | {rows[1][4]['mean']:.3f} |\n"
            f"| min pairwise similarity | {rows[0][4]['min']:.3f} | {rows[1][4]['min']:.3f} |\n"
            + (f"| mean similarity to reference | {rows[0][5].mean():.3f} | {rows[1][5].mean():.3f} |\n"
               if rows[0][5] is not None else "")
            + f"| faces missing | {len(rows[0][3])} | {len(rows[1][3])} |\n"
        )
        print(f"\nwrote {outp}")


def demo():
    """Offline self-check on the scoring math. No models, no images."""
    rng = np.random.default_rng(0)

    def norm(v):
        return v / np.linalg.norm(v, axis=-1, keepdims=True)

    # identical vectors -> sim 1; orthogonal -> sim 0
    a = norm(np.array([[1.0, 0, 0], [1.0, 0, 0], [0, 1.0, 0]]))
    sim = cosine_matrix(a)
    assert np.allclose(np.diag(sim), 1.0), "diagonal must be 1"
    assert np.allclose(sim, sim.T), "matrix must be symmetric"
    assert abs(sim[0, 1] - 1.0) < 1e-6, "identical rows -> 1"
    assert abs(sim[0, 2] - 0.0) < 1e-6, "orthogonal rows -> 0"

    ps = pairwise_stats(sim)
    assert ps["pairs"] == 3 and ps["min"] <= ps["mean"], ps

    # a tight cluster must out-score a loose one
    tight = norm(np.array([1.0, 0, 0]) + 0.02 * rng.standard_normal((8, 3)))
    loose = norm(np.array([1.0, 0, 0]) + 0.40 * rng.standard_normal((8, 3)))
    assert pairwise_stats(cosine_matrix(tight))["mean"] > pairwise_stats(cosine_matrix(loose))["mean"]

    # ref similarity ordering
    ref = norm(np.array([1.0, 0, 0]))
    rs = ref_similarities(tight, ref)
    assert rs.shape == (8,) and rs.max() <= 1.0 + 1e-6

    # single image -> nan, no crash
    assert np.isnan(pairwise_stats(np.ones((1, 1)))["mean"])

    print("demo: all checks passed")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dir", help="directory of images to score")
    p.add_argument("--ref", help="reference face image (the canonical character)")
    p.add_argument("--out", help="write report.md + heatmap here")
    p.add_argument("--filter", type=float, metavar="T",
                   help="move images with sim-to-ref < T (or no face) to <dir>_rejected/")
    p.add_argument("--compare", nargs=2, metavar=("DIR_A", "DIR_B"), help="compare two sets")
    p.add_argument("--demo", action="store_true", help="offline self-check")
    args = p.parse_args()

    if args.demo:
        demo()
    elif args.compare:
        run_compare(args)
    elif args.dir:
        run_dir(args)
    else:
        p.error("need one of --dir, --compare, or --demo")


if __name__ == "__main__":
    main()

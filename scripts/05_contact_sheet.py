#!/usr/bin/env python3
"""
Assemble contact sheets for the deliverable. Runs on the Mac.

  python 05_contact_sheet.py                       # one sheet per outputs/<pipeline>/ + a compare sheet
  python 05_contact_sheet.py --dir outputs/sdxl_ref
"""
import argparse
from pathlib import Path

import yaml
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).parent.parent
CELL = 420
PAD = 12
LABEL_H = 26


def _font(sz=15):
    for p in ("/System/Library/Fonts/Helvetica.ttc", "/Library/Fonts/Arial.ttf"):
        if Path(p).exists():
            return ImageFont.truetype(p, sz)
    return ImageFont.load_default()


def titles():
    sc = yaml.safe_load((ROOT / "character" / "scenes.yaml").read_text())
    return {f"scene_{s['id']}": s["title"] for s in sc["scenes"]}


def sheet(img_paths, out_path, header, sub=None):
    t = titles()
    cols = 4
    rows = (len(img_paths) + cols - 1) // cols
    W = cols * CELL + (cols + 1) * PAD
    top = 44 + (18 if sub else 0)
    H = top + rows * (CELL + LABEL_H + PAD) + PAD
    canvas = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(canvas)
    d.text((PAD, 12), header, fill="black", font=_font(20))
    if sub:
        d.text((PAD, 36), sub, fill="#555", font=_font(13))

    for i, p in enumerate(sorted(img_paths)):
        r, c = divmod(i, cols)
        x = PAD + c * (CELL + PAD)
        y = top + r * (CELL + LABEL_H + PAD)
        im = Image.open(p).convert("RGB")
        im.thumbnail((CELL, CELL))
        ox = x + (CELL - im.width) // 2
        canvas.paste(im, (ox, y))
        name = p.stem.split("_seed")[0]
        label = f"{name}  {t.get(name, '')}".strip()
        d.text((x, y + CELL + 4), label[:52], fill="black", font=_font(13))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, quality=92)
    print("wrote", out_path)


def compare_sheet(a_dir, b_dir, out_path):
    t = titles()
    a = {p.stem: p for p in a_dir.glob("scene_*.png")}
    b = {p.stem: p for p in b_dir.glob("scene_*.png")}
    keys = sorted(set(a) & set(b))
    W = 2 * CELL + 3 * PAD + 120
    H = 60 + len(keys) * (CELL + PAD)
    canvas = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(canvas)
    d.text((PAD + 120, 12), a_dir.name, fill="black", font=_font(18))
    d.text((PAD * 2 + 120 + CELL, 12), b_dir.name, fill="black", font=_font(18))
    for i, k in enumerate(keys):
        y = 48 + i * (CELL + PAD)
        d.text((PAD, y + CELL // 2), f"{k}\n{t.get(k, '')}", fill="black", font=_font(12))
        for j, src in enumerate((a[k], b[k])):
            im = Image.open(src).convert("RGB"); im.thumbnail((CELL, CELL))
            canvas.paste(im, (120 + PAD + j * (CELL + PAD), y))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, quality=92)
    print("wrote", out_path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", help="a single outputs/<pipeline> dir")
    a = ap.parse_args()
    rep = ROOT / "reports"

    if a.dir:
        d = ROOT / a.dir
        sheet(list(d.glob("*.png")), rep / f"contact_{d.name}.png", f"Mara Vance - {d.name}")
        return

    dirs = [p for p in (ROOT / "outputs").iterdir() if p.is_dir() and not p.name.startswith("_")]
    for d in dirs:
        imgs = list(d.glob("scene_*.png"))
        if imgs:
            sheet(imgs, rep / f"contact_{d.name}.png", f"Mara Vance - {d.name}",
                  "same character, 12 scenes, one pipeline")
    if len(dirs) == 2:
        compare_sheet(dirs[0], dirs[1], rep / "contact_compare.png")


if __name__ == "__main__":
    main()

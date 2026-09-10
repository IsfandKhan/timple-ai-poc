#!/usr/bin/env python3
"""Faster model fetch: parallel chunked downloads via huggingface_hub + hf_transfer.
Reads config/models.txt (DEST_DIR | FILENAME | URL), places files flat under ComfyUI/models/.
Idempotent — skips files already present. Use instead of the wget loop in 00_setup.sh
when the host network is slow.

  HF_TOKEN=... COMFY=/workspace/ComfyUI python scripts/fetch_models.py
"""
import os
import re
import shutil
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

try:
    import hf_transfer  # noqa: F401
except ImportError:
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
    print("(hf_transfer not installed -- falling back to plain download; "
          "pip install hf_transfer for speed)")

from huggingface_hub import hf_hub_download  # noqa: E402

ROOT = Path(__file__).parent.parent
COMFY = Path(os.environ.get("COMFY", "/workspace/ComfyUI"))
TOKEN = os.environ.get("HF_TOKEN")
URL_RE = re.compile(r"https://huggingface\.co/([^/]+/[^/]+)/resolve/([^/]+)/(.+)")

fails = []
for raw in (ROOT / "config" / "models.txt").read_text().splitlines():
    line = raw.strip()
    if not line or line.startswith("#"):
        continue
    dest, fname, url = (p.strip() for p in line.split("|"))
    target = COMFY / "models" / dest / fname
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.stat().st_size > 100_000:
        print(f"  have {fname}")
        continue
    m = URL_RE.match(url)
    if not m:
        print(f"  !! not an HF url: {url}")
        fails.append(fname)
        continue
    repo, rev, path = m.groups()
    print(f"  get {fname}  ({repo}/{path})")
    try:
        got = Path(hf_hub_download(repo_id=repo, filename=path, revision=rev, token=TOKEN,
                                   local_dir=str(target.parent)))
        if got.resolve() != target.resolve():
            got.replace(target)
            # tidy any now-empty nested dirs the repo path created
            for parent in got.parents:
                if parent == target.parent:
                    break
                try:
                    parent.rmdir()
                except OSError:
                    break
        print(f"     -> {target}  ({target.stat().st_size / 1e9:.2f} GB)")
    except Exception as e:  # noqa: BLE001
        print(f"  !! FAILED {fname}: {e}")
        fails.append(fname)

shutil.rmtree(COMFY / "models" / ".cache", ignore_errors=True)
if fails:
    print("\nFAILED:", ", ".join(fails))
    sys.exit(1)
print("\nall models present")

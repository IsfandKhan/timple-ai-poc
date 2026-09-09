"""Tiny ComfyUI API client. Shared by 01_build_dataset.py and 03_generate.py.

ComfyUI must be running: python main.py --listen 0.0.0.0 --port 8188
"""
import io
import json
import time
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

HOST = "http://127.0.0.1:8188"


def _req(path, data=None):
    url = f"{HOST}{path}"
    if data is not None:
        data = json.dumps(data).encode()
    r = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(r, timeout=600) as resp:
        return json.loads(resp.read())


def load_workflow(name: str) -> dict:
    """Load an API-format workflow template from comfyui/<name>.api.json."""
    p = Path(__file__).parent.parent / "comfyui" / f"{name}.api.json"
    return json.loads(p.read_text())


def set_input(wf: dict, title: str, key: str, value):
    """Patch node input by the node's _meta.title (set titles in ComfyUI, right-click > Title)."""
    for node in wf.values():
        if node.get("_meta", {}).get("title") == title:
            node["inputs"][key] = value
            return
    raise KeyError(f"no node titled {title!r} in workflow")


def upload_image(path: Path, subfolder="") -> str:
    """Upload an input image (pose ref, face ref). Returns the name ComfyUI stores it as."""
    boundary = uuid.uuid4().hex
    body = io.BytesIO()
    def part(name, val, filename=None, ctype=None):
        body.write(f"--{boundary}\r\n".encode())
        disp = f'form-data; name="{name}"'
        if filename:
            disp += f'; filename="{filename}"'
        body.write(f"Content-Disposition: {disp}\r\n".encode())
        if ctype:
            body.write(f"Content-Type: {ctype}\r\n".encode())
        body.write(b"\r\n")
        body.write(val if isinstance(val, bytes) else str(val).encode())
        body.write(b"\r\n")
    part("image", path.read_bytes(), path.name, "image/png")
    part("overwrite", "true")
    if subfolder:
        part("subfolder", subfolder)
    body.write(f"--{boundary}--\r\n".encode())
    r = urllib.request.Request(f"{HOST}/upload/image", data=body.getvalue(),
                               headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    with urllib.request.urlopen(r, timeout=120) as resp:
        j = json.loads(resp.read())
    return (j.get("subfolder", "") + "/" + j["name"]).lstrip("/")


def run(wf: dict, out_dir: Path, prefix: str, timeout=900):
    """Queue a workflow, wait, save all produced images to out_dir/<prefix>*.png. Returns list of paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cid = uuid.uuid4().hex
    pid = _req("/prompt", {"prompt": wf, "client_id": cid})["prompt_id"]

    deadline = time.time() + timeout
    while time.time() < deadline:
        hist = _req(f"/history/{pid}")
        if pid in hist:
            break
        time.sleep(1.5)
    else:
        raise TimeoutError(f"workflow {pid} did not finish in {timeout}s")

    saved = []
    outputs = hist[pid]["outputs"]
    idx = 0
    for node_out in outputs.values():
        for img in node_out.get("images", []):
            q = urllib.parse.urlencode({"filename": img["filename"],
                                        "subfolder": img.get("subfolder", ""),
                                        "type": img.get("type", "output")})
            with urllib.request.urlopen(f"{HOST}/view?{q}", timeout=120) as resp:
                data = resp.read()
            name = f"{prefix}.png" if len(saved) == 0 and _single(outputs) else f"{prefix}_{idx}.png"
            (out_dir / name).write_bytes(data)
            saved.append(out_dir / name)
            idx += 1
    return saved


def _single(outputs):
    return sum(len(o.get("images", [])) for o in outputs.values()) == 1

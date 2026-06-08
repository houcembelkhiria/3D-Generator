#!/usr/bin/env python3
"""Benchmark one end-to-end generation against a running backend.

Usage:
    # 1. Start backend with bf16 off (default):
    #    uvicorn app.main:app --reload --port 8000
    python scripts/bench_weights.py --prompt "red sports car" --texture

    # 2. Restart backend with bf16 on:
    #    HY3D_BF16_TEXGEN=1 uvicorn app.main:app --reload --port 8000
    python scripts/bench_weights.py --prompt "red sports car" --texture

Output: one GLB per run into generated/bench/, named with the launch-time
bf16 flag state reported by /api/v1/unity/launcher-status… no wait, we
read that from a different probe. Uses the OpenAPI-exposed settings where
possible; falls back to labelling by current env.

The script does NOT orchestrate two backend processes itself — the bf16
flag is read at backend startup so each measurement needs its own server.
Run it twice (once per flag state) and compare the two JSON result files.

Generated timings come from the backend's own `generation_time` field plus
any per-stage lines it printed to the terminal (not captured here — watch
the backend logs live).
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

DEFAULT_BACKEND = os.environ.get("BENCH_BACKEND", "http://127.0.0.1:8000")
BENCH_DIR = Path("generated/bench")


def http_post_json(url: str, payload: dict, timeout: float = 600) -> dict:
    data = json.dumps(payload).encode()
    req = Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    with urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def http_get(url: str, timeout: float = 30):
    with urlopen(url, timeout=timeout) as resp:
        return resp.read()


def label_for_current_env() -> str:
    """The server reads HY3D_BF16_TEXGEN at startup; we assume the env this
    script sees matches what the server saw. If not, pass --label explicitly."""
    return "bf16" if os.environ.get("HY3D_BF16_TEXGEN", "").lower() in ("1", "true") else "fp32"


def run_text_to_3d(
    backend: str, prompt: str, *, steps: int, texture: bool, seed: int
) -> dict:
    url = f"{backend}/api/v1/text-to-3d"
    payload = {
        "text": prompt,
        "seed": seed,
        "steps": steps,
        "guidance_scale": 5.0,
        "octree_resolution": 128,
        "num_chunks": 8000,
        "texture": texture,
        "face_count": 20000,
        "output_type": "glb",
    }
    print(f"[bench] POST {url}  steps={steps}  texture={texture}  seed={seed}")
    t0 = time.time()
    body = http_post_json(url, payload)
    wall = time.time() - t0
    body["_client_wall_seconds"] = round(wall, 2)
    return body


def run_image_to_3d(
    backend: str, image_path: Path, *, steps: int, texture: bool, seed: int
) -> dict:
    if not image_path.is_file():
        raise FileNotFoundError(image_path)
    img_b64 = base64.b64encode(image_path.read_bytes()).decode()
    url = f"{backend}/api/v1/image-to-3d"
    payload = {
        "image": img_b64,
        "seed": seed,
        "steps": steps,
        "guidance_scale": 5.0,
        "octree_resolution": 128,
        "num_chunks": 8000,
        "texture": texture,
        "face_count": 20000,
        "output_type": "glb",
    }
    print(f"[bench] POST {url}  steps={steps}  texture={texture}  seed={seed}")
    t0 = time.time()
    body = http_post_json(url, payload)
    wall = time.time() - t0
    body["_client_wall_seconds"] = round(wall, 2)
    return body


def save_artifacts(result: dict, backend: str, label: str, prompt_slug: str) -> Path:
    BENCH_DIR.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    basename = f"{label}_{prompt_slug}_{stamp}"

    # Metadata JSON
    meta_path = BENCH_DIR / f"{basename}.json"
    meta_path.write_text(json.dumps(result, indent=2))

    # Download GLB if the response has a preview/download URL
    preview = result.get("preview_url") or result.get("download_url")
    glb_path = None
    if preview:
        glb_url = preview if preview.startswith("http") else f"{backend}{preview}"
        try:
            blob = http_get(glb_url)
            glb_path = BENCH_DIR / f"{basename}.glb"
            glb_path.write_bytes(blob)
        except (HTTPError, URLError) as exc:
            print(f"[bench] WARNING could not download GLB from {glb_url}: {exc}")

    print(f"[bench] saved   {meta_path}")
    if glb_path:
        print(f"[bench] saved   {glb_path}")
    return meta_path


def slugify(s: str, maxlen: int = 30) -> str:
    return "".join(c if c.isalnum() else "-" for c in s.lower())[:maxlen].strip("-")


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backend", default=DEFAULT_BACKEND, help="Backend base URL")
    p.add_argument("--prompt", help="Text-to-3D prompt")
    p.add_argument("--image", type=Path, help="Image-to-3D source image path")
    p.add_argument("--steps", type=int, default=5, help="Shape-gen inference steps")
    p.add_argument("--seed", type=int, default=1234, help="Generation seed")
    p.add_argument("--no-texture", action="store_true", help="Skip texgen (faster, tests shape-gen only)")
    p.add_argument(
        "--label",
        default=None,
        help="Override label on output files (default: 'bf16' or 'fp32' inferred from HY3D_BF16_TEXGEN env)",
    )
    args = p.parse_args(argv)

    if not args.prompt and not args.image:
        p.error("provide --prompt (text-to-3d) or --image (image-to-3d)")

    label = args.label or label_for_current_env()
    prompt_slug = slugify(args.prompt or args.image.stem)

    print(f"[bench] backend = {args.backend}")
    print(f"[bench] label   = {label}  (HY3D_BF16_TEXGEN env = {os.environ.get('HY3D_BF16_TEXGEN', '<unset>')})")
    print(f"[bench] NOTE the label reflects THIS script's env — confirm the running backend saw the same.")

    try:
        if args.prompt:
            result = run_text_to_3d(
                args.backend, args.prompt,
                steps=args.steps, texture=not args.no_texture, seed=args.seed,
            )
        else:
            result = run_image_to_3d(
                args.backend, args.image,
                steps=args.steps, texture=not args.no_texture, seed=args.seed,
            )
    except HTTPError as exc:
        body = exc.read().decode(errors="replace")
        print(f"[bench] HTTP {exc.code} — {body}", file=sys.stderr)
        return 2
    except URLError as exc:
        print(f"[bench] could not reach {args.backend} — is the backend running? ({exc})", file=sys.stderr)
        return 2

    wall = result.get("_client_wall_seconds")
    gen = result.get("generation_time")
    print()
    print(f"[bench] client wall-clock : {wall} s")
    print(f"[bench] backend gen_time  : {gen} s")

    save_artifacts(result, args.backend, label, prompt_slug)

    print()
    print("To compare runs:")
    print("  ls -lh generated/bench/")
    print("  jq '{_client_wall_seconds, generation_time}' generated/bench/*.json")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

#!/usr/bin/env python3
"""Parse a unity3dgen:// URL and emit a SpawnRequest JSON on stdout.

Expected URL shape:
    unity3dgen://spawn?url=<absolute-glb-url>&scene=new|existing&name=<label>&id=<model-id>

The schema matches UnityProject/Assets/Editor/SpawnRequest.cs.
"""
from __future__ import annotations

import json
import sys
import time
from urllib.parse import parse_qs, unquote, urlparse


def parse(uri: str) -> dict[str, str]:
    p = urlparse(uri)
    if p.scheme != "unity3dgen":
        raise ValueError(f"unsupported scheme: {p.scheme}")

    q = {k: v[0] for k, v in parse_qs(p.query, keep_blank_values=False).items()}

    url = unquote(q.get("url", ""))
    if not url:
        raise ValueError("missing 'url' parameter")

    scene = q.get("scene", "existing").lower()
    if scene not in ("new", "existing"):
        scene = "existing"

    return {
        "id": q.get("id") or f"model_{int(time.time() * 1000)}",
        "url": url,
        "scene": scene,
        "name": unquote(q.get("name", "")),
    }


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: parse.py <unity3dgen-uri>", file=sys.stderr)
        return 2
    try:
        req = parse(sys.argv[1])
    except ValueError as e:
        print(f"parse error: {e}", file=sys.stderr)
        return 1
    json.dump(req, sys.stdout)
    return 0


if __name__ == "__main__":
    sys.exit(main())

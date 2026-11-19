#!/usr/bin/env python
"""Download Wan2.1-T2V-1.3B (Diffusers layout) for area "video" (host step, network on, no GPU).

The repository `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` at the pinned commit goes to `models/wan2.1-t2v-1.3b/` as plain
files (no symbolic links, MM-20), without `assets/` and `examples/` (images for the model card; README "Pins").
huggingface_hub's own download metadata (`<local_dir>/.cache/huggingface/`) is moved out to `hf-meta/`. No token is
used or read: `token=False` on every call, `HF_HUB_DISABLE_IMPLICIT_TOKEN=1`, and `HF_HOME` points to the fresh
`hf-home/` of this directory; a token in the environment refuses the run (its value is never read). Xet transfer is
disabled (plain HTTPS). The OCR models (PaddleOCR reward, docTR deciding metric) are the SD3.5 OCR recipe's (`<sd35-ocr-runtime>/ocr-models/`).

`models/MODELS.json` records repo, revision, license and every file's SHA-256 and bytes; `--verify` re-hashes.
Run with any venv that has huggingface_hub. Licence Apache-2.0 (the model card's `license`); about 28.9 GB.

Usage (host):
  env -u HF_TOKEN python fetch_models.py            # download, write MODELS.json
  python fetch_models.py --verify                   # re-hash every file
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODELS = HERE / "models"
NAME = "wan2.1-t2v-1.3b"
REPO = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
REVISION = "0fad780a534b6463e45facd96134c9f345acfa5b"
IGNORE = ["assets/*", "examples/*"]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return "sha256:" + h.hexdigest()


def plain_files(root: Path) -> dict[str, dict]:
    out = {}
    for p in sorted(root.rglob("*")):
        if p.is_symlink():
            raise SystemExit(f"{p} is a symbolic link; the directory must be plain (MM-20)")
        if p.is_file():
            out[p.relative_to(root).as_posix()] = {"sha256": sha256_file(p), "bytes": p.stat().st_size}
    return out


def download() -> int:
    if os.environ.get("HF_TOKEN") is not None:
        raise SystemExit("HF_TOKEN is set; run with `env -u HF_TOKEN` (this script needs no credential)")
    for name in ("HUGGING_FACE_HUB_TOKEN", "HF_TOKEN_PATH"):
        os.environ.pop(name, None)          # removed, never read
    os.environ["HF_HOME"] = str(HERE / "hf-home")
    os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    (HERE / "hf-home").mkdir(parents=True, exist_ok=True)
    from huggingface_hub import HfApi, snapshot_download

    info = HfApi().model_info(REPO, revision=REVISION, token=False)
    if info.gated:
        raise SystemExit(f"{REPO} is gated; this area downloads ungated repositories only")
    dest = MODELS / NAME
    snapshot_download(REPO, revision=REVISION, local_dir=str(dest), ignore_patterns=IGNORE, token=False)
    meta = dest / ".cache"
    if meta.exists():
        target = HERE / "hf-meta" / NAME
        if target.exists():
            shutil.rmtree(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(meta), str(target))
    files = plain_files(dest)
    want = sorted(s.rfilename for s in info.siblings
                  if not any(s.rfilename.startswith(p.rstrip("*")) for p in IGNORE))
    if sorted(files) != want:
        raise SystemExit(f"{dest} holds {sorted(set(files) ^ set(want))} differently from the repository listing")
    record = {"schema": "ptc_mm.video_models", "schema_version": "1.0", "tags": ["prototype", "unpromotable"],
              "models": {NAME: {"repo": REPO, "revision": REVISION, "license": "apache-2.0", "gated": False,
                                "excluded": IGNORE, "files": files,
                                "bytes": sum(f["bytes"] for f in files.values())}}}
    (MODELS / "MODELS.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"{NAME}: {len(files)} files, {record['models'][NAME]['bytes'] / 1e9:.2f} GB", flush=True)
    return 0


def verify() -> int:
    record = json.loads((MODELS / "MODELS.json").read_text(encoding="utf-8"))
    bad = 0
    for name, m in record["models"].items():
        ok = plain_files(MODELS / name) == m["files"]
        bad += not ok
        print(f"{name}: {'ok' if ok else 'MISMATCH'}", flush=True)
    return 1 if bad else 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--verify", action="store_true")
    return verify() if p.parse_args(argv).verify else download()


if __name__ == "__main__":
    sys.exit(main())

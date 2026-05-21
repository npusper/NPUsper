#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import shutil
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PATCH_ROOT = ROOT / "qai_hub_models_patch" / "modified"
PATCH_FILES = ["model.py", "model_adaptation.py", "test_nstep.py"]


def resolve_site_root() -> Path:
    spec = importlib.util.find_spec("qai_hub_models")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("qai_hub_models is not importable in this interpreter")
    return Path(list(spec.submodule_search_locations)[0])


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Apply the local qai_hub_models whisper patch to the current Python environment."
    )
    ap.add_argument("--check-only", action="store_true",
                    help="Only print current and target paths without copying files")
    args = ap.parse_args()

    site_root = resolve_site_root()
    target_dir = site_root / "models" / "_shared" / "hf_whisper"
    if not target_dir.is_dir():
        raise RuntimeError(f"Target directory not found: {target_dir}")

    print(f"qai_hub_models site root: {site_root}")
    print(f"Target patch dir:        {target_dir}")

    for name in PATCH_FILES:
        src = PATCH_ROOT / name
        dst = target_dir / name
        if not src.is_file():
            raise RuntimeError(f"Patch source not found: {src}")
        print(f"  {src} -> {dst}")
        if not args.check_only:
            shutil.copy2(src, dst)

    if not args.check_only:
        pycache = target_dir / "__pycache__"
        if pycache.exists():
            shutil.rmtree(pycache)
        print("Patch applied.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

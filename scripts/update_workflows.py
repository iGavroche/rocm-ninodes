#!/usr/bin/env python
"""
Update ComfyUI workflow JSONs to current ROCm Ninodes naming conventions.

Usage:
  uv run python scripts/update_workflows.py --paths "C:\\Users\\Nino\\ComfyUI\\user\\default\\workflows" "comfyui_workflows"

Behavior:
- Scans .json files recursively in given paths
- Applies conservative string replacements based on known mappings
- Writes .bak alongside modified files
"""
import argparse
import json
import sys
from pathlib import Path

MAPPINGS = {
    # Class renames
    "ROCMOptimizedUNetLoader": "ROCmDiffusionLoader",
}


def migrate_file(path: Path) -> bool:
    try:
        original = path.read_text(encoding="utf-8")
    except Exception:
        return False
    updated = original
    for old, new in MAPPINGS.items():
        updated = updated.replace(old, new)

    if updated != original:
        backup = path.with_suffix(path.suffix + ".bak")
        backup.write_text(original, encoding="utf-8")
        path.write_text(updated, encoding="utf-8")
        return True
    return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", nargs="+", required=True, help="Directories to scan recursively for .json workflows")
    args = parser.parse_args()

    changed = 0
    checked = 0
    for raw in args.paths:
        root = Path(raw)
        if not root.exists():
            continue
        for p in root.rglob("*.json"):
            checked += 1
            if migrate_file(p):
                changed += 1

    print(f"Checked {checked} workflow(s); updated {changed}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())




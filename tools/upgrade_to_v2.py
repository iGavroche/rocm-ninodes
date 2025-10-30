"""
Upgrade helper for ROCm Ninodes v2

Actions:
- Backup legacy rocm_nodes.py (if present)
- Remove temporary extraction files (temp_*.py)
- Verify new package structure exists
- Print next steps
"""

from __future__ import annotations

import os
import shutil
import glob
import sys


def main() -> int:
    repo_root = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    print(f"[ROCm Ninodes] Upgrade to v2 starting in: {repo_root}")

    # 1) Backup legacy monolithic file if present
    legacy_file = os.path.join(repo_root, "rocm_nodes.py")
    backup_dir = os.path.join(repo_root, "backup")
    if os.path.exists(legacy_file):
        os.makedirs(backup_dir, exist_ok=True)
        backup_target = os.path.join(backup_dir, "rocm_nodes.py.bak")
        shutil.copy2(legacy_file, backup_target)
        print(f"[OK] Backed up legacy rocm_nodes.py -> {backup_target}")
    else:
        print("[INFO] No legacy rocm_nodes.py found (already migrated)")

    # 2) Remove temp extraction files
    removed = 0
    for path in glob.glob(os.path.join(repo_root, "temp_*.py")):
        try:
            os.remove(path)
            removed += 1
        except OSError:
            pass
    if removed:
        print(f"[OK] Removed {removed} temporary file(s)")
    else:
        print("[INFO] No temporary files to remove")

    # 3) Verify package structure
    pkg_dir = os.path.join(repo_root, "rocm_nodes")
    core_dir = os.path.join(pkg_dir, "core")
    utils_dir = os.path.join(pkg_dir, "utils")

    missing = []
    for p in [pkg_dir, core_dir, utils_dir, os.path.join(pkg_dir, "nodes.py")]:
        if not os.path.exists(p):
            missing.append(p)

    if missing:
        print("[ERROR] Missing required paths:")
        for p in missing:
            print(f"  - {p}")
        print("[HINT] Pull the latest code or re-clone the repository")
        return 1

    print("[OK] Verified package structure (rocm_nodes/, core/, utils/, nodes.py)")

    # 4) Final instructions
    print("\nNext steps:")
    print("  1) Restart ComfyUI completely")
    print("  2) Verify nodes appear under 'ROCm Ninodes' categories")
    print("  3) If nodes do not appear, clear ComfyUI cache and restart")

    print("\n[ROCm Ninodes] Upgrade to v2 completed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())



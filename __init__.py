"""
ROCm Ninodes: ROCm Optimized Nodes for ComfyUI
Optimized operations for AMD GPUs with ROCm support

This is the top-level __init__.py that ComfyUI loads.
It imports from the rocm_nodes package structure.
"""

import sys
import os

__version__ = "2.0.2"
__author__ = "iGavroche"
__email__ = "nino2k@proton.me"
__description__ = "ROCm-optimized ComfyUI nodes for AMD GPU performance"

# Ensure the current directory is on sys.path for package imports
_current_dir = os.path.dirname(os.path.abspath(__file__))
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)

# Import from the new package structure
try:
    from rocm_nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
    print("[ROCm Ninodes] Successfully loaded from rocm_nodes package")
except ImportError as e:
    # Fallback for testing or if package structure is incomplete
    print(f"[ROCm Ninodes] Failed to import from rocm_nodes package: {e}")
    try:
        # Try importing from old monolithic structure (backward compatibility)
        import importlib.util
        legacy_path = os.path.join(_current_dir, "rocm_nodes.py")
        if os.path.exists(legacy_path):
            spec = importlib.util.spec_from_file_location("rocm_nodes_legacy", legacy_path)
            legacy_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(legacy_module)
            NODE_CLASS_MAPPINGS = legacy_module.NODE_CLASS_MAPPINGS
            NODE_DISPLAY_NAME_MAPPINGS = legacy_module.NODE_DISPLAY_NAME_MAPPINGS
            print("[ROCm Ninodes] Loaded from legacy rocm_nodes.py")
        else:
            raise ImportError(f"Legacy rocm_nodes.py not found at {legacy_path}")
    except (ImportError, AttributeError) as e2:
        print(f"[ROCm Ninodes] Failed to import from legacy structure: {e2}")
        # Last resort: empty mappings
        NODE_CLASS_MAPPINGS = {}
        NODE_DISPLAY_NAME_MAPPINGS = {}
        print("[ROCm Ninodes] WARNING: No nodes available - check installation")

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', '__version__']

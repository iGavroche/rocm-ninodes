"""
ROCm Ninodes: ROCm Optimized Nodes for ComfyUI
Optimized operations for AMD GPUs with ROCm support

This is the top-level __init__.py that ComfyUI loads.
It imports from the rocm_nodes package structure.
"""

import sys
import os

__version__ = "2.3.3"
__author__ = "iGavroche"
__email__ = "nino2k@proton.me"
__description__ = "ROCm-optimized ComfyUI nodes for AMD GPU performance"

_current_dir = os.path.dirname(os.path.abspath(__file__))
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)

# Probe ROCm memory allocator config at import time
try:
    from rocm_nodes.utils.memory import setup_rocm_memory_config
    setup_rocm_memory_config()
except Exception:
    pass

# Import from the new package structure
try:
    from rocm_nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
    print("[ROCm Ninodes] Successfully loaded from rocm_nodes package")
except ImportError as e:
    print(f"[ROCm Ninodes] Failed to import from rocm_nodes package: {e}")
    # Last resort: empty mappings
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}
    print("[ROCm Ninodes] WARNING: No nodes available - check installation")

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', '__version__']

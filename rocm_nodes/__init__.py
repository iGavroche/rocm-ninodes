"""
RocM Ninodes: ROCM Optimized Nodes for ComfyUI

Main package for ROCm-optimized nodes for AMD GPUs with ROCm support.
Organized into core nodes and utility modules.
"""

__version__ = "2.0.0"
__author__ = "iGavroche"
__email__ = "nino2k@proton.me"
__description__ = "RocM-optimized ComfyUI nodes for AMD GPU performance"

# Import node mappings for ComfyUI compatibility
try:
    from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
except ImportError:
    # Fallback for testing environments
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', '__version__']


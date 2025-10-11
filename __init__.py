"""
RocM Ninodes: ROCM Optimized Nodes for ComfyUI
Optimized operations for AMD GPUs with ROCm support
"""

__version__ = "1.0.0"
__author__ = "Nino"
__email__ = "nino2k@proton.me"
__description__ = "RocM-optimized ComfyUI nodes for AMD GPU performance"

# Import nodes only if comfy modules are available
try:
    from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
except ImportError:
    # Fallback for testing environments
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', '__version__']

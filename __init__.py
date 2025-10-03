"""
RocM-Nino: ROCM Optimized Nodes for ComfyUI
Optimized operations for AMD GPUs with ROCm support
"""

__version__ = "1.0.0"
__author__ = "Nino"
__email__ = "nino@example.com"
__description__ = "ROCM Optimized Nodes for ComfyUI - AMD GPU Performance Optimizations"

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', '__version__']

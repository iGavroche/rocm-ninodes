"""
Node registry for ROCM Ninodes.

This module imports all node classes and creates the NODE_CLASS_MAPPINGS
and NODE_DISPLAY_NAME_MAPPINGS required by ComfyUI.
"""

# Import all node classes from core modules
from .core.vae import (
    ROCMOptimizedVAEDecode,
    ROCMOptimizedVAEDecodeTiled,
    ROCMVAEPerformanceMonitor,
)

from .core.sampler import (
    ROCMOptimizedKSampler,
    ROCMOptimizedKSamplerAdvanced,
    ROCMSamplerPerformanceMonitor,
)

from .core.checkpoint import (
    ROCMOptimizedCheckpointLoader,
)

from .core.unet_loader import (
    ROCmDiffusionLoader,
)

from .core.gguf_loader import (
    ROCmGGUFLoader,
)

from .core.lora import (
    ROCMLoRALoader,
)

from .core.monitors import (
    ROCMFluxBenchmark,
    ROCMMemoryOptimizer,
)

# Define node class mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "ROCMOptimizedCheckpointLoader": ROCMOptimizedCheckpointLoader,
    "ROCmDiffusionLoader": ROCmDiffusionLoader,
    "ROCmGGUFLoader": ROCmGGUFLoader,
    "ROCMOptimizedVAEDecode": ROCMOptimizedVAEDecode,
    "ROCMOptimizedVAEDecodeTiled": ROCMOptimizedVAEDecodeTiled,
    "ROCMVAEPerformanceMonitor": ROCMVAEPerformanceMonitor,
    "ROCMOptimizedKSampler": ROCMOptimizedKSampler,
    "ROCMOptimizedKSamplerAdvanced": ROCMOptimizedKSamplerAdvanced,
    "ROCMSamplerPerformanceMonitor": ROCMSamplerPerformanceMonitor,
    "ROCMFluxBenchmark": ROCMFluxBenchmark,
    "ROCMMemoryOptimizer": ROCMMemoryOptimizer,
    "ROCMLoRALoader": ROCMLoRALoader,
}

# Define display name mappings for ComfyUI
NODE_DISPLAY_NAME_MAPPINGS = {
    "ROCMOptimizedCheckpointLoader": "ROCm Checkpoint Loader",
    "ROCmDiffusionLoader": "ROCm Diffusion Loader",
    "ROCmGGUFLoader": "ROCm GGUF Loader",
    "ROCMOptimizedVAEDecode": "ROCm VAE Decode",
    "ROCMOptimizedVAEDecodeTiled": "ROCm VAE Decode Tiled", 
    "ROCMVAEPerformanceMonitor": "ROCm VAE Performance Monitor",
    "ROCMOptimizedKSampler": "ROCm KSampler",
    "ROCMOptimizedKSamplerAdvanced": "ROCm KSampler Advanced",
    "ROCMSamplerPerformanceMonitor": "ROCm Sampler Performance Monitor",
    "ROCMFluxBenchmark": "ROCm Flux Benchmark",
    "ROCMMemoryOptimizer": "ROCm Memory Optimizer",
    "ROCMLoRALoader": "ROCm LoRA Loader",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']


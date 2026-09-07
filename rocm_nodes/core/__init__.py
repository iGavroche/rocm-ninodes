"""
Core node implementations for ROCM Ninodes.

This module contains all ComfyUI node classes organized by functionality:
- VAE nodes (decode operations)
- Sampler nodes (generation/sampling)
- Checkpoint loader nodes
- UNet/Diffusion Model loader nodes
- LoRA loader nodes
- Monitor/utility nodes
"""

from .vae import (
    ROCmVAEDecode,
    ROCmVAEDecodeTiled,
    ROCmVAEPerformanceMonitor,
)
from .sampler import (
    ROCmKSampler,
    ROCmKSamplerAdvanced,
    ROCmSamplerPerformanceMonitor,
    ROCmSamplerCustomAdvanced,
    ROCmSamplerCustomAdvancedBenchmark,
)
from .checkpoint import ROCmCheckpointLoader
from .unet_loader import ROCmDiffusionLoader
from .lora import ROCmLoRALoader
from .monitors import ROCmFluxBenchmark, ROCmMemoryOptimizer
from .textgen_ltx2 import ROCmTextGenerateLTX2Prompt
try:
    from .h3_easycache import ROCmH3EasyCache
    _HAS_H3_EASYCACHE = True
except ImportError:
    ROCmH3EasyCache = None
    _HAS_H3_EASYCACHE = False

__all__ = [
    # VAE nodes
    'ROCmVAEDecode',
    'ROCmVAEDecodeTiled',
    'ROCmVAEPerformanceMonitor',
    # Sampler nodes
    'ROCmKSampler',
    'ROCmKSamplerAdvanced',
    'ROCmSamplerPerformanceMonitor',
    'ROCmSamplerCustomAdvanced',
    'ROCmSamplerCustomAdvancedBenchmark',
    # Loader nodes
    'ROCmCheckpointLoader',
    'ROCmDiffusionLoader',
    'ROCmLoRALoader',
    # Monitor nodes
    'ROCmFluxBenchmark',
    'ROCmMemoryOptimizer',
    'ROCmTextGenerateLTX2Prompt',
]

if _HAS_H3_EASYCACHE:
    __all__.append('ROCmH3EasyCache')

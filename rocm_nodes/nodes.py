"""
Node registry for ROCM Ninodes.

This module imports all node classes and creates the NODE_CLASS_MAPPINGS
and NODE_DISPLAY_NAME_MAPPINGS required by ComfyUI.
"""

# Import all node classes from core modules
from .core.vae import (
    ROCmVAEDecode,
    ROCmVAEDecodeTiled,
    ROCmVAEPerformanceMonitor,
)

from .core.sampler import (
    ROCmKSampler,
    ROCmKSamplerAdvanced,
    ROCmSamplerPerformanceMonitor,
    ROCmSamplerCustomAdvanced,
    ROCmSamplerCustomAdvancedBenchmark,
)

from .core.checkpoint import (
    ROCmCheckpointLoader,
)

from .core.unet_loader import (
    ROCmDiffusionLoader,
)

from .core.gguf_loader import (
    ROCmGGUFLoader,
)

from .core.lora import (
    ROCmLoRALoader,
)

from .core.monitors import (
    ROCmFluxBenchmark,
    ROCmMemoryOptimizer,
)
from .core.textgen_ltx2 import ROCmTextGenerateLTX2Prompt

# Define node class mappings for ComfyUI.
# Old keys (ROCMOptimized*) are kept as backward-compatible aliases so that
# existing workflows saved with the old naming still load correctly.
NODE_CLASS_MAPPINGS = {
    # --- New canonical names (ROCm prefix) ---
    "ROCmCheckpointLoader": ROCmCheckpointLoader,
    "ROCmDiffusionLoader": ROCmDiffusionLoader,
    "ROCmGGUFLoader": ROCmGGUFLoader,
    "ROCmVAEDecode": ROCmVAEDecode,
    "ROCmVAEDecodeTiled": ROCmVAEDecodeTiled,
    "ROCmVAEPerformanceMonitor": ROCmVAEPerformanceMonitor,
    "ROCmKSampler": ROCmKSampler,
    "ROCmKSamplerAdvanced": ROCmKSamplerAdvanced,
    "ROCmSamplerCustomAdvanced": ROCmSamplerCustomAdvanced,
    "ROCmSamplerPerformanceMonitor": ROCmSamplerPerformanceMonitor,
    "ROCmSamplerCustomAdvancedBenchmark": ROCmSamplerCustomAdvancedBenchmark,
    "ROCmFluxBenchmark": ROCmFluxBenchmark,
    "ROCmMemoryOptimizer": ROCmMemoryOptimizer,
    "ROCmLoRALoader": ROCmLoRALoader,
    "ROCmTextGenerateLTX2Prompt": ROCmTextGenerateLTX2Prompt,

    # --- Legacy aliases (backward compat) ---
    "ROCMOptimizedCheckpointLoader": ROCmCheckpointLoader,
    "ROCMOptimizedVAEDecode": ROCmVAEDecode,
    "ROCMOptimizedVAEDecodeV2Phase3": ROCmVAEDecode,
    "ROCMOptimizedVAEDecodeTiled": ROCmVAEDecodeTiled,
    "ROCMVAEPerformanceMonitor": ROCmVAEPerformanceMonitor,
    "ROCMOptimizedKSampler": ROCmKSampler,
    "ROCMOptimizedKSamplerAdvanced": ROCmKSamplerAdvanced,
    "ROCMSamplerCustomAdvanced": ROCmSamplerCustomAdvanced,
    "ROCMSamplerPerformanceMonitor": ROCmSamplerPerformanceMonitor,
    "ROCMSamplerCustomAdvancedBenchmark": ROCmSamplerCustomAdvancedBenchmark,
    "ROCMFluxBenchmark": ROCmFluxBenchmark,
    "ROCMMemoryOptimizer": ROCmMemoryOptimizer,
    "ROCMLoRALoader": ROCmLoRALoader,
}

# Define display name mappings for ComfyUI
NODE_DISPLAY_NAME_MAPPINGS = {
    "ROCmCheckpointLoader": "ROCm Checkpoint Loader",
    "ROCmDiffusionLoader": "ROCm Diffusion Loader",
    "ROCmGGUFLoader": "ROCm GGUF Loader",
    "ROCmVAEDecode": "ROCm VAE Decode",
    "ROCmVAEDecodeTiled": "ROCm VAE Decode Tiled",
    "ROCmVAEPerformanceMonitor": "ROCm VAE Performance Monitor",
    "ROCMOptimizedVAEDecodeV2Phase3": "ROCm VAE Decode",
    "ROCmKSampler": "ROCm KSampler",
    "ROCmKSamplerAdvanced": "ROCm KSampler Advanced",
    "ROCmSamplerCustomAdvanced": "ROCm SamplerCustomAdvanced",
    "ROCmSamplerPerformanceMonitor": "ROCm Sampler Performance Monitor",
    "ROCmSamplerCustomAdvancedBenchmark": "ROCm SamplerCustomAdvanced Benchmark",
    "ROCmFluxBenchmark": "ROCm Flux Benchmark",
    "ROCmMemoryOptimizer": "ROCm Memory Optimizer",
    "ROCmLoRALoader": "ROCm LoRA Loader",
    "ROCmTextGenerateLTX2Prompt": "ROCm Text Generate LTX2 Prompt",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

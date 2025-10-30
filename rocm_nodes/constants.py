"""
Constants and configuration for ROCM Ninodes.

Shared constants used across the package, including:
- Default tile sizes
- Memory configuration
- Architecture-specific settings
"""

# Default tile sizes for VAE operations
DEFAULT_TILE_SIZE = 768
MIN_TILE_SIZE = 256
MAX_TILE_SIZE = 1024
DEFAULT_TILE_OVERLAP = 96

# Memory configuration
DEFAULT_MEMORY_SAFETY_GB = 2.0
MEMORY_MULTIPLIER_FP32 = 1.5
MEMORY_MULTIPLIER_FP8 = 0.5
MEMORY_MULTIPLIER_INT8 = 0.25
MEMORY_MULTIPLIER_INT4 = 0.25

# Architecture-specific settings
GFX1151_TILE_SIZE_RANGE = (768, 1024)
GFX1151_PREFERRED_PRECISION = "fp32"

# Quantization detection
QUANTIZATION_TYPES = ["fp8", "int8", "int4", "bf16", "fp32"]

# Performance targets
FLUX_TARGET_TIME_SECONDS = 110  # 78% improvement baseline
WAN_TARGET_TIME_SECONDS = 93    # 5.6% improvement baseline
VAE_DECODE_TARGET_SECONDS = 10
CHECKPOINT_LOAD_TARGET_SECONDS = 30

__all__ = [
    'DEFAULT_TILE_SIZE',
    'MIN_TILE_SIZE',
    'MAX_TILE_SIZE',
    'DEFAULT_TILE_OVERLAP',
    'DEFAULT_MEMORY_SAFETY_GB',
    'MEMORY_MULTIPLIER_FP32',
    'MEMORY_MULTIPLIER_FP8',
    'MEMORY_MULTIPLIER_INT8',
    'MEMORY_MULTIPLIER_INT4',
    'GFX1151_TILE_SIZE_RANGE',
    'GFX1151_PREFERRED_PRECISION',
    'QUANTIZATION_TYPES',
    'FLUX_TARGET_TIME_SECONDS',
    'WAN_TARGET_TIME_SECONDS',
    'VAE_DECODE_TARGET_SECONDS',
    'CHECKPOINT_LOAD_TARGET_SECONDS',
]


"""
Utility functions for ROCM Ninodes.

This module contains helper functions organized by purpose:
- Memory management utilities
- ROCm diagnostics
- Quantization detection and handling
- Benchmark utilities
- Debug utilities
"""

from .memory import (
    simple_memory_cleanup,
    gentle_memory_cleanup,
    aggressive_memory_cleanup,
    emergency_memory_cleanup,
    get_gpu_memory_info,
    check_memory_safety,
)
from .diagnostics import log_rocm_diagnostics
from .quantization import (
    detect_model_quantization,
    check_quantized_memory_safety,
    quantized_memory_cleanup,
)
from .debug import (
    DEBUG_MODE,
    save_debug_data,
    capture_timing,
    capture_memory_usage,
    log_debug,
)

__all__ = [
    # Memory utilities
    'simple_memory_cleanup',
    'gentle_memory_cleanup',
    'aggressive_memory_cleanup',
    'emergency_memory_cleanup',
    'get_gpu_memory_info',
    'check_memory_safety',
    # Diagnostics
    'log_rocm_diagnostics',
    # Quantization
    'detect_model_quantization',
    'check_quantized_memory_safety',
    'quantized_memory_cleanup',
    # Debug
    'DEBUG_MODE',
    'save_debug_data',
    'capture_timing',
    'capture_memory_usage',
    'log_debug',
]


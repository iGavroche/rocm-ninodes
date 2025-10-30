"""
Quantization detection and quantization-aware utilities for ROCM Ninodes.

Provides functions for detecting model quantization types and performing
quantization-aware memory calculations.
"""

from typing import Tuple, Optional, Any

import torch

from ..utils.memory import get_gpu_memory_info


def detect_model_quantization(model: Any) -> str:
    """
    Detect if a model is quantized and return quantization type
    
    Args:
        model: Model object with model_dtype() method
        
    Returns:
        Quantization type string: "fp8", "int8", "int4", "bf16", "fp32", or "unknown"
    """
    try:
        # Check model dtype
        model_dtype = model.model_dtype()
        dtype_name = str(model_dtype).lower()
        
        if 'int8' in dtype_name:
            return "int8"
        elif 'int4' in dtype_name:
            return "int4"
        elif 'float8' in dtype_name or 'fp8' in dtype_name:
            return "fp8"
        elif 'bfloat16' in dtype_name or 'bf16' in dtype_name:
            return "bf16"
        else:
            return "fp32"
    except Exception as e:
        print(f"⚠️ Could not detect model quantization: {e}")
        return "unknown"


def check_quantized_memory_safety(model_size: float, quantization_type: str = "fp8") -> Tuple[bool, str]:
    """
    Check memory safety with quantization-aware estimation
    
    Args:
        model_size: Model size in bytes (FP32 equivalent)
        quantization_type: Type of quantization ("fp8", "int8", "int4", "fp32")
        
    Returns:
        Tuple of (is_safe, message)
    """
    if not torch.cuda.is_available():
        return True, "CUDA not available"
    
    total_memory, allocated_memory, reserved_memory, free_memory = get_gpu_memory_info()
    
    if total_memory is None:
        return True, "Memory info unavailable"
    
    # Calculate memory requirements based on quantization
    if quantization_type == "fp8":
        required_memory = model_size * 0.5  # 50% of FP32
        safety_margin = 1.2  # Less aggressive for quantized
    elif quantization_type in ["int8", "int4"]:
        required_memory = model_size * 0.25  # 25% of FP32
        safety_margin = 1.1  # Even less aggressive
    else:
        required_memory = model_size  # Full FP32
        safety_margin = 1.5  # Conservative for non-quantized
    
    required_gb = (required_memory * safety_margin) / (1024**3)
    free_gb = free_memory / (1024**3)
    
    if free_gb < required_gb:
        return False, f"Only {free_gb:.2f}GB free, need {required_gb:.2f}GB for {quantization_type}"
    
    return True, f"Memory OK for {quantization_type}: {free_gb:.2f}GB free"


def quantized_memory_cleanup(quantization_type: str) -> bool:
    """
    Quantization-aware memory cleanup - less aggressive for quantized models
    
    Args:
        quantization_type: Type of quantization ("fp8", "int8", "int4", "fp32")
        
    Returns:
        True if cleanup succeeded, False otherwise
    """
    try:
        if not torch.cuda.is_available():
            return False
        
        # Different cleanup strategies based on quantization type
        if quantization_type in ["fp8", "int8", "int4"]:
            # Quantized models are already memory-efficient, minimal cleanup
            torch.cuda.empty_cache()
            print(f"🧹 Light memory cleanup for {quantization_type} model")
        else:
            # Standard cleanup for non-quantized models
            from ..utils.memory import gentle_memory_cleanup
            gentle_memory_cleanup()
            print(f"🧹 Standard memory cleanup for {quantization_type} model")
        
        return True
    except Exception as e:
        print(f"   Quantized memory cleanup error: {e}")
        return False


__all__ = [
    'detect_model_quantization',
    'check_quantized_memory_safety',
    'quantized_memory_cleanup',
]


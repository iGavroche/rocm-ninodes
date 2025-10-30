"""
Memory management utilities for ROCM Ninodes.

Provides functions for GPU memory cleanup, monitoring, and safety checks
optimized for ROCm and AMD GPUs.
"""

import gc
import logging
from typing import Tuple, Optional

import torch


def simple_memory_cleanup() -> bool:
    """Simple memory cleanup for ROCm"""
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
            return True
        return False
    except Exception as e:
        print(f"   Memory cleanup error: {e}")
        return False


def gentle_memory_cleanup() -> bool:
    """Gentle memory cleanup - less aggressive for mature ROCm drivers"""
    try:
        if not torch.cuda.is_available():
            return False
        
        # Single cache clear with synchronization
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        # Light garbage collection
        gc.collect()
        
        return True
    except Exception as e:
        print(f"   Memory cleanup error: {e}")
        return False


def aggressive_memory_cleanup() -> bool:
    """Gentle memory cleanup - renamed for compatibility"""
    return gentle_memory_cleanup()


def emergency_memory_cleanup() -> bool:
    """Emergency memory cleanup - gentle approach for mature ROCm"""
    try:
        if not torch.cuda.is_available():
            return False
        
        # Gentle cleanup - single pass
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        gc.collect()
        
        return True
    except Exception as e:
        logging.error(f"Emergency memory cleanup failed: {e}")
        return False


def get_gpu_memory_info() -> Tuple[Optional[int], Optional[int], Optional[int], Optional[int]]:
    """Get accurate GPU memory information
    
    Returns:
        Tuple of (total_memory, allocated_memory, reserved_memory, free_memory)
        Returns (None, None, None, None) if CUDA is not available
    """
    if not torch.cuda.is_available():
        return None, None, None, None
    
    total_memory = torch.cuda.get_device_properties(0).total_memory
    allocated_memory = torch.cuda.memory_allocated(0)
    reserved_memory = torch.cuda.memory_reserved(0)
    free_memory = total_memory - reserved_memory
    
    return total_memory, allocated_memory, reserved_memory, free_memory


def check_memory_safety(required_memory_gb: float = 2.0) -> Tuple[bool, str]:
    """Check if there's enough memory for the operation
    
    Args:
        required_memory_gb: Required memory in GB
        
    Returns:
        Tuple of (is_safe, message)
    """
    if not torch.cuda.is_available():
        return True, "CUDA not available"
    
    total_memory, allocated_memory, reserved_memory, free_memory = get_gpu_memory_info()
    
    if total_memory is None:
        return True, "Memory info unavailable"
    
    free_gb = free_memory / (1024**3)
    required_gb = required_memory_gb
    
    if free_gb < required_gb:
        return False, f"Only {free_gb:.2f}GB free, need {required_gb:.2f}GB"
    
    return True, f"Memory OK: {free_gb:.2f}GB free"


__all__ = [
    'simple_memory_cleanup',
    'gentle_memory_cleanup',
    'aggressive_memory_cleanup',
    'emergency_memory_cleanup',
    'get_gpu_memory_info',
    'check_memory_safety',
]


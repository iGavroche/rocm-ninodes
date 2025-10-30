"""
ROCm diagnostics and diagnostic utilities for ROCM Ninodes.

Provides functions for logging ROCm configuration, memory diagnostics,
and system information for debugging and troubleshooting.
"""

import os

import torch


def log_rocm_diagnostics() -> None:
    """Log ROCm memory management configuration for debugging"""
    try:
        print("ROCm Memory Management Diagnostics:")
        print(f"   PYTORCH_CUDA_ALLOC_CONF: {os.environ.get('PYTORCH_CUDA_ALLOC_CONF', 'NOT SET')}")
        print(f"   PYTORCH_HIP_ALLOC_CONF: {os.environ.get('PYTORCH_HIP_ALLOC_CONF', 'NOT SET')}")
        
        if torch.cuda.is_available():
            print(f"   CUDA Available: {torch.cuda.is_available()}")
            print(f"   CUDA Device: {torch.cuda.get_device_name(0)}")
            
            # Get detailed memory info
            total_memory = torch.cuda.get_device_properties(0).total_memory
            allocated_memory = torch.cuda.memory_allocated(0)
            reserved_memory = torch.cuda.memory_reserved(0)
            free_memory = total_memory - allocated_memory
            fragmentation = reserved_memory - allocated_memory
            
            print(f"   Total Memory: {total_memory / 1024**3:.2f}GB")
            print(f"   Allocated Memory: {allocated_memory / 1024**3:.2f}GB")
            print(f"   Reserved Memory: {reserved_memory / 1024**3:.2f}GB")
            print(f"   Free Memory: {free_memory / 1024**3:.2f}GB")
            print(f"   Fragmentation: {fragmentation / 1024**3:.2f}GB")
            
            # Check fragmentation level
            if fragmentation > 500 * 1024**2:  # 500MB
                print(f"   WARNING: High fragmentation detected! This will cause OOM errors.")
                print(f"   Solution: Force memory defragmentation before operations.")
            
            # Check if PyTorch is using the right allocator
            try:
                # Try to get allocator info (this might not work on all PyTorch versions)
                allocator_info = torch.cuda.memory_stats(0)
                print(f"   Memory Allocator: Active (stats available)")
            except:
                print(f"   Memory Allocator: Unknown (stats not available)")
        else:
            print("   CUDA Not Available - using CPU")
            
    except Exception as e:
        print(f"   Diagnostic Error: {e}")


__all__ = ['log_rocm_diagnostics']


"""
ROCm diagnostics and diagnostic utilities for ROCM Ninodes.

Provides functions for logging ROCm configuration, memory diagnostics,
and system information for debugging and troubleshooting.
"""

import os

import torch
from .memory import detect_rocm_version


def log_rocm_diagnostics() -> None:
    """Log ROCm memory management configuration for debugging"""
    try:
        print("ROCm Memory Management Diagnostics:")

        # ── ROCm version ────────────────────────────────────────────────────
        rocm_ver = detect_rocm_version()
        if rocm_ver is not None:
            print(f"   ROCm Version: {rocm_ver[0]}.{rocm_ver[1]}.{rocm_ver[2]}")
            hip_ver = getattr(torch.version, 'hip', 'unknown')
            print(f"   torch.version.hip: {hip_ver}")
        else:
            print("   ROCm Version: Not detected (not on ROCm?)")

        # ── TheRock build system detection ───────────────────────────────────
        if os.path.exists('/opt/rocm/core'):
            print("   TheRock Build System: detected (/opt/rocm/core)")
        elif os.path.exists('/opt/rocm'):
            print("   TheRock Build System: not detected (legacy /opt/rocm layout)")

        print(f"   PYTORCH_CUDA_ALLOC_CONF: {os.environ.get('PYTORCH_CUDA_ALLOC_CONF', 'NOT SET')}")
        print(f"   PYTORCH_HIP_ALLOC_CONF: {os.environ.get('PYTORCH_HIP_ALLOC_CONF', 'NOT SET')}")

        # ── expandable_segments check ────────────────────────────────────────
        hip_alloc = os.environ.get('PYTORCH_HIP_ALLOC_CONF', '')
        cuda_alloc = os.environ.get('PYTORCH_CUDA_ALLOC_CONF', '')
        has_expandable = 'expandable_segments' in hip_alloc or 'expandable_segments' in cuda_alloc
        print(f"   expandable_segments: {'enabled' if has_expandable else 'not set (opt-in via PYTORCH_HIP_ALLOC_CONF)'}")

        # ── TORCH_BLAS_PREFER_HIPBLASLT ─────────────────────────────────────
        blas_pref = os.environ.get('TORCH_BLAS_PREFER_HIPBLASLT', 'not set')
        print(f"   TORCH_BLAS_PREFER_HIPBLASLT: {blas_pref}")

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
            print(f"   Free Memory (total - alloc): {free_memory / 1024**3:.2f}GB")
            print(f"   Fragmentation: {fragmentation / 1024**3:.2f}GB")

            # Check fragmentation level
            if fragmentation > 500 * 1024**2:  # 500MB
                print(f"   WARNING: High fragmentation detected! This can cause OOM errors.")
                print(f"   Solution: Consider setting PYTORCH_HIP_ALLOC_CONF=expandable_segments:True")
                print(f"             or use ROCm Memory Optimizer node with 'aggressive' setting.")

            # Check if PyTorch is using the right allocator
            try:
                allocator_info = torch.cuda.memory_stats(0)
                print(f"   Memory Allocator: Active (stats available)")
            except:
                print(f"   Memory Allocator: Unknown (stats not available)")
        else:
            print("   CUDA Not Available - using CPU")

    except Exception as e:
        print(f"   Diagnostic Error: {e}")


__all__ = ['log_rocm_diagnostics']


"""
Memory management utilities for ROCM Ninodes.

Provides functions for GPU memory cleanup, monitoring, and safety checks
optimized for ROCm and AMD GPUs, including APU (unified memory) awareness.
"""

import gc
import logging
import os
from typing import Tuple, Optional

import torch


_rocm_version_cache = None


def detect_rocm_version() -> Optional[Tuple[int, int, int]]:
    """Parse ROCm version from torch.version.hip.

    Returns:
        (major, minor, patch) tuple, or None if not on ROCm
    """
    global _rocm_version_cache
    if _rocm_version_cache is not None:
        return _rocm_version_cache
    hip_ver = getattr(torch.version, 'hip', None)
    if hip_ver is None:
        _rocm_version_cache = None
        return None
    parts = hip_ver.split(".")
    try:
        major = int(parts[0])
        minor = int(parts[1]) if len(parts) > 1 else 0
        patch = int(parts[2]) if len(parts) > 2 else 0
        _rocm_version_cache = (major, minor, patch)
        return _rocm_version_cache
    except (ValueError, IndexError):
        _rocm_version_cache = None
        return None


def is_rocm_7_14_plus() -> bool:
    """Check if running on ROCm 7.14 or newer."""
    ver = detect_rocm_version()
    return ver is not None and ver >= (7, 14, 0)


def probe_expandable_segments() -> Optional[bool]:
    """Probe whether expandable_segments works with the current ROCm/PyTorch.

    We attempt to set expandable_segments in the allocator config and
    check for errors. Returns True/False if probe succeeded, None if
    not on ROCm or probe inconclusive.
    """
    if not torch.cuda.is_available():
        return None
    if not is_rocm_7_14_plus():
        return None

    current = os.environ.get('PYTORCH_HIP_ALLOC_CONF', '')
    current_cuda = os.environ.get('PYTORCH_CUDA_ALLOC_CONF', '')
    if 'expandable_segments' in current or 'expandable_segments' in current_cuda:
        return True

    # Don't force-enable — user can opt in via env var if stable on their system.
    return False


def setup_rocm_memory_config():
    """Probe and log ROCm memory allocator configuration at startup.

    Does NOT force any setting — only reports what is available.
    expandable_segments is left as an opt-in via environment variable
    since reports on ROCm 7.14 are mixed.
    """
    if not torch.cuda.is_available():
        return

    rocm_ver = detect_rocm_version()
    hip_alloc = os.environ.get('PYTORCH_HIP_ALLOC_CONF', 'not set')
    cuda_alloc = os.environ.get('PYTORCH_CUDA_ALLOC_CONF', 'not set')

    if rocm_ver is not None:
        has_expandable = probe_expandable_segments()
        if has_expandable is False and rocm_ver >= (7, 14, 0):
            print(
                f"  ROCm {rocm_ver[0]}.{rocm_ver[1]}.{rocm_ver[2]} — "
                f"PYTORCH_HIP_ALLOC_CONF={hip_alloc} | "
                f"PYTORCH_CUDA_ALLOC_CONF={cuda_alloc} | "
                f"expandable_segments=not set (opt-in via env var)"
            )


def discard_between_chunks():
    """Aggressive synchronized memory discard between temporal tiles.

    This is the recommended cleanup sequence for the HIP caching
    allocator on APUs (gfx1151). Unlike simple empty_cache(), it
    synchronizes first so the allocator can fully reclaim blocks.
    """
    if not torch.cuda.is_available():
        return
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()


def free_memory_between_tiles(tensor_refs=None):
    """Free tensor references then run aggressive cleanup.

    Handles the common pattern in temporal tiling where the input
    latent chunk must be freed before the next chunk is sliced.

    Args:
        tensor_refs: Optional list of tensor variables to delete
    """
    if tensor_refs:
        for ref in tensor_refs:
            try:
                del ref
            except Exception:
                pass
    discard_between_chunks()


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

        torch.cuda.empty_cache()
        torch.cuda.synchronize()
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

        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        gc.collect()
        return True
    except Exception as e:
        logging.error(f"Emergency memory cleanup failed: {e}")
        return False


def get_gpu_memory_info(is_apu: bool = False) -> Tuple[Optional[int], Optional[int], Optional[int], Optional[int]]:
    """Get accurate GPU memory information

    For discrete GPUs: returns VRAM stats from torch.cuda.
    For APUs (unified memory): returns system-wide available memory
    since GPU and CPU share the same pool.

    Args:
        is_apu: Whether the GPU is an APU with unified memory (e.g. Strix Halo)

    Returns:
        Tuple of (total_memory, allocated_memory, reserved_memory, free_memory)
        Returns (None, None, None, None) if CUDA is not available
    """
    if not torch.cuda.is_available():
        return None, None, None, None

    if is_apu:
        total_memory = torch.cuda.get_device_properties(0).total_memory
        allocated_memory = torch.cuda.memory_allocated(0)
        reserved_memory = torch.cuda.memory_reserved(0)
        import psutil
        system = psutil.virtual_memory()
        free_memory = int(system.available)
        return total_memory, allocated_memory, reserved_memory, free_memory

    total_memory = torch.cuda.get_device_properties(0).total_memory
    allocated_memory = torch.cuda.memory_allocated(0)
    reserved_memory = torch.cuda.memory_reserved(0)
    free_memory = total_memory - reserved_memory

    return total_memory, allocated_memory, reserved_memory, free_memory


def check_memory_safety(required_memory_gb: float = 2.0, is_apu: bool = False) -> Tuple[bool, str]:
    """Check if there's enough memory for the operation

    For APUs, checks against system-wide available memory.
    For discrete GPUs, checks against VRAM.

    Args:
        required_memory_gb: Required memory in GB
        is_apu: Whether the GPU is an APU with unified memory

    Returns:
        Tuple of (is_safe, message)
    """
    if not torch.cuda.is_available():
        return True, "CUDA not available"

    _, _, _, free_memory = get_gpu_memory_info(is_apu=is_apu)

    if free_memory is None:
        return True, "Memory info unavailable"

    free_gb = free_memory / (1024**3)
    required_gb = required_memory_gb

    if free_gb < required_gb:
        return False, f"Only {free_gb:.2f}GB free, need {required_gb:.2f}GB"

    return True, f"Memory OK: {free_gb:.2f}GB free"


def is_apu_architecture(arch_name: str) -> bool:
    """Detect if the GPU architecture is an APU (unified memory)"""
    apu_archs = ["gfx1151", "gfx1150"]
    return any(a in arch_name for a in apu_archs)


__all__ = [
    'simple_memory_cleanup',
    'gentle_memory_cleanup',
    'aggressive_memory_cleanup',
    'emergency_memory_cleanup',
    'get_gpu_memory_info',
    'check_memory_safety',
    'is_apu_architecture',
    'detect_rocm_version',
    'is_rocm_7_14_plus',
    'probe_expandable_segments',
    'setup_rocm_memory_config',
    'discard_between_chunks',
    'free_memory_between_tiles',
]

"""
GPU architecture detection and model type classification.

Shared utilities used by both VAE decode and sampler nodes.
"""

import os
import torch

from ..constants import (
    MAX_TILE_SIZE,
    GFX1151_TILE_SIZE_RANGE, GFX1151_BATCH_CAP, GFX1151_IS_APU,
    GFX1100_TILE_SIZE_RANGE, GFX1100_BATCH_CAP, GFX1100_IS_APU,
    GFX1030_TILE_SIZE_RANGE, GFX1030_BATCH_CAP, GFX1030_IS_APU,
    CDNA_TILE_SIZE_RANGE, CDNA_BATCH_CAP, CDNA_IS_APU,
    DEFAULT_AMD_TILE_SIZE_RANGE, DEFAULT_AMD_BATCH_CAP, DEFAULT_AMD_IS_APU,
)
from .memory import detect_rocm_version


def detect_architecture():
    """Detect GPU architecture and return structured info for tuning.

    Returns:
        dict with keys:
            - arch_name: raw gcnArchName string
            - family: "rdna3_5" | "rdna3" | "rdna2" | "cdna" | "generic_amd" | "cpu"
            - is_apu: bool (unified memory, e.g. Strix Halo)
            - tile_size_max: recommended upper tile bound
            - batch_cap: max batch for tiled operations
            - preferred_precision: "fp16" | "bf16" | "fp32"
            - rocm_version: (major, minor, patch) tuple or None
            - is_rocm_7_14_plus: bool
    """
    rocm_ver = detect_rocm_version()

    if not torch.cuda.is_available():
        return {
            "arch_name": None, "family": "cpu", "is_apu": False,
            "tile_size_max": 2048, "batch_cap": 4,
            "preferred_precision": "fp32",
            "rocm_version": rocm_ver,
            "is_rocm_7_14_plus": rocm_ver is not None and rocm_ver >= (7, 14, 0),
        }

    try:
        arch_name = torch.cuda.get_device_properties(0).gcnArchName
    except Exception:
        arch_name = ""

    base = {
        "rocm_version": rocm_ver,
        "is_rocm_7_14_plus": rocm_ver is not None and rocm_ver >= (7, 14, 0),
    }

    if 'gfx1151' in arch_name or 'gfx1150' in arch_name:
        base.update({
            "arch_name": arch_name, "family": "rdna3_5", "is_apu": GFX1151_IS_APU,
            "tile_size_max": GFX1151_TILE_SIZE_RANGE[1],
            "batch_cap": GFX1151_BATCH_CAP,
            "preferred_precision": "fp16",
        })
        return base
    elif 'gfx110' in arch_name:
        base.update({
            "arch_name": arch_name, "family": "rdna3", "is_apu": GFX1100_IS_APU,
            "tile_size_max": GFX1100_TILE_SIZE_RANGE[1],
            "batch_cap": GFX1100_BATCH_CAP,
            "preferred_precision": "fp16",
        })
        return base
    elif 'gfx103' in arch_name:
        base.update({
            "arch_name": arch_name, "family": "rdna2", "is_apu": GFX1030_IS_APU,
            "tile_size_max": GFX1030_TILE_SIZE_RANGE[1],
            "batch_cap": GFX1030_BATCH_CAP,
            "preferred_precision": "fp16",
        })
        return base
    elif 'gfx90a' in arch_name or 'gfx942' in arch_name:
        base.update({
            "arch_name": arch_name, "family": "cdna", "is_apu": CDNA_IS_APU,
            "tile_size_max": CDNA_TILE_SIZE_RANGE[1],
            "batch_cap": CDNA_BATCH_CAP,
            "preferred_precision": "bf16",
        })
        return base
    else:
        base.update({
            "arch_name": arch_name, "family": "generic_amd", "is_apu": DEFAULT_AMD_IS_APU,
            "tile_size_max": DEFAULT_AMD_TILE_SIZE_RANGE[1],
            "batch_cap": DEFAULT_AMD_BATCH_CAP,
            "preferred_precision": "fp16",
        })
        return base


def detect_model_sampling_type(model) -> dict:
    """Detect model sampling type and properties from a ComfyUI model object.

    Returns:
        dict with keys:
            - model_type: "flow" | "eps" | "v_prediction" | "unknown"
            - latent_channels: int
            - is_pixel_space: bool (z-image-turbo style, no VAE)
            - memory_usage_factor: float
            - has_high_memory: bool (memory_usage_factor > 5)
            - model_architecture: str (e.g. "ltx", "flux", "wan", "sd", "unknown")
    """
    result = {
        "model_type": "unknown",
        "latent_channels": 4,
        "is_pixel_space": False,
        "memory_usage_factor": 1.0,
        "has_high_memory": False,
        "model_architecture": "unknown",
    }

    try:
        inner = model.model
    except Exception:
        return result

    # Model type
    try:
        mt = getattr(inner, 'model_type', None)
        if mt is not None:
            mt_name = getattr(mt, 'name', str(mt)).lower()
            if 'flow' in mt_name:
                result["model_type"] = "flow"
            elif 'v_prediction' in mt_name or 'v' == mt_name:
                result["model_type"] = "v_prediction"
            elif 'eps' in mt_name:
                result["model_type"] = "eps"
    except Exception:
        pass

    # Latent format
    try:
        lf = getattr(inner, 'latent_format', None)
        if lf is not None:
            lc = getattr(lf, 'latent_channels', None)
            if lc is not None:
                result["latent_channels"] = lc
            sdr = getattr(lf, 'spacial_downscale_ratio', None)
            if sdr is not None and sdr == 1 and lc == 3:
                result["is_pixel_space"] = True
    except Exception:
        pass

    # Memory factor
    try:
        muf = getattr(inner, 'memory_usage_factor', None)
        if muf is not None:
            result["memory_usage_factor"] = muf
            result["has_high_memory"] = muf > 5
    except Exception:
        pass

    # Model architecture detection (class name based)
    try:
        class_name = getattr(inner, '__class__', None)
        if class_name is not None:
            cn = class_name.__name__.lower()
            if any(x in cn for x in ('ltx', 'ltxv', 'lightricks')):
                result["model_architecture"] = "ltx"
            elif 'flux' in cn:
                result["model_architecture"] = "flux"
            elif 'wan' in cn:
                result["model_architecture"] = "wan"
            elif 'stable_diffusion' in cn or 'sd_' in cn:
                result["model_architecture"] = "sd"
    except Exception:
        pass

    return result


def select_precision(precision_mode: str, is_quantized: bool, arch_info: dict) -> torch.dtype:
    """Select optimal compute precision based on user setting, model type, and architecture.

    For the sampler: simpler than VAE's version since there's no VAE model dtype to preserve.
    """
    if is_quantized:
        return None

    if precision_mode != "auto":
        dtype_map = {
            "fp32": torch.float32,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
        }
        return dtype_map[precision_mode]

    if arch_info["family"] != "cpu":
        pref = arch_info.get("preferred_precision", "fp16")
        dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
        return dtype_map.get(pref, torch.float16)

    return None


def _check_blas_prefer_hipblaslt(arch_info: dict):
    """Check and warn if TORCH_BLAS_PREFER_HIPBLASLT is not set on gfx1151.

    ROCm docs recommend this env var for LLM performance on RDNA3/RDNA3.5
    with PyTorch < 2.14. It becomes the default in PyTorch 2.14.
    """
    if arch_info.get("family") not in ("rdna3_5", "rdna3"):
        return
    val = os.environ.get("TORCH_BLAS_PREFER_HIPBLASLT", "")
    if val != "1":
        print(
            f"  💡 TORCH_BLAS_PREFER_HIPBLASLT is not set. "
            f"For optimal LLM performance on {arch_info['family']}, set: "
            f"export TORCH_BLAS_PREFER_HIPBLASLT=1"
        )


def apply_rocm_backend_settings(arch_info: dict):
    """Apply ROCm-specific backend settings based on architecture."""
    if arch_info["family"] == "cpu":
        return

    is_amd = arch_info["family"] != "cpu"
    if not is_amd:
        return

    # allow_fp16_accumulation:
    # - On ROCm < 7.14: disabled on rdna3_5 (gfx1151) — causes numerical drift
    #   / illegal memory access in bf16 flow-matching models (LTX, etc.)
    # - On ROCm 7.14+: re-enabled since AMD may have fixed the issue
    is_rdna3_5 = arch_info.get("family") == "rdna3_5"
    is_rocm_7_14_plus = arch_info.get("is_rocm_7_14_plus", False)

    if not is_rdna3_5 or (is_rdna3_5 and is_rocm_7_14_plus):
        torch.backends.cuda.matmul.allow_fp16_accumulation = True
        if is_rdna3_5 and is_rocm_7_14_plus:
            print(
                f"  ROCm 7.14+ detected — re-enabling fp16 accumulation "
                f"on rdna3_5 (was disabled on older ROCm due to numerical drift)"
            )

    # Check TORCH_BLAS_PREFER_HIPBLASLT for LLM perf
    _check_blas_prefer_hipblaslt(arch_info)

    has_hip = bool(getattr(torch.version, 'hip', None))
    if not has_hip:
        return

    supported_attention = ["gfx90a", "gfx942", "gfx950", "gfx1100", "gfx1101", "gfx1150", "gfx1151"]
    if arch_info.get("arch_name") and any(a in arch_info["arch_name"] for a in supported_attention):
        if hasattr(torch.backends.cuda, 'sdpa_kernel'):
            pass  # PyTorch SDPA is available; let it route to HIP SDPA internally

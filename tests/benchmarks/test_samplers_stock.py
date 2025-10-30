"""
Baseline performance checks for stock-equivalent ROCm samplers.
Skips automatically if ComfyUI environment is not available.
"""
import os
import time
import pytest

try:
    import comfy  # type: ignore
    import comfy.sample  # type: ignore
    import comfy.samplers  # type: ignore
    HAS_COMFY = True
except Exception:
    HAS_COMFY = False


pytestmark = pytest.mark.skipif(not HAS_COMFY, reason="ComfyUI not available in test environment")


def _make_dummy_latent(h=64, w=64):
    import torch
    return {"samples": torch.randn(1, 4, h, w)}


def test_rocm_ksampler_stock_path_smoke():
    from rocm_nodes.core.sampler import ROCMOptimizedKSampler

    node = ROCMOptimizedKSampler()
    latent = _make_dummy_latent()

    t0 = time.time()
    out, = node.sample(
        model=None,
        seed=0,
        steps=4,
        cfg=1.0,
        sampler_name='euler',
        scheduler='simple',
        positive={"conditioning": []},
        negative={"conditioning": []},
        latent_image=latent,
        denoise=1.0,
    )
    dt = time.time() - t0

    assert isinstance(out, dict)
    assert "samples" in out
    # basic timing sanity (not a perf assertion)
    assert dt < 60


def test_rocm_ksampler_advanced_stock_path_smoke():
    from rocm_nodes.core.sampler import ROCMOptimizedKSamplerAdvanced

    node = ROCMOptimizedKSamplerAdvanced()
    latent = _make_dummy_latent()

    t0 = time.time()
    out, = node.sample(
        model=None,
        add_noise="enable",
        noise_seed=0,
        steps=4,
        cfg=1.0,
        sampler_name='euler',
        scheduler='simple',
        positive={"conditioning": []},
        negative={"conditioning": []},
        latent_image=latent,
        start_at_step=0,
        end_at_step=4,
        return_with_leftover_noise="disable",
        denoise=1.0,
    )
    dt = time.time() - t0

    assert isinstance(out, dict)
    assert "samples" in out
    assert dt < 60



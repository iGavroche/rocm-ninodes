"""
Unit tests for ROCMOptimizedKSamplerAdvanced
"""
import types
import pytest
import torch

# Ensure comfy mocks exist (conftest sets base modules)
import sys


def _install_comfy_sample_mocks(monkeypatch, captured):
    import comfy

    def fix_empty_latent_channels(model, x):
        return x

    def prepare_noise(latent, seed, batch_inds=None):
        # Deterministic noise same shape as latent
        torch.manual_seed(seed % (2**31))
        return torch.randn_like(latent)

    def sample(model, noise, steps, cfg, sampler_name, scheduler,
               positive, negative, latent_image, denoise,
               noise_mask=None, callback=None, disable_pbar=False, seed=0,
               start_step=0, last_step=None, force_full_denoise=True):
        # Capture args for assertions and return a tensor matching latent shape
        captured.update({
            'noise': noise,
            'steps': steps,
            'cfg': cfg,
            'sampler_name': sampler_name,
            'scheduler': scheduler,
            'latent_image': latent_image,
            'denoise': denoise,
            'start_step': start_step,
            'last_step': last_step,
            'seed': seed,
        })
        return torch.zeros_like(latent_image)

    monkeypatch.setattr(comfy.sample, 'fix_empty_latent_channels', fix_empty_latent_channels, raising=False)
    monkeypatch.setattr(comfy.sample, 'prepare_noise', prepare_noise, raising=False)
    monkeypatch.setattr(comfy.sample, 'sample', sample, raising=False)


class TestROCMOptimizedKSamplerAdvanced:
    def test_add_noise_disabled_uses_latent_as_noise(self, monkeypatch, sample_conditioning):
        from rocm_nodes.core.sampler import ROCMOptimizedKSamplerAdvanced

        # Minimal comfy.utils and preview
        import comfy
        comfy.utils.PROGRESS_BAR_ENABLED = False
        # latent_preview may be imported; ensure dummy
        sys.modules.setdefault('latent_preview', types.ModuleType('latent_preview'))
        import latent_preview
        def prepare_callback(model, steps):
            return None
        setattr(latent_preview, 'prepare_callback', prepare_callback)

        captured = {}
        _install_comfy_sample_mocks(monkeypatch, captured)

        node = ROCMOptimizedKSamplerAdvanced()

        latent = {"samples": torch.randn(1, 4, 32, 32)}

        out, = node.sample(
            model=types.SimpleNamespace(model_dtype=lambda: torch.float32),
            add_noise="disable",
            noise_seed=123,
            steps=4,
            cfg=1.0,
            sampler_name='euler',
            scheduler='simple',
            positive=sample_conditioning,
            negative=sample_conditioning,
            latent_image=latent,
            start_at_step=0,
            end_at_step=10000,
            return_with_leftover_noise="disable",
            use_rocm_optimizations=False,
            precision_mode="auto",
            memory_optimization=False,
            denoise=1.0,
            compatibility_mode=False,
        )

        assert isinstance(out, dict)
        assert 'samples' in out
        # When add_noise is disabled, noise must be a tensor with the same shape
        assert isinstance(captured['noise'], torch.Tensor)
        assert tuple(captured['noise'].shape) == tuple(latent['samples'].shape)
        # last_step should become None when >= steps
        assert captured['last_step'] is None

    def test_step_bounds_normalization(self, monkeypatch, sample_conditioning):
        from rocm_nodes.core.sampler import ROCMOptimizedKSamplerAdvanced

        import comfy
        comfy.utils.PROGRESS_BAR_ENABLED = False
        sys.modules.setdefault('latent_preview', types.ModuleType('latent_preview'))
        import latent_preview
        setattr(latent_preview, 'prepare_callback', lambda model, steps: None)

        captured = {}
        _install_comfy_sample_mocks(monkeypatch, captured)

        node = ROCMOptimizedKSamplerAdvanced()
        latent = {"samples": torch.randn(1, 4, 16, 16)}

        out, = node.sample(
            model=types.SimpleNamespace(model_dtype=lambda: torch.float32),
            add_noise="enable",
            noise_seed=0,
            steps=10,
            cfg=1.0,
            sampler_name='euler',
            scheduler='simple',
            positive=sample_conditioning,
            negative=sample_conditioning,
            latent_image=latent,
            start_at_step=-5,   # should clamp to 0
            end_at_step=10,     # should normalize to None
            return_with_leftover_noise="disable",
            use_rocm_optimizations=False,
            precision_mode="auto",
            memory_optimization=False,
            denoise=1.0,
            compatibility_mode=False,
        )

        assert isinstance(out, dict)
        assert captured['start_step'] == 0
        assert captured['last_step'] is None



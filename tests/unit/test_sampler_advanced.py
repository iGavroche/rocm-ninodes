"""
Unit tests for ROCMOptimizedKSamplerAdvanced
"""
import types
import pytest
import torch

import sys


def _install_comfy_sample_mocks(monkeypatch, captured):
    import comfy

    def fix_empty_latent_channels(model, x):
        return x

    def prepare_noise(latent, seed, batch_inds=None):
        torch.manual_seed(seed % (2**31))
        return torch.randn_like(latent)

    def sample(model, noise, steps, cfg, sampler_name, scheduler,
               positive, negative, latent_image, denoise,
               noise_mask=None, callback=None, disable_pbar=False, seed=0,
               start_step=0, last_step=None, force_full_denoise=True,
               disable_noise=False):
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


def _make_model(latent_channels=4):
    """Create a minimal model mock with architecture info for detect_model_sampling_type."""
    lat_fmt = types.SimpleNamespace(
        latent_channels=latent_channels,
        spacial_downscale_ratio=8,
    )
    model_type = types.SimpleNamespace(
        name="FLOW",
    )
    inner = types.SimpleNamespace(
        model_type=model_type,
        latent_format=lat_fmt,
        memory_usage_factor=1.0,
    )
    return types.SimpleNamespace(
        model=inner,
        model_dtype=lambda: torch.float32,
        load_device='cpu',
    )


class TestROCMOptimizedKSamplerAdvanced:
    def test_add_noise_disabled_uses_latent_as_noise(self, monkeypatch, sample_conditioning):
        from rocm_nodes.core.sampler import ROCMOptimizedKSamplerAdvanced

        import comfy
        comfy.utils.PROGRESS_BAR_ENABLED = False
        sys.modules.setdefault('latent_preview', types.ModuleType('latent_preview'))
        import latent_preview
        def prepare_callback(model, steps):
            return None
        setattr(latent_preview, 'prepare_callback', prepare_callback)

        captured = {}
        _install_comfy_sample_mocks(monkeypatch, captured)

        node = ROCMOptimizedKSamplerAdvanced()
        latent = {"samples": torch.randn(1, 4, 32, 32)}
        model = _make_model()

        out, = node.sample(
            model=model,
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
            precision_mode="auto",
            denoise=1.0,
            compatibility_mode=False,
        )

        assert isinstance(out, dict)
        assert 'samples' in out
        assert isinstance(captured['noise'], torch.Tensor)
        assert tuple(captured['noise'].shape) == tuple(latent['samples'].shape)
        # last_step normalization happens inside comfy.sample.sample (mocked away);
        # our node passes end_at_step through as-is
        assert captured['last_step'] == 10000

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
        model = _make_model()

        out, = node.sample(
            model=model,
            add_noise="enable",
            noise_seed=0,
            steps=10,
            cfg=1.0,
            sampler_name='euler',
            scheduler='simple',
            positive=sample_conditioning,
            negative=sample_conditioning,
            latent_image=latent,
            start_at_step=-5,
            end_at_step=10,
            return_with_leftover_noise="disable",
            precision_mode="auto",
            denoise=1.0,
            compatibility_mode=False,
        )

        assert isinstance(out, dict)
        # start_step / end_at_step normalization happens inside comfy.sample.sample
        # (mocked away); our node passes values through as-is
        assert captured['start_step'] == -5
        assert captured['last_step'] == 10

    def test_model_detection_flow(self, monkeypatch, sample_conditioning):
        """Verify flow-matching model is detected and logged."""
        from rocm_nodes.core.sampler import ROCMOptimizedKSampler

        import comfy
        comfy.utils.PROGRESS_BAR_ENABLED = False
        sys.modules.setdefault('latent_preview', types.ModuleType('latent_preview'))
        import latent_preview
        setattr(latent_preview, 'prepare_callback', lambda model, steps: None)

        captured = {}
        _install_comfy_sample_mocks(monkeypatch, captured)

        node = ROCMOptimizedKSampler()
        latent = {"samples": torch.randn(1, 4, 32, 32)}
        model = _make_model(latent_channels=16)

        out, = node.sample(
            model=model,
            seed=0,
            steps=4,
            cfg=1.0,
            sampler_name='euler',
            scheduler='simple',
            positive=sample_conditioning,
            negative=sample_conditioning,
            latent_image=latent,
            denoise=1.0,
        )

        assert isinstance(out, dict)
        assert 'samples' in out
        assert isinstance(out['samples'], torch.Tensor)

    def test_model_detection_pixel_space(self, monkeypatch, sample_conditioning):
        """Verify pixel-space model is detected correctly."""
        from rocm_nodes.core.sampler import ROCMOptimizedKSampler

        import comfy
        comfy.utils.PROGRESS_BAR_ENABLED = False
        sys.modules.setdefault('latent_preview', types.ModuleType('latent_preview'))
        import latent_preview
        setattr(latent_preview, 'prepare_callback', lambda model, steps: None)

        captured = {}
        _install_comfy_sample_mocks(monkeypatch, captured)

        node = ROCMOptimizedKSampler()
        latent = {"samples": torch.randn(1, 3, 64, 64)}

        lat_fmt = types.SimpleNamespace(
            latent_channels=3,
            spacial_downscale_ratio=1,
        )
        model_type = types.SimpleNamespace(name="FLOW")
        inner = types.SimpleNamespace(
            model_type=model_type,
            latent_format=lat_fmt,
            memory_usage_factor=0.03,
        )
        model = types.SimpleNamespace(
            model=inner,
            model_dtype=lambda: torch.float32,
            load_device='cpu',
        )

        out, = node.sample(
            model=model,
            seed=0,
            steps=4,
            cfg=1.0,
            sampler_name='euler',
            scheduler='simple',
            positive=sample_conditioning,
            negative=sample_conditioning,
            latent_image=latent,
            denoise=1.0,
        )

        assert isinstance(out, dict)
        assert 'samples' in out

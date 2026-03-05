"""
Unit tests for ROCm Checkpoint Loader cache.

Tests cache hit/miss, force_reload bypass, use_cache behavior, and single-slot eviction.
"""

import pytest
from unittest.mock import Mock, patch

from rocm_nodes.core import checkpoint as checkpoint_module
from rocm_nodes.core.checkpoint import ROCMOptimizedCheckpointLoader


def _clear_checkpoint_cache():
    """Reset module-level cache so tests are isolated."""
    checkpoint_module._checkpoint_cache.clear()
    checkpoint_module._checkpoint_cache_key = None


@pytest.fixture(autouse=True)
def reset_cache():
    """Clear checkpoint cache before and after each test."""
    _clear_checkpoint_cache()
    yield
    _clear_checkpoint_cache()


@pytest.fixture
def loader():
    """Create loader instance."""
    return ROCMOptimizedCheckpointLoader()


@pytest.fixture
def mock_model_clip_vae():
    """Fake (model, clip, vae) tuple returned by load_checkpoint_guess_config."""
    return (Mock(name="model"), Mock(name="clip"), Mock(name="vae"))


class TestCheckpointLoaderInputTypes:
    """Test INPUT_TYPES and optional cache parameters."""

    def test_required_ckpt_name(self, loader):
        input_types = loader.INPUT_TYPES()
        assert "ckpt_name" in input_types["required"]

    def test_optional_use_cache_and_force_reload(self, loader):
        input_types = loader.INPUT_TYPES()
        optional = input_types["optional"]
        assert "use_cache" in optional
        assert optional["use_cache"][1]["default"] is True
        assert "force_reload" in optional
        assert optional["force_reload"][1]["default"] is False


class TestCheckpointLoaderCache:
    """Test cache hit, miss, force_reload, and eviction."""

    @patch("rocm_nodes.core.checkpoint.gentle_memory_cleanup")
    @patch("rocm_nodes.core.checkpoint.folder_paths.get_full_path_or_raise")
    @patch("rocm_nodes.core.checkpoint.comfy.sd.load_checkpoint_guess_config")
    def test_cache_miss_then_hit(
        self, mock_load, mock_get_path, mock_cleanup, loader, mock_model_clip_vae
    ):
        """First load is cache miss (load called); second load with same ckpt is cache hit (load not called)."""
        mock_get_path.return_value = "/fake/path/ckpt.safetensors"
        mock_load.return_value = mock_model_clip_vae

        out1 = loader.load_checkpoint("ckpt.safetensors", use_cache=True, force_reload=False)
        assert out1 == mock_model_clip_vae
        assert mock_load.call_count == 1

        out2 = loader.load_checkpoint("ckpt.safetensors", use_cache=True, force_reload=False)
        assert out2 == mock_model_clip_vae
        assert mock_load.call_count == 1  # still 1: cache hit
        assert out1[0] is out2[0] and out1[1] is out2[1] and out1[2] is out2[2]  # same refs

    @patch("rocm_nodes.core.checkpoint.gentle_memory_cleanup")
    @patch("rocm_nodes.core.checkpoint.folder_paths.get_full_path_or_raise")
    @patch("rocm_nodes.core.checkpoint.comfy.sd.load_checkpoint_guess_config")
    def test_force_reload_bypasses_cache(
        self, mock_load, mock_get_path, mock_cleanup, loader, mock_model_clip_vae
    ):
        """force_reload=True causes load from disk even when cache has the checkpoint."""
        mock_get_path.return_value = "/fake/path/ckpt.safetensors"
        mock_load.return_value = mock_model_clip_vae

        loader.load_checkpoint("ckpt.safetensors", use_cache=True, force_reload=False)
        assert mock_load.call_count == 1

        loader.load_checkpoint("ckpt.safetensors", use_cache=True, force_reload=True)
        assert mock_load.call_count == 2

    @patch("rocm_nodes.core.checkpoint.gentle_memory_cleanup")
    @patch("rocm_nodes.core.checkpoint.folder_paths.get_full_path_or_raise")
    @patch("rocm_nodes.core.checkpoint.comfy.sd.load_checkpoint_guess_config")
    def test_use_cache_false_still_loads_and_caches(
        self, mock_load, mock_get_path, mock_cleanup, loader, mock_model_clip_vae
    ):
        """use_cache=False still loads and stores in cache; next call with use_cache=True can hit."""
        mock_get_path.return_value = "/fake/path/ckpt.safetensors"
        mock_load.return_value = mock_model_clip_vae

        out1 = loader.load_checkpoint("ckpt.safetensors", use_cache=False, force_reload=False)
        assert out1 == mock_model_clip_vae
        assert mock_load.call_count == 1

        # Cache was populated; with use_cache=True we get cache hit
        out2 = loader.load_checkpoint("ckpt.safetensors", use_cache=True, force_reload=False)
        assert out2 == mock_model_clip_vae
        assert mock_load.call_count == 1
        assert out1[0] is out2[0] and out1[1] is out2[1] and out1[2] is out2[2]

    @patch("rocm_nodes.core.checkpoint.gentle_memory_cleanup")
    @patch("rocm_nodes.core.checkpoint.folder_paths.get_full_path_or_raise")
    @patch("rocm_nodes.core.checkpoint.comfy.sd.load_checkpoint_guess_config")
    def test_eviction_when_loading_different_ckpt(
        self, mock_load, mock_get_path, mock_cleanup, loader, mock_model_clip_vae
    ):
        """Loading a different ckpt_name evicts previous cache and calls gentle_memory_cleanup."""
        mock_get_path.side_effect = lambda folder, name: f"/fake/{name}"
        mock_load.return_value = mock_model_clip_vae

        loader.load_checkpoint("ckpt_a.safetensors", use_cache=True, force_reload=False)
        assert mock_load.call_count == 1
        assert checkpoint_module._checkpoint_cache_key == "ckpt_a.safetensors"
        assert "ckpt_a.safetensors" in checkpoint_module._checkpoint_cache

        # Different checkpoint: evict and load
        other_triple = (Mock(name="m2"), Mock(name="c2"), Mock(name="v2"))
        mock_load.return_value = other_triple
        loader.load_checkpoint("ckpt_b.safetensors", use_cache=True, force_reload=False)

        assert mock_load.call_count == 2
        assert mock_cleanup.call_count >= 1
        assert checkpoint_module._checkpoint_cache_key == "ckpt_b.safetensors"
        assert "ckpt_a.safetensors" not in checkpoint_module._checkpoint_cache
        assert "ckpt_b.safetensors" in checkpoint_module._checkpoint_cache

    @patch("rocm_nodes.core.checkpoint.gentle_memory_cleanup")
    @patch("rocm_nodes.core.checkpoint.folder_paths.get_full_path_or_raise")
    @patch("rocm_nodes.core.checkpoint.comfy.sd.load_checkpoint_guess_config")
    def test_return_types_and_names(self, mock_load, mock_get_path, mock_cleanup, loader):
        """Loader returns MODEL, CLIP, VAE."""
        mock_get_path.return_value = "/fake/path/ckpt.safetensors"
        mock_load.return_value = (Mock(), Mock(), Mock())

        assert loader.RETURN_TYPES == ("MODEL", "CLIP", "VAE")
        assert loader.RETURN_NAMES == ("MODEL", "CLIP", "VAE")
        result = loader.load_checkpoint("ckpt.safetensors")
        assert len(result) == 3

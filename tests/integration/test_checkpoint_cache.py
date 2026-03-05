"""
Integration tests for ROCm Checkpoint Loader cache.

Verifies that when the same checkpoint is requested twice (e.g. two API runs),
the second run does not reload from disk (load_checkpoint_guess_config called once).
"""

import pytest
from unittest.mock import Mock, patch

from rocm_nodes.core import checkpoint as checkpoint_module
from rocm_nodes.core.checkpoint import ROCMOptimizedCheckpointLoader


def _clear_checkpoint_cache():
    checkpoint_module._checkpoint_cache.clear()
    checkpoint_module._checkpoint_cache_key = None


@pytest.fixture(autouse=True)
def reset_cache():
    _clear_checkpoint_cache()
    yield
    _clear_checkpoint_cache()


@pytest.fixture
def mock_model_clip_vae():
    return (Mock(name="model"), Mock(name="clip"), Mock(name="vae"))


@patch("rocm_nodes.core.checkpoint.gentle_memory_cleanup")
@patch("rocm_nodes.core.checkpoint.folder_paths.get_full_path_or_raise")
@patch("rocm_nodes.core.checkpoint.comfy.sd.load_checkpoint_guess_config")
def test_two_runs_same_checkpoint_only_loads_once(
    mock_load, mock_get_path, mock_cleanup, mock_model_clip_vae
):
    """
    Simulate two API runs with the same checkpoint: load_checkpoint_guess_config
    must be called only once (second run uses cache).
    """
    mock_get_path.return_value = "/fake/checkpoints/ltx-2-19b-dev-fp8.safetensors"
    mock_load.return_value = mock_model_clip_vae

    loader = ROCMOptimizedCheckpointLoader()

    # First "API run"
    out1 = loader.load_checkpoint(
        "ltx-2-19b-dev-fp8.safetensors", use_cache=True, force_reload=False
    )
    assert len(out1) == 3
    assert mock_load.call_count == 1

    # Second "API run" with same workflow (same checkpoint)
    out2 = loader.load_checkpoint(
        "ltx-2-19b-dev-fp8.safetensors", use_cache=True, force_reload=False
    )
    assert len(out2) == 3
    assert mock_load.call_count == 1  # no second load
    assert out1[0] is out2[0] and out1[1] is out2[1] and out1[2] is out2[2]


@patch("rocm_nodes.core.checkpoint.gentle_memory_cleanup")
@patch("rocm_nodes.core.checkpoint.folder_paths.get_full_path_or_raise")
@patch("rocm_nodes.core.checkpoint.comfy.sd.load_checkpoint_guess_config")
def test_force_reload_on_second_run_calls_load_again(
    mock_load, mock_get_path, mock_cleanup, mock_model_clip_vae
):
    """Second run with force_reload=True triggers a reload (e.g. user requested refresh)."""
    mock_get_path.return_value = "/fake/checkpoints/ckpt.safetensors"
    mock_load.return_value = mock_model_clip_vae

    loader = ROCMOptimizedCheckpointLoader()
    loader.load_checkpoint("ckpt.safetensors", use_cache=True, force_reload=False)
    assert mock_load.call_count == 1

    loader.load_checkpoint("ckpt.safetensors", use_cache=True, force_reload=True)
    assert mock_load.call_count == 2

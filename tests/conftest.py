"""
Test suite for ROCM Ninodes
Comprehensive testing infrastructure for all ROCM-optimized nodes
"""
import sys
import os
import pytest
import torch
import numpy as np
from pathlib import Path
from unittest.mock import Mock

# Add ComfyUI to path
sys.path.insert(0, '/home/nino/ComfyUI')

# Mock ComfyUI modules BEFORE importing nodes
import types
comfy = types.ModuleType('comfy')
comfy.model_management = types.ModuleType('comfy.model_management')
comfy.utils = types.ModuleType('comfy.utils')
comfy.sd = types.ModuleType('comfy.sd')
comfy.samplers = types.ModuleType('comfy.samplers')
comfy.sample = types.ModuleType('comfy.sample')

# Mock comfy.cli_args and latent_preview so rocm_nodes.core.vae can be imported
comfy.cli_args = types.ModuleType('comfy.cli_args')
comfy.cli_args.args = type('Args', (), {'preview_method': None, 'base_directory': None})()
comfy.cli_args.LatentPreviewMethod = type('LatentPreviewMethod', (), {})
latent_preview = types.ModuleType('latent_preview')
latent_preview.get_previewer = lambda *args, **kwargs: None

# Mock WAN and LTX packages
comfy.ldm = types.ModuleType('comfy.ldm')
comfy.ldm.wan = types.ModuleType('comfy.ldm.wan')
comfy.ldm.wan.vae = types.ModuleType('comfy.ldm.wan.vae')
comfy.ldm.wan.vae2_2 = types.ModuleType('comfy.ldm.wan.vae2_2')

class MockWanVAE:
    pass

class MockWanVAE2_2:
    pass

comfy.ldm.wan.vae.WanVAE = MockWanVAE
comfy.ldm.wan.vae2_2.WanVAE = MockWanVAE2_2

comfy.ldm.lightricks = types.ModuleType('comfy.ldm.lightricks')
comfy.ldm.lightricks.vae = types.ModuleType('comfy.ldm.lightricks.vae')
comfy.ldm.lightricks.vae.causal_video_autoencoder = types.ModuleType('comfy.ldm.lightricks.vae.causal_video_autoencoder')

class MockVideoVAE:
    pass

comfy.ldm.lightricks.vae.causal_video_autoencoder.VideoVAE = MockVideoVAE

# Add modules to sys.modules BEFORE any imports
sys.modules['comfy'] = comfy
sys.modules['comfy.model_management'] = comfy.model_management
sys.modules['comfy.utils'] = comfy.utils
sys.modules['comfy.sd'] = comfy.sd
sys.modules['comfy.samplers'] = comfy.samplers
sys.modules['comfy.sample'] = comfy.sample
sys.modules['comfy.cli_args'] = comfy.cli_args
sys.modules['latent_preview'] = latent_preview
sys.modules['comfy.ldm'] = comfy.ldm
sys.modules['comfy.ldm.wan'] = comfy.ldm.wan
sys.modules['comfy.ldm.wan.vae'] = comfy.ldm.wan.vae
sys.modules['comfy.ldm.wan.vae2_2'] = comfy.ldm.wan.vae2_2
sys.modules['comfy.ldm.lightricks'] = comfy.ldm.lightricks
sys.modules['comfy.ldm.lightricks.vae'] = comfy.ldm.lightricks.vae
sys.modules['comfy.ldm.lightricks.vae.causal_video_autoencoder'] = comfy.ldm.lightricks.vae.causal_video_autoencoder

# Mock functions
comfy.model_management.load_models_gpu = lambda *args, **kwargs: None
comfy.sd.load_checkpoint_guess_config = lambda *args, **kwargs: (Mock(), Mock(), Mock())
comfy.model_management.get_free_memory = lambda *args, **kwargs: 16 * 1024**3
comfy.model_management.minimum_inference_memory = lambda *args, **kwargs: 8 * 1024**3
comfy.model_management.extra_reserved_memory = lambda *args, **kwargs: 0
comfy.utils.tiled_scale = lambda *args, **kwargs: (torch.randn(1, 3, 512, 512), 512, 512)

class MockProgressBar:
    def __init__(self, total, node_id=None):
        self.total = total
        self.current = 0
    def update_absolute(self, value, total=None, preview=None):
        self.current = value
    def update(self, value):
        pass

comfy.utils.ProgressBar = MockProgressBar

# Need to mock folder_paths for constants import
folder_paths = types.ModuleType('folder_paths')
sys.modules['folder_paths'] = folder_paths


class MockModel:
    def __init__(self):
        self.model = None


class MockPatcher:
    def __init__(self):
        self.model_patches_models = lambda: []
        self.current_loaded_device = lambda: torch.device('cpu')
        self.model_size = lambda: 1024
        self.loaded_size = lambda: 1024
        self.model_patches_to = lambda x: x
        self.model_dtype = lambda: torch.float32
        self.partially_load = lambda device, extra_memory, force_patch_weights=False: True
        self.is_clone = lambda: False
        self.parent = None
        self.load_device = torch.device('cpu')
        self.model = MockModel()
        self.device = torch.device('cpu')

    def model_memory_required(self, device):
        return 1024

    def model_load(self, lowvram_model_memory, force_patch_weights=False):
        pass

    def model_use_more_vram(self, use_more_vram, force_patch_weights=False):
        pass


class MockFirstStageModel:
    def __init__(self):
        self.dtype = torch.float32

    def to(self, dtype):
        self.dtype = dtype
        return self

    def decode(self, samples):
        if len(samples.shape) == 5:
            B, C, T, H, W = samples.shape
            return torch.randn(B, T, H*8, W*8, 3)
        else:
            B, C, H, W = samples.shape
            return torch.randn(B, 3, H*8, W*8)


class MockVAE:
    def __init__(self):
        self.device = torch.device('cpu')
        self.first_stage_model = MockFirstStageModel()
        self.vae_dtype = torch.float32
        self.patcher = MockPatcher()
        self.output_device = torch.device('cpu')
        self.latent_channels = 4

    def process_output(self, x):
        return x.permute(0, 2, 3, 1)

    def memory_used_decode(self, shape, dtype):
        return 1024

    def spacial_compression_decode(self, shape=None):
        return 8

    def temporal_compression_decode(self, shape=None):
        return 1

    def decode_tiled(self, samples, tile_size, overlap):
        B, C, H, W = samples.shape
        return torch.randn(B, 3, H*8, W*8)

    def decode(self, samples):
        if len(samples.shape) == 5:
            B, C, T, H, W = samples.shape
            return torch.randn(B, T, H*8, W*8, 3)
        else:
            B, C, H, W = samples.shape
            return torch.randn(B, H*8, W*8, 3)


class MockLTXFirstStageModel:
    """Mocks the Lightricks VideoVAE first_stage_model behavior"""
    def __init__(self):
        self.dtype = torch.float16

    def to(self, dtype):
        self.dtype = dtype
        return self

    def decode(self, samples):
        B, C, T, H, W = samples.shape
        return torch.randn(B, 3, 1 + (T - 1) * 8, H * 32, W * 32)

    def state_dict(self):
        return {
            "decoder.up_blocks.0.res_blocks.0.conv1.conv.weight": torch.randn(1024, 128, 3, 3),
        }


class MockLTXVAE:
    """Mock VAE that mimics Lightricks LTX Video VAE (latent_channels=128)"""
    def __init__(self):
        self.device = torch.device('cpu')
        self.first_stage_model = MockLTXFirstStageModel()
        self.vae_dtype = torch.float16
        self.patcher = MockPatcher()
        self.output_device = torch.device('cpu')
        self.latent_channels = 128

    def process_output(self, x):
        return x

    def memory_used_decode(self, shape, dtype):
        return 1024 * 1024

    def spacial_compression_decode(self, shape=None):
        return 32

    def temporal_compression_decode(self, shape=None):
        return 8

    def decode(self, samples):
        B, C, T, H, W = samples.shape
        out_frames = 1 + (T - 1) * 8
        return torch.randn(B, out_frames, H * 32, W * 32, 3)


class MockPixelSpaceFirstStageModel:
    def __init__(self):
        self.dtype = torch.float32

    def to(self, dtype):
        self.dtype = dtype
        return self

    def decode(self, samples):
        return samples


class MockPixelSpaceVAE:
    """Mock VAE that mimics z-image pixel-space passthrough (latent_channels=3)"""
    def __init__(self):
        self.device = torch.device('cpu')
        self.first_stage_model = MockPixelSpaceFirstStageModel()
        self.vae_dtype = torch.float32
        self.patcher = MockPatcher()
        self.output_device = torch.device('cpu')
        self.latent_channels = 3

    def process_output(self, x):
        return x

    def memory_used_decode(self, shape, dtype):
        return 1024

    def spacial_compression_decode(self, shape=None):
        return 1

    def temporal_compression_decode(self, shape=None):
        return None

    def decode(self, samples):
        return samples


class MockWanFirstStageModel(MockWanVAE):
    def __init__(self):
        super().__init__()
        self.dtype = torch.float32

    def to(self, dtype):
        self.dtype = dtype
        return self

    def decode(self, samples):
        B, C, T, H, W = samples.shape
        return torch.randn(B, 3, T, H * 8, W * 8)


class MockWanVAE:
    """Mock VAE that mimics WAN VAE (latent_channels=16)"""
    def __init__(self):
        self.device = torch.device('cpu')
        self.first_stage_model = MockWanFirstStageModel()
        self.vae_dtype = torch.float32
        self.patcher = MockPatcher()
        self.output_device = torch.device('cpu')
        self.latent_channels = 16

    def process_output(self, x):
        return x

    def memory_used_decode(self, shape, dtype):
        return 1024 * 1024

    def spacial_compression_decode(self, shape=None):
        return 8

    def temporal_compression_decode(self, shape=None):
        return 4

    def decode(self, samples):
        B, C, T, H, W = samples.shape
        return torch.randn(B, T, H * 8, W * 8, 3)


# Fixtures
@pytest.fixture
def sample_vae():
    """Provide a mock VAE for testing"""
    return MockVAE()

@pytest.fixture
def sample_latent():
    """Provide a sample latent tensor"""
    return {
        "samples": torch.randn(1, 4, 32, 32)  # 256x256 image
    }

@pytest.fixture
def sample_video_latent():
    """Provide a sample video latent tensor"""
    return {
        "samples": torch.randn(1, 4, 8, 32, 32)  # 8 frames, 256x256
    }

@pytest.fixture
def sample_ltx_vae():
    """Provide a mock LTX Video VAE"""
    return MockLTXVAE()

@pytest.fixture
def sample_ltx_latent():
    """Provide a sample LTX video latent (128 channels, 5D)"""
    return {
        "samples": torch.randn(1, 128, 8, 4, 6)  # [B, 128, T, H, W]
    }

@pytest.fixture
def sample_wan_vae():
    """Provide a mock WAN VAE"""
    return MockWanVAE()

@pytest.fixture
def sample_wan_latent():
    """Provide a sample WAN video latent (16 channels, 5D)"""
    return {
        "samples": torch.randn(1, 16, 8, 32, 32)  # [B, 16, T, H, W]
    }

@pytest.fixture
def sample_pixel_vae():
    """Provide a mock pixel-space VAE (z-image style)"""
    return MockPixelSpaceVAE()

@pytest.fixture
def sample_pixel_latent():
    """Provide a sample pixel-space latent (3 channels, already pixels)"""
    return {
        "samples": torch.randn(1, 3, 64, 64)  # Already RGB pixels
    }

@pytest.fixture
def sample_model():
    """Provide a mock model for testing"""
    return MockModel()

@pytest.fixture
def sample_conditioning():
    """Provide sample conditioning for testing"""
    return {
        "conditioning": torch.randn(1, 77, 768)
    }

@pytest.fixture
def is_amd_gpu():
    """Check if running on AMD GPU"""
    return torch.cuda.is_available()

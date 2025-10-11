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

# Add modules to sys.modules BEFORE any imports
sys.modules['comfy'] = comfy
sys.modules['comfy.model_management'] = comfy.model_management
sys.modules['comfy.utils'] = comfy.utils
sys.modules['comfy.sd'] = comfy.sd
sys.modules['comfy.samplers'] = comfy.samplers
sys.modules['comfy.sample'] = comfy.sample

# Mock functions
comfy.model_management.load_models_gpu = lambda *args, **kwargs: None
comfy.model_management.get_free_memory = lambda *args, **kwargs: 16 * 1024**3
comfy.model_management.minimum_inference_memory = lambda *args, **kwargs: 8 * 1024**3
comfy.model_management.extra_reserved_memory = lambda *args, **kwargs: 0
comfy.utils.tiled_scale = lambda *args, **kwargs: (torch.randn(1, 3, 512, 512), 512, 512)

# Mock classes
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
        if len(samples.shape) == 5:  # Video
            B, C, T, H, W = samples.shape
            return torch.randn(B, T, H*8, W*8, 3)
        else:  # Image
            B, C, H, W = samples.shape
            return torch.randn(B, 3, H*8, W*8)

class MockVAE:
    def __init__(self):
        self.device = torch.device('cpu')
        self.first_stage_model = MockFirstStageModel()
        self.vae_dtype = torch.float32
        self.patcher = MockPatcher()
        self.output_device = torch.device('cpu')
    
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
        if len(samples.shape) == 5:  # Video tensor (B, C, T, H, W)
            B, C, T, H, W = samples.shape
            return torch.randn(B, 3, T, H*8, W*8)
        else:  # Image tensor (B, C, H, W)
            B, C, H, W = samples.shape
            return torch.randn(B, 3, H*8, W*8)

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

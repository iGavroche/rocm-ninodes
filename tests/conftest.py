import pytest
import torch
import sys
import os

# Add ComfyUI to path
sys.path.insert(0, '/home/nino/ComfyUI')

# Mock ComfyUI modules
import types
comfy = types.ModuleType('comfy')
comfy.model_management = types.ModuleType('comfy.model_management')
comfy.utils = types.ModuleType('comfy.utils')

# Mock model management functions
def mock_load_models_gpu(models, memory_required=0, **kwargs):
    pass

def mock_get_free_memory(device):
    return 8 * 1024 * 1024 * 1024  # 8GB

def mock_minimum_inference_memory():
    return 1024 * 1024 * 1024  # 1GB

def mock_extra_reserved_memory():
    return 512 * 1024 * 1024  # 512MB

# Set up mock functions
comfy.model_management.load_models_gpu = mock_load_models_gpu
comfy.model_management.get_free_memory = mock_get_free_memory
comfy.model_management.minimum_inference_memory = mock_minimum_inference_memory
comfy.model_management.extra_reserved_memory = mock_extra_reserved_memory

# Mock tiled_scale function
def mock_tiled_scale(samples, function, tile_x, tile_y, overlap, upscale_amount, out_channels, output_device, pbar=None):
    # Simple mock that just calls the function
    return function(samples)

comfy.utils.tiled_scale = mock_tiled_scale

# Add to sys.modules
sys.modules['comfy'] = comfy
sys.modules['comfy.model_management'] = comfy.model_management
sys.modules['comfy.utils'] = comfy.utils

class MockModel:
    """Mock model for testing"""
    def __init__(self):
        self.dtype = torch.float32
    
    def to(self, dtype):
        self.dtype = dtype
        return self
    
    def eval(self):
        return self
    
    def __call__(self, *args, **kwargs):
        return None

class MockPatcher:
    """Mock patcher for model management"""
    def __init__(self):
        self.model = MockModel()
        self.patch = None
        self.parent = None
        self.load_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
    def model_patches_models(self):
        return []
        
    def current_loaded_device(self):
        return self.load_device
        
    def model_size(self):
        return 100 * 1024 * 1024  # 100MB
        
    def loaded_size(self):
        return 50 * 1024 * 1024  # 50MB
        
    def model_patches_to(self, device):
        pass
        
    def model_dtype(self):
        return torch.float32
        
    def partially_load(self, device, extra_memory, force_patch_weights=False):
        return True
        
    def is_clone(self, other):
        return False

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
            # Return in (B, C, H, W) format - the process_output will convert to (B, H, W, C)
            # The input latent has shape (B, 4, H, W) where H and W are h//8 and w//8
            # The test expects the output to have the same height and width as the original resolution
            # So we need to return (B, 3, H*8, W*8) which becomes (B, H*8, W*8, 3) after permute
            return torch.randn(B, 3, H*8, W*8)

@pytest.fixture
def sample_vae():
    """Create a mock VAE for testing"""
    class MockVAE:
        def __init__(self):
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.output_device = self.device
            self.vae_dtype = torch.float32
            self.latent_channels = 4
            self.upscale_ratio = 8
            
            # Mock first stage model
            self.first_stage_model = MockFirstStageModel()
            
            # Mock patcher for model management
            self.patcher = MockPatcher()
        
        def decode(self, samples):
            # Handle both dict and tensor inputs
            if isinstance(samples, dict):
                samples = samples["samples"]
            
            if len(samples.shape) == 5:  # Video
                B, C, T, H, W = samples.shape
                return (torch.randn(B, T, H*8, W*8, 3),)
            else:  # Image
                B, C, H, W = samples.shape
                # Return correct dimensions: (B, H*8, W*8, 3) for (B, H, W, C) format
                # The test expects shape[1] == h and shape[2] == w, so we need (B, H*8, W*8, 3)
                return (torch.randn(B, H*8, W*8, 3),)
        
        def memory_used_decode(self, shape, dtype):
            return 100 * 1024 * 1024  # 100MB
        
        def spacial_compression_decode(self):
            return 8
        
        def temporal_compression_decode(self):
            return None
            
        def decode_tiled(self, samples, tile_x, tile_y, overlap, tile_t=None, overlap_t=None):
            if len(samples.shape) == 5:  # Video
                B, C, T, H, W = samples.shape
                return torch.randn(B, T, H*8, W*8, 3)
            else:  # Image
                B, C, H, W = samples.shape
                return torch.randn(B, H*8, W*8, 3)
        
        def process_output(self, output):
            # Convert from (B, C, H, W) to (B, H, W, C) format
            if len(output.shape) == 4:
                return output.permute(0, 2, 3, 1).contiguous()
            return output
    
    return MockVAE()

@pytest.fixture
def sample_clip():
    """Create a mock CLIP for testing"""
    class MockCLIP:
        def __init__(self):
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    return MockCLIP()

@pytest.fixture
def flux_workflow_data():
    """Load Flux workflow test data"""
    import json
    workflow_path = "/home/nino/ComfyUI/custom_nodes/rocm_ninodes/flux_dev_optimized.json"
    if os.path.exists(workflow_path):
        with open(workflow_path, 'r') as f:
            return json.load(f)
    return {}

@pytest.fixture
def wan_workflow_data():
    """Load WAN workflow test data"""
    import json
    workflow_path = "/home/nino/ComfyUI/custom_nodes/rocm_ninodes/example_workflow_wan_video.json"
    if os.path.exists(workflow_path):
        with open(workflow_path, 'r') as f:
            return json.load(f)
    return {}

@pytest.fixture
def sample_latent():
    """Create a sample latent tensor for testing"""
    return {
        "batch_index": [0],
        "samples": torch.randn(1, 4, 64, 64)
    }

@pytest.fixture
def sample_video_latent():
    """Create a sample video latent tensor for testing"""
    return {
        "batch_index": [0],
        "samples": torch.randn(1, 4, 8, 64, 64)  # B, C, T, H, W
    }

@pytest.fixture
def is_amd_gpu():
    """Check if running on AMD GPU"""
    return torch.cuda.is_available() and hasattr(torch.cuda, 'get_device_properties')

#!/usr/bin/env python3
"""
Test the instrumentation system
"""
import sys
import os
sys.path.insert(0, '/home/nino/ComfyUI')

from custom_nodes.rocm_ninodes.instrumentation import instrumentation
from custom_nodes.rocm_ninodes.nodes import ROCMOptimizedVAEDecode
import torch

def test_instrumentation():
    """Test if instrumentation is working"""
    print("Testing instrumentation...")
    
    # Create a mock VAE
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
            return 1024  # Mock memory usage
        
        def spacial_compression_decode(self, shape=None):
            return 8  # Mock compression ratio
        
        def temporal_compression_decode(self, shape=None):
            return 1  # Mock temporal compression
        
        def decode_tiled(self, samples, tile_size, overlap):
            B, C, H, W = samples.shape
            return torch.randn(B, 3, H*8, W*8)
        
        def decode(self, samples):
            B, C, H, W = samples.shape
            return torch.randn(B, 3, H*8, W*8)
    
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
    
    class MockModel:
        def __init__(self):
            self.model = None
    
    class MockFirstStageModel:
        def decode(self, samples):
            B, C, H, W = samples.shape
            return torch.randn(B, 3, H*8, W*8)
    
    # Create test data
    vae = MockVAE()
    samples = {
        "samples": torch.randn(1, 4, 32, 32)
    }
    
    # Test the instrumented node
    node = ROCMOptimizedVAEDecode()
    
    try:
        result = node.decode(
            vae=vae,
            samples=samples,
            tile_size=512,
            overlap=64,
            use_rocm_optimizations=True
        )
        print(f"✓ Node execution successful")
        print(f"✓ Result type: {type(result)}")
        print(f"✓ Result length: {len(result) if isinstance(result, (list, tuple)) else 'N/A'}")
        
        # Check if data was captured
        inputs = instrumentation.load_test_data("ROCMOptimizedVAEDecode", "inputs")
        outputs = instrumentation.load_test_data("ROCMOptimizedVAEDecode", "outputs")
        benchmarks = instrumentation.load_benchmark_data("ROCMOptimizedVAEDecode")
        
        print(f"✓ Inputs captured: {len(inputs)}")
        print(f"✓ Outputs captured: {len(outputs)}")
        print(f"✓ Benchmarks captured: {len(benchmarks)}")
        
        if benchmarks:
            print(f"✓ Sample execution time: {benchmarks[0]['execution_time']:.6f}s")
        
    except Exception as e:
        print(f"✗ Node execution failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_instrumentation()

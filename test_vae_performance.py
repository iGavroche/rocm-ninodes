#!/usr/bin/env python3
"""
Simple VAE Performance Test
Test the actual performance of the current VAE decode implementation
"""
import sys
import os
sys.path.insert(0, '/home/nino/ComfyUI')

import torch
import time
import psutil
from custom_nodes.rocm_ninodes.nodes import ROCMOptimizedVAEDecodeInstrumented

def create_mock_vae():
    """Create a mock VAE for testing"""
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
    
    return MockVAE()

def test_vae_performance():
    """Test VAE decode performance"""
    print("🔍 VAE Performance Test")
    print("=" * 40)
    
    node = ROCMOptimizedVAEDecodeInstrumented()
    vae = create_mock_vae()
    
    # Test different sizes
    test_cases = [
        {"name": "256x256", "latent": torch.randn(1, 4, 32, 32)},
        {"name": "512x512", "latent": torch.randn(1, 4, 64, 64)},
        {"name": "1024x1024", "latent": torch.randn(1, 4, 128, 128)},
    ]
    
    for test_case in test_cases:
        samples = {"samples": test_case["latent"]}
        
        print(f"\n📊 Testing {test_case['name']}...")
        
        # Warm up
        for _ in range(2):
            node.decode(
                vae=vae,
                samples=samples,
                tile_size=512,
                overlap=64,
                use_rocm_optimizations=True
            )
        
        # Actual test
        execution_times = []
        for i in range(5):
            start_time = time.time()
            
            result = node.decode(
                vae=vae,
                samples=samples,
                tile_size=512,
                overlap=64,
                use_rocm_optimizations=True
            )
            
            execution_time = time.time() - start_time
            execution_times.append(execution_time)
            
            print(f"  Run {i+1}: {execution_time:.6f}s")
        
        avg_time = sum(execution_times) / len(execution_times)
        print(f"  Average: {avg_time:.6f}s")
        
        # Check if this is reasonable
        if test_case["name"] == "256x256" and avg_time > 0.1:
            print(f"  ⚠️  WARNING: {test_case['name']} is slower than expected!")
        elif test_case["name"] == "512x512" and avg_time > 0.5:
            print(f"  ⚠️  WARNING: {test_case['name']} is slower than expected!")
        elif test_case["name"] == "1024x1024" and avg_time > 2.0:
            print(f"  ⚠️  WARNING: {test_case['name']} is slower than expected!")

if __name__ == "__main__":
    test_vae_performance()

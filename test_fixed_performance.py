#!/usr/bin/env python3
"""
Quick performance test to verify the fix
"""
import sys
import os
sys.path.insert(0, '/home/nino/ComfyUI')

import torch
import time
import statistics
from custom_nodes.rocm_ninodes.nodes import ROCMOptimizedVAEDecode
from custom_nodes.rocm_ninodes.nodes_v2_fixed import ROCMOptimizedVAEDecodeV2FixedInstrumented

def create_simple_mock_vae():
    """Create a minimal mock VAE for testing"""
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

def test_performance():
    """Test performance of both nodes"""
    print("=== Performance Comparison Test (FIXED) ===")
    
    # Create test data
    vae = create_simple_mock_vae()
    samples = {
        "samples": torch.randn(1, 4, 32, 32)  # 256x256 image
    }
    
    # Test original node
    print("\nTesting Original ROCMOptimizedVAEDecode...")
    original_node = ROCMOptimizedVAEDecode()
    original_times = []
    
    for i in range(5):
        start_time = time.time()
        try:
            result = original_node.decode(
                vae=vae,
                samples=samples,
                tile_size=512,
                overlap=64,
                use_rocm_optimizations=True,
                precision_mode='auto',
                batch_optimization=True,
                video_chunk_size=8,
                memory_optimization_enabled=True
            )
            execution_time = time.time() - start_time
            original_times.append(execution_time)
            print(f"  Iteration {i+1}: {execution_time:.6f}s")
        except Exception as e:
            print(f"  Iteration {i+1}: FAILED - {e}")
    
    # Test FIXED V2 node
    print("\nTesting ROCMOptimizedVAEDecodeV2Fixed...")
    v2_fixed_node = ROCMOptimizedVAEDecodeV2FixedInstrumented()
    v2_fixed_times = []
    
    for i in range(5):
        start_time = time.time()
        try:
            result = v2_fixed_node.decode(
                vae=vae,
                samples=samples,
                tile_size=512,
                overlap=64,
                use_rocm_optimizations=True,
                precision_mode='auto',
                batch_optimization=True,
                video_chunk_size=8,
                memory_optimization_enabled=True,
                adaptive_tiling=True
            )
            execution_time = time.time() - start_time
            v2_fixed_times.append(execution_time)
            print(f"  Iteration {i+1}: {execution_time:.6f}s")
        except Exception as e:
            print(f"  Iteration {i+1}: FAILED - {e}")
    
    # Calculate statistics
    if original_times and v2_fixed_times:
        original_avg = statistics.mean(original_times)
        v2_fixed_avg = statistics.mean(v2_fixed_times)
        improvement = ((original_avg - v2_fixed_avg) / original_avg) * 100
        
        print(f"\n=== Results ===")
        print(f"Original Average: {original_avg:.6f}s")
        print(f"V2 Fixed Average: {v2_fixed_avg:.6f}s")
        print(f"Improvement: {improvement:.1f}%")
        
        if improvement < 0:
            print(f"⚠️  Still slower: {abs(improvement):.1f}% slower")
        else:
            print(f"✅ Performance improvement: {improvement:.1f}%")
    else:
        print("❌ No successful iterations to compare")

if __name__ == "__main__":
    test_performance()

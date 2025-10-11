#!/usr/bin/env python3
"""
Benchmark script to compare ROCMOptimizedVAEDecode vs ROCMOptimizedVAEDecodeV2
Phase 1 Optimization Performance Testing
"""
import sys
import os
sys.path.insert(0, '/home/nino/ComfyUI')

import torch
import time
import json
import statistics
from pathlib import Path
from custom_nodes.rocm_ninodes.nodes import ROCMOptimizedVAEDecode
from custom_nodes.rocm_ninodes.nodes_v2_phase1 import ROCMOptimizedVAEDecodeV2Instrumented

def create_mock_vae():
    """Create a comprehensive mock VAE for testing"""
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
    
    return MockVAE()

def benchmark_node(node_class, node_name, test_cases, iterations=10):
    """Benchmark a node with multiple test cases"""
    print(f"\n=== Benchmarking {node_name} ===")
    
    node = node_class()
    vae = create_mock_vae()
    
    results = {
        'node_name': node_name,
        'test_cases': [],
        'overall_stats': {}
    }
    
    for i, test_case in enumerate(test_cases):
        print(f"  Test case {i+1}: {test_case['name']}")
        
        samples = {
            "samples": torch.randn(test_case['batch_size'], 4, test_case['height']//8, test_case['width']//8)
        }
        
        execution_times = []
        
        for iteration in range(iterations):
            start_time = time.time()
            
            try:
                # Use different parameters based on node version
                if "V2" in node_name:
                    result = node.decode(
                        vae=vae,
                        samples=samples,
                        tile_size=test_case.get('tile_size', 768),
                        overlap=test_case.get('overlap', 96),
                        use_rocm_optimizations=True,
                        precision_mode=test_case.get('precision_mode', 'auto'),
                        batch_optimization=True,
                        video_chunk_size=8,
                        memory_optimization_enabled=test_case.get('memory_optimization', True),
                        adaptive_tiling=test_case.get('adaptive_tiling', True)
                    )
                else:
                    # Original node parameters
                    result = node.decode(
                        vae=vae,
                        samples=samples,
                        tile_size=test_case.get('tile_size', 768),
                        overlap=test_case.get('overlap', 96),
                        use_rocm_optimizations=True,
                        precision_mode=test_case.get('precision_mode', 'auto'),
                        batch_optimization=True,
                        video_chunk_size=8,
                        memory_optimization_enabled=test_case.get('memory_optimization', True)
                    )
                
                execution_time = time.time() - start_time
                execution_times.append(execution_time)
                
            except Exception as e:
                print(f"    ✗ Iteration {iteration+1} failed: {e}")
                continue
        
        if execution_times:
            test_result = {
                'name': test_case['name'],
                'parameters': test_case,
                'iterations': len(execution_times),
                'execution_times': execution_times,
                'average_time': statistics.mean(execution_times),
                'median_time': statistics.median(execution_times),
                'min_time': min(execution_times),
                'max_time': max(execution_times),
                'std_dev': statistics.stdev(execution_times) if len(execution_times) > 1 else 0
            }
            
            results['test_cases'].append(test_result)
            
            print(f"    ✓ Average: {test_result['average_time']:.6f}s")
            print(f"    ✓ Median: {test_result['median_time']:.6f}s")
            print(f"    ✓ Min: {test_result['min_time']:.6f}s")
            print(f"    ✓ Max: {test_result['max_time']:.6f}s")
    
    # Calculate overall statistics
    all_times = []
    for test_case in results['test_cases']:
        all_times.extend(test_case['execution_times'])
    
    if all_times:
        results['overall_stats'] = {
            'total_executions': len(all_times),
            'average_time': statistics.mean(all_times),
            'median_time': statistics.median(all_times),
            'min_time': min(all_times),
            'max_time': max(all_times),
            'std_dev': statistics.stdev(all_times) if len(all_times) > 1 else 0
        }
    
    return results

def run_phase1_benchmark():
    """Run Phase 1 optimization benchmark"""
    print("=== ROCMOptimizedVAEDecode Phase 1 Optimization Benchmark ===")
    
    # Define test cases
    test_cases = [
        {
            'name': 'Small Image (256x256)',
            'batch_size': 1,
            'height': 256,
            'width': 256,
            'tile_size': 512,
            'overlap': 64
        },
        {
            'name': 'Medium Image (512x512)',
            'batch_size': 1,
            'height': 512,
            'width': 512,
            'tile_size': 768,
            'overlap': 96
        },
        {
            'name': 'Large Image (1024x1024)',
            'batch_size': 1,
            'height': 1024,
            'width': 1024,
            'tile_size': 1024,
            'overlap': 128
        },
        {
            'name': 'Batch Processing (4x512x512)',
            'batch_size': 4,
            'height': 512,
            'width': 512,
            'tile_size': 768,
            'overlap': 96
        },
        {
            'name': 'High Resolution (1280x1280)',
            'batch_size': 1,
            'height': 1280,
            'width': 1280,
            'tile_size': 1280,
            'overlap': 160
        }
    ]
    
    # Benchmark original node
    original_results = benchmark_node(
        ROCMOptimizedVAEDecode, 
        "ROCMOptimizedVAEDecode (Original)", 
        test_cases, 
        iterations=5
    )
    
    # Benchmark Phase 1 optimized node
    v2_results = benchmark_node(
        ROCMOptimizedVAEDecodeV2Instrumented, 
        "ROCMOptimizedVAEDecodeV2 (Phase 1)", 
        test_cases, 
        iterations=5
    )
    
    # Calculate performance improvements
    improvements = []
    for i, (original, v2) in enumerate(zip(original_results['test_cases'], v2_results['test_cases'])):
        improvement = ((original['average_time'] - v2['average_time']) / original['average_time']) * 100
        improvements.append({
            'test_case': original['name'],
            'original_time': original['average_time'],
            'v2_time': v2['average_time'],
            'improvement_percent': improvement
        })
    
    # Overall improvement
    overall_improvement = 0
    if (original_results['overall_stats'] and v2_results['overall_stats'] and 
        'average_time' in original_results['overall_stats'] and 
        'average_time' in v2_results['overall_stats']):
        overall_improvement = ((original_results['overall_stats']['average_time'] - 
                               v2_results['overall_stats']['average_time']) / 
                              original_results['overall_stats']['average_time']) * 100
    
    # Save results
    benchmark_results = {
        'timestamp': time.time(),
        'phase': 'Phase 1 - Memory and Tile Optimization',
        'original_results': original_results,
        'v2_results': v2_results,
        'improvements': improvements,
        'overall_improvement_percent': overall_improvement,
        'summary': {
            'original_average': original_results['overall_stats']['average_time'],
            'v2_average': v2_results['overall_stats']['average_time'],
            'improvement_percent': overall_improvement,
            'target_achieved': overall_improvement >= 25  # Phase 1 target: 25-35%
        }
    }
    
    # Save to file
    os.makedirs("test_data/optimization", exist_ok=True)
    with open("test_data/optimization/phase1_benchmark_results.json", "w") as f:
        json.dump(benchmark_results, f, indent=2)
    
    # Print summary
    print(f"\n=== Phase 1 Benchmark Summary ===")
    print(f"Original Average Time: {original_results['overall_stats']['average_time']:.6f}s")
    print(f"V2 Average Time: {v2_results['overall_stats']['average_time']:.6f}s")
    print(f"Overall Improvement: {overall_improvement:.1f}%")
    print(f"Target (25-35%): {'✓ ACHIEVED' if overall_improvement >= 25 else '✗ NOT ACHIEVED'}")
    
    print(f"\nPer-Test Improvements:")
    for improvement in improvements:
        print(f"  {improvement['test_case']}: {improvement['improvement_percent']:.1f}%")
    
    print(f"\nBenchmark results saved to test_data/optimization/phase1_benchmark_results.json")
    
    return benchmark_results

if __name__ == "__main__":
    run_phase1_benchmark()

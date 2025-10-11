"""
Performance benchmark tests for ROCM nodes
"""
import pytest
import torch
import time
import statistics
import sys
import os

# Add the custom nodes directory to path
sys.path.insert(0, '/home/nino/ComfyUI/custom_nodes/rocm_ninodes')

from nodes import ROCMOptimizedVAEDecodeInstrumented

class TestPerformanceBenchmarks:
    """Performance benchmark tests"""
    
    def test_vae_decode_performance(self, sample_vae):
        """Benchmark VAE decode performance"""
        node = ROCMOptimizedVAEDecodeInstrumented()
        
        # Test different resolutions
        test_cases = [
            {"name": "256x256", "latent": torch.randn(1, 4, 32, 32)},
            {"name": "512x512", "latent": torch.randn(1, 4, 64, 64)},
            {"name": "1024x1024", "latent": torch.randn(1, 4, 128, 128)},
        ]
        
        results = {}
        
        for test_case in test_cases:
            samples = {"samples": test_case["latent"]}
            execution_times = []
            
            # Run multiple iterations
            for _ in range(5):
                start_time = time.time()
                
                result = node.decode(
                    vae=sample_vae,
                    samples=samples,
                    tile_size=512,
                    overlap=64,
                    use_rocm_optimizations=True
                )
                
                execution_time = time.time() - start_time
                execution_times.append(execution_time)
            
            avg_time = statistics.mean(execution_times)
            results[test_case["name"]] = avg_time
            
            print(f"{test_case['name']}: {avg_time:.6f}s average")
        
        # Basic performance assertions
        assert results["256x256"] < 1.0  # Should be fast for small images
        assert results["512x512"] < 2.0  # Should be reasonable for medium images
        assert results["1024x1024"] < 5.0  # Should be acceptable for large images
    
    def test_tile_size_performance(self, sample_vae):
        """Benchmark different tile sizes"""
        node = ROCMOptimizedVAEDecodeInstrumented()
        
        samples = {"samples": torch.randn(1, 4, 64, 64)}  # 512x512
        
        tile_sizes = [256, 512, 768, 1024]
        results = {}
        
        for tile_size in tile_sizes:
            execution_times = []
            
            for _ in range(3):
                start_time = time.time()
                
                result = node.decode(
                    vae=sample_vae,
                    samples=samples,
                    tile_size=tile_size,
                    overlap=64,
                    use_rocm_optimizations=True
                )
                
                execution_time = time.time() - start_time
                execution_times.append(execution_time)
            
            avg_time = statistics.mean(execution_times)
            results[tile_size] = avg_time
            
            print(f"Tile size {tile_size}: {avg_time:.6f}s average")
        
        # All tile sizes should be reasonably fast
        for tile_size, avg_time in results.items():
            assert avg_time < 2.0, f"Tile size {tile_size} too slow: {avg_time:.6f}s"
    
    def test_precision_performance(self, sample_vae):
        """Benchmark different precision modes"""
        node = ROCMOptimizedVAEDecodeInstrumented()
        
        samples = {"samples": torch.randn(1, 4, 64, 64)}  # 512x512
        
        precision_modes = ["auto", "fp32", "fp16"]
        results = {}
        
        for precision_mode in precision_modes:
            execution_times = []
            
            for _ in range(3):
                start_time = time.time()
                
                result = node.decode(
                    vae=sample_vae,
                    samples=samples,
                    tile_size=512,
                    overlap=64,
                    use_rocm_optimizations=True,
                    precision_mode=precision_mode
                )
                
                execution_time = time.time() - start_time
                execution_times.append(execution_time)
            
            avg_time = statistics.mean(execution_times)
            results[precision_mode] = avg_time
            
            print(f"Precision {precision_mode}: {avg_time:.6f}s average")
        
        # All precision modes should be reasonably fast
        for precision_mode, avg_time in results.items():
            assert avg_time < 2.0, f"Precision {precision_mode} too slow: {avg_time:.6f}s"
    
    def test_memory_usage(self, sample_vae):
        """Test memory usage patterns"""
        node = ROCMOptimizedVAEDecodeInstrumented()
        
        # Test with different batch sizes
        batch_sizes = [1, 2, 4]
        
        for batch_size in batch_sizes:
            samples = {"samples": torch.randn(batch_size, 4, 32, 32)}
            
            # Clear memory before test
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            start_time = time.time()
            
            result = node.decode(
                vae=sample_vae,
                samples=samples,
                tile_size=512,
                overlap=64,
                use_rocm_optimizations=True,
                memory_optimization_enabled=True
            )
            
            execution_time = time.time() - start_time
            
            print(f"Batch size {batch_size}: {execution_time:.6f}s")
            
            # Should complete successfully
            assert isinstance(result, tuple)
            assert len(result) == 1
            assert isinstance(result[0], torch.Tensor)
            assert result[0].shape[0] == batch_size
    
    def test_consistency_performance(self, sample_vae):
        """Test performance consistency across multiple runs"""
        node = ROCMOptimizedVAEDecodeInstrumented()
        
        samples = {"samples": torch.randn(1, 4, 32, 32)}
        execution_times = []
        
        # Run multiple times
        for _ in range(10):
            start_time = time.time()
            
            result = node.decode(
                vae=sample_vae,
                samples=samples,
                tile_size=512,
                overlap=64,
                use_rocm_optimizations=True
            )
            
            execution_time = time.time() - start_time
            execution_times.append(execution_time)
        
        avg_time = statistics.mean(execution_times)
        std_dev = statistics.stdev(execution_times) if len(execution_times) > 1 else 0
        
        print(f"Average time: {avg_time:.6f}s")
        print(f"Standard deviation: {std_dev:.6f}s")
        
        # Performance should be consistent
        assert avg_time < 1.0, f"Average time too slow: {avg_time:.6f}s"
        assert std_dev < 0.1, f"Performance too inconsistent: {std_dev:.6f}s std dev"

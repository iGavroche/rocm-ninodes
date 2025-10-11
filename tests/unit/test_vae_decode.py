"""
Unit tests for ROCMOptimizedVAEDecode
"""
import pytest
import torch
import sys
import os

# Add the custom nodes directory to path
sys.path.insert(0, '/home/nino/ComfyUI/custom_nodes/rocm_ninodes')

from nodes import ROCMOptimizedVAEDecodeInstrumented

class TestROCMOptimizedVAEDecode:
    """Test suite for ROCMOptimizedVAEDecode"""
    
    def test_basic_decode(self, sample_vae, sample_latent):
        """Test basic VAE decode functionality"""
        node = ROCMOptimizedVAEDecodeInstrumented()
        
        result = node.decode(
            vae=sample_vae,
            samples=sample_latent,
            tile_size=512,
            overlap=64,
            use_rocm_optimizations=True
        )
        
        assert isinstance(result, tuple)
        assert len(result) == 1
        assert isinstance(result[0], torch.Tensor)
        assert len(result[0].shape) == 4  # Should be (B, H, W, C)
    
    def test_different_tile_sizes(self, sample_vae, sample_latent):
        """Test VAE decode with different tile sizes"""
        node = ROCMOptimizedVAEDecodeInstrumented()
        
        tile_sizes = [256, 512, 768, 1024]
        for tile_size in tile_sizes:
            result = node.decode(
                vae=sample_vae,
                samples=sample_latent,
                tile_size=tile_size,
                overlap=64,
                use_rocm_optimizations=True
            )
            
            assert isinstance(result, tuple)
            assert len(result) == 1
            assert isinstance(result[0], torch.Tensor)
    
    def test_different_resolutions(self, sample_vae):
        """Test VAE decode with different resolutions"""
        node = ROCMOptimizedVAEDecodeInstrumented()
        
        resolutions = [(256, 256), (512, 512), (1024, 1024)]
        for w, h in resolutions:
            latent = {
                "samples": torch.randn(1, 4, h//8, w//8)
            }
            
            result = node.decode(
                vae=sample_vae,
                samples=latent,
                tile_size=512,
                overlap=64,
                use_rocm_optimizations=True
            )
            
            assert isinstance(result, tuple)
            assert len(result) == 1
            assert isinstance(result[0], torch.Tensor)
            assert len(result[0].shape) == 4
    
    def test_precision_modes(self, sample_vae, sample_latent):
        """Test VAE decode with different precision modes"""
        node = ROCMOptimizedVAEDecodeInstrumented()
        
        precision_modes = ["auto", "fp32", "fp16"]
        for precision_mode in precision_modes:
            result = node.decode(
                vae=sample_vae,
                samples=sample_latent,
                tile_size=512,
                overlap=64,
                use_rocm_optimizations=True,
                precision_mode=precision_mode
            )
            
            assert isinstance(result, tuple)
            assert len(result) == 1
            assert isinstance(result[0], torch.Tensor)
    
    def test_batch_processing(self, sample_vae):
        """Test VAE decode with batch processing"""
        node = ROCMOptimizedVAEDecodeInstrumented()
        
        # Test with batch size > 1
        batch_latent = {
            "samples": torch.randn(2, 4, 32, 32)  # Batch size 2
        }
        
        result = node.decode(
            vae=sample_vae,
            samples=batch_latent,
            tile_size=512,
            overlap=64,
            use_rocm_optimizations=True,
            batch_optimization=True
        )
        
        assert isinstance(result, tuple)
        assert len(result) == 1
        assert isinstance(result[0], torch.Tensor)
        assert result[0].shape[0] == 2  # Batch size should be preserved
    
    def test_memory_optimization(self, sample_vae, sample_latent):
        """Test VAE decode with memory optimization enabled/disabled"""
        node = ROCMOptimizedVAEDecodeInstrumented()
        
        for memory_opt in [True, False]:
            result = node.decode(
                vae=sample_vae,
                samples=sample_latent,
                tile_size=512,
                overlap=64,
                use_rocm_optimizations=True,
                memory_optimization_enabled=memory_opt
            )
            
            assert isinstance(result, tuple)
            assert len(result) == 1
            assert isinstance(result[0], torch.Tensor)
    
    def test_adaptive_tiling(self, sample_vae, sample_latent):
        """Test VAE decode with adaptive tiling enabled/disabled"""
        node = ROCMOptimizedVAEDecodeInstrumented()
        
        for adaptive_tiling in [True, False]:
            result = node.decode(
                vae=sample_vae,
                samples=sample_latent,
                tile_size=512,
                overlap=64,
                use_rocm_optimizations=True,
                adaptive_tiling=adaptive_tiling
            )
            
            assert isinstance(result, tuple)
            assert len(result) == 1
            assert isinstance(result[0], torch.Tensor)
    
    def test_error_handling(self, sample_vae):
        """Test VAE decode error handling"""
        node = ROCMOptimizedVAEDecodeInstrumented()
        
        # Test with invalid samples
        invalid_samples = {"samples": "invalid"}
        
        with pytest.raises(Exception):
            node.decode(
                vae=sample_vae,
                samples=invalid_samples,
                tile_size=512,
                overlap=64,
                use_rocm_optimizations=True
            )
    
    def test_performance_consistency(self, sample_vae, sample_latent):
        """Test that VAE decode produces consistent results"""
        node = ROCMOptimizedVAEDecodeInstrumented()
        
        # Run multiple times with same inputs
        results = []
        for _ in range(3):
            result = node.decode(
                vae=sample_vae,
                samples=sample_latent,
                tile_size=512,
                overlap=64,
                use_rocm_optimizations=True
            )
            results.append(result[0])
        
        # All results should have the same shape
        for result in results:
            assert result.shape == results[0].shape
    
    def test_large_image(self, sample_vae):
        """Test VAE decode with large image"""
        node = ROCMOptimizedVAEDecodeInstrumented()
        
        # Test with 1024x1024 image
        large_latent = {
            "samples": torch.randn(1, 4, 128, 128)  # 1024x1024
        }
        
        result = node.decode(
            vae=sample_vae,
            samples=large_latent,
            tile_size=768,
            overlap=96,
            use_rocm_optimizations=True
        )
        
        assert isinstance(result, tuple)
        assert len(result) == 1
        assert isinstance(result[0], torch.Tensor)
        assert len(result[0].shape) == 4

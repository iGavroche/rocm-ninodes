"""
Integration tests for ROCM workflows
"""
import pytest
import torch
import sys
import os
import json

# Add the custom nodes directory to path
sys.path.insert(0, '/home/nino/ComfyUI/custom_nodes/rocm_ninodes')

from nodes import ROCMOptimizedVAEDecodeInstrumented

class TestWorkflowIntegration:
    """Integration tests for ROCM workflows"""
    
    def test_flux_workflow_simulation(self, sample_vae):
        """Test simulation of Flux workflow components"""
        node = ROCMOptimizedVAEDecodeInstrumented()
        
        # Simulate Flux workflow latent
        flux_latent = {
            "samples": torch.randn(1, 4, 64, 64)  # 512x512 image
        }
        
        result = node.decode(
            vae=sample_vae,
            samples=flux_latent,
            tile_size=768,
            overlap=96,
            use_rocm_optimizations=True,
            precision_mode="auto",
            batch_optimization=True
        )
        
        assert isinstance(result, tuple)
        assert len(result) == 1
        assert isinstance(result[0], torch.Tensor)
        assert len(result[0].shape) == 4
    
    @pytest.mark.skip(reason="Video processing needs more complex implementation")
    def test_wan_video_workflow_simulation(self, sample_vae):
        """Test simulation of WAN video workflow components"""
        node = ROCMOptimizedVAEDecodeInstrumented()
        
        # Simulate WAN video workflow latent
        wan_latent = {
            "samples": torch.randn(1, 4, 16, 32, 32)  # 16 frames, 256x256
        }
        
        result = node.decode(
            vae=sample_vae,
            samples=wan_latent,
            tile_size=512,
            overlap=64,
            use_rocm_optimizations=True,
            video_chunk_size=8,
            memory_optimization_enabled=True
        )
        
        assert isinstance(result, tuple)
        assert len(result) == 1
        assert isinstance(result[0], torch.Tensor)
    
    def test_high_resolution_workflow(self, sample_vae):
        """Test high resolution workflow simulation"""
        node = ROCMOptimizedVAEDecodeInstrumented()
        
        # Test 1280x1280 image
        high_res_latent = {
            "samples": torch.randn(1, 4, 160, 160)  # 1280x1280
        }
        
        result = node.decode(
            vae=sample_vae,
            samples=high_res_latent,
            tile_size=1024,
            overlap=128,
            use_rocm_optimizations=True,
            adaptive_tiling=True
        )
        
        assert isinstance(result, tuple)
        assert len(result) == 1
        assert isinstance(result[0], torch.Tensor)
    
    def test_batch_workflow_simulation(self, sample_vae):
        """Test batch processing workflow simulation"""
        node = ROCMOptimizedVAEDecodeInstrumented()
        
        # Test batch processing
        batch_latent = {
            "samples": torch.randn(4, 4, 32, 32)  # Batch size 4
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
        assert result[0].shape[0] == 4  # Batch size preserved
    
    def test_memory_constrained_workflow(self, sample_vae):
        """Test memory-constrained workflow simulation"""
        node = ROCMOptimizedVAEDecodeInstrumented()
        
        # Test with memory optimizations
        memory_latent = {
            "samples": torch.randn(1, 4, 80, 80)  # 640x640
        }
        
        result = node.decode(
            vae=sample_vae,
            samples=memory_latent,
            tile_size=512,
            overlap=64,
            use_rocm_optimizations=True,
            memory_optimization_enabled=True,
            adaptive_tiling=True
        )
        
        assert isinstance(result, tuple)
        assert len(result) == 1
        assert isinstance(result[0], torch.Tensor)
    
    def test_precision_workflow_simulation(self, sample_vae):
        """Test different precision modes in workflow context"""
        node = ROCMOptimizedVAEDecodeInstrumented()
        
        precision_modes = ["auto", "fp32", "fp16"]
        for precision_mode in precision_modes:
            result = node.decode(
                vae=sample_vae,
                samples={"samples": torch.randn(1, 4, 32, 32)},
                tile_size=512,
                overlap=64,
                use_rocm_optimizations=True,
                precision_mode=precision_mode
            )
            
            assert isinstance(result, tuple)
            assert len(result) == 1
            assert isinstance(result[0], torch.Tensor)
    
    def test_workflow_consistency(self, sample_vae):
        """Test workflow consistency across multiple runs"""
        node = ROCMOptimizedVAEDecodeInstrumented()
        
        # Test consistency
        latent = {"samples": torch.randn(1, 4, 32, 32)}
        
        results = []
        for _ in range(3):
            result = node.decode(
                vae=sample_vae,
                samples=latent,
                tile_size=512,
                overlap=64,
                use_rocm_optimizations=True
            )
            results.append(result[0])
        
        # All results should have same shape
        for result in results:
            assert result.shape == results[0].shape

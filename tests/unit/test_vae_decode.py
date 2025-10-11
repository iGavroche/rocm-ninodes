"""
Unit tests for ROCMOptimizedVAEDecode node
"""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from nodes import ROCMOptimizedVAEDecode


class TestROCMOptimizedVAEDecode:
    """Test cases for ROCMOptimizedVAEDecode node"""
    
    def test_input_types(self):
        """Test that INPUT_TYPES returns correct structure"""
        input_types = ROCMOptimizedVAEDecode.INPUT_TYPES()
        
        assert "required" in input_types
        required = input_types["required"]
        
        # Check required inputs
        assert "samples" in required
        assert "vae" in required
        assert "tile_size" in required
        assert "overlap" in required
        assert "use_rocm_optimizations" in required
        assert "precision_mode" in required
        assert "batch_optimization" in required
    
    def test_return_types(self):
        """Test that RETURN_TYPES is correct"""
        assert ROCMOptimizedVAEDecode.RETURN_TYPES == ("IMAGE",)
        assert ROCMOptimizedVAEDecode.RETURN_NAMES == ("IMAGE",)
        assert ROCMOptimizedVAEDecode.FUNCTION == "decode"
        assert ROCMOptimizedVAEDecode.CATEGORY == "RocM Ninodes/VAE"
    
    def test_decode_basic(self, sample_latent, sample_vae):
        """Test basic decode functionality"""
        node = ROCMOptimizedVAEDecode()
        
        result = node.decode(
            vae=sample_vae,
            samples=sample_latent,
            tile_size=768,
            overlap=96,
            use_rocm_optimizations=True,
            precision_mode="auto",
            batch_optimization=True
        )
        
        assert isinstance(result, tuple)
        assert len(result) == 1
        assert isinstance(result[0], torch.Tensor)
        assert len(result[0].shape) == 4  # B, H, W, C
    
    def test_decode_video(self, sample_video_latent, sample_vae):
        """Test video decode functionality"""
        node = ROCMOptimizedVAEDecode()
        
        result = node.decode(
            vae=sample_vae,
            samples=sample_video_latent,
            tile_size=768,
            overlap=96,
            use_rocm_optimizations=True,
            precision_mode="auto",
            batch_optimization=True,
            video_chunk_size=4,
            memory_optimization_enabled=True
        )
        
        assert isinstance(result, tuple)
        assert len(result) == 1
        assert isinstance(result[0], torch.Tensor)
        # Should be reshaped to 4D for ComfyUI
        assert len(result[0].shape) == 4  # B*T, H, W, C
    
    def test_precision_modes(self, sample_latent, sample_vae):
        """Test different precision modes"""
        node = ROCMOptimizedVAEDecode()
        
        precision_modes = ["auto", "fp32", "fp16", "bf16"]
        
        for precision in precision_modes:
            result = node.decode(
                vae=sample_vae,
                samples=sample_latent,
                precision_mode=precision,
                use_rocm_optimizations=True
            )
            
            assert isinstance(result, tuple)
            assert len(result) == 1
            assert isinstance(result[0], torch.Tensor)
    
    def test_rocm_optimizations(self, sample_latent, sample_vae, is_amd_gpu):
        """Test ROCm optimizations"""
        node = ROCMOptimizedVAEDecode()
        
        # Test with ROCm optimizations enabled
        result = node.decode(
            vae=sample_vae,
            samples=sample_latent,
            use_rocm_optimizations=True,
            precision_mode="auto"
        )
        
        assert isinstance(result, tuple)
        assert len(result) == 1
        
        # Test with ROCm optimizations disabled
        result = node.decode(
            vae=sample_vae,
            samples=sample_latent,
            use_rocm_optimizations=False,
            precision_mode="auto"
        )
        
        assert isinstance(result, tuple)
        assert len(result) == 1
    
    def test_tile_size_optimization(self, sample_latent, sample_vae):
        """Test tile size optimization"""
        node = ROCMOptimizedVAEDecode()
        
        tile_sizes = [256, 512, 768, 1024, 1536]
        
        for tile_size in tile_sizes:
            result = node.decode(
                vae=sample_vae,
                samples=sample_latent,
                tile_size=tile_size,
                overlap=min(96, tile_size // 4),
                use_rocm_optimizations=True
            )
            
            assert isinstance(result, tuple)
            assert len(result) == 1
            assert isinstance(result[0], torch.Tensor)
    
    def test_memory_optimization(self, sample_latent, sample_vae):
        """Test memory optimization settings"""
        node = ROCMOptimizedVAEDecode()
        
        # Test with memory optimization enabled
        result = node.decode(
            vae=sample_vae,
            samples=sample_latent,
            batch_optimization=True,
            memory_optimization_enabled=True
        )
        
        assert isinstance(result, tuple)
        
        # Test with memory optimization disabled
        result = node.decode(
            vae=sample_vae,
            samples=sample_latent,
            batch_optimization=False,
            memory_optimization_enabled=False
        )
        
        assert isinstance(result, tuple)
    
    def test_error_handling(self, sample_latent, sample_vae):
        """Test error handling and fallbacks"""
        node = ROCMOptimizedVAEDecode()
        
        # Test with invalid parameters
        with pytest.raises(Exception):
            node.decode(
                vae=None,
                samples=sample_latent,
                use_rocm_optimizations=True
            )
    
    def test_performance_timing(self, sample_latent, sample_vae):
        """Test that performance timing works"""
        node = ROCMOptimizedVAEDecode()
        
        import time
        start_time = time.time()
        
        result = node.decode(
            vae=sample_vae,
            samples=sample_latent,
            use_rocm_optimizations=True
        )
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        assert execution_time > 0
        assert isinstance(result, tuple)


class TestROCMOptimizedVAEDecodeTiled:
    """Test cases for ROCMOptimizedVAEDecodeTiled node"""
    
    def test_input_types(self):
        """Test that INPUT_TYPES returns correct structure"""
        input_types = ROCMOptimizedVAEDecodeTiled.INPUT_TYPES()
        
        assert "required" in input_types
        required = input_types["required"]
        
        # Check required inputs
        assert "samples" in required
        assert "vae" in required
        assert "tile_size" in required
        assert "overlap" in required
        assert "temporal_size" in required
        assert "temporal_overlap" in required
        assert "rocm_optimizations" in required
    
    def test_return_types(self):
        """Test that RETURN_TYPES is correct"""
        assert ROCMOptimizedVAEDecodeTiled.RETURN_TYPES == ("IMAGE",)
        assert ROCMOptimizedVAEDecodeTiled.RETURN_NAMES == ("IMAGE",)
        assert ROCMOptimizedVAEDecodeTiled.FUNCTION == "decode"
        assert ROCMOptimizedVAEDecodeTiled.CATEGORY == "RocM Ninodes/VAE"
    
    def test_decode_tiled(self, sample_latent, sample_vae):
        """Test tiled decode functionality"""
        node = ROCMOptimizedVAEDecodeTiled()
        
        result = node.decode(
            vae=sample_vae,
            samples=sample_latent,
            tile_size=768,
            overlap=96,
            temporal_size=64,
            temporal_overlap=8,
            rocm_optimizations=True
        )
        
        assert isinstance(result, tuple)
        assert len(result) == 1
        assert isinstance(result[0], torch.Tensor)
    
    def test_temporal_parameters(self, sample_video_latent, sample_vae):
        """Test temporal parameters for video processing"""
        node = ROCMOptimizedVAEDecodeTiled()
        
        result = node.decode(
            vae=sample_vae,
            samples=sample_video_latent,
            tile_size=512,
            overlap=64,
            temporal_size=32,
            temporal_overlap=4,
            rocm_optimizations=True
        )
        
        assert isinstance(result, tuple)
        assert len(result) == 1
        assert isinstance(result[0], torch.Tensor)
    
    def test_rocm_optimizations_tiled(self, sample_latent, sample_vae):
        """Test ROCm optimizations for tiled decode"""
        node = ROCMOptimizedVAEDecodeTiled()
        
        # Test with ROCm optimizations
        result = node.decode(
            vae=sample_vae,
            samples=sample_latent,
            rocm_optimizations=True
        )
        
        assert isinstance(result, tuple)
        
        # Test without ROCm optimizations
        result = node.decode(
            vae=sample_vae,
            samples=sample_latent,
            rocm_optimizations=False
        )
        
        assert isinstance(result, tuple)


@pytest.mark.parametrize("resolution", [(256, 256), (512, 512), (1024, 1024)])
def test_different_resolutions(resolution, sample_vae):
    """Test VAE decode with different resolutions"""
    node = ROCMOptimizedVAEDecode()
    
    # Create latent with specified resolution
    w, h = resolution
    latent = {
        "samples": torch.randn(1, 4, h//8, w//8)
    }
    
    result = node.decode(
        vae=sample_vae,
        samples=latent,
        use_rocm_optimizations=True
    )
    
    assert isinstance(result, tuple)
    assert len(result) == 1
    assert isinstance(result[0], torch.Tensor)
    # VAE decode should return a 4D tensor (B, H, W, C)
    assert len(result[0].shape) == 4
    # The output should have the correct batch size
    assert result[0].shape[0] == 1


@pytest.mark.parametrize("batch_size", [1, 2, 4])
def test_different_batch_sizes(batch_size, sample_vae):
    """Test VAE decode with different batch sizes"""
    node = ROCMOptimizedVAEDecode()
    
    # Create latent with specified batch size
    latent = {
        "samples": torch.randn(batch_size, 4, 64, 64)
    }
    
    result = node.decode(
        vae=sample_vae,
        samples=latent,
        use_rocm_optimizations=True
    )
    
    assert isinstance(result, tuple)
    assert len(result) == 1
    assert isinstance(result[0], torch.Tensor)
    assert result[0].shape[0] == batch_size

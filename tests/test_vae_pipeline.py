#!/usr/bin/env python3
"""
Test VAE decode optimizations using simulated encoder data
Tests the complete VAE encode->decode pipeline
"""

import pytest
import torch
import numpy as np
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import PIL.Image

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from nodes import ROCMOptimizedVAEDecode

class TestVAEPipelineOptimization:
    """Test complete VAE encode->decode pipeline"""
    
    @pytest.fixture
    def node(self):
        return ROCMOptimizedVAEDecode()
    
    def create_mock_vae(self, vae_type="SD"):
        """Create mock VAE for testing"""
        mock_vae = Mock()
        mock_vae.device = torch.device('cpu')
        mock_vae.vae_dtype = torch.float32
        mock_vae.output_device = torch.device('cpu')
        
        # Mock first_stage_model
        mock_vae.first_stage_model = Mock()
        mock_vae.first_stage_model.dtype = torch.float32
        
        # Mock patcher
        mock_vae.patcher = Mock()
        
        # Mock memory calculation
        def mock_memory_used_decode(shape, dtype):
            elements = 1
            for dim in shape:
                elements *= dim
            return elements * 4 / 1024**3
        
        mock_vae.memory_used_decode = mock_memory_used_decode
        
        # Mock decode function
        def mock_decode(samples):
            if isinstance(samples, dict):
                samples_tensor = samples["samples"]
            else:
                samples_tensor = samples
            
            B, C, H, W = samples_tensor.shape
            output_h, output_w = H * 8, W * 8
            output = torch.randn(B, 3, output_h, output_w, dtype=torch.float32)
            return (output,)
        
        mock_vae.decode = mock_decode
        mock_vae.first_stage_model.decode = mock_decode
        
        # Mock VAE properties
        mock_vae.spacial_compression_decode = lambda: 8
        mock_vae.upscale_ratio = 8
        mock_vae.latent_channels = 4 if vae_type == "SD" else 16
        
        return mock_vae
    
    def simulate_vae_encode(self, image_tensor):
        """Simulate VAE encoding process"""
        # Simulate encoding: (B, 3, H, W) -> (B, C, H/8, W/8)
        B, C, H, W = image_tensor.shape
        latent_h, latent_w = H // 8, W // 8
        
        # Simulate different architectures
        if H <= 512:  # SD
            latent_channels = 4
        else:  # SDXL
            latent_channels = 16
        
        # Create latent tensor
        latent = torch.randn(B, latent_channels, latent_h, latent_w, dtype=torch.float32)
        return latent
    
    def test_complete_sd_pipeline(self, node):
        """Test complete SD VAE pipeline"""
        mock_vae = self.create_mock_vae("SD")
        
        # Simulate input image (512x512 RGB)
        input_image = torch.randn(1, 3, 512, 512, dtype=torch.float32)
        
        # Simulate VAE encoding
        latent = self.simulate_vae_encode(input_image)
        assert latent.shape == (1, 4, 64, 64)  # SD: 512/8 = 64
        
        # Test VAE decoding
        samples = {"samples": latent}
        
        with patch('nodes.COMFY_AVAILABLE', True):
            with patch('nodes.comfy') as mock_comfy:
                mock_comfy.utils.tiled_scale = Mock(return_value=torch.randn(1, 3, 512, 512))
                
                result = node.decode(mock_vae, samples, use_rocm_optimizations=True)
        
        # Verify output
        assert isinstance(result, tuple)
        assert len(result) == 1
        output_tensor = result[0]
        assert isinstance(output_tensor, torch.Tensor)
        assert output_tensor.shape == (1, 3, 512, 512)
        assert output_tensor.dtype == torch.float32
    
    def test_complete_sdxl_pipeline(self, node):
        """Test complete SDXL VAE pipeline"""
        mock_vae = self.create_mock_vae("SDXL")
        
        # Simulate input image (1024x1024 RGB)
        input_image = torch.randn(1, 3, 1024, 1024, dtype=torch.float32)
        
        # Simulate VAE encoding
        latent = self.simulate_vae_encode(input_image)
        assert latent.shape == (1, 16, 128, 128)  # SDXL: 1024/8 = 128
        
        # Test VAE decoding
        samples = {"samples": latent}
        
        with patch('nodes.COMFY_AVAILABLE', True):
            with patch('nodes.comfy') as mock_comfy:
                mock_comfy.utils.tiled_scale = Mock(return_value=torch.randn(1, 3, 1024, 1024))
                
                result = node.decode(mock_vae, samples, use_rocm_optimizations=True)
        
        # Verify output
        assert isinstance(result, tuple)
        assert len(result) == 1
        output_tensor = result[0]
        assert isinstance(output_tensor, torch.Tensor)
        assert output_tensor.shape == (1, 3, 1024, 1024)
        assert output_tensor.dtype == torch.float32
    
    def test_pil_compatibility(self, node):
        """Test that VAE decode output is compatible with PIL"""
        mock_vae = self.create_mock_vae("SD")
        
        # Create latent
        latent = torch.randn(1, 4, 64, 64, dtype=torch.float32)
        samples = {"samples": latent}
        
        with patch('nodes.COMFY_AVAILABLE', True):
            with patch('nodes.comfy') as mock_comfy:
                mock_comfy.utils.tiled_scale = Mock(return_value=torch.randn(1, 3, 512, 512))
                
                result = node.decode(mock_vae, samples, use_rocm_optimizations=True)
        
        # Get the output tensor
        output_tensor = result[0]
        
        # Convert to numpy for PIL compatibility testing
        output_np = output_tensor.detach().cpu().numpy()
        
        # Test PIL compatibility
        try:
            # Simulate ComfyUI's image processing
            # Convert from (B, C, H, W) to (H, W, C) for PIL
            if len(output_np.shape) == 4:
                # Take first image from batch
                img_array = output_np[0]  # (C, H, W)
                img_array = np.transpose(img_array, (1, 2, 0))  # (H, W, C)
            
            # Clamp values to [0, 255] and convert to uint8
            img_array = np.clip(img_array, 0, 255).astype(np.uint8)
            
            # Test PIL Image creation
            pil_image = PIL.Image.fromarray(img_array)
            
            # Verify PIL image properties
            assert pil_image.size == (512, 512)
            assert pil_image.mode == 'RGB'
            
            print(f"✅ PIL compatibility test passed")
            print(f"   Input shape: {output_tensor.shape}")
            print(f"   PIL image size: {pil_image.size}")
            print(f"   PIL image mode: {pil_image.mode}")
            
        except Exception as e:
            pytest.fail(f"PIL compatibility test failed: {e}")
    
    def test_different_image_sizes(self, node):
        """Test VAE decode with different image sizes"""
        test_sizes = [
            (256, 256),   # Small
            (512, 512),   # Medium
            (768, 768),   # Large
            (1024, 1024), # Very large
        ]
        
        for H, W in test_sizes:
            # Determine architecture based on size
            if H <= 512:
                vae_type = "SD"
                latent_channels = 4
            else:
                vae_type = "SDXL"
                latent_channels = 16
            
            mock_vae = self.create_mock_vae(vae_type)
            
            # Create latent
            latent_h, latent_w = H // 8, W // 8
            latent = torch.randn(1, latent_channels, latent_h, latent_w, dtype=torch.float32)
            samples = {"samples": latent}
            
            with patch('nodes.COMFY_AVAILABLE', True):
                with patch('nodes.comfy') as mock_comfy:
                    mock_comfy.utils.tiled_scale = Mock(return_value=torch.randn(1, 3, H, W))
                    
                    result = node.decode(mock_vae, samples, use_rocm_optimizations=True)
            
            # Verify output
            output_tensor = result[0]
            assert output_tensor.shape == (1, 3, H, W), f"Failed for {H}x{W}"
            assert output_tensor.dtype == torch.float32
    
    def test_batch_processing(self, node):
        """Test VAE decode with batch processing"""
        mock_vae = self.create_mock_vae("SD")
        
        # Create batch of latents
        batch_size = 4
        latent = torch.randn(batch_size, 4, 64, 64, dtype=torch.float32)
        samples = {"samples": latent}
        
        with patch('nodes.COMFY_AVAILABLE', True):
            with patch('nodes.comfy') as mock_comfy:
                mock_comfy.utils.tiled_scale = Mock(return_value=torch.randn(batch_size, 3, 512, 512))
                
                result = node.decode(mock_vae, samples, use_rocm_optimizations=True)
        
        # Verify output
        output_tensor = result[0]
        assert output_tensor.shape == (batch_size, 3, 512, 512)
        assert output_tensor.dtype == torch.float32
    
    def test_performance_comparison(self, node):
        """Compare performance with different optimization settings"""
        mock_vae = self.create_mock_vae("SD")
        
        # Create test latent
        latent = torch.randn(1, 4, 64, 64, dtype=torch.float32)
        samples = {"samples": latent}
        
        test_configs = [
            {'use_rocm_optimizations': False, 'precision_mode': 'fp32'},
            {'use_rocm_optimizations': True, 'precision_mode': 'fp32'},
            {'use_rocm_optimizations': True, 'precision_mode': 'fp16'},
            {'use_rocm_optimizations': True, 'precision_mode': 'auto'},
        ]
        
        results = {}
        for config in test_configs:
            config_name = f"rocm_{config['use_rocm_optimizations']}_precision_{config['precision_mode']}"
            
            try:
                import time
                start_time = time.time()
                
                with patch('nodes.COMFY_AVAILABLE', True):
                    with patch('nodes.comfy') as mock_comfy:
                        mock_comfy.utils.tiled_scale = Mock(return_value=torch.randn(1, 3, 512, 512))
                        
                        result = node.decode(mock_vae, samples, **config)
                
                end_time = time.time()
                
                results[config_name] = {
                    'execution_time': end_time - start_time,
                    'output_shape': result[0].shape,
                    'success': True
                }
                
            except Exception as e:
                results[config_name] = {
                    'execution_time': None,
                    'error': str(e),
                    'success': False
                }
        
        # Print results
        print("\nPerformance Comparison:")
        for config_name, result in results.items():
            if result['success']:
                print(f"{config_name}: {result['execution_time']:.4f}s, shape: {result['output_shape']}")
            else:
                print(f"{config_name}: FAILED - {result['error']}")
        
        # Verify all configs succeeded
        for config_name, result in results.items():
            assert result['success'], f"Config {config_name} failed: {result.get('error', 'Unknown error')}"


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "-s"])

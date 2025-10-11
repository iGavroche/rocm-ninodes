#!/usr/bin/env python3
"""
Comprehensive test for PIL compatibility fixes
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

class TestPILCompatibilityFix:
    """Test PIL compatibility fixes"""
    
    @pytest.fixture
    def node(self):
        return ROCMOptimizedVAEDecode()
    
    def create_mock_vae_with_problematic_process_output(self):
        """Create mock VAE with problematic process_output"""
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
        
        # PROBLEMATIC: Mock process_output that returns wrong format
        def problematic_process_output(tensor):
            """Simulate problematic process_output"""
            # Return wrong format that would cause PIL error
            return torch.randn(1, 1, 512, dtype=torch.uint8)  # This would cause the PIL error
        
        mock_vae.process_output = problematic_process_output
        
        # Mock VAE properties
        mock_vae.spacial_compression_decode = lambda: 8
        mock_vae.upscale_ratio = 8
        mock_vae.latent_channels = 16  # SDXL
        
        return mock_vae
    
    def create_mock_vae_with_crashing_process_output(self):
        """Create mock VAE with crashing process_output"""
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
        
        # CRASHING: Mock process_output that raises exception
        def crashing_process_output(tensor):
            """Simulate crashing process_output"""
            raise Exception("process_output crashed!")
        
        mock_vae.process_output = crashing_process_output
        
        # Mock VAE properties
        mock_vae.spacial_compression_decode = lambda: 8
        mock_vae.upscale_ratio = 8
        mock_vae.latent_channels = 16  # SDXL
        
        return mock_vae
    
    def test_problematic_process_output_handling(self, node):
        """Test handling of problematic process_output"""
        mock_vae = self.create_mock_vae_with_problematic_process_output()
        
        # Create test samples
        samples_tensor = torch.randn(1, 16, 64, 64, dtype=torch.float32)
        samples = {"samples": samples_tensor}
        
        with patch('nodes.COMFY_AVAILABLE', True):
            with patch('nodes.comfy') as mock_comfy:
                mock_comfy.utils.tiled_scale = Mock(return_value=torch.randn(1, 3, 512, 512))
                
                result = node.decode(mock_vae, samples, use_rocm_optimizations=True)
        
        # Verify output is correct despite problematic process_output
        assert isinstance(result, tuple)
        assert len(result) == 1
        output_tensor = result[0]
        assert isinstance(output_tensor, torch.Tensor)
        assert output_tensor.shape == (1, 3, 512, 512)  # Should be corrected
        assert output_tensor.dtype == torch.float32
    
    def test_crashing_process_output_handling(self, node):
        """Test handling of crashing process_output"""
        mock_vae = self.create_mock_vae_with_crashing_process_output()
        
        # Create test samples
        samples_tensor = torch.randn(1, 16, 64, 64, dtype=torch.float32)
        samples = {"samples": samples_tensor}
        
        with patch('nodes.COMFY_AVAILABLE', True):
            with patch('nodes.comfy') as mock_comfy:
                mock_comfy.utils.tiled_scale = Mock(return_value=torch.randn(1, 3, 512, 512))
                
                result = node.decode(mock_vae, samples, use_rocm_optimizations=True)
        
        # Verify output is correct despite crashing process_output
        assert isinstance(result, tuple)
        assert len(result) == 1
        output_tensor = result[0]
        assert isinstance(output_tensor, torch.Tensor)
        assert output_tensor.shape == (1, 3, 512, 512)  # Should be corrected
        assert output_tensor.dtype == torch.float32
    
    def test_pil_compatibility_with_fixes(self, node):
        """Test PIL compatibility with all fixes applied"""
        mock_vae = self.create_mock_vae_with_problematic_process_output()
        
        # Create test samples
        samples_tensor = torch.randn(1, 16, 64, 64, dtype=torch.float32)
        samples = {"samples": samples_tensor}
        
        with patch('nodes.COMFY_AVAILABLE', True):
            with patch('nodes.comfy') as mock_comfy:
                mock_comfy.utils.tiled_scale = Mock(return_value=torch.randn(1, 3, 512, 512))
                
                result = node.decode(mock_vae, samples, use_rocm_optimizations=True)
        
        # Get the output tensor
        output_tensor = result[0]
        
        # Test PIL compatibility
        try:
            # Convert to numpy for PIL compatibility testing
            output_np = output_tensor.detach().cpu().numpy()
            
            # Simulate ComfyUI's image processing
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
            
            print(f"✅ PIL compatibility test passed with fixes")
            print(f"   Input shape: {output_tensor.shape}")
            print(f"   PIL image size: {pil_image.size}")
            print(f"   PIL image mode: {pil_image.mode}")
            
        except Exception as e:
            pytest.fail(f"PIL compatibility test failed: {e}")
    
    def test_different_vae_types_with_fixes(self, node):
        """Test different VAE types with fixes"""
        test_cases = [
            (4, "SD"),    # 4-channel SD
            (16, "SDXL"), # 16-channel SDXL
        ]
        
        for channels, vae_type in test_cases:
            # Create mock VAE
            mock_vae = Mock()
            mock_vae.device = torch.device('cpu')
            mock_vae.vae_dtype = torch.float32
            mock_vae.output_device = torch.device('cpu')
            mock_vae.first_stage_model = Mock()
            mock_vae.first_stage_model.dtype = torch.float32
            mock_vae.patcher = Mock()
            
            def mock_memory_used_decode(shape, dtype):
                elements = 1
                for dim in shape:
                    elements *= dim
                return elements * 4 / 1024**3
            
            mock_vae.memory_used_decode = mock_memory_used_decode
            
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
            
            # Add problematic process_output
            def problematic_process_output(tensor):
                return torch.randn(1, 1, 512, dtype=torch.uint8)  # Wrong format
            
            mock_vae.process_output = problematic_process_output
            
            mock_vae.spacial_compression_decode = lambda: 8
            mock_vae.upscale_ratio = 8
            mock_vae.latent_channels = channels
            
            # Create test samples
            samples_tensor = torch.randn(1, channels, 64, 64, dtype=torch.float32)
            samples = {"samples": samples_tensor}
            
            with patch('nodes.COMFY_AVAILABLE', True):
                with patch('nodes.comfy') as mock_comfy:
                    mock_comfy.utils.tiled_scale = Mock(return_value=torch.randn(1, 3, 512, 512))
                    
                    result = node.decode(mock_vae, samples, use_rocm_optimizations=True)
            
            # Verify output is correct
            output_tensor = result[0]
            assert output_tensor.shape == (1, 3, 512, 512), f"Failed for {vae_type}"
            assert output_tensor.dtype == torch.float32, f"Failed for {vae_type}"


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "-s"])

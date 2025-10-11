#!/usr/bin/env python3
"""
Comprehensive tests for different VAE channel support
Tests SD (4-channel) and SDXL (16-channel) latents
"""

import pytest
import torch
import numpy as np
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from nodes import ROCMOptimizedVAEDecode

class TestVAEChannelSupport:
    """Test VAE decode with different channel counts"""
    
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
    
    def test_sd_4_channel_latent(self, node):
        """Test SD VAE with 4-channel latent"""
        mock_vae = self.create_mock_vae("SD")
        
        # Create 4-channel latent (SD)
        samples_tensor = torch.randn(1, 4, 64, 64, dtype=torch.float32)
        samples = {"samples": samples_tensor}
        
        # Mock ComfyUI modules
        with patch('nodes.COMFY_AVAILABLE', True):
            with patch('nodes.comfy') as mock_comfy:
                mock_comfy.utils.tiled_scale = Mock(return_value=torch.randn(1, 3, 512, 512))
                
                result = node.decode(mock_vae, samples, use_rocm_optimizations=True)
        
        # Verify output - now in (B, H, W, C) format for ComfyUI compatibility
        assert isinstance(result, tuple)
        assert len(result) == 1
        output_tensor = result[0]
        assert isinstance(output_tensor, torch.Tensor)
        assert output_tensor.shape[0] == 1  # Batch size
        assert output_tensor.shape[1] == 512  # Height (64 * 8)
        assert output_tensor.shape[2] == 512  # Width (64 * 8)
        assert output_tensor.shape[3] == 3  # RGB channels (now last dimension)
    
    def test_sdxl_16_channel_latent(self, node):
        """Test SDXL VAE with 16-channel latent"""
        mock_vae = self.create_mock_vae("SDXL")
        
        # Create 16-channel latent (SDXL)
        samples_tensor = torch.randn(1, 16, 64, 64, dtype=torch.float32)
        samples = {"samples": samples_tensor}
        
        # Mock ComfyUI modules
        with patch('nodes.COMFY_AVAILABLE', True):
            with patch('nodes.comfy') as mock_comfy:
                mock_comfy.utils.tiled_scale = Mock(return_value=torch.randn(1, 3, 512, 512))
                
                result = node.decode(mock_vae, samples, use_rocm_optimizations=True)
        
        # Verify output - now in (B, H, W, C) format for ComfyUI compatibility
        assert isinstance(result, tuple)
        assert len(result) == 1
        output_tensor = result[0]
        assert isinstance(output_tensor, torch.Tensor)
        assert output_tensor.shape[0] == 1  # Batch size
        assert output_tensor.shape[1] == 512  # Height (64 * 8)
        assert output_tensor.shape[2] == 512  # Width (64 * 8)
        assert output_tensor.shape[3] == 3  # RGB channels (now last dimension)
    
    def test_unsupported_channel_count(self, node):
        """Test error handling for unsupported channel counts"""
        mock_vae = self.create_mock_vae("SD")
        
        # Create unsupported channel count
        samples_tensor = torch.randn(1, 8, 64, 64, dtype=torch.float32)
        samples = {"samples": samples_tensor}
        
        # Mock ComfyUI modules
        with patch('nodes.COMFY_AVAILABLE', True):
            with pytest.raises(ValueError, match="Unsupported latent channels: 8"):
                node.decode(mock_vae, samples, use_rocm_optimizations=True)
    
    def test_different_batch_sizes(self, node):
        """Test different batch sizes with both channel types"""
        test_cases = [
            (1, 4, 32, 32),   # Single SD image
            (2, 4, 64, 64),   # Batch of SD images
            (1, 16, 32, 32),  # Single SDXL image
            (4, 16, 64, 64),  # Batch of SDXL images
        ]
        
        for B, C, H, W in test_cases:
            vae_type = "SD" if C == 4 else "SDXL"
            mock_vae = self.create_mock_vae(vae_type)
            
            samples_tensor = torch.randn(B, C, H, W, dtype=torch.float32)
            samples = {"samples": samples_tensor}
            
            # Mock ComfyUI modules
            with patch('nodes.COMFY_AVAILABLE', True):
                with patch('nodes.comfy') as mock_comfy:
                    expected_h, expected_w = H * 8, W * 8
                    mock_comfy.utils.tiled_scale = Mock(return_value=torch.randn(B, 3, expected_h, expected_w))
                    
                    result = node.decode(mock_vae, samples, use_rocm_optimizations=True)
            
            # Verify output shape - now in (B, H, W, C) format for ComfyUI compatibility
            output_tensor = result[0]
            assert output_tensor.shape == (B, expected_h, expected_w, 3), f"Failed for {B}x{C}x{H}x{W}"
    
    def test_different_resolutions(self, node):
        """Test different resolutions with both channel types"""
        resolutions = [
            (32, 32),   # Small
            (64, 64),   # Medium
            (128, 128), # Large
            (256, 256), # Very large
        ]
        
        for H, W in resolutions:
            for C in [4, 16]:  # Test both SD and SDXL
                vae_type = "SD" if C == 4 else "SDXL"
                mock_vae = self.create_mock_vae(vae_type)
                
                samples_tensor = torch.randn(1, C, H, W, dtype=torch.float32)
                samples = {"samples": samples_tensor}
                
                # Mock ComfyUI modules
                with patch('nodes.COMFY_AVAILABLE', True):
                    with patch('nodes.comfy') as mock_comfy:
                        expected_h, expected_w = H * 8, W * 8
                        mock_comfy.utils.tiled_scale = Mock(return_value=torch.randn(1, 3, expected_h, expected_w))
                        
                        result = node.decode(mock_vae, samples, use_rocm_optimizations=True)
                
                # Verify output shape - now in (B, H, W, C) format for ComfyUI compatibility
                output_tensor = result[0]
                assert output_tensor.shape == (1, expected_h, expected_w, 3), f"Failed for {C}ch {H}x{W}"
    
    def test_fallback_behavior(self, node):
        """Test fallback behavior when VAE decode fails"""
        mock_vae = self.create_mock_vae("SD")
        
        # Create samples that will cause VAE decode to fail
        samples_tensor = torch.randn(1, 4, 32, 32, dtype=torch.float32)
        samples = {"samples": samples_tensor}
        
        # Mock ComfyUI modules to simulate failure
        with patch('nodes.COMFY_AVAILABLE', True):
            with patch('nodes.comfy') as mock_comfy:
                # Make tiled_scale fail
                mock_comfy.utils.tiled_scale.side_effect = Exception("Tiled decode failed")
                
                # Make VAE decode fail
                mock_vae.first_stage_model.decode.side_effect = Exception("VAE decode failed")
                mock_vae.decode.side_effect = Exception("VAE decode failed")
                
                result = node.decode(mock_vae, samples, use_rocm_optimizations=True)
        
        # Should still return a valid output (fallback)
        assert isinstance(result, tuple)
        assert len(result) == 1
        output_tensor = result[0]
        assert isinstance(output_tensor, torch.Tensor)
        assert output_tensor.shape == (1, 3, 256, 256)  # 32 * 8 = 256
    
    def test_sdxl_fallback_behavior(self, node):
        """Test fallback behavior for SDXL when VAE decode fails"""
        mock_vae = self.create_mock_vae("SDXL")
        
        # Create SDXL samples that will cause VAE decode to fail
        samples_tensor = torch.randn(1, 16, 64, 64, dtype=torch.float32)
        samples = {"samples": samples_tensor}
        
        # Mock ComfyUI modules to simulate failure
        with patch('nodes.COMFY_AVAILABLE', True):
            with patch('nodes.comfy') as mock_comfy:
                # Make tiled_scale fail
                mock_comfy.utils.tiled_scale.side_effect = Exception("Tiled decode failed")
                
                # Make VAE decode fail
                mock_vae.first_stage_model.decode.side_effect = Exception("VAE decode failed")
                mock_vae.decode.side_effect = Exception("VAE decode failed")
                
                result = node.decode(mock_vae, samples, use_rocm_optimizations=True)
        
        # Should still return a valid output (fallback)
        assert isinstance(result, tuple)
        assert len(result) == 1
        output_tensor = result[0]
        assert isinstance(output_tensor, torch.Tensor)
        assert output_tensor.shape == (1, 3, 512, 512)  # 64 * 8 = 512
    
    def test_instrumentation_capture(self, node):
        """Test that instrumentation captures different channel types"""
        mock_vae = self.create_mock_vae("SDXL")
        
        # Create SDXL samples
        samples_tensor = torch.randn(1, 16, 32, 32, dtype=torch.float32)
        samples = {"samples": samples_tensor}
        
        # Mock ComfyUI modules and instrumentation
        with patch('nodes.COMFY_AVAILABLE', True):
            with patch('nodes.comfy') as mock_comfy:
                with patch('nodes.INSTRUMENTATION_AVAILABLE', True):
                    with patch('nodes.instrumentation') as mock_instrumentation:
                        mock_comfy.utils.tiled_scale = Mock(return_value=torch.randn(1, 3, 256, 256))
                        
                        result = node.decode(mock_vae, samples, use_rocm_optimizations=True)
        
        # Verify instrumentation was called
        assert mock_instrumentation.capture_inputs.called
        assert mock_instrumentation.capture_outputs.called
        assert mock_instrumentation.capture_performance.called
        
        # Verify input capture included channel info
        captured_inputs = mock_instrumentation.capture_inputs.call_args[0][1]
        assert 'samples' in captured_inputs


class TestRealDataChannelAnalysis:
    """Analyze real captured data for channel patterns"""
    
    def test_analyze_captured_channel_data(self):
        """Analyze captured data to understand channel usage patterns"""
        import os
        import pickle
        
        test_data_dir = "test_data/inputs"
        if not os.path.exists(test_data_dir):
            pytest.skip("No captured data available")
        
        channel_counts = {}
        total_cases = 0
        
        for filename in os.listdir(test_data_dir):
            if filename.startswith('ROCMOptimizedVAEDecode') and filename.endswith('.pkl'):
                try:
                    with open(os.path.join(test_data_dir, filename), 'rb') as f:
                        data = pickle.load(f)
                        if 'inputs' in data and 'samples' in data['inputs']:
                            samples_data = data['inputs']['samples']
                            if isinstance(samples_data, dict) and 'samples' in samples_data:
                                shape = samples_data['samples']['shape']
                                if len(shape) >= 2:
                                    channels = shape[1]
                                    if channels not in channel_counts:
                                        channel_counts[channels] = 0
                                    channel_counts[channels] += 1
                                    total_cases += 1
                except Exception as e:
                    print(f"Warning: Could not analyze {filename}: {e}")
        
        print(f"\nChannel Analysis from Real Data:")
        print(f"Total cases analyzed: {total_cases}")
        for channels, count in sorted(channel_counts.items()):
            percentage = (count / total_cases) * 100 if total_cases > 0 else 0
            print(f"  {channels} channels: {count} cases ({percentage:.1f}%)")
        
        # Verify we have data
        assert total_cases > 0, "No channel data found in captured inputs"
        
        # Check if we have the new 16-channel case
        if 16 in channel_counts:
            print(f"✅ Found {channel_counts[16]} cases with 16-channel latents (SDXL)")
        else:
            print("ℹ️  No 16-channel latents found in current data")


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "-s"])

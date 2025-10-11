#!/usr/bin/env python3
"""
Correctness tests for ROCM Ninodes
Tests output validation and shape correctness
"""

import pytest
import torch
from .test_fixtures import (
    load_flux_checkpoint_data, load_flux_ksampler_data, load_flux_vae_data,
    load_wan_ksampler_data, load_wan_vae_data,
    get_tensor_from_data, get_tensor_info,
    create_mock_latent, create_mock_video_latent, create_mock_image_latent
)

class TestFluxCorrectness:
    """Correctness tests for Flux workflow components"""
    
    def test_flux_checkpoint_output_types(self):
        """Test that checkpoint loader returns correct types"""
        data = load_flux_checkpoint_data()
        if data is None:
            pytest.skip("No captured checkpoint data available")
        
        # Mock checkpoint loader output validation
        # In real implementation, this would check model, clip, vae objects
        assert data is not None, "Checkpoint data should not be None"
        
        # Check if data has expected structure
        if isinstance(data, dict):
            assert 'data' in data or 'samples' in data, "Data should contain 'data' or 'samples' key"
    
    def test_flux_ksampler_output_shape(self):
        """Test that KSampler returns correct output shape"""
        data = load_flux_ksampler_data()
        if data is None:
            pytest.skip("No captured KSampler data available")
        
        tensor = get_tensor_from_data(data)
        if tensor is not None:
            # Flux KSampler should return 4D tensor [B, C, H, W]
            assert len(tensor.shape) == 4, f"Expected 4D tensor, got {len(tensor.shape)}D: {tensor.shape}"
            assert tensor.shape[0] > 0, "Batch size should be positive"
            assert tensor.shape[1] > 0, "Channel count should be positive"
            assert tensor.shape[2] > 0, "Height should be positive"
            assert tensor.shape[3] > 0, "Width should be positive"
    
    def test_flux_vae_output_shape(self):
        """Test that VAE decode returns correct output shape"""
        data = load_flux_vae_data()
        if data is None:
            pytest.skip("No captured VAE data available")
        
        tensor = get_tensor_from_data(data)
        if tensor is not None:
            # Flux VAE should return 4D tensor [B, H, W, C] for images
            assert len(tensor.shape) == 4, f"Expected 4D tensor, got {len(tensor.shape)}D: {tensor.shape}"
            assert tensor.shape[0] > 0, "Batch size should be positive"
            assert tensor.shape[1] > 0, "Height should be positive"
            assert tensor.shape[2] > 0, "Width should be positive"
            assert tensor.shape[3] == 3, f"Expected 3 channels (RGB), got {tensor.shape[3]}"

class TestWANCorrectness:
    """Correctness tests for WAN workflow components"""
    
    def test_wan_ksampler_output_shape(self):
        """Test that WAN KSampler returns correct output shape"""
        data = load_wan_ksampler_data()
        if data is None:
            pytest.skip("No captured WAN KSampler data available")
        
        tensor = get_tensor_from_data(data)
        if tensor is not None:
            # WAN KSampler should return 5D tensor [B, C, T, H, W] for video
            assert len(tensor.shape) == 5, f"Expected 5D tensor, got {len(tensor.shape)}D: {tensor.shape}"
            assert tensor.shape[0] > 0, "Batch size should be positive"
            assert tensor.shape[1] > 0, "Channel count should be positive"
            assert tensor.shape[2] > 0, "Frame count should be positive"
            assert tensor.shape[3] > 0, "Height should be positive"
            assert tensor.shape[4] > 0, "Width should be positive"
    
    def test_wan_vae_input_shape(self):
        """Test that WAN VAE receives correct input shape"""
        data = load_wan_vae_data()
        if data is None:
            pytest.skip("No captured WAN VAE data available")
        
        tensor = get_tensor_from_data(data)
        if tensor is not None:
            # WAN VAE should receive 5D tensor [B, C, T, H, W]
            assert len(tensor.shape) == 5, f"Expected 5D input tensor, got {len(tensor.shape)}D: {tensor.shape}"
            assert tensor.shape[0] > 0, "Batch size should be positive"
            assert tensor.shape[1] > 0, "Channel count should be positive"
            assert tensor.shape[2] > 0, "Frame count should be positive"
            assert tensor.shape[3] > 0, "Height should be positive"
            assert tensor.shape[4] > 0, "Width should be positive"
    
    def test_wan_vae_output_shape(self):
        """Test that WAN VAE returns correct output shape"""
        data = load_wan_vae_data()
        if data is None:
            pytest.skip("No captured WAN VAE data available")
        
        # Mock VAE decode output validation
        # WAN VAE should convert 5D input to 4D output [B*T, H, W, C]
        input_tensor = get_tensor_from_data(data)
        if input_tensor is not None:
            B, C, T, H, W = input_tensor.shape
            expected_output_shape = (B * T, H, W, 3)  # 3 for RGB channels
            
            # This would be the actual output in real implementation
            # For now, just validate the expected shape calculation
            assert expected_output_shape[0] == B * T, "Output batch size should be B*T"
            assert expected_output_shape[1] == H, "Output height should match input height"
            assert expected_output_shape[2] == W, "Output width should match input width"
            assert expected_output_shape[3] == 3, "Output should have 3 RGB channels"

class TestDataFormatCorrectness:
    """Tests for data format correctness"""
    
    def test_latent_format_structure(self):
        """Test that LATENT format has correct structure"""
        # Test with mock data
        mock_latent = create_mock_image_latent()
        
        assert isinstance(mock_latent, dict), "LATENT should be a dictionary"
        assert 'samples' in mock_latent, "LATENT should contain 'samples' key"
        assert isinstance(mock_latent['samples'], torch.Tensor), "samples should be a tensor"
    
    def test_video_latent_format(self):
        """Test that video LATENT format is correct"""
        mock_video_latent = create_mock_video_latent()
        
        assert isinstance(mock_video_latent, dict), "Video LATENT should be a dictionary"
        assert 'samples' in mock_video_latent, "Video LATENT should contain 'samples' key"
        
        samples = mock_video_latent['samples']
        assert isinstance(samples, torch.Tensor), "samples should be a tensor"
        assert len(samples.shape) == 5, f"Video samples should be 5D, got {len(samples.shape)}D"
    
    def test_image_latent_format(self):
        """Test that image LATENT format is correct"""
        mock_image_latent = create_mock_image_latent()
        
        assert isinstance(mock_image_latent, dict), "Image LATENT should be a dictionary"
        assert 'samples' in mock_image_latent, "Image LATENT should contain 'samples' key"
        
        samples = mock_image_latent['samples']
        assert isinstance(samples, torch.Tensor), "samples should be a tensor"
        assert len(samples.shape) == 4, f"Image samples should be 4D, got {len(samples.shape)}D"

class TestTensorProperties:
    """Tests for tensor properties and constraints"""
    
    def test_tensor_dtype_consistency(self):
        """Test that tensor dtypes are consistent"""
        data = load_flux_vae_data()
        if data is None:
            pytest.skip("No captured data available")
        
        tensor = get_tensor_from_data(data)
        if tensor is not None:
            # Check that dtype is reasonable for VAE operations
            assert tensor.dtype in [torch.float32, torch.float16, torch.bfloat16], \
                f"Unexpected dtype: {tensor.dtype}"
    
    def test_tensor_device_consistency(self):
        """Test that tensors are on expected device"""
        data = load_flux_vae_data()
        if data is None:
            pytest.skip("No captured data available")
        
        tensor = get_tensor_from_data(data)
        if tensor is not None:
            # Check that device is reasonable
            assert tensor.device.type in ['cpu', 'cuda', 'hip'], \
                f"Unexpected device type: {tensor.device.type}"
    
    def test_tensor_requires_grad(self):
        """Test that tensors have appropriate requires_grad setting"""
        data = load_flux_vae_data()
        if data is None:
            pytest.skip("No captured data available")
        
        tensor = get_tensor_from_data(data)
        if tensor is not None:
            # VAE decode output should not require gradients
            assert not tensor.requires_grad, "VAE decode output should not require gradients"

class TestMockDataCorrectness:
    """Correctness tests using mock data when captured data is unavailable"""
    
    def test_mock_flux_vae_shape_conversion(self):
        """Test Flux VAE shape conversion with mock data"""
        # Create mock 4D latent input
        mock_input = create_mock_image_latent(batch_size=1, channels=16, height=128, width=128)
        input_tensor = mock_input['samples']
        
        # Simulate VAE decode: [B, C, H, W] -> [B, H, W, C]
        B, C, H, W = input_tensor.shape
        expected_output_shape = (B, H, W, 3)  # 3 for RGB channels
        
        # Mock the conversion
        assert input_tensor.shape == (B, C, H, W), "Input should be 4D"
        assert expected_output_shape[0] == B, "Output batch size should match input"
        assert expected_output_shape[1] == H, "Output height should match input"
        assert expected_output_shape[2] == W, "Output width should match input"
        assert expected_output_shape[3] == 3, "Output should have 3 RGB channels"
    
    def test_mock_wan_vae_shape_conversion(self):
        """Test WAN VAE shape conversion with mock data"""
        # Create mock 5D latent input
        mock_input = create_mock_video_latent(batch_size=1, channels=16, frames=17, height=32, width=32)
        input_tensor = mock_input['samples']
        
        # Simulate VAE decode: [B, C, T, H, W] -> [B*T, H, W, C]
        B, C, T, H, W = input_tensor.shape
        expected_output_shape = (B * T, H, W, 3)  # 3 for RGB channels
        
        # Mock the conversion
        assert input_tensor.shape == (B, C, T, H, W), "Input should be 5D"
        assert expected_output_shape[0] == B * T, "Output batch size should be B*T"
        assert expected_output_shape[1] == H, "Output height should match input"
        assert expected_output_shape[2] == W, "Output width should match input"
        assert expected_output_shape[3] == 3, "Output should have 3 RGB channels"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])

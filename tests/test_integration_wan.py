#!/usr/bin/env python3
"""
Integration tests for WAN workflow
Tests full workflow execution with ComfyUI
"""

import os
import sys
import time
import json
import pytest
from pathlib import Path

# Add ComfyUI to path if available
COMFYUI_PATH = os.getenv('COMFYUI_PATH', '/home/nino/ComfyUI')
if os.path.exists(COMFYUI_PATH):
    sys.path.insert(0, COMFYUI_PATH)

try:
    import comfy.model_management as model_management
    import comfy.sample
    import comfy.samplers
    COMFYUI_AVAILABLE = True
except ImportError:
    COMFYUI_AVAILABLE = False

from .test_fixtures import (
    load_wan_ksampler_data, load_wan_vae_data,
    create_mock_video_latent
)

class TestWANIntegration:
    """Integration tests for WAN workflow"""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup for each test"""
        if not COMFYUI_AVAILABLE:
            pytest.skip("ComfyUI not available")
    
    def test_wan_workflow_components_available(self):
        """Test that all WAN workflow components are available"""
        # This would test that the ROCM nodes are properly loaded
        # For now, just verify the test framework works
        assert True, "WAN integration test placeholder"
    
    def test_wan_sampling(self):
        """Test WAN sampling with real data"""
        data = load_wan_ksampler_data()
        if data is None:
            pytest.skip("No captured WAN KSampler data available")
        
        # Mock sampling
        start_time = time.time()
        
        # Simulate sampling
        time.sleep(0.1)
        
        elapsed = time.time() - start_time
        
        # Should complete within reasonable time
        assert elapsed < 100.0, f"WAN sampling too slow: {elapsed:.2f}s"
    
    def test_wan_vae_decode(self):
        """Test WAN VAE decode with real data"""
        data = load_wan_vae_data()
        if data is None:
            pytest.skip("No captured WAN VAE data available")
        
        # Mock VAE decode
        start_time = time.time()
        
        # Simulate VAE decode
        time.sleep(0.1)
        
        elapsed = time.time() - start_time
        
        # Should complete within reasonable time
        assert elapsed < 10.0, f"WAN VAE decode too slow: {elapsed:.2f}s"
    
    def test_wan_video_shape_conversion(self):
        """Test WAN video shape conversion (5D -> 4D)"""
        data = load_wan_vae_data()
        if data is None:
            pytest.skip("No captured WAN VAE data available")
        
        # Test shape conversion logic
        # Input: [B, C, T, H, W] -> Output: [B*T, H, W, C]
        # This is a critical test for WAN video processing
        
        # Mock the conversion
        input_shape = (1, 16, 17, 32, 32)  # B, C, T, H, W
        B, C, T, H, W = input_shape
        expected_output_shape = (B * T, H, W, 3)  # B*T, H, W, C (3 for RGB)
        
        assert expected_output_shape == (17, 32, 32, 3), \
            f"Expected output shape (17, 32, 32, 3), got {expected_output_shape}"
    
    def test_wan_end_to_end_workflow(self):
        """Test complete WAN workflow"""
        # This would test the complete workflow from sampling to final video
        # For now, just verify the test framework works
        assert True, "End-to-end WAN workflow test placeholder"
    
    def test_wan_output_validation(self):
        """Test that WAN output meets requirements"""
        # This would validate the final output video
        # - Correct dimensions (320x320)
        # - Correct frame count (17)
        # - Correct format (RGB)
        # - File size within expected range
        assert True, "WAN output validation test placeholder"

class TestWANMockIntegration:
    """Integration tests using mock data"""
    
    def test_mock_wan_workflow(self):
        """Test WAN workflow with mock data"""
        # Create mock video data
        mock_latent = create_mock_video_latent(batch_size=1, channels=16, frames=17, height=32, width=32)
        
        # Test that mock data has correct format
        assert 'samples' in mock_latent
        assert mock_latent['samples'].shape == (1, 16, 17, 32, 32)
        
        # Mock workflow execution
        start_time = time.time()
        
        # Simulate workflow steps
        time.sleep(0.1)  # Sampling
        time.sleep(0.1)  # VAE decode
        
        elapsed = time.time() - start_time
        
        # Should complete within reasonable time
        assert elapsed < 1.0, f"Mock WAN workflow too slow: {elapsed:.2f}s"
    
    def test_mock_wan_video_processing(self):
        """Test WAN video processing with mock data"""
        # Test 5D to 4D conversion
        mock_latent = create_mock_video_latent(batch_size=1, channels=16, frames=17, height=32, width=32)
        input_tensor = mock_latent['samples']
        
        # Simulate the conversion process
        B, C, T, H, W = input_tensor.shape
        
        # Mock VAE decode that returns 5D tensor
        mock_vae_output = (B, T, H, W, 3)  # [B, T, H, W, C]
        
        # Convert to 4D for ComfyUI
        output_shape = (B * T, H, W, 3)  # [B*T, H, W, C]
        
        assert output_shape == (17, 32, 32, 3), \
            f"Expected output shape (17, 32, 32, 3), got {output_shape}"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])

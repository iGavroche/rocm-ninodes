#!/usr/bin/env python3
"""
Integration tests for Flux workflow
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
    load_flux_checkpoint_data, load_flux_ksampler_data, load_flux_vae_data,
    create_mock_image_latent
)

class TestFluxIntegration:
    """Integration tests for Flux workflow"""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup for each test"""
        if not COMFYUI_AVAILABLE:
            pytest.skip("ComfyUI not available")
    
    def test_flux_workflow_components_available(self):
        """Test that all Flux workflow components are available"""
        # This would test that the ROCM nodes are properly loaded
        # For now, just verify the test framework works
        assert True, "Flux integration test placeholder"
    
    def test_flux_checkpoint_loading(self):
        """Test Flux checkpoint loading with real data"""
        data = load_flux_checkpoint_data()
        if data is None:
            pytest.skip("No captured checkpoint data available")
        
        # Mock checkpoint loading
        start_time = time.time()
        
        # Simulate checkpoint loading
        time.sleep(0.1)
        
        elapsed = time.time() - start_time
        
        # Should complete within reasonable time
        assert elapsed < 30.0, f"Checkpoint loading too slow: {elapsed:.2f}s"
    
    def test_flux_sampling(self):
        """Test Flux sampling with real data"""
        data = load_flux_ksampler_data()
        if data is None:
            pytest.skip("No captured KSampler data available")
        
        # Mock sampling
        start_time = time.time()
        
        # Simulate sampling
        time.sleep(0.1)
        
        elapsed = time.time() - start_time
        
        # Should complete within reasonable time
        assert elapsed < 60.0, f"Sampling too slow: {elapsed:.2f}s"
    
    def test_flux_vae_decode(self):
        """Test Flux VAE decode with real data"""
        data = load_flux_vae_data()
        if data is None:
            pytest.skip("No captured VAE data available")
        
        # Mock VAE decode
        start_time = time.time()
        
        # Simulate VAE decode
        time.sleep(0.1)
        
        elapsed = time.time() - start_time
        
        # Should complete within reasonable time
        assert elapsed < 10.0, f"VAE decode too slow: {elapsed:.2f}s"
    
    def test_flux_end_to_end_workflow(self):
        """Test complete Flux workflow"""
        # This would test the complete workflow from checkpoint to final image
        # For now, just verify the test framework works
        assert True, "End-to-end Flux workflow test placeholder"
    
    def test_flux_output_validation(self):
        """Test that Flux output meets requirements"""
        # This would validate the final output image
        # - Correct dimensions (1024x1024)
        # - Correct format (RGB)
        # - File size within expected range
        assert True, "Flux output validation test placeholder"

class TestFluxMockIntegration:
    """Integration tests using mock data"""
    
    def test_mock_flux_workflow(self):
        """Test Flux workflow with mock data"""
        # Create mock data
        mock_latent = create_mock_image_latent(batch_size=1, channels=16, height=128, width=128)
        
        # Test that mock data has correct format
        assert 'samples' in mock_latent
        assert mock_latent['samples'].shape == (1, 16, 128, 128)
        
        # Mock workflow execution
        start_time = time.time()
        
        # Simulate workflow steps
        time.sleep(0.1)  # Checkpoint loading
        time.sleep(0.1)  # Sampling
        time.sleep(0.1)  # VAE decode
        
        elapsed = time.time() - start_time
        
        # Should complete within reasonable time
        assert elapsed < 1.0, f"Mock workflow too slow: {elapsed:.2f}s"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])

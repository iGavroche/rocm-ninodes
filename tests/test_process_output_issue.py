#!/usr/bin/env python3
"""
Test to isolate the process_output issue
"""

import torch
import numpy as np
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from nodes import ROCMOptimizedVAEDecode

def test_process_output_issue():
    """Test the process_output issue"""
    
    node = ROCMOptimizedVAEDecode()
    
    # Create mock VAE with problematic process_output
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
    
    # Mock decode function that returns correct format
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
    
    # CRITICAL: Mock process_output to simulate the problematic behavior
    def problematic_process_output(tensor):
        """Simulate problematic process_output that causes PIL issues"""
        print(f"process_output input: {tensor.shape}, {tensor.dtype}")
        
        # Simulate what might be happening - wrong tensor manipulation
        # This could be causing the (1, 1, 512) issue
        if len(tensor.shape) == 4:  # (B, C, H, W)
            # Wrong manipulation that could cause issues
            # Let's simulate a problematic case
            problematic_tensor = tensor[:, :1, :1, :]  # Take only first channel and first pixel
            print(f"process_output output (problematic): {problematic_tensor.shape}, {problematic_tensor.dtype}")
            return problematic_tensor
        return tensor
    
    mock_vae.process_output = problematic_process_output
    
    # Mock VAE properties
    mock_vae.spacial_compression_decode = lambda: 8
    mock_vae.upscale_ratio = 8
    mock_vae.latent_channels = 16  # SDXL
    
    # Create test samples
    samples_tensor = torch.randn(1, 16, 64, 64, dtype=torch.float32)
    samples = {"samples": samples_tensor}
    
    print("Testing with problematic process_output...")
    
    try:
        with patch('nodes.COMFY_AVAILABLE', True):
            with patch('nodes.comfy') as mock_comfy:
                mock_comfy.utils.tiled_scale = Mock(return_value=torch.randn(1, 3, 512, 512))
                
                result = node.decode(mock_vae, samples, use_rocm_optimizations=True)
        
        print(f"Result: {result[0].shape}, {result[0].dtype}")
        
    except Exception as e:
        print(f"Error: {e}")

def test_fix_process_output():
    """Test the fix for process_output issue"""
    
    node = ROCMOptimizedVAEDecode()
    
    # Create mock VAE with fixed process_output
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
    
    # FIXED: Mock process_output that preserves correct format
    def fixed_process_output(tensor):
        """Fixed process_output that preserves correct tensor format"""
        print(f"process_output input: {tensor.shape}, {tensor.dtype}")
        
        # Just return the tensor as-is, don't manipulate it
        print(f"process_output output (fixed): {tensor.shape}, {tensor.dtype}")
        return tensor
    
    mock_vae.process_output = fixed_process_output
    
    # Mock VAE properties
    mock_vae.spacial_compression_decode = lambda: 8
    mock_vae.upscale_ratio = 8
    mock_vae.latent_channels = 16  # SDXL
    
    # Create test samples
    samples_tensor = torch.randn(1, 16, 64, 64, dtype=torch.float32)
    samples = {"samples": samples_tensor}
    
    print("\nTesting with fixed process_output...")
    
    try:
        with patch('nodes.COMFY_AVAILABLE', True):
            with patch('nodes.comfy') as mock_comfy:
                mock_comfy.utils.tiled_scale = Mock(return_value=torch.randn(1, 3, 512, 512))
                
                result = node.decode(mock_vae, samples, use_rocm_optimizations=True)
        
        print(f"Result: {result[0].shape}, {result[0].dtype}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_process_output_issue()
    test_fix_process_output()

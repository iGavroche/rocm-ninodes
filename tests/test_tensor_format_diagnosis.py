#!/usr/bin/env python3
"""
Diagnostic test to capture the exact tensor format issue
"""

import torch
import numpy as np
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import PIL.Image

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from nodes import ROCMOptimizedVAEDecode

def test_tensor_format_diagnosis():
    """Test to diagnose the exact tensor format issue"""
    
    node = ROCMOptimizedVAEDecode()
    
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
    
    # Mock process_output to return the exact problematic format
    def problematic_process_output(tensor):
        """Return the exact format that causes PIL error"""
        print(f"process_output input: {tensor.shape}, {tensor.dtype}")
        
        # Return the exact problematic format: (1, 1, 512) with uint8
        problematic_tensor = torch.zeros(1, 1, 512, dtype=torch.uint8)
        print(f"process_output output (PROBLEMATIC): {problematic_tensor.shape}, {problematic_tensor.dtype}")
        return problematic_tensor
    
    mock_vae.process_output = problematic_process_output
    
    mock_vae.spacial_compression_decode = lambda: 8
    mock_vae.upscale_ratio = 8
    mock_vae.latent_channels = 16  # SDXL
    
    # Create test samples
    samples_tensor = torch.randn(1, 16, 64, 64, dtype=torch.float32)
    samples = {"samples": samples_tensor}
    
    print("Testing with EXACTLY the problematic process_output...")
    
    with patch('nodes.COMFY_AVAILABLE', True):
        with patch('nodes.comfy') as mock_comfy:
            mock_comfy.utils.tiled_scale = Mock(return_value=torch.randn(1, 3, 512, 512))
            
            result = node.decode(mock_vae, samples, use_rocm_optimizations=True)
    
    print(f"Node result: {result[0].shape}, {result[0].dtype}")
    
    # Test PIL compatibility with the result
    try:
        output_tensor = result[0]
        output_np = output_tensor.detach().cpu().numpy()
        
        print(f"Numpy array shape: {output_np.shape}")
        print(f"Numpy array dtype: {output_np.dtype}")
        
        # Simulate ComfyUI's processing
        if len(output_np.shape) == 4:
            img_array = output_np[0]  # (C, H, W)
            img_array = np.transpose(img_array, (1, 2, 0))  # (H, W, C)
        
        print(f"After transpose: {img_array.shape}")
        
        # Clamp and convert to uint8
        img_array = np.clip(img_array, 0, 255).astype(np.uint8)
        print(f"After uint8 conversion: {img_array.shape}, {img_array.dtype}")
        
        # Test PIL
        pil_image = PIL.Image.fromarray(img_array)
        print(f"PIL image created successfully: {pil_image.size}, {pil_image.mode}")
        
    except Exception as e:
        print(f"PIL error: {e}")
        print(f"Error type: {type(e)}")

def test_comfyui_save_images_simulation():
    """Simulate ComfyUI's save_images function"""
    
    node = ROCMOptimizedVAEDecode()
    
    # Create mock VAE that returns problematic format
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
    
    # Mock process_output to return problematic format
    def problematic_process_output(tensor):
        # Return the exact format that causes PIL error: (1, 1, 512) uint8
        return torch.zeros(1, 1, 512, dtype=torch.uint8)
    
    mock_vae.process_output = problematic_process_output
    
    mock_vae.spacial_compression_decode = lambda: 8
    mock_vae.upscale_ratio = 8
    mock_vae.latent_channels = 16  # SDXL
    
    # Create test samples
    samples_tensor = torch.randn(1, 16, 64, 64, dtype=torch.float32)
    samples = {"samples": samples_tensor}
    
    print("\nSimulating ComfyUI save_images function...")
    
    with patch('nodes.COMFY_AVAILABLE', True):
        with patch('nodes.comfy') as mock_comfy:
            mock_comfy.utils.tiled_scale = Mock(return_value=torch.randn(1, 3, 512, 512))
            
            result = node.decode(mock_vae, samples, use_rocm_optimizations=True)
    
    # Simulate ComfyUI's save_images processing
    try:
        output_tensor = result[0]
        print(f"Node output: {output_tensor.shape}, {output_tensor.dtype}")
        
        # Convert to numpy (this is what ComfyUI does)
        output_np = output_tensor.detach().cpu().numpy()
        print(f"Numpy conversion: {output_np.shape}, {output_np.dtype}")
        
        # Simulate the exact ComfyUI save_images processing
        # This is the line that fails: img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
        
        # Process each image in the batch
        for i in range(output_np.shape[0]):
            img_data = output_np[i]  # Get single image
            print(f"Single image shape: {img_data.shape}")
            
            # This is the exact ComfyUI processing
            clipped = np.clip(img_data, 0, 255)
            print(f"After clip: {clipped.shape}, {clipped.dtype}")
            
            uint8_data = clipped.astype(np.uint8)
            print(f"After uint8: {uint8_data.shape}, {uint8_data.dtype}")
            
            # This is where the error occurs
            try:
                img = PIL.Image.fromarray(uint8_data)
                print(f"PIL image created: {img.size}, {img.mode}")
            except Exception as e:
                print(f"PIL ERROR: {e}")
                print(f"Problematic data: shape={uint8_data.shape}, dtype={uint8_data.dtype}")
                
                # Let's see what PIL expects
                print(f"PIL typekey: {(uint8_data.shape, uint8_data.dtype.str)}")
                
    except Exception as e:
        print(f"Error in simulation: {e}")

if __name__ == "__main__":
    test_tensor_format_diagnosis()
    test_comfyui_save_images_simulation()

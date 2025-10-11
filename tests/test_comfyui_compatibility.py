#!/usr/bin/env python3
"""
Test to verify ComfyUI save_images compatibility
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

def test_comfyui_save_images_compatibility():
    """Test compatibility with ComfyUI's save_images function"""
    
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
    
    # Mock process_output to return correct format
    def correct_process_output(tensor):
        return tensor  # Return as-is
    
    mock_vae.process_output = correct_process_output
    
    mock_vae.spacial_compression_decode = lambda: 8
    mock_vae.upscale_ratio = 8
    mock_vae.latent_channels = 16  # SDXL
    
    # Create test samples
    samples_tensor = torch.randn(1, 16, 64, 64, dtype=torch.float32)
    samples = {"samples": samples_tensor}
    
    print("Testing ComfyUI save_images compatibility...")
    
    with patch('nodes.COMFY_AVAILABLE', True):
        with patch('nodes.comfy') as mock_comfy:
            mock_comfy.utils.tiled_scale = Mock(return_value=torch.randn(1, 3, 512, 512))
            
            result = node.decode(mock_vae, samples, use_rocm_optimizations=True)
    
    # Get the output tensor
    output_tensor = result[0]
    print(f"Node output: {output_tensor.shape}, {output_tensor.dtype}")
    
    # Test ComfyUI save_images compatibility
    try:
        # Convert to numpy (this is what ComfyUI does)
        output_np = output_tensor.detach().cpu().numpy()
        print(f"Numpy conversion: {output_np.shape}, {output_np.dtype}")
        
        # Simulate ComfyUI's save_images processing
        # This is the exact line from ComfyUI: img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
        
        # Process each image in the batch
        for i in range(output_np.shape[0]):
            img_data = output_np[i]  # Get single image (C, H, W)
            print(f"Single image shape: {img_data.shape}")
            
            # This is the exact ComfyUI processing
            clipped = np.clip(img_data, 0, 255)
            print(f"After clip: {clipped.shape}, {clipped.dtype}")
            
            uint8_data = clipped.astype(np.uint8)
            print(f"After uint8: {uint8_data.shape}, {uint8_data.dtype}")
            
            # Check if this matches the problematic format
            if uint8_data.shape == (1, 1, 512):
                print(f"❌ PROBLEMATIC FORMAT DETECTED: {uint8_data.shape}")
                print(f"This matches the PIL error!")
                return False
            
            # Test PIL Image creation
            try:
                img = PIL.Image.fromarray(uint8_data)
                print(f"✅ PIL Image created successfully: {img.size}, {img.mode}")
            except Exception as e:
                print(f"❌ PIL ERROR: {e}")
                print(f"Problematic data: shape={uint8_data.shape}, dtype={uint8_data.dtype}")
                return False
        
        print("✅ All images processed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error in ComfyUI simulation: {e}")
        return False

def test_tensor_format_edge_cases():
    """Test edge cases that might cause PIL errors"""
    
    print("\n=== Testing Edge Cases ===")
    
    # Test case 1: Normal format
    normal_tensor = torch.randn(1, 3, 512, 512, dtype=torch.float32)
    print(f"Normal tensor: {normal_tensor.shape}, {normal_tensor.dtype}")
    
    # Test case 2: Wrong channel order
    wrong_channels = torch.randn(1, 512, 512, 3, dtype=torch.float32)
    print(f"Wrong channels: {wrong_channels.shape}, {wrong_channels.dtype}")
    
    # Test case 3: Wrong dimensions
    wrong_dims = torch.randn(1, 1, 512, dtype=torch.float32)
    print(f"Wrong dimensions: {wrong_dims.shape}, {wrong_dims.dtype}")
    
    # Test each case with PIL
    test_cases = [
        ("Normal", normal_tensor),
        ("Wrong Channels", wrong_channels),
        ("Wrong Dimensions", wrong_dims)
    ]
    
    for name, tensor in test_cases:
        try:
            # Convert to numpy
            np_array = tensor.detach().cpu().numpy()
            
            # Process first image
            if len(np_array.shape) == 4:
                img_data = np_array[0]  # (C, H, W)
            else:
                img_data = np_array
            
            # Convert to uint8
            uint8_data = np.clip(img_data, 0, 255).astype(np.uint8)
            
            # Test PIL
            img = PIL.Image.fromarray(uint8_data)
            print(f"✅ {name}: PIL Image created - {img.size}, {img.mode}")
            
        except Exception as e:
            print(f"❌ {name}: PIL Error - {e}")
            print(f"   Shape: {uint8_data.shape}, Dtype: {uint8_data.dtype}")

if __name__ == "__main__":
    success = test_comfyui_save_images_compatibility()
    test_tensor_format_edge_cases()
    
    if success:
        print("\n🎉 All tests passed! The VAE decode node is compatible with ComfyUI's save_images function.")
    else:
        print("\n❌ Tests failed! There's still an issue with PIL compatibility.")

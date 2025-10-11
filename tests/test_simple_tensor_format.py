#!/usr/bin/env python3
"""
Simple test to verify tensor format without ComfyUI dependencies
"""

import torch
import numpy as np
import PIL.Image

def test_tensor_format_compatibility():
    """Test tensor format compatibility with PIL"""
    
    print("=== Testing Tensor Format Compatibility ===")
    
    # Test case 1: Correct format (B, C, H, W)
    correct_tensor = torch.randn(1, 3, 512, 512, dtype=torch.float32)
    print(f"Correct tensor: {correct_tensor.shape}, {correct_tensor.dtype}")
    
    # Test case 2: Problematic format (B, H, W, C) - this might cause issues
    problematic_tensor = torch.randn(1, 512, 512, 3, dtype=torch.float32)
    print(f"Problematic tensor: {problematic_tensor.shape}, {problematic_tensor.dtype}")
    
    # Test case 3: The exact problematic format from the error
    error_tensor = torch.zeros(1, 1, 512, dtype=torch.uint8)
    print(f"Error tensor: {error_tensor.shape}, {error_tensor.dtype}")
    
    test_cases = [
        ("Correct Format", correct_tensor),
        ("Problematic Format", problematic_tensor),
        ("Error Format", error_tensor)
    ]
    
    for name, tensor in test_cases:
        print(f"\n--- Testing {name} ---")
        
        try:
            # Convert to numpy
            np_array = tensor.detach().cpu().numpy()
            print(f"Numpy array: {np_array.shape}, {np_array.dtype}")
            
            # Process first image
            if len(np_array.shape) == 4:
                img_data = np_array[0]  # (C, H, W) or (H, W, C)
            else:
                img_data = np_array
            
            print(f"Single image: {img_data.shape}, {img_data.dtype}")
            
            # Check if this matches the problematic format
            if img_data.shape == (1, 1, 512):
                print(f"❌ This matches the PIL error format!")
                print(f"   Shape: {img_data.shape}, Dtype: {img_data.dtype}")
                continue
            
            # Convert to uint8
            uint8_data = np.clip(img_data, 0, 255).astype(np.uint8)
            print(f"After uint8: {uint8_data.shape}, {uint8_data.dtype}")
            
            # Test PIL Image creation
            try:
                img = PIL.Image.fromarray(uint8_data)
                print(f"✅ PIL Image created: {img.size}, {img.mode}")
            except Exception as e:
                print(f"❌ PIL Error: {e}")
                print(f"   Problematic data: shape={uint8_data.shape}, dtype={uint8_data.dtype}")
                
        except Exception as e:
            print(f"❌ Error processing {name}: {e}")

def test_comfyui_save_images_simulation():
    """Simulate ComfyUI's save_images function"""
    
    print("\n=== ComfyUI Save Images Simulation ===")
    
    # Simulate our VAE decode output
    vae_output = torch.randn(1, 3, 512, 512, dtype=torch.float32)
    print(f"VAE Decode Output: {vae_output.shape}, {vae_output.dtype}")
    
    # Simulate ComfyUI's processing
    try:
        # Convert to numpy
        output_np = vae_output.detach().cpu().numpy()
        print(f"After detach().cpu().numpy(): {output_np.shape}, {output_np.dtype}")
        
        # Simulate ComfyUI's image processing
        # This is where the issue might be occurring
        for i, img in enumerate(output_np):
            print(f"Processing image {i}: {img.shape}, {img.dtype}")
            
            # Check if this is the problematic case
            if img.shape == (1, 1, 512):
                print(f"   ❌ Found problematic shape: {img.shape}")
                print(f"   This matches the PIL error!")
                return
            
            # Normal processing
            if len(img.shape) == 3 and img.shape[0] == 3:  # (C, H, W)
                # Convert to (H, W, C)
                img_array = np.transpose(img, (1, 2, 0))
            elif len(img.shape) == 3 and img.shape[2] == 3:  # (H, W, C)
                img_array = img
            else:
                print(f"   ❌ Unexpected shape: {img.shape}")
                return
            
            # Clamp and convert to uint8
            img_array = np.clip(img_array, 0, 255).astype(np.uint8)
            print(f"   Final array: {img_array.shape}, {img_array.dtype}")
            
            # Create PIL image
            pil_image = PIL.Image.fromarray(img_array)
            print(f"   ✅ PIL Image created: {pil_image.size}, {pil_image.mode}")
            
    except Exception as e:
        print(f"❌ Error in ComfyUI simulation: {e}")

if __name__ == "__main__":
    test_tensor_format_compatibility()
    test_comfyui_save_images_simulation()

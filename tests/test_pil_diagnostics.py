#!/usr/bin/env python3
"""
Diagnostic test to understand PIL compatibility issues
"""

import torch
import numpy as np
import PIL.Image

def test_tensor_formats():
    """Test different tensor formats for PIL compatibility"""
    
    print("=== Tensor Format Analysis ===")
    
    # Test 1: Our VAE decode output format
    print("\n1. VAE Decode Output Format:")
    vae_output = torch.randn(1, 3, 512, 512, dtype=torch.float32)
    print(f"   Shape: {vae_output.shape}")
    print(f"   Dtype: {vae_output.dtype}")
    
    # Convert to numpy
    vae_np = vae_output.detach().cpu().numpy()
    print(f"   Numpy shape: {vae_np.shape}")
    print(f"   Numpy dtype: {vae_np.dtype}")
    
    # Test PIL conversion
    try:
        # Take first image from batch
        img_array = vae_np[0]  # (3, 512, 512)
        img_array = np.transpose(img_array, (1, 2, 0))  # (512, 512, 3)
        img_array = np.clip(img_array, 0, 255).astype(np.uint8)
        
        pil_image = PIL.Image.fromarray(img_array)
        print(f"   ✅ PIL conversion successful: {pil_image.size}, {pil_image.mode}")
    except Exception as e:
        print(f"   ❌ PIL conversion failed: {e}")
    
    # Test 2: The problematic format from the error
    print("\n2. Problematic Format (from error):")
    problematic = np.random.randint(0, 255, (1, 1, 512), dtype=np.uint8)
    print(f"   Shape: {problematic.shape}")
    print(f"   Dtype: {problematic.dtype}")
    
    try:
        pil_image = PIL.Image.fromarray(problematic)
        print(f"   ✅ PIL conversion successful: {pil_image.size}, {pil_image.mode}")
    except Exception as e:
        print(f"   ❌ PIL conversion failed: {e}")
    
    # Test 3: Common ComfyUI image formats
    print("\n3. Common ComfyUI Image Formats:")
    
    # Format 1: (H, W, C) - most common
    format1 = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    print(f"   Format 1 (H, W, C): {format1.shape}, {format1.dtype}")
    try:
        pil_image = PIL.Image.fromarray(format1)
        print(f"   ✅ PIL conversion successful: {pil_image.size}, {pil_image.mode}")
    except Exception as e:
        print(f"   ❌ PIL conversion failed: {e}")
    
    # Format 2: (B, H, W, C) - batch format
    format2 = np.random.randint(0, 255, (1, 512, 512, 3), dtype=np.uint8)
    print(f"   Format 2 (B, H, W, C): {format2.shape}, {format2.dtype}")
    try:
        img_array = format2[0]  # Take first image
        pil_image = PIL.Image.fromarray(img_array)
        print(f"   ✅ PIL conversion successful: {pil_image.size}, {pil_image.mode}")
    except Exception as e:
        print(f"   ❌ PIL conversion failed: {e}")
    
    # Format 3: (C, H, W) - channel first
    format3 = np.random.randint(0, 255, (3, 512, 512), dtype=np.uint8)
    print(f"   Format 3 (C, H, W): {format3.shape}, {format3.dtype}")
    try:
        img_array = np.transpose(format3, (1, 2, 0))  # Convert to (H, W, C)
        pil_image = PIL.Image.fromarray(img_array)
        print(f"   ✅ PIL conversion successful: {pil_image.size}, {pil_image.mode}")
    except Exception as e:
        print(f"   ❌ PIL conversion failed: {e}")

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
    test_tensor_formats()
    test_comfyui_save_images_simulation()

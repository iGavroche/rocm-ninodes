#!/usr/bin/env python3
"""
Debug script for VAE decode issues
Loads the saved debug data and tests VAE decode in isolation
"""

import pickle
import torch
import sys
import os

# Add ComfyUI to path
sys.path.append('/home/nino/ComfyUI')

def debug_vae_decode():
    """Debug VAE decode with saved data"""
    
    # Load debug data - find the most recent file
    import glob
    debug_files = glob.glob('test_data/debug/wan_vae_input_debug_*.pkl')
    if not debug_files:
        print("No debug data found in test_data/debug/")
        print("Run the WAN workflow first to generate test data")
        return
    
    # Use the most recent file
    latest_file = max(debug_files, key=os.path.getctime)
    print(f"Loading debug data from: {latest_file}")
    
    try:
        with open(latest_file, 'rb') as f:
            debug_data = pickle.load(f)
        print("Loaded debug data:")
        for key, value in debug_data.items():
            if key != 'chunk_reshaped_tensor':  # Don't print the large tensor
                print(f"  {key}: {value}")
    except Exception as e:
        print(f"Error loading debug data: {e}")
        return
    
    # Create a mock tensor with the same shape
    shape = debug_data['chunk_reshaped_shape']
    dtype = debug_data['chunk_reshaped_dtype']
    device = debug_data['chunk_reshaped_device']
    
    print(f"\nCreating mock tensor:")
    print(f"  Shape: {shape}")
    print(f"  Dtype: {dtype}")
    print(f"  Device: {device}")
    
    # Create mock tensor
    mock_tensor = torch.randn(shape, dtype=dtype, device=device)
    
    # Test different input formats
    print(f"\nTesting different input formats:")
    
    # Format 1: Direct tensor (what we were doing wrong)
    print(f"1. Direct tensor: {type(mock_tensor)}")
    print(f"   Shape: {mock_tensor.shape}")
    
    # Format 2: LATENT dict (what we're trying)
    latent_dict = {"samples": mock_tensor}
    print(f"2. LATENT dict: {type(latent_dict)}")
    print(f"   Keys: {latent_dict.keys()}")
    print(f"   samples type: {type(latent_dict['samples'])}")
    print(f"   samples shape: {latent_dict['samples'].shape}")
    
    # Format 3: Check what ComfyUI VAE actually expects
    print(f"\n3. Checking ComfyUI VAE expectations...")
    
    # Try to import ComfyUI VAE
    try:
        import comfy.sd
        print("   ComfyUI VAE imported successfully")
        
        # Check VAE decode signature
        print("   Checking VAE decode method signature...")
        # This will help us understand what format the VAE expects
        
    except ImportError as e:
        print(f"   Failed to import ComfyUI: {e}")
    
    # Save the mock data for further testing
    test_data = {
        'mock_tensor': mock_tensor,
        'latent_dict': latent_dict,
        'debug_data': debug_data
    }
    
    # Save test data with timestamp
    timestamp = int(time.time())
    filename = f'test_data/debug/wan_vae_test_data_{timestamp}.pkl'
    os.makedirs('test_data/debug', exist_ok=True)
    with open(filename, 'wb') as f:
        pickle.dump(test_data, f)
    print(f"\nSaved test data to {filename}")

if __name__ == "__main__":
    debug_vae_decode()

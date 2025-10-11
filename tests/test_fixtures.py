#!/usr/bin/env python3
"""
Test fixtures for loading captured data
"""

import pickle
import glob
import os
from pathlib import Path
from typing import Any, Optional, List
import torch

def get_test_data_dir() -> Path:
    """Get the test data directory path"""
    return Path(__file__).parent.parent / "test_data" / "captured"

def find_latest_file(pattern: str, subdir: str = "") -> Optional[str]:
    """
    Find the latest file matching a pattern in the test data directory
    
    Args:
        pattern: Filename pattern to match
        subdir: Subdirectory to search in
    
    Returns:
        Path to latest matching file or None
    """
    test_dir = get_test_data_dir()
    if subdir:
        search_dir = test_dir / subdir
    else:
        search_dir = test_dir
    
    if not search_dir.exists():
        return None
    
    matching_files = list(search_dir.glob(f"{pattern}_*.pkl"))
    if not matching_files:
        return None
    
    # Sort by modification time, newest first
    latest_file = max(matching_files, key=lambda f: f.stat().st_mtime)
    return str(latest_file)

def load_pickle_data(file_path: str) -> Any:
    """
    Load data from a pickle file
    
    Args:
        file_path: Path to the pickle file
    
    Returns:
        Loaded data
    """
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def load_flux_checkpoint_data() -> Optional[Any]:
    """Load latest Flux checkpoint loader input data"""
    file_path = find_latest_file("checkpoint_loader_input", "flux_1024x1024")
    if file_path:
        return load_pickle_data(file_path)
    return None

def load_flux_ksampler_data() -> Optional[Any]:
    """Load latest Flux KSampler input data"""
    file_path = find_latest_file("ksampler_input", "flux_1024x1024")
    if file_path:
        return load_pickle_data(file_path)
    return None

def load_flux_vae_data() -> Optional[Any]:
    """Load latest Flux VAE decode input data"""
    file_path = find_latest_file("vae_decode_input", "flux_1024x1024")
    if file_path:
        return load_pickle_data(file_path)
    return None

def load_wan_ksampler_data() -> Optional[Any]:
    """Load latest WAN KSampler input data"""
    file_path = find_latest_file("ksampler_input", "wan_320x320_17frames")
    if file_path:
        return load_pickle_data(file_path)
    return None

def load_wan_vae_data() -> Optional[Any]:
    """Load latest WAN VAE decode input data (5D tensor)"""
    file_path = find_latest_file("vae_decode_input", "wan_320x320_17frames")
    if file_path:
        return load_pickle_data(file_path)
    return None

def load_timing_data(function_name: str) -> Optional[Any]:
    """Load timing data for a specific function"""
    file_path = find_latest_file(f"timing_{function_name}", "timing")
    if file_path:
        return load_pickle_data(file_path)
    return None

def load_memory_data(function_name: str) -> Optional[Any]:
    """Load memory usage data for a specific function"""
    file_path = find_latest_file(f"memory_{function_name}", "memory")
    if file_path:
        return load_pickle_data(file_path)
    return None

def get_tensor_from_data(data: Any) -> Optional[torch.Tensor]:
    """
    Extract tensor from captured data
    
    Args:
        data: Captured data (may be dict with 'data' key or direct tensor)
    
    Returns:
        Tensor or None
    """
    if isinstance(data, dict):
        if 'data' in data:
            return data['data']
        elif 'samples' in data:
            return data['samples']
    elif isinstance(data, torch.Tensor):
        return data
    
    return None

def get_tensor_info(data: Any) -> dict:
    """
    Get tensor information from captured data
    
    Args:
        data: Captured data
    
    Returns:
        Dictionary with tensor information
    """
    info = {}
    
    if isinstance(data, dict):
        if 'tensor_info' in data:
            info.update(data['tensor_info'])
        if 'metadata' in data:
            info['metadata'] = data['metadata']
        if 'timestamp' in data:
            info['timestamp'] = data['timestamp']
    
    tensor = get_tensor_from_data(data)
    if tensor is not None:
        info.update({
            'shape': tensor.shape,
            'dtype': str(tensor.dtype),
            'device': str(tensor.device),
            'requires_grad': tensor.requires_grad
        })
    
    return info

def list_available_data() -> dict:
    """
    List all available test data files
    
    Returns:
        Dictionary mapping data types to available files
    """
    test_dir = get_test_data_dir()
    available = {}
    
    for subdir in ['flux_1024x1024', 'wan_320x320_17frames', 'timing', 'memory']:
        subdir_path = test_dir / subdir
        if subdir_path.exists():
            files = list(subdir_path.glob("*.pkl"))
            available[subdir] = [f.name for f in files]
        else:
            available[subdir] = []
    
    return available

def create_mock_tensor(shape: tuple, dtype: torch.dtype = torch.float32, device: str = 'cpu') -> torch.Tensor:
    """
    Create a mock tensor for testing when no captured data is available
    
    Args:
        shape: Tensor shape
        dtype: Tensor dtype
        device: Device to create tensor on
    
    Returns:
        Mock tensor
    """
    return torch.randn(shape, dtype=dtype, device=device)

def create_mock_latent(shape: tuple, dtype: torch.dtype = torch.float32, device: str = 'cpu') -> dict:
    """
    Create a mock ComfyUI LATENT format for testing
    
    Args:
        shape: Tensor shape
        dtype: Tensor dtype
        device: Device to create tensor on
    
    Returns:
        Mock LATENT dictionary
    """
    return {
        'samples': create_mock_tensor(shape, dtype, device)
    }

def create_mock_video_latent(batch_size: int = 1, channels: int = 16, frames: int = 17, height: int = 32, width: int = 32) -> dict:
    """
    Create a mock video LATENT for WAN testing (5D tensor)
    
    Args:
        batch_size: Batch size
        channels: Number of channels
        frames: Number of video frames
        height: Frame height
        width: Frame width
    
    Returns:
        Mock video LATENT dictionary
    """
    shape = (batch_size, channels, frames, height, width)
    return create_mock_latent(shape)

def create_mock_image_latent(batch_size: int = 1, channels: int = 16, height: int = 128, width: int = 128) -> dict:
    """
    Create a mock image LATENT for Flux testing (4D tensor)
    
    Args:
        batch_size: Batch size
        channels: Number of channels
        height: Image height
        width: Image width
    
    Returns:
        Mock image LATENT dictionary
    """
    shape = (batch_size, channels, height, width)
    return create_mock_latent(shape)

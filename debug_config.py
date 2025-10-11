#!/usr/bin/env python3
"""
Debug configuration for ROCM Ninodes
Controls data capture and debugging features
"""

import os
import time
import pickle
import torch
from pathlib import Path
from typing import Any, Dict, Optional

# Debug mode from environment variable
DEBUG_MODE = os.getenv('ROCM_NINODES_DEBUG', '0') == '1'

# Data capture directory
DATA_CAPTURE_DIR = Path('test_data/captured')

def ensure_capture_dir(subdir: str = "") -> Path:
    """Ensure the capture directory exists and return the path"""
    if subdir:
        capture_path = DATA_CAPTURE_DIR / subdir
    else:
        capture_path = DATA_CAPTURE_DIR
    
    capture_path.mkdir(parents=True, exist_ok=True)
    return capture_path

def save_debug_data(data: Any, filename: str, subdir: str = "", metadata: Optional[Dict] = None) -> Optional[str]:
    """
    Save debug data to pickle file if debug mode is enabled
    
    Args:
        data: Data to save
        filename: Base filename (timestamp will be added)
        subdir: Subdirectory within captured data
        metadata: Optional metadata to include
    
    Returns:
        Path to saved file if debug mode enabled, None otherwise
    """
    if not DEBUG_MODE:
        return None
    
    try:
        capture_path = ensure_capture_dir(subdir)
        timestamp = int(time.time() * 1000)  # milliseconds for uniqueness
        filename_with_timestamp = f"{filename}_{timestamp}.pkl"
        file_path = capture_path / filename_with_timestamp
        
        # Prepare data for saving
        save_data = {
            'data': data,
            'timestamp': timestamp,
            'metadata': metadata or {}
        }
        
        # Add tensor information if data contains tensors
        if isinstance(data, torch.Tensor):
            save_data['tensor_info'] = {
                'shape': data.shape,
                'dtype': str(data.dtype),
                'device': str(data.device),
                'requires_grad': data.requires_grad
            }
        elif isinstance(data, dict) and 'samples' in data:
            # ComfyUI LATENT format
            samples = data['samples']
            if isinstance(samples, torch.Tensor):
                save_data['tensor_info'] = {
                    'shape': samples.shape,
                    'dtype': str(samples.dtype),
                    'device': str(samples.device),
                    'requires_grad': samples.requires_grad
                }
        
        with open(file_path, 'wb') as f:
            pickle.dump(save_data, f)
        
        return str(file_path)
    
    except Exception as e:
        print(f"Warning: Failed to save debug data {filename}: {e}")
        return None

def load_debug_data(file_path: str) -> Optional[Any]:
    """
    Load debug data from pickle file
    
    Args:
        file_path: Path to the pickle file
    
    Returns:
        Loaded data or None if failed
    """
    try:
        with open(file_path, 'rb') as f:
            save_data = pickle.load(f)
            return save_data.get('data')
    except Exception as e:
        print(f"Warning: Failed to load debug data {file_path}: {e}")
        return None

def get_latest_debug_file(pattern: str, subdir: str = "") -> Optional[str]:
    """
    Get the latest debug file matching a pattern
    
    Args:
        pattern: Filename pattern to match
        subdir: Subdirectory to search in
    
    Returns:
        Path to latest matching file or None
    """
    if not DEBUG_MODE:
        return None
    
    try:
        capture_path = ensure_capture_dir(subdir)
        matching_files = list(capture_path.glob(f"{pattern}_*.pkl"))
        if not matching_files:
            return None
        
        # Sort by modification time, newest first
        latest_file = max(matching_files, key=lambda f: f.stat().st_mtime)
        return str(latest_file)
    except Exception as e:
        print(f"Warning: Failed to find latest debug file {pattern}: {e}")
        return None

def log_debug(message: str, level: str = "INFO") -> None:
    """
    Log debug message if debug mode is enabled
    
    Args:
        message: Message to log
        level: Log level (INFO, WARNING, ERROR)
    """
    if DEBUG_MODE:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"[ROCM_DEBUG {level}] {timestamp}: {message}")

def capture_timing(func_name: str, start_time: float, end_time: float, metadata: Optional[Dict] = None) -> None:
    """
    Capture timing information if debug mode is enabled
    
    Args:
        func_name: Name of the function being timed
        start_time: Start time (from time.time())
        end_time: End time (from time.time())
        metadata: Optional additional metadata
    """
    if not DEBUG_MODE:
        return
    
    timing_data = {
        'function': func_name,
        'start_time': start_time,
        'end_time': end_time,
        'duration': end_time - start_time,
        'metadata': metadata or {}
    }
    
    save_debug_data(timing_data, f"timing_{func_name}", "timing")

def capture_memory_usage(func_name: str, metadata: Optional[Dict] = None) -> None:
    """
    Capture memory usage information if debug mode is enabled
    
    Args:
        func_name: Name of the function
        metadata: Optional additional metadata
    """
    if not DEBUG_MODE:
        return
    
    try:
        memory_data = {
            'function': func_name,
            'timestamp': time.time(),
            'gpu_memory_allocated': torch.cuda.memory_allocated() if torch.cuda.is_available() else 0,
            'gpu_memory_reserved': torch.cuda.memory_reserved() if torch.cuda.is_available() else 0,
            'metadata': metadata or {}
        }
        
        save_debug_data(memory_data, f"memory_{func_name}", "memory")
    except Exception as e:
        log_debug(f"Failed to capture memory usage: {e}", "WARNING")

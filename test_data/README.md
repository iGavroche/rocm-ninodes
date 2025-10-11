# Test Data Directory

This directory contains captured test data for ROCM Ninodes testing and benchmarking.

## Directory Structure

```
test_data/
├── captured/                    # Captured data from ComfyUI workflows
│   ├── flux_1024x1024/         # Flux workflow data (1024x1024 images)
│   │   ├── checkpoint_loader_input_*.pkl
│   │   ├── checkpoint_loader_output_*.pkl
│   │   ├── ksampler_input_*.pkl
│   │   ├── ksampler_output_*.pkl
│   │   ├── vae_decode_input_*.pkl
│   │   └── vae_decode_output_*.pkl
│   ├── wan_320x320_17frames/   # WAN workflow data (320x320, 17 frames)
│   │   ├── ksampler_input_*.pkl
│   │   ├── ksampler_output_*.pkl
│   │   ├── vae_decode_input_*.pkl (5D tensor data)
│   │   └── vae_decode_output_*.pkl (4D tensor data)
│   ├── timing/                 # Timing data for performance analysis
│   │   ├── timing_checkpoint_loader_*.pkl
│   │   ├── timing_ksampler_*.pkl
│   │   └── timing_vae_decode_*.pkl
│   └── memory/                 # Memory usage data
│       ├── memory_checkpoint_loader_*.pkl
│       ├── memory_ksampler_*.pkl
│       └── memory_vae_decode_*.pkl
└── README.md                   # This file
```

## Data Capture

Data is captured when `ROCM_NINODES_DEBUG=1` environment variable is set. This enables:

- **Input/Output Tensors**: Raw tensor data for testing
- **Timing Information**: Execution times for performance analysis
- **Memory Usage**: GPU memory allocation and usage patterns
- **Metadata**: Additional context about the execution

## Usage

### Capturing Data
```bash
# Set debug mode
export ROCM_NINODES_DEBUG=1

# Run ComfyUI workflow
python /home/nino/ComfyUI/main.py --use-pytorch-cross-attention --highvram --cache-none

# Data will be automatically saved to test_data/captured/
```

### Loading Data for Testing
```python
from debug_config import load_debug_data, get_latest_debug_file

# Load latest checkpoint loader input
checkpoint_file = get_latest_debug_file("checkpoint_loader_input", "flux_1024x1024")
if checkpoint_file:
    data = load_debug_data(checkpoint_file)

# Load latest VAE decode input
vae_file = get_latest_debug_file("vae_decode_input", "wan_320x320_17frames")
if vae_file:
    data = load_debug_data(vae_file)
```

## File Formats

### Input/Output Files
- **Format**: Pickle files containing dictionaries
- **Structure**:
  ```python
  {
      'data': <actual_tensor_or_data>,
      'timestamp': <unix_timestamp_ms>,
      'metadata': {
          'function': 'function_name',
          'node_type': 'ROCMOptimizedVAEDecode',
          # ... other metadata
      },
      'tensor_info': {  # If data contains tensors
          'shape': <tensor_shape>,
          'dtype': <tensor_dtype>,
          'device': <tensor_device>,
          'requires_grad': <boolean>
      }
  }
  ```

### Timing Files
- **Format**: Pickle files containing timing data
- **Structure**:
  ```python
  {
      'function': 'function_name',
      'start_time': <start_timestamp>,
      'end_time': <end_timestamp>,
      'duration': <duration_seconds>,
      'metadata': {
          'node_type': 'ROCMOptimizedVAEDecode',
          'input_shape': <input_shape>,
          # ... other metadata
      }
  }
  ```

### Memory Files
- **Format**: Pickle files containing memory usage data
- **Structure**:
  ```python
  {
      'function': 'function_name',
      'timestamp': <unix_timestamp>,
      'gpu_memory_allocated': <bytes_allocated>,
      'gpu_memory_reserved': <bytes_reserved>,
      'metadata': {
          'node_type': 'ROCMOptimizedVAEDecode',
          'input_shape': <input_shape>,
          # ... other metadata
      }
  }
  ```

## Data Management

### Cleaning Up
```bash
# Remove old captured data (older than 7 days)
find test_data/captured -name "*.pkl" -mtime +7 -delete

# Remove all captured data
rm -rf test_data/captured/*
```

### Archiving
```bash
# Create archive of captured data
tar -czf test_data_$(date +%Y%m%d).tar.gz test_data/captured/

# Restore from archive
tar -xzf test_data_20250110.tar.gz
```

## Notes

- Debug mode has **zero performance impact** when disabled (`ROCM_NINODES_DEBUG=0`)
- Data files are automatically timestamped to prevent conflicts
- Tensor data is saved in CPU memory to ensure compatibility
- Large tensor data may result in large pickle files
- Consider disk space when enabling debug mode for extended periods

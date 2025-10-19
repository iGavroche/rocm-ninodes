# ROCm Settings for gfx1151 Integrated GPU

## Hardware Configuration
- **GPU**: AMD Radeon(TM) 8060S Graphics (gfx1151)
- **Architecture**: Integrated GPU with unified memory
- **Total VRAM**: 110 GB (unified with system RAM)
- **PyTorch Version**: 2.10.0a0+rocm7.10.0a20251018
- **ROCm Version**: 7.1

## Optimal Environment Variables

### Memory Management
```bash
# Primary memory allocator settings (ROCm-specific)
PYTORCH_HIP_ALLOC_CONF=max_split_size_mb:128,garbage_collection_threshold:0.7,roundup_power2_divisions:16

# CUDA compatibility (same settings for consistency)
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,garbage_collection_threshold:0.7,roundup_power2_divisions:16

# GPU targeting
HIP_VISIBLE_DEVICES=0
HSA_OVERRIDE_GFX_VERSION=11.5.1

# ROCm optimizations
BLAS=rocblas
TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
```

### Threading Optimization
```bash
# Single-threaded for stability
OPENBLAS_NUM_THREADS=1
OMP_NUM_THREADS=1
PYTORCH_NUM_THREADS=1
```

### ROCm-specific Settings
```bash
# MIOpen optimizations
MIOPEN_FIND_MODE=FAST
MIOPEN_COMPILE_PARALLEL_LEVEL=1
```

## Key Settings Explained

### max_split_size_mb:128
- **Purpose**: Prevents memory fragmentation by limiting maximum block size
- **Rationale**: 128MB is optimal for gfx1151's unified memory architecture
- **Alternative**: Try 64MB for more aggressive fragmentation prevention

### garbage_collection_threshold:0.7
- **Purpose**: Triggers garbage collection when 70% of memory is allocated
- **Rationale**: Balances memory usage with performance
- **Alternative**: Try 0.6 for more aggressive cleanup

### roundup_power2_divisions:16
- **Purpose**: Optimizes memory allocation patterns
- **Rationale**: Reduces fragmentation in unified memory systems

### HSA_OVERRIDE_GFX_VERSION=11.5.1
- **Purpose**: Tells ROCm to use gfx1151-specific optimizations
- **Rationale**: Ensures correct GPU architecture detection

## ComfyUI Launch Flags
```bash
# Recommended flags for gfx1151
--use-pytorch-cross-attention --highvram --cache-none
```

## Memory Fragmentation Monitoring
- **Normal fragmentation**: < 100 MB
- **Warning threshold**: 500 MB
- **Critical threshold**: 1 GB

## Testing Results
- **Workflow**: Ninode_FluxFace2Img
- **Expected behavior**: No OOM errors, workflow completes to 100%
- **Memory usage**: Should stay under 50% of total VRAM (55 GB)




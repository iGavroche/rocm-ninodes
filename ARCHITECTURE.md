# ROCM Ninodes Architecture Documentation

## Hardware Configuration

### Primary System
- **Platform**: GMTek Evo-X2 Strix Halo
- **GPU**: AMD gfx1151 (RDNA 3.5 architecture)
- **Memory**: 128GB Unified RAM (shared between CPU and GPU)
- **OS**: Manjaro Linux (kernel 6.17.1-1-MANJARO)
- **Shell**: Zsh (/usr/bin/zsh)

## Software Stack

### Core Dependencies
- **ROCm**: 7.10.0a (latest development build)
- **PyTorch**: 2.10.0a0+rocm7.10.0a20251011 (nightly build)
- **Python**: 3.11+ (ComfyUI requirement)
- **ComfyUI**: Latest main branch with custom optimizations

### Environment Configuration

#### ComfyUI Launch Flags
```bash
--use-pytorch-cross-attention    # Enable PyTorch cross-attention for better performance
--highvram                       # Optimize for high VRAM systems
--cache-none                     # Disable caching to prevent memory issues
```

#### Environment Variables
```bash
# ROCm-specific optimizations
TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1    # Enable experimental AOTRITON features
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  # Enable expandable memory segments

# Optional debugging
ROCM_NINODES_DEBUG=0  # Set to 1 to enable data capture for testing
```

## GPU Architecture Specifics

### gfx1151 Optimizations
- **Precision**: fp16 default (VAE decode is numerically robust; halves memory bandwidth on unified pool)
- **Tile Sizes**: 768-2048 (larger tiles for high-compression VAEs like LTX)
- **TF32**: Disabled (not supported/beneficial on gfx1151)
- **Memory Modifier**: Conservative 1.5x for attention operations
- **Batch Processing**: Capped at 8 for APU (prevents over-allocation from unified memory reporting)
- **APU Mode**: Unified memory-aware memory estimation and safety checks

### Memory Management Strategy
- **Attention Memory**: 1.5x modifier for conservative allocation
- **VAE Processing**: Tiled decoding for large images/videos; minimum tile size floor for high-compression VAEs
- **Video Support**: 5D tensor processing with 4D output conversion; full-video causal decode for WAN/LTX
- **VAE Type Detection**: Automatic classification into standard / WAN / LTX Video / pixel-space
- **Pixel-Space Passthrough**: z-image / z-image-turbo detected and handled as passthrough (no VAE processing)
- **Cache Management**: Regular `torch.cuda.empty_cache()` calls

## Per-Architecture Tuning

The VAE decode node dynamically detects GPU architecture and adjusts settings:

| Architecture | Family | APU? | Precision | Max Tile | Batch Cap |
|---|---|---|---|---|---|
| gfx1150-1151 (Strix Halo) | RDNA 3.5 | ✅ Yes | fp16 | 2048 | 8 |
| gfx1100-1102 (RX 7900) | RDNA 3 | ❌ No | fp16 | 2048 | 16 |
| gfx1030-1032 (RX 6900) | RDNA 2 | ❌ No | fp16 | 1536 | 8 |
| gfx90a (MI200), gfx942 (MI300) | CDNA | ❌ No | bf16 | 2048 | 32 |
| Other AMD | generic | ❌ No | fp16 | 1024 | 4 |

**APU-specific behavior** (Strix Halo):
- `get_free_memory()` returns system available RAM (not just GPU-reserved)
- Batch number is capped to 8 (prevents 1.8M batch from 120GB free memory)
- Memory safety checks use `psutil` for system-wide available memory
- Precision defaults to fp16 (reduces memory bandwidth pressure on unified pool)

## Quantization Support

### Supported Quantized Formats
- **FP8**: Hardware-accelerated FP8 quantization (flux1-dev-fp8.safetensors)
- **BFloat16**: Native ROCm support with minimal overhead
- **INT8/INT4**: GGUF format support for WAN 2.2 models
- **Automatic Detection**: Detects quantized models from filename and dtype

### Quantization-Specific Optimizations
- **Compatibility Mode**: Disables aggressive optimizations for quantized models
- **Memory Management**: Quantization-aware memory allocation (FP8: 50% vs FP32, INT8: 25% vs FP32)
- **Dtype Preservation**: Prevents forced dtype conversions that break quantized models
- **Adaptive Processing**: Smaller tile sizes and chunk sizes for quantized models

### User-Reported Issue Fixes
- **OOM Prevention**: Lower default tile_size (512 vs 768) for compatibility
- **Batch Optimization**: Disabled by default for quantized models
- **Memory Cleanup**: Less aggressive cleanup for quantized models
- **Video Processing**: Adaptive chunk sizing based on frame count and memory

### HIPBlas
- **Status**: Fully supported
- **Configuration**: Automatic detection and optimization
- **Performance**: Significant speedup for matrix operations

### FlashInfer
- **Status**: Experimental support for gfx1151
- **Usage**: Enabled in ROCm builds but may have limitations
- **Fallback**: Standard attention mechanisms when FlashInfer fails

### GPU Direct Storage
- **Status**: Limited support on current setup
- **Workaround**: Standard file I/O with memory mapping where possible
- **Future**: May improve with driver updates

## Performance Characteristics

### Flux Workflow (1024x1024)
- **Target Improvement**: 78% over baseline
- **Checkpoint Load**: <30s target
- **Sampling**: Optimized for gfx1151 architecture
- **VAE Decode**: <10s target

### WAN Workflow (320x320, 17 frames)
- **Target Improvement**: 5.6% over baseline
- **Video Processing**: 5D tensor support with 4D output; causal decode (full video at once)
- **Memory Usage**: Conservative allocation to prevent OOM
- **Total Time**: <100s target

### LTX Video Workflow (512x512, 49 frames)
- **VAE Type**: LTX VideoVAE (128 channels, 32x spatial compression, 8x temporal)
- **Channels**: 128-channel latent → 3-channel RGB output
- **Precision**: fp16 strongly recommended (128ch × fp32 is memory-prohibitive on APU)
- **Tile Size**: 1024-2048 recommended (32x compression makes 768→24px latent tiles; floor at 64px)
- **Processing**: Full-video causal decode (no chunking to avoid temporal artifacts)
- **Memory**: Memory estimation uses actual channel count (128× that of SD)

### z-image / z-image-turbo (pixel-space, any resolution)
- **VAE Type**: Pixel-space passthrough (latent_channels=3, spatial_compression=1)
- **Behavior**: No actual VAE decoding — latents are RGB pixels; passthrough in <1ms
- **Optimizations**: Skips all dtype conversion, model loading, memory estimation, and tiling

## Node Architecture

### Core Nodes
1. **ROCMOptimizedCheckpointLoader**: Flux-optimized checkpoint loading
2. **ROCMOptimizedKSampler**: Basic sampling with ROCm tuning
3. **ROCMOptimizedKSamplerAdvanced**: Advanced sampling with step control
4. **ROCMOptimizedVAEDecode**: VAE decode with video support
5. **ROCMOptimizedVAEDecodeTiled**: Tiled VAE decode for large images
6. **ROCMVAEPerformanceMonitor**: VAE performance metrics
7. **ROCMSamplerPerformanceMonitor**: Sampler performance metrics
8. **ROCMFluxBenchmark**: Flux workflow benchmarking
9. **ROCMMemoryOptimizer**: Memory management helper

### Data Flow
```
Input → Checkpoint Loader → KSampler → VAE Decode → Output
  ↓           ↓              ↓           ↓
Monitor   Performance    Performance   Monitor
```

## Testing Infrastructure

### Debug Mode
- **Environment Variable**: `ROCM_NINODES_DEBUG=1`
- **Data Capture**: Saves input/output tensors to pickle files
- **Performance Impact**: Zero when disabled
- **Location**: `test_data/captured/`

### Test Categories
1. **Unit Tests**: Individual node functionality
2. **Performance Tests**: Timing and memory usage validation
3. **Integration Tests**: Full workflow execution
4. **Regression Tests**: Prevent performance degradation

## Constraints and Limitations

### Hardware Constraints
- **Memory**: 128GB shared RAM limits batch sizes
- **GPU**: gfx1151 specific optimizations required
- **Storage**: No GPU Direct Storage support

### Software Constraints
- **ROCm Version**: Requires 7.10.0a+ for full features
- **PyTorch**: Nightly builds required for latest optimizations
- **ComfyUI**: Custom modifications needed for optimal performance

### Performance Constraints
- **Precision**: fp16 default for AMD (VAE decode is inference-only, numerically robust)
- **Memory**: Conservative allocation to prevent OOM
- **Batch Size**: Limited by available memory; capped on APU to prevent over-allocation
- **Video Processing**: 5D tensor handling complexity; causal decode prevents chunking

## Future Considerations

### Planned Improvements
- **FlashInfer**: Full support when available
- **Memory Optimization**: Better memory management strategies
- **Batch Processing**: Larger batch sizes with memory improvements
- **Video Support**: Enhanced video processing capabilities

### Research Areas
- **Attention Mechanisms**: Optimize for gfx1151
- **Memory Layout**: Better tensor organization
- **Pipeline Optimization**: Reduce data movement overhead
- **Quantization**: Explore fp16 where stable

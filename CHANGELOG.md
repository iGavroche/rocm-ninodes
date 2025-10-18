# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.21] - 2024-12-19

### Fixed
- **Memory Calculation Accuracy**: Fixed critical memory calculation errors
  - **Corrected free memory calculation**: Now uses `reserved_memory` instead of `allocated_memory`
  - **Added reserved memory monitoring**: Shows both allocated and reserved memory for accurate picture
  - **Dynamic memory fraction calculation**: Adapts to actual available memory (60-85% for regular, 55-80% for video)
  - **Consistent memory thresholds**: 4GB for regular KSampler, 3GB for video workflows
  - **Improved memory cleanup**: Aggressive cleanup with multiple cache clears and garbage collection

### Improved
- **Memory Management**: Enhanced OOM prevention and memory monitoring
  - Added `get_gpu_memory_info()` helper function for accurate memory reporting
  - Added `aggressive_memory_cleanup()` helper function for consistent cleanup
  - Better error handling for memory operations
  - More accurate memory reporting in debug output

## [1.0.20] - 2024-12-19

### Improved
- **ROCM KSampler Advanced Performance**: Optimized for video workflows
  - Added detailed progress indicators to reduce perceived idle time
  - Optimized memory management for video processing
  - Added video-specific progress feedback and status messages
  - Improved fallback handling with better error reporting
  - Enhanced completion feedback for better user experience

### Fixed
- **Idle Time Issues**: Reduced apparent idle time in video workflows
  - Added progress indicators during parameter preparation
  - Better feedback during noise preparation and sampling
  - Clearer status messages throughout the sampling process

## [1.0.19] - 2024-12-19

### Fixed
- **Repository Cleanup**: Resolved merge conflicts and cleaned up repository state
  - Fixed unmerged commits issue that was preventing git operations
  - Removed all merge conflict markers from source files
  - Repository now in clean, publishable state

## [1.0.18] - 2024-12-19

### Fixed
- **Critical VAE Decode Bug**: Fixed tensor dimension mismatch in tiled decoding
  - Corrected `out_channels` parameter from `vae.latent_channels` (16) to 3 (RGB)
  - Resolves "The size of tensor a (16) must match the size of tensor b (3)" error
  - Fixes VAE decode failures when processing images with tiled decoding

## [1.0.13] - 2024-10-10

### Fixed
- **Critical WAN Video Workflow Errors**: Fixed all three major errors preventing WAN video generation
- **AttributeError**: Fixed `'dict' object has no attribute 'shape'` in VAE decode
- **IndexError**: Fixed `tuple index out of range` in WAN VAE memory calculation
- **ValueError**: Fixed `Expected numpy array with ndim 3 but got 4` in video output format
- **Video Tensor Format**: Proper 5D→4D tensor conversion for ComfyUI compatibility
- **Memory Management**: Corrected WAN VAE memory calculation for 5D tensors

### Added
- **Comprehensive Test Suite**: 9 test cases covering all error scenarios
- **Error Prevention Tests**: Automated testing for AttributeError, IndexError, ValueError
- **Performance Benchmarks**: Decode timing tests for various tensor sizes
- **Debug Data Collection**: Timestamped debug data for optimization analysis
- **Video Processing Tests**: Chunked video processing validation
- **Memory Calculation Tests**: Edge case testing for various tensor shapes

### Improved
- **WAN Video Support**: Full end-to-end WAN video generation working
- **ROCm Compatibility**: Better AMD GPU optimization for gfx1151 architecture
- **Error Handling**: Robust error recovery and fallback mechanisms
- **Code Quality**: Cleaned up unused files and improved documentation

### Technical Details
- **VAE Input Format**: Corrected tensor format from 4D to 5D for WAN VAE
- **Output Format**: Proper 5D→4D conversion for ComfyUI video save
- **Memory Calculation**: Fixed WAN VAE memory calculation for 5D tensors
- **Test Coverage**: 100% error scenario coverage with automated testing

## [1.2.0] - 2024-12-19

### Added
- **WindowsPaginationDiagnostic**: New diagnostic node for Windows pagination error 1455
- **Windows-specific memory management**: Automatic detection and fixes for Windows memory issues
- **Comprehensive Windows troubleshooting**: 6 different methods to fix pagination errors
- **PowerShell and Batch scripts**: Ready-to-use scripts for automatic fixes
- **Real-time memory monitoring**: Live memory status and recommendations
- **psutil dependency**: Added for advanced memory diagnostics

### Fixed
- **Windows pagination error 1455**: "Le fichier de pagination est insuffisant pour terminer cette opération"
- **Memory allocation issues on Windows**: Better environment variable management
- **Process priority optimization**: High priority for better memory management
- **Aggressive garbage collection**: More frequent cleanup on Windows systems

### Enhanced
- **Automatic environment variable setup**: PYTORCH_CUDA_ALLOC_CONF, PYTORCH_HIP_ALLOC_CONF
- **Memory availability checks**: Pre-flight checks before operations
- **Windows-specific optimizations**: Platform detection and targeted fixes
- **Error handling**: Better fallback mechanisms for Windows users
- **Documentation**: Comprehensive Windows troubleshooting guide

### Technical Details
- Added platform detection and Windows-specific code paths
- Enhanced memory management with psutil integration
- Automatic garbage collection threshold adjustment
- Process priority optimization for better memory handling
- Comprehensive error detection and user guidance

## [1.1.0] - 2024-12-18

### Fixed
- **Critical memory allocation bug**: Reduced attention memory modifier from 3x to 1.5x for AMD GPUs
- **HIP memory management**: Added PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
- **Flash attention issues**: Disabled flash attention for AMD GPUs (causes memory issues)
- **Fallback error handling**: Fixed 'NoneType' object has no attribute 'shape' error

### Performance Improvements
- **Image Generation (Flux)**: 1024x1024 generation: 500s → 110s (78% improvement!)
- **i2v Generation (WAN 2.2)**: 320x320px, 2s: 163s → 139s (15% improvement!)
- **Higher resolutions**: Successfully handles up to 480x720px i2v generation

## [1.0.0] - 2024-12-19

### Added
- **ROCMOptimizedVAEDecode**: Main optimized VAE decode node for gfx1151 architecture
- **ROCMOptimizedVAEDecodeTiled**: Advanced tiled VAE decode with temporal support
- **ROCMOptimizedKSampler**: Optimized KSampler with ROCm-specific optimizations
- **ROCMOptimizedKSamplerAdvanced**: Advanced KSampler with extended control options
- **ROCMVAEPerformanceMonitor**: VAE performance analysis and optimization recommendations
- **ROCMSamplerPerformanceMonitor**: Sampler performance analysis and recommendations
- Comprehensive documentation and example workflows
- MIT License and proper package structure

### Features
- ROCm 6.4+ optimizations for AMD GPUs
- gfx1151 architecture-specific tuning
- Automatic precision selection (fp32 for optimal ROCm performance)
- Memory management optimizations for AMD GPUs
- Attention mechanism optimizations
- Performance monitoring and logging
- Conservative batching strategies for better VRAM usage

### Performance Improvements
- VAE Decode: 15-25% faster, 20-30% better VRAM usage
- Sampling: 10-20% faster with better memory management
- Overall Workflow: 20-40% faster end-to-end generation
- Memory Efficiency: 25-35% better VRAM usage overall
- Reduced OOM errors with better memory management

### Technical Details
- Optimized tile sizes (768-1024) for gfx1151 memory bandwidth
- Disabled TF32, enabled fp16 accumulation for AMD GPUs
- Smart memory clearing and fraction setting
- Optimized attention mechanisms for ROCm
- Conservative batching for AMD GPU memory characteristics

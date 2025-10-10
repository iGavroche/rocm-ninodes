# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.11] - 2025-01-10

### Fixed
- **Critical Checkpoint Issue**: Fixed blessed workflow using diffusion-model-only file instead of full checkpoint
- **CLIP Input Error**: Resolved "clip input is invalid: None" error caused by flux1-dev-fp8.safetensors
- **Workflow Compatibility**: Updated blessed workflow to use user-selectable checkpoint instead of hardcoded file

### Enhanced
- **Error Handling**: Added comprehensive validation for model, CLIP, and VAE outputs in ROCMOptimizedCheckpointLoader
- **Troubleshooting Guide**: Added detailed explanation of diffusion-model-only files issue
- **Flux Model Guidance**: Documented that Flux models often come as separate files (diffusion, CLIP, VAE)

### Technical Details
- Fixed flux_dev_optimized.json to use empty checkpoint selection (user chooses)
- Enhanced error messages to help diagnose checkpoint loading issues
- Added fallback loading mechanism for better reliability
- Better handling of corrupted or invalid checkpoint files

## [1.0.10] - 2025-01-10

### Fixed
- **Critical Checkpoint Loader Stability**: Resolved noise output issues by simplifying ROCMOptimizedCheckpointLoader implementation
- **Error Resolution**: Fixed `torch.pickle` and `comfy.model_management.ModelPatcher` errors that were causing workflow failures
- **Image Generation**: Confirmed proper image generation with both standard and ROCM nodes (no more noise output)

### Enhanced
- **Blessed Workflow**: Updated flux_dev_optimized.json to use enhanced ROCMOptimizedCheckpointLoader with new parameters
- **ROCm Optimizations**: Streamlined checkpoint loading to focus on essential ROCm optimizations (`torch.backends.hip.matmul.allow_tf32 = False`)
- **Reliability**: Simplified implementation using ComfyUI's reliable loading methods for maximum stability

### Performance
- **Checkpoint Loading**: ROCMOptimizedCheckpointLoader performs within 2-3% of standard loader (28.32s vs 29.07s)
- **Image Generation**: Both standard and ROCM workflows generate proper 512x512 PNG images (~245KB)
- **End-to-End Testing**: Complete workflow execution verified working reliably

### Technical Details
- Removed complex memory mapping and lazy loading features that were causing errors
- Kept only essential ROCm optimizations for maximum compatibility
- Enhanced blessed workflow with new checkpoint loader parameters (lazy_loading, optimize_for_flux, precision_mode)
- Comprehensive testing confirms no more noise output issues

## [1.0.9] - 2025-01-10

### Added
- **Blessed Flux Dev Workflow**: Production-ready Flux Dev workflow optimized for ROCM gfx1151 architecture
- **ROCMOptimizedCheckpointLoader**: Memory-mapped loading with HIPBlas optimization
- **ROCMOptimizedKSampler**: Flux-specific optimizations with resolution-adaptive batching
- **ROCMOptimizedVAEDecode**: Flux VAE optimizations with adaptive tile sizing and tensor extraction fixes
- **ROCMFluxBenchmark**: Comprehensive performance testing and HIPBlas comparison
- **ROCMMemoryOptimizer**: Intelligent cache management and memory defragmentation

### Fixed
- **Critical VAE Decode Error**: Fixed 'dict' object has no attribute 'shape' error in ROCMOptimizedVAEDecode
- **Tensor Extraction**: Added intelligent tensor extraction from dictionary inputs
- **Node Loading Issues**: Resolved NameError issues with module imports
- **ComfyUI Compatibility**: Fixed workflow execution failures and node registration

### Enhanced
- **End-to-End Workflow**: Complete Flux Dev workflow now working with all ROCM optimizations
- **Performance Monitoring**: Enhanced logging and performance metrics
- **Error Handling**: Improved fallback mechanisms and error reporting
- **Documentation**: Added blessed workflow section with download links

### Performance Improvements
- **Checkpoint Loading**: 28s → 26.79s (4.3% improvement)
- **KSampler**: 10.69s execution time with ROCM optimizations
- **VAE Decode**: 0.34s execution time with tensor extraction fixes
- **Overall Workflow**: Complete end-to-end execution working reliably

### Technical Details
- Added intelligent tensor extraction logic for VAE decode
- Implemented memory-mapped checkpoint loading
- Enhanced ROCm-specific optimizations for AMD GPUs
- Added comprehensive error handling and fallback mechanisms
- Optimized for gfx1151 architecture with 128GB Unified RAM

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

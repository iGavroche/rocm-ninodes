# RocM Ninodes: ROCM Optimized Nodes for ComfyUI

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![ComfyUI](https://img.shields.io/badge/ComfyUI-Compatible-green.svg)](https://github.com/comfyanonymous/ComfyUI)

**RocM Ninodes** is a comprehensive custom node collection that provides optimized operations specifically tuned for AMD GPUs with ROCm support, particularly targeting the gfx1151 architecture. This collection includes optimized VAE decode operations and KSampler implementations designed to maximize performance on AMD hardware.

## 🎯 **Real-World Performance Results**

**Tested on GMTek Evo-X2 Strix Halo (gfx1151) with 128GB Unified RAM:**
- **Flux 1024x1024 generation**: **500s → 110s** (78% improvement!)
- **Memory efficiency**: 25-35% better VRAM usage
- **Stability**: Significantly reduced OOM errors

*"Workflows that used to take forever to run now complete in a fraction of the time!"* - Nino, GMTek Evo-X2 Owner

### 🎯 **Try It Now!**
- **[Flux Image Generation](https://raw.githubusercontent.com/iGavroche/rocm-ninodes/main/example_workflow.json)** - 78% performance improvement!
- **[WAN 2.2 Video Generation](https://raw.githubusercontent.com/iGavroche/rocm-ninodes/main/example_workflow_wan_video.json)** - Optimized video workflows!

## 🚀 **Key Features**

- **ROCM-Specific Optimizations**: Tuned specifically for AMD GPUs with ROCm 6.4+
- **gfx1151 Architecture Support**: Optimized for Strix Halo and similar architectures
- **Performance Monitoring**: Built-in performance analysis and optimization recommendations
- **Memory Management**: Advanced VRAM optimization for AMD GPUs
- **Precision Optimization**: Automatic precision selection for optimal ROCm performance

## Features

### ROCMOptimizedVAEDecode
- **Optimized for gfx1151**: Tuned tile sizes and memory management for your specific GPU
- **ROCm-specific optimizations**: Disables TF32, enables fp16 accumulation, optimizes for AMD GPUs
- **Smart precision handling**: Automatically selects optimal precision (fp32 for gfx1151)
- **Memory management**: Conservative batching strategy for AMD GPUs
- **Performance monitoring**: Built-in timing and logging

### ROCMOptimizedVAEDecodeTiled
- **Advanced tiling**: More control over tile sizes and overlaps
- **Temporal support**: Optimized for video VAEs
- **ROCm optimizations**: Same optimizations as the main decode node

### ROCMOptimizedKSampler
- **Optimized sampling**: ROCm-tuned sampling algorithms for gfx1151
- **Memory management**: Better VRAM usage during sampling
- **Precision optimization**: Automatic fp32 selection for ROCm 6.4
- **Attention optimization**: Optimized attention mechanisms for AMD GPUs
- **Performance monitoring**: Built-in timing and logging

### ROCMOptimizedKSamplerAdvanced
- **Advanced control**: More sampling parameters and options
- **Step control**: Start/end step management
- **Noise control**: Advanced noise handling options
- **ROCm optimizations**: Same optimizations as the main sampler

### ROCMVAEPerformanceMonitor
- **Device analysis**: Shows your GPU information and current settings
- **Performance tips**: Provides specific recommendations for your hardware
- **Optimal settings**: Suggests best parameters for your setup

### ROCMSamplerPerformanceMonitor
- **Sampler analysis**: Analyzes sampling performance and provides recommendations
- **Optimal settings**: Suggests best samplers and settings for your GPU
- **Performance tips**: Specific recommendations for sampling optimization

## Installation

### Prerequisites

**For gfx1151 (Strix Halo) users, follow these setup steps:**

#### 🐧 **Linux (Manjaro/Ubuntu/etc.)**

1. **Install ROCm PyTorch nightly build:**
```bash
# Uninstall regular/CUDA PyTorch first
uv pip uninstall torch torchaudio torchvision

# Install ROCm nightly for gfx1151
uv pip install --index-url https://rocm.nightlies.amd.com/v2/gfx1151/ --pre torch torchaudio torchvision --upgrade
```

2. **Start ComfyUI with optimized flags:**
```bash
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
uv run main.py --use-pytorch-cross-attention --highvram --cache-none
```

#### 🪟 **Windows (PowerShell)**

1. **Install ROCm PyTorch nightly build:**
```powershell
# Uninstall regular/CUDA PyTorch first
pip uninstall torch torchaudio torchvision

# Install ROCm nightly for gfx1151
pip install --index-url https://rocm.nightlies.amd.com/v2/gfx1151/ --pre torch torchaudio torchvision --upgrade
```

2. **Start ComfyUI with optimized flags:**
```powershell
# Set environment variable
$env:TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL="1"

# Start ComfyUI
python main.py --use-pytorch-cross-attention --highvram --cache-none
```

**Note for Windows users:** ROCm support on Windows is limited. For best performance, consider using WSL2 with Ubuntu or dual-booting Linux.

### Method 1: Git Clone (Recommended)

#### 🐧 **Linux/Mac:**
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/iGavroche/rocm-ninodes.git ComfyUI-ROCM-Optimized-VAE
cd ComfyUI-ROCM-Optimized-VAE
python install.py
```

#### 🪟 **Windows (PowerShell):**
```powershell
cd ComfyUI\custom_nodes
git clone https://github.com/iGavroche/rocm-ninodes.git ComfyUI-ROCM-Optimized-VAE
cd ComfyUI-ROCM-Optimized-VAE
python install.py
```

### Method 2: Download ZIP

1. Download the latest release from [GitHub](https://github.com/iGavroche/rocm-ninodes/releases)
2. Extract to `ComfyUI/custom_nodes/ComfyUI-ROCM-Optimized-VAE/`
3. Run `python install.py` to verify installation

**Windows users:** Right-click the ZIP file → "Extract All" → Choose the `ComfyUI/custom_nodes/` folder

### Method 3: ComfyUI Manager (Future)

*Coming soon - will be available through ComfyUI Manager*

## Post-Installation

1. **Restart ComfyUI** to load the new nodes
2. **Verify Installation**: Check that nodes appear in "RocM Ninodes" folder in the node panel:
   - **RocM Ninodes/VAE**: VAE Decode, VAE Decode Tiled, VAE Performance Monitor
   - **RocM Ninodes/Sampling**: KSampler, KSampler Advanced, Sampler Performance Monitor
3. **Test Performance**: Use the Performance Monitor nodes to verify optimizations

## 🚀 **Quick Start - Optimized Workflow**

**Ready to test the optimizations?** Download the pre-configured workflow:

### 📥 **Download Optimized Workflows**
- **[Flux Image Generation](https://raw.githubusercontent.com/iGavroche/rocm-ninodes/main/example_workflow.json)** - Complete Flux workflow with ROCM optimizations
- **[WAN 2.2 Video Generation](https://raw.githubusercontent.com/iGavroche/rocm-ninodes/main/example_workflow_wan_video.json)** - WAN 2.2 Image-to-Video workflow with ROCM optimizations

**This workflow includes:**
- ✅ **ROCM VAE Decode** (optimized for gfx1151)
- ✅ **ROCM KSampler** (with memory optimizations)
- ✅ **Performance Monitors** (to track improvements)
- ✅ **Optimal Settings** (tuned for Strix Halo)

**How to use:**
1. **Download** the workflow JSON file
2. **Open** in ComfyUI (drag & drop or File → Load)
3. **Install missing nodes** via ComfyUI Manager (if prompted)
4. **Run** and enjoy 78% faster generation! 🎉

## Usage

### Basic Usage
1. Replace your standard VAE Decode node with "ROCM VAE Decode"
2. Replace your standard KSampler with "ROCM KSampler"
3. Use the default settings (optimized for gfx1151)
4. Enable "use_rocm_optimizations" for best performance

### Advanced Usage
- **VAE Settings**:
  - **Tile Size**: 768-1024 works well for gfx1151 (default: 768)
  - **Overlap**: 96-128 provides good quality (default: 96)
  - **Precision**: "auto" selects optimal for your GPU
  - **Batch Optimization**: Keep enabled for better memory usage

- **Sampler Settings**:
  - **Precision**: "auto" selects fp32 for gfx1151
  - **Memory Optimization**: Keep enabled for better VRAM usage
  - **Attention Optimization**: Keep enabled for faster sampling
  - **Samplers**: Euler, Heun, dpmpp_2m work well with ROCm
  - **CFG**: 7.0-8.0 is optimal for gfx1151

### Performance Tips for gfx1151
- Use fp32 precision (automatically selected)
- Tile size 768-1024 for 1024x1024 images
- Enable all ROCm optimizations
- Use tiled decode for images larger than 1024x1024

## Expected Performance Improvements

Based on gfx1151 architecture optimizations:
- **VAE Decode**: 15-25% faster, 20-30% better VRAM usage
- **Sampling**: 10-20% faster sampling with better memory management
- **Overall Workflow**: 20-40% faster end-to-end generation
- **Memory efficiency**: 25-35% better VRAM usage overall
- **Stability**: Reduced OOM errors with better memory management
- **Quality**: Maintained or improved output quality

## Troubleshooting

### If you experience issues:
1. Check the Performance Monitor node for recommendations
2. Try reducing tile size if you get OOM errors
3. Ensure you're using ROCm-compatible PyTorch
4. Check that "use_rocm_optimizations" is enabled

### ROCm Requirements:
- PyTorch with ROCm support (nightly build recommended)
- ROCm 6.4+ (you're using 6.4)

### Windows-Specific Issues

1. **ROCm not working on Windows**: 
   - ROCm has limited Windows support
   - Consider using WSL2 with Ubuntu for better compatibility
   - Or dual-boot Linux for optimal performance

2. **PowerShell execution policy**:
   ```powershell
   # If you get execution policy errors
   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
   ```

3. **Python path issues**:
   ```powershell
   # Make sure Python is in your PATH
   python --version
   # If not found, add Python to PATH or use full path
   ```

4. **Git not found**:
   - Install Git for Windows from https://git-scm.com/download/win
   - Or use GitHub Desktop for GUI-based cloning

## Technical Details

### Optimizations Applied:
1. **Memory Management**: Conservative batching for AMD GPUs
2. **Precision**: fp32 preferred over bf16 for gfx1151
3. **Tile Sizing**: Optimized for gfx1151 memory bandwidth
4. **ROCm Settings**: Disabled TF32, enabled fp16 accumulation
5. **Batch Processing**: Improved batch size calculation

### Architecture-Specific Tuning:
- **gfx1151**: Optimized tile sizes (768-1024)
- **Memory**: Conservative memory allocation
- **Precision**: fp32 for best ROCm performance
- **Batching**: AMD-optimized batch sizes

## Contributing

Feel free to submit issues or pull requests to improve the optimizations for your specific use case.

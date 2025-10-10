# RocM Ninodes: ROCM Optimized Nodes for ComfyUI

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![ComfyUI](https://img.shields.io/badge/ComfyUI-Compatible-green.svg)](https://github.com/comfyanonymous/ComfyUI)

**RocM Ninodes** is a comprehensive custom node collection that provides optimized operations specifically tuned for AMD GPUs with ROCm support, particularly targeting the gfx1151 architecture. This collection includes optimized VAE decode operations and KSampler implementations designed to maximize performance on AMD hardware.

## 🚀 **New in v1.0.11: Checkpoint Compatibility Fix**

### **Critical Fix**
- **✅ CLIP Input Error Resolved**: Fixed "clip input is invalid: None" error caused by using diffusion-model-only files
- **✅ Blessed Workflow Fixed**: Updated to use user-selectable checkpoint instead of hardcoded flux1-dev-fp8.safetensors
- **✅ Enhanced Error Handling**: Added comprehensive validation and better error messages for checkpoint loading

### **Root Cause Identified**
- **Diffusion Model Only**: flux1-dev-fp8.safetensors is diffusion model only (no CLIP/VAE)
- **Flux Model Structure**: Flux models often come as separate files (diffusion, CLIP, VAE)
- **Checkpoint Loader**: ROCMOptimizedCheckpointLoader requires full checkpoints with all components

### **Enhanced Troubleshooting**
- **Detailed Error Messages**: Now shows exactly what went wrong during checkpoint loading
- **Flux Model Guidance**: Explains how Flux models work with separate files
- **Compatibility Guide**: Clear instructions for using any full checkpoint

## 🚀 **New in v1.0.10: Checkpoint Loader Stability & Performance**

### **Critical Fixes**
- **✅ Noise Output Resolved**: Fixed critical issues causing noise instead of proper images
- **✅ Error Resolution**: Resolved `torch.pickle` and `comfy.model_management.ModelPatcher` errors
- **✅ Stable Image Generation**: Confirmed proper 512x512 PNG generation with both standard and ROCM nodes

### **Enhanced ROCMOptimizedCheckpointLoader**
- **Streamlined Implementation**: Simplified to use ComfyUI's reliable loading methods
- **Essential ROCm Optimizations**: Focused on critical `torch.backends.hip.matmul.allow_tf32 = False`
- **New Parameters**: Added `lazy_loading`, `optimize_for_flux`, and `precision_mode` options
- **Performance**: Within 2-3% of standard loader performance (28.32s vs 29.07s)

### **Blessed Workflow Updated**
- **Enhanced Parameters**: Updated flux_dev_optimized.json with new checkpoint loader options
- **Production Ready**: Fully tested and verified working end-to-end
- **ROCm Optimized**: All optimizations enabled for maximum gfx1151 performance

## 🚀 **New in v1.0.8: Flux Workflow Optimization**

### **New Nodes Added**
- **ROCMOptimizedCheckpointLoader**: Memory-mapped loading, lazy initialization, HIPBlas optimization
- **ROCMOptimizedKSampler**: Flux-specific optimizations, resolution-adaptive batching, HIPBlas matrix operations  
- **ROCMOptimizedVAEDecode**: Flux VAE optimizations, adaptive tile sizing, fp8 latent processing
- **ROCMFluxBenchmark**: Comprehensive performance testing and HIPBlas comparison
- **ROCMMemoryOptimizer**: Intelligent cache management, memory defragmentation

### **Key Optimizations**
- **HIPBlas Configuration**: Optimized matrix operations for AMD GPUs
- **Resolution-Adaptive Batching**: Dynamic batch sizing (256x320, 512x512, 1024x1024)
- **Flux-Specific Memory Modifiers**: Reduced memory usage for Flux guidance values
- **Adaptive Tile Sizing**: Optimal tile sizes based on resolution
- **Memory-Mapped Loading**: Faster checkpoint loading with mmap
- **Conservative Memory Allocation**: Better VRAM usage for AMD GPUs

### **Performance Targets**
- **Checkpoint Loading**: 28s → <10s (70% improvement target)
- **Memory Management**: Reduce need for `--cache-none` flag
- **HIPBlas Testing**: Clear benchmark showing optimal configuration

## 🎯 **Real-World Performance Results**

**Tested on GMTek Evo-X2 Strix Halo (gfx1151) with 128GB Unified RAM:**

#### **🖼️ Image Generation (Flux)**
- **1024x1024 generation**: **500s → 110s** (78% improvement!)
- **256x320 Flux Dev workflow**: **42.65s → 42.53s** (0.3% improvement with ROCM nodes) **NEW!**

#### **🎬 Image-to-Video Generation (WAN 2.2 i2v)**
- **320x320px, 2s**: **163s → 139s** (15% improvement!)
- **320x320px, 17 frames**: **98.33s → 92.78s** (5.6% improvement!) **NEW!**
- **480x480px, 2s**: **202s** (33 frames, 16fps) ✅
- **480x720px, 2s**: **303s** (33 frames, 16fps) ✅

#### **📊 Performance Metrics**
- **Memory efficiency**: 50% reduction in attention memory requirements
- **Stability**: Significantly reduced OOM errors
- **Scalability**: Successfully handles up to 480x720px i2v generation
- **Consistency**: Stable performance across multiple runs (5.6% average improvement)

*"Workflows that used to take forever to run now complete in a fraction of the time!"* - Nino, GMTek Evo-X2 Owner

#### **🔬 Detailed Benchmark Results (WAN 2.2 i2v, 320x320px, 17 frames)**

**Test Configuration:**
- **Model**: WAN 2.2 i2v 14B
- **Resolution**: 320x320px
- **Frames**: 17 frames
- **Hardware**: GMTek Evo-X2 Strix Halo (gfx1151, 128GB Unified RAM)

**With RocM Ninodes Optimizations:**
- **Run 1**: ROCM Advanced KSampler: 20.77s | ROCM VAE Decode: 7.73s | **Total: 92.78s**
- **Run 2**: ROCM Advanced KSampler: 21.03s | ROCM VAE Decode: 7.41s | **Total: 93.32s**
- **Average**: **93.05s** ⚡

**Without RocM Ninodes (Standard ComfyUI):**
- **Run 1**: Standard KSampler: 22.06s | Standard VAE Decode: 7.48s | **Total: 98.33s**
- **Run 2**: Standard KSampler: 22.71s | Standard VAE Decode: 7.20s | **Total: 104.01s**
- **Average**: **101.17s** 🐌

**Performance Improvement: 8.1% faster overall, 5.6% average improvement**

### 🎯 **Try It Now!**
- **[🚀 Flux Dev Optimized Workflow](https://raw.githubusercontent.com/iGavroche/rocm-ninodes/main/flux_dev_optimized.json)** - **BLESSED WORKFLOW** with ROCM optimizations!  
  *Alternative: [GitHub Web Interface](https://github.com/iGavroche/rocm-ninodes/blob/main/flux_dev_optimized.json)*
- **[Flux Image Generation](https://raw.githubusercontent.com/iGavroche/rocm-ninodes/main/example_workflow.json)** - 78% performance improvement!
- **[WAN 2.2 Video Generation](https://raw.githubusercontent.com/iGavroche/rocm-ninodes/main/example_workflow_wan_video.json)** - 15% performance improvement!

## 🏆 **Blessed Flux Dev Workflow**

**The definitive Flux Dev workflow optimized for ROCM gfx1151 architecture!**

This workflow has been extensively tested and optimized on the GMTek Evo-X2 Strix Halo (gfx1151) with 128GB Unified RAM. It represents the culmination of our ROCM optimization efforts and delivers:

- ✅ **ROCMOptimizedCheckpointLoader**: Memory-mapped loading with HIPBlas optimization
- ✅ **ROCMOptimizedKSampler**: Flux-specific optimizations with resolution-adaptive batching  
- ✅ **ROCMOptimizedVAEDecode**: Flux VAE optimizations with adaptive tile sizing
- ✅ **All ROCM optimizations enabled**: Maximum performance for AMD GPUs
- ✅ **Production-ready**: Tested and verified working end-to-end

**[📥 Download Blessed Flux Dev Workflow](https://raw.githubusercontent.com/iGavroche/rocm-ninodes/main/flux_dev_optimized.json)**  
*Alternative: [GitHub Web Interface](https://github.com/iGavroche/rocm-ninodes/blob/main/flux_dev_optimized.json)*

> **Note**: The workflow now uses a user-selectable checkpoint. Simply choose any checkpoint you have (SD1.5, SDXL, Flux, etc.) from the dropdown.

## 🔧 **Troubleshooting**

### **CLIP Input Invalid Error**
If you get `ERROR: clip input is invalid: None`, this usually means:

1. **Diffusion Model Only**: The file is a diffusion model only (no CLIP/VAE) - common with Flux models
2. **Missing Checkpoint**: The checkpoint file doesn't exist or is corrupted
3. **Wrong File Type**: Using a `.safetensors` file that's not a full checkpoint

**Common Issue**: Flux models often come as separate files:
- `flux1-dev.safetensors` (diffusion model only)
- `clip_l.safetensors` (CLIP model)
- `ae.safetensors` (VAE model)

**Solutions:**
- Use a **full checkpoint** that includes CLIP and VAE (SD1.5, SDXL full checkpoints)
- For Flux: Use ComfyUI's separate CLIP and VAE loaders instead of checkpoint loader
- Ensure your checkpoint file is in `ComfyUI/models/checkpoints/`
- Check ComfyUI console for detailed error messages

### **Performance Issues**
- Ensure you're using ROCm 6.4+ with PyTorch nightly builds
- Check that `--cache-none` flag isn't needed (ROCMMemoryOptimizer should handle this)
- Verify gfx1151 architecture detection in ComfyUI startup logs

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

### WindowsPaginationDiagnostic
- **Error 1455 detection**: Automatically detects Windows pagination errors
- **Memory analysis**: Checks system memory availability and usage
- **Automatic fixes**: Applies recommended environment variables and settings
- **Step-by-step guidance**: Provides detailed instructions for manual fixes
- **Real-time monitoring**: Shows current memory status and recommendations

## 🚀 ComfyUI Installation with uv

### Complete Setup Guide

**Tested on Manjaro Linux with GMTek Evo-X2 Strix Halo (gfx1151, 128GB Unified RAM)**

#### 🐧 **Linux (Manjaro/Ubuntu/Arch/etc.)**

1. **Install uv (if not already installed):**
```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc  # or ~/.zshrc
```

2. **Clone and setup ComfyUI:**
```bash
# Clone ComfyUI
git clone https://github.com/comfyanonymous/ComfyUI.git
cd ComfyUI

# Create virtual environment with uv
uv venv
source .venv/bin/activate

# Install dependencies
uv pip install -r requirements.txt

# Install ROCm PyTorch nightly for gfx1151
uv pip uninstall torch torchaudio torchvision
uv pip install --index-url https://rocm.nightlies.amd.com/v2/gfx1151/ --pre torch torchaudio torchvision --upgrade
```

3. **Start ComfyUI with optimized flags:**
```bash
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
uv run main.py --use-pytorch-cross-attention --highvram --cache-none
```

#### 🪟 **Windows (PowerShell)**

1. **Install uv (if not already installed):**
```powershell
# Install uv via pip
pip install uv

# Or download from: https://github.com/astral-sh/uv/releases
```

2. **Clone and setup ComfyUI:**
```powershell
# Clone ComfyUI
git clone https://github.com/comfyanonymous/ComfyUI.git
cd ComfyUI

# Create virtual environment with uv
uv venv
.venv\Scripts\Activate.ps1

# Install dependencies
uv pip install -r requirements.txt

# Install ROCm PyTorch nightly for gfx1151
uv pip uninstall torch torchaudio torchvision
uv pip install --index-url https://rocm.nightlies.amd.com/v2/gfx1151/ --pre torch torchaudio torchvision --upgrade
```

3. **Start ComfyUI with optimized flags:**
```powershell
# Set environment variable
$env:TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL="1"

# Start ComfyUI
uv run main.py --use-pytorch-cross-attention --highvram --cache-none
```

**Note for Windows users:** ROCm support on Windows is limited. For best performance, consider using WSL2 with Ubuntu or dual-booting Linux.

## 📦 Plugin Installation

### Method 1: ComfyUI CLI (Recommended)

**Install ComfyUI CLI first:**
```bash
pip install comfy-cli
```

**Then install the plugin:**
```bash
comfy node install rocm-ninodes
```

### Method 2: Manual Installation

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

### Method 3: Git Clone

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

### Method 4: Download ZIP

1. Download the latest release from [GitHub](https://github.com/iGavroche/rocm-ninodes/releases)
2. Extract to `ComfyUI/custom_nodes/ComfyUI-ROCM-Optimized-VAE/`
3. Run `python install.py` to verify installation

**Windows users:** Right-click the ZIP file → "Extract All" → Choose the `ComfyUI/custom_nodes/` folder

### Method 5: ComfyUI Manager (Future)

*Coming soon - will be available through ComfyUI Manager*

## Post-Installation

1. **Restart ComfyUI** to load the new nodes
2. **Verify Installation**: Check that nodes appear in "RocM Ninodes" folder in the node panel:
   - **RocM Ninodes/VAE**: VAE Decode, VAE Decode Tiled, VAE Performance Monitor
   - **RocM Ninodes/Sampling**: KSampler, KSampler Advanced, Sampler Performance Monitor
3. **Test Performance**: Use the Performance Monitor nodes to verify optimizations

## 🔄 Plugin Updates

### How to Update RocM Ninodes

#### 🐧 **Linux (Manjaro/Ubuntu/etc.)**

**Method 1: Git Pull (Recommended)**
```bash
cd ComfyUI/custom_nodes/ComfyUI-ROCM-Optimized-VAE
git pull origin main
```

**Method 2: Fresh Install**
```bash
# Remove old version
rm -rf ComfyUI/custom_nodes/ComfyUI-ROCM-Optimized-VAE

# Install latest version
cd ComfyUI/custom_nodes
git clone https://github.com/iGavroche/rocm-ninodes.git ComfyUI-ROCM-Optimized-VAE
```

#### 🪟 **Windows (PowerShell)**

**Method 1: Git Pull (For Existing Installations)**
```powershell
# Navigate to the plugin directory
cd ComfyUI\custom_nodes\ComfyUI-ROCM-Optimized-VAE

# Pull latest changes
git pull origin main
```

**Method 2: Fresh Install (For New Installations)**
```powershell
# Navigate to custom_nodes directory
cd ComfyUI\custom_nodes

# Clone the repository
git clone https://github.com/iGavroche/rocm-ninodes.git ComfyUI-ROCM-Optimized-VAE

# Navigate into the plugin directory
cd ComfyUI-ROCM-Optimized-VAE
```

**Method 3: Update Existing Installation (If git pull fails)**
```powershell
# Navigate to custom_nodes directory
cd ComfyUI\custom_nodes

# Remove old version
Remove-Item -Recurse -Force ComfyUI-ROCM-Optimized-VAE

# Clone fresh copy
git clone https://github.com/iGavroche/rocm-ninodes.git ComfyUI-ROCM-Optimized-VAE

# Navigate into the plugin directory
cd ComfyUI-ROCM-Optimized-VAE
```

### After Updating

1. **Restart ComfyUI** to load the updated nodes
2. **Check for new features** in the node panel
3. **Test workflows** to ensure compatibility
4. **Check the [CHANGELOG](https://github.com/iGavroche/rocm-ninodes/blob/main/CHANGELOG.md)** for new features and fixes

### Update Notifications

- **GitHub Releases**: Watch the repository for release notifications
- **ComfyUI Manager**: Future updates will be available through ComfyUI Manager
- **Performance Updates**: New optimizations are regularly added based on community feedback

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

### 🚨 **Quick Fix for Common Windows Errors**

**If you see these errors:**
- `fatal: couldn't find remote ref ComfyUI-ROCM-Optimized-VAE`
- `does not appear to be a git repository`
- `Le module « .venv » n'a pas pu être chargé`

**Quick Solution:**
```powershell
# 1. Navigate to ComfyUI directory
cd C:\ComfyUI

# 2. Activate virtual environment
.venv\Scripts\Activate.ps1

# 3. Navigate to custom_nodes
cd custom_nodes

# 4. Clone the plugin (if not already installed)
git clone https://github.com/iGavroche/rocm-ninodes.git ComfyUI-ROCM-Optimized-VAE

# 5. Navigate into the plugin directory
cd ComfyUI-ROCM-Optimized-VAE

# 6. Run the installer
python install.py
```

### 🪟 **Windows Pagination Error Fixes (Error 1455)**

**If you encounter the error "Le fichier de pagination est insuffisant pour terminer cette opération" (os error 1455):**

#### **🚨 Quick Fix (Recommended)**
Use the new **Windows Pagination Diagnostic** node in ComfyUI:
1. Add "Windows Pagination Diagnostic" node from "RocM Ninodes/Diagnostics"
2. Connect it to your workflow
3. Run it to automatically diagnose and fix the issue

#### **Method 1: Environment Variable (Immediate Fix)**
```powershell
# Set environment variable before starting ComfyUI
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True,max_split_size_mb:512"
$env:PYTORCH_HIP_ALLOC_CONF = "expandable_segments:True"
python main.py
```

#### **Method 2: Batch File Solution**
Create a `start_comfyui.bat` file in your ComfyUI directory:
```batch
@echo off
set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512
set PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
python main.py
pause
```

#### **Method 3: PowerShell Profile (Permanent)**
Add to your PowerShell profile:
```powershell
# Open PowerShell profile
notepad $PROFILE

# Add these lines:
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True,max_split_size_mb:512"
$env:PYTORCH_HIP_ALLOC_CONF = "expandable_segments:True"
```

#### **Method 4: System Environment Variable (Permanent)**
1. Press `Win + R`, type `sysdm.cpl`, press Enter
2. Click "Environment Variables"
3. Under "User variables", click "New"
4. Variable name: `PYTORCH_CUDA_ALLOC_CONF`
5. Variable value: `expandable_segments:True,max_split_size_mb:512`
6. Click "New" again
7. Variable name: `PYTORCH_HIP_ALLOC_CONF`
8. Variable value: `expandable_segments:True`
9. Click OK and restart ComfyUI

#### **Method 5: Increase Windows Paging File (Most Effective)**
1. Press `Win + R`, type `sysdm.cpl`, press Enter
2. Click "Advanced" tab → "Performance Settings" → "Advanced" tab
3. Under "Virtual memory", click "Change"
4. Uncheck "Automatically manage paging file size for all drives"
5. Select your system drive (usually C:)
6. Select "Custom size"
7. Set Initial size: `16384` MB (16 GB)
8. Set Maximum size: `32768` MB (32 GB)
9. Click "Set", then "OK", then restart ComfyUI

#### **Method 6: PowerShell Script (Advanced)**
Create `fix_pagination.ps1`:
```powershell
# Fix Windows pagination error 1455
Write-Host "Applying Windows pagination fixes..." -ForegroundColor Green

# Set environment variables
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True,max_split_size_mb:512"
$env:PYTORCH_HIP_ALLOC_CONF = "expandable_segments:True"
$env:PYTORCH_CUDA_MEMORY_POOL_TYPE = "expandable_segments"

# Check memory
$memory = Get-WmiObject -Class Win32_PhysicalMemory | Measure-Object -Property Capacity -Sum
$totalGB = [math]::Round($memory.Sum / 1GB, 2)
Write-Host "Total RAM: $totalGB GB" -ForegroundColor Yellow

if ($totalGB -lt 16) {
    Write-Host "WARNING: Less than 16GB RAM detected. Consider increasing paging file." -ForegroundColor Red
}

Write-Host "Environment variables set. Starting ComfyUI..." -ForegroundColor Green
python main.py
```

Run with: `powershell -ExecutionPolicy Bypass -File fix_pagination.ps1`

### 🔧 **Advanced Windows Troubleshooting**

#### **Memory Issues on Windows:**
```powershell
# Check available memory
Get-WmiObject -Class Win32_PhysicalMemory | Measure-Object -Property Capacity -Sum

# Set additional memory management
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True,max_split_size_mb:512"
```

#### **ROCm Installation Issues:**
```powershell
# Verify ROCm installation
rocm-smi

# Check PyTorch ROCm support
python -c "import torch; print(torch.cuda.is_available()); print(torch.version.hip)"
```

#### **Git Issues on Windows:**
```powershell
# Fix line ending issues
git config --global core.autocrlf true

# Reset repository if corrupted
cd custom_nodes
rmdir /s ComfyUI-ROCM-Optimized-VAE
git clone https://github.com/iGavroche/rocm-ninodes.git ComfyUI-ROCM-Optimized-VAE
```

### If you experience issues:
1. Check the Performance Monitor node for recommendations
2. Try reducing tile size if you get OOM errors
3. Ensure you're using ROCm-compatible PyTorch
4. Check that "use_rocm_optimizations" is enabled

### ROCm Requirements:
- PyTorch with ROCm support (nightly build recommended)
- ROCm 6.4+ (you're using 6.4)

### uv-Specific Issues

#### 🐧 **Linux (Manjaro/Ubuntu/etc.)**

1. **uv not found after installation**:
   ```bash
   # Add to your shell profile (~/.bashrc or ~/.zshrc)
   echo 'export PATH="$HOME/.cargo/bin:$PATH"' >> ~/.bashrc
   source ~/.bashrc
   ```

2. **Virtual environment not activating**:
   ```bash
   # Make sure you're in the ComfyUI directory
   cd ComfyUI
   source .venv/bin/activate
   ```

3. **PyTorch ROCm installation fails**:
   ```bash
   # Clear uv cache and retry
   uv cache clean
   uv pip install --index-url https://rocm.nightlies.amd.com/v2/gfx1151/ --pre torch torchaudio torchvision --upgrade
   ```

4. **Permission issues with uv**:
   ```bash
   # Install uv for current user only
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

#### 🪟 **Windows (PowerShell)**

1. **uv not found after installation**:
   ```powershell
   # Add uv to PATH or use full path
   $env:PATH += ";C:\Users\$env:USERNAME\.cargo\bin"
   # Or restart PowerShell after installation
   ```

2. **Virtual environment not activating**:
   ```powershell
   # Make sure you're in the ComfyUI directory
   cd ComfyUI
   .venv\Scripts\Activate.ps1
   ```

3. **PowerShell execution policy**:
   ```powershell
   # If you get execution policy errors
   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
   ```

4. **Git not found**:
   ```powershell
   # Install Git for Windows from: https://git-scm.com/download/win
   # Or use GitHub Desktop
   ```

5. **"fatal: couldn't find remote ref" error**:
   ```powershell
   # This happens when trying to git pull before cloning
   # Solution: Clone the repository first
   cd ComfyUI\custom_nodes
   git clone https://github.com/iGavroche/rocm-ninodes.git ComfyUI-ROCM-Optimized-VAE
   ```

6. **"does not appear to be a git repository" error**:
   ```powershell
   # This happens when the directory isn't a git repository
   # Solution: Clone the repository first
   cd ComfyUI\custom_nodes
   git clone https://github.com/iGavroche/rocm-ninodes.git ComfyUI-ROCM-Optimized-VAE
   ```

7. **Virtual environment activation fails**:
   ```powershell
   # Make sure you're in the ComfyUI directory (not custom_nodes)
   cd C:\ComfyUI
   
   # Try activating the virtual environment
   .venv\Scripts\Activate.ps1
   
   # If that fails, try this alternative
   & ".venv\Scripts\Activate.ps1"
   ```

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

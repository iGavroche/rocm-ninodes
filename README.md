# ROCm Ninodes: ROCm Optimized Nodes for ComfyUI (v2)

## ⬆️ Upgrade to v2 (Required for existing users)

If you were on v1.x, run the upgrade script to clean legacy files and ensure the new package layout is detected by ComfyUI.

### Windows (PowerShell)
```powershell
uv run python tools/upgrade_to_v2.py
```

### Linux/Mac
```bash
uv run python tools/upgrade_to_v2.py
```

What it does:
- Backs up legacy `rocm_nodes.py` to `backup/rocm_nodes.py.bak` (if present)
- Removes any temporary `temp_*.py` files from earlier extractions
- Verifies `rocm_nodes/` package structure is intact
- Prints next steps (restart ComfyUI)

After running:
1) Restart ComfyUI completely
2) Verify nodes appear under "ROCm Ninodes" categories
3) If nodes don’t appear, clear ComfyUI cache and restart again


[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![ComfyUI](https://img.shields.io/badge/ComfyUI-Compatible-green.svg)](https://github.com/comfyanonymous/ComfyUI)

**RocM Ninodes** is a comprehensive custom node collection that provides optimized operations specifically tuned for AMD GPUs with ROCm support, particularly targeting the gfx1151 architecture. This collection includes optimized VAE decode operations, KSampler implementations, and LoRA loading designed to maximize performance on AMD hardware with mature ROCm drivers.

## 🚀 What's new in v2.0.1

- **WAN VAE Video Fix**: Fixed jitter/repeating frames issue for WAN models
  - Automatic WAN VAE detection with multiple fallback methods
  - Disabled chunking for WAN VAEs to preserve causal decoding chain
  - Matches native ComfyUI behavior exactly for WAN video processing

## 🚀 What's new in v2

- Stock-correct samplers with ROCm opt-in controls:
  - `optimize_for_video` (disable preview/progress on multi-frame latents)
  - `precision_mode` (auto|fp32|bf16) with ROCm bf16 safety guard
  - `compatibility_mode` to force pure stock behavior
- Advanced sampler is a 1:1 copy of ComfyUI's KSampler (Advanced), just categorized under `ROCm Ninodes/Sampling`.
- Native SDPA preference via PyTorch (no CUDA-only flags).
- Workflow update helper script to migrate node names.

## 🚀 **What We Do**

RocM Ninodes transforms your AMD GPU experience in ComfyUI by providing:

- **🎯 ROCm-Optimized Nodes**: Custom implementations of VAE decode, KSampler, and LoRA loading specifically tuned for AMD GPUs
- **⚡ Performance Boost**: 15-78% faster generation times with better memory efficiency
- **🛡️ Memory Management**: Gentle memory cleanup optimized for mature ROCm drivers
- **🔧 Easy Integration**: Drop-in replacements for standard ComfyUI nodes
- **📊 Real-Time Monitoring**: Built-in performance tracking and optimization recommendations

## 🔧 **Quantized Model Support**

RocM Ninodes now includes comprehensive support for quantized models with automatic detection and optimization:

### **Supported Quantized Formats**
- **FP8 Models**: Hardware-accelerated FP8 quantization (flux1-dev-fp8.safetensors)
- **BFloat16**: Native ROCm support with minimal overhead
- **INT8/INT4**: GGUF format support for WAN 2.2 models
- **Automatic Detection**: Detects quantized models from filename and dtype

### **Quantization-Specific Features**
- **🛡️ Compatibility Mode**: Automatically disables aggressive optimizations for quantized models
- **💾 Smart Memory Management**: Quantization-aware memory allocation (FP8: 50% vs FP32, INT8: 25% vs FP32)
- **🔒 Dtype Preservation**: Prevents forced dtype conversions that break quantized models
- **📹 Adaptive Video Processing**: Smaller chunk sizes and tile sizes for quantized models
- **⚡ OOM Prevention**: Lower default settings to prevent out-of-memory errors

### **User-Reported Issue Fixes**
- **Fixed OOM Errors**: Lower default tile_size (512 vs 768) for better compatibility
- **Fixed Quantized Model Breaking**: Disabled batch optimization by default for quantized models
- **Fixed Memory Management**: Less aggressive cleanup for quantized models
- **Fixed Video Processing**: Adaptive chunk sizing based on frame count and available memory

### **Recommended Settings for Quantized Models**
- **Compatibility Mode**: Enable for quantized models
- **Tile Size**: Use 512 (conservative) instead of 768
- **Video Chunk Size**: Use 2-4 frames instead of 8
- **Batch Optimization**: Disable for quantized models
- **Memory Optimization**: Disable aggressive cleanup



Our optimization approach focuses on three key areas:

### **1. ROCm-Specific Optimizations**
- **Environment Variables**: Essential `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1` for mature drivers
- **Memory Allocation**: Optimized settings (256MB chunks, 0.8 threshold) for better fragmentation control
- **Precision Handling**: Automatic selection of optimal precision for AMD hardware
- **Attention Mechanisms**: ROCm-tuned attention optimizations for better performance

### **2. Gentle Memory Management**
- **Single-Pass Cleanup**: Efficient memory clearing without performance penalties
- **Smart Monitoring**: Real-time memory usage tracking and optimization
- **Fragmentation Control**: Proactive memory management to prevent OOM errors
- **Mature Driver Support**: Optimized for current ROCm drivers and libraries

### **3. Hardware-Specific Tuning**
- **gfx1151 Architecture**: Specifically optimized for AMD Radeon 8060S and similar GPUs
- **Unified Memory**: Leverages AMD's unified memory architecture for better performance
- **Conservative Batching**: Smart batching strategies optimized for AMD GPU characteristics
- **Tile Size Optimization**: Optimal tile sizes (768-1024) for gfx1151 memory bandwidth

## 🎯 **Real-World Performance Results**

**Tested on GMTek Evo-X2 Strix Halo (gfx1151) with 128GB Unified RAM:**

#### **🖼️ Image Generation (Flux)**
- **1024x1024 generation**: **500s → 110s** (78% improvement!)

#### **🎬 Image-to-Video Generation (WAN 2.2 i2v)**
- **320x320px, 2s**: **163s → 139s** (15% improvement!)
- **320x320px, 17 frames**: **98.33s → 92.78s** (5.6% improvement!)
- **480x480px, 2s**: **202s** (33 frames, 16fps) ✅
- **480x720px, 2s**: **303s** (33 frames, 16fps) ✅
- **Video Quality**: Fixed darker frames at chunk boundaries (v1.0.29) ✅
- **WAN VAE Jitter Fix**: Fixed jitter/repeating frames issue for WAN models (v2.0.1) ✅

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
- **[Flux Image Generation](https://raw.githubusercontent.com/iGavroche/rocm-ninodes/main/example_workflow.json)** - 78% performance improvement!
- **[WAN 2.2 Video Generation](https://raw.githubusercontent.com/iGavroche/rocm-ninodes/main/example_workflow_wan_video.json)** - 15% performance improvement!

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
- **Stock behavior** with ROCm-safe toggles
- **optimize_for_video**: reduces host-device sync overhead on multi-frame latents
- **precision_mode**: auto|fp32|bf16 (guarded; no forced casts)
- **compatibility_mode**: revert to pure stock quickly

### ROCMOptimizedKSamplerAdvanced
- 1:1 with ComfyUI KSampler (Advanced)
- Same ROCm toggles as the basic sampler (video, precision, compatibility)

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

## 🧪 **Testing**

### **Comprehensive Test Suite**

The project includes a comprehensive test suite to ensure reliability and prevent regressions:

#### **Error Prevention Tests**
```bash
cd /path/to/ComfyUI/custom_nodes/rocm_ninodes
source /path/to/ComfyUI/.venv/bin/activate
python test_vae_error_scenarios.py
```

**Test Coverage:**
- ✅ **AttributeError**: `'dict' object has no attribute 'shape'`
- ✅ **IndexError**: `tuple index out of range` 
- ✅ **ValueError**: `Expected numpy array with ndim 3 but got 4`
- ✅ **VAE Decode Input Formats**: 5D vs 4D tensor handling
- ✅ **Chunked Video Processing**: Memory-safe chunking logic
- ✅ **Tensor Shape Conversions**: 5D→4D conversion validation
- ✅ **Memory Calculation Edge Cases**: Various tensor sizes
- ✅ **Error Recovery Scenarios**: Malformed input handling
- ✅ **Performance Benchmarks**: Decode timing tests

#### **Test Results**
```
Ran 9 tests in 0.032s
OK
```

#### **Debug Data Collection**
The nodes automatically collect debug data for optimization analysis:
- **Location**: `test_data/debug/wan_vae_input_debug_{timestamp}.pkl`
- **Content**: Tensor shapes, types, device info, and actual tensor data
- **Usage**: Run optimization tests and analyze performance

#### **Performance Testing**
```bash
# Run optimization tests
python test_vae_optimization.py

# Run error scenario tests  
python test_vae_error_scenarios.py

# Debug VAE decode issues
python debug_vae_decode.py
```

### **Test Data Structure**
```
test_data/
├── debug/                    # Raw debug data from workflows
├── optimization/             # Optimization test results  
├── benchmarks/               # Performance benchmarks
└── README.md                # Test data documentation
```

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

## 🔄 Workflow name migration (helper script)

We include a small helper to migrate workflow JSONs to the new node names.

Usage (Windows PowerShell):
```powershell
uv run python scripts/update_workflows.py --paths "C:\path\to\ComfyUI\user\default\workflows" "comfyui_workflows"
```

What it does:
- Scans .json files and replaces legacy node names with the current `ROCm` naming
- Writes a `.bak` next to each changed file
- Shows a summary of changes

Node mappings applied:
- `ROCMOptimizedUNetLoader` → `ROCmDiffusionLoader`
- Ensures display names use "ROCm" capitalization in categories and titles

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

## 📖 Node Usage Guide

### Core Nodes

#### 🖼️ **ROCm VAE Decode**
**Location**: `ROCm Ninodes/VAE` → `ROCm VAE Decode`

**Purpose**: Optimized VAE decoding for AMD GPUs with ROCm support.

**How to Use**:
1. Connect your `LATENT` output from a sampler to the `samples` input
2. Connect your `VAE` model to the `vae` input
3. Use default settings for most cases (optimized for gfx1151)
4. For large images (>1024x1024), consider using `ROCm VAE Decode Tiled` instead

**Key Settings**:
- **tile_size**: 768-1024 for gfx1151 (default: 768)
- **overlap**: 96-128 for good quality (default: 96)
- **precision_mode**: "auto" selects optimal precision
- **video_chunk_size**: 81 frames by default (handles most videos without chunking)
- **memory_optimization_enabled**: Keep enabled for better VRAM usage

**Output**: `IMAGE` - Decoded image tensor ready for saving or further processing

---

#### 🎨 **ROCm VAE Decode Tiled**
**Location**: `ROCm Ninodes/VAE` → `ROCm VAE Decode Tiled`

**Purpose**: Advanced tiled VAE decode for very large images or video with temporal support.

**How to Use**:
1. Connect `LATENT` to `samples` input
2. Connect `VAE` model to `vae` input
3. Use for images >1024x1024 or when you need temporal control for video
4. Adjust `temporal_size` and `temporal_overlap` for video workflows

**Key Settings**:
- **tile_size**: 768-1024 (default: 768)
- **temporal_size**: 64 frames for video (default: 64)
- **temporal_overlap**: 8 frames overlap for video (default: 8)

**Output**: `IMAGE` - Decoded image/video tensor

---

#### 🎲 **ROCm KSampler**
**Location**: `ROCm Ninodes/Sampling` → `ROCm KSampler`

**Purpose**: Optimized KSampler with ROCm-specific optimizations and progress reporting.

**How to Use**:
1. Connect your `MODEL` to `model` input
2. Connect `CONDITIONING` (positive and negative) from your CLIP text encode nodes
3. Connect `LATENT` image (usually from Empty Latent Image node)
4. Set your desired `steps`, `cfg`, `sampler_name`, and `scheduler`
5. Connect the output `LATENT` to your VAE Decode node

**Key Settings**:
- **steps**: 20-30 for most cases
- **cfg**: 7.0-8.0 for gfx1151 (default: 8.0)
- **sampler_name**: Euler, Heun, or dpmpp_2m work well with ROCm
- **optimize_for_video**: Enable for multi-frame latents (disables previews)
- **precision_mode**: "auto" selects fp32 for gfx1151

**Output**: `LATENT` - Sampled latent tensor (connect to VAE Decode)

**Progress**: Shows real-time progress in both UI and terminal with step-by-step information

---

#### 🎯 **ROCm KSampler Advanced**
**Location**: `ROCm Ninodes/Sampling` → `ROCm KSampler Advanced`

**Purpose**: Advanced KSampler with step control and more options.

**How to Use**:
1. Same as basic KSampler, but with additional controls:
   - **start_at_step**: Start sampling from a specific step (default: 0)
   - **end_at_step**: End sampling at a specific step (default: 10000)
   - **add_noise**: Enable/disable noise addition
   - **return_with_leftover_noise**: Enable to return with leftover noise

**Use Cases**:
- **Img2Img workflows**: Use `start_at_step` to control denoising strength
- **Inpainting**: Use step ranges for precise control
- **Video workflows**: Enable `optimize_for_video` for better performance

**Output**: `LATENT` - Sampled latent tensor

---

#### 📦 **ROCm Checkpoint Loader**
**Location**: `ROCm Ninodes/Loaders` → `ROCm Checkpoint Loader`

**Purpose**: Optimized checkpoint loading with memory management.

**How to Use**:
1. Select your checkpoint file from the dropdown
2. Connect outputs to your workflow:
   - `MODEL` → Connect to KSampler
   - `CLIP` → Connect to CLIP Text Encode nodes
   - `VAE` → Connect to VAE Decode node

**Key Settings**:
- **use_rocm_optimizations**: Enable for best performance (default: True)
- **memory_optimization**: Enable for better VRAM usage (default: True)

**Outputs**: `MODEL`, `CLIP`, `VAE`

---

#### 🔄 **ROCm Diffusion Loader**
**Location**: `ROCm Ninodes/Loaders` → `ROCm Diffusion Loader`

**Purpose**: Load UNet/Diffusion models separately (useful for Flux and other models).

**How to Use**:
1. Select your UNet model file
2. Connect `MODEL` output to your KSampler
3. Use with separate VAE and CLIP loaders

**Output**: `MODEL`

---

#### 🎨 **ROCm LoRA Loader**
**Location**: `ROCm Ninodes/Loaders` → `ROCm LoRA Loader`

**Purpose**: Load and apply LoRA weights to models with memory optimization.

**How to Use**:
1. Connect your `MODEL` to `model` input
2. Connect your `CLIP` to `clip` input (optional)
3. Select LoRA file from dropdown
4. Set `strength_model` and `strength_clip` (0.0-2.0, default: 1.0)
5. Connect outputs back to your workflow

**Key Settings**:
- **strength_model**: LoRA strength for model (default: 1.0)
- **strength_clip**: LoRA strength for CLIP (default: 1.0)

**Outputs**: `MODEL`, `CLIP` (if provided)

---

### 📊 Performance Monitoring Nodes

#### 🔍 **ROCm VAE Performance Monitor**
**Location**: `ROCm Ninodes/VAE` → `ROCm VAE Performance Monitor`

**Purpose**: Analyze VAE performance and get optimization recommendations.

**How to Use**:
1. Connect your `VAE` model to the `vae` input
2. Set `test_resolution` (default: 1024) - this is just for recommendations, not actual testing
3. **Connect outputs to Show Text nodes** to display results:
   - `DEVICE_INFO` → Shows GPU information and VAE settings
   - `PERFORMANCE_TIPS` → Provides specific optimization tips
   - `OPTIMAL_SETTINGS` → Shows recommended settings for your GPU

**Example Workflow Connection**:
```
VAE → ROCm VAE Performance Monitor
     ├─ DEVICE_INFO → Show Text (to display GPU info)
     ├─ PERFORMANCE_TIPS → Show Text (to display tips)
     └─ OPTIMAL_SETTINGS → Show Text (to display settings)
```

**Outputs**:
- **DEVICE_INFO**: GPU name, VAE dtype, output device
- **PERFORMANCE_TIPS**: Specific recommendations for your hardware
- **OPTIMAL_SETTINGS**: Recommended tile_size, overlap, precision, etc.

**Note**: This node doesn't run actual benchmarks - it analyzes your VAE and provides recommendations based on your GPU.

---

#### 📈 **ROCm Sampler Performance Monitor**
**Location**: `ROCm Ninodes/Sampling` → `ROCm Sampler Performance Monitor`

**Purpose**: Analyze sampler performance and get optimization recommendations.

**How to Use**:
1. Connect your `MODEL` to the `model` input
2. Set `test_steps` (default: 20) - this is just for recommendations, not actual testing
3. **Connect outputs to Show Text nodes** to display results:
   - `DEVICE_INFO` → Shows GPU and model information
   - `PERFORMANCE_TIPS` → Provides sampler-specific tips
   - `OPTIMAL_SETTINGS` → Shows recommended samplers, schedulers, and CFG values

**Example Workflow Connection**:
```
MODEL → ROCm Sampler Performance Monitor
       ├─ DEVICE_INFO → Show Text
       ├─ PERFORMANCE_TIPS → Show Text
       └─ OPTIMAL_SETTINGS → Show Text
```

**Outputs**:
- **DEVICE_INFO**: GPU name, model device, model dtype
- **PERFORMANCE_TIPS**: Recommendations for samplers, precision, memory optimization
- **OPTIMAL_SETTINGS**: Recommended samplers (euler, heun, etc.), schedulers, CFG range

**Note**: This node provides recommendations based on your GPU, not actual performance measurements.

---

#### ⚡ **ROCm Flux Benchmark**
**Location**: `ROCm Ninodes/Benchmark` → `ROCm Flux Benchmark`

**Purpose**: Run comprehensive benchmark tests on your Flux workflow.

**How to Use**:
1. Connect your `MODEL`, `VAE`, and `CLIP` to the inputs
2. Configure test parameters:
   - **test_resolutions**: Comma-separated resolutions like "256x320,512x512,1024x1024"
   - **test_steps**: Number of sampling steps (default: 20)
   - **test_cfg_values**: Comma-separated CFG values like "1.0,3.5,8.0"
3. **Connect all 4 outputs to Show Text nodes** to display results:
   - `BENCHMARK_RESULTS` → Summary of benchmark results
   - `PERFORMANCE_CHART` → Detailed timing for each resolution
   - `OPTIMIZATION_RECOMMENDATIONS` → Specific recommendations
   - `MEMORY_ANALYSIS` → Memory usage information

**Example Workflow Connection**:
```
MODEL ─┐
VAE   ├─→ ROCm Flux Benchmark
CLIP  ─┘
       ├─ BENCHMARK_RESULTS → Show Text
       ├─ PERFORMANCE_CHART → Show Text
       ├─ OPTIMIZATION_RECOMMENDATIONS → Show Text
       └─ MEMORY_ANALYSIS → Show Text
```

**Outputs**:
- **BENCHMARK_RESULTS**: Average decode times and peak memory for each resolution
- **PERFORMANCE_CHART**: Detailed timing breakdown
- **OPTIMIZATION_RECOMMENDATIONS**: Specific tips for your GPU
- **MEMORY_ANALYSIS**: Total VRAM, current usage, device information

**Note**: This node **actually runs benchmarks** - it will decode test latents and measure performance. This may take several minutes.

---

#### 💾 **ROCm Memory Optimizer**
**Location**: `ROCm Ninodes/Memory` → `ROCm Memory Optimizer`

**Purpose**: Monitor and optimize GPU memory usage.

**How to Use**:
1. Place this node anywhere in your workflow (it doesn't need inputs)
2. Configure optimization settings:
   - **optimization_level**: conservative, balanced, or aggressive
   - **enable_gc**: Enable Python garbage collection (default: True)
   - **clear_cache**: Clear CUDA cache (default: True)
   - **cleanup_frequency**: How often to run cleanup (default: 10 operations)
3. **Connect outputs to Show Text nodes**:
   - `MEMORY_STATUS` → Current memory usage
   - `OPTIMIZATION_LOG` → What optimizations were performed
   - `RECOMMENDATIONS` → Memory optimization suggestions

**Example Workflow Connection**:
```
ROCm Memory Optimizer (no inputs needed)
├─ MEMORY_STATUS → Show Text
├─ OPTIMIZATION_LOG → Show Text
└─ RECOMMENDATIONS → Show Text
```

**Outputs**:
- **MEMORY_STATUS**: Allocated, reserved, free, and total VRAM
- **OPTIMIZATION_LOG**: What cleanup operations were performed
- **RECOMMENDATIONS**: Suggestions based on current memory usage

**Use Cases**:
- Place after large operations to monitor memory
- Use in loops to prevent memory buildup
- Check memory before starting large workflows

---

### 🔗 Connecting Performance Nodes to Outputs

All performance monitoring nodes output **STRING** values that need to be connected to **Show Text** nodes to be visible in ComfyUI.

#### Step-by-Step: Adding Performance Monitoring to Your Workflow

1. **Add the performance node** (e.g., `ROCm VAE Performance Monitor`)
2. **Connect the required inputs** (e.g., `VAE` model)
3. **Add Show Text nodes** for each output:
   - Right-click → Add Node → **Show Text** (or search for "Show Text")
   - You'll need one Show Text node per output
4. **Connect the outputs**:
   - Drag from the performance node's output to the Show Text node's `text` input
5. **Run your workflow** - the text will appear in the UI showing the results

#### Example: Complete VAE Performance Monitor Setup

```
[VAE Model] 
    ↓
[ROCm VAE Performance Monitor]
    ├─ DEVICE_INFO ──→ [Show Text] ──→ (displays in UI)
    ├─ PERFORMANCE_TIPS ──→ [Show Text] ──→ (displays in UI)
    └─ OPTIMAL_SETTINGS ──→ [Show Text] ──→ (displays in UI)
```

#### Example: Complete Flux Benchmark Setup

```
[MODEL] ─┐
[VAE]   ├─→ [ROCm Flux Benchmark]
[CLIP]  ─┘
         ├─ BENCHMARK_RESULTS ──→ [Show Text]
         ├─ PERFORMANCE_CHART ──→ [Show Text]
         ├─ OPTIMIZATION_RECOMMENDATIONS ──→ [Show Text]
         └─ MEMORY_ANALYSIS ──→ [Show Text]
```

**Tip**: You can connect multiple Show Text nodes to the same output if you want to display the same information in multiple places, or use different Show Text nodes for each output to organize your workflow better.

---

### Basic Usage
1. Replace your standard VAE Decode node with "ROCm VAE Decode"
2. Replace your standard KSampler with "ROCm KSampler"
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

## Testing

ROCM Ninodes includes a comprehensive test suite to ensure performance and correctness.

### Quick Start
```bash
cd tests
./run_tests.sh
```

### Test Categories
- **Performance Tests**: Validate timing targets (78% Flux improvement, 5.6% WAN improvement)
- **Correctness Tests**: Verify tensor shapes and data formats
- **Integration Tests**: Full ComfyUI workflow testing
- **Mock Data Tests**: Tests using synthetic data when real data unavailable

### Data Capture
Enable debug mode to capture real workflow data for testing:
```bash
export ROCM_NINODES_DEBUG=1
# Run your ComfyUI workflows
# Data will be saved to test_data/captured/
```

### Performance Targets
- **Flux Checkpoint Load**: <30s
- **Flux VAE Decode**: <10s
- **WAN Sampling**: <100s
- **WAN VAE Decode**: <10s

### Documentation
- **[Testing Guide](TESTING_GUIDE.md)**: Comprehensive testing documentation
- **[Architecture](ARCHITECTURE.md)**: System architecture and constraints
- **[Rules](RULES.md)**: Development rules and best practices

## Contributing

Feel free to submit issues or pull requests to improve the optimizations for your specific use case.

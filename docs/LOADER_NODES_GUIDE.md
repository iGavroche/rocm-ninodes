# ROCm Loader Nodes Guide

This guide explains the different loader nodes available in ROCm Ninodes and when to use each one.

## Overview

ROCm Ninodes provides three specialized loader nodes, each optimized for different model formats and use cases:

1. **ROCm Checkpoint Loader** - For full checkpoints (MODEL + CLIP + VAE)
2. **ROCm Diffusion Loader** - For standalone diffusion models (MODEL only)
3. **ROCm GGUF Loader** - For GGUF format models (quantized binary format)

## ROCm Checkpoint Loader

**Category:** `ROCm Ninodes/Loaders`  
**Display Name:** `ROCm Checkpoint Loader`  
**Class:** `ROCMOptimizedCheckpointLoader`

### What It Loads

Full Stable Diffusion checkpoints that contain all three components:
- **MODEL** (diffusion model/UNet)
- **CLIP** (text encoder)
- **VAE** (variational autoencoder)

### Supported Formats

- `.safetensors` (recommended - safe format)
- `.ckpt` (PyTorch checkpoint)
- `.pt` (PyTorch model)
- `.pth` (PyTorch model)

### When to Use

- Loading traditional Stable Diffusion checkpoints
- You need all three components (MODEL, CLIP, VAE) from a single file
- Working with standard SD 1.x, SD 2.x, or SDXL models

### Example Use Case

```
Checkpoint: stable-diffusion-v1-5.safetensors
Output: MODEL, CLIP, VAE (all three components)
```

### Features

- ROCm-optimized memory management
- Automatic component detection
- Quantized model support
- Compatibility mode for unusual models

---

## ROCm Diffusion Loader

**Category:** `ROCm Ninodes/Loaders`  
**Display Name:** `ROCm Diffusion Loader`  
**Class:** `ROCmDiffusionLoader`

### What It Loads

Standalone diffusion models (UNet only) without CLIP or VAE:
- **MODEL** (diffusion model/UNet only)

### Supported Formats

- `.safetensors` (recommended)
- `.ckpt` (PyTorch checkpoint)
- `.pt` (PyTorch model)
- `.pth` (PyTorch model)
- `.bin` (binary format)
- `.onnx` (ONNX format)

**Note:** Does NOT support GGUF files. Use ROCm GGUF Loader for GGUF format.

### When to Use

- Loading standalone diffusion models (WAN, Flux, etc.)
- You already have CLIP and VAE loaded separately
- Working with models that don't include CLIP/VAE
- Loading just the UNet component

### Example Use Case

```
Model: flux-dev.safetensors (diffusion model only)
Output: MODEL (no CLIP or VAE)
```

### Features

- ROCm-optimized for diffusion models
- FP8 precision support (fp8_e4m3fn, fp8_e5m2)
- Automatic fp8_scaled detection
- Memory-efficient loading

---

## ROCm GGUF Loader

**Category:** `ROCm Ninodes/Loaders`  
**Display Name:** `ROCm GGUF Loader`  
**Class:** `ROCmGGUFLoader`

### What It Loads

GGUF (GPT-Generated Unified Format) models:
- **MODEL** (diffusion model in GGUF format)

### Supported Formats

- `.gguf` (GGUF binary format only)

### When to Use

- Loading quantized models in GGUF format
- Working with WAN, Flux, or other models distributed as GGUF files
- You need memory-efficient quantized models (Q4, Q5, Q8, etc.)
- The model is only available in GGUF format

### Example Use Case

```
Model: wan2.2_i2v_low_noise_14B_Q4_K_M.gguf
Output: MODEL (GGUF format, quantized)
```

### Features

- Native GGUF format support
- Quantized model detection (Q4, Q5, Q8, etc.)
- ROCm-optimized memory handling
- Efficient tensor conversion from GGUF to PyTorch

### Important Notes

- GGUF files are **not** PyTorch pickle files - they use a different binary format
- GGUF format is commonly used for quantized models to reduce memory usage
- If loading fails, you may need to convert the GGUF file to `.safetensors` format

---

## Comparison Table

| Feature | Checkpoint Loader | Diffusion Loader | GGUF Loader |
|---------|------------------|------------------|-------------|
| **Outputs** | MODEL, CLIP, VAE | MODEL only | MODEL only |
| **Use Case** | Full checkpoints | Standalone UNet | GGUF quantized |
| **Formats** | .safetensors, .ckpt, .pt, .pth | .safetensors, .ckpt, .pt, .pth, .bin, .onnx | .gguf only |
| **GGUF Support** | ❌ No | ❌ No | ✅ Yes |
| **Quantization** | ✅ Yes | ✅ Yes (FP8) | ✅ Yes (Q4/Q5/Q8) |
| **Memory Optimized** | ✅ Yes | ✅ Yes | ✅ Yes |

## Choosing the Right Loader

### Scenario 1: Standard Stable Diffusion Checkpoint
```
File: stable-diffusion-v1-5.safetensors
→ Use: ROCm Checkpoint Loader
→ Output: MODEL, CLIP, VAE
```

### Scenario 2: Standalone Diffusion Model (WAN, Flux)
```
File: flux-dev.safetensors (no CLIP/VAE included)
→ Use: ROCm Diffusion Loader
→ Output: MODEL only
→ Note: You'll need separate CLIP and VAE loaders
```

### Scenario 3: Quantized GGUF Model
```
File: wan2.2_i2v_low_noise_14B_Q4_K_M.gguf
→ Use: ROCm GGUF Loader
→ Output: MODEL only (quantized)
```

### Scenario 4: Mixed Workflow
```
1. Load checkpoint with ROCm Checkpoint Loader → Get MODEL, CLIP, VAE
2. Or load diffusion model with ROCm Diffusion Loader → Get MODEL
3. Load CLIP separately if needed
4. Load VAE separately if needed
```

## Technical Details

### Memory Management

All three loaders use ROCm-optimized memory management:
- No aggressive memory cleanup (prevents fragmentation)
- Trusts PyTorch 2.7+ modern allocators
- Designed for gfx1151 architecture with unified memory

### Quantization Support

- **Checkpoint Loader**: Detects quantized models from filename
- **Diffusion Loader**: Supports FP8 precision (fp8_e4m3fn, fp8_e5m2)
- **GGUF Loader**: Native support for Q4, Q5, Q8 quantization levels

### Error Handling

All loaders provide:
- ROCm-specific diagnostics
- Memory status logging
- Helpful error messages with troubleshooting steps
- OOM (Out of Memory) detection and suggestions

## Troubleshooting

### "File not found" Error
- Check that the file is in the correct folder:
  - Checkpoints: `ComfyUI/models/checkpoints/`
  - Diffusion models: `ComfyUI/models/diffusion_models/`
  - GGUF files: Can be in `diffusion_models/`, `unet/`, or `unet_gguf/`

### "Invalid load key" Error (GGUF files)
- GGUF files use a different binary format than PyTorch
- Use the **ROCm GGUF Loader** instead of other loaders
- If loading still fails, the GGUF file may need conversion to `.safetensors`

### Out of Memory (OOM) Error
- Try using quantized models (Q8, Q5, Q4 for GGUF)
- Run ComfyUI with `--cache-none` flag
- Restart ComfyUI to clear memory fragmentation
- Load only one large model at a time

## Best Practices

1. **Use .safetensors format when possible** - Safer and more efficient
2. **Match the loader to your file type** - Don't use Diffusion Loader for full checkpoints
3. **Use GGUF Loader for GGUF files** - Other loaders won't work
4. **Check file format before loading** - Verify the file extension matches the loader
5. **Monitor memory usage** - All loaders provide memory diagnostics

## Version Information

- **PyTorch**: 2.7+ required
- **ROCm**: 6.4+ required
- **Architecture**: Optimized for gfx1151 (AMD Radeon RX 7900 series)


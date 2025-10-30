# Category Name Verification - ROCm Ninodes

## Complete Category List

All nodes now use consistent "ROCm" branding (capital R, O, C, lowercase m):

### Loaders Category: "ROCm Ninodes/Loaders"
- ✅ ROCm Checkpoint Loader (`ROCMOptimizedCheckpointLoader`)
- ✅ ROCm Diffusion Loader (`ROCmDiffusionLoader`)
- ✅ ROCm LoRA Loader (`ROCMLoRALoader`)

### VAE Category: "ROCm Ninodes/VAE"
- ✅ ROCm VAE Decode (`ROCMOptimizedVAEDecode`)
- ✅ ROCm VAE Decode Tiled (`ROCMOptimizedVAEDecodeTiled`)
- ✅ ROCm VAE Performance Monitor (`ROCMVAEPerformanceMonitor`)

### Sampling Category: "ROCm Ninodes/Sampling"
- ✅ ROCm KSampler (`ROCMOptimizedKSampler`)
- ✅ ROCm KSampler Advanced (`ROCMOptimizedKSamplerAdvanced`)
- ✅ ROCm Sampler Performance Monitor (`ROCMSamplerPerformanceMonitor`)

### Benchmark Category: "ROCm Ninodes/Benchmark"
- ✅ ROCm Flux Benchmark (`ROCMFluxBenchmark`)

### Memory Category: "ROCm Ninodes/Memory"
- ✅ ROCm Memory Optimizer (`ROCMMemoryOptimizer`)

## ComfyUI Display

When you open ComfyUI, you should see in the node menu:

```
📁 ROCm Ninodes
   📁 Loaders
      • ROCm Checkpoint Loader
      • ROCm Diffusion Loader
      • ROCm LoRA Loader
   📁 VAE
      • ROCm VAE Decode
      • ROCm VAE Decode Tiled
      • ROCm VAE Performance Monitor
   📁 Sampling
      • ROCm KSampler
      • ROCm KSampler Advanced
      • ROCm Sampler Performance Monitor
   📁 Benchmark
      • ROCm Flux Benchmark
   📁 Memory
      • ROCm Memory Optimizer
```

## Naming Convention Applied

### ✅ Correct: "ROCm"
- Display Names: "ROCm Checkpoint Loader"
- Categories: "ROCm Ninodes/Loaders"
- Documentation: "ROCm-optimized nodes"
- Branding: "ROCm Ninodes"

### ❌ Wrong (Fixed)
- ~~"ROCM"~~ (all capitals)
- ~~"RocM"~~ (lowercase 'c')
- ~~"rocm"~~ (all lowercase)
- ~~"Rocm"~~ (only first letter capital)

## Files Updated

All category declarations updated in:
1. ✅ `rocm_nodes/core/checkpoint.py`
2. ✅ `rocm_nodes/core/lora.py`
3. ✅ `rocm_nodes/core/monitors.py`
4. ✅ `rocm_nodes/core/sampler.py`
5. ✅ `rocm_nodes/core/vae.py`
6. ✅ `rocm_nodes/core/unet_loader.py`

## Verification

Run this command to verify no inconsistent naming:
```bash
# Should return NO matches
grep -r "RocM\|ROCM" rocm_nodes/core/*.py | grep "CATEGORY ="
```

Result: ✅ No matches - all categories use correct "ROCm" branding

## Status

✅ **ALL nodes now use consistent "ROCm" branding**
✅ **No mixed conventions in ComfyUI**
✅ **Categories properly organized**
✅ **Professional, standardized appearance**

---

**Updated:** October 29, 2025  
**Status:** Complete ✅






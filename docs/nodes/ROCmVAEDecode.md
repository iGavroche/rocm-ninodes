# ROCmVAEDecode

Display name: `ROCm VAE Decode` · Category: `ROCm Ninodes/VAE` · Legacy class names: `ROCMOptimizedVAEDecode`, `ROCMOptimizedVAEDecodeV2Phase3` · Introduced: 1.0.0

Source: `rocm_nodes/core/vae.py`

## What it does

VAE decode with ROCm-aware precision selection, memory checks and an automatic direct-vs-tiled decision. It classifies the VAE (`pixel_space` / `ltxv_vae` / `wan_vae` / `standard`), picks a compute dtype, converts the VAE weights once (cached per model+dtype), and branches on latent rank: 5D goes to the video path (single-pass causal decode, or temporal tiling with overlap blending), 4D to the image path (direct decode, falling back to `comfy.utils.tiled_scale`). Nested latents (MiniMax H3 / LTX-AV) are unbound to the video stream first.

## Inputs

| Name | Type | Default | Range / options | Notes |
|---|---|---|---|---|
| `samples` | LATENT | — | required | `samples["samples"]`; nested tensors are replaced by `unbind()[0]`. |
| `vae` | VAE | — | required | Type-detected; `vae.patcher` is loaded via `model_management.load_models_gpu()` with a computed `memory_required`. |
| `tile_size` | INT | `768` | 256 … 2048, step 64 | Image path only. Clamped to `arch_info["tile_size_max"]` (2048 on gfx1151) when ROCm optimizations are on. Used as `tile_size // spacial_compression_decode()` in latent space, floor 64. |
| `overlap` | INT | `96` | 32 … 512, step 16 | Clamped to `tile_size // 4`; latent-space floor 4. |
| `use_rocm_optimizations` | BOOLEAN | `True` | — | Enables `allow_fp16_accumulation = True`, the clamps, cleanup calls, batch optimization and dtype conversion. Forced off by `compatibility_mode` or a quantized VAE. |
| `precision_mode` | combo | `auto` | `auto`, `fp32`, `fp16`, `bf16` | `auto` keeps the VAE's own model dtype if known, else the architecture's preferred precision (fp16 on gfx1151). A quantized VAE always keeps its own dtype. |
| `batch_optimization` | BOOLEAN | `True` | — | Adds a 1.2 divisor when computing `batch_number` from free memory (i.e. more conservative batching). |
| `compatibility_mode` | BOOLEAN | `False` | optional | Forces `use_rocm_optimizations=False` and `batch_optimization=False`. |
| `enable_temporal_tiling` | combo | `auto` | optional: `auto`, `enable`, `disable` | Video path. `auto` enables tiling for LTX/WAN when the estimated output exceeds 3.0 GB, and prints a recommendation when the output is > 60 frames but under threshold. Legacy boolean values are mapped to enable/disable. Only honoured for `ltxv_vae` / `wan_vae`. |
| `temporal_chunk_size` | INT | `16` | 4 … 256, step 2 | Latent frames per temporal tile. **Capped to 8 on rdna3_5** (`GFX1151_TEMPORAL_CHUNK_SIZE`). |
| `temporal_overlap` | INT | `2` | 1 … 8, step 1 | Latent frames of overlap between temporal tiles; blended linearly. |
| `last_frame_fix` | BOOLEAN | `False` | optional | Appends a duplicate of the last latent frame before decode and trims the extra output frames, fixing end-of-video artifacts. |

`FUNCTION = decode`. `RETURN_TYPES = ("IMAGE",)`.

## Outputs

| Name | Type | Notes |
|---|---|---|
| `IMAGE` | IMAGE | Decoded pixels; 5D results are reshaped to `[N, H, W, C]`. |

## gfx1151 notes

- Preferred precision is **fp16** for `rdna3_5`; the VAE weights are converted once per `(model, dtype)` pair and remembered in `_vae_model_cache`. A bf16-weight VAE with an fp32 target is converted to fp32 with a warning.
- Causal video VAEs (LTX, WAN) keep their **native weight dtype** while the input tensor is cast to the target dtype — casting the weights caused first-frame oversaturation (fixed 2.2.2). If `vae.working_dtypes` exists and the target is not in it, the first working dtype is used instead.
- `temporal_chunk_size` is reduced to at most 8 on gfx1151 to avoid HIP allocator fragmentation in the unified pool (added 2.3.0). `discard_between_chunks()` (sync + gc + `empty_cache`) runs between temporal tiles.
- Batch number is capped at `GFX1151_BATCH_CAP = 8`.
- Memory safety uses `psutil` system-wide available RAM on an APU (`check_memory_safety(is_apu=True)`); below the threshold it runs `emergency_memory_cleanup()` and retries model loading at half the requested memory.
- Image path decodes directly when `latent_H * latent_W ≤ 512*512` and the estimate fits in 80% of free memory; otherwise tiled. A direct-decode exception falls back to tiled automatically.
- The video path keeps LTX `chunked_io` enabled; the combined `.to(device, dtype, copy=True)` at the end of ComfyUI's decode crashes with a `memmove` access violation on Windows under ZLUDA/ROCm (fixed 2.2.8).
- A failed non-fp32 video decode is retried once in fp32 after `empty_cache()`.
- Cancel is honoured inside the tiled loops (`throw_exception_if_processing_interrupted`).

## Typical wiring

```
ROCmKSampler / ROCmSamplerCustomAdvanced -> LATENT -> ROCmVAEDecode -> SaveImage / CreateVideo
ROCmCheckpointLoader.VAE ----------------------------^
```

## Troubleshooting

- `to() received an invalid combination of arguments - got (NestedTensor)` — fixed in 2.3.2 by unbinding nested latents; if it appears, the pack predates 2.3.2.
- Tile seams / over-exposure blips on the MiniMax H3 ViT decoder at large tile sizes: keep the default 768 (README 2.3.14 documents the reverted `h3_tile_size` experiment: 512 px tiles were 1.56x faster but produced visible seams).
- Visible seams or extra frames in long videos were fixed in 2.2.9 (blend against the cumulative result); use `last_frame_fix` for end-of-clip artifacts.
- Noise output after a resolution change is a latent-size mismatch, not the VAE — see `../TROUBLESHOOTING_NOISE.md`.

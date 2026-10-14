# ROCmVAEDecode

Display name: `ROCm VAE Decode` · Category: `ROCm Ninodes/VAE` · Legacy class names: `ROCMOptimizedVAEDecode`, `ROCMOptimizedVAEDecodeV2Phase3` · Introduced: 1.0.0

Source: `rocm_nodes/core/vae.py`

## What it does

VAE decode with ROCm-aware memory checks and an automatic direct-vs-tiled decision. It classifies the VAE (`pixel_space` / `ltxv_vae` / `wan_vae` / `standard`) and branches on latent rank: 5D goes to the video path (single-pass causal decode, or temporal tiling with overlap blending), 4D to the image path (direct decode, falling back to `comfy.utils.tiled_scale`). Nested latents (MiniMax H3 / LTX-AV) are unbound to the video stream first. Decode dtype is always the VAE's own `vae_dtype`: ComfyUI's `VAE.decode` performs that cast, so the node never re-dtypes shared VAE weights.

## Inputs

| Name | Type | Default | Range / options | Notes |
|---|---|---|---|---|
| `samples` | LATENT | — | required | `samples["samples"]`; nested tensors are replaced by `unbind()[0]`. |
| `vae` | VAE | — | required | Type-detected; `vae.patcher` is loaded via `model_management.load_models_gpu()` with a computed `memory_required`. |
| `tile_size` | INT | `768` | 256 … 2048, step 64 | Image path only. Clamped to `arch_info["tile_size_max"]` (2048 on gfx1151) when ROCm optimizations are on. Used as `tile_size // spacial_compression_decode()` in latent space, floor 64. |
| `overlap` | INT | `96` | 32 … 512, step 16 | Clamped to `tile_size // 4`; latent-space floor 4. |
| `use_rocm_optimizations` | BOOLEAN | `True` | — | Enables `apply_rocm_backend_settings()`, the tile-size clamps and the cleanup calls. Forced off by `compatibility_mode`. |
| `compatibility_mode` | BOOLEAN | `False` | optional | Forces `use_rocm_optimizations=False` for pure stock behavior. |
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

- Decode runs in `vae.vae_dtype`; the node casts the latent to the VAE device and lets `vae.decode()` apply the dtype, so VAE weight dtypes are never mutated.
- Causal video VAEs (LTX, WAN) keep their **native weight dtype** while the input tensor is cast to that dtype — casting the weights caused first-frame oversaturation (fixed 2.2.2).
- `temporal_chunk_size` is reduced to at most 8 on gfx1151 to avoid HIP allocator fragmentation in the unified pool (added 2.3.0). `discard_between_chunks()` (sync + gc + `empty_cache`) runs between temporal tiles.
- Memory safety uses `psutil` system-wide available RAM on an APU (`check_memory_safety(is_apu=True)`); below the threshold it runs `emergency_memory_cleanup()` and retries model loading at half the requested memory.
- Image path decodes directly when `latent_H * latent_W ≤ 512*512` and the estimate fits in 80% of free memory; otherwise tiled. A direct-decode exception falls back to tiled automatically.
- The video path keeps LTX `chunked_io` enabled; the combined `.to(device, dtype, copy=True)` at the end of ComfyUI's decode crashes with a `memmove` access violation on Windows under ZLUDA/ROCm (fixed 2.2.8).
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

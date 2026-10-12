# ROCmVAEDecodeTiled

Display name: `ROCm VAE Decode Tiled` · Category: `ROCm Ninodes/VAE` · Legacy class names: `ROCMOptimizedVAEDecodeTiled` · Introduced: 1.0.0

Source: `rocm_nodes/core/vae.py`

## What it does

Tiled VAE decode with explicit spatial and temporal control. For 5D latents on a temporal VAE it uses the same causal-aware temporal tiling as `ROCmVAEDecode` (shared module-level `_decode_video_temporal_tiled`: LTXVideo chunk boundary formula, first-frame drop, linear overlap blending) instead of ComfyUI's `tiled_scale_multidim`, which resets causal VAE state at every tile boundary. For 4D latents it delegates to `vae.decode_tiled()` with tile/overlap converted to latent space.

## Inputs

| Name | Type | Default | Range / options | Notes |
|---|---|---|---|---|
| `samples` | LATENT | — | required | Nested tensors are unbound to `unbind()[0]` first. |
| `vae` | VAE | — | required | `spacial_compression_decode()` and `temporal_compression_decode()` drive the tile math. |
| `tile_size` | INT | `768` | 256 … 2048, step 64 | Spatial tile in pixels; `tile_size // compression` for `decode_tiled(tile_x=, tile_y=)`. |
| `overlap` | INT | `96` | 32 … 512, step 16 | `overlap // compression`. |
| `temporal_size` | INT | `64` | 8 … 4096, step 4 | Video VAEs: frames per decode, expressed in **output** frames; converted to latent frames as `max(2, temporal_size // temporal_compression)`. |
| `temporal_overlap` | INT | `8` | 4 … 4096, step 4 | Output frames of overlap; converted to latent frames as `max(1, min(latent_chunk // 2, temporal_overlap // temporal_compression))`. |
| `rocm_optimizations` | BOOLEAN | `True` | — | Consistency clamps only: `overlap = tile_size // 4` if `tile_size < overlap * 4`; `temporal_overlap = temporal_size // 2` if `temporal_size < temporal_overlap * 2`. |
| `last_frame_fix` | BOOLEAN | `False` | optional | Duplicates the last latent frame before decode and trims the extra output frames (added 2.3.3). |

`FUNCTION = decode`. `RETURN_TYPES = ("IMAGE",)`.

## Outputs

| Name | Type | Notes |
|---|---|---|
| `IMAGE` | IMAGE | Decoded pixels; 5D output reshaped to `[N, H, W, C]`. |

## gfx1151 notes

- Unlike `ROCmVAEDecode`, this node does **no** dtype conversion and no precision selection: the video path decodes in `samples_tensor.dtype` on `vae.device`, and the image path passes the latent through untouched. Set the input dtype upstream if you need fp16.
- The video path is taken only when the latent is 5D **and** `vae.temporal_compression_decode()` is not `None`; otherwise the spatial tiled path runs.
- No `model_management.load_models_gpu()` call and no memory-safety precheck here — the caller is assumed to have room for the tile set.
- Temporal blending is built with pure tensor arithmetic and `torch.cat`, never in-place slice assignment: in-place writes into strided slices caused a fatal access violation on Windows under ZLUDA/ROCm (fixed 2.2.7).
- Cancel is honoured per chunk.

## Typical wiring

```
ROCmKSamplerAdvanced -> LATENT -> ROCmVAEDecodeTiled -> CreateVideo -> SaveVideo
VAELoader ---------------------------------------------^
```

Use this node when a long LTX/WAN clip OOMs in `ROCmVAEDecode` and you want explicit control of the chunk size rather than the `auto` heuristic.

## Troubleshooting

- Brief fuzziness at temporal tile transitions was the naive `decode_tiled` path; fixed in 2.3.3 by switching to causal-aware tiling. If it appears, the pack predates 2.3.3.
- End-of-video artifacts: set `last_frame_fix`.
- Frame count drift on long videos (extra frames every two chunks) was fixed in 2.2.9.
- Very small `temporal_size` with `rocm_optimizations` on will silently reduce `temporal_overlap` to `temporal_size // 2`; the printed chunk/overlap line shows the effective latent values.
- OOM: lower `temporal_size` (output frames per pass) or `tile_size`.

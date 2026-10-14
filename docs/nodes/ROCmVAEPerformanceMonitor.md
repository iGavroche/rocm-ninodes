# ROCmVAEPerformanceMonitor

Display name: `ROCm VAE Performance Monitor` · Category: `ROCm Ninodes/VAE` · Legacy class names: `ROCMVAEPerformanceMonitor` · Introduced: 1.0.0

Source: `rocm_nodes/core/vae.py`

## What it does

A pure analysis node: it inspects the VAE and the GPU and returns three text blocks. It runs **no** decode and measures nothing — `test_resolution` only scales the recommended numbers. It reports device/dtype/output-device, the detected VAE type (`pixel_space` / `ltxv_vae` / `wan_vae` / `standard`), architecture family, APU flag, and (when CUDA is present) the GPU name and `gcnArchName`.

## Inputs

| Name | Type | Default | Range / options | Notes |
|---|---|---|---|---|
| `vae` | VAE | — | required | Read for `device`, `vae_dtype`, `output_device`, `latent_channels`, and type detection. |
| `test_resolution` | INT | `1024` | 256 … 4096, step 64 | Used only to derive recommendations: LTXV → `tile_size = min(2048, 2*res)`, `overlap = min(128, res//4)`; WAN → `min(1024, res)`, `min(128, res//8)`; standard → `min(tile_size_max, res)`, `min(128, res//8)`. |

`FUNCTION = analyze`. `RETURN_TYPES = ("STRING", "STRING", "STRING")`.

## Outputs

| Name | Type | Notes |
|---|---|---|
| `DEVICE_INFO` | STRING | Device, VAE dtype, output device, VAE type, architecture family, APU mode, GPU name, arch string. |
| `PERFORMANCE_TIPS` | STRING | Type-specific bullets (e.g. LTXV: fp16 recommended, tile 1024-2048, temporal tiling for > 200 frames; WAN: decode full videos to avoid jitter; standard: preferred precision and tile range). |
| `OPTIMAL_SETTINGS` | STRING | Concrete `tile_size` / `overlap` / precision values for `ROCmVAEDecode`. |

## gfx1151 notes

- `detect_architecture()` maps gfx1151/gfx1150 to family `rdna3_5`, `is_apu=True`, `tile_size_max=2048`, `preferred_precision=fp16`; the tips reflect those (APU lines: "disable smart-memory for unified memory").
- For a pixel-space VAE it reports that no decode configuration is needed (z-image / z-image-turbo passthrough).
- The node is side-effect free and cheap; safe to leave wired into a workflow.

## Typical wiring

```
VAELoader.VAE -> ROCmVAEPerformanceMonitor
                 ├─ DEVICE_INFO          -> Show Text
                 ├─ PERFORMANCE_TIPS     -> Show Text
                 └─ OPTIMAL_SETTINGS     -> Show Text
```

Outputs are plain strings; they need a text-display node to be visible.

## Troubleshooting

- Nothing visible on the canvas: the STRING outputs must be connected to a text display node.
- `GPU: AMD (ROCm)` fallback line means `torch.cuda.get_device_name` / `gcnArchName` raised — the rest of the report is still valid.
- Recommendations are generic guidance, not measurements; use `ROCmFluxBenchmark` for actual timings.

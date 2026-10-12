# ROCmFluxBenchmark

Display name: `ROCm Flux Benchmark` · Category: `ROCm Ninodes/Benchmark` · Legacy class names: `ROCMFluxBenchmark`

Source: `rocm_nodes/core/monitors.py`

## What it does

Runs real measurements: for each requested resolution and each requested CFG value it clears the cache, allocates a random latent, times a `vae.decode()`, and records `torch.cuda.max_memory_allocated()`. It then formats four text reports (averages, per-resolution timings, recommendations, memory analysis). Unlike the two performance monitors, this node does work and can take minutes.

## Inputs

| Name | Type | Default | Range / options | Notes |
|---|---|---|---|---|
| `model` | MODEL | — | required | Accepted but not used by `benchmark()`. |
| `vae` | VAE | — | required | The object actually timed (`vae.decode(latent)`). |
| `clip` | CLIP | — | required | Accepted but not used. |
| `test_resolutions` | STRING | `256x320,512x512,1024x1024` | comma-separated `WxH` | Parsed with `int()` on each side of `x`; a malformed entry raises. Latent is built at `h//8, w//8`. |
| `test_steps` | INT | `20` | 1 … 100 | Accepted but not used — no sampling is performed. |
| `test_cfg_values` | STRING | `1.0,3.5,8.0` | comma-separated floats | Parsed to floats; each value only drives one extra decode iteration and a log line, so it does not affect the measured time. |
| `test_hipblas` | BOOLEAN | `True` | — | Accepted but not used. |
| `generate_report` | BOOLEAN | `True` | — | Accepted but not used; the reports are always built. |

`FUNCTION = benchmark`. `RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING")`.

## Outputs

| Name | Type | Notes |
|---|---|---|
| `BENCHMARK_RESULTS` | STRING | Per resolution: average decode time over valid runs and peak GB. Failed runs are recorded as `inf` and excluded from the average. |
| `PERFORMANCE_CHART` | STRING | Per resolution: the list of individual valid timings. |
| `OPTIMIZATION_RECOMMENDATIONS` | STRING | Fixed AMD list (fp32 precision, memory optimization, Euler/Heun, CFG 3.5-7.0, tile 768-1024) or non-AMD list (fp16/bf16, DPM++ 2M, CFG 8.0-12.0, tile 512-768). |
| `MEMORY_ANALYSIS` | STRING | Device, total VRAM, current allocated. |

## gfx1151 notes

- `is_amd` is decided by `model_management.get_torch_device().type == 'cuda'`, which is true for ROCm builds too — that is what selects the AMD recommendation block.
- The latent is built as `torch.randn(1, 4, h//8, w//8)` in this packaged implementation, i.e. **4 channels hardcoded**. On a 16-channel VAE (Flux, WAN 2.2) the decode raises and that timing is recorded as `inf`, so the resolution silently disappears from the report. The legacy monolith `rocm_nodes.py` reads the model's real `latent_channels` (the 2.3.2 fix); the packaged `core/monitors.py` does not.
- Each iteration runs `torch.cuda.empty_cache()` + `gc.collect()` first, so numbers reflect a cold allocator.
- `torch.cuda.max_memory_allocated()` is a global counter and is never reset between resolutions, so `memory_peak` is the high-water mark across the whole run, not per resolution.

## Typical wiring

```
MODEL ┐
VAE   ├-> ROCmFluxBenchmark ├─ BENCHMARK_RESULTS / PERFORMANCE_CHART
CLIP  ┘                      ├─ OPTIMIZATION_RECOMMENDATIONS
                             └─ MEMORY_ANALYSIS   (each -> Show Text)
```

## Troubleshooting

- A resolution missing from `BENCHMARK_RESULTS` means every decode for it failed; the console shows `Error testing <W>x<H> CFG <cfg>: ...` with the real exception (typically the channel-count mismatch above).
- Long runtime is expected: resolutions × CFG values decodes, each timed separately.
- `ValueError` on a malformed `test_resolutions` / `test_cfg_values` string — use `WxH` and plain floats.
- Any machine numbers quoted for this pack come from `Benchmarks.md` / `README.md`; this page deliberately has none.

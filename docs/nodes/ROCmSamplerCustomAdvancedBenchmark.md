# ROCmSamplerCustomAdvancedBenchmark

Display name: `ROCmSamplerCustomAdvanced Benchmark` · Category: `ROCm Ninodes/Sampling` · Legacy class names: `ROCMSamplerCustomAdvancedBenchmark` · Introduced: 2.2.0

Source: `rocm_nodes/core/sampler.py`

## What it does

A/B harness for one sampling pass. Phase 1 runs the stock path (`guider.sample()` with no callback, progress bar enabled). Phase 2 runs the ROCm path: `apply_rocm_backend_settings()`, `emergency_memory_cleanup()` for high-memory models, a lightweight progress callback, and `disable_pbar` for video latents. Both phases run on identical inputs, each preceded by `torch.cuda.empty_cache()` + `gc.collect()`, and the node reports wall time and allocated-memory delta for both.

## Inputs

| Name | Type | Default | Range / options | Notes |
|---|---|---|---|---|
| `noise` | NOISE | — | required | Used for both phases; `noise.seed` forwarded. |
| `guider` | GUIDER | — | required | `guider.sample()`; `guider.model_patcher` is inspected for model type. |
| `sampler` | SAMPLER | — | required | Passed through. |
| `sigmas` | SIGMAS | — | required | `total_steps = sigmas.shape[-1] - 1`. |
| `latent_image` | LATENT | — | required | `noise_mask` honoured. Note: unlike `ROCmSamplerCustomAdvanced`, this node does **not** call `fix_empty_latent_channels()`. |

`FUNCTION = run`. `RETURN_TYPES = ("LATENT", "STRING")`.

## Outputs

| Name | Type | Notes |
|---|---|---|
| `LATENT` | LATENT | Latent dict copy carrying the **ROCm** phase result (stock result is discarded). |
| `BENCHMARK_REPORT` | STRING | Stock time + peak delta, ROCm time + peak delta, speed % (`(stock/rocm)*100 - 100`), memory delta, step count, model type / latent channels / memory factor, GPU name. |

## gfx1151 notes

- Sampling work is doubled: both phases run to completion on every queue. Use it for measurement, not production.
- Memory figures are `torch.cuda.memory_allocated()` deltas around each phase, not true peak allocation; on gfx1151 unified memory this is the torch allocator view, not system RAM.
- Phase 2 applies the same `apply_rocm_backend_settings()` as the production node, so on gfx1151 with ROCm < 7.14 fp16 accumulation stays disabled.
- `torch.cuda.empty_cache()` is called unconditionally between phases; it requires CUDA to be available (no guard).
- Neither phase generates latent previews; phase 1 runs with ComfyUI's stock progress bar (`disable_pbar=False`), phase 2 uses a lightweight bar callback and disables the bar for video latents.

## Typical wiring

```
RandomNoise + BasicGuider + KSamplerSelect + BasicScheduler
  -> ROCmSamplerCustomAdvancedBenchmark
       ├─ LATENT            -> ROCmVAEDecode
       └─ BENCHMARK_REPORT  -> a text display node
```

## Troubleshooting

- A negative speed % means the ROCm path was slower than stock for this workload — expected on models where the ROCm layer only adds logging.
- `inf`/zero division is guarded: `time_speedup` is 0 when the ROCm phase measured 0 s.
- Because the stock phase runs first with the progress bar enabled, cancelling during phase 1 never reaches phase 2.
- Compare against `ROCmSamplerCustomAdvanced` timings rather than trusting absolute numbers across runs; the first run after a restart includes kernel warmup.

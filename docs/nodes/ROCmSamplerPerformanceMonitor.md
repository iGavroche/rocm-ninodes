# ROCmSamplerPerformanceMonitor

Display name: `ROCm Sampler Performance Monitor` · Category: `ROCm Ninodes/Sampling` · Legacy class names: `ROCMSamplerPerformanceMonitor` · Introduced: 1.0.0

Source: `rocm_nodes/core/sampler.py`

## What it does

Analysis-only node: it inspects the model and the GPU and returns three text blocks. No sampling happens. It reports architecture family, APU mode, model device and dtype, GPU name, detected sampling type (`flow` / `eps` / `v_prediction` / `unknown`), latent channels and the model's memory usage factor, then derives tips and recommended settings from those.

## Inputs

| Name | Type | Default | Range / options | Notes |
|---|---|---|---|---|
| `model` | MODEL | — | required | Read via `detect_model_sampling_type(model)` plus `model.model_dtype()` and `model_management.get_torch_device()`. |

`FUNCTION = analyze`. `RETURN_TYPES = ("STRING", "STRING", "STRING")`.

## Outputs

| Name | Type | Notes |
|---|---|---|
| `DEVICE_INFO` | STRING | Architecture, APU mode, model device, model dtype, GPU name, sampling type, latent channels, memory factor. Falls back to `Model info unavailable` / `GPU: AMD (ROCm)` if the introspection calls raise. |
| `PERFORMANCE_TIPS` | STRING | Model-type bullets (flow-matching → euler / dpmpp_sde / dpmpp_2m and `sgm_uniform`; large latent ≥ 128ch → memory warning; pixel-space → no VAE decode needed) plus architecture bullets (preferred precision, APU smart-memory note). |
| `OPTIMAL_SETTINGS` | STRING | On AMD: `Recommended precision: <pref>`, `Recommended samplers: euler, dpmpp_2m, dpmpp_sde`, `Recommended schedulers: sgm_uniform, normal`. Non-AMD: precision `auto`, samplers `dpmpp_2m, dpmpp_sde`. |

## gfx1151 notes

- gfx1151/gfx1150 → family `rdna3_5`, `is_apu=True`, preferred precision `fp16`, so the tips read "FP16 RECOMMENDED FOR rdna3_5" and "APU mode: disable smart-memory for unified memory".
- `memory_usage_factor > 5` marks a model high-memory; the sampler nodes act on that, this node only reports it.
- Pixel-space detection (`spacial_downscale_ratio == 1` and 3 latent channels) reports that no VAE decode is needed downstream.
- Side-effect free and cheap.

## Typical wiring

```
ROCmCheckpointLoader.MODEL -> ROCmSamplerPerformanceMonitor
                              ├─ DEVICE_INFO      -> Show Text
                              ├─ PERFORMANCE_TIPS -> Show Text
                              └─ OPTIMAL_SETTINGS -> Show Text
```

## Troubleshooting

- Nothing visible: STRING outputs need a text display node.
- `Model info unavailable` means `model.model_dtype()` or `get_torch_device()` raised; the model-type lines below it are still produced.
- These are recommendations, not measurements. For measured numbers use `ROCmFluxBenchmark`.

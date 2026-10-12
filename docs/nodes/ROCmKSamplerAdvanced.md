# ROCmKSamplerAdvanced

Display name: `ROCm KSampler Advanced` · Category: `ROCm Ninodes/Sampling` · Legacy class names: `ROCMOptimizedKSamplerAdvanced` · Introduced: 1.0.0

Source: `rocm_nodes/core/sampler.py`

## What it does

The stock KSampler (Advanced) behaviour — step-range control, optional noise, leftover noise — under the ROCm category, plus the same ROCm layer as `ROCmKSampler`: architecture detection, `apply_rocm_backend_settings()`, emergency cleanup for high-memory models, model-type logging, and a step callback with elapsed/remaining/per-step timing. The class docstring describes it as a 1:1 copy of ComfyUI's node with no ROCm-specific change to the sampling math.

## Inputs

| Name | Type | Default | Range / options | Notes |
|---|---|---|---|---|
| `model` | MODEL | — | required | Inspected by `detect_model_sampling_type()`. |
| `add_noise` | combo | `enable` | `enable`, `disable` | `disable` → `disable_noise=True` and noise is a zeros tensor on CPU with the latent's dtype/layout. |
| `noise_seed` | INT | `0` | 0 … 0xffffffffffffffff | `prepare_noise()` and `sample(seed=...)`. |
| `steps` | INT | `20` | 1 … 10000 | Passed through. |
| `cfg` | FLOAT | `8.0` | 0.0 … 100.0, step 0.1, round 0.01 | Passed through. |
| `sampler_name` | combo | first entry | `comfy.samplers.KSampler.SAMPLERS` | Passed through. |
| `scheduler` | combo | first entry | `comfy.samplers.KSampler.SCHEDULERS` | Passed through. |
| `positive` | CONDITIONING | — | required | Passed through. |
| `negative` | CONDITIONING | — | required | Passed through. |
| `latent_image` | LATENT | — | required | `fix_empty_latent_channels()` applied; `batch_index` / `noise_mask` honoured. |
| `start_at_step` | INT | `0` | 0 … 10000 | `start_step=`. |
| `end_at_step` | INT | `10000` | 0 … 10000 | `last_step=`. |
| `return_with_leftover_noise` | combo | `disable` | `disable`, `enable` | `enable` → `force_full_denoise=False`. |
| `optimize_for_video` | BOOLEAN | `False` | optional | With a 5D latent (`shape[2] > 1`) suppresses latent previews and labels the log `Video`. |
| `precision_mode` | combo | `auto` | optional: `auto`, `fp32`, `bf16` | Computes a local `use_bf16` flag from `torch.cuda.is_bf16_supported()`; the flag is unused, so currently a no-op hint. |
| `compatibility_mode` | BOOLEAN | `False` | optional | Skips all ROCm backend tuning and cleanup. |

`sample()` also accepts a `denoise` keyword defaulting to `1.0`, but it is not exposed in `INPUT_TYPES` (same as stock).

`FUNCTION = sample`. `RETURN_TYPES = ("LATENT",)`; this class defines no `RETURN_NAMES`.

## Outputs

| Name | Type | Notes |
|---|---|---|
| `LATENT` | LATENT | Input latent dict with `samples` replaced. |

## gfx1151 notes

- Same `apply_rocm_backend_settings()` behaviour as `ROCmKSampler`: fp16 accumulation only re-enabled on gfx1151 at ROCm ≥ 7.14; hipBLASLt advisory; TF32 left at the ROCm default.
- High-memory models get `emergency_memory_cleanup()` before and `gentle_memory_cleanup()` after sampling.
- `comfy.sample.sample()` is called with `disable_pbar=False` regardless of `optimize_for_video`, so the enhanced callback always runs; the video flag only controls preview generation.
- Progress prints at every step; UI bar updates are throttled to 0.3 s.

## Typical wiring

```
ROCmDiffusionLoader.MODEL + conditionings + EmptyLatentVideo
  -> ROCmKSamplerAdvanced (start_at_step=0, end_at_step=20)
  -> LATENT -> ROCmVAEDecode
```

Chain two instances for stage-split sampling: first with `return_with_leftover_noise=enable`, second with `add_noise=disable` and `start_at_step` at the handoff step.

## Troubleshooting

- Nothing sampled: `end_at_step` ≤ `start_at_step`.
- Jitter/repeating frames on WAN video was a VAE-side issue (fixed 2.0.1), not this node.
- Use `compatibility_mode` to compare against pure stock behaviour on the same seed.
- Noise output after a resolution change: latent size mismatch, see `../TROUBLESHOOTING_NOISE.md`.

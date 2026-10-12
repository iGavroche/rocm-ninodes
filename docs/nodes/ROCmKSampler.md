# ROCmKSampler

Display name: `ROCm KSampler` · Category: `ROCm Ninodes/Sampling` · Legacy class names: `ROCMOptimizedKSampler` · Introduced: 1.0.0

Source: `rocm_nodes/core/sampler.py`

## What it does

Stock-equivalent KSampler under the ROCm category. It calls `comfy.sample.sample()` with the same arguments ComfyUI's own node uses, and layers on: architecture detection plus `apply_rocm_backend_settings()`, an emergency memory cleanup for models whose `memory_usage_factor > 5`, model-type detection logging (flow-matching / pixel-space / high-memory), and a step callback that drives the UI progress bar and prints elapsed / remaining / per-step timing.

## Inputs

| Name | Type | Default | Range / options | Notes |
|---|---|---|---|---|
| `model` | MODEL | — | required | Also inspected by `detect_model_sampling_type()` and used for the latent previewer. |
| `seed` | INT | `0` | 0 … 0xffffffffffffffff | `comfy.sample.prepare_noise(...)` and passed to `sample()`. |
| `steps` | INT | `20` | 1 … 10000 | Also sizes `comfy.utils.ProgressBar`. |
| `cfg` | FLOAT | `8.0` | 0.0 … 100.0, step 0.1, round 0.01 | Passed through. |
| `sampler_name` | combo | first entry | `comfy.samplers.KSampler.SAMPLERS` | Passed through. |
| `scheduler` | combo | first entry | `comfy.samplers.KSampler.SCHEDULERS` | Passed through. |
| `positive` | CONDITIONING | — | required | Passed through. |
| `negative` | CONDITIONING | — | required | Passed through. |
| `latent_image` | LATENT | — | required | `samples` go through `comfy.sample.fix_empty_latent_channels()`; `batch_index` and `noise_mask` are honoured. |
| `denoise` | FLOAT | `1.0` | 0.0 … 1.0, step 0.01 | Passed through. |
| `optimize_for_video` | BOOLEAN | `False` | optional | Only takes effect when the latent is 5D with `shape[2] > 1`: then latent previews are not generated and the log line is labelled `Video`. |
| `precision_mode` | combo | `auto` | optional: `auto`, `fp32`, `bf16` | Queries `torch.cuda.is_bf16_supported()` into a local flag; the flag is not used by the sampling call, so this is currently a no-op hint. |
| `compatibility_mode` | BOOLEAN | `False` | optional | Skips `apply_rocm_backend_settings()`, the high-memory cleanup, the model-type logging, the video preview suppression and the bf16 check. |

`FUNCTION = sample`. `RETURN_TYPES = ("LATENT",)`.

## Outputs

| Name | Type | Notes |
|---|---|---|
| `LATENT` | LATENT | Copy of the input latent dict with `samples` replaced by the sampler output. |

## gfx1151 notes

- `apply_rocm_backend_settings()` on `rdna3_5` (gfx1151/gfx1150): `torch.backends.cuda.matmul.allow_fp16_accumulation = True` only on ROCm ≥ 7.14 — on older ROCm it is deliberately skipped because it caused numerical drift and `hipErrorIllegalAddress` in bf16 flow-matching models (CHANGELOG 2.2.5 / 2.2.2). It also warns when `torch.backends.cuda.preferred_blas_library()` is not hipBLASLt, suggesting `TORCH_BLAS_PREFER_HIPBLASLT=1`.
- TF32 is left at the ROCm default; forcing it off caused a ~2x slowdown on LTX 2.3 (fixed 2.2.6).
- High-memory models (`memory_usage_factor > 5`, e.g. LTX 128ch, Ideogram4) get `emergency_memory_cleanup()` before sampling and `gentle_memory_cleanup()` after.
- Previews are throttled to one update per 0.3 s and generated every 5th step; a previewer failure prints the exception type and retries next step instead of silently dropping previews.
- `comfy.sample.sample()` is always called with `disable_pbar=False` so the callback runs (fixed 2.0.3).

## Typical wiring

```
ROCmCheckpointLoader.MODEL + CLIPTextEncode x2 + EmptyLatentImage
  -> ROCmKSampler -> LATENT -> ROCmVAEDecode -> SaveImage
```

## Troubleshooting

- `hipErrorIllegalAddress` / banding on bf16 flow-matching models: check the ROCm version; on < 7.14 the node already avoids fp16 accumulation, so look elsewhere (e.g. another node setting it globally, such as `ROCmGGUFLoader`).
- Previews missing on Windows: the previewer exception is now logged (fixed 2.2.8) — read the console line rather than assuming the sampler hung.
- Noise-only output is almost always a latent-size mismatch, see `../TROUBLESHOOTING_NOISE.md`.
- Set `compatibility_mode` to isolate whether a ROCm-side setting is responsible for a regression.

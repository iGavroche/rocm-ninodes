# ROCmSamplerCustomAdvanced

Display name: `ROCmSamplerCustomAdvanced` (`ROCm SamplerCustomAdvanced` in the node's own `io.Schema`) · Category: `ROCm Ninodes/Sampling` · Legacy class names: `ROCMSamplerCustomAdvanced` · Introduced: 2.2.0

Source: `rocm_nodes/core/sampler.py` (V3 `io.ComfyNode`, `define_schema()`)

## What it does

Drop-in replacement for ComfyUI's `SamplerCustomAdvanced` with the same five inputs and the same `guider.sample()` call. Added value: ROCm backend tuning, `emergency_memory_cleanup()` for high-memory models, a per-step callback with ETA and 10%-milestone console logging, automatic video detection (5D latent with T > 1 → no previews, progress bar disabled), a pre-flight free-memory warning, and a per-node flash-attention override that is restored afterwards. It also emits the denoised `x0` latent as a second output.

## Inputs

| Name | Type | Default | Range / options | Notes |
|---|---|---|---|---|
| `noise` | NOISE | — | required | `noise.generate_noise(latent)`, `noise.seed` forwarded. |
| `guider` | GUIDER | — | required | Source of `model_patcher`; `guider.sample(...)`. |
| `sampler` | SAMPLER | — | required | Passed through. |
| `sigmas` | SIGMAS | — | required | `total_steps = sigmas.shape[-1] - 1`. |
| `latent_image` | LATENT | — | required | `fix_empty_latent_channels()` is called with the latent's `downscale_ratio_spacial` / `downscale_ratio_temporal`; `noise_mask` honoured. |
| `compatibility_mode` | BOOLEAN (optional, advanced) | `False` | — | Skips `apply_rocm_backend_settings()`, ROCm logging, the high-memory cleanup and the VRAM safety check. Does not affect the flash-attention override. |
| `flash_attention` | combo (optional, advanced) | `auto` | `auto`, `disable_flash`, `force_flash` | `disable_flash`: flash SDP off, math + mem-efficient on. `force_flash`: flash on while keeping math and mem-efficient available as fallback. The three prior SDP states are saved and restored in a `finally` block. No-op when CUDA is unavailable or on `auto`. |

## Outputs

| Name | Type | Notes |
|---|---|---|
| `output` | LATENT | Latent dict copy with `downscale_ratio_*` keys popped and `samples` set to the sampler output moved to `comfy.model_management.intermediate_device()`. |
| `denoised_output` | LATENT | `process_latent_out(x0)` when the callback captured an `x0`; otherwise identical to `output`. For nested (audio-video) samples, `x0` is only unpacked when samples are nested but `x0` is not — matching stock (fixed 2.3.1). |

## gfx1151 notes

- `apply_rocm_backend_settings()` for `rdna3_5`: fp16 matmul accumulation only on ROCm ≥ 7.14 (older ROCm caused drift / `hipErrorIllegalAddress` on bf16 flow-matching models); hipBLASLt advisory; TF32 untouched.
- High-memory models (`memory_usage_factor > 5`, e.g. LTX 128ch) get `emergency_memory_cleanup()` before sampling.
- VRAM safety check: if free memory < 2.0 GB or usage > 95% of device total, it prints a warning naming the detected architecture and suggesting: disable CacheDiT, remove `--reserve-vram`, reduce frame count, or use `--use-pytorch-attention`. Free memory on an APU comes from `psutil` system-wide available RAM (`get_gpu_memory_info(is_apu=True)`).
- The previewer is created once outside the callback (fixed 2.2.1); previews every 5th step, bar updates throttled to 0.3 s, console timing only at step 0, step last, and each new 10% bucket.
- `throw_exception_if_processing_interrupted()` runs each step so Cancel stops immediately (added 2.2.4).
- `disable_pbar=True` for video latents; the callback still runs.

## Typical wiring

```
RandomNoise + BasicGuider(ROCmDiffusionLoader.MODEL, cond) + KSamplerSelect + BasicScheduler
  -> ROCmSamplerCustomAdvanced -> output -> ROCmVAEDecode
                                 -> denoised_output (x0, for analysis / audio VAE)
```

For MiniMax H3, put `VedaSparseAttention` and/or `ROCmH3EasyCache` on the MODEL wire feeding the guider, and this node last in the sampling chain.

## Troubleshooting

- `hipErrorIllegalAddress` mid-sample: set `flash_attention=disable_flash` to route through math/efficient SDP.
- OOM on large models: the pre-flight warning names the mitigations; `compatibility_mode` removes the node's own global torch-state changes for bisection.
- `'NestedTensor' object has no attribute 'reshape'` after sampling was fixed in 2.3.1; if it appears, the pack is older than 2.3.1.
- Missing previews on Windows are a previewer failure, now logged with the exception type rather than swallowed.

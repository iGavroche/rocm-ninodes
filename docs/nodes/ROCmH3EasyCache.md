# ROCmH3EasyCache

Display name: `ROCm H3 EasyCache` · Category: `ROCm Ninodes/Optimization` · Introduced: 2.3.4

Source: `rocm_nodes/core/h3_easycache.py` (V3 `io.ComfyNode`)

## What it does

Wraps ComfyUI's core EasyCache (`comfy_extras.nodes_easycache`) with defaults tuned for the 4-step MiniMax H3 pipeline. It clones the model, installs an `EasyCacheHolder` in `transformer_options["easycache"]`, and registers three wrappers — `OUTER_SAMPLE`, `CALC_COND_BATCH`, `DIFFUSION_MODEL` — under the key `easycache`. The holder compares the denoising delta between steps against `reuse_threshold`; when the change is small the transformer call is skipped and the previous result reused. With 4 steps, 1 of the 3 eligible steps can be skipped, which the docstring describes as a ~25% speedup when the trajectory is smooth.

## Inputs

| Name | Type | Default | Range / options | Notes |
|---|---|---|---|---|
| `model` | MODEL | — | required | Cloned before patching. Tooltip: MiniMax H3 model after loader + LoRA + Spectrum. `model.model.latent_format.latent_channels` is read for the holder's `output_channels`. |
| `reuse_threshold` | FLOAT | `0.2` | 0.0 … 3.0, step 0.01 | Skip aggressiveness; higher skips more. 0.2 is the stated safe default for 4-step H3; the tooltip suggests 0.3+ if quality holds. |
| `start_percent` | FLOAT (advanced) | `0.15` | 0.0 … 1.0, step 0.01 | Relative step where caching begins — 0.15 ≈ step 2 of 4, leaving step 1 to build cache history. |
| `end_percent` | FLOAT (advanced) | `0.95` | 0.0 … 1.0, step 0.01 | Relative step where caching stops — 0.95 covers steps 2-3 of 4 so the last step always runs. |
| `verbose` | BOOLEAN (advanced) | `False` | — | Per-step cache decisions in the log. |

## Outputs

| Name | Type | Notes |
|---|---|---|
| `model` | MODEL | Patched clone. |

`execute()` is a classmethod; the schema is declared via `io.Schema(node_id="ROCmH3EasyCache", ...)`.

## gfx1151 notes

- Hard-coded holder parameters that are not exposed: `subsample_factor=8`, `offload_cache_diff=False`, `output_channels=<model latent channels>`.
- The node registers only if `comfy_extras.nodes_easycache` imports. On an older ComfyUI the import is guarded in `rocm_nodes/nodes.py` and `rocm_nodes/core/__init__.py`, so the other nodes still load and the console prints `ROCmH3EasyCache unavailable (needs recent ComfyUI)` (fixed 2.3.5 — before that the whole extension failed to load).
- No ROCm-specific code path: the speedup comes from skipping transformer calls, which matters most where those calls are the bottleneck on the APU.
- `LOG.info` prints the effective threshold and percentage range at patch time.

## Typical wiring

```
ROCmDiffusionLoader / checkpoint MODEL -> LoRA loaders -> [Spectrum] -> ROCmH3EasyCache
  -> BasicGuider -> ROCmSamplerCustomAdvanced
```

Place it between the model loader / LoRA chain and the sampler, per the class docstring.

## Troubleshooting

- Node missing from the menu: ComfyUI has no `comfy_extras/nodes_easycache.py` — update ComfyUI and check the startup line about the node being unavailable.
- Quality loss / detail smearing at 4 steps: lower `reuse_threshold` toward 0.1, or set `start_percent` higher so fewer steps are eligible.
- No speedup: with only 4 steps and a non-smooth trajectory nothing is skipped; `verbose=True` shows the per-step decisions.
- It patches `transformer_options`, so it must sit after any node that replaces the transformer options wholesale.

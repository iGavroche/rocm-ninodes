# ROCmCheckpointLoader

Display name: `ROCm Checkpoint Loader` · Category: `ROCm Ninodes/Loaders` · Legacy class names: `ROCMOptimizedCheckpointLoader`

Source: `rocm_nodes/core/checkpoint.py`

## What it does

Wraps ComfyUI's native `comfy.sd.load_checkpoint_guess_config()` and adds ROCm diagnostics (GPU/ROCm version, allocated/reserved memory, fragmentation warning). It does not change how the checkpoint is loaded or which dtypes are used. Its real functional addition is a module-level single-slot in-memory cache: repeated API runs with the same `ckpt_name` return the cached `(MODEL, CLIP, VAE)` without touching disk, and loading a *different* checkpoint evicts the previous slot so RAM stays bounded on a unified-memory box.

## Inputs

| Name | Type | Default | Range / options | Notes |
|---|---|---|---|---|
| `ckpt_name` | combo (`folder_paths.get_filename_list("checkpoints")`) | first entry | installed checkpoints | Resolved with `get_full_path_or_raise("checkpoints", ...)`. Filename is scanned for `fp8`, `int8`, `int4`, `quantized`, `bnb`, `gguf`; a match only prints a "preserving original precision" notice. |
| `use_cache` | BOOLEAN | `True` | — | On cache hit (same `ckpt_name`, no `force_reload`) returns the cached tuple and prints `Checkpoint cache hit`. |
| `force_reload` | BOOLEAN | `False` | — | Bypasses the cache, loads from disk, and overwrites the cache slot. |
| `compatibility_mode` | BOOLEAN | `False` | — | Runs `_validate_components()`: raises if `clip is None` or the CLIP object has no `encode` attribute. Does not otherwise change loading. |

## Outputs

| Name | Type | Notes |
|---|---|---|
| `MODEL` | MODEL | `out[0]` of `load_checkpoint_guess_config` |
| `CLIP` | CLIP | `out[1]` |
| `VAE` | VAE | `out[2]` |

`FUNCTION = load_checkpoint`. Raises `ValueError` if the native loader returns fewer than 3 items.

## gfx1151 notes

- No dtype forcing, no `expandable_segments` manipulation, no aggressive cleanup on the happy path — the file's docstring states modern PyTorch (2.7+) allocators are trusted. Cleanup happens only on slot eviction (`gc.collect()` + `gentle_memory_cleanup()` = `empty_cache` + `synchronize` + `gc`).
- The cache is one slot per process (`_checkpoint_cache_key`), keyed by filename only — not by mtime or content hash. Swapping between two checkpoints every run therefore always reloads.
- Memory diagnostics use `torch.cuda` counters; on an APU that is the unified pool, so "total" is the reported device memory, not a discrete VRAM budget.
- Cache reuse of an FP8 checkpoint with `MixedPrecisionOps` required a ComfyUI-side fix (`comfy/ops.py` `QuantizedTensor` handling in `_apply`); see CHANGELOG 2.0.10.

## Typical wiring

```
ROCmCheckpointLoader
  ├─ MODEL → ROCmKSampler / ROCmSamplerCustomAdvanced (via guider)
  ├─ CLIP  → CLIPTextEncode → ROCmKSampler (positive/negative)
  └─ VAE   → ROCmVAEDecode
```

## Troubleshooting

- `torch.cuda.OutOfMemoryError` is caught, logged with current memory status plus a suggestion list (close GPU users, restart ComfyUI, smaller/quantized model, check other loaded models, update PyTorch/ROCm), then re-raised.
- `Checkpoint loading failed` prints the resolved path and whether it exists — a wrong `ckpt_name` or a moved `checkpoints` folder shows up here.
- Large checkpoints print `Loading model components (may take 2-6 minutes)`; a run that stalls is not necessarily hung.
- Stale cache after editing a checkpoint on disk: set `force_reload`.
- `CLIP model is None` / `missing 'encode' method` only appear with `compatibility_mode` on; they mean the file has no usable text encoder.

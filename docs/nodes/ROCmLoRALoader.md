# ROCmLoRALoader

Display name: `ROCm LoRA Loader` · Category: `ROCm Ninodes/Loaders` · Legacy class names: `ROCMLoRALoader`

Source: `rocm_nodes/core/lora.py`

## What it does

Applies a LoRA to a MODEL (and optionally a CLIP) using ComfyUI's native path: `comfy.utils.load_torch_file(lora_path, safe_load=True)` followed by `comfy.sd.load_lora_for_models(model, clip, lora, strength_model, strength_clip)`. The ROCm-specific part is memory hygiene: `gentle_memory_cleanup()` (`empty_cache` + `synchronize` + `gc.collect()`) runs before loading, between the file read and the patch application, after success, and again on any exception, with allocated/reserved/fragmentation figures printed around the operation.

## Inputs

| Name | Type | Default | Range / options | Notes |
|---|---|---|---|---|
| `model` | MODEL | — | required | Model to patch. |
| `lora_name` | combo (`folder_paths.get_filename_list("loras")`) | first entry | installed LoRAs | Resolved with `folder_paths.get_full_path("loras", ...)`; `FileNotFoundError` if it returns `None`. |
| `strength_model` | FLOAT | `1.0` | -10.0 … 10.0, step 0.01 | Passed to `load_lora_for_models`. |
| `strength_clip` | FLOAT | `1.0` | -10.0 … 10.0, step 0.01 | Passed to `load_lora_for_models`. |
| `clip` | CLIP | — | optional | May be left unconnected (`clip=None`); the CLIP output is then whatever `load_lora_for_models` returns for a `None` input. |

`FUNCTION = load_lora`. `RETURN_TYPES = ("MODEL", "CLIP")`.

## Outputs

| Name | Type | Notes |
|---|---|---|
| `MODEL` | MODEL | LoRA-patched model. |
| `CLIP` | CLIP | LoRA-patched CLIP, or `None` when no CLIP was supplied. |

## gfx1151 notes

- No dtype or precision changes; the LoRA patch itself is stock ComfyUI.
- The cleanup calls are the reason to prefer this node over stock `LoraLoader` in long chains on unified memory: they keep reserved-but-unallocated memory from accumulating between LoRA applications.
- Cleanup is skipped entirely when `torch.cuda.is_available()` is false.
- Negative strengths are accepted by the widget and passed straight through, matching stock semantics.

## Typical wiring

```
ROCmCheckpointLoader.MODEL ─┐
ROCmCheckpointLoader.CLIP ──┼-> ROCmLoRALoader -> MODEL/CLIP -> ROCmKSampler
                             (repeat per LoRA)
```

## Troubleshooting

- `LoRA file not found: <name>` — file missing from `models/loras`.
- Any failure prints `[ERROR] ROCM LoRA Loader: Failed to load ...`, runs an emergency `gentle_memory_cleanup()`, then re-raises.
- `'str' object has no attribute 'keys'` from `comfy.lora.load_lora` was the legacy bug (a path string passed where a dict was expected); fixed in 2.3.2 by switching to `load_torch_file` + `load_lora_for_models`. If it appears, an old workflow is hitting a stale node implementation.
- A LoRA whose keys do not match the model simply has no effect (stock `load_lora_for_models` behaviour); this node adds no extra validation.

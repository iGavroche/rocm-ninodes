# ROCmGGUFLoader

Display name: `ROCm GGUF Loader` · Category: `ROCm Ninodes/Loaders`

Source: `rocm_nodes/core/gguf_loader.py`

## What it does

Reads a GGUF diffusion model directly (no safetensors conversion) with lazy dequantization: `_gguf_sd_loader()` parses the file with `gguf.GGUFReader`, wraps every tensor in a `ROCmGGMLTensor` subclass that keeps the raw quantized block data, and `comfy.sd.load_diffusion_model_state_dict()` builds the model with `custom_operations = ROCmGGMLOps`. Weights are dequantized per forward pass in `cast_bias_weight()` / `get_weight()`, so peak memory stays near the on-disk quantized size instead of the dequantized size. The implementation follows City96's ComfyUI-GGUF structure.

## Inputs

| Name | Type | Default | Range / options | Notes |
|---|---|---|---|---|
| `gguf_name` | combo (`_get_gguf_files()`) | first entry | any `.gguf` found in `diffusion_models`, `unet`, `unet_gguf`, `checkpoints` | Path resolution tries those four folders in order; `FileNotFoundError` if none resolve. There is deliberately no dtype selector (removed in 2.0.8). |

`FUNCTION = load_gguf`. `RETURN_TYPES = ("MODEL",)`.

## Outputs

| Name | Type | Notes |
|---|---|---|
| `MODEL` | MODEL | From `comfy.sd.load_diffusion_model_state_dict(sd, model_options={"custom_operations": ops})`; `RuntimeError: Could not detect model type of: <path>` if it returns `None`. |

## gfx1151 notes

- Before loading, on any AMD device this node sets globally: `torch.backends.cuda.matmul.allow_tf32 = False` and `torch.backends.cuda.matmul.allow_fp16_accumulation = True`, and reads `gcnArchName` for a `gfx1151 architecture detected` log line. This is the node's own inline code, not the shared `apply_rocm_backend_settings()` helper, so it does not carry the ROCm < 7.14 gfx1151 fp16-accumulation guard.
- `dequantize_tensor()` defaults to `torch.float32` when no dtype is requested, and uses non-blocking device transfers when `comfy.model_management.device_supports_non_blocking` exists.
- On-device block dequantization is implemented for `BF16`, `Q8_0`, `Q5_1`, `Q5_0`, `Q4_1`, `Q4_0`, `Q6_K`, `Q5_K`, `Q4_K`, `Q3_K`, `Q2_K`. Anything else falls back to the slow CPU `gguf.quants.dequantize()`. The K-quants were added in 2.3.2 (CHANGELOG: ~5x faster Qwen Image inference).
- `simple_memory_cleanup()` (`empty_cache` + `gc`) runs after a successful load on AMD.
- `general.architecture` is validated against the image list `flux, sd1, sdxl, sd3, aura, hidream, cosmos, ltxv, hyvid, wan, lumina2, qwen_image`; unknown names only warn. An arch of `None` or `pig` loads in `sd.cpp` compatibility mode.
- Requires the `gguf` package. If it is missing the module still imports and the node raises an install hint at run time.
- The module tries to load City96's ComfyUI-GGUF (`ops.py`, `loader.py`, `dequant.py`, `nodes.py`) from `<base_path>/custom_nodes/ComfyUI-GGUF` or a sibling `custom_nodes/ComfyUI-GGUF`, and prints which implementation is in use.

## Typical wiring

```
ROCmGGUFLoader.MODEL -> ROCmKSampler / BasicGuider -> ROCmSamplerCustomAdvanced
(+ separate VAELoader / CLIPLoader - GGUF files carry only the diffusion model)
```

## Troubleshooting

- `GGUF library not installed` -> `pip install gguf` (or `uv add gguf`); `install.py` installs it automatically.
- `Could not detect model type of: <path>` -> the GGUF architecture is not recognized by ComfyUI's diffusion-model loader.
- `Multiple dispatch failed for torch.nn.linear` was a historical bug (the custom tensor subclass reaching `torch.nn.linear` directly), fixed in 2.0.7 by converting every `ROCmGGMLTensor` to a plain `torch.Tensor` before ops. If it reappears, a GGUF model reached a node that does not use this loader's ops.
- Slow inference on a quant type missing from the list above: it is silently using the CPU dequantize fallback.
- OOM during load is caught, memory status and suggestions printed, then re-raised.

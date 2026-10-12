# ROCmDiffusionLoader

Display name: `ROCm Diffusion Loader` · Category: `ROCm Ninodes/Loaders`

Source: `rocm_nodes/core/unet_loader.py`

## What it does

Loads a standalone diffusion model (UNet/DiT) through ComfyUI's native `comfy.sd.load_diffusion_model()` and adds ROCm diagnostics. Its second job is file discovery: `_get_model_files()` merges ComfyUI's own `diffusion_models` listing with scans of the `unet`, `unet_gguf` and `diffusion_models` folders plus a raw `os.listdir` pass, so formats ComfyUI's default filter hides still appear in the dropdown. GGUF files are explicitly out of scope — use `ROCmGGUFLoader`.

## Inputs

| Name | Type | Default | Range / options | Notes |
|---|---|---|---|---|
| `unet_name` | combo (`_get_model_files()`) | first entry | `.safetensors`, `.ckpt`, `.pt`, `.pth`, `.bin`, `.onnx` | `MODEL_EXTENSIONS` filters the scanned folders. Path resolution tries folders `diffusion_models`, `unet`, `unet_gguf` in order; raises `FileNotFoundError` if none resolve. |
| `weight_dtype` | combo | `default` | `default`, `fp8_e4m3fn`, `fp8_e5m2` | Applied only when not `default` **and** the filename is not `fp8_scaled`/`fp8-scaled`: then `model_options["dtype"]` becomes `torch.float8_e4m3fn` or `torch.float8_e5m2`. For `fp8_scaled` files an explicit choice is ignored (with a warning) so ComfyUI's own scale detection runs. |

`FUNCTION = load_unet`. `RETURN_TYPES = ("MODEL",)`.

## Outputs

| Name | Type | Notes |
|---|---|---|
| `MODEL` | MODEL | Result of `comfy.sd.load_diffusion_model(unet_path, model_options=...)`; `ValueError` if `None`. |

## gfx1151 notes

- `model_options` is **always** passed, as `{}` when nothing is overridden. The code comments state this matches stock ComfyUI exactly and that omitting it produced noise-only output.
- No dtype is forced for non-fp8 models; weights keep whatever the file declares.
- Filename indicators `fp8`, `int8`, `int4`, `quantized`, `bnb` only trigger an informational print.
- The docstring recommends running ComfyUI with `--cache-none` for workflows holding several >20 GB models.
- Diagnostics print allocated/reserved memory and a fragmentation warning above 0.5 GB.

## Typical wiring

```
ROCmDiffusionLoader.MODEL → ROCmKSampler / BasicGuider → ROCmSamplerCustomAdvanced
(+ separate VAELoader and CLIPLoader for VAE / text encoding)
```

## Troubleshooting

- `Could not find model '<name>' in any of: diffusion_models, unet, unet_gguf` — the file is in a folder this node does not probe.
- A GGUF file selected here fails to parse; that is the wrong loader.
- `fp8_scaled` model with a non-`default` `weight_dtype` prints `[WARNING] fp8_scaled models require 'default' dtype for automatic scaling` and continues with auto-detection.
- OOM is caught, memory status plus a suggestion list are printed, then the error is re-raised.
- Pure noise output after changing resolution is a latent-size mismatch, not this node — see `../TROUBLESHOOTING_NOISE.md`.

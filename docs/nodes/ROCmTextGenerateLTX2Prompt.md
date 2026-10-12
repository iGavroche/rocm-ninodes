# ROCmTextGenerateLTX2Prompt

Display name: `ROCm Text Generate LTX2 Prompt` · Category: `ROCm Ninodes/Generative AI` · Introduced: 2.0.11

Source: `rocm_nodes/core/textgen_ltx2.py`

## What it does

Drop-in replacement for ComfyUI's `TextGenerateLTX2Prompt`: it builds the same Gemma-style chat prompt with the LTX2 T2V or I2V system prompt copied verbatim from `comfy_extras/nodes_textgen.py`, then runs `clip.tokenize(...)` → `clip.generate(...)` → `clip.decode(...)` with the same parameters. The ROCm additions are: forcing the CLIP execution device to `model_management.get_torch_device()`, and a `gentle_memory_cleanup()` before and after generation.

## Inputs

| Name | Type | Default | Range / options | Notes |
|---|---|---|---|---|
| `clip` | CLIP | — | required | LLM text encoder (e.g. Gemma3 12B for LTX2). `set_clip_options({"execution_device": device})` is applied when the method exists. |
| `prompt` | STRING (multiline) | `""` | — | Inserted as `User Raw Input Prompt: <prompt>.` inside the turn template. |
| `max_length` | INT | `256` | 1 … 2048 | `clip.generate(max_length=...)`. |
| `image` | IMAGE | — | optional | Presence selects the I2V system prompt and injects `<image_soft_token>`; `clip.tokenize(image=...)`. |
| `sampling_mode` | combo | `on` | `on`, `off` | `on` → `do_sample=True` (temperature/top_k/top_p active); `off` → greedy. |
| `temperature` | FLOAT | `0.7` | 0.01 … 2.0, step 0.000001 | Passed to `clip.generate`. |
| `top_k` | INT | `64` | 0 … 1000 | Passed to `clip.generate`. |
| `top_p` | FLOAT | `0.95` | 0.0 … 1.0, step 0.01 | Passed to `clip.generate`. |
| `min_p` | FLOAT | `0.05` | 0.0 … 1.0, step 0.01 | Passed to `clip.generate`. |
| `repetition_penalty` | FLOAT | `1.05` | 0.0 … 5.0, step 0.01 | Passed to `clip.generate`. |
| `seed` | INT | `0` | 0 … 0xFFFFFFFFFFFFFFFF | Passed to `clip.generate`. |

`FUNCTION = generate`. `RETURN_TYPES = ("STRING",)`, `RETURN_NAMES = ("generated_text",)`.

## Outputs

| Name | Type | Notes |
|---|---|---|
| `generated_text` | STRING | `clip.decode(generated_ids, skip_special_tokens=True)` — the expanded video prompt, fed to the positive conditioning. |

## gfx1151 notes

- Generation is one forward pass per token with a manual KV cache, and the LTX2 system prompt is ~2k tokens, so the first pass dominates. `docs/LTX2_PROMPT_NODE.md` explains why this path is slow on gfx1151 (the "small input" attention path targets MI200/MI300X SDPA).
- The node's own guidance, repeated in its DESCRIPTION: run ComfyUI with `--use-pytorch-attention` for best speed on ROCm. Omit it if SDPA with custom attention masks misbehaves on a given build.
- `gentle_memory_cleanup()` (`empty_cache` + `synchronize` + `gc`) runs before and after generation; it is a no-op without CUDA.
- No dtype or precision overrides — the CLIP keeps whatever precision ComfyUI loaded it with.

## Typical wiring

```
CLIPLoader (Gemma3 12B) + prompt (+ optional LoadImage)
  -> ROCmTextGenerateLTX2Prompt -> generated_text -> CLIPTextEncode -> guider / KSampler
```

## Troubleshooting

- Very slow generation: check that `--use-pytorch-attention` is set; see `../LTX2_PROMPT_NODE.md`.
- Wrong attention path / crashes with SDPA on some ROCm+PyTorch combos: drop the flag and retest.
- Garbled or truncated output: raise `max_length`; the strict single-paragraph format is enforced only by the system prompt, not by code.
- `sampling_mode=off` makes temperature/top_k/top_p irrelevant (greedy decoding).

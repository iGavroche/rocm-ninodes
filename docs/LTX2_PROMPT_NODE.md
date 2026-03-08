# ROCm LTX2 Prompt Node

## Why the stock TextGenerateLTX2Prompt is slow on gfx1151

ComfyUI's **TextGenerateLTX2Prompt** uses the same Gemma3 12B text encoder as the rest of the LTX2 pipeline. Generation is slow on ROCm (e.g. gfx1151 / Strix Halo) because:

1. **Attention path** – The LLM uses a "small input" attention path that selects either PyTorch SDPA or a basic einsum-based implementation. On ROCm, the fastest SDPA backends target MI200/MI300X; gfx1151 may use a slower fallback.
2. **One token per step** – Generation runs one forward pass per token with a manual KV cache; there is no batched or flash decoding in this path.
3. **Long context** – The LTX2 system prompt is very long (~2k+ tokens), so the first forward is expensive and each step still does substantial work (e.g. sliding window 1024).

## ROCm Text Generate LTX2 Prompt node

The **ROCm Text Generate LTX2 Prompt** node in ROCm Ninodes is a drop-in replacement:

- **Same inputs**: `clip`, `prompt`, `max_length`, optional `image`, and sampling parameters (sampling_mode, temperature, top_k, top_p, min_p, repetition_penalty, seed).
- **Same output**: A single string (the generated video prompt).
- **Same behavior**: Identical LTX2 T2V/I2V system prompts and the same `clip.tokenize` → `clip.generate` → `clip.decode` flow.

### Recommended ComfyUI flags for ROCm

For best speed when using this node (or the stock TextGenerateLTX2Prompt) on ROCm/AMD GPUs, run ComfyUI with:

```bash
--use-pytorch-attention
```

This makes the text encoder use PyTorch SDPA instead of the basic attention implementation, which can be faster on ROCm. If you see incorrect outputs or crashes with SDPA on certain ROCm/PyTorch versions, omit this flag (known issues have been reported with custom attention masks on some builds).

### Optional: memory cleanup

The ROCm node runs a gentle memory cleanup before and after generation to reduce OOM risk on limited VRAM; this is a no-op when memory is not under pressure.

## When to use which node

- Use **ROCm Text Generate LTX2 Prompt** when you want the same result as the stock node with a single place for ROCm-related settings and future optimizations.
- Keep using the stock **TextGenerateLTX2Prompt** if you prefer; both work with the same clip and produce the same output for the same inputs.

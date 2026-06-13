# ROCM SamplerCustomAdvanced

## Overview

A drop-in replacement for ComfyUI's built-in `SamplerCustomAdvanced` (V3 API, `model/sampling/custom_sampling`) with ROCm-aware optimizations.

**Menu location:** `ROCm Ninodes/Sampling > ROCM SamplerCustomAdvanced`

## Why This Exists

The stock `SamplerCustomAdvanced` in `comfy_extras/nodes_custom_sampler.py` accepts five pluggable sub-components:

```
noise (RandomNoise)      ─┐
guider (CFGGuider)       ─┤
sampler (KSamplerSelect) ─┤── SamplerCustomAdvanced → (output, denoised_output)
sigmas (BasicScheduler)  ─┤
latent_image (...)       ─┘
```

This is the standard pattern for LTX Video, LTX 2.3 AV, and many other advanced ComfyUI workflows. The ROCm version preserves the exact same interface while adding:

| Optimization | What It Does | Impact |
|---|---|---|
| Architecture detection | Detects AMD GPU family (gfx110x, gfx1151, gfx90a, etc.) | Tunes backend settings per GPU |
| ROCm backend tuning | Disables TF32, enables fp16 accumulation on AMD | Small throughput gain |
| Emergency memory defrag | Frees fragmented VRAM before sampling | Prevents OOM on large models |
| Precision management | Auto-selects fp16 vs bf16 based on GPU capability | Matches hardware capability |
| Enhanced callback | Per-step timing, ETA, preview generation | Real-time visibility |
| Video workflow mode | Disables previews for multi-frame latents | Faster long-video generation |
| Post-sample cleanup | Gentle memory defrag after generation | Reduces fragmentation |

## Inputs

| Input | Type | Required | Notes |
|---|---|---|---|
| `noise` | NOISE | Yes | e.g. RandomNoise |
| `guider` | GUIDER | Yes | e.g. CFGGuider, BasicGuider |
| `sampler` | SAMPLER | Yes | e.g. KSamplerSelect |
| `sigmas` | SIGMAS | Yes | e.g. BasicScheduler |
| `latent_image` | LATENT | Yes | e.g. LTXVConcatAVLatent |
| `compatibility_mode` | BOOLEAN | No (advanced) | Skip all ROCm optimizations, use pure stock behavior |

## Outputs

| Output | Type | Description |
|---|---|---|
| `output` | LATENT | The sampled latent |
| `denoised_output` | LATENT | The denoised (x0) prediction, if available |

## Usage

### Replace SamplerCustomAdvanced in an Existing Workflow

1. Delete the stock `SamplerCustomAdvanced` node
2. Add `ROCM SamplerCustomAdvanced` from `ROCm Ninodes/Sampling`
3. Wire the same connections (noise, guider, sampler, sigmas, latent_image)
4. Run — the console will show GPU architecture, model detection, and per-step progress

### LTX 2.3 / LTX Video Workflows

For LTX-based workflows (like `LTX2.3-Director-App-Mariah.json`), the `latent_image` input typically comes from `LTXVConcatAVLatent`. ROCMSamplerCustomAdvanced detects the flow-matching model type (128ch latent, memory factor 5.5x) and applies appropriate memory management before and after sampling. Video (5D latent with T > 1) is auto-detected — previews are disabled for multi-frame latents to reduce overhead.

### When to Use `compatibility_mode`

Enable it via the node's advanced properties panel if you suspect the ROCm optimizations are causing issues. This returns the exact stock behavior for debugging.

## Benchmark Node

`ROCMSamplerCustomAdvancedBenchmark` (`ROCm Ninodes/Sampling > ROCm SamplerCustomAdvanced Benchmark`) runs both the stock and ROCm-optimized sampler in sequence on identical inputs and outputs a comparison string.

**Inputs:** Same as SamplerCustomAdvanced (no extra params).

**Outputs:**
- `LATENT` — result from the ROCm-optimized run
- `BENCHMARK_REPORT` — multiline string with timing and memory comparison

Example output:

```
Benchmark Results
──────────────────────────────────────────────────
Stock:  42.35s, 5840MB peak
ROCm:   41.12s, 5720MB peak
Speed:  +3.0%
Memory: +120MB delta
Steps:  40
Model:  flow (128ch, mem factor 5.5x)
GPU:    AMD Radeon Graphics (gfx1151)
──────────────────────────────────────────────────
```

## Implementation Details

See `rocm_nodes/core/sampler.py`:

```
ROCMSamplerCustomAdvanced      (V3 io.ComfyNode, define_schema)
ROCMSamplerCustomAdvancedBenchmark  (V1 legacy API, INPUT_TYPES)
```

The node uses the ComfyUI V3 (ComfyNode) API via `comfy_api.latest.io`, which ComfyUI's execution system detects automatically from the class inheritance.

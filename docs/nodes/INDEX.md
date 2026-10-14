# Node index

All nodes shipped by ROCm Ninodes (`rocm_nodes/nodes.py` `NODE_CLASS_MAPPINGS` plus `VedaSparseAttention`, registered in the top-level `__init__.py`). Target hardware for the notes in each file: AMD gfx1151 (Strix Halo, RDNA 3.5), 128 GB unified memory, ROCm 7.x, Manjaro Linux.

## Loaders

| Class | Display name | Category | Purpose | Doc |
|---|---|---|---|---|
| `ROCmCheckpointLoader` | ROCm Checkpoint Loader | ROCm Ninodes/Loaders | Native checkpoint load (MODEL/CLIP/VAE) with ROCm diagnostics and a single-slot in-memory cache. | [ROCmCheckpointLoader.md](ROCmCheckpointLoader.md) |
| `ROCmDiffusionLoader` | ROCm Diffusion Loader | ROCm Ninodes/Loaders | Standalone diffusion-model loader with broad file discovery and optional fp8 weight dtype. | [ROCmDiffusionLoader.md](ROCmDiffusionLoader.md) |
| `ROCmGGUFLoader` | ROCm GGUF Loader | ROCm Ninodes/Loaders | GGUF diffusion models with lazy on-device dequantization (Q2_K…Q6_K, Q4_0/Q5_0/Q8_0/BF16). | [ROCmGGUFLoader.md](ROCmGGUFLoader.md) |
| `ROCmLoRALoader` | ROCm LoRA Loader | ROCm Ninodes/Loaders | Stock LoRA application wrapped in memory cleanup and fragmentation reporting. | [ROCmLoRALoader.md](ROCmLoRALoader.md) |

## Sampling

| Class | Display name | Category | Purpose | Doc |
|---|---|---|---|---|
| `ROCmKSampler` | ROCm KSampler | ROCm Ninodes/Sampling | Stock KSampler plus ROCm backend tuning, high-memory cleanup and timed progress. | [ROCmKSampler.md](ROCmKSampler.md) |
| `ROCmKSamplerAdvanced` | ROCm KSampler Advanced | ROCm Ninodes/Sampling | Stock KSampler (Advanced) with step ranges, plus the same ROCm layer. | [ROCmKSamplerAdvanced.md](ROCmKSamplerAdvanced.md) |
| `ROCmSamplerCustomAdvanced` | ROCmSamplerCustomAdvanced | ROCm Ninodes/Sampling | Drop-in `SamplerCustomAdvanced` with memory management, VRAM precheck, `attention_backend` override and `x0` output. | [ROCmSamplerCustomAdvanced.md](ROCmSamplerCustomAdvanced.md) |
| `ROCmSamplerPerformanceMonitor` | ROCm Sampler Performance Monitor | ROCm Ninodes/Sampling | Reports model type, latent format and architecture; recommends samplers/schedulers. No measurement. | [ROCmSamplerPerformanceMonitor.md](ROCmSamplerPerformanceMonitor.md) |

## VAE

| Class | Display name | Category | Purpose | Doc |
|---|---|---|---|---|
| `ROCmVAEDecode` | ROCm VAE Decode | ROCm Ninodes/VAE | VAE decode with precision selection, direct/tiled decision and causal-aware temporal tiling for LTX/WAN. | [ROCmVAEDecode.md](ROCmVAEDecode.md) |
| `ROCmVAEDecodeTiled` | ROCm VAE Decode Tiled | ROCm Ninodes/VAE | Explicit spatial + temporal tiled decode with causal chunk blending and `last_frame_fix`. | [ROCmVAEDecodeTiled.md](ROCmVAEDecodeTiled.md) |
| `ROCmVAEPerformanceMonitor` | ROCm VAE Performance Monitor | ROCm Ninodes/VAE | Reports VAE type/architecture and derives tile/overlap/precision recommendations. No measurement. | [ROCmVAEPerformanceMonitor.md](ROCmVAEPerformanceMonitor.md) |

## Audio

| Class | Display name | Category | Purpose | Doc |
|---|---|---|---|---|
| `ROCmSeedVoiceConversion` | ROCm Seed Voice Conversion | 🎛 ROCm/audio | Seed-VC zero-shot voice conversion (speech 22 kHz / singing 44 kHz) with an fp16 BigVGAN vocoder tuned for gfx1151. | [ROCmSeedVoiceConversion.md](ROCmSeedVoiceConversion.md) |

## Monitors and benchmark

| Class | Display name | Category | Purpose | Doc |
|---|---|---|---|---|
| `ROCmFluxBenchmark` | ROCm Flux Benchmark | ROCm Ninodes/Benchmark | Times real VAE decodes across resolutions/CFG values and reports timings, peak memory, recommendations. | [ROCmFluxBenchmark.md](ROCmFluxBenchmark.md) |
| `ROCmMemoryOptimizer` | ROCm Memory Optimizer | ROCm Ninodes/Memory | Memory status panel with frequency-gated gc / `empty_cache` / sync. | [ROCmMemoryOptimizer.md](ROCmMemoryOptimizer.md) |

## Text generation

| Class | Display name | Category | Purpose | Doc |
|---|---|---|---|---|
| `ROCmTextGenerateLTX2Prompt` | ROCm Text Generate LTX2 Prompt | ROCm Ninodes/Generative AI | Drop-in `TextGenerateLTX2Prompt` with device pinning and memory cleanup around the LLM call. | [ROCmTextGenerateLTX2Prompt.md](ROCmTextGenerateLTX2Prompt.md) |

## Attention and step caching

| Class | Display name | Category | Purpose | Doc |
|---|---|---|---|---|
| `VedaSparseAttention` | Veda Sparse Attention (MiniMax H3) | ROCm Ninodes/Optimization | Learned block-sparse attention for MiniMax-H3 via a predictor bundle and a Triton INT8 kernel. | [VedaSparseAttention.md](VedaSparseAttention.md) |
| `ROCmH3EasyCache` | ROCm H3 EasyCache | ROCm Ninodes/Optimization | Core EasyCache pre-tuned for 4-step MiniMax H3 (threshold 0.2, 15-95% range). | [ROCmH3EasyCache.md](ROCmH3EasyCache.md) |

## Legacy class names

Old workflow keys still registered in `NODE_CLASS_MAPPINGS` (all resolve to the canonical class above):

| Legacy key | Canonical class |
|---|---|
| `ROCMOptimizedCheckpointLoader` | `ROCmCheckpointLoader` |
| `ROCMOptimizedVAEDecode` | `ROCmVAEDecode` |
| `ROCMOptimizedVAEDecodeV2Phase3` | `ROCmVAEDecode` |
| `ROCMOptimizedVAEDecodeTiled` | `ROCmVAEDecodeTiled` |
| `ROCMVAEPerformanceMonitor` | `ROCmVAEPerformanceMonitor` |
| `ROCMOptimizedKSampler` | `ROCmKSampler` |
| `ROCMOptimizedKSamplerAdvanced` | `ROCmKSamplerAdvanced` |
| `ROCMSamplerCustomAdvanced` | `ROCmSamplerCustomAdvanced` |
| `ROCMSamplerPerformanceMonitor` | `ROCmSamplerPerformanceMonitor` |
| `ROCMFluxBenchmark` | `ROCmFluxBenchmark` |
| `ROCMMemoryOptimizer` | `ROCmMemoryOptimizer` |
| `ROCMLoRALoader` | `ROCmLoRALoader` |

Notes:

- `ROCmSamplerCustomAdvanced` is also given a display name inside its `io.Schema` (`ROCm SamplerCustomAdvanced`); the value in the table above is the `NODE_DISPLAY_NAME_MAPPINGS` entry.
- `ROCmH3EasyCache` is registered only when the installed ComfyUI provides `comfy_extras.nodes_easycache`; on older ComfyUI the import is skipped with a console message and the other nodes still load.
- `VedaSparseAttention` is registered only when `veda_comfy` imports; the package ships inside `custom_nodes/rocm-ninodes/veda_comfy/`.

# VedaSparseAttention

Display name: `Veda Sparse Attention (MiniMax H3)` · Category: `ROCm Ninodes/Optimization` · Introduced: 2.3.8

Source: `veda_comfy/nodes.py` in this pack (`custom_nodes/rocm-ninodes/veda_comfy/`), registered directly by the top-level `__init__.py`:

```python
from veda_comfy.nodes import VedaSparseAttention, register_model_folder
register_model_folder()
NODE_CLASS_MAPPINGS['VedaSparseAttention'] = VedaSparseAttention
NODE_DISPLAY_NAME_MAPPINGS['VedaSparseAttention'] = 'Veda Sparse Attention (MiniMax H3)'
```

`veda_comfy` is **not** installed in `/home/nino/ComfyUI/.venv314/lib/python3.14/site-packages/` — the package lives inside this custom-node directory (version `0.3.1` in `veda_comfy/__init__.py`), so everything below is read from that local source. If the import fails, the console prints `Veda sparse attention unavailable: <error>` and the node is simply absent.

## What it does

Installs learned block-sparse attention for MiniMax-H3. A small trained predictor bundle scores query/key tile pairs before each step and keeps only the tiles worth attending to; a Triton INT8 kernel then computes just those. `execute()` loads the bundle, verifies the model is a `MiniMaxH3Model` whose `(blocks, heads, head_dim)` matches what the predictor was trained on, parses the sparsity/index settings, and returns a cloned model with the patch installed (`comfy_patch.apply`: transformer options + `ON_PREPARE_STATE` / `ON_CLEANUP` callbacks). It is an attention override, so it belongs last on the MODEL wire before the sampler or guider.

## Inputs

| Name | Type | Default | Range / options | Notes |
|---|---|---|---|---|
| `model` | MODEL | — | required | Must wrap a `MiniMaxH3Model` diffusion model; otherwise a bilingual `ValueError` naming the actual class. |
| `predictor` | combo | `predictors.DEFAULT_PREDICTOR` = `minimax_h3_t2va_veda_8nfe_600step_preview_fp8.safetensors` | files in `models/veda` (`.safetensors` only) plus the known released bundles (T2VA and R2VA previews) | Bundles are cached in-process keyed by `(mtime, size)`, and the cache holds one bundle at a time (~0.5 GB host memory). A missing file raises with the Hugging Face URL and target folder; the node never downloads. |
| `generated_sparsity` | STRING (advanced) | `32` | `"NN%"` (0 ≤ NN < 100), a positive integer, or `"trained"` | Key tiles of 128 tokens each query tile keeps. A whole number keeps that many whatever the video size (what two-stage workflows need; a percentage collapses on their small first pass). `"trained"` takes the budget the chosen predictor declares: `90%` for T2VA, `32` tiles for R2VA. `100%` or higher is rejected. |
| `reference_sparsity` | STRING (advanced) | `32` | same grammar | Same budget for references: first/last frames, guide frames, reference images and videos. `"0%"` keeps full attention to and from them. |
| `full_attention_layers` | STRING (advanced) | `""` | 0-based indices and ranges, e.g. `0, 1, 47-49` | DiT blocks that keep full attention. Empty = all sparse. Indices ≥ the model's block count raise, naming the valid `0..N-1` range. Separators: ASCII/full-width commas, semicolons, spaces; dashes `- ~ – —`. |
| `full_attention_steps` | STRING (advanced) | `""` | same grammar | 0-based sampling steps that keep full attention (e.g. `0` for the first step). Empty = all sparse. |
| `verbose` | BOOLEAN (advanced) | `False` | — | Timing and diagnostics on the node, per-backend availability list, predictor description, and which attention forward the sparse layers use. |

Hidden input: `io.Hidden.unique_id` — used to route status text back to this node.

## Outputs

| Name | Type | Notes |
|---|---|---|
| `model` | MODEL | Clone with Veda sparse attention applied. |

## gfx1151 notes

- Backend selection (`veda_comfy/backends/__init__.py`): `candidates()` returns `triton-int8` for any `cuda`-kind device with compute capability ≥ (8, 0), `mlx` for MPS, nothing otherwise. `hardware.describe()` reports HIP devices as `kind='cuda'` with `family` set to the `gcnArchName` (e.g. `gfx1151`), `subtype='rocm'` and `unified_memory=True`, which is what admits ROCm to the Triton INT8 backend. A self-test still gates every device, and a backend that fails to load only costs one line in the report.
- The INT8 kernel launch options are chosen per GPU architecture. README 2.3.11: on the real 480x864 portrait grid (56 heads, 90% sparse) 8 warps / 2 stages measured 17.6 ms vs 35.3 ms for the shipped 4/3 — a 2.0x kernel speedup on gfx1151; README 2.3.12 adds a 23% faster INT8 quantize pass at 2 warps (~3.5% of the whole attention call).
- README performance section (GMTek Evo-X2 Strix Halo, gfx1151, 128 GB unified RAM, ROCm 7.16; MiniMax H3 TURBO int8 + PDMD 4-step LoRA, 480x864, 244 frames, 4 steps): with Veda at 90% sparsity sampler 366.5 s, VAE decode 95.0 s, total prompt 472.6 s, attention computed 26.8% of full; dense baseline sampler 691.6 s, VAE decode 96.1 s, total 1016 s → **1.89x faster sampling, 2.15x faster overall**. README 2.3.8 also quotes 1.47x at 768x768/8 steps and 1.40x at 4-step turbo. The first run after each ComfyUI restart includes the one-time Triton compile.
- The node self-reports on itself via `send_progress_text`: usable backend + device short name, the sparsity plan, any full-attention layers/steps, and how much attention was actually computed.
- Any kernel error falls back to full attention rather than failing the render (`comfy_patch.py`: "never lose the user's render"); the node then shows `Veda done, fell back to full attention` and, in verbose mode, where it failed.
- If another patch already owns the attention (e.g. KJNodes low-VRAM forward), Veda takes it over and warns, naming what it took over; the full-attention layers stay with the previous owner.

## Typical wiring

```
ROCmDiffusionLoader / checkpoint MODEL -> LoRA loaders -> VedaSparseAttention
  -> BasicGuider -> ROCmSamplerCustomAdvanced -> ROCmVAEDecode
```

For MiniMax H3 audio-video, `ROCmH3EasyCache` also sits on the MODEL wire; the shipped templates are `comfyui_workflows/Veda MiniMax H3 T2VA (text to audio-video).json` and `... R2VA (reference to audio-video).json`.

## Troubleshooting

- `Veda accelerates MiniMax-H3 only. Connect the MODEL of a MiniMax-H3 checkpoint (got <Class>)` — wrong model on the wire.
- `This model has A x B x C (blocks x heads x head dim), but the predictor was trained for D x E x F` — predictor/model mismatch; pick the bundle for this model.
- Predictor missing: the error gives the exact filename, the `models/veda` path and the download URL; opening a Veda template lets ComfyUI's missing-model dialog fetch it.
- Status line `... ; using full attention` means no backend resolved for the device (missing Triton, failed self-test). On CUDA the node suggests `pip install triton` (`triton-windows` on Windows).
- Warning that ComfyUI's own "Model Sparse Attention" node is also applied: on H3 it replaces the attention blocks, so Veda would not run — remove one of the two.
- `full_attention_layers: this model has blocks 0-N; <x> do not exist` — index out of range.
- Sparsity strings must be `"trained"`, `"NN%"` (< 100), or a positive integer; anything else raises with the accepted grammar.

"""Calibrates Veda's FP8 kernel against the INT8 attention ComfyUI ships.

    python tools/compare_int8.py

ComfyUI's low-precision attention is SageAttention (`--use-sage-attention`,
`attention_sage` in `comfy/ldm/modules/attention.py`), which quantizes Q and
K to INT8 per block. That is the bar a precision variant has to clear: if
INT8 is good enough to ship as ComfyUI's fast path, FP8 is good enough when
it is at least as close to an fp32 reference.

The problem is deliberately **dense** - every block kept, every slot real -
so our kernel and the baselines compute the same attention and the only
difference left is arithmetic.

Baselines, in order of authority:
  * fp32 reference (`core.reference`) - ground truth.
  * torch SDPA in bf16 - what ComfyUI runs by default.
  * SageAttention, called exactly as ComfyUI calls it, when installed.
  * a SageAttention-v1-style INT8 model in plain torch - per-block
    quantize/dequantize of Q and K, fp32 scores, fp16 PV. Mathematically
    the same arithmetic as the INT8 kernel (one scale pair per block pair
    descales an INT32 accumulator), so it stands in for the real thing on
    a machine that cannot build it.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

import torch  # noqa: E402

from veda_comfy import backends  # noqa: E402
from veda_comfy import hardware  # noqa: E402
from veda_comfy.backends import base  # noqa: E402
from veda_comfy.core import reference  # noqa: E402
from veda_comfy.core import tiling  # noqa: E402

# SageAttention v1's block sizes (BLKQ / BLKK in its kernels).
_Q_BLOCK = 128
_K_BLOCK = 64


def _dense_problem(device, tiles=32, heads=8, head_dim=128, seed=0):
    """A problem with no padding: every one of the 128*tiles slots is real.

    The grid divides the tile shape exactly and the global region is a
    whole number of tiles, so `slot_valid` is all true and a full block
    mask really is dense attention.
    """
    shape = tiling.TileShape(1, 8, 16)
    span = tiling.TiledSpan(256, (tiles - 2, 8, 16), shape)
    seq_len = 256 + (tiles - 2) * tiling.TILE_SIZE
    layout = tiling.build_tile_layout([span], seq_len, device)
    assert bool(layout.slot_valid.all()), 'problem is not padding-free'
    generator = torch.Generator().manual_seed(seed)
    out = []
    for _ in range(3):
        x = torch.randn(seq_len, heads, head_dim, generator=generator)
        out.append(x.to(device=device, dtype=torch.bfloat16))
    mask = torch.ones(heads, layout.n_tiles, layout.n_tiles,
                      dtype=torch.bool, device=device)
    return (*out, mask, layout)


def _quant_dequant(x: torch.Tensor, block: int) -> torch.Tensor:
    """x through INT8 and back, one scale per (block of rows, head)."""
    rows, heads, dim = x.shape
    blocks = x.float().view(rows // block, block, heads, dim)
    scale = blocks.abs().amax(dim=(1, 3), keepdim=True) / 127.0
    scale = scale.clamp(min=torch.finfo(torch.float32).tiny)
    return ((blocks / scale).round().clamp(-127, 127) * scale).view(x.shape)


def _sage_v1_model(q, k, v):
    """SageAttention v1's arithmetic, in plain torch.

    smooth_k is off because that is how ComfyUI calls it.
    """
    qd = _quant_dequant(q, _Q_BLOCK)
    kd = _quant_dequant(k, _K_BLOCK)
    scale = q.shape[-1] ** -0.5
    scores = torch.einsum('qhd,khd->hqk', qd, kd) * scale
    probs = torch.softmax(scores, dim=-1).half()
    return torch.einsum('hqk,khd->qhd', probs, v.half()).to(q.dtype)


def _sage_real(q, k, v):
    """The installed SageAttention, with ComfyUI's own kwargs."""
    from sageattention import sageattn  # noqa: PLC0415
    args = dict(is_causal=False, tensor_layout='NHD', sm_scale=None,
                smooth_k=False)
    return sageattn(q[None], k[None], v[None], **args)[0]


def _sdpa(q, k, v):
    out = torch.nn.functional.scaled_dot_product_attention(
        *(t.transpose(0, 1)[None] for t in (q, k, v)))
    return out[0].transpose(0, 1)


def _int8_block_sparse_model(q, k, v, mask, layout):
    """The INT8 arithmetic, block-sparse, in plain torch.

    Same quantisation as the kernel but with an exact masked softmax, so a
    disagreement is the kernel's sparse walk or its padding mask and not
    quantisation.
    """
    tile = tiling.TILE_SIZE
    scale = q.shape[-1] ** -0.5
    qd = _quant_dequant(q, tile)
    kd = _quant_dequant(k, _K_BLOCK)
    allowed = mask.repeat_interleave(tile, 1).repeat_interleave(tile, 2)
    allowed = allowed & layout.slot_valid.bool()[None, None, :]
    scores = torch.einsum('qhd,khd->hqk', qd, kd) * scale
    probs = torch.softmax(scores.masked_fill(~allowed, float('-inf')),
                          dim=-1).nan_to_num(0.0).half().float()
    out = torch.einsum('hqk,khd->qhd', probs, v.half().float())
    return (out * layout.slot_valid.to(out.dtype)[:, None, None]).to(q.dtype)


def _padding_stage(device, info):
    """Second stage: the self-test problem, which is mostly padding.

    The dense stage above cannot see a padding-mask fault because it has no
    padding. This one isolates it: against the INT8 model the only thing
    left is the kernel's sparse walk.
    """
    print('\n--- the self-test problem (partial tiles, global rows) ---')
    ours = backends._load('triton-int8', info)
    q, k, v, mask, layout = base.selftest_problem(device, torch.bfloat16)
    real = layout.slot_valid.bool()
    exact = reference.block_sparse_attention(q, k, v, mask, layout)[real]
    model = _int8_block_sparse_model(q, k, v, mask, layout)[real].float()
    got = ours.attend(q, k, v, mask, layout)[real].float()
    exact = exact.float()
    scale = exact.abs().max().item()
    for label, x in (('INT8 model vs fp32', model), ('our kernel vs fp32',
                                                     got)):
        print(f'  {label:22s} rel L2 {(x - exact).norm() / exact.norm():7.3%}'
              f'   max err {(x - exact).abs().max().item() / scale:7.3%}')
    delta = (got - model).norm().item() / model.norm().item()
    print(f'  our kernel vs the INT8 model: rel L2 {delta:.3%}')
    print('  -> quantisation, not the sparse walk' if delta < 0.01 else
          '  -> the kernel disagrees with its own arithmetic: a real fault')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--tiles', type=int, default=32)
    parser.add_argument('--heads', type=int, default=8)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit('needs a CUDA GPU')
    device = torch.device('cuda')
    info = hardware.describe(device)
    ours = backends._load('triton-int8', info)

    q, k, v, mask, layout = _dense_problem(device, args.tiles, args.heads)
    tokens = q.shape[0]
    print(f'{info.label}: dense attention, {tokens} tokens x {args.heads} '
          f'heads x {q.shape[2]}')
    want = reference.block_sparse_attention(q, k, v, mask, layout).float()
    denom = want.norm().item()

    rows = [
        ('torch SDPA (bf16)', lambda: _sdpa(q, k, v)),
        ('ComfyUI INT8 model (torch)', lambda: _sage_v1_model(q, k, v)),
    ]
    try:
        import sageattention  # noqa: F401,PLC0415
        rows.append(('SageAttention (installed)', lambda: _sage_real(q, k, v)))
    except ImportError as error:
        print(f'  (SageAttention itself is not installed: {error.name}; '
              'the torch model below stands in for it)')
    rows.append((f'{ours.name} (ours)', lambda: ours.attend(q, k, v, mask,
                                                            layout)))

    results = {}
    print(f'\n{"":28s}  rel L2 vs fp32   max err / absmax')
    for label, run in rows:
        got = run().float()
        results[label] = got
        rel = (got - want).norm().item() / denom
        peak = ((got - want).abs().max().item()
                / want.abs().max().item())
        print(f'  {label:26s}  {rel:12.3%}   {peak:14.3%}')

    model = results['ComfyUI INT8 model (torch)']
    got = results[f'{ours.name} (ours)']
    rel = (got - model).norm().item() / model.norm().item()
    print(f'\nour kernel against the INT8 path directly: rel L2 {rel:.3%}')
    model_err = (model - want).norm().item() / denom
    our_err = (got - want).norm().item() / denom
    verdict = ('we are at least as close to the reference as the INT8 '
               'ComfyUI ships' if our_err <= model_err * 1.05 else
               'we are further from the reference than ComfyUI\'s INT8')
    print(f'{verdict} ({our_err:.3%} against {model_err:.3%}).')
    _padding_stage(device, info)


if __name__ == '__main__':
    main()

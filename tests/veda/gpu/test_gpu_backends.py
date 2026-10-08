"""GPU checks: this GPU's backend must be correct, sparse, and match an
fp32 reference to within its own declared precision.

    pytest tests/gpu -q            # on a CUDA or Apple silicon machine

Skipped without an accelerator, and skipped (never silently passed) when
the kernel's own dependency is missing.
"""

import pytest
import torch

from veda_comfy import backends
from veda_comfy import hardware
from veda_comfy.backends import base
from veda_comfy.core import reference
from veda_comfy.core import selection
from veda_comfy.core import tiling


def _devices():
    if torch.cuda.is_available():
        return [torch.device('cuda', i)
                for i in range(torch.cuda.device_count())]
    if torch.backends.mps.is_available():
        return [torch.device('mps')]
    return []


pytestmark = pytest.mark.skipif(not _devices(), reason='needs a GPU')


def _load(name, device):
    info = hardware.describe(device)
    try:
        return backends._load(name, info)
    except base.BackendUnavailable as error:
        pytest.skip(f'{name} unavailable on {info.label}: {error}')


@pytest.mark.parametrize('device', _devices(), ids=str)
def test_auto_resolution_picks_a_sparse_kernel(device):
    backends.reset()
    resolution = backends.resolve(device)
    assert resolution.backend is not None, resolution.report()
    info = resolution.device
    if info.kind == 'cuda':
        print(f'{info.label}: {resolution.report()}')


def _problem(device, seq_len=6000, heads=4, seed=0):
    """Mid-size problem: target 7x16x24 (2688 rows), reference 1x16x24,
    text / audio as global rows."""
    spans = [tiling.TiledSpan(300, (1, 16, 24), tiling.TileShape(1, 8, 16)),
             tiling.TiledSpan(1000, (7, 16, 24), tiling.TileShape(4, 4, 8))]
    layout = tiling.build_tile_layout(spans, seq_len, device)
    g = torch.Generator().manual_seed(seed)
    x = [torch.randn(seq_len, heads, 128, generator=g).to(device,
                                                          torch.bfloat16)
         for _ in range(3)]
    q, k, v = (tiling.gather_tiles(t, layout, torch.arange(heads,
                                                           device=device))
               for t in x)
    n = layout.n_video_tiles
    scores = torch.randn(heads, n, n, generator=g).to(device)
    blocks = selection.column_blocks(layout, selection.Budget(ratio=0.1),
                                     selection.Budget(ratio=0.2))
    mask = selection.block_mask(*selection.select(scores, layout, blocks),
                                layout)
    return q, k, v, mask, layout


@pytest.mark.parametrize('device', _devices(), ids=str)
def test_backend_matches_reference(device):
    """The device's kernel against the fp32 reference on a real problem."""
    info = hardware.describe(device)
    names = backends.candidates(info)
    if not names:
        pytest.skip(f'no kernel for {info.label}')
    backend = _load(names[0], device)
    base.self_test(backend, device)
    q, k, v, mask, layout = _problem(device)
    out = backend.attend(q, k, v, mask, layout)
    want = reference.block_sparse_attention(q, k, v, mask, layout)
    real = layout.slot_valid.bool()
    scale = max(1.0, want[real].float().abs().max().item())
    err = (out[real].float() - want[real].float()).abs().max().item() / scale
    print(f'{backend.name}: max err {err:.3%} '
          f'(tolerance {backend.tolerance:.1%})')
    assert err < backend.tolerance, f'{backend.name}: max error {err:.3%}'


@pytest.mark.parametrize('device', _devices(), ids=str)
def test_int8_matches_comfyui_s_own_quantisation(device):
    """Our INT8 kernel against a torch model of SageAttention's arithmetic.

    This is the accuracy contract: ComfyUI's low-precision attention is
    SageAttention, so matching that model to within rounding means a user
    who turns Veda on gets the quality ComfyUI would have given them.
    `tools/compare_int8.py` runs the same comparison against the installed
    package when there is one.
    """
    if hardware.describe(device).kind != 'cuda':
        pytest.skip('triton-int8 is CUDA only')
    backend = _load('triton-int8', device)
    q, k, v, mask, layout = _problem(device)
    real = layout.slot_valid.bool()
    tile = tiling.TILE_SIZE
    scale = q.shape[-1] ** -0.5

    def quantised(x, block):
        blocks = x.float().view(x.shape[0] // block, block, *x.shape[1:])
        step = (blocks.abs().amax(dim=(1, 3), keepdim=True) / 127.0).clamp(
            min=torch.finfo(torch.float32).tiny)
        return ((blocks / step).round().clamp(-127, 127) * step).view(x.shape)

    allowed = mask.repeat_interleave(tile, 1).repeat_interleave(tile, 2)
    allowed = allowed & layout.slot_valid.bool()[None, None, :]
    scores = torch.einsum('qhd,khd->hqk', quantised(q, tile),
                          quantised(k, 64)) * scale
    probs = torch.softmax(scores.masked_fill(~allowed, float('-inf')),
                          dim=-1).nan_to_num(0.0).half().float()
    want = torch.einsum('hqk,khd->qhd', probs, v.half().float())[real]
    got = backend.attend(q, k, v, mask, layout)[real].float()
    rel = (got - want).norm().item() / want.norm().item()
    print(f'against SageAttention\'s arithmetic: rel L2 {rel:.3%}')
    assert rel < 0.01, f'INT8 kernel disagrees with its own arithmetic: {rel}'


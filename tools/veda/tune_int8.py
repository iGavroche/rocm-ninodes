"""Measures the INT8 kernel's launch options on this GPU.

    python tools/tune_int8.py --latent-t 102

The kernel ships with one hard-coded configuration because recompiling
inside ComfyUI would cost the user a stall on every new shape. This picks
that configuration: it runs the real problem shape through every variant
and prints the times, so the constant in `sparse_int8.py` is a measurement
and not a guess. Re-run it on a new architecture.
"""

from __future__ import annotations

import argparse
import itertools
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

import torch  # noqa: E402

from veda_comfy.core import selection  # noqa: E402
from veda_comfy.core import tiling  # noqa: E402
from veda_comfy.kernels.sage import sparse_int8  # noqa: E402

sparse_int8.USE_TMA = True  # the sweep decides, not the shipped default

HEADS, DIM = 24, 128


def _problem(device, latent_t, sparsity):
    grid = (latent_t, 24, 42)  # 16:9, as tools/bench_attention.py uses
    shape = tiling.TileShape(1, 8, 16)
    text, audio = 512, 2 * round(latent_t * 17 / 5 / 24 * 40)
    span = tiling.TiledSpan(text + audio, grid, shape)
    seq_len = text + audio + grid[0] * grid[1] * grid[2]
    layout = tiling.build_tile_layout([span], seq_len, device)
    budget = selection.Budget.sparsity(sparsity)
    scores = torch.randn(HEADS, layout.n_video_tiles, layout.n_video_tiles,
                         device=device)
    blocks = selection.column_blocks(layout, budget, budget)
    mask = selection.block_mask(*selection.select(scores, layout, blocks),
                                layout)
    qkv = [torch.randn(layout.num_slots, HEADS, DIM, device=device,
                       dtype=torch.bfloat16) for _ in range(3)]
    return qkv, mask, layout


def _time(fn, repeat=5):
    for _ in range(2):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(repeat):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / repeat * 1e3


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--latent-t', type=int, default=102)
    parser.add_argument('--sparsity', type=float, default=90.0)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit('needs a CUDA GPU')
    device = torch.device('cuda')
    (q, k, v), mask, layout = _problem(device, args.latent_t, args.sparsity)
    index, count = selection.tile_index_list(mask & layout.kv_ok)
    capability = torch.cuda.get_device_capability(device)
    print(f'SM{capability[0]}{capability[1]}: {layout.num_slots} slots x '
          f'{HEADS} heads, {args.sparsity:g}% sparse')
    base = None
    for tma, warps, stages, key in itertools.product(
            (False, True) if hasattr(sparse_int8.tl,
                                     'make_tensor_descriptor')
            and capability[0] >= 9 else (False,),
            (4, 8), (2, 3), (64, 128)):
        sparse_int8.OVERRIDE = dict(tma=tma, num_warps=warps,
                                    num_stages=stages, key_block=key)
        try:
            ms = _time(lambda: sparse_int8.attend(q, k, v, index, count,
                                                  layout.valid_count))
        except Exception as error:
            print(f'  {"tma" if tma else "ptr"} w{warps} s{stages} '
                  f'k{key}: {type(error).__name__}: '
                  f'{str(error).splitlines()[0][:60]}')
            continue
        base = ms if base is None else base
        print(f'  {"tma" if tma else "ptr"} w{warps} s{stages} k{key}: '
              f'{ms:7.1f} ms  ({base / ms:4.2f}x)')
    sparse_int8.OVERRIDE = None


if __name__ == '__main__':
    main()

"""Times one H3 attention layer: full attention vs Veda, on this machine.

    python tools/bench_attention.py                      # 16:9, 5 s
    python tools/bench_attention.py --latent-t 102 --profile

Synthetic q / k / v with H3's real shapes (56 heads x 128, packed text +
audio + target video), a random predictor and the trained plan geometry;
what a step costs depends on the kept-block count, not on the scores, so
this measures the real kernel cost. Prints per-layer milliseconds and the
attention speed-up (excluding the first call, which compiles kernels).
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from veda_comfy import backends  # noqa: E402
from veda_comfy.core import bundle as veda_bundle  # noqa: E402
from veda_comfy.core import engine as veda_engine  # noqa: E402
from veda_comfy.core import h3_layout  # noqa: E402
from veda_comfy.core import plans  # noqa: E402
from veda_comfy.core import selection  # noqa: E402
from veda_comfy.core import tiling  # noqa: E402

HEADS, DIM, LAYERS = 56, 128, 50
CANVAS = {'16:9': (24, 42), '9:16': (42, 24), '1:1': (24, 24),
          '4:3': (24, 32)}


def _sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    elif device.type == 'mps':
        torch.mps.synchronize()


def _time(fn, device, repeat):
    fn()  # warm-up / compile
    _sync(device)
    start = time.perf_counter()
    for _ in range(repeat):
        fn()
    _sync(device)
    return (time.perf_counter() - start) / repeat * 1000.0


def _random_bundle(grid):
    shapes = [tiling.TileShape(4, 4, 8), tiling.TileShape(2, 8, 8)]
    head_shape = [[i % 2 for i in range(HEADS)] for _ in range(LAYERS)]
    table = plans.PlanTable([plans.TilePlan('bench', grid, shapes,
                                            head_shape)])
    weights = [torch.randn(HEADS, 3 * DIM, DIM, dtype=torch.bfloat16) * 1e-2
               for _ in range(LAYERS)]
    return veda_bundle.PredictorBundle('random', LAYERS, HEADS, DIM, 0.1,
                                       table, weights, list(weights), {})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--aspect', default='16:9', choices=sorted(CANVAS))
    parser.add_argument('--latent-t', type=int, default=37)
    parser.add_argument('--sparsity', type=float, default=90.0)
    parser.add_argument('--repeat', type=int, default=5)
    parser.add_argument('--device', default=None)
    parser.add_argument('--profile', action='store_true',
                        help='break the Veda time down by phase (CUDA)')
    args = parser.parse_args()
    device = torch.device(args.device or (
        'cuda' if torch.cuda.is_available() else
        'mps' if torch.backends.mps.is_available() else 'cpu'))
    grid = (args.latent_t, *CANVAS[args.aspect])
    text, audio = 512, 2 * round(args.latent_t * 17 / 5 / 24 * 40)
    video = grid[0] * grid[1] * grid[2]
    seq_len = text + audio + video
    spec = h3_layout.LayoutSpec(seq_len, h3_layout.SpanSpec(
        'target', text + audio, grid), ())
    resolution = backends.resolve(device, notify=lambda t: print('...', t))
    print(f'{resolution.device.label}: {resolution.report()}')
    if resolution.backend is None:
        sys.exit('no sparse backend works here')
    bundle = _random_bundle(grid)
    budget = selection.Budget.sparsity(args.sparsity)
    engine = veda_engine.VedaEngine(bundle, budget, budget,
                                    resolution.backend, device)
    plan = bundle.plans.select(grid).plan
    q, k, v = (torch.randn(seq_len, HEADS, DIM, device=device,
                           dtype=torch.bfloat16) for _ in range(3))
    start = time.perf_counter()
    engine.attention(q, k, v, 0, spec, plan)
    _sync(device)
    first_ms = (time.perf_counter() - start) * 1000.0
    dense_ms = _time(lambda: F.scaled_dot_product_attention(
        q.transpose(0, 1)[None], k.transpose(0, 1)[None],
        v.transpose(0, 1)[None]), device, args.repeat)
    veda_ms = _time(lambda: engine.attention(q, k, v, 0, spec, plan),
                    device, args.repeat)
    print(f'{args.aspect} latent_t {args.latent_t}: {seq_len} tokens, '
          f'{args.sparsity:g}% sparse')
    print(f'  full attention (SDPA): {dense_ms:8.1f} ms / layer')
    print(f'  Veda ({resolution.backend.display}): {veda_ms:8.1f} ms / '
          f'layer -> {dense_ms / veda_ms:.2f}x')
    print(f'  first Veda call (kernel compile): {first_ms / 1000.0:.1f} s')
    print(f'  attention computed: {100 * engine.stats.compute_fraction():.1f}%'
          ' of full attention')
    if args.profile:
        timer = engine.enable_timing()
        if timer is None:
            sys.exit('--profile needs CUDA')
        for _ in range(args.repeat):
            engine.attention(q, k, v, 0, spec, plan)
        phases = timer.summary()
        total = sum(phases.values())
        for name, ms in sorted(phases.items(), key=lambda kv: -kv[1]):
            print(f'    {name:8s} {ms / args.repeat:8.1f} ms  '
                  f'({100 * ms / total:4.1f}%)')


if __name__ == '__main__':
    main()

"""Shared test helpers: synthetic predictor bundles and attention inputs."""

from __future__ import annotations

import json

import torch
from safetensors.torch import save_file

from veda_comfy.core import tiling

FORMAT = 'miowtion-veda-predictor-v1'


def plan_json(name, grid, shapes, num_layers, num_heads, seed=0):
    """A plan using `shapes`, heads split between the first two per layer."""
    generator = torch.Generator().manual_seed(seed)
    head_shape = []
    for _ in range(num_layers):
        if len(shapes) == 1:
            head_shape.append([0] * num_heads)
        else:
            pick = torch.randint(0, 2, (num_heads,), generator=generator)
            head_shape.append([int(x) for x in pick])
    return {'geometry': name, 'grid': list(grid),
            'shapes': [str(s) for s in shapes], 'head_shape': head_shape,
            'provenance': {}}


def write_bundle(path, num_layers, num_heads, plans, head_dim=128,
                 dtype='bfloat16', seed=0, keep_ratio=0.1):
    """Writes a bundle in the release format; returns the fp32 weights."""
    generator = torch.Generator().manual_seed(seed)
    weights, tensors = {}, {}
    for layer in range(num_layers):
        for name in ('proj_q', 'proj_k'):
            w = torch.randn(num_heads, 3 * head_dim, head_dim,
                            generator=generator) * 0.02
            key = f'layers.{layer}.{name}'
            weights[key] = w
            if dtype == 'float8_e4m3fn':
                amax = w.abs().amax(dim=(1, 2))
                tiny = torch.finfo(torch.float32).tiny
                scale = (amax / 448.0).clamp(min=tiny)
                tensors[key] = (w / scale.view(-1, 1, 1)).to(
                    torch.float8_e4m3fn)
                tensors[key + '.__scale'] = scale
            else:
                tensors[key] = w.to(getattr(torch, dtype))
    metadata = {
        'format': FORMAT, 'num_layers': str(num_layers),
        'num_heads': str(num_heads), 'head_dim': str(head_dim),
        'dtype': dtype, 'keep_ratio': repr(keep_ratio), 'source': 'test',
        'source_weights': 'live', 'step': '0',
        'plans': json.dumps({p['geometry']: p for p in plans}),
    }
    save_file(tensors, path, metadata=metadata)
    return weights


def default_plans(num_layers, num_heads):
    """Plans for the small grids the tests use."""
    shapes = [tiling.TileShape(2, 8, 8), tiling.TileShape(4, 4, 8)]
    return [plan_json('16x9_t5', (5, 8, 14), shapes, num_layers, num_heads),
            plan_json('1x1_t5', (5, 8, 8), shapes, num_layers, num_heads, 1)]

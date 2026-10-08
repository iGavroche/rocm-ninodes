"""Predictor bundles: predictor weights plus the tile plans they were
trained against, in one safetensors file (format written by Miowtion's
`scripts/export_predictor.py`).

Weights are held on the host in bf16 whatever the storage dtype: the scoring
arithmetic upcasts the projections to fp32 anyway, and a host copy lets the
predictor follow ComfyUI's model offloading instead of pinning ~0.5 GB of
VRAM for the whole session. fp8 bundles carry a per-head amax scale next to
every tensor (`<name>.__scale`).
"""

from __future__ import annotations

import dataclasses
import json
import os

import torch
from safetensors import safe_open

from . import plans as veda_plans

FORMAT = 'miowtion-veda-predictor-v1'
_SCALE_SUFFIX = '.__scale'
_STORAGE = {'float32': torch.float32, 'bfloat16': torch.bfloat16,
            'float8_e4m3fn': torch.float8_e4m3fn}
# Bundles written before the dtype was recorded are fp32.
_LEGACY_DTYPE = 'float32'


class BundleError(ValueError):
    """The file is not a usable predictor bundle (message is user-facing)."""


@dataclasses.dataclass
class PredictorBundle:
    """A loaded bundle.

    Attributes:
        path: Source file.
        num_layers: Predictor layers (= DiT blocks it scores).
        num_heads: Heads per layer.
        head_dim: Head dimension D.
        keep_ratio: Keep ratio the predictor was trained with.
        plans: Tile plans the predictor was trained against.
        proj_q: Per layer [H, 3D, D] bf16 host tensors.
        proj_k: Per layer [H, 3D, D] bf16 host tensors.
        metadata: Raw string metadata (provenance).
    """

    path: str
    num_layers: int
    num_heads: int
    head_dim: int
    keep_ratio: float
    plans: veda_plans.PlanTable
    proj_q: list[torch.Tensor]
    proj_k: list[torch.Tensor]
    metadata: dict[str, str]

    def describe(self) -> str:
        step = self.metadata.get('step', '?')
        return (f'{os.path.basename(self.path)} · {self.num_layers} layers x '
                f'{self.num_heads} heads · trained keep '
                f'{self.keep_ratio:g} · step {step} · plans: '
                f'{self.plans.summary()}')


def read_metadata(path: str) -> dict[str, str]:
    """The file's metadata without reading tensors.

    Raises:
        BundleError: If the file is not a Veda predictor bundle.
    """
    try:
        with safe_open(path, framework='pt', device='cpu') as f:
            metadata = dict(f.metadata() or {})
    except Exception as error:  # safetensors raises several types
        raise BundleError(f'{os.path.basename(path)} is not a readable '
                          f'safetensors file ({error})') from error
    if metadata.get('format') != FORMAT:
        raise BundleError(
            f'{os.path.basename(path)} is not a Veda predictor bundle '
            f'(format {metadata.get("format")!r}, expected {FORMAT!r}). '
            'Put only Veda predictor files in models/veda.')
    return metadata


def load_bundle(path: str) -> PredictorBundle:
    """Loads a bundle strictly: every tensor must match the declared shape.

    Raises:
        BundleError: On any format or shape mismatch.
    """
    metadata = read_metadata(path)
    try:
        num_layers = int(metadata['num_layers'])
        num_heads = int(metadata['num_heads'])
        head_dim = int(metadata['head_dim'])
        keep_ratio = float(metadata['keep_ratio'])
        plan_json = json.loads(metadata['plans'])
    except (KeyError, ValueError) as error:
        raise BundleError(f'{os.path.basename(path)}: incomplete metadata '
                          f'({error})') from error
    stored = metadata.get('dtype', _LEGACY_DTYPE)
    if stored not in _STORAGE:
        raise BundleError(f'{os.path.basename(path)}: unknown storage dtype '
                          f'{stored!r}')
    with safe_open(path, framework='pt', device='cpu') as f:
        tensors = {k: f.get_tensor(k) for k in f.keys()}
    shape = (num_heads, 3 * head_dim, head_dim)
    proj_q, proj_k = [], []
    for layer in range(num_layers):
        for name, out in (('proj_q', proj_q), ('proj_k', proj_k)):
            key = f'layers.{layer}.{name}'
            value = tensors.pop(key, None)
            if value is None or tuple(value.shape) != shape:
                raise BundleError(
                    f'{os.path.basename(path)}: {key} missing or not '
                    f'{list(shape)}')
            if stored == 'float8_e4m3fn':
                scale = tensors.pop(key + _SCALE_SUFFIX, None)
                if scale is None or tuple(scale.shape) != (num_heads,):
                    raise BundleError(f'{os.path.basename(path)}: {key} has '
                                      'no per-head fp8 scale')
                value = value.float() * scale.float().view(-1, 1, 1)
            out.append(value.to(torch.bfloat16).contiguous())
    if tensors:
        raise BundleError(f'{os.path.basename(path)}: unexpected tensors '
                          f'{sorted(tensors)[:4]}')
    table = veda_plans.PlanTable(
        [veda_plans.TilePlan.from_json(p) for p in plan_json.values()])
    for plan in table.plans.values():
        if plan.num_layers != num_layers or plan.num_heads != num_heads:
            raise BundleError(f'{os.path.basename(path)}: plan {plan.name} '
                              f'is {plan.num_layers}x{plan.num_heads}, '
                              f'predictor is {num_layers}x{num_heads}')
    return PredictorBundle(path=path, num_layers=num_layers,
                           num_heads=num_heads, head_dim=head_dim,
                           keep_ratio=keep_ratio, plans=table, proj_q=proj_q,
                           proj_k=proj_k, metadata=metadata)

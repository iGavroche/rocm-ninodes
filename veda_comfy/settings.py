"""User settings of the Veda node and their parsing.

Every parse error is a ValueError whose message is shown on the node as is,
so it says what was wrong and how to write it instead.
"""

from __future__ import annotations

import dataclasses
import re
from collections.abc import Sequence

from . import status as veda_status
from .core import selection

# Separators users type: ASCII and full-width commas / semicolons, spaces.
_SPLIT = re.compile(r'[\s,;，；、]+')
_RANGE = re.compile(r'^(\d+)\s*[-~–—]\s*(\d+)$')
_PERCENT = re.compile(r'^(\d+(?:\.\d+)?)\s*[%％]$')
_TILES = re.compile(r'^(\d+)$')


def parse_index_list(text: str, what: str) -> frozenset[int]:
    """'0, 1, 47-49' -> {0, 1, 47, 48, 49} (0-based, inclusive ranges).

    Raises:
        ValueError: On anything that is not an index or a range.
    """
    indices: set[int] = set()
    text = (text or '').strip()
    # Normalize spaces around range dashes so '47 - 49' is one token.
    text = re.sub(r'\s*([-~–—])\s*', r'\1', text)
    for token in _SPLIT.split(text):
        if not token:
            continue
        if token.isdigit():
            indices.add(int(token))
            continue
        match = _RANGE.match(token)
        if match is None:
            raise ValueError(veda_status.bilingual(
                f'{what}: cannot read {token!r}. Use 0-based indices and '
                'ranges, e.g. "0, 1, 47-49".',
                f'{what}：看不懂 {token!r}。请写从 0 开始的下标和范围，'
                '例如 "0, 1, 47-49"。'))
        lo, hi = sorted((int(match.group(1)), int(match.group(2))))
        indices.update(range(lo, hi + 1))
    return frozenset(indices)


def parse_sparsity(text: str, what: str) -> selection.Budget:
    """'90%' -> skip 90% of the key tiles; '24' -> keep exactly 24 tiles.

    Raises:
        ValueError: On anything else, or a percentage outside [0, 100).
    """
    value = (text or '').strip()
    match = _PERCENT.match(value)
    if match is not None:
        percent = float(match.group(1))
        if percent >= 100.0:
            raise ValueError(veda_status.bilingual(
                f'{what}: {value} would skip every tile; use a percentage '
                'below 100%, e.g. "90%".',
                f'{what}：{value} 会跳过所有 tile；请写小于 100% 的比例，'
                '例如 "90%"。'))
        return selection.Budget.sparsity(percent)
    match = _TILES.match(value)
    if match is not None and int(match.group(1)) > 0:
        return selection.Budget(tiles=float(match.group(1)))
    raise ValueError(veda_status.bilingual(
        f'{what}: cannot read {value!r}. Write a sparsity percentage such '
        'as "90%" (skip 90% of the tiles), or a whole number such as "24" '
        '(keep 24 tiles of 128 tokens per query tile).',
        f'{what}：看不懂 {value!r}。请写稀疏比例如 "90%"（跳过 90% 的 tile），'
        '或整数如 "24"（每个 query tile 保留 24 个 128-token 的 tile）。'))


def format_index_list(indices: Sequence[int] | frozenset[int]) -> str:
    """{0, 1, 47, 48, 49} -> '0-1, 47-49'."""
    values = sorted(indices)
    parts, start = [], None
    for i, value in enumerate(values):
        if start is None:
            start = value
        if i + 1 == len(values) or values[i + 1] != value + 1:
            parts.append(str(start) if start == value else f'{start}-{value}')
            start = None
    return ', '.join(parts)


def format_budget(budget: selection.Budget) -> str:
    """'90%' or '24 tiles' (the way the user wrote it)."""
    if budget.tiles is not None:
        return f'{budget.tiles:g} tiles'
    return f'{100.0 * (1.0 - min(budget.ratio, 1.0)):g}%'


@dataclasses.dataclass(frozen=True)
class VedaSettings:
    """Everything the node configures.

    Attributes:
        generated: Budget of the generated (target) video's key tiles.
        reference: Budget of the reference / condition key tiles (first /
            last frames, guide frames, reference images and videos);
            keeps_all means they stay in full attention (untiled).
        dense_layers: 0-based DiT blocks that run full attention.
        dense_steps: 0-based sampling steps that run full attention.
        verbose: Show performance diagnostics and log every decision.
    """

    generated: selection.Budget
    reference: selection.Budget
    dense_layers: frozenset[int] = frozenset()
    dense_steps: frozenset[int] = frozenset()
    verbose: bool = False

    def describe(self) -> str:
        return (f'generated {format_budget(self.generated)} · reference '
                f'{format_budget(self.reference)}')

    def describe_full_attention(self) -> str | None:
        parts = []
        if self.dense_layers:
            parts.append(f'layers {format_index_list(self.dense_layers)}')
        if self.dense_steps:
            parts.append(f'steps {format_index_list(self.dense_steps)}')
        return ' · '.join(parts) or None


def step_index(sigma: float, sample_sigmas: Sequence[float]) -> int | None:
    """0-based sampling step whose interval contains sigma.

    Step i integrates from sample_sigmas[i] down to sample_sigmas[i + 1];
    multi-stage samplers evaluate the model inside that interval, which
    still counts as step i.
    """
    sigmas = [float(s) for s in sample_sigmas]
    if len(sigmas) < 2:
        return None
    tolerance = 1e-5 * max(1.0, abs(sigma))
    for i in range(len(sigmas) - 1):
        high, low = sigmas[i], sigmas[i + 1]
        if high + tolerance >= sigma > low + tolerance:
            return i
    if sigma >= sigmas[0]:
        return 0
    return len(sigmas) - 2 if sigma >= sigmas[-1] - tolerance else None

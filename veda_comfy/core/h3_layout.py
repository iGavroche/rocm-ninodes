"""Reads ComfyUI's MiniMax-H3 packed layout into Veda spans.

ComfyUI's `comfy.ldm.minimax.model.PackedLayout` (published to attention
patches as `transformer_options["minimax_h3_layout"]`) packs

    text | keyframes (cond) | references | target audio | target video

with contiguous `segments` [(start, stop, kind)], the same order as
Miowtion's training layout. Veda needs, per video-like span, its token grid
(T, H, W) and the guarantee that its rows are T, H, W row-major:

  * target: the `video` segment; its grid is the latent grid after the 1x2x2
    patch, read from `layout.signature`.
  * reference: `cond` (FL2VA keyframes / AddGuide frames) and `ref_img`
    (R2VA reference images and videos). Their grids are not in the signature, so
    they are recovered from `layout.position_ids` and the row order is
    verified against them; a segment that does not verify stays global
    (dense), which is always correct, just slower.

Text and every audio segment are global rows. Duck-typed on purpose: this
module must not import ComfyUI.
"""

from __future__ import annotations

import dataclasses

import torch

_REFERENCE_KINDS = ('cond', 'ref_img')


@dataclasses.dataclass(frozen=True)
class SpanSpec:
    """A video-like span: rows [start, start + T*H*W), T/H/W row-major."""

    kind: str
    start: int
    grid: tuple[int, int, int]


@dataclasses.dataclass(frozen=True)
class LayoutSpec:
    """What Veda needs from one packed layout (hashable cache key).

    Attributes:
        seq_len: Packed sequence length S.
        target: The target video span.
        references: Condition spans that can be tiled, in packed order.
        skipped: Condition segments that stay global, with the reason.
    """

    seq_len: int
    target: SpanSpec
    references: tuple[SpanSpec, ...]
    skipped: tuple[str, ...] = ()


class LayoutError(ValueError):
    """The layout cannot be mapped; the caller runs dense attention."""


def _segment_grid(position_ids: torch.Tensor, start: int,
                  stop: int) -> tuple[int, int, int] | None:
    """(T, H, W) of a segment if its rows are T, H, W row-major, else None.

    Positions are (t, h, w) floats; a row-major grid has exactly
    T * H * W rows whose positions are the outer product of the sorted
    unique values per axis.
    """
    pos = position_ids[start:stop]
    axes = [torch.unique(pos[:, i]) for i in range(3)]
    t, h, w = (a.numel() for a in axes)
    if t * h * w != stop - start:
        return None
    expected = torch.stack(torch.meshgrid(*axes, indexing='ij'),
                           dim=-1).reshape(-1, 3)
    if not torch.equal(expected, pos.to(expected.dtype)):
        return None
    return t, h, w


def describe(layout) -> LayoutSpec:
    """Maps a ComfyUI H3 PackedLayout to a LayoutSpec.

    Raises:
        LayoutError: If the layout has no target video segment or its size
            does not match the latent grid.
    """
    try:
        segments = list(layout.segments)
        _, latent_t, latent_h, latent_w, _ = layout.signature
        seq_len = int(layout.seq_len)
    except (AttributeError, TypeError, ValueError) as error:
        raise LayoutError(
            f'not a MiniMax-H3 packed layout ({error})') from error
    video = [(a, b) for a, b, kind in segments if kind == 'video']
    if len(video) != 1:
        raise LayoutError(f'expected one target video segment, '
                          f'found {len(video)}')
    a, b = video[0]
    grid = (int(latent_t), int(latent_h) // 2, int(latent_w) // 2)
    if grid[0] * grid[1] * grid[2] != b - a:
        raise LayoutError(f'video segment of {b - a} rows does not match the '
                          f'latent grid {grid}')
    references, skipped = [], []
    position_ids = getattr(layout, 'position_ids', None)
    for start, stop, kind in segments:
        if kind not in _REFERENCE_KINDS:
            continue
        span_grid = (None if position_ids is None
                     else _segment_grid(position_ids, start, stop))
        if span_grid is None:
            skipped.append(f'{kind}[{start}:{stop}] has no row-major grid')
            continue
        references.append(SpanSpec(kind, int(start), span_grid))
    return LayoutSpec(seq_len=seq_len, target=SpanSpec('target', int(a), grid),
                      references=tuple(references), skipped=tuple(skipped))

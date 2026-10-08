"""Exact block-sparse attention in fp32: the ground truth for every backend.

Token level and O(N^2) memory, so only for tests and the small backend
self-tests; never on a real sequence.
"""

from __future__ import annotations

import torch

from . import tiling


def block_sparse_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                           block_mask: torch.Tensor,
                           layout: tiling.TileLayout) -> torch.Tensor:
    """Masked softmax attention on tile-ordered tensors.

    Args:
        q, k, v: [N, H', D] tile order, N = n_tiles * 128.
        block_mask: [H', n_tiles, n_tiles] bool, rows are query tiles.
        layout: Tile layout (key slot validity).

    Returns:
        [N, H', D] in q's dtype; rows of padding slots are 0.
    """
    tile = tiling.TILE_SIZE
    scale = q.shape[-1] ** -0.5
    allowed = block_mask.repeat_interleave(tile, 1).repeat_interleave(tile, 2)
    allowed = allowed & layout.slot_valid.bool()[None, None, :]
    scores = torch.einsum('qhd,khd->hqk', q.float(), k.float()) * scale
    scores = scores.masked_fill(~allowed, float('-inf'))
    probs = torch.softmax(scores, dim=-1).nan_to_num(0.0)
    out = torch.einsum('hqk,khd->qhd', probs, v.float())
    out = out * layout.slot_valid.to(out.dtype)[:, None, None]
    return out.to(q.dtype)


def dense_attention(q: torch.Tensor, k: torch.Tensor,
                    v: torch.Tensor) -> torch.Tensor:
    """Plain softmax attention, [S, H, D] -> [S, H, D] (fp32 math)."""
    scale = q.shape[-1] ** -0.5
    scores = torch.einsum('qhd,khd->hqk', q.float(), k.float()) * scale
    out = torch.einsum('hqk,khd->qhd', torch.softmax(scores, -1), v.float())
    return out.to(q.dtype)

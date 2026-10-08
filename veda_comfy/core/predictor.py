"""Tile scores: which key tiles each query tile should attend.

Per tile, q and k are pooled into [mean | max | min] (3D features, the
paper's TripPool). Per layer and head a residual projection P maps them to
D dims, `x_hat = pool @ P[h] + mean`, and block logits are
`q_hat . k_hat / sqrt(D)` in fp32. Same arithmetic as Miowtion's
`miowtion/veda/predictor.py`; inference only, so everything runs without
autograd.
"""

from __future__ import annotations

import math

import torch

from . import tiling


@torch.no_grad()
def pool_video_tiles(x: torch.Tensor,
                     layout: tiling.TileLayout) -> torch.Tensor:
    """Masked mean / max / min over the real rows of every video tile.

    Global tiles are never scored (they are always attended), so they are
    not pooled.

    Args:
        x: [N, H', D] tile-ordered rows, padding slots zero (gather_tiles).
        layout: Its tile layout.

    Returns:
        [H', n_video_tiles, 3D] fp32 features; exactly 0 for empty tiles.
    """
    n_tiles = layout.n_video_tiles
    heads, dim = x.shape[1], x.shape[2]
    tiles = x[:n_tiles * tiling.TILE_SIZE].view(n_tiles, tiling.TILE_SIZE,
                                                heads, dim)
    count = layout.valid_count[:n_tiles].clamp(min=1).to(torch.float32)
    # Sum with fp32 accumulation instead of upcasting the whole tensor.
    mean = tiles.sum(dim=1, dtype=torch.float32) / count[:, None, None]
    # One pass for both extremes (~1/3 less pooling traffic than amin +
    # amax on an RTX 5070).
    tmin, tmax = torch.aminmax(tiles, dim=1)
    # Padding rows are 0, a legal value, so max/min are recomputed with
    # masking for the partial tiles only.
    partial = layout.partial_video_tiles
    if partial.numel():
        sub = tiles.index_select(0, partial)
        valid = (torch.arange(tiling.TILE_SIZE, device=x.device)[None, :]
                 < layout.valid_count.index_select(0, partial)[:, None])
        valid = valid[:, :, None, None]
        tmax.index_copy_(0, partial,
                         sub.masked_fill(~valid, float('-inf')).amax(dim=1))
        tmin.index_copy_(0, partial,
                         sub.masked_fill(~valid, float('inf')).amin(dim=1))
    feats = torch.cat([mean, tmax.float(), tmin.float()], dim=-1)
    # where, not multiply: -inf * 0 would be NaN for empty tiles.
    feats = torch.where(layout.kv_ok[:n_tiles, None, None], feats, 0.0)
    return feats.permute(1, 0, 2).contiguous()


@torch.no_grad()
def tile_logits(feats_q: torch.Tensor, feats_k: torch.Tensor,
                proj_q: torch.Tensor, proj_k: torch.Tensor) -> torch.Tensor:
    """Block logits.

    Args:
        feats_q: [H', n_q, 3D] fp32 pooled query features.
        feats_k: [H', n_k, 3D] fp32 pooled key features.
        proj_q: [H', 3D, D] projections of these heads (any float dtype).
        proj_k: [H', 3D, D].

    Returns:
        [H', n_q, n_k] fp32 logits.
    """
    dim = proj_q.shape[-1]
    q_hat = torch.bmm(feats_q, proj_q.float()) + feats_q[..., :dim]
    k_hat = torch.bmm(feats_k, proj_k.float()) + feats_k[..., :dim]
    return torch.bmm(q_hat, k_hat.transpose(1, 2)) / math.sqrt(dim)

import math

import pytest
import torch

from veda_comfy.core import selection
from veda_comfy.core import tiling


def _layout():
    # reference: one 1x8x16 span; target 5x8x8 tiled 2x8x8 -> 3 tiles
    spans = [tiling.TiledSpan(20, (1, 8, 16), tiling.TileShape(1, 8, 16)),
             tiling.TiledSpan(200, (5, 8, 8), tiling.TileShape(2, 8, 8))]
    return tiling.build_tile_layout(spans, 200 + 320 + 10)


def test_budget_sparsity():
    assert selection.Budget.sparsity(90.0).ratio == pytest.approx(0.1)
    assert selection.Budget.sparsity(0.0).keeps_all
    with pytest.raises(ValueError):
        selection.Budget.sparsity(100.0)


def test_equal_cost_budget():
    budget = selection.Budget(ratio=0.1)
    # 320 real tokens -> n_ideal 3; padded to 4 columns -> 0.1 * 9 / 4
    assert budget.per_row(320, 4) == pytest.approx(0.225)
    assert selection.Budget(tiles=5).per_row(320, 4) == 5


def test_split_budget_and_bresenham_mean():
    assert selection.split_budget(2.3, 10) == (2, 3, 0.3)
    assert selection.split_budget(12, 10) == (10, 10, 0.0)
    assert selection.split_budget(0.2, 10)[0] == 1
    extra = selection.bresenham_extra(1000, 0.3, 'cpu')
    assert int(extra.sum()) == 300


def test_select_forces_diagonal_and_respects_blocks():
    layout = _layout()
    n_video = layout.n_video_tiles
    assert layout.n_ref_tiles == 1 and n_video == 4
    scores = torch.randn(2, n_video, n_video)
    scores[:, :, 0] = -1e9  # reference scores worst; its own row keeps it
    blocks = selection.column_blocks(layout, selection.Budget(tiles=1),
                                     selection.Budget(tiles=1))
    index, keep = selection.select(scores, layout, blocks)
    mask = selection.block_mask(index, keep, layout)
    video = mask[:, :n_video, :n_video]
    assert video.diagonal(dim1=1, dim2=2).all()
    # one reference column and one target column per row (budget 1 each)
    assert (video[:, :, :1].sum(-1) == 1).all()
    assert (video[:, :, 1:].sum(-1) == 1).all()
    # global rows / columns always on
    assert mask[:, n_video:, :].all()
    assert mask[:, :, n_video:].all()


def test_keep_all_block_keeps_every_nonempty_tile():
    layout = _layout()
    scores = torch.randn(1, 4, 4)
    blocks = selection.column_blocks(layout, selection.Budget(ratio=1.0),
                                     selection.Budget(ratio=1.0))
    index, keep = selection.select(scores, layout, blocks)
    mask = selection.block_mask(index, keep, layout)
    assert torch.equal(mask[0], layout.kv_ok.expand(layout.n_tiles, -1))


def test_kept_count_matches_fractional_budget():
    spans = [tiling.TiledSpan(0, (16, 16, 16), tiling.TileShape(4, 4, 8))]
    layout = tiling.build_tile_layout(spans, 4096)
    n = layout.n_video_tiles
    budget = selection.Budget(ratio=0.1)
    blocks = selection.column_blocks(layout, budget, budget)
    index, keep = selection.select(torch.randn(3, n, n), layout, blocks)
    expected = budget.per_row(4096, n)
    assert keep.sum(-1).float().mean().item() == pytest.approx(
        expected, abs=1.0 / n)
    assert math.floor(expected) <= keep.sum(-1).min()

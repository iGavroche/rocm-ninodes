import pytest
import torch

from veda_comfy.core import tiling


def test_all_shapes_hold_one_tile():
    shapes = tiling.all_shapes()
    assert len(shapes) == 36
    assert all(s.t * s.h * s.w == tiling.TILE_SIZE for s in shapes)


def test_least_padding_shape_prefers_exact_then_cubic():
    assert tiling.least_padding_shape((1, 8, 16)) == tiling.TileShape(1, 8, 16)
    assert tiling.least_padding_shape((8, 8, 8)).num_tiles((8, 8, 8)) == 4
    shape = tiling.least_padding_shape((4, 4, 4))  # smaller than one tile
    assert shape.t <= 4 or shape.h <= 4 or shape.w <= 4


def test_span_tiles_order_and_prefix():
    span = tiling.TiledSpan(10, (3, 4, 5), tiling.TileShape(2, 4, 16))
    tiles = tiling.span_tiles(span)
    # every real row exactly once, padding is a suffix of each tile
    real = tiles[tiles >= 0]
    assert torch.equal(real.sort().values, torch.arange(10, 10 + 60))
    valid = tiles >= 0
    assert torch.equal(valid, valid.sort(dim=1, descending=True).values)
    # first tile: t 0-1, h 0-3, w 0-4 in t, h, w row-major
    expected = torch.tensor([10 + t * 20 + h * 5 + w
                             for t in range(2) for h in range(4)
                             for w in range(5)])
    assert torch.equal(tiles[0, :40], expected)


def test_build_tile_layout_globals_after_spans():
    spans = [tiling.TiledSpan(5, (1, 8, 16), tiling.TileShape(1, 8, 16)),
             tiling.TiledSpan(140, (2, 8, 8), tiling.TileShape(2, 8, 8))]
    layout = tiling.build_tile_layout(spans, 300)
    assert layout.n_ref_tiles == 1
    assert layout.n_video_tiles == 2
    assert layout.target_tokens == 128 and layout.ref_tokens == 128
    globals_ = layout.perm[layout.n_video_tiles * 128:]
    expected = torch.cat([torch.arange(0, 5), torch.arange(133, 140),
                          torch.arange(268, 300)])
    assert torch.equal(globals_[globals_ >= 0], expected)
    assert layout.valid_count.tolist() == [128, 128, 44]
    assert layout.partial_tiles.tolist() == [2]
    assert layout.partial_video_tiles.numel() == 0


def test_build_tile_layout_rejects_overlap():
    spans = [tiling.TiledSpan(0, (1, 8, 16), tiling.TileShape(1, 8, 16)),
             tiling.TiledSpan(100, (1, 8, 16), tiling.TileShape(1, 8, 16))]
    with pytest.raises(ValueError):
        tiling.build_tile_layout(spans, 400)


def test_gather_scatter_roundtrip_is_exact():
    spans = [tiling.TiledSpan(7, (3, 6, 10), tiling.TileShape(2, 4, 16))]
    layout = tiling.build_tile_layout(spans, 7 + 180 + 30)
    x = torch.randn(layout.seq_len, 3, 128)
    heads = torch.tensor([0, 2])
    tiled = tiling.gather_tiles(x, layout, heads)
    assert tiled.shape == (layout.num_slots, 2, 128)
    assert torch.equal(tiled[layout.pad_slots], torch.zeros_like(
        tiled[layout.pad_slots]))
    out = torch.full((layout.seq_len + 1, 3, 128), float('nan'))
    tiling.scatter_tiles_(out, tiled, layout, heads)
    assert torch.equal(out[:-1, [0, 2]], x[:, [0, 2]])
    assert torch.isnan(out[:-1, 1]).all()

import pytest
import torch

from reference_backend import ReferenceBackend
from veda_comfy.core import engine as veda_engine
from veda_comfy.core import h3_layout
from veda_comfy.core import predictor
from veda_comfy.core import reference
from veda_comfy.core import selection
from veda_comfy.core import tiling
import veda_testing


def test_pool_ignores_padding_rows():
    spans = [tiling.TiledSpan(0, (3, 4, 5), tiling.TileShape(2, 4, 16))]
    layout = tiling.build_tile_layout(spans, 60 + 9)
    x = torch.randn(layout.seq_len, 2, 128) - 5.0  # all negative
    tiled = tiling.gather_tiles(x, layout, torch.arange(2))
    feats = predictor.pool_video_tiles(tiled, layout)
    assert feats.shape == (2, layout.n_video_tiles, 384)
    for tile in range(layout.n_video_tiles):
        rows = layout.perm[tile * 128:(tile + 1) * 128]
        real = x[rows[rows >= 0]]
        torch.testing.assert_close(feats[:, tile, :128], real.mean(0))
        # max of real rows is negative: the zero padding must not leak in
        torch.testing.assert_close(feats[:, tile, 128:256], real.amax(0))
        torch.testing.assert_close(feats[:, tile, 256:], real.amin(0))


def test_logits_untrained_equal_mean_pooled_qk():
    feats_q = torch.randn(2, 3, 384)
    feats_k = torch.randn(2, 5, 384)
    zero = torch.zeros(2, 384, 128)
    logits = predictor.tile_logits(feats_q, feats_k, zero, zero)
    want = feats_q[..., :128] @ feats_k[..., :128].transpose(1, 2) / 128**0.5
    torch.testing.assert_close(logits, want)


def _spec(references=True):
    spans = ([h3_layout.SpanSpec('ref_img', 30, (1, 8, 16))]
             if references else [])
    return h3_layout.LayoutSpec(
        seq_len=30 + 128 + 40 + 5 * 8 * 14 + 3,
        target=h3_layout.SpanSpec('target', 198, (5, 8, 14)),
        references=tuple(spans))


@pytest.fixture
def bundle(tmp_path):
    from veda_comfy.core import bundle as veda_bundle
    path = str(tmp_path / 'p.safetensors')
    veda_testing.write_bundle(path, 2, 4, veda_testing.default_plans(2, 4))
    return veda_bundle.load_bundle(path)


def _qkv(seq_len, heads=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(seq_len, heads, 128, generator=g) for _ in range(3)]


@pytest.mark.parametrize('references', [True, False])
def test_engine_keep_all_equals_dense(bundle, references):
    spec = _spec(references)
    keep_all = selection.Budget(tiles=10**6)
    engine = veda_engine.VedaEngine(bundle, keep_all, keep_all,
                                    ReferenceBackend(),
                                    torch.device('cpu'))
    q, k, v = _qkv(spec.seq_len)
    plan = engine.plan_for(spec).plan
    out = engine.attention(q, k, v, 1, spec, plan)
    torch.testing.assert_close(out, reference.dense_attention(q, k, v),
                               rtol=1e-4, atol=1e-4)


def test_engine_matches_reference_per_head_group(bundle, monkeypatch):
    """Chunked, grouped engine output == per-group reference attention."""
    spec = _spec(True)
    budget = selection.Budget(ratio=0.3)
    engine = veda_engine.VedaEngine(bundle, budget, budget,
                                    ReferenceBackend(),
                                    torch.device('cpu'))
    monkeypatch.setattr(engine, '_chunk_bytes', lambda: 1)  # 1 head/chunk
    q, k, v = _qkv(spec.seq_len)
    plan = engine.plan_for(spec).plan
    out = engine.attention(q, k, v, 0, spec, plan)
    want = torch.empty_like(out)
    for group in plan.head_groups(0, 'cpu'):
        layout = engine._tile_layout(spec, group.shape)
        blocks = selection.column_blocks(layout, budget, budget)
        for head in group.heads:
            heads = head.view(1)
            q_t, k_t, v_t = (tiling.gather_tiles(t, layout, heads)
                             for t in (q, k, v))
            scores = predictor.tile_logits(
                predictor.pool_video_tiles(q_t, layout),
                predictor.pool_video_tiles(k_t, layout),
                bundle.proj_q[0][heads], bundle.proj_k[0][heads])
            mask = selection.block_mask(*selection.select(scores, layout,
                                                          blocks), layout)
            o = reference.block_sparse_attention(q_t, k_t, v_t, mask, layout)
            buf = torch.empty(spec.seq_len + 1, 4, 128)
            tiling.scatter_tiles_(buf, o, layout, heads)
            want[:, head] = buf[:-1, head]
    torch.testing.assert_close(out, want, rtol=1e-4, atol=1e-4)
    assert engine.stats.kept_fraction() < 0.9
    assert 0.0 < engine.stats.compute_fraction() < 1.0


@pytest.mark.parametrize('head_chunks', [1, 2, 4])
def test_head_chunks_bound_the_chunking_without_changing_the_result(
        bundle, monkeypatch, head_chunks):
    """KJNodes' Low VRAM node asks for N head groups to lower peak memory.
    Veda's own chunking is otherwise bounded only by free memory, so the
    request has to reach it; heads are independent, so the result is the
    same whatever the split."""
    spec = _spec(True)
    budget = selection.Budget(ratio=0.3)
    engine = veda_engine.VedaEngine(bundle, budget, budget,
                                    ReferenceBackend(),
                                    torch.device('cpu'))
    # Room for every head at once, so only head_chunks can split it.
    monkeypatch.setattr(engine, '_chunk_bytes', lambda: 1 << 40)
    q, k, v = _qkv(spec.seq_len)
    plan = engine.plan_for(spec).plan
    unbounded = engine.attention(q, k, v, 0, spec, plan)
    widest = engine.chunking['heads_per_chunk']
    out = engine.attention(q, k, v, 0, spec, plan, head_chunks=head_chunks)
    cap = -(-q.shape[1] // head_chunks)  # what the node asked for
    assert engine.chunking['heads_per_chunk'] == min(widest, cap)
    torch.testing.assert_close(out, unbounded, rtol=0, atol=0)

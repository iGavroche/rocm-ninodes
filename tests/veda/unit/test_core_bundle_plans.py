import pytest
import torch
from safetensors.torch import save_file

import veda_testing
from veda_comfy.core import bundle as veda_bundle
from veda_comfy.core import plans
from veda_comfy.core import selection
from veda_comfy.core import tiling


@pytest.mark.parametrize('dtype', ['bfloat16', 'float8_e4m3fn', 'float32'])
def test_load_bundle_dequantizes(tmp_path, dtype):
    path = str(tmp_path / 'p.safetensors')
    weights = veda_testing.write_bundle(path, 3, 4,
                                        veda_testing.default_plans(3, 4),
                                        dtype=dtype)
    b = veda_bundle.load_bundle(path)
    assert (b.num_layers, b.num_heads, b.head_dim) == (3, 4, 128)
    assert b.generated == selection.Budget(ratio=0.1)
    tolerance = {'bfloat16': 1e-2, 'float32': 1e-2, 'float8_e4m3fn': 7e-2}
    for layer in range(3):
        got = b.proj_q[layer].float()
        want = weights[f'layers.{layer}.proj_q']
        assert got.dtype == torch.float32 and b.proj_q[layer].dtype == (
            torch.bfloat16)
        rel = (got - want).abs().max() / want.abs().max()
        assert rel < tolerance[dtype]
    assert len(b.plans.plans) == 2


def test_rejects_foreign_safetensors(tmp_path):
    path = str(tmp_path / 'lora.safetensors')
    save_file({'x': torch.zeros(2)}, path, metadata={'format': 'pt'})
    with pytest.raises(veda_bundle.BundleError, match='not a Veda predictor'):
        veda_bundle.load_bundle(path)


def test_rejects_shape_mismatch(tmp_path):
    path = str(tmp_path / 'p.safetensors')
    veda_testing.write_bundle(path, 2, 4, veda_testing.default_plans(3, 4))
    with pytest.raises(veda_bundle.BundleError, match='plan'):
        veda_bundle.load_bundle(path)


def _table():
    shapes = [tiling.TileShape(4, 4, 8), tiling.TileShape(2, 8, 8)]
    head_shape = [[0, 1]]
    return plans.PlanTable([
        plans.TilePlan('16x9_t37', (37, 24, 42), shapes, head_shape),
        plans.TilePlan('16x9_t72', (72, 24, 42), shapes, head_shape),
        plans.TilePlan('1x1_t37', (37, 24, 24), shapes, head_shape),
    ])


def test_plan_selection_by_aspect_then_duration():
    table = _table()
    exact = table.select((37, 24, 42))
    assert exact.exact and exact.plan.name == '16x9_t37'
    assert 'trained for this size' in exact.how
    longer = table.select((62, 24, 42))  # nearer 72 than 37
    assert not longer.exact and longer.plan.name == '16x9_t72'
    assert 'nearest trained size: 1344x768 · 10.1 s' in longer.how
    smaller = table.select((37, 15, 27))  # 864x480, still 16:9
    assert not smaller.exact and smaller.plan.name == '16x9_t37'
    square = table.select((72, 32, 32))
    assert square.plan.name == '1x1_t37'
    portrait = table.select((37, 42, 24))  # only the transpose fits
    assert portrait.plan.name == '16x9_t37_T'
    assert portrait.plan.shapes[0] == tiling.TileShape(4, 8, 4)


def test_head_groups_partition_heads():
    plan = _table().plans['16x9_t37']
    groups = plan.head_groups(0, 'cpu')
    assert [g.shape for g in groups] == [tiling.TileShape(4, 4, 8),
                                         tiling.TileShape(2, 8, 8)]
    assert sorted(torch.cat([g.heads for g in groups]).tolist()) == [0, 1]


def test_summary():
    assert _table().summary() == '16:9, 1:1 x 5.2 / 10.1 s'


@pytest.mark.parametrize('declared,generated,reference', [
    # Bundles before the R2VA release state one keep_ratio for both.
    ({'keep_ratio': '0.1'},
     selection.Budget(ratio=0.1), selection.Budget(ratio=0.1)),
    # R2VA states a kind and a value per stream.
    ({'target_budget_kind': 'tiles', 'target_budget_value': '32.0',
      'ref_budget_kind': 'tiles', 'ref_budget_value': '32.0'},
     selection.Budget(tiles=32.0), selection.Budget(tiles=32.0)),
    # A reference budget may differ from the generated one.
    ({'target_budget_kind': 'tiles', 'target_budget_value': '32.0',
      'ref_budget_kind': 'ratio', 'ref_budget_value': '0.25'},
     selection.Budget(tiles=32.0), selection.Budget(ratio=0.25)),
])
def test_both_budget_schemas_load(tmp_path, declared, generated, reference):
    path = str(tmp_path / 'p.safetensors')
    veda_testing.write_bundle(path, 2, 2, veda_testing.default_plans(2, 2))
    _rewrite_metadata(path, declared, drop=('keep_ratio',))
    loaded = veda_bundle.load_bundle(path)
    assert loaded.generated == generated
    assert loaded.reference == reference


def test_a_bundle_that_declares_no_budget_is_rejected(tmp_path):
    path = str(tmp_path / 'p.safetensors')
    veda_testing.write_bundle(path, 2, 2, veda_testing.default_plans(2, 2))
    _rewrite_metadata(path, {}, drop=('keep_ratio',))
    with pytest.raises(veda_bundle.BundleError, match='incomplete metadata'):
        veda_bundle.load_bundle(path)


def test_an_unknown_budget_kind_says_what_is_allowed(tmp_path):
    path = str(tmp_path / 'p.safetensors')
    veda_testing.write_bundle(path, 2, 2, veda_testing.default_plans(2, 2))
    _rewrite_metadata(path, {'target_budget_kind': 'quantile',
                             'target_budget_value': '0.5'},
                      drop=('keep_ratio',))
    with pytest.raises(veda_bundle.BundleError, match="'ratio' or 'tiles'"):
        veda_bundle.load_bundle(path)


def _rewrite_metadata(path, add, drop=()):
    """Rewrites a bundle's metadata in place, keeping its tensors."""
    from safetensors import safe_open
    with safe_open(path, framework='pt', device='cpu') as f:
        metadata = dict(f.metadata() or {})
        tensors = {k: f.get_tensor(k) for k in f.keys()}
    for key in drop:
        metadata.pop(key, None)
    metadata.update(add)
    save_file(tensors, path, metadata=metadata)

import dataclasses
import importlib.util

import pytest
import torch

from veda_comfy import backends
from veda_comfy import hardware
from veda_comfy import settings
from veda_comfy.backends import base
from reference_backend import ReferenceBackend
from veda_comfy.core import reference


@pytest.mark.parametrize('text,want', [
    ('', set()), ('0, 1, 47-49', {0, 1, 47, 48, 49}),
    ('3；5，7 9', {3, 5, 7, 9}), ('10 - 12', {10, 11, 12}),
    ('5-3', {3, 4, 5}), ('0~2', {0, 1, 2})])
def test_parse_index_list(text, want):
    assert settings.parse_index_list(text, 'x') == want


@pytest.mark.parametrize('text,ratio,tiles', [
    ('90%', 0.1, None), (' 87.5 % ', 0.125, None), ('0%', 1.0, None),
    ('90％', 0.1, None), ('24', None, 24.0)])
def test_parse_sparsity(text, ratio, tiles):
    budget = settings.parse_sparsity(text, 'x')
    assert budget.tiles == tiles
    if ratio is not None:
        assert budget.ratio == pytest.approx(ratio)


@pytest.mark.parametrize('text', ['100%', '0', '0.9', 'ninety', '', '-5%'])
def test_parse_sparsity_explains_errors(text):
    with pytest.raises(ValueError, match='generated_sparsity'):
        settings.parse_sparsity(text, 'generated_sparsity')


def test_format_budget():
    assert settings.format_budget(settings.parse_sparsity('90%', 'x')) == (
        '90%')
    assert settings.format_budget(settings.parse_sparsity('24', 'x')) == (
        '24 tiles')


def test_parse_index_list_explains_errors():
    with pytest.raises(ValueError, match='0-based'):
        settings.parse_index_list('first', 'full_attention_steps')


def test_format_index_list():
    assert settings.format_index_list({0, 1, 47, 48, 49, 7}) == '0-1, 7, 47-49'


def test_step_index():
    sigmas = [1.0, 0.75, 0.5, 0.25, 0.0]
    assert settings.step_index(1.0, sigmas) == 0
    assert settings.step_index(0.75, sigmas) == 1
    assert settings.step_index(0.6, sigmas) == 1  # 2nd-order midpoint
    assert settings.step_index(0.25, sigmas) == 3
    assert settings.step_index(1.2, sigmas) == 0


def _info(cc, name='NVIDIA GeForce RTX 4090', kind='cuda'):
    base_info = hardware.DeviceInfo(
        kind, 0, name, cc, hardware._FAMILIES.get(cc, 'x'),
        hardware._cuda_subtype(cc, name) if cc else kind, 'linux', 'x86_64',
        None, False, '12.8' if cc else None)
    return base_info


@pytest.mark.parametrize('cc,name,family,subtype', [
    ((8, 6), 'NVIDIA GeForce RTX 3090', 'sm86', 'rtx30'),
    ((8, 9), 'NVIDIA GeForce RTX 4090', 'sm89', 'rtx40'),
    ((8, 0), 'NVIDIA A100-SXM4-80GB', 'sm80', 'a100'),
    ((9, 0), 'NVIDIA H100 80GB HBM3', 'sm90', 'hopper'),
    ((10, 0), 'NVIDIA B200', 'sm100', 'b200'),
    ((12, 0), 'NVIDIA GeForce RTX 5090', 'sm120', 'rtx50'),
    ((12, 0), 'NVIDIA RTX PRO 6000 Blackwell Workstation Edition', 'sm120',
     'rtx-pro-blackwell'),
    ((12, 1), 'NVIDIA GB10', 'sm121', 'dgx-spark'),
])
def test_sm_families_and_candidates(cc, name, family, subtype):
    info = _info(cc, name)
    assert (info.family, info.subtype) == (family, subtype)
    assert backends.candidates(info) == ['triton-int8']


def test_non_cuda_candidates():
    mps = dataclasses.replace(_info(None, 'Apple M3', 'mps'), family='mps')
    assert backends.candidates(mps) == ['mlx']
    cpu = dataclasses.replace(_info(None, 'CPU', 'cpu'), family='cpu')
    assert backends.candidates(cpu) == []
    # Anything older than SM80 has no kernel either.
    assert backends.candidates(_info((7, 5), 'NVIDIA T4')) == []


def test_reference_backend_passes_self_test_on_cpu():
    base.self_test(ReferenceBackend(), torch.device('cpu'), torch.float32)


def test_self_test_catches_a_kernel_that_ignores_the_mask():
    class Dense(base.Backend):
        name = 'dense'
        dtypes = (torch.float32,)

        def attend(self, q, k, v, block_mask, layout):
            mask = torch.ones_like(block_mask) & layout.kv_ok
            return reference.block_sparse_attention(q, k, v, mask, layout)

    with pytest.raises(base.BackendUnavailable, match='ignores the block'):
        base.self_test(Dense(), torch.device('cpu'), torch.float32)


def test_self_test_catches_wrong_results():
    class Wrong(ReferenceBackend):
        name = 'wrong'

        def attend(self, *args):
            return super().attend(*args) * 1.5

    with pytest.raises(base.BackendUnavailable, match='wrong results'):
        base.self_test(Wrong(), torch.device('cpu'), torch.float32)


def test_resolve_on_a_device_with_no_kernel():
    backends.reset()
    resolution = backends.resolve(torch.device('cpu'))
    assert resolution.backend is None
    assert resolution.attempts == []


@pytest.mark.skipif(importlib.util.find_spec('mlx') is None
                    or not torch.backends.mps.is_available(),
                    reason='needs Apple silicon with mlx')
def test_mlx_and_torch_backends_on_mps():
    from veda_comfy.backends import mlx_gather
    info = hardware.describe('mps')
    base.self_test(mlx_gather.create(info), torch.device('mps'))

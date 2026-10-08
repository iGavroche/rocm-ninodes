"""End to end through ComfyUI's real MiniMax-H3 code (tiny random model).

Runs `MiniMaxH3Model.forward` with Veda installed exactly as the node
installs it, for T2VA, FL2VA (keyframe) and R2VA (reference image + video)
layouts. Keep-all budgets must reproduce full attention; sparse budgets
must run the sparse path on every block. Skipped without a ComfyUI checkout
(see tests/conftest.py).
"""

import pytest
import torch

from conftest import COMFYUI_ROOT
import veda_testing

pytestmark = pytest.mark.skipif(COMFYUI_ROOT is None,
                                reason='needs a ComfyUI checkout')

LAYERS, HEADS, HIDDEN = 2, 2, 256


@pytest.fixture(autouse=True)
def on_reference_backend(monkeypatch):
    """Resolves every device to the exact reference backend.

    Every shipped backend needs a GPU, so on CPU CI `backends.resolve`
    finds nothing and the patch goes dense - which would leave the whole
    override path untested. `reference_backend` stands in: it is exact, so
    a failure here is about the code under test and never about precision.
    """
    import reference_backend
    from veda_comfy import backends
    from veda_comfy import hardware

    def resolve(device, notify=None):
        del notify  # nothing slow to announce
        return backends.Resolution(hardware.describe(device),
                                   reference_backend.ReferenceBackend(),
                                   [('reference', 'ok')])

    monkeypatch.setattr(backends, 'resolve', resolve)


@pytest.fixture(scope='module')
def h3():
    import comfy.ops
    from comfy.ldm.minimax import model as h3_model
    torch.manual_seed(0)
    model = h3_model.MiniMaxH3Model(
        hidden_size=HIDDEN, num_layers=LAYERS, token_refiner_num_layers=1,
        num_attention_heads=HEADS, attention_head_dim=128,
        ffn_hidden_size=512, text_dim=HIDDEN, timestep_input_dim=256,
        time_embed_hidden_size=HIDDEN, time_embed_dim=128,
        dtype=torch.float32, device='cpu',
        operations=comfy.ops.disable_weight_init)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if 'norm' in name and name.endswith('weight'):
                p.fill_(1.0)
            else:
                p.normal_(0.0, 0.05)
        model.rope.inv_freq.copy_(
            1.0 / (10000 ** (torch.arange(16, dtype=torch.float32) / 16)))
    return h3_model, model.eval().requires_grad_(False)


@pytest.fixture(scope='module')
def bundle(tmp_path_factory):
    from veda_comfy.core import bundle as veda_bundle
    path = str(tmp_path_factory.mktemp('veda') / 'tiny.safetensors')
    veda_testing.write_bundle(path, LAYERS, HEADS,
                              veda_testing.default_plans(LAYERS, HEADS))
    return veda_bundle.load_bundle(path)


def _inputs(h3_model, case):
    generator = torch.Generator().manual_seed(1)
    video = torch.randn(1, 24, 5, 16, 28, generator=generator)
    audio = torch.randn(1, 32, 2, 20, generator=generator)
    context = torch.randn(1, 40, HIDDEN, generator=generator)
    payload = {'seed': 0}
    keyframes = refs = None
    if case == 'fl2va':
        z = torch.randn(1, 24, 1, 16, 28, generator=generator)
        keyframes = [{'latent': z, 'resolved_frame_index': 0}]
        payload['keyframes'] = keyframes
        payload['cond_video_latents'] = [z]
    if case == 'r2va':
        image = torch.randn(1, 24, 1, 16, 16, generator=generator)
        clip = torch.randn(1, 24, 2, 8, 8, generator=generator)
        refs = [{'kind': 'image', 'latent_h': 16, 'latent_w': 16,
                 'latent': image},
                {'kind': 'video', 'latent_t': 2, 'latent_h': 8,
                 'latent_w': 8, 'ref_audio_t': 0, 'latent': clip}]
        payload['refs'] = refs
        payload['cond_video_latents'] = [image, clip]
    payload['layout'] = h3_model.PackedLayout(40, 5, 16, 28, 20,
                                              keyframes=keyframes, refs=refs)
    return [video, audio], context, payload


def _forward(h3, case, patch=None):
    h3_model, model = h3
    x, context, payload = _inputs(h3_model, case)
    options = {'sigmas': torch.tensor([0.5]),
               'sample_sigmas': torch.tensor([1.0, 0.5, 0.0])}
    if patch is not None:
        patch.install(options)
    with torch.no_grad():
        out = model(x, torch.tensor([500.0]), context,
                    transformer_options=options, minimax_payload=payload)
    return out, payload['layout']


def _patch(bundle, generated, reference, **kwargs):
    from veda_comfy import comfy_patch
    from veda_comfy import settings
    from veda_comfy.core import selection
    s = settings.VedaSettings(generated=selection.Budget(**generated),
                              reference=selection.Budget(**reference),
                              **kwargs)
    return comfy_patch.VedaPatch(bundle, s, node_id=None)


KEEP_ALL = {'tiles': 10**6}


@pytest.mark.parametrize('case', ['t2va', 'fl2va', 'r2va'])
def test_keep_all_reproduces_full_attention(h3, bundle, case):
    dense, _ = _forward(h3, case)
    patch = _patch(bundle, KEEP_ALL, KEEP_ALL)
    veda, _ = _forward(h3, case, patch)
    assert patch.calls['sparse'] == LAYERS
    for a, b in zip(dense, veda):
        torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize('case', ['t2va', 'fl2va', 'r2va'])
def test_sparse_runs_every_block(h3, bundle, case):
    dense, layout = _forward(h3, case)
    patch = _patch(bundle, {'ratio': 0.1}, {'ratio': 0.1})
    veda, _ = _forward(h3, case, patch)
    assert patch.calls['sparse'] == LAYERS
    assert all(torch.isfinite(t).all() for t in veda)
    assert any((a - b).abs().max() > 1e-4 for a, b in zip(dense, veda))
    engine = next(iter(patch._engines.values()))
    spec = engine.layout_spec(layout)
    expected = {'t2va': 0, 'fl2va': 1, 'r2va': 2}[case]
    assert len(spec.references) == expected


def test_reference_full_attention_leaves_conditions_untiled(h3, bundle):
    patch = _patch(bundle, {'ratio': 0.1}, {'ratio': 1.0})
    _forward(h3, 'r2va', patch)
    engine = next(iter(patch._engines.values()))
    assert all(layout.n_ref_tiles == 0
               for layout in engine._tile_layouts.values())


def test_full_attention_layers_and_steps(h3, bundle):
    patch = _patch(bundle, {'ratio': 0.1}, {'ratio': 0.1},
                   dense_layers=frozenset({0}))
    _forward(h3, 't2va', patch)
    assert patch.calls['full-attention layer'] == 1
    assert patch.calls['sparse'] == LAYERS - 1
    patch = _patch(bundle, {'ratio': 0.1}, {'ratio': 0.1},
                   dense_steps=frozenset({1}))  # sigma 0.5 is step 1
    _forward(h3, 't2va', patch)
    assert patch.calls['full-attention step'] == LAYERS
    assert patch.calls['sparse'] == 0


def test_declined_calls_reach_the_previous_override(h3, bundle):
    seen = []

    def previous(func, *args, **kwargs):
        seen.append(kwargs['transformer_options'].get('block_index'))
        return func(*args, **kwargs)

    patch = _patch(bundle, {'ratio': 0.1}, {'ratio': 0.1},
                   dense_layers=frozenset({1}))
    h3_model, model = h3
    x, context, payload = _inputs(h3_model, 't2va')
    options = {'optimized_attention_override': previous}
    patch.install(options)
    patch.install(options)  # idempotent once on top
    with torch.no_grad():
        model(x, torch.tensor([500.0]), context, transformer_options=options,
              minimax_payload=payload)
    assert seen == [1]
    assert len(patch.installed) == 1


@pytest.mark.parametrize('verbose', [False, True])
def test_node_text_reports_sparsity_not_call_counts(h3, bundle, verbose):
    patch = _patch(bundle, {'ratio': 0.1}, {'ratio': 0.1}, verbose=verbose)
    shown = []
    patch.status.show = lambda text, *a, **k: shown.append(text)
    patch.status.warn = lambda text, *a, **k: shown.append(text)
    _forward(h3, 't2va', patch)
    patch.on_cleanup()
    running, summary = shown[0], shown[-1]
    assert running.startswith('Veda running · reference (fp32)')
    assert 'Video: 448x256' in running and 'Sparsity: generated 90%' in running
    assert summary.startswith('Veda done · reference (fp32)')
    assert '% of full attention' in summary
    assert ('Attention calls' in summary) == verbose
    assert 'sparse /' not in summary
    assert patch.calls == {}  # reset for the next run


def _low_vram_forward(attn, x, rope_freqs=None, transformer_options={}):
    """The H3 attention forward as KJNodes' "MiniMax H3 Low VRAM Attention"
    patches it: one attention call per head group (`minimax_head_chunks`),
    written into a preallocated output."""
    from comfy.ldm.modules.attention import optimized_attention
    import comfy.quant_ops
    s, heads, dim = x.shape[0], attn.heads, attn.head_dim
    q, k, v = attn.qkv_proj(x).split(heads * dim, dim=-1)
    q = q.reshape(1, s, heads, dim)
    k = k.reshape(1, s, heads, dim)
    comfy.quant_ops.ck.rms_rope_split_half_(
        q, k, rope_freqs, attn.q_norm.weight, attn.k_norm.weight,
        epsilon=attn.q_norm.eps, rot_dim=rope_freqs.shape[-3] * 2)
    q, k = q[0].transpose(0, 1)[None], k[0].transpose(0, 1)[None]
    v = v.reshape(s, heads, dim).transpose(0, 1)[None]
    n = min(transformer_options.get('minimax_head_chunks', 1), heads)
    out = torch.empty(s, heads * dim, dtype=x.dtype)
    start = 0
    for i in range(n):
        end = start + heads // n + (1 if i < heads % n else 0)
        part = optimized_attention(
            q[:, start:end], k[:, start:end], v[:, start:end], end - start,
            mask=None, skip_reshape=True,
            transformer_options=transformer_options)
        out[:, start * dim:end * dim] = part.squeeze(0)
        start = end
    return attn.out_proj(out)


@pytest.fixture
def low_vram(h3, monkeypatch):
    import types
    _, model = h3
    for block in model.blocks:
        monkeypatch.setattr(block.attn, 'forward',
                            types.MethodType(_low_vram_forward, block.attn))


def test_kjnodes_head_chunks_still_run_sparse(h3, bundle, low_vram):
    """Without the takeover Veda saw HEADS / chunks heads per call and
    switched itself off ("the model has 14 heads, the predictor expects
    56")."""
    del low_vram
    reference = _patch(bundle, {'ratio': 0.1}, {'ratio': 0.1})
    expected, _ = _forward(h3, 't2va', reference)
    patch = _patch(bundle, {'ratio': 0.1}, {'ratio': 0.1})
    h3_model, model = h3
    x, context, payload = _inputs(h3_model, 't2va')
    options = {'minimax_head_chunks': HEADS}
    patch.install(options)
    patch.install(options)  # every step: the held count survives
    with torch.no_grad():
        out = model(x, torch.tensor([500.0]), context,
                    transformer_options=options, minimax_payload=payload)
    assert patch.calls == {'sparse': LAYERS}
    for a, b in zip(expected, out):
        torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-5)


def test_declined_calls_keep_the_kjnodes_head_split(h3, bundle, low_vram):
    del low_vram
    dense, _ = _forward(h3, 't2va')
    seen = []

    def previous(func, q, k, v, heads, **kwargs):
        seen.append(heads)
        return func(q, k, v, heads, **kwargs)

    patch = _patch(bundle, {'ratio': 0.1}, {'ratio': 0.1},
                   dense_layers=frozenset(range(LAYERS)))
    h3_model, model = h3
    x, context, payload = _inputs(h3_model, 't2va')
    options = {'optimized_attention_override': previous,
               'minimax_head_chunks': HEADS}
    patch.install(options)
    with torch.no_grad():
        out = model(x, torch.tensor([500.0]), context,
                    transformer_options=options, minimax_payload=payload)
    assert seen == [1] * HEADS * LAYERS
    for a, b in zip(dense, out):
        torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-4)


def _bypassing_forward(seen):
    """An H3 attention forward that never calls the attention override,
    like KJNodes' "MiniMax H3 Mem Eff Sage Attention Patch"."""
    def make(attn):
        stock = type(attn).forward

        def forward(x, rope_freqs=None, transformer_options={}):
            seen.append((transformer_options.get('block_index'),
                         transformer_options.get('minimax_head_chunks')))
            options = dict(transformer_options)
            options.pop('optimized_attention_override', None)
            return stock(attn, x, rope_freqs=rope_freqs,
                         transformer_options=options)
        return forward
    return make


def test_forward_replacement_hides_every_call_from_veda(h3, bundle,
                                                        monkeypatch):
    """What users saw with Mem Eff Sage: Veda printed "ready", then
    nothing; now the run ends with a warning."""
    from veda_comfy import status
    _, model = h3
    make = _bypassing_forward([])
    for block in model.blocks:
        monkeypatch.setattr(block.attn, 'forward', make(block.attn))
    patch = _patch(bundle, {'ratio': 0.1}, {'ratio': 0.1})
    warned = []
    patch.status.warn = lambda text, zh=None: warned.append(
        status.bilingual(text, zh))
    patch.run.prepared = True  # ON_PREPARE_STATE ran
    _forward(h3, 't2va', patch)
    assert patch.calls.get('sparse', 0) == 0
    patch.on_cleanup()
    assert warned and warned[-1].startswith('Veda did not run')
    assert '\nVeda 未运行' in warned[-1]  # warnings carry Chinese too


def test_phased_guider_second_phase_does_not_warn(h3, bundle):
    """Spectrum H3 samples in phases: the model phase reaches Veda, the
    forecast phase runs without the model. The second cleanup must not
    claim Veda never ran."""
    from veda_comfy import status
    patch = _patch(bundle, {'ratio': 0.1}, {'ratio': 0.1})
    warned = []
    patch.status.warn = lambda text, zh=None: warned.append(
        status.bilingual(text, zh))
    patch.run.prepared = True
    _forward(h3, 't2va', patch)
    assert patch.calls['sparse'] == LAYERS
    patch.on_cleanup()          # end of the model phase
    patch.run.prepared = True   # forecast phase: no model call at all
    patch.on_cleanup()
    assert warned == []

@pytest.mark.parametrize('take_forward', [None, _low_vram_forward])
def test_routed_forward_runs_sparse_and_keeps_the_replacement(
        h3, bundle, monkeypatch, take_forward):
    """Sparse blocks go through ComfyUI's forward, or KJNodes' Low VRAM
    forward when that node left it under `sol_take_forward`."""
    _, model = h3
    seen = []
    make = _bypassing_forward(seen)
    patch = _patch(bundle, {'ratio': 0.1}, {'ratio': 0.1},
                   dense_layers=frozenset({1}))
    for block in model.blocks:
        monkeypatch.setattr(block.attn, 'forward', patch._routed_forward(
            block.attn, make(block.attn)))
    h3_model, _ = h3
    x, context, payload = _inputs(h3_model, 't2va')
    options = {'minimax_head_chunks': 2, 'sol_take_forward': take_forward}
    patch.install(options)
    with torch.no_grad():
        out = model(x, torch.tensor([500.0]), context,
                    transformer_options=options, minimax_payload=payload)
    assert all(torch.isfinite(t).all() for t in out)
    assert patch.calls == {'sparse': LAYERS - 1, 'full-attention layer': 1}
    # The replacing forward gets block 1 and its head split back.
    assert seen == [(1, 2)]

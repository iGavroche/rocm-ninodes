"""Hooks Veda into ComfyUI's MiniMax-H3 attention.

H3's `Attention.forward` (comfy/ldm/minimax/model.py) calls
`optimized_attention` with q / k / v already through the (LoRA-patched)
qkv projection, QK-norm and RoPE, and with `transformer_options` carrying
`block_index`, `minimax_h3_layout` and the sampling sigmas. Veda replaces
exactly that call through `transformer_options["optimized_attention_override"]`:

  * weights are untouched, so any LoRA, fine-tune or quantized H3 checkpoint
    works, and T2VA / FL2VA / R2VA conditioning nodes are unchanged;
  * no `patches_replace` slot is taken, so block patches such as H3's Fun
    ControlNet keep working;
  * calls Veda declines run the override that was installed before it (or
    ComfyUI's attention), so other attention nodes still apply there.

The override is re-installed on every step (ON_PREPARE_STATE), like
ComfyUI's own sparse attention node, so a node applied later cannot
silently replace it. KJNodes' "MiniMax H3 Low VRAM Attention" splits each
call into head groups (`minimax_head_chunks`), which would hand Veda a
slice of heads it cannot map onto the predictor; Veda takes that split
over instead (see `install`).

Some nodes replace H3's whole attention forward with one that never calls
`optimized_attention` (KJNodes' "MiniMax H3 Mem Eff Sage Attention
Patch"), so the override would never run. When such a node comes before
Veda, Veda routes that forward per block (`_routed_forward`): blocks Veda
runs sparse take a forward that calls the override, the others keep the
replacing node's forward. When it comes after Veda nothing can be routed;
the node says so, and a run in which no H3 call reached Veda says so too.

Node text: a few readable lines (kernel, video and plan, sparsity, share
of full attention computed); `verbose` adds timing and call diagnostics.
"""

from __future__ import annotations

import collections
import contextlib
import logging
import re
import weakref

import torch

import comfy.patcher_extension

from . import backends
from . import settings as veda_settings
from . import status as veda_status
from .core import engine as veda_engine
from .core import h3_layout
from .core import plans as veda_plans

try:  # ComfyUI with the comfy-aimdo allocation graph (0.38+)
    from comfy.model_prefetch import pause_malloc_graph as _pause_malloc_graph
except ImportError:  # older ComfyUI: nothing to pause
    _pause_malloc_graph = contextlib.nullcontext

_KEY = 'veda_sparse_attention'
# KJNodes' "MiniMax H3 Low VRAM Attention" asks the H3 attention forward to
# call attention once per head group; Veda moves the request to its own key
# and splits only the calls it runs as full attention.
_HEAD_CHUNKS = 'minimax_head_chunks'
_HELD_HEAD_CHUNKS = 'veda_held_head_chunks'
# KJNodes' convention: a forward that still calls `optimized_attention` is
# marked with this attribute, and a node that installs one for sparse
# attention to use leaves it under `sol_take_forward` (its Low VRAM node).
_COMPOSES = '_uses_optimized_attention'
TAKE_FORWARD = 'sol_take_forward'
_ATTN_FORWARD = re.compile(r'diffusion_model\.blocks\.(\d+)\.attn\.forward')
# Display names of known forward replacements, by function name.
_KNOWN_FORWARDS = {
    'minimax_sageattn_forward':
        '"MiniMax H3 Mem Eff Sage Attention Patch" (KJNodes)',
}
_TIMED_PHASES = (('gather', 'gather'), ('score', 'score'),
                 ('select', 'select'), ('attend', 'kernel'),
                 ('scatter', 'scatter'))


def _by_head_groups(attend, q, chunks: int, skip_output_reshape: bool):
    """Full attention in `chunks` head groups, split like KJNodes' Low VRAM
    node so its peak-memory saving holds on the calls Veda declines.

    Args:
        attend: `attend(start, end)` runs heads [start, end) and returns
            [B, S, h*D], or [B, h, S, D] with skip_output_reshape.
        q: [B, H, S, D], for the shapes.
        chunks: Number of head groups (> 1).
        skip_output_reshape: The output layout attention was asked for.
    """
    batch, heads, seq_len, dim = q.shape
    chunks = min(chunks, heads)
    out = None
    start = 0
    for i in range(chunks):
        end = start + heads // chunks + (1 if i < heads % chunks else 0)
        part = attend(start, end)
        if skip_output_reshape:
            if out is None:
                out = part.new_empty(batch, heads, seq_len, dim)
            out[:, start:end] = part
        else:
            if out is None:
                out = part.new_empty(batch, seq_len, heads * dim)
            out[..., start * dim:end * dim] = part
        del part
        start = end
    return out


def replaced_forwards(model_patcher) -> dict[int, object]:
    """H3 attention forwards patched in by other nodes that never call
    `optimized_attention`, by block index."""
    found = {}
    patches = getattr(model_patcher, 'object_patches', None) or {}
    for key, value in patches.items():
        match = _ATTN_FORWARD.fullmatch(key)
        if match and not getattr(value, _COMPOSES, False):
            found[int(match.group(1))] = value
    return found


def describe_forward(forward) -> str:
    """Who installed an attention forward, for the node text."""
    name = getattr(forward, '__name__', type(forward).__name__)
    known = _KNOWN_FORWARDS.get(name)
    if known:
        return known
    module = getattr(forward, '__module__', None)
    return f'{name} ({module})' if module else name


def _is_interrupt(error: BaseException) -> bool:
    # comfy.model_management.InterruptProcessingException, without importing
    # it at module scope.
    return type(error).__name__ == 'InterruptProcessingException'


class _Run:
    """Per sampling run: what happened, for the summary on the node."""

    def __init__(self):
        self.calls = collections.Counter()
        self.evaluations = 0          # model calls (layer 0 reached)
        self.video = None             # e.g. '1344x768 · 5.2 s'
        self.failed = None            # error text if the sparse path failed
        self.prepared = False         # sampling started (ON_PREPARE_STATE)
        self.announced: set = set()


class VedaPatch:
    """Options and runtime state of one patched model."""

    def __init__(self, bundle, settings: veda_settings.VedaSettings,
                 node_id: str | None):
        self.bundle = bundle
        self.settings = settings
        self.status = veda_status.NodeStatus(node_id)
        self.installed: set = set()
        self._engines: dict[str, veda_engine.VedaEngine | None] = {}
        self._timers: dict[str, veda_engine.PhaseTimer] = {}
        self._steps = weakref.WeakKeyDictionary()
        self.run = _Run()
        # Display names of the forwards routed by `take_over_forwards`.
        self.taken_over: list[str] = []
        # Whether any MiniMax-H3 attention call ever reached this patch.
        # A guider can sample in phases (Spectrum H3 forecasts the last
        # steps without the model); only a run that never reached Veda at
        # all is a wiring problem.
        self.ever_ran = False

    @property
    def calls(self) -> collections.Counter:
        return self.run.calls

    # -- per device ------------------------------------------------------

    def _engine(self, device: torch.device) -> veda_engine.VedaEngine | None:
        key = str(device)
        if key not in self._engines:
            resolution = backends.resolve(
                device,
                notify=self.status.show)
            if resolution.backend is None:
                self.status.warn(
                    f'Veda off: no sparse kernel works on '
                    f'{resolution.device.label}; using full attention\n'
                    f'{resolution.report()}',
                    zh=f'Veda 未启用：{resolution.device.label} 上没有可用的'
                       '稀疏 kernel，本次使用全注意力（原因见上）。')
                self._engines[key] = None
            else:
                engine = veda_engine.VedaEngine(
                    self.bundle, self.settings.generated,
                    self.settings.reference, resolution.backend, device)
                if self.settings.verbose:
                    timer = engine.enable_timing()
                    if timer is not None:
                        self._timers[key] = timer
                    logging.info('Veda: backends on %s: %s',
                                 resolution.device.label, resolution.report())
                self._engines[key] = engine
        return self._engines[key]

    def _step(self, transformer_options) -> int | None:
        sigmas = transformer_options.get('sigmas')
        schedule = transformer_options.get('sample_sigmas')
        if sigmas is None or schedule is None:
            return None
        try:
            return self._steps[sigmas]
        except (KeyError, TypeError):
            pass
        step = veda_settings.step_index(float(sigmas.flatten()[0]),
                                        schedule.flatten().tolist())
        try:
            self._steps[sigmas] = step
        except TypeError:
            pass
        return step

    # -- the override ------------------------------------------------------

    def make_override(self, previous):
        """The attention override; declined calls go to `previous`."""
        patch = self

        def override(func, q, k, v, heads, mask=None, attn_precision=None,
                     skip_reshape=False, skip_output_reshape=False,
                     **kwargs):
            options = kwargs.get('transformer_options') or {}

            def dense(reason: str):
                patch.run.calls[reason] += 1
                kw = dict(mask=mask, attn_precision=attn_precision,
                          skip_reshape=skip_reshape,
                          skip_output_reshape=skip_output_reshape, **kwargs)

                def attend(q, k, v, heads):
                    if previous is None:
                        return func(q, k, v, heads, **kw)
                    return previous(func, q, k, v, heads, **kw)

                chunks = options.get(_HELD_HEAD_CHUNKS)
                if reason == 'other attention' or not chunks:
                    return attend(q, k, v, heads)
                return _by_head_groups(
                    lambda a, b: attend(q[:, a:b], k[:, a:b], v[:, a:b],
                                        b - a),
                    q, chunks, skip_output_reshape)

            layout = options.get('minimax_h3_layout')
            if (mask is not None or not skip_reshape or q.dim() != 4
                    or q.shape[0] != 1 or q.shape != k.shape
                    or q.shape != v.shape or layout is None
                    or getattr(layout, 'seq_len', None) != q.shape[2]):
                return dense('other attention')
            patch.ever_ran = True
            if options.get('block_index') == 0:
                patch.run.evaluations += 1
            reason = patch._dense_reason(q, options)
            if reason is not None:
                return dense(reason)
            try:
                with _pause_malloc_graph():
                    return patch._sparse(q, k, v, layout, options,
                                         skip_output_reshape, dense)
            except Exception as error:  # never lose the user's render
                if _is_interrupt(error):
                    raise
                patch.run.failed = f'{type(error).__name__}: {error}'
                logging.error('Veda: sparse attention failed', exc_info=True)
                patch.status.warn(
                    'Veda hit an error and finishes this run with full '
                    f'attention:\n{patch.run.failed}',
                    zh='Veda 出错，本次剩余部分改用全注意力，渲染照常完成'
                       '（错误见上）。')
                return dense('error')

        return override

    def _block_reason(self, options, device=None) -> str | None:
        """Why this block runs full attention, from what is known before
        q / k / v exist; None if it may go sparse."""
        s = self.settings
        if self.run.failed:
            return 'error'
        layer = options.get('block_index')
        if not isinstance(layer, int) or layer >= self.bundle.num_layers:
            return 'layer outside the predictor'
        if layer in s.dense_layers:
            return 'full-attention layer'
        if s.dense_steps and self._step(options) in s.dense_steps:
            return 'full-attention step'
        if s.generated.keeps_all and s.reference.keeps_all:
            return 'sparsity 0%'
        if device is not None and self._engines.get(str(device), 0) is None:
            return 'no sparse kernel'
        return None

    def _dense_reason(self, q, options) -> str | None:
        """Why this H3 call runs full attention, or None to go sparse."""
        bundle = self.bundle
        reason = self._block_reason(options)
        if reason is not None:
            return reason
        if q.shape[1] != bundle.num_heads or q.shape[3] != bundle.head_dim:
            self._announce(('shape', tuple(q.shape)),
                           f'Veda off: an attention call has {q.shape[1]} '
                           f'heads of dim {q.shape[3]}, the predictor '
                           f'expects {bundle.num_heads} x {bundle.head_dim}. '
                           'Another node may split the heads; using full '
                           'attention.',
                           warn=True,
                           zh=f'Veda 未启用：注意力调用有 {q.shape[1]} 个头'
                              f'（维度 {q.shape[3]}），打分器需要 '
                              f'{bundle.num_heads} x {bundle.head_dim}。可能有'
                              '其他节点把头拆开了；本次使用全注意力。')
            return 'head mismatch'
        return None

    def _sparse(self, q, k, v, layout, options, skip_output_reshape, dense):
        # ComfyUI records each block's allocations into a malloc graph
        # (comfy-aimdo) and expects everything allocated inside a block to
        # be gone by its end. Veda keeps device state across calls (tile
        # layouts, plan head groups, statistics) and its kernels allocate
        # their own workspaces, so the caller runs this with the graph
        # paused, like ComfyUI's own sparse attention node does for its
        # persistent state. Otherwise the process aborts natively.
        engine = self._engine(q.device)
        if engine is None:
            return dense('no sparse kernel')
        try:
            spec = engine.layout_spec(layout)
        except h3_layout.LayoutError as error:
            self._announce(('layout', str(error)),
                           f'Veda off for this video: cannot read the H3 '
                           f'layout ({error}); using full attention.',
                           warn=True,
                           zh='Veda 本次未启用：读不懂这个视频的 H3 序列布局'
                              '（原因见上），本次使用全注意力。')
            return dense('layout')
        choice = engine.plan_for(spec)
        self.run.video = veda_plans.describe_grid(spec.target.grid)
        self._announce(('plan', spec.target.grid),
                       self._running_text(engine, spec, choice),
                       warn=not choice.exact)
        batch, heads, seq_len, dim = q.shape
        out = engine.attention(q[0].transpose(0, 1), k[0].transpose(0, 1),
                               v[0].transpose(0, 1), options['block_index'],
                               spec, choice.plan,
                               head_chunks=options.get(_HELD_HEAD_CHUNKS, 1))
        self.run.calls['sparse'] += 1
        if skip_output_reshape:
            return out.transpose(0, 1).unsqueeze(0)
        return out.reshape(batch, seq_len, heads * dim)

    # -- node text ---------------------------------------------------------

    def _running_text(self, engine, spec, choice) -> str:
        lines = [f'Veda running · {engine.backend.display}',
                 f'Video: {veda_plans.describe_grid(spec.target.grid)}',
                 f'Tile plan: {choice.how}']
        lines.append(f'Sparsity: {self.settings.describe()}')
        if spec.references:
            count = len(spec.references)
            mode = ('full attention' if self.settings.reference.keeps_all
                    else 'tiled, ' + veda_settings.format_budget(
                        self.settings.reference) + ' sparse')
            lines.append(f'References: {count} span'
                         f'{"s" if count > 1 else ""} ({mode})')
        full = self.settings.describe_full_attention()
        if full:
            lines.append(f'Full attention: {full}')
        if self.settings.verbose and spec.skipped:
            lines.append('Untiled references: ' + '; '.join(spec.skipped))
        return '\n'.join(lines)

    def _summary(self) -> str | None:
        run = self.run
        engines = [e for e in self._engines.values() if e is not None]
        sparse = run.calls.get('sparse', 0)
        full = sum(n for r, n in run.calls.items()
                   if r not in ('sparse', 'other attention'))
        if not sparse and not full:
            return None
        backend = engines[0].backend.display if engines else 'full attention'
        headline = ('Veda done' if sparse and not run.failed
                    else 'Veda done, fell back to full attention')
        lines = [f'{headline} · {backend}']
        if run.video:
            lines.append(f'Video: {run.video}')
        work = [e.stats.compute_fraction() for e in engines]
        work = [w for w in work if w is not None]
        if work:
            share = sum(work) / len(work)
            if full:  # dense calls do all of their work
                share = (share * sparse + full) / (sparse + full)
            lines.append(f'Attention computed: {100 * share:.1f}% of full '
                         f'attention ({100 * (1 - share):.1f}% skipped)')
        configured = self.settings.describe_full_attention()
        if configured:
            lines.append(f'Full attention: {configured}')
        if run.failed:
            lines.append(f'Fell back to full attention after: {run.failed}')
        if self.settings.verbose:
            lines += self._diagnostics(engines)
        return '\n'.join(lines)

    def _diagnostics(self, engines) -> list[str]:
        run, lines = self.run, ['-- diagnostics --']
        for key, timer in self._timers.items():
            phases = timer.summary()
            total = sum(phases.values()) / 1000.0
            if not total:
                continue
            per_eval = total / max(1, run.evaluations)
            lines.append(f'Veda attention time: {total:.2f} s '
                         f'({per_eval:.2f} s per model call x '
                         f'{run.evaluations})')
            lines.append('  ' + ' · '.join(
                f'{label} {phases.get(name, 0.0) / 1000.0:.2f} s'
                for name, label in _TIMED_PHASES))
            self._timers[key] = self._engines[key].enable_timing()
        for engine in engines:
            kept = engine.stats.kept_fraction()
            if kept is not None:
                lines.append(f'Video tiles kept: {100 * kept:.1f}% of '
                             'video x video tile pairs')
            chunking = engine.chunking
            if chunking.get('chunks_per_layer'):
                free = chunking.get('free_bytes')
                lines.append(
                    f'Chunks: {chunking["chunks_per_layer"]} per layer, '
                    f'{chunking["heads_per_chunk"]} heads each'
                    + (f' ({free / 2**30:.1f} GB free)' if free else ''))
                workspace = chunking.get('workspace_bytes')
                if workspace:
                    lines.append('Attention workspace: '
                                 f'{workspace / 2**20:.0f} MB at once')
        reasons = ', '.join(f'{r} {n}' for r, n in sorted(run.calls.items())
                            if r != 'sparse')
        lines.append(f'Attention calls: {run.calls.get("sparse", 0)} sparse'
                     + (f'; full attention: {reasons}' if reasons else ''))
        lines.append(f'Predictor: {self.bundle.describe()}')
        return lines

    def _announce(self, key, text: str, warn: bool = False,
                  zh: str | None = None) -> None:
        if key in self.run.announced:
            return
        self.run.announced.add(key)
        if warn:
            self.status.warn(text, zh=zh)
        else:
            self.status.show(text, zh=zh)

    # -- other nodes' attention forwards ----------------------------------

    def _routed_forward(self, attn, replaced):
        """Block `attn`'s forward when another node replaced it with one that
        bypasses the override: blocks Veda may run sparse take a forward
        that calls the override, the rest keep `replaced`."""
        patch = self
        stock = type(attn).forward

        def forward(x, rope_freqs=None, transformer_options={}):
            options = (transformer_options
                       if isinstance(transformer_options, dict) else {})
            first = x[0] if isinstance(x, list) else x
            reason = patch._block_reason(options, first.device)
            if reason is None:
                own = options.get(TAKE_FORWARD)
                if own is not None:  # KJNodes' Low VRAM forward
                    return own(attn, x, rope_freqs=rope_freqs,
                               transformer_options=transformer_options)
                if isinstance(x, list):  # KJNodes' block patch hands [h]
                    x = x.pop()
                return stock(attn, x, rope_freqs=rope_freqs,
                             transformer_options=transformer_options)
            patch.run.calls[reason] += 1
            if options.get('block_index') == 0:
                patch.run.evaluations += 1
            held = options.get(_HELD_HEAD_CHUNKS)
            if held:  # the replacing forward reads the split itself
                transformer_options = dict(transformer_options)
                transformer_options[_HEAD_CHUNKS] = held
            return replaced(x, rope_freqs=rope_freqs,
                            transformer_options=transformer_options)

        forward._uses_optimized_attention = True
        return forward

    def take_over_forwards(self, model_patcher, diffusion) -> None:
        """Routes the H3 attention forwards that bypass the override (see
        `_routed_forward`) on `model_patcher`, a clone Veda owns."""
        names = []
        for index, replaced in sorted(replaced_forwards(model_patcher).items()):
            if index >= len(diffusion.blocks):
                continue
            attn = diffusion.blocks[index].attn
            model_patcher.add_object_patch(
                f'diffusion_model.blocks.{index}.attn.forward',
                self._routed_forward(attn, replaced))
            name = describe_forward(replaced)
            if name not in names:
                names.append(name)
        self.taken_over = names

    # -- lifecycle ---------------------------------------------------------

    def on_prepare(self, model_patcher, model_options) -> None:
        """Every sampling step: re-install, and catch nodes after Veda that
        replace the H3 attention forward (Veda cannot route those)."""
        self.run.prepared = True
        self.install(model_options['transformer_options'])
        later = replaced_forwards(model_patcher)
        if later:
            names = ', '.join(sorted({describe_forward(f)
                                      for f in later.values()}))
            self._announce(
                ('replaced', names),
                f'Veda is not running: {names} replaces the MiniMax-H3 '
                'attention after Veda. Move the Veda node after it (last '
                'before the sampler).', warn=True,
                zh=f'Veda 未运行：{names} 在 Veda 之后替换了 MiniMax-H3 的'
                   '注意力。请把 Veda 节点移到它后面（采样器之前的最后一个）。')

    def install(self, transformer_options: dict) -> None:
        """Puts the override on top of whatever override is on the hook;
        idempotent once it is on top.

        Also takes over a head-group split requested by KJNodes' Low VRAM
        node: the H3 forward then hands Veda all heads at once, Veda's
        sparse path splits into at least that many head chunks itself, and
        declined calls are split here.
        """
        chunks = transformer_options.get(_HEAD_CHUNKS)
        if isinstance(chunks, int) and chunks > 1:
            transformer_options[_HELD_HEAD_CHUNKS] = chunks
            transformer_options[_HEAD_CHUNKS] = 1
        current = transformer_options.get('optimized_attention_override')
        if current in self.installed:
            return
        override = self.make_override(current)
        self.installed.add(override)
        transformer_options['optimized_attention_override'] = override

    def on_cleanup(self) -> None:
        """End of a sampling run: show the summary, reset per-run state."""
        summary = self._summary()
        if summary:
            self.status.show(summary)
        elif (self.run.prepared and not self.ever_ran
              and not any(isinstance(key, tuple) and key[0] == 'replaced'
                          for key in self.run.announced)):
            self.status.warn(
                'Veda did not run: no MiniMax-H3 attention call reached it '
                'in this render. Another node probably replaces the H3 '
                'attention; place Veda after it, last before the sampler.',
                zh='Veda 未运行：这次渲染没有任何 MiniMax-H3 注意力调用到达 '
                   'Veda。可能有其他节点替换了 H3 注意力；请把 Veda 放在它'
                   '之后、采样器之前。')
        for engine in self._engines.values():
            if engine is not None:
                engine.stats = veda_engine.Stats()
        self.run = _Run()


def apply(model, bundle, settings: veda_settings.VedaSettings,
          node_id: str | None):
    """Returns a clone of `model` with Veda attention installed."""
    patch = VedaPatch(bundle, settings, node_id)
    patched = model.clone()
    patch.install(patched.model_options.setdefault('transformer_options', {}))
    patch.take_over_forwards(patched,
                             patched.get_model_object('diffusion_model'))
    patched.add_callback_with_key(
        comfy.patcher_extension.CallbacksMP.ON_PREPARE_STATE, _KEY,
        lambda model_patcher, timestep, model_options: patch.on_prepare(
            model_patcher, model_options))
    patched.add_callback_with_key(
        comfy.patcher_extension.CallbacksMP.ON_CLEANUP, _KEY,
        lambda model_patcher: patch.on_cleanup())
    return patched, patch

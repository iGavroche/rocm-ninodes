"""One Veda attention call, end to end, on any device.

For each head group of the layer's tile plan (heads sharing a tile shape),
in chunks of heads bounded by memory:

    gather q/k/v into tile order -> pool video tiles -> predictor logits ->
    top-k selection -> block mask -> backend kernel -> scatter back

Everything except the kernel is device-agnostic torch, identical for every
backend. Caches (layout specs, tile layouts, plan choices) live on the
engine, i.e. per patched model and device, and are dropped with it.
"""

from __future__ import annotations

import collections
import contextlib
import weakref

import torch

from . import bundle as veda_bundle
from . import h3_layout
from . import plans as veda_plans
from . import predictor
from . import selection
from . import tiling

# Bounds of one tile-ordered q / k / v / out copy of a head chunk. Four of
# them are alive at once, plus the backend's own temporaries, so a chunk
# takes up to 1/8 of the free memory (about half of it in total). Larger
# chunks matter: every chunk launches ~60 small kernels, and on Windows
# (WDDM) launch latency made 8 chunks per layer cost ~15 ms more than 2
# (RTX 5070, 38k tokens).
_CHUNK_SHARE = 8
_MIN_CHUNK_BYTES = 128 * 2**20
_MAX_CHUNK_BYTES = 1024 * 2**20
_DEFAULT_CHUNK_BYTES = 256 * 2**20
# Free memory alone is the wrong bound: on a GPU with room to spare the
# chunk grows until Veda holds several GB that ComfyUI then cannot use to
# stage weights (measured 3.24 GB peak against 0.64 GB for full attention,
# RTX 5070, 1344x768 x 5.2 s). So the chunk is also kept to a share of the
# layer's own output buffer, which scales with the problem and not with
# the machine; the free-memory bound still applies on top for small GPUs.
_WORKING_SET_SHARE = 6


class Stats:
    """What the sparse calls computed, accumulated on the device and read
    once at the end of a run (reading per call would synchronize).

    Work is counted in 128x128 blocks per head: kept video blocks plus the
    always-dense global rows / columns, against the (S / 128)^2 blocks of
    full attention, so `compute_fraction` is the share of full-attention
    work actually done.
    """

    def __init__(self):
        self.kept = None
        self.video_pairs = 0
        self.work = None
        self.dense_work = 0.0

    def add(self, keep: torch.Tensor, layout: tiling.TileLayout) -> None:
        heads = keep.shape[0]
        kept = keep.sum()
        global_blocks = layout.n_global_tiles * (layout.n_tiles
                                                 + layout.n_video_tiles)
        work = kept + heads * global_blocks
        self.kept = kept if self.kept is None else self.kept + kept
        self.work = work if self.work is None else self.work + work
        self.video_pairs += heads * layout.n_video_tiles ** 2
        self.dense_work += heads * (layout.seq_len / tiling.TILE_SIZE) ** 2

    def kept_fraction(self) -> float | None:
        """Kept share of the video x video tile pairs."""
        if self.kept is None or not self.video_pairs:
            return None
        return float(self.kept) / self.video_pairs

    def compute_fraction(self) -> float | None:
        """Share of full-attention work done (all rows, all tokens)."""
        if self.work is None or not self.dense_work:
            return None
        return min(1.0, float(self.work) / self.dense_work)


class PhaseTimer:
    """Opt-in per-phase GPU time (CUDA events; read once at the end)."""

    def __init__(self):
        self._events = collections.defaultdict(list)

    @contextlib.contextmanager
    def __call__(self, name: str):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        yield
        end.record()
        self._events[name].append((start, end))

    def summary(self) -> dict[str, float]:
        """Milliseconds per phase (synchronizes the device)."""
        torch.cuda.synchronize()
        return {name: sum(s.elapsed_time(e) for s, e in pairs)
                for name, pairs in self._events.items()}


def _no_timer(name: str):
    del name
    return contextlib.nullcontext()


class VedaEngine:
    """Sparse attention for one predictor bundle and settings on a device.

    Args:
        bundle: The predictor bundle.
        generated: Budget of the generated (target) video's key tiles.
        reference: Budget of the reference / condition key tiles;
            keeps_all leaves the conditions global (full attention both
            ways), which is also what happens when the layout has none.
        backend: A `backends.base.Backend`.
        device: The device of q / k / v.
    """

    def __init__(self, bundle: veda_bundle.PredictorBundle,
                 generated: selection.Budget, reference: selection.Budget,
                 backend, device: torch.device):
        self.bundle = bundle
        self.generated = generated
        self.reference = reference
        self.backend = backend
        self.device = device
        self.stats = Stats()
        self.timer = _no_timer
        # Last call's chunking, for diagnostics: free memory, heads per
        # chunk, chunks per layer.
        self.chunking: dict[str, int] = {}
        self._specs = weakref.WeakKeyDictionary()
        self._tile_layouts: dict[tuple, tiling.TileLayout] = {}
        self._pinned = device.type == 'cuda'

    def enable_timing(self) -> PhaseTimer | None:
        """Times every phase from now on; returns the timer, or None off
        CUDA (no event timers there)."""
        if self.device.type != 'cuda':
            return None
        self.timer = PhaseTimer()
        return self.timer

    # -- caches ---------------------------------------------------------

    def layout_spec(self, layout) -> h3_layout.LayoutSpec:
        """LayoutSpec of a ComfyUI PackedLayout (cached per layout object).

        Raises:
            h3_layout.LayoutError: If the layout cannot be mapped.
        """
        try:
            spec = self._specs.get(layout)
        except TypeError:  # not weak-referenceable: describe every time
            return h3_layout.describe(layout)
        if spec is None:
            spec = h3_layout.describe(layout)
            self._specs[layout] = spec
        return spec

    def plan_for(self, spec: h3_layout.LayoutSpec) -> veda_plans.PlanChoice:
        return self.bundle.plans.select(spec.target.grid)

    def _tile_layout(self, spec: h3_layout.LayoutSpec,
                     shape: tiling.TileShape) -> tiling.TileLayout:
        tile_references = not self.reference.keeps_all
        key = (spec, shape, tile_references)
        layout = self._tile_layouts.get(key)
        if layout is None:
            spans = []
            if tile_references:
                spans = [tiling.TiledSpan(s.start, s.grid,
                                          tiling.least_padding_shape(s.grid))
                         for s in spec.references]
            spans.append(tiling.TiledSpan(spec.target.start, spec.target.grid,
                                          shape))
            layout = tiling.build_tile_layout(spans, spec.seq_len,
                                              self.device)
            self._tile_layouts[key] = layout
        return layout

    def _weights(self, layer: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Layer projections on the device ([H, 3D, D] bf16 each).

        Host copies stay authoritative so the predictor follows ComfyUI's
        offloading; on CUDA they are pinned once so the copy is async.
        """
        pq, pk = self.bundle.proj_q[layer], self.bundle.proj_k[layer]
        if self._pinned and not pq.is_pinned():
            try:
                pq = self.bundle.proj_q[layer] = pq.pin_memory()
                pk = self.bundle.proj_k[layer] = pk.pin_memory()
            except RuntimeError:
                self._pinned = False
        return (pq.to(self.device, non_blocking=True),
                pk.to(self.device, non_blocking=True))

    def _chunk_bytes(self) -> int:
        if self.device.type == 'cuda':
            free, _ = torch.cuda.mem_get_info(self.device)
            self.chunking['free_bytes'] = free
            return int(min(max(free // _CHUNK_SHARE, _MIN_CHUNK_BYTES),
                           _MAX_CHUNK_BYTES))
        return _DEFAULT_CHUNK_BYTES

    def reset(self) -> None:
        """Drops per-geometry caches (between sampling runs)."""
        self._tile_layouts.clear()
        self._specs = weakref.WeakKeyDictionary()
        self.stats = Stats()

    # -- the call -------------------------------------------------------

    @torch.no_grad()
    def attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                  layer: int, spec: h3_layout.LayoutSpec,
                  plan: veda_plans.TilePlan,
                  head_chunks: int = 1) -> torch.Tensor:
        """Veda block-sparse attention of one layer.

        Args:
            q, k, v: [S, H, D] (any strides; post QK-norm and RoPE).
            layer: DiT block index.
            spec: Layout spec of this sequence.
            plan: Tile plan to use (from plan_for()).
            head_chunks: Least number of head groups to split the layer
                into, from a node asking for lower peak memory (KJNodes'
                Low VRAM attention). Without it the chunk size is only
                bounded by free memory, which on a GPU with room to spare
                means one or two large chunks.

        Returns:
            [S, H, D] contiguous, q's dtype.
        """
        timer = self.timer
        seq_len, heads, dim = q.shape
        dtype = q.dtype
        if dtype not in self.backend.dtypes:
            q, k, v = (t.to(self.backend.dtypes[0]) for t in (q, k, v))
        chunks = 0
        self.chunking['heads_per_chunk'] = 0
        # Veda's own transient, measured rather than derived: the bytes
        # torch holds at the peak of a chunk, counted from before the
        # layer's output buffer so the number is everything this call
        # adds. Two counter reads per chunk, no synchronization.
        measuring = self.device.type == 'cuda'
        base = torch.cuda.memory_allocated(self.device) if measuring else 0
        out = q.new_empty(seq_len + 1, heads, dim)
        proj_q, proj_k = self._weights(layer)
        chunk_bytes = self._chunk_bytes()
        for group in plan.head_groups(layer, self.device):
            layout = self._tile_layout(spec, group.shape)
            blocks = selection.column_blocks(layout, self.generated,
                                             self.reference)
            per_head = layout.num_slots * dim * q.element_size()
            step = max(1, chunk_bytes // per_head)
            step = min(step, max(1, out.numel() * out.element_size()
                                 // (per_head * _WORKING_SET_SHARE)))
            if head_chunks > 1:
                step = min(step, -(-heads // head_chunks))
            # The widest group, not the last one: the groups of a layer
            # differ in size and the widest is what has to fit.
            self.chunking['heads_per_chunk'] = max(
                self.chunking.get('heads_per_chunk', 0),
                min(step, len(group.heads)))
            for heads_chunk in group.heads.split(step):
                chunks += 1
                with timer('gather'):
                    q_t = tiling.gather_tiles(q, layout, heads_chunk)
                    k_t = tiling.gather_tiles(k, layout, heads_chunk)
                with timer('score'):
                    scores = predictor.tile_logits(
                        predictor.pool_video_tiles(q_t, layout),
                        predictor.pool_video_tiles(k_t, layout),
                        proj_q.index_select(0, heads_chunk),
                        proj_k.index_select(0, heads_chunk))
                with timer('select'):
                    index, keep = selection.select(scores, layout, blocks)
                    del scores
                    self.stats.add(keep, layout)
                    mask = selection.block_mask(index, keep, layout)
                with timer('gather'):
                    v_t = tiling.gather_tiles(v, layout, heads_chunk)
                with timer('attend'):
                    o_t = self.backend.attend(q_t, k_t, v_t, mask, layout)
                if measuring:
                    self.chunking['workspace_bytes'] = max(
                        self.chunking.get('workspace_bytes', 0),
                        torch.cuda.memory_allocated(self.device) - base)
                del q_t, k_t, v_t, mask
                with timer('scatter'):
                    tiling.scatter_tiles_(out, o_t, layout, heads_chunk)
        self.chunking['chunks_per_layer'] = chunks
        out = out[:seq_len]
        return out if out.dtype == dtype else out.to(dtype)

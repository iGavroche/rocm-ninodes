"""The contract every attention backend implements, and its self-test.

A backend computes block-sparse attention on tile-ordered tensors for one
block mask. Everything before that (layout, predictor, selection) is shared
torch code in `veda_comfy.core` and identical on every device, so backends
stay small and independent: a backend module imports only this file, torch,
and its own kernel package. Changing one backend can therefore not break
another, which matters because no single machine can regression-test them
all.

Every backend must pass `self_test` on the actual device before it is used.
The test also checks that the result is really sparse: a kernel that
silently ignores the block mask fails it; that is not hypothetical, an
earlier FlashAttention-4 build accepted the mask and ignored it.
"""

from __future__ import annotations

import abc

import torch

from ..core import reference
from ..core import selection
from ..core import tiling


class BackendUnavailable(RuntimeError):
    """The backend cannot run here; the message is shown to the user."""


class Backend(abc.ABC):
    """Block-sparse attention on tile-ordered tensors.

    Attributes:
        name: Stable id used in logs and settings, e.g. 'triton-int8'.
        display: Name shown on the node, e.g. 'Triton INT8 (SM120)'.
        dtypes: Input dtypes the kernel takes natively; others are cast.
        tolerance: Largest pointwise error the self-test accepts, as a
            fraction of the reference's absmax. This is a property of the
            kernel's arithmetic, not a universal constant: a 16-bit kernel
            lands near 0.5%, while a quantised one is bounded by its own
            format and must say so, or the self-test rejects a correct
            kernel for being quantised.
    """

    name: str = 'backend'
    display: str = 'backend'
    dtypes: tuple[torch.dtype, ...] = (torch.bfloat16, torch.float16)
    tolerance: float = 0.02

    @abc.abstractmethod
    def attend(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
               block_mask: torch.Tensor,
               layout: tiling.TileLayout) -> torch.Tensor:
        """Block-sparse attention.

        Args:
            q, k, v: [N, H', D] contiguous, tile order (padding slots are
                zero), N = n_tiles * 128, D = 128, dtype in `dtypes`.
            block_mask: [H', n_tiles, n_tiles] bool, rows are query tiles;
                empty key tiles are never set.
            layout: Tile layout (key slot validity: `slot_valid`, `kv_ok`,
                `full_tile`).

        Returns:
            [N, H', D] in q's dtype. Rows of padding slots are unspecified.
        """

    def warmup_note(self) -> str | None:
        """What the first call does that takes a while, if anything."""
        return None


def selftest_problem(device: torch.device, dtype: torch.dtype, seed: int = 0):
    """A small problem with every hard case: history and target spans,
    partial tiles, global rows, per-row budgets.

    Returns:
        (q, k, v, block_mask, layout); q/k/v [N, 2, 128] in dtype.
    """
    # text 70 | history 1x8x16 (one full tile) | audio 40 | target 5x8x8
    # (tiled 4x4x8 -> 8x8x8 padded, partial tiles) ; 110 global rows.
    spans = [tiling.TiledSpan(70, (1, 8, 16), tiling.TileShape(1, 8, 16)),
             tiling.TiledSpan(238, (5, 8, 8), tiling.TileShape(4, 4, 8))]
    seq_len = 238 + 320
    layout = tiling.build_tile_layout(spans, seq_len, device)
    generator = torch.Generator().manual_seed(seed)
    heads, dim = 2, 128
    # A peaky softmax makes skipped blocks matter, so a kernel that ignores
    # the mask is caught.
    x = torch.randn(3, seq_len, heads, dim, generator=generator) * 1.5
    q, k, v = (tiling.gather_tiles(t.to(device=device, dtype=dtype), layout,
                                   torch.arange(heads, device=device))
               for t in x)
    scores = torch.randn(heads, layout.n_video_tiles, layout.n_video_tiles,
                         generator=generator).to(device)
    blocks = selection.column_blocks(layout, selection.Budget(tiles=2),
                                     selection.Budget(tiles=1))
    index, keep = selection.select(scores, layout, blocks)
    return q, k, v, selection.block_mask(index, keep, layout), layout


def self_test(backend: Backend, device: torch.device,
              dtype: torch.dtype = torch.bfloat16) -> None:
    """Checks a backend against the fp32 reference on `device`.

    Raises:
        BackendUnavailable: If the result is wrong or not sparse.
    """
    if dtype not in backend.dtypes:
        dtype = backend.dtypes[0]
    q, k, v, mask, layout = selftest_problem(device, dtype)
    try:
        out = backend.attend(q, k, v, mask, layout)
    except BackendUnavailable:
        raise
    except Exception as error:  # kernels fail in many ways; report, not crash
        raise BackendUnavailable(
            f'{backend.name} failed its self-test: {type(error).__name__}: '
            f'{error}') from error
    real = layout.slot_valid.bool()
    want = reference.block_sparse_attention(q, k, v, mask, layout)[real]
    dense = reference.block_sparse_attention(
        q, k, v, torch.ones_like(mask) & layout.kv_ok, layout)[real]
    got = out[real].float()
    err = (got - want.float()).abs().max().item()
    gap = (dense.float() - want.float()).abs().max().item()
    if (got - dense.float()).abs().max().item() < 0.5 * gap:
        raise BackendUnavailable(f'{backend.name} ignores the block mask on '
                                 'this device (computes dense attention)')
    limit = backend.tolerance * max(1.0, want.float().abs().max().item())
    if not err <= limit:
        raise BackendUnavailable(f'{backend.name} gives wrong results on this '
                                 f'device (max error {err:.3g}, allowed '
                                 f'{limit:.3g})')

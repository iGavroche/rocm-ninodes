"""Triton INT8 block-sparse attention: the default CUDA path.

This is the same arithmetic ComfyUI runs behind `--use-sage-attention`
(`from sageattention import sageattn`), so its accuracy is ComfyUI's, with
Veda's block sparsity on top. On an RTX 5070 it reaches ~125 TFLOPS dense
against ~45 for our bf16 CuTe kernel, and ~1.3% relative error against an
fp32 reference where FP8 costs ~5.3%.

Covers every CUDA GPU from SM80 on, because Triton does: one backend
instead of one per SM family, and no fallback behind it (see
backends/__init__.py for why that is deliberate).

Self-contained on purpose (see backends/base.py): imports only this file,
torch, and `veda_comfy.kernels.sage`.
"""

from __future__ import annotations

import functools
import sys

import torch

from . import base
from ..core import selection

_MIN_CC = (8, 0)


@functools.cache
def _kernel():
    from ..kernels.sage import sparse_int8  # pylint: disable=import-outside-toplevel
    return sparse_int8


class TritonInt8Backend(base.Backend):
    """SageAttention's INT8 arithmetic, walking only the kept tiles."""

    name = 'triton-int8'
    display = 'Triton INT8'
    dtypes = (torch.bfloat16, torch.float16)
    # INT8 Q and K with one scale per block: measured 1.3% relative error
    # on a dense problem and 2.7% pointwise on the padding-heavy self-test
    # problem, which is the format's floor rather than a fault. 5% still
    # catches the real ones (a mis-strided V read came out at 650%).
    tolerance = 0.05

    def __init__(self, label: str):
        self.display = f'Triton INT8 ({label})'

    def attend(self, q, k, v, block_mask, layout):
        index, count = selection.tile_index_list(block_mask & layout.kv_ok)
        with torch.no_grad():
            return _kernel().attend(q, k, v, index, count,
                                    layout.valid_count)

    def warmup_note(self) -> str:
        return 'compiling Triton kernels for this GPU (first run only)'


def create(info) -> base.Backend:
    if info.kind != 'cuda' or info.cc is None or info.cc < _MIN_CC:
        raise base.BackendUnavailable(
            'triton-int8 needs a CUDA GPU of SM80 or newer')
    try:
        _kernel()
    except ImportError as error:
        package = ('triton-windows' if sys.platform == 'win32'
                   else 'triton')
        raise base.BackendUnavailable(
            f'Triton is not installed ({error}); pip install {package}'
        ) from error
    return TritonInt8Backend(info.family.upper())

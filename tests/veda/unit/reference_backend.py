"""An exact backend over `core.reference`, for tests only.

Every shipped backend needs a GPU, so CPU CI would otherwise exercise none
of the Backend contract: not `self_test`, not the engine's chunking and
scatter. This stands in. It is deliberately not in `veda_comfy.backends` -
it is O(N^2) and far too slow to offer anyone - and it is exact, so a test
that fails here is about the code under test and never about precision.
"""

from __future__ import annotations

import torch

from veda_comfy.backends import base
from veda_comfy.core import reference


class ReferenceBackend(base.Backend):
    """Exact fp32 block-sparse attention."""

    name = 'reference'
    display = 'reference (fp32)'
    dtypes = (torch.float32, torch.bfloat16, torch.float16)

    def attend(self, q, k, v, block_mask, layout):
        return reference.block_sparse_attention(q, k, v, block_mask, layout)

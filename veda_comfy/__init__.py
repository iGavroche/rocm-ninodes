"""Veda learned block-sparse attention for MiniMax-H3 in ComfyUI.

Layout:
  core/      device-agnostic torch: tiling, plans, predictor, selection,
             the per-call engine (no ComfyUI imports)
  backends/  one module per kernel family, isolated from each other
  kernels/   sage/: the Triton INT8 block-sparse kernel, derived from
             SageAttention v1; see docs/features/int8_kernel.md
  nodes.py, comfy_patch.py, status.py  the ComfyUI side
"""

__version__ = '0.3.0'

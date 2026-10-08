"""Test setup: import `veda_comfy` from the repo, find ComfyUI if present.

ComfyUI-backed tests need a ComfyUI checkout: set COMFYUI_ROOT, or run the
tests from `ComfyUI/custom_nodes/<this repo>`; they are skipped otherwise.
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(_HERE))  # rocm-ninodes repo root
sys.path.insert(0, ROOT)
sys.path.insert(0, _HERE)
# The repo's top-level tests/conftest.py stubs `comfy` and `folder_paths`
# into sys.modules for the rocm_nodes tests; these tests import the real
# ComfyUI instead, so drop the stubs before anything imports them.
for _name in [n for n in list(sys.modules)
              if n == 'comfy' or n.startswith('comfy.')
              or n in ('folder_paths', 'latent_preview')]:
    del sys.modules[_name]


def _comfyui_root():
    candidates = [os.environ.get('COMFYUI_ROOT'),
                  os.path.dirname(os.path.dirname(ROOT))]
    for path in candidates:
        if path and os.path.isfile(os.path.join(path, 'comfy', 'ops.py')):
            return path
    return None


def _pin_comfyui_to_cpu():
    """Makes ComfyUI report a CPU device, whatever torch was installed.

    ComfyUI only leaves its default `cpu_state = CPUState.GPU` when `--cpu`
    is passed or MPS is detected (comfy/model_management.py). A CPU-only
    torch build matches neither, so `get_torch_device()` falls through to
    `torch.cuda.current_device()` and raises "Torch not compiled with CUDA
    enabled" - which is what CI's Linux and Windows runners hit. Setting
    the flag before anything imports `model_management` also makes the
    three CI platforms agree: without it macOS runs these tests against
    MPS and probes the mlx backend, and Linux does not.
    """
    import comfy.cli_args
    comfy.cli_args.args.cpu = True


COMFYUI_ROOT = _comfyui_root()
if COMFYUI_ROOT is not None:
    if COMFYUI_ROOT not in sys.path:
        sys.path.append(COMFYUI_ROOT)
    _pin_comfyui_to_cpu()

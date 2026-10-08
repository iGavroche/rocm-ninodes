"""Backend registry: which kernel runs on which device.

    CUDA sm80 and up    triton-int8
    Apple MPS           mlx
    anything else       nothing (the node runs the model's own attention)

One kernel per device family, no fallback chain and nothing for the user
to pick. Earlier there were five: four FA4 CuTe backends (one per SM
family, plus an FP8 variant) behind a FlexAttention and a plain-torch
path. `triton-int8` replaced all of them because it is better on every
axis that mattered - 1.50x the FA4 kernel's speed on the same sparse
problem, the same accuracy as ComfyUI's own low-precision attention, and
SM80-and-up coverage from a single Triton kernel - and because a chain of
fallbacks silently hands a user a slower or less accurate kernel than the
one they think they are running. See docs/features/int8_kernel.md.

Backend modules are imported lazily and only here, so a missing Triton
costs one line in the report, never the node.
"""

from __future__ import annotations

import dataclasses
import importlib
import logging
import threading
from collections.abc import Callable

import torch

from . import base
from .. import hardware

_MODULES = {'triton-int8': 'triton_int8', 'mlx': 'mlx_gather'}
_MIN_CC = (8, 0)


def _load(name: str, info: hardware.DeviceInfo) -> base.Backend:
    module = importlib.import_module(f'.{_MODULES[name]}', __name__)
    return module.create(info)


@dataclasses.dataclass
class Resolution:
    """Outcome of picking a backend for one device.

    Attributes:
        device: The device info.
        backend: The backend to use, None if nothing works (run dense).
        attempts: (candidate, 'ok' or why it was skipped), in order.
    """

    device: hardware.DeviceInfo
    backend: base.Backend | None
    attempts: list[tuple[str, str]]

    def report(self) -> str:
        return '; '.join(f'{name}: {status}' for name, status in self.attempts)


def candidates(info: hardware.DeviceInfo) -> list[str]:
    """The backend to try on a device; empty if none applies."""
    if info.kind == 'cuda' and info.cc:
        return ['triton-int8'] if info.cc >= _MIN_CC else []
    if info.kind == 'mps':
        return ['mlx']
    return []


_LOCK = threading.RLock()
_RESOLVED: dict[str, Resolution] = {}


def resolve(device: torch.device,
            notify: Callable[[str], None] | None = None) -> Resolution:
    """Loads and self-tests the backend for `device` (cached).

    Args:
        device: The device attention runs on.
        notify: Called with a short status line before slow steps (kernel
            compilation on the first call).
    """
    info = hardware.describe(device)
    key = str(device)
    with _LOCK:
        if key in _RESOLVED:
            return _RESOLVED[key]
        attempts: list[tuple[str, str]] = []
        chosen = None
        for name in candidates(info):
            try:
                backend = _load(name, info)
                note = backend.warmup_note()
                if notify is not None and note:
                    notify(f'{backend.display}: {note}')
                base.self_test(backend, device)
            except base.BackendUnavailable as error:
                attempts.append((name, str(error)))
                continue
            except Exception as error:  # a broken backend must not crash
                logging.warning('Veda: backend %s failed to load', name,
                                exc_info=True)
                attempts.append((name, f'{type(error).__name__}: {error}'))
                continue
            attempts.append((backend.name, 'ok'))
            chosen = backend
            break
        resolution = Resolution(info, chosen, attempts)
        _RESOLVED[key] = resolution
        return resolution


def probe(device: torch.device) -> list[tuple[str, str, str | None]]:
    """(name, display name, error or None) for the device's backend,
    without compiling or self-testing (cheap; for the status shown before
    sampling)."""
    info = hardware.describe(device)
    out = []
    for name in candidates(info):
        try:
            backend = _load(name, info)
            out.append((backend.name, backend.display, None))
        except Exception as error:  # report every failure the same way
            out.append((name, name, str(error)))
    return out


def reset() -> None:
    """Forgets resolutions (tests, or after installing kernels)."""
    with _LOCK:
        _RESOLVED.clear()

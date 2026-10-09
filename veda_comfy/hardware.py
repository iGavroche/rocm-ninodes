"""What the attention runs on: device kind, SM family and product subtype.

The SM family decides whether a kernel exists at all (see
`backends/`); the subtype only changes labels, memory heuristics and install
hints. SM120 is split on purpose: compute capability 12.0 covers GeForce
RTX 50 and the RTX PRO Blackwell cards, 12.1 is GB10 (DGX Spark, aarch64,
unified memory, usually CUDA 13).
"""

from __future__ import annotations

import dataclasses
import functools
import platform
import sys

import torch

# (major, minor) -> SM family. The family is what the node reports and
# what hardware notes are filed under; the Triton kernel itself compiles
# per device, so nothing dispatches on it.
_FAMILIES = {
    (8, 0): 'sm80', (8, 6): 'sm86', (8, 7): 'sm87', (8, 9): 'sm89',
    (9, 0): 'sm90',
    (10, 0): 'sm100', (10, 3): 'sm103', (11, 0): 'sm110',
    (12, 0): 'sm120', (12, 1): 'sm121',
}


@dataclasses.dataclass(frozen=True)
class DeviceInfo:
    """Static facts about one torch device.

    Attributes:
        kind: 'cuda', 'mps', 'cpu' or another torch device type.
        index: Device index for CUDA, else None.
        name: Marketing name, e.g. 'NVIDIA GeForce RTX 4090'.
        cc: CUDA compute capability (major, minor), None off CUDA.
        family: 'sm89', 'sm120', 'sm121', ..., 'mps', 'cpu', 'rocm'.
        subtype: Product class, e.g. 'rtx40', 'rtx-pro-blackwell',
            'dgx-spark', 'apple-silicon'.
        os: 'linux', 'windows' or 'darwin'.
        machine: CPU architecture, e.g. 'x86_64', 'aarch64', 'arm64'.
        total_memory: Device memory in bytes (None when unknown).
        unified_memory: GPU and CPU share memory (Apple silicon, GB10).
        cuda: torch's CUDA version string, e.g. '12.8', None off CUDA.
    """

    kind: str
    index: int | None
    name: str
    cc: tuple[int, int] | None
    family: str
    subtype: str
    os: str
    machine: str
    total_memory: int | None
    unified_memory: bool
    cuda: str | None

    @property
    def short_name(self) -> str:
        """Marketing name without the vendor prefix, e.g. 'RTX 5070'."""
        return self.name.replace('NVIDIA ', '').replace('GeForce ', '')

    @property
    def label(self) -> str:
        if self.kind == 'cuda':
            short = self.name.replace('NVIDIA ', '').replace('GeForce ', '')
            return f'{short} ({self.family.upper()})'
        if self.kind == 'mps':
            return f'{self.name} (Metal)'
        return self.name

    @property
    def cuda_major(self) -> int | None:
        return int(self.cuda.split('.')[0]) if self.cuda else None


def _os() -> str:
    if sys.platform.startswith('win'):
        return 'windows'
    if sys.platform == 'darwin':
        return 'darwin'
    return 'linux'


def _cuda_subtype(cc: tuple[int, int], name: str) -> str:
    upper = name.upper()
    if cc == (12, 1) or 'GB10' in upper:
        return 'dgx-spark'
    if cc == (12, 0):
        return 'rtx-pro-blackwell' if 'RTX PRO' in upper else 'rtx50'
    if cc[0] in (10, 11):
        return {(10, 3): 'b300', (11, 0): 'thor'}.get(cc, 'b200')
    if cc == (9, 0):
        return 'hopper'
    if cc == (8, 9):
        return 'rtx40' if 'RTX 40' in upper else 'ada-pro'
    if cc == (8, 6):
        return 'rtx30' if 'RTX 30' in upper else 'ampere-pro'
    if cc == (8, 7):
        return 'orin'
    if cc == (8, 0):
        return 'a100'
    return 'other'


@functools.lru_cache(maxsize=16)
def _describe(device_str: str) -> DeviceInfo:
    device = torch.device(device_str)
    os_name, machine = _os(), platform.machine().lower()
    if device.type == 'cuda':
        index = device.index if device.index is not None else (
            torch.cuda.current_device())
        props = torch.cuda.get_device_properties(index)
        cc = (props.major, props.minor)
        if torch.version.hip:
            gcn = getattr(props, 'gcnArchName', '').split(':')[0]
            return DeviceInfo('cuda', index, props.name, cc,
                              gcn or 'rocm', 'rocm', os_name, machine,
                              props.total_memory, True, None)
        family = _FAMILIES.get(cc, f'sm{cc[0]}{cc[1]}')
        subtype = _cuda_subtype(cc, props.name)
        unified = bool(getattr(props, 'is_integrated', False)) or (
            subtype == 'dgx-spark')
        return DeviceInfo('cuda', index, props.name, cc, family, subtype,
                          os_name, machine, props.total_memory, unified,
                          torch.version.cuda)
    if device.type == 'mps':
        # platform.processor() gives "Apple M3 Pro" on macOS, "arm"
        # elsewhere; a process spawn for a display label is not worth it.
        name = 'Apple silicon'
        if os_name == 'darwin':
            name = platform.processor() or name
        return DeviceInfo('mps', None, name, None, 'mps', 'apple-silicon',
                          os_name, machine, None, True, None)
    return DeviceInfo(device.type, device.index, device.type.upper(), None,
                      device.type, device.type, os_name, machine, None,
                      device.type == 'cpu', None)


def describe(device: torch.device | str) -> DeviceInfo:
    """DeviceInfo of a torch device (cached)."""
    device = torch.device(device)
    if device.type == 'cuda' and device.index is None:
        device = torch.device('cuda', torch.cuda.current_device())
    return _describe(str(device))

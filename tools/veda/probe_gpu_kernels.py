"""Reports what this GPU and the installed Triton can do, for kernel work.

    python tools/probe_gpu_kernels.py

Prints the device, the tensor-core throughput of the dtypes the kernel
could use, and which Triton features are available here - notably TMA,
whose shape differs between consumer and datacenter Blackwell.

The point of this tool is to answer capability questions by asking the
hardware instead of reading someone's `assert`. An upstream gate that says
"arch 10 only" usually means "only arch 10 has an implementation", not
"the silicon cannot". Run this on every new architecture before concluding
a path is unavailable.
"""

from __future__ import annotations

import time

import torch


def _tflops(fn, flops: float, repeat: int = 20) -> float:
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(repeat):
        fn()
    torch.cuda.synchronize()
    return flops * repeat / (time.perf_counter() - start) / 1e12


def _throughput(device) -> None:
    n = 4096
    flops = 2.0 * n ** 3
    print('\ndense 4096^3 GEMM, by operand dtype')
    for label, dtype in (('bf16', torch.bfloat16), ('fp16', torch.float16)):
        a = torch.randn(n, n, device=device, dtype=dtype)
        b = torch.randn(n, n, device=device, dtype=dtype)
        print(f'  {label:5s} {_tflops(lambda: a @ b, flops):7.1f} TFLOPS')
    a8 = torch.randint(-127, 127, (n, n), device=device, dtype=torch.int8)
    b8 = a8.t().contiguous().t()
    try:
        print(f'  int8  {_tflops(lambda: torch._int_mm(a8, b8), flops):7.1f} '
              'TOPS')
    except Exception as error:
        print(f'  int8  unavailable: {type(error).__name__}: {error}')
    try:
        af = torch.randn(n, n, device=device).to(torch.float8_e4m3fn)
        bf = af.t().contiguous().t()
        scale = torch.ones((), device=device)
        rate = _tflops(lambda: torch._scaled_mm(af, bf, scale, scale), flops)
        print(f'  fp8   {rate:7.1f} TFLOPS')
    except Exception as error:
        print(f'  fp8   unavailable: {type(error).__name__}: {error}')


def _triton_features(capability) -> None:
    print('\nTriton')
    try:
        import triton
        import triton.language as tl
    except ImportError as error:
        print(f'  not installed: {error}')
        return
    print(f'  version {triton.__version__}')
    for name in ('make_tensor_descriptor', 'async_task', 'range'):
        print(f'  tl.{name}: {"yes" if hasattr(tl, name) else "no"}')
    major = capability[0] if capability else 0
    if major >= 9:
        # TMA exists from Hopper on, but the destination differs: SM90 and
        # the datacenter Blackwells (SM100/103) have thread block clusters
        # and distributed shared memory, so they can target
        # `.shared::cluster` and multicast a tile to several CTAs. Consumer
        # Blackwell (SM120/121) has TMA but no clusters, so every bulk copy
        # must land in `.shared::cta` and num_ctas must stay 1. Asking for a
        # cluster there is a compile error at best and a silent fallback at
        # worst.
        cluster = major in (9, 10)
        where = ('shared::cluster (thread block clusters available)'
                 if cluster else 'shared::cta only (no clusters, '
                 'keep num_ctas at 1)')
        print(f'  TMA: yes, destination {where}')
    else:
        print('  TMA: no (needs SM90 or newer)')


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit('needs a CUDA GPU')
    device = torch.device('cuda')
    capability = torch.cuda.get_device_capability(device)
    props = torch.cuda.get_device_properties(device)
    print(f'{props.name}: SM{capability[0]}{capability[1]}, '
          f'{props.multi_processor_count} SMs, '
          f'{props.total_memory / 2**30:.1f} GiB, '
          f'torch {torch.__version__}')
    _throughput(device)
    _triton_features(capability)


if __name__ == '__main__':
    main()

# ROCmMemoryOptimizer

Display name: `ROCm Memory Optimizer` · Category: `ROCm Ninodes/Memory` · Legacy class names: `ROCMMemoryOptimizer`

Source: `rocm_nodes/core/monitors.py`

## What it does

A memory status panel with an optional cleanup action. Each execution increments an internal counter, reports allocated/reserved/free/total, and — only when the counter is a multiple of `cleanup_frequency` — runs garbage collection and/or `torch.cuda.empty_cache()`. `aggressive` additionally calls `torch.cuda.synchronize()`. It returns three text blocks; it does not pass data through, so it must be pulled into execution by a consumer of its outputs.

## Inputs

| Name | Type | Default | Range / options | Notes |
|---|---|---|---|---|
| `optimization_level` | combo | `balanced` | `conservative`, `balanced`, `aggressive` | Only `aggressive` does anything extra: `torch.cuda.synchronize()`. `conservative` and `balanced` behave identically. |
| `enable_gc` | BOOLEAN | `True` | — | `gc.collect()`, gated on the frequency counter. |
| `clear_cache` | BOOLEAN | `True` | — | `torch.cuda.empty_cache()`, gated on the frequency counter. |
| `cleanup_frequency` | INT | `10` | 1 … 100 | Cleanup runs when `operation_count % cleanup_frequency == 0`, i.e. on the 10th execution by default. |

`FUNCTION = optimize_memory`. `RETURN_TYPES = ("STRING", "STRING", "STRING")`. The counter lives on the node instance (`__init__` sets `self.operation_count = 0`).

## Outputs

| Name | Type | Notes |
|---|---|---|
| `MEMORY_STATUS` | STRING | Allocated / Reserved / Free / Total in GB (`free = total - reserved`), or `CUDA not available - no GPU memory to optimize`. |
| `OPTIMIZATION_LOG` | STRING | Level name plus which actions actually ran this execution (`Garbage collection performed`, `CUDA cache cleared`, `CUDA synchronization performed`). |
| `RECOMMENDATIONS` | STRING | Threshold-based: > 80% allocated → high-usage warnings; > 60% → moderate; otherwise "Memory usage is good". |

## gfx1151 notes

- On gfx1151 the reported "total" is the unified memory pool as torch sees it, not a discrete VRAM budget; other nodes in this pack use `psutil` system-wide available RAM for APU decisions, this node uses only `torch.cuda` counters.
- The cleanup calls are the same primitives as `gentle_memory_cleanup()` in `utils/memory.py`, but `synchronize()` only happens in `aggressive` mode.
- Frequent `empty_cache()` on an APU is counterproductive — the pack's general stance (see `utils/memory.py`: "less aggressive for mature ROCm drivers") is to let the allocator work. Keep `cleanup_frequency` high.
- No model is loaded or unloaded; this node cannot free an unused model from VRAM — that is ComfyUI's own model management.

## Typical wiring

```
ROCmMemoryOptimizer -> MEMORY_STATUS / OPTIMIZATION_LOG / RECOMMENDATIONS -> Show Text
```

Because it has no data inputs, it only executes when one of its STRING outputs feeds a node that ComfyUI considers part of the output chain (e.g. a text display node).

## Troubleshooting

- Nothing happens on the first runs: cleanup is gated by `cleanup_frequency` (default 10 executions). Set it to 1 to act every time.
- `OPTIMIZATION_LOG` lists an action only when it actually ran, so a bare header line means the frequency gate did not trigger.
- Node outputs are strings; without a consumer the node is pruned from the execution graph and never runs.
- `Free` here is `total - reserved`, which is usually smaller than what ComfyUI's own memory manager considers available.

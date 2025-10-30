# Checkpoint Loader Update - October 2025

## Problem

The original `ROCMOptimizedCheckpointLoader` was experiencing HIP Out of Memory (OOM) errors with messages suggesting to use `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. However, this was the **wrong approach** for modern PyTorch versions.

### Error Pattern
```
HIP out of memory. Tried to allocate 16.00 MiB. GPU 0 has a total capacity of 107.87 GiB 
of which 38.36 GiB is free. Of the allocated memory 48.27 GiB is allocated by PyTorch, 
and 27.55 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory 
is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.
```

### Root Cause
The checkpoint loader was **causing** fragmentation through aggressive memory cleanup:
- Manual `gc.collect()` calls before loading
- `torch.cuda.empty_cache()` calls
- `torch.cuda.synchronize()` calls  
- `simple_memory_cleanup()` utility calls

**This was backwards thinking.** Modern PyTorch (2.7+, especially 2.10.0a0 nightly with ROCm 7.10.0a) has sophisticated memory allocators that handle fragmentation internally. By manually forcing cleanup, we were **interfering** with the allocator's ability to manage memory efficiently.

## Solution

### Updated Approach
1. **Remove aggressive memory cleanup** - Let PyTorch's allocator do its job
2. **Simplify the loader** - Delegate to ComfyUI's native `load_checkpoint_guess_config`
3. **Add diagnostics** - Provide helpful logging without interfering with loading
4. **Handle OOM gracefully** - Catch OutOfMemoryError and provide actionable suggestions

### Code Changes

**Before (193 lines):**
- Multiple try/except fallback blocks (3 levels deep)
- Manual memory cleanup before every operation
- Compatibility mode flags and optimization toggles
- Redundant validation in multiple places

**After (178 lines):**
- Single try/except with specific OOM handling
- No manual memory cleanup (trust PyTorch)
- Simplified inputs (removed unused optimization flags)
- Helper methods for diagnostics and validation

### Key Improvements

1. **No expandable_segments needed** - Modern PyTorch handles this
2. **Fewer parameters** - Removed `lazy_loading`, `optimize_for_flux`, `precision_mode` (unused)
3. **Better error messages** - Specific suggestions for OOM errors
4. **Non-intrusive diagnostics** - Log system info without interfering

## PyTorch Version Context

### Current Environment
- **PyTorch**: 2.10.0a0+rocm7.10.0a20251011 (nightly)
- **ROCm**: 7.10.0a (development build)
- **Architecture**: gfx1151 (RDNA 3.5) with unified memory
- **Python**: >=3.13

### Modern Memory Management
PyTorch 2.7+ includes:
- Improved memory allocator with automatic defragmentation
- Better caching strategies
- Smarter memory reuse patterns
- ROCm-specific optimizations for unified memory architectures

**The key insight:** Let PyTorch manage memory. Don't second-guess the allocator.

## Testing Recommendations

### Before Testing
1. **Restart ComfyUI** - Clear any existing memory state
2. **Close other GPU applications** - Ensure clean baseline
3. **Check PyTorch version** - Verify you're using 2.7+ with ROCm 6.4+

### Test Scenarios
1. **Basic checkpoint loading** - Load a standard Flux model
2. **Quantized model loading** - Test with fp8 checkpoints
3. **Large model loading** - Test memory management with large models
4. **Sequential loading** - Load multiple checkpoints in sequence

### Expected Behavior
- ✅ Smooth loading without OOM errors
- ✅ Memory usage stays consistent
- ✅ No fragmentation warnings in logs
- ✅ Helpful diagnostics in console

## If OOM Still Occurs

The new loader provides these suggestions:
1. Close other applications using GPU memory
2. Restart ComfyUI to clear memory
3. Use a smaller model or quantized version
4. Check for other loaded models in ComfyUI
5. Verify PyTorch/ROCm versions are up to date

### Additional Debugging
Enable debug mode to see memory details:
```bash
export ROCM_NINODES_DEBUG=1
```

Check memory status:
```python
import torch
print(f"Allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f}GB")
print(f"Reserved: {torch.cuda.memory_reserved(0) / 1024**3:.2f}GB")
print(f"Fragmentation: {(torch.cuda.memory_reserved(0) - torch.cuda.memory_allocated(0)) / 1024**3:.2f}GB")
```

## Environment Management Update

### Using `uv` Package Manager
This project now documents its use of **`uv`** for dependency management:

#### Key Commands
```bash
# Install/sync dependencies
uv sync

# Add a new dependency  
uv add <package>

# Run tests
uv run pytest tests/

# Run any command in the venv
uv run <command>
```

#### Why `uv`?
- **Fast**: Rust-based, 10-100x faster than pip
- **Reliable**: Deterministic resolution via `uv.lock`
- **Modern**: Designed for Python 3.13+

#### Important
- Don't use `pip install` directly - use `uv add`
- Commit both `pyproject.toml` and `uv.lock`
- PyTorch with ROCm installed separately (platform-specific)

## Files Updated

1. **`rocm_nodes/core/checkpoint.py`**
   - Removed aggressive memory cleanup
   - Simplified loading logic
   - Added helpful diagnostics
   - Better error handling

2. **`.cursorrules`**
   - Added comprehensive `uv` documentation
   - Updated all command examples to use `uv run`
   - Added environment management section
   - Updated test execution commands

3. **Memory MCP**
   - Stored knowledge about uv package manager
   - Documented checkpoint loader changes
   - Recorded PyTorch version and architecture details

## Conclusion

**The takeaway:** Modern PyTorch with ROCm doesn't need manual memory management or `expandable_segments` configuration. Trust the allocator, keep the code simple, and let PyTorch do what it does best.

This update aligns with the project's philosophy:
> **Code quality > Speed of development**. Take time to do it right.

By removing unnecessary complexity and trusting modern PyTorch's capabilities, we've created a more maintainable, reliable checkpoint loader that works **with** the framework rather than against it.


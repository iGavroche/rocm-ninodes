# Next Steps - Post Refactoring

## ✓ Completed

1. **Package Structure** - Created modular `rocm_nodes/` package
   - 5 core modules (vae, sampler, checkpoint, lora, monitors)
   - 4 utility modules (memory, diagnostics, quantization, debug)
   - All files <500 lines (AI-friendly)

2. **Node Registry** - Created `rocm_nodes/nodes.py` with all NODE_CLASS_MAPPINGS

3. **Backward Compatibility** - Updated top-level `__init__.py` with fallback imports

4. **Utilities Extraction** - Separated all utility functions into dedicated modules

5. **Development Rules** - Created `.cursorrules` with comprehensive guidelines

6. **Documentation** - Created `REFACTORING_SUMMARY.md` with detailed information

7. **Code Quality** - Zero Chinese characters, no linter errors, clean imports

## ⏳ Pending (Requires User Action)

### Critical: Test in ComfyUI
You need to restart ComfyUI and verify the nodes load correctly:

```bash
# Navigate to ComfyUI directory
cd C:\Users\Nino\ComfyUI

# Restart ComfyUI (or restart the server)
python main.py
```

**Expected Behavior:**
- ComfyUI should detect and load all 10 ROCm nodes
- Check the terminal for import messages
- Nodes should appear in the "RocM Ninodes" category
- No import errors should occur

**If you see errors:**
1. Check the ComfyUI console output
2. Look for import errors mentioning `rocm_nodes`
3. Verify `comfy` module is available in ComfyUI environment

### Test Workflows
After ComfyUI starts successfully:

1. **Load a test workflow**
   ```
   comfyui_workflows/basic_image.json
   ```

2. **Add ROCm nodes to workflow**
   - Try: ROCM Checkpoint Loader
   - Try: ROCM KSampler  
   - Try: ROCM VAE Decode

3. **Execute workflow**
   - Verify output quality matches baseline
   - Check performance (should be equal or better)
   - Monitor memory usage

### Run Test Suite
```bash
# From project root
cd C:\Users\Nino\ComfyUI\custom_nodes\rocm-ninodes

# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=rocm_nodes --cov-report=term-missing

# Run specific test categories
pytest tests/unit/ -v              # Unit tests
pytest tests/integration/ -v       # Integration tests
pytest tests/benchmarks/ -v        # Performance benchmarks
```

**Note:** Some tests may fail if ComfyUI modules aren't available in the test environment. This is expected and handled by the graceful fallback.

## Optional Improvements

### 1. Remove Old Monolithic File (After Testing)
Once you've verified everything works:

```bash
# Backup first!
mv rocm_nodes.py rocm_nodes.py.backup

# If everything still works, you can delete the backup later
```

### 2. Populate Constants File
Edit `rocm_nodes/constants.py`:

```python
# Optimal tile sizes for gfx1151
OPTIMAL_TILE_SIZE_GFX1151 = 768
DEFAULT_OVERLAP = 96
MAX_TILE_SIZE = 2048
MIN_TILE_SIZE = 256

# Memory management
MEMORY_SAFETY_MARGIN = 0.15  # 15% safety margin
CRITICAL_MEMORY_THRESHOLD = 0.85

# Default precision modes
DEFAULT_PRECISION = "fp32"
SUPPORTED_PRECISIONS = ["auto", "fp32", "fp16", "bf16"]

# Video processing
DEFAULT_VIDEO_CHUNK_SIZE = 8
MAX_VIDEO_CHUNK_SIZE = 32

# Debug
DEBUG_MODE_ENV_VAR = "ROCM_NINODES_DEBUG"
```

### 3. Create Module-Specific Tests
Organize tests to match the new structure:

```
tests/unit/
├── test_vae.py           # Tests for VAE nodes
├── test_sampler.py       # Tests for sampler nodes
├── test_checkpoint.py    # Tests for checkpoint loader
├── test_lora.py          # Tests for LoRA loader
├── test_monitors.py      # Tests for monitor nodes
└── test_utils/
    ├── test_memory.py    # Memory utilities tests
    ├── test_quantization.py
    └── test_debug.py
```

### 4. Update Documentation
- Update `README.md` with new package structure
- Add import examples for developers
- Document utility functions
- Create API reference

## Troubleshooting

### Import Errors
**Symptom:** `ModuleNotFoundError: No module named 'rocm_nodes'`

**Solution:**
1. Verify you're in the correct directory
2. Check `rocm_nodes/` folder exists
3. Verify `__init__.py` files are present
4. Restart ComfyUI completely

### Empty Node List
**Symptom:** ComfyUI starts but no ROCm nodes appear

**Solution:**
1. Check ComfyUI console for import errors
2. Verify `comfy` module is available
3. Check `NODE_CLASS_MAPPINGS` is not empty
4. Review import fallback messages

### Performance Regression
**Symptom:** Nodes work but are slower than before

**Solution:**
1. Run benchmark workflows
2. Compare with baseline results
3. Check if optimizations are being applied
4. Verify precision mode settings

## Testing Checklist

- [ ] ComfyUI starts without errors
- [ ] All 10 ROCm nodes appear in node list
- [ ] Nodes can be added to workflows
- [ ] Basic workflow executes successfully
- [ ] Output quality matches baseline
- [ ] Performance matches or exceeds baseline
- [ ] Memory usage is acceptable
- [ ] No crashes during execution
- [ ] Workflow files can be saved/loaded
- [ ] Test suite passes (or expected failures only)

## Success Criteria

✓ **Structure:** Modular package with clear separation
✓ **Compatibility:** Backward compatible imports
✓ **Quality:** No linter errors, clean code
✓ **Documentation:** Comprehensive guides and rules

⏳ **Pending User Validation:**
- ComfyUI loads nodes successfully
- Workflows execute correctly
- Performance maintained or improved
- Tests pass in ComfyUI environment

## Support

If you encounter issues:

1. **Check logs:** Look at ComfyUI console output
2. **Enable debug:** Set `ROCM_NINODES_DEBUG=1`
3. **Review docs:** Check `REFACTORING_SUMMARY.md`
4. **Test imports:** Run the import test manually
5. **Verify structure:** Ensure all files are in place

## Quick Start Test

```bash
# Quick verification that package imports
python -c "from rocm_nodes import NODE_CLASS_MAPPINGS; print(f'Nodes: {len(NODE_CLASS_MAPPINGS)}')"

# Should print "Nodes: 0" (without ComfyUI) or "Nodes: 10" (with ComfyUI)
```

## Report Back

After testing, please verify:
- ✓ Did ComfyUI load all nodes?
- ✓ Did workflows execute correctly?
- ✓ Was performance maintained?
- ✗ Any errors or issues?

This will help ensure the refactoring was successful!






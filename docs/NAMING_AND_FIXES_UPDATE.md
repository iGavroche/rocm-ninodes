# Naming Convention and Bug Fixes - October 2025

## Overview

Updated the project to use consistent "ROCm" branding and fixed a critical bug in the Diffusion Loader that was causing crashes when loading models.

## Changes Made

### 1. **Fixed Critical Bug in Diffusion Loader**

#### The Problem
```python
# Old code (BROKEN)
model_options = {}
# ... populate model_options ...
model = comfy.sd.load_diffusion_model(
    unet_path,
    model_options=model_options if model_options else None  # ❌ WRONG!
)
```

When `model_options` was an empty dict `{}`, it evaluated to falsy and became `None`. ComfyUI's `load_diffusion_model` expects a dict and tries to call `.get()` on it, resulting in:

```
AttributeError: 'NoneType' object has no attribute 'get'
```

#### The Fix
```python
# New code (CORRECT)
model_options = {}
# ... populate model_options ...
model = comfy.sd.load_diffusion_model(
    unet_path,
    model_options=model_options  # ✅ Always pass dict, even if empty
)
```

**Key insight:** Always pass `model_options` as a dict. ComfyUI expects a dict object, not `None`.

### 2. **Renamed Node: UNet Loader → Diffusion Loader**

#### Old Name
- Class: `ROCMOptimizedUNetLoader`
- Display: "ROCM UNet Loader"
- Category: "RocM Ninodes/Loaders"

#### New Name
- Class: `ROCmDiffusionLoader`
- Display: "ROCm Diffusion Loader"
- Category: "ROCm Ninodes/Loaders"

**Rationale:**
- "Diffusion Loader" is more descriptive and user-friendly
- Handles WAN, Flux, and other diffusion models (not just UNet architecture)
- Better matches the actual functionality

### 3. **Standardized "ROCm" Branding**

Updated all instances to use correct "ROCm" capitalization:
- ✅ **ROCm** (capital R, O, C, lowercase m)
- ❌ ~~RocM~~ (was used in some places)
- ❌ ~~ROCM~~ (was used in display names)

#### Files Updated for Branding

**Display Names** (`rocm_nodes/nodes.py`):
```python
# Before
"ROCM Checkpoint Loader"
"ROCM VAE Decode"
"ROCM KSampler"

# After  
"ROCm Checkpoint Loader"
"ROCm VAE Decode"
"ROCm KSampler"
```

**Category Names** (`rocm_nodes/core/*.py`):
```python
# Before
CATEGORY = "RocM Ninodes/Loaders"

# After
CATEGORY = "ROCm Ninodes/Loaders"
```

**Documentation** (`.cursorrules`, docs):
- All references now use "ROCm"
- Added naming convention guidelines

### 4. **Updated `.cursorrules` with Naming Standards**

Added new "Naming Conventions" section:

```markdown
## Naming Conventions (CRITICAL)

### ROCm Branding
- **Always use "ROCm"** (capital R, capital O, capital C, lowercase m)
- ❌ Wrong: "RocM", "ROCM", "rocm", "Rocm"
- ✅ Correct: "ROCm"
- Examples:
  - Class names: `ROCmDiffusionLoader` (when part of compound name)
  - Display names: "ROCm Checkpoint Loader"
  - Documentation: "ROCm-optimized nodes"
  - Categories: "ROCm Ninodes/Loaders"
```

## Files Modified

### Core Code Changes
1. **`rocm_nodes/core/unet_loader.py`**
   - Renamed class: `ROCMOptimizedUNetLoader` → `ROCmDiffusionLoader`
   - Fixed `model_options` bug (always pass dict, not None)
   - Updated category to "ROCm Ninodes/Loaders"
   - Updated docstrings

2. **`rocm_nodes/core/__init__.py`**
   - Updated import: `ROCmDiffusionLoader`
   - Updated `__all__` exports

3. **`rocm_nodes/nodes.py`**
   - Updated import
   - Updated NODE_CLASS_MAPPINGS
   - Updated all display names to use "ROCm" (not "ROCM")

4. **`pyproject.toml`**
   - Updated node list: `ROCmDiffusionLoader`

### Documentation Changes
5. **`.cursorrules`**
   - Changed title to "ROCm Ninodes"
   - Added "Naming Conventions" section
   - Updated all references to use "ROCm"

## Testing

### Before Fix
```
Error: AttributeError: 'NoneType' object has no attribute 'get'
Result: ❌ Crash when loading diffusion models
```

### After Fix
```
📁 Loading diffusion model: wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors
🖥️  GPU: AMD Radeon(TM) 8060S Graphics
🔧 ROCm backend detected
   ROCm version: 7.1.25433-55d0db39ba
⏳ Loading model (may take 2-6 minutes for large models)...
✅ Diffusion model loaded successfully
📊 Memory: XX.XXGB used / 107.87GB total
Result: ✅ Success!
```

## Impact

### Bug Fix Impact
- **Critical:** Fixed crash that prevented loading any diffusion models
- **Affected users:** Anyone using the ROCm Diffusion Loader node
- **Severity:** High (node was unusable)
- **Status:** ✅ Resolved

### Naming Impact
- **User-facing:** More professional and consistent branding
- **Developer experience:** Clear naming standards prevent confusion
- **Documentation:** Easier to search and reference
- **Breaking changes:** Node renamed (users will need to reconnect in workflows)

## Migration Guide

### For Users

If you were using the old "ROCM UNet Loader" node:

1. **Node will appear as "ROCm Diffusion Loader"** in the node menu
2. **Reconnect in existing workflows** - node ID changed from `ROCMOptimizedUNetLoader` to `ROCmDiffusionLoader`
3. **Functionality is identical** - same inputs, same outputs
4. **Bug is fixed** - no more crashes when loading models!

### For Developers

If you were importing the node:

```python
# Old
from rocm_nodes.core.unet_loader import ROCMOptimizedUNetLoader

# New
from rocm_nodes.core.unet_loader import ROCmDiffusionLoader
```

## Naming Convention Going Forward

### Class Names
```python
# Pattern: ROCm + DescriptiveName
class ROCmDiffusionLoader:        # ✅ Correct
class ROCMDiffusionLoader:        # ❌ Wrong
class RocMDiffusionLoader:        # ❌ Wrong
```

### Display Names
```python
# Pattern: "ROCm " + Descriptive Name
"ROCm Diffusion Loader"           # ✅ Correct
"ROCM Diffusion Loader"           # ❌ Wrong
"RocM Diffusion Loader"           # ❌ Wrong
```

### Categories
```python
# Pattern: "ROCm Ninodes/" + Subcategory
CATEGORY = "ROCm Ninodes/Loaders" # ✅ Correct
CATEGORY = "ROCM Ninodes/Loaders" # ❌ Wrong
CATEGORY = "RocM Ninodes/Loaders" # ❌ Wrong
```

### Documentation
```markdown
# Pattern: ROCm (as one word, with capital R, O, C, lowercase m)
ROCm-optimized nodes              # ✅ Correct
ROCM-optimized nodes              # ❌ Wrong
RocM-optimized nodes              # ❌ Wrong
```

## Why "ROCm" Specifically?

**ROCm** stands for "Radeon Open Compute platform" and is AMD's official branding:
- **R**adeon **O**pen **C**ompute **m**odule
- Official AMD documentation uses "ROCm"
- Matches industry standard capitalization
- Distinguishes from "ROCM" (looks like acronym screaming)
- More professional than "RocM" (looks like typo)

## Summary

### What Was Fixed
✅ Critical bug preventing diffusion model loading  
✅ Inconsistent naming across codebase  
✅ Missing naming convention guidelines  

### What Changed
- Node renamed: "UNet Loader" → "Diffusion Loader"
- All branding: "ROCM"/"RocM" → "ROCm"
- Bug fixed: `model_options=None` → `model_options={}`

### Result
- ✅ Node works correctly
- ✅ Professional, consistent branding
- ✅ Clear naming standards for future development

## Related Documentation

- `.cursorrules` - Naming conventions and project standards
- `docs/LOADER_NODES_SUMMARY.md` - Overview of all loader nodes
- `docs/UNET_LOADER_ADDED.md` - Original diffusion loader documentation (use new name)

---

**Version:** Updated October 29, 2025  
**Affected Nodes:** ROCm Diffusion Loader (formerly UNet Loader)  
**Status:** ✅ Complete






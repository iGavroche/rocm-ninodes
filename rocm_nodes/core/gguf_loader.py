"""
GGUF loader node for ROCm Ninodes.

Contains GGUF model loading implementation:
- ROCmGGUFLoader: ROCm-optimized GGUF model loader

GGUF (GPT-Generated Unified Format) is a binary format used for quantized models.
This loader handles GGUF files for diffusion models with ROCm-specific optimizations.

Based on City96's ComfyUI-GGUF implementation, optimized for ROCm/gfx1151.
Uses lazy dequantization - tensors stay quantized until needed during forward passes.
"""

import os
import warnings
import numpy as np
from typing import Tuple, Optional, Dict, Any, List

import torch
import comfy.sd
import comfy.model_patcher
import comfy.ops
import comfy.model_management
import comfy.lora
import folder_paths
from ..utils.memory import simple_memory_cleanup

# Try to import required libraries - provide helpful errors if not available
try:
    import gguf
    GGUF_AVAILABLE = True
except ImportError:
    GGUF_AVAILABLE = False

# Try to import City96's GGUF utilities (CRITICAL: Use their implementation if available)
try:
    import sys
    import importlib.util
    # Try to import from ComfyUI-GGUF custom node
    gguf_path = "/home/nino/ComfyUI/custom_nodes/ComfyUI-GGUF"
    if os.path.exists(gguf_path):
        # Import ops.py first (contains GGMLTensor and GGMLOps)
        spec = importlib.util.spec_from_file_location("gguf_ops", os.path.join(gguf_path, "ops.py"))
        if spec and spec.loader:
            # Add the directory to path so imports work
            sys.path.insert(0, gguf_path)
            try:
                gguf_ops = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(gguf_ops)
                GGMLOps = gguf_ops.GGMLOps
                GGMLTensor = gguf_ops.GGMLTensor
                
                # Import loader.py
                spec = importlib.util.spec_from_file_location("gguf_loader", os.path.join(gguf_path, "loader.py"))
                if spec and spec.loader:
                    gguf_loader = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(gguf_loader)
                    gguf_sd_loader = gguf_loader.gguf_sd_loader
                    
                    # Import dequant.py
                    spec = importlib.util.spec_from_file_location("gguf_dequant", os.path.join(gguf_path, "dequant.py"))
                    if spec and spec.loader:
                        gguf_dequant = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(gguf_dequant)
                        is_quantized = gguf_dequant.is_quantized
                        
                        CITY96_GGUF_AVAILABLE = True
                        print("[INFO] Using City96's ComfyUI-GGUF implementation (recommended)")
                    else:
                        CITY96_GGUF_AVAILABLE = False
                else:
                    CITY96_GGUF_AVAILABLE = False
            finally:
                # Remove from path
                if gguf_path in sys.path:
                    sys.path.remove(gguf_path)
        else:
            CITY96_GGUF_AVAILABLE = False
    else:
        CITY96_GGUF_AVAILABLE = False
        print("[INFO] ComfyUI-GGUF not found, using our implementation")
except Exception as e:
    CITY96_GGUF_AVAILABLE = False
    print(f"[INFO] Could not load City96's GGUF implementation: {e}")
    print("[INFO] Using our implementation instead")


# ============================================================================
# Dequantization utilities (based on City96's ComfyUI-GGUF)
# ============================================================================

TORCH_COMPATIBLE_QTYPES = (None, gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16)

def is_torch_compatible(tensor):
    """Check if tensor is compatible with PyTorch operations (F32/F16)."""
    return tensor is None or getattr(tensor, "tensor_type", None) in TORCH_COMPATIBLE_QTYPES

def is_quantized(tensor):
    """Check if tensor is quantized (not F32/F16)."""
    return not is_torch_compatible(tensor)

def to_uint32(x):
    """Convert uint8 array to uint32 (no native uint32 in PyTorch)."""
    x = x.view(torch.uint8).to(torch.int32)
    return (x[:, 0] | x[:, 1] << 8 | x[:, 2] << 16 | x[:, 3] << 24).unsqueeze(1)

def split_block_dims(blocks, *args):
    """Split blocks into dimensions."""
    n_max = blocks.shape[1]
    dims = list(args) + [n_max - sum(args)]
    return torch.split(blocks, dims, dim=1)

# Full weights dequantization
def dequantize_blocks_BF16(blocks, block_size, type_size, dtype=None):
    """Dequantize BF16 blocks."""
    return (blocks.view(torch.int16).to(torch.int32) << 16).view(torch.float32)

# Legacy Quants
def dequantize_blocks_Q8_0(blocks, block_size, type_size, dtype=None):
    """Dequantize Q8_0 blocks (2 bytes scale + 32 bytes quantized)."""
    d, x = split_block_dims(blocks, 2)
    d = d.view(torch.float16).to(dtype)
    x = x.view(torch.int8)
    return (d * x)

def dequantize_blocks_Q5_1(blocks, block_size, type_size, dtype=None):
    """Dequantize Q5_1 blocks."""
    n_blocks = blocks.shape[0]
    
    d, m, qh, qs = split_block_dims(blocks, 2, 2, 4)
    d = d.view(torch.float16).to(dtype)
    m = m.view(torch.float16).to(dtype)
    qh = to_uint32(qh)
    
    qh = qh.reshape((n_blocks, 1)) >> torch.arange(32, device=d.device, dtype=torch.int32).reshape(1, 32)
    ql = qs.reshape((n_blocks, -1, 1, block_size // 2)) >> torch.tensor([0, 4], device=d.device, dtype=torch.uint8).reshape(1, 1, 2, 1)
    qh = (qh & 1).to(torch.uint8)
    ql = (ql & 0x0F).reshape((n_blocks, -1))
    
    qs = (ql | (qh << 4))
    return (d * qs) + m

def dequantize_blocks_Q5_0(blocks, block_size, type_size, dtype=None):
    """Dequantize Q5_0 blocks."""
    n_blocks = blocks.shape[0]
    
    d, qh, qs = split_block_dims(blocks, 2, 4)
    d = d.view(torch.float16).to(dtype)
    qh = to_uint32(qh)
    
    qh = qh.reshape(n_blocks, 1) >> torch.arange(32, device=d.device, dtype=torch.int32).reshape(1, 32)
    ql = qs.reshape(n_blocks, -1, 1, block_size // 2) >> torch.tensor([0, 4], device=d.device, dtype=torch.uint8).reshape(1, 1, 2, 1)
    
    qh = (qh & 1).to(torch.uint8)
    ql = (ql & 0x0F).reshape(n_blocks, -1)
    
    qs = (ql | (qh << 4)).to(torch.int8) - 16
    return (d * qs)

def dequantize_blocks_Q4_1(blocks, block_size, type_size, dtype=None):
    """Dequantize Q4_1 blocks."""
    n_blocks = blocks.shape[0]
    
    d, m, qs = split_block_dims(blocks, 2, 2)
    d = d.view(torch.float16).to(dtype)
    m = m.view(torch.float16).to(dtype)
    
    qs = qs.reshape((n_blocks, -1, 1, block_size // 2)) >> torch.tensor([0, 4], device=d.device, dtype=torch.uint8).reshape(1, 1, 2, 1)
    qs = (qs & 0x0F).reshape(n_blocks, -1)
    
    return (d * qs) + m

def dequantize_blocks_Q4_0(blocks, block_size, type_size, dtype=None):
    """Dequantize Q4_0 blocks."""
    n_blocks = blocks.shape[0]
    
    d, qs = split_block_dims(blocks, 2)
    d = d.view(torch.float16).to(dtype)
    
    qs = qs.reshape((n_blocks, -1, 1, block_size // 2)) >> torch.tensor([0, 4], device=d.device, dtype=torch.uint8).reshape((1, 1, 2, 1))
    qs = (qs & 0x0F).reshape((n_blocks, -1)).to(torch.int8) - 8
    return (d * qs)

# Dequantization function mapping
dequantize_functions = {}
if GGUF_AVAILABLE:
    dequantize_functions = {
        gguf.GGMLQuantizationType.BF16: dequantize_blocks_BF16,
        gguf.GGMLQuantizationType.Q8_0: dequantize_blocks_Q8_0,
        gguf.GGMLQuantizationType.Q5_1: dequantize_blocks_Q5_1,
        gguf.GGMLQuantizationType.Q5_0: dequantize_blocks_Q5_0,
        gguf.GGMLQuantizationType.Q4_1: dequantize_blocks_Q4_1,
        gguf.GGMLQuantizationType.Q4_0: dequantize_blocks_Q4_0,
    }

def dequantize(data, qtype, oshape, dtype=None):
    """
    Dequantize tensor back to usable shape/dtype.
    
    Args:
        data: Raw tensor data (uint8 view)
        qtype: Quantization type
        oshape: Original shape
        dtype: Target dtype (default: float32)
    """
    if dtype is None:
        dtype = torch.float32
    
    if qtype not in dequantize_functions:
        raise ValueError(f"Unsupported quantization type: {qtype}")
    
    block_size, type_size = gguf.GGML_QUANT_SIZES[qtype]
    dequantize_blocks = dequantize_functions[qtype]
    
    rows = data.reshape(
        (-1, data.shape[-1])
    ).view(torch.uint8)
    
    n_blocks = rows.numel() // type_size
    blocks = rows.reshape((n_blocks, type_size))
    blocks = dequantize_blocks(blocks, block_size, type_size, dtype)
    return blocks.reshape(oshape)

def dequantize_tensor(tensor, dtype=None, dequant_dtype=None):
    """
    Main entry point for dequantizing a tensor.
    
    Args:
        tensor: ROCmGGMLTensor or regular tensor
        dtype: Target dtype for output
        dequant_dtype: Dtype for intermediate dequantization (None = use dtype)
    
    Returns:
        Dequantized tensor (regular torch.Tensor)
    """
    if tensor is None:
        return None
    
    qtype = getattr(tensor, "tensor_type", None)
    oshape = getattr(tensor, "tensor_shape", tensor.shape)
    
    # ROCm optimization: Use fp32 by default for gfx1151 (better stability)
    # For quantized GGUF models, fp32 is optimal for ROCm
    if dtype is None:
        dtype = torch.float32
    
    # ROCm optimization: Use non-blocking transfers
    device = tensor.device
    non_blocking = comfy.model_management.device_supports_non_blocking(device) if hasattr(comfy.model_management, 'device_supports_non_blocking') else True
    
    if qtype in TORCH_COMPATIBLE_QTYPES:
        # Non-quantized: just convert dtype
        return tensor.to(dtype, non_blocking=non_blocking)
    elif qtype in dequantize_functions:
        # Quantized: dequantize using our functions
        dequant_dtype = dtype if dequant_dtype == "target" else (dequant_dtype or dtype)
        dequantized = dequantize(tensor.data, qtype, oshape, dtype=dequant_dtype)
        return dequantized.to(dtype, non_blocking=non_blocking)
    else:
        # Fallback to gguf library (slower)
        try:
            new = gguf.quants.dequantize(tensor.cpu().numpy(), qtype)
            return torch.from_numpy(new).to(device, dtype=dtype, non_blocking=non_blocking)
        except Exception as e:
            raise ValueError(f"Could not dequantize tensor (type {qtype}): {e}")


class ROCmGGUFLoader:
    """
    ROCm-optimized GGUF model loader for AMD GPUs (gfx1151)
    
    Features:
    - Native GGUF file format support (not PyTorch pickle format)
    - ROCm-specific memory optimizations
    - Quantized model support (Q4, Q5, Q8, etc.)
    - Efficient tensor conversion from GGUF to PyTorch
    - Memory-efficient loading for large models
    
    Designed for:
    - PyTorch 2.7+ with ROCm 6.4+
    - Modern memory allocators (no expandable_segments needed)
    - gfx1151 architecture with unified memory
    - WAN, Flux, and other diffusion models in GGUF format
    """
    
    @classmethod
    def _get_gguf_files(cls) -> List[str]:
        """
        Get all GGUF files from common model folders.
        
        Checks multiple folder names: diffusion_models, unet, unet_gguf, checkpoints
        """
        all_files = []
        
        # Check multiple folder names
        folder_names = ["diffusion_models", "unet", "unet_gguf", "checkpoints"]
        
        for folder_name in folder_names:
            try:
                files = folder_paths.get_filename_list(folder_name)
                # Filter to only GGUF files
                for filename in files:
                    if filename.lower().endswith('.gguf'):
                        all_files.append(filename)
            except Exception:
                continue
        
        # Also scan manually for GGUF files
        try:
            model_paths = folder_paths.get_folder_paths("diffusion_models")
            if model_paths:
                for model_path in model_paths:
                    if not os.path.exists(model_path):
                        continue
                    try:
                        for filename in os.listdir(model_path):
                            if filename.lower().endswith('.gguf'):
                                all_files.append(filename)
                    except (OSError, PermissionError):
                        continue
        except Exception:
            pass
        
        # Remove duplicates and sort
        return sorted(list(set(all_files)))
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "gguf_name": (cls._get_gguf_files(), {
                    "tooltip": "GGUF model file to load"
                })
            }
        }
    
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("MODEL",)
    FUNCTION = "load_gguf"
    CATEGORY = "ROCm Ninodes/Loaders"
    DESCRIPTION = "ROCm-optimized GGUF model loader (PyTorch 2.7+, ROCm 6.4+)"
    
    def _create_simple_ops(self):
        """
        Create basic GGMLOps structure for lazy dequantization.
        Based on City96's GGMLOps, minimal implementation for now.
        Full dequantization will be implemented later for optimization.
        """
        TORCH_COMPATIBLE_QTYPES = (None, gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16)
        
        def is_quantized(tensor):
            """Check if tensor is quantized (not F32/F16)."""
            if tensor is None:
                return False
            tensor_type = getattr(tensor, "tensor_type", None)
            return tensor_type not in TORCH_COMPATIBLE_QTYPES
        
        # Create GGMLLayer base class (matches City96's structure)
        class ROCmGGMLLayer(torch.nn.Module):
            """Base class for GGML layers that handle quantized weights."""
            comfy_cast_weights = True
            dequant_dtype = None
            patch_dtype = None
            largest_layer = False
            torch_compatible_tensor_types = {None, gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16}
            
            def is_ggml_quantized(self, *, weight=None, bias=None):
                if weight is None:
                    weight = self.weight
                if bias is None:
                    bias = self.bias
                return is_quantized(weight) or is_quantized(bias)
            
            def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
                weight, bias = state_dict.get(f"{prefix}weight"), state_dict.get(f"{prefix}bias")
                # Use modified load for linear due to not initializing on creation (like City96)
                if self.is_ggml_quantized(weight=weight, bias=bias) or isinstance(self, torch.nn.Linear):
                    return self.ggml_load_from_state_dict(state_dict, prefix, *args, **kwargs)
                return super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)
            
            def ggml_load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
                prefix_len = len(prefix)
                for k, v in state_dict.items():
                    if k[prefix_len:] == "weight":
                        self.weight = torch.nn.Parameter(v, requires_grad=False)
                    elif k[prefix_len:] == "bias" and v is not None:
                        self.bias = torch.nn.Parameter(v, requires_grad=False)
                    else:
                        unexpected_keys.append(k)
                
                # For Linear layer with missing weight
                if self.weight is None and isinstance(self, torch.nn.Linear):
                    v = torch.zeros(self.in_features, self.out_features)
                    self.weight = torch.nn.Parameter(v, requires_grad=False)
                    missing_keys.append(prefix + "weight")
                
                # For vram estimation
                if getattr(self.weight, "is_largest_weight", False):
                    self.largest_layer = True
            
            def get_weight(self, tensor, dtype):
                """
                Get dequantized weight, handling LoRA patches if present.
                
                Args:
                    tensor: ROCmGGMLTensor or regular tensor
                    dtype: Target dtype
                
                Returns:
                    Dequantized tensor (regular torch.Tensor)
                """
                if tensor is None:
                    return None
                
                # Check if it's a ROCmGGMLTensor (by checking for tensor_type attribute)
                is_ggml_tensor = hasattr(tensor, "tensor_type")
                
                # Consolidate and load patches to GPU in async (for LoRA support)
                # ROCm optimization: Use non-blocking transfers
                patch_list = []
                device = tensor.device
                non_blocking = comfy.model_management.device_supports_non_blocking(device)
                for patches, key in getattr(tensor, "patches", []):
                    # Move patches to device (like City96's move_patch_to_device)
                    if isinstance(patches, torch.Tensor):
                        patch_list.append((patches.to(device, non_blocking=non_blocking), key))
                    elif isinstance(patches, tuple):
                        patch_list.extend([(p.to(device, non_blocking=non_blocking), key) for p in patches if isinstance(p, torch.Tensor)])
                    else:
                        patch_list.append((patches, key))
                
                # Dequantize tensor while patches load (or just convert if F32/F16)
                if is_ggml_tensor:
                    weight = dequantize_tensor(tensor, dtype, self.dequant_dtype)
                    # CRITICAL: Always convert ROCmGGMLTensor to regular tensor
                    # Even F32/F16 wrapped in ROCmGGMLTensor must be converted
                    if hasattr(weight, "tensor_type"):
                        weight = torch.Tensor(weight)
                else:
                    # Regular tensor - just convert dtype
                    weight = tensor.to(dtype) if dtype else tensor
                
                # Apply patches (LoRA)
                if len(patch_list) > 0:
                    # Extract key from first patch (all patches for a tensor share the same key)
                    key = patch_list[0][1] if patch_list else None
                    if key is not None:
                        if self.patch_dtype is None:
                            weight = comfy.lora.calculate_weight(patch_list, weight, key)
                        else:
                            # For testing, may degrade image quality
                            patch_dtype = dtype if self.patch_dtype == "target" else self.patch_dtype
                            weight = comfy.lora.calculate_weight(patch_list, weight, key, patch_dtype)
                
                return weight
            
            def _save_to_state_dict(self, *args, **kwargs):
                """Save to state dict - route to ggml_save if quantized."""
                if self.is_ggml_quantized():
                    return self.ggml_save_to_state_dict(*args, **kwargs)
                return super()._save_to_state_dict(*args, **kwargs)
            
            def ggml_save_to_state_dict(self, destination, prefix, keep_vars):
                """
                Create fake state dict for VRAM estimation.
                This is a fake state dict for vram estimation.
                """
                # This is a fake state dict for vram estimation
                weight = torch.zeros_like(self.weight, device=torch.device("meta"))
                destination[prefix + "weight"] = weight
                if self.bias is not None:
                    bias = torch.zeros_like(self.bias, device=torch.device("meta"))
                    destination[prefix + "bias"] = bias
                
                # Take into account space required for dequantizing the largest tensor
                if self.largest_layer:
                    shape = getattr(self.weight, "tensor_shape", self.weight.shape)
                    dtype = self.dequant_dtype if self.dequant_dtype and self.dequant_dtype != "target" else torch.float16
                    temp = torch.empty(*shape, device=torch.device("meta"), dtype=dtype)
                    destination[prefix + "temp.weight"] = temp
                
                return
            
            def cast_bias_weight(self, input=None, dtype=None, device=None, bias_dtype=None):
                """
                Cast bias and weight to appropriate dtype/device, dequantizing if needed.
                
                Args:
                    input: Input tensor (used to infer dtype/device)
                    dtype: Target dtype
                    device: Target device
                    bias_dtype: Target dtype for bias (defaults to dtype)
                
                Returns:
                    Tuple of (weight, bias) as regular torch.Tensors
                """
                if input is not None:
                    if dtype is None:
                        dtype = getattr(input, "dtype", torch.float32)
                    if bias_dtype is None:
                        bias_dtype = dtype
                    if device is None:
                        device = input.device
                
                bias = None
                non_blocking = comfy.model_management.device_supports_non_blocking(device)
                # ROCm optimization: Use non-blocking transfers for better performance
                if self.bias is not None:
                    bias = self.get_weight(self.bias.to(device, non_blocking=non_blocking), dtype)
                    bias = comfy.ops.cast_to(bias, bias_dtype, device, non_blocking=non_blocking, copy=False)
                
                weight = self.get_weight(self.weight.to(device, non_blocking=non_blocking), dtype)
                weight = comfy.ops.cast_to(weight, dtype, device, non_blocking=non_blocking, copy=False)
                return weight, bias
            
            def forward_comfy_cast_weights(self, input, *args, **kwargs):
                """
                Forward pass with automatic casting - routes to ggml or standard forward.
                CRITICAL: Always use our cast_bias_weight if weights are ROCmGGMLTensor,
                even for non-quantized F32/F16 tensors, because PyTorch can't handle
                our custom tensor subclass in operations.
                """
                # Check if weights are ROCmGGMLTensor (even if not quantized)
                weight_is_ggml = hasattr(self.weight, "tensor_type") if self.weight is not None else False
                bias_is_ggml = hasattr(self.bias, "tensor_type") if self.bias is not None else False
                
                if self.is_ggml_quantized() or weight_is_ggml or bias_is_ggml:
                    # Always use our dequantization path for ROCmGGMLTensor objects
                    out = self.forward_ggml_cast_weights(input, *args, **kwargs)
                else:
                    # Only use parent's method if weights are regular tensors
                    out = super().forward_comfy_cast_weights(input, *args, **kwargs)
                
                # Prevent propagating custom tensor class
                # Check if output is our custom tensor type
                if hasattr(out, "tensor_type"):
                    out = torch.Tensor(out)
                return out
        
        # Create GGMLOps class (matches City96's structure)
        class ROCmGGMLOps(comfy.ops.manual_cast):
            """
            ROCm-optimized GGMLOps - dequantize weights on the fly.
            Minimal implementation based on City96's approach.
            """
            class Linear(ROCmGGMLLayer, comfy.ops.manual_cast.Linear):
                def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
                    torch.nn.Module.__init__(self)
                    # Don't initialize weight/bias here to avoid memory spike (like City96)
                    self.in_features = in_features
                    self.out_features = out_features
                    self.weight = None
                    self.bias = None
                
                def forward_ggml_cast_weights(self, input):
                    """
                    Forward pass with dequantization on-the-fly.
                    Uses cast_bias_weight() to dequantize before operations.
                    """
                    weight, bias = self.cast_bias_weight(input)
                    return torch.nn.functional.linear(input, weight, bias)
        
        return ROCmGGMLOps()
    
    def _gguf_sd_loader(self, file_path: str, handle_prefix: str = "model.diffusion_model.") -> Dict[str, Any]:
        """
        Load GGUF file and return state dict with GGMLTensor objects (like City96's implementation).
        
        This is memory-efficient: tensors stay as memmap until dequantized on-the-fly.
        """
        if not GGUF_AVAILABLE:
            raise ImportError(
                "GGUF library not installed. Please install it with: pip install gguf\n"
                "Or use: uv add gguf"
            )
        
        reader = gguf.GGUFReader(file_path)
        
        # Filter and strip prefix
        has_prefix = False
        if handle_prefix:
            prefix_len = len(handle_prefix)
            tensor_names = set(tensor.name for tensor in reader.tensors)
            has_prefix = any(s.startswith(handle_prefix) for s in tensor_names)
        
        tensors = []
        for tensor in reader.tensors:
            sd_key = tensor_name = tensor.name
            if has_prefix:
                if not tensor_name.startswith(handle_prefix):
                    continue
                sd_key = tensor_name[prefix_len:]
            tensors.append((sd_key, tensor))
        
        # Main loading loop - create GGMLTensor objects (wraps raw data, no dequantization)
        state_dict = {}
        qtype_dict = {}
        
        for sd_key, tensor in tensors:
            tensor_name = tensor.name
            
            # Convert numpy memmap to torch tensor (still quantized, memory-efficient)
            # NOTE: This matches City96's approach exactly
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="The given NumPy array is not writable")
                torch_tensor = torch.from_numpy(tensor.data)  # memmap - very memory efficient
            
            # Get shape - exactly like City96's get_orig_shape() function
            shape = self._get_orig_shape(reader, tensor_name)
            if shape is None:
                # GGUF stores shapes in reverse order (C-style), PyTorch uses F-style
                # Reverse the shape to match PyTorch's expected format
                shape = torch.Size(tuple(int(v) for v in reversed(tensor.shape)))
            
            # CRITICAL: Wrap ALL tensors in ROCmGGMLTensor (like City96 wraps all in GGMLTensor)
            # This ensures consistent shape handling for ComfyUI's architecture detection
            if tensor.tensor_type in {gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16}:
                # Non-quantized: reshape directly, then wrap in ROCmGGMLTensor
                torch_tensor = torch_tensor.view(*shape)
            
            # Always wrap in ROCmGGMLTensor (matches City96 line 120)
            state_dict[sd_key] = self._create_ggml_tensor(torch_tensor, tensor_type=tensor.tensor_type, tensor_shape=shape)
            
            # Track tensor types
            tensor_type_str = getattr(tensor.tensor_type, "name", repr(tensor.tensor_type))
            qtype_dict[tensor_type_str] = qtype_dict.get(tensor_type_str, 0) + 1
        
        print(f"[INFO] GGUF qtypes: {', '.join(f'{k} ({v})' for k, v in qtype_dict.items())}")
        
        # Mark largest tensor for VRAM estimation (like City96)
        # Find quantized tensors and mark the largest one
        quantized_tensors = {k: v for k, v in state_dict.items() if self._is_quantized_tensor(v)}
        if len(quantized_tensors) > 0:
            max_key = max(quantized_tensors.keys(), key=lambda k: quantized_tensors[k].numel())
            if hasattr(state_dict[max_key], 'is_largest_weight'):
                state_dict[max_key].is_largest_weight = True
        
        return state_dict
    
    def _is_quantized_tensor(self, tensor) -> bool:
        """Check if tensor is quantized (not F32/F16)."""
        if tensor is None:
            return False
        tensor_type = getattr(tensor, "tensor_type", None)
        TORCH_COMPATIBLE_QTYPES = (None, gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16)
        return tensor_type not in TORCH_COMPATIBLE_QTYPES
    
    def _get_orig_shape(self, reader, tensor_name: str) -> Optional[torch.Size]:
        """Get original shape from GGUF metadata if available."""
        field_key = f"comfy.gguf.orig_shape.{tensor_name}"
        field = reader.get_field(field_key)
        if field is None:
            return None
        if len(field.types) != 2 or field.types[0] != gguf.GGUFValueType.ARRAY or field.types[1] != gguf.GGUFValueType.INT32:
            return None
        return torch.Size(tuple(int(field.parts[part_idx][0]) for part_idx in field.data))
    
    
    def _create_ggml_tensor(self, tensor: torch.Tensor, tensor_type: int, tensor_shape: torch.Size) -> torch.Tensor:
        """
        Create a GGMLTensor-like wrapper that preserves quantization info.
        CRITICAL: The tensor must expose the correct shape for ComfyUI's model detection.
        
        Based on City96's GGMLTensor implementation - exactly replicates their approach.
        This allows ComfyUI to detect the correct architecture even for quantized tensors.
        """
        # Create GGMLTensor class exactly like City96's (ROCm-optimized version)
        class ROCmGGMLTensor(torch.Tensor):
            """
            ROCm-optimized GGMLTensor - based on City96's implementation.
            Stores quantized weights and exposes correct shape for model detection.
            """
            def __init__(self, *args, tensor_type, tensor_shape, patches=[], **kwargs):
                super().__init__()
                self.tensor_type = tensor_type
                self.tensor_shape = tensor_shape
                self.patches = patches

            def __new__(cls, *args, tensor_type, tensor_shape, patches=[], **kwargs):
                return super().__new__(cls, *args, **kwargs)

            def to(self, *args, **kwargs):
                new = super().to(*args, **kwargs)
                new.tensor_type = getattr(self, "tensor_type", None)
                new.tensor_shape = getattr(self, "tensor_shape", new.data.shape)
                new.patches = getattr(self, "patches", []).copy()
                return new

            def clone(self, *args, **kwargs):
                return self

            def detach(self, *args, **kwargs):
                return self

            def copy_(self, *args, **kwargs):
                # fixes .weight.copy_ in comfy/clip_model/CLIPTextModel
                try:
                    return super().copy_(*args, **kwargs)
                except Exception as e:
                    print(f"[WARNING] ignoring 'copy_' on tensor: {e}")

            def new_empty(self, size, *args, **kwargs):
                # Intel Arc fix, ref#50 (also helps with ROCm)
                new_tensor = super().new_empty(size, *args, **kwargs)
                return ROCmGGMLTensor(
                        new_tensor,
                        tensor_type = getattr(self, "tensor_type", None),
                        tensor_shape = size,
                        patches = getattr(self, "patches", []).copy()
                )

            @property
            def shape(self):
                # CRITICAL: Return tensor_shape for model detection, not actual data shape
                # This is what makes architecture detection work correctly
                if not hasattr(self, "tensor_shape"):
                    self.tensor_shape = self.size()
                return self.tensor_shape
            
            def numel(self):
                # Return numel based on tensor_shape (for VRAM estimation)
                if hasattr(self, "tensor_shape"):
                    result = 1
                    for dim in self.tensor_shape:
                        result *= dim
                    return result
                return super().numel()
        
        # Create the tensor exactly like City96 does
        return ROCmGGMLTensor(tensor, tensor_type=tensor_type, tensor_shape=tensor_shape)
    
    def load_gguf(self, gguf_name: str) -> Tuple:
        """
        Load GGUF model using ROCm-optimized loading with diagnostics.
        
        GGUF files are not PyTorch pickle files, so we need special handling.
        This implementation attempts to use ComfyUI's loader with compatibility patches,
        or falls back to error message with helpful guidance.
        """
        # Get model path - try multiple folder names
        gguf_path = None
        folder_names = ["diffusion_models", "unet", "unet_gguf", "checkpoints"]
        
        for folder_name in folder_names:
            try:
                gguf_path = folder_paths.get_full_path_or_raise(folder_name, gguf_name)
                break
            except Exception:
                continue
        
        if gguf_path is None:
            raise FileNotFoundError(f"Could not find GGUF model '{gguf_name}' in any of: {', '.join(folder_names)}")
        
        # Log diagnostics
        print(f"\n[LOADING] Loading GGUF model: {gguf_name}")
        self._log_system_info()
        self._log_memory_status()
        print("[LOADING] GGUF files require special handling (not PyTorch format)...")
        
        try:
            # Check if gguf library is available
            if not GGUF_AVAILABLE:
                raise ImportError(
                    "GGUF library not installed. Please install it with:\n"
                    "  pip install gguf\n"
                    "Or if using uv:\n"
                    "  uv add gguf"
                )
            
            # Use City96's efficient approach: load GGUF state dict directly (no safetensors conversion)
            print("[LOADING] Loading GGUF file using efficient lazy dequantization...")
            file_size = os.path.getsize(gguf_path) / (1024**3)
            print(f"[INFO] File size: {file_size:.2f} GB")
            
            # Load GGUF state dict with GGMLTensor wrappers (memory-efficient, no upfront dequantization)
            # Using our ROCm-optimized implementation (based on City96's approach)
            print("[INFO] Using ROCm-optimized GGUF loader")
            sd = self._gguf_sd_loader(gguf_path)
            # Create ops wrapper for lazy dequantization (we'll implement full GGMLOps later)
            ops = self._create_simple_ops()
            
            tensor_count = len(sd)
            print(f"[INFO] Loaded {tensor_count} tensors from GGUF file (lazy dequantization enabled)")
            
            # Prepare model options with custom operations for lazy dequantization (like City96)
            model_options = {
                "custom_operations": ops
            }
            
            # ROCm optimization: Use fp32 by default for gfx1151 (better stability)
            # GGUF models are already quantized, so dtype selector is not needed
            # For quantized models, we dequantize to float32, which is optimal for ROCm
            
            # Debug: Print some tensor shapes to verify they're correct (helpful for troubleshooting)
            sample_keys = [
                "single_blocks.0.linear1.weight", 
                "double_blocks.0.img_mod.lin.weight"
            ]
            for key in sample_keys:
                if key in sd:
                    tensor = sd[key]
                    print(f"[DEBUG] Tensor '{key}' shape: {tensor.shape}")
            
            # ROCm optimizations: Configure backend settings before loading
            is_amd = False
            is_gfx1151 = False
            try:
                if torch.cuda.is_available():
                    device_name = torch.cuda.get_device_name(0)
                    is_amd = "AMD" in device_name or "Radeon" in device_name
                    if is_amd:
                        # GENERAL ROCM OPTIMIZATIONS (apply to all AMD GPUs)
                        torch.backends.cuda.matmul.allow_tf32 = False  # TF32 not supported on AMD
                        torch.backends.cuda.matmul.allow_fp16_accumulation = True  # Better performance on AMD
                        
                        # Detect gfx1151 for architecture-specific optimizations
                        try:
                            arch = torch.cuda.get_device_properties(0).gcnArchName
                            if 'gfx1151' in arch:
                                is_gfx1151 = True
                                print("[CONFIG] gfx1151 architecture detected - using fp32 precision")
                        except:
                            pass
            except Exception:
                pass
            
            # Load model directly from state dict (no safetensors conversion needed!)
            # City96 doesn't pass metadata - ComfyUI detects architecture from tensor shapes
            print("[LOADING] Loading model from GGUF state dict (lazy dequantization)...")
            model = comfy.sd.load_diffusion_model_state_dict(sd, model_options=model_options)
            
            # Validate output
            if model is None:
                raise ValueError(f"GGUF model loading returned None for: {gguf_name}")
            
            # ROCm optimization: Memory cleanup after loading large model
            if is_amd:
                simple_memory_cleanup()
            
            print("[SUCCESS] GGUF model loaded successfully")
            self._log_memory_status()
            
            return (model,)
            
        except torch.cuda.OutOfMemoryError as e:
            print(f"\n[ERROR] GPU Out of Memory Error")
            print(f"Error: {e}")
            self._log_memory_status()
            self._suggest_solutions()
            raise
        
        except Exception as e:
            print(f"\n[ERROR] GGUF model loading failed: {e}")
            print(f"Model: {gguf_name}")
            print(f"Path: {gguf_path}")
            print(f"Path exists: {os.path.exists(gguf_path)}")
            raise
    
    def _log_system_info(self) -> None:
        """Log GPU and ROCm configuration (non-intrusive)"""
        try:
            if torch.cuda.is_available():
                device_name = torch.cuda.get_device_name(0)
                print(f"[GPU] GPU: {device_name}")
                
                # Check for AMD/ROCm
                is_amd = "AMD" in device_name or "Radeon" in device_name
                if is_amd:
                    print("[CONFIG] ROCm backend detected")
                    # Check ROCm version if available
                    if hasattr(torch.version, 'hip') and torch.version.hip:
                        print(f"   ROCm version: {torch.version.hip}")
                else:
                    print("[INFO] Non-AMD GPU detected")
            else:
                print("[INFO] CUDA not available - using CPU")
        except Exception as e:
            print(f"[WARNING] GPU detection failed: {e}")
    
    def _log_memory_status(self) -> None:
        """Log current memory usage (helpful for debugging OOM)"""
        try:
            if torch.cuda.is_available():
                total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                allocated = torch.cuda.memory_allocated(0) / 1024**3
                reserved = torch.cuda.memory_reserved(0) / 1024**3
                free = total - allocated
                
                print(f"[MEMORY] Memory: {allocated:.2f}GB used / {total:.2f}GB total ({free:.2f}GB free)")
                
                # Warn about high fragmentation
                fragmentation = reserved - allocated
                if fragmentation > 0.5:  # 500MB
                    print(f"[WARNING] Memory fragmentation detected: {fragmentation:.2f}GB")
                    print(f"[INFO] Consider restarting ComfyUI to clear fragmentation")
        except Exception:
            pass  # Silent failure for diagnostics
    
    def _suggest_solutions(self) -> None:
        """Suggest solutions for OOM errors"""
        print("\n[INFO] Possible solutions:")
        print("   1. Run ComfyUI with --cache-none flag (disables model caching)")
        print("   2. Restart ComfyUI to clear all GPU memory")
        print("   3. Close other applications using GPU memory")
        print("   4. Use a lower quantization level (Q8 instead of Q4)")
        print("   5. Load only one large model at a time")
        print("   6. Reduce batch size or image resolution")
        print("   7. Split workflow into multiple separate workflows")


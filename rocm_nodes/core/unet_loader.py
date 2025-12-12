"""
Diffusion Model loader node for ROCm Ninodes.

Contains diffusion model loading implementation:
- ROCmDiffusionLoader: ROCm-optimized diffusion model loader

Modern PyTorch (2.7+) with ROCm has improved memory allocators that don't require
manual memory cleanup or expandable_segments configuration. This loader leverages
ComfyUI's native memory management while providing ROCm-specific diagnostics.
"""

import os
from typing import Tuple, Optional, Dict, Any, List

import torch
import comfy.sd
import folder_paths


class ROCmDiffusionLoader:
    """
    ROCm-optimized UNet/Diffusion Model loader for AMD GPUs (gfx1151)
    
    Features:
    - Delegates to ComfyUI's native diffusion model loading (memory-optimized)
    - Provides ROCm-specific diagnostics and validation
    - Detects quantized models for compatibility warnings
    - No aggressive memory cleanup (prevents fragmentation)
    - Supports multiple model formats: .safetensors, .gguf, .ckpt, .pt, .pth, .bin, .onnx
    
    Designed for:
    - PyTorch 2.7+ with ROCm 6.4+
    - Modern memory allocators (no expandable_segments needed)
    - gfx1151 architecture with unified memory
    - WAN, Flux, and other diffusion models
    """
    
    # Supported model file extensions
    MODEL_EXTENSIONS = {".safetensors", ".gguf", ".ckpt", ".pt", ".pth", ".bin", ".onnx"}
    
    @classmethod
    def _get_model_files(cls) -> List[str]:
        """
        Get all model files from the diffusion_models folder with supported extensions.
        
        This includes .safetensors, .gguf, .ckpt, .pt, .pth, .bin, and .onnx files.
        """
        model_files = []
        try:
            # Get the diffusion_models folder path
            model_paths = folder_paths.get_folder_paths("diffusion_models")
            if not model_paths:
                return []
            
            # Scan all diffusion_models folders (user can have multiple)
            for model_path in model_paths:
                if not os.path.exists(model_path):
                    continue
                
                # List all files in the directory
                for filename in os.listdir(model_path):
                    file_path = os.path.join(model_path, filename)
                    # Only include files (not directories) with supported extensions
                    if os.path.isfile(file_path):
                        _, ext = os.path.splitext(filename.lower())
                        if ext in cls.MODEL_EXTENSIONS:
                            model_files.append(filename)
            
            # Remove duplicates and sort
            model_files = sorted(list(set(model_files)))
            return model_files
        except Exception as e:
            # Fallback to ComfyUI's default if our scan fails
            print(f"⚠️  Warning: Could not scan diffusion_models folder: {e}")
            print("   Falling back to ComfyUI's default file list")
            return folder_paths.get_filename_list("diffusion_models")
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "unet_name": (cls._get_model_files(), {
                    "tooltip": "Diffusion model file to load (supports .safetensors, .gguf, .ckpt, .pt, .pth, .bin, .onnx)"
                })
            },
            "optional": {
                "weight_dtype": (["default", "fp8_e4m3fn", "fp8_e5m2"], {
                    "default": "default",
                    "tooltip": "Weight data type - use fp8 for quantized models"
                })
            }
        }
    
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("MODEL",)
    FUNCTION = "load_unet"
    CATEGORY = "ROCm Ninodes/Loaders"
    DESCRIPTION = "ROCm-optimized Diffusion Model loader (PyTorch 2.7+, ROCm 6.4+)"
    
    def load_unet(self, unet_name: str, weight_dtype: str = "default") -> Tuple:
        """
        Load UNet/Diffusion Model using ComfyUI's native loader with ROCm diagnostics.
        
        Modern PyTorch (2.7+) handles memory allocation efficiently, so we avoid
        manual cleanup that can cause fragmentation. Let PyTorch's allocator do its job.
        
        Note: For workflows with multiple large models (>20GB each), run ComfyUI with
              --cache-none flag to prevent model caching issues.
        """
        # Detect quantized models from filename
        quantized_indicators = ['fp8', 'int8', 'int4', 'quantized', 'bnb', 'gguf']
        unet_name_lower = unet_name.lower()
        is_quantized = any(indicator in unet_name_lower for indicator in quantized_indicators)
        
        if is_quantized:
            print(f"🔍 Detected quantized diffusion model: {unet_name}")
            print("💡 Quantized models use specialized dtypes - preserving original precision")
        
        # Get model path
        unet_path = folder_paths.get_full_path_or_raise("diffusion_models", unet_name)
        
        # Log diagnostics (non-intrusive)
        print(f"\n📁 Loading diffusion model: {unet_name}")
        self._log_system_info()
        self._log_memory_status()
        print("⏳ Loading model (may take 2-6 minutes for large models)...")
        
        try:
            # CRITICAL: fp8_scaled models require ComfyUI's automatic scaling detection
            # Never pass explicit dtype for fp8_scaled - it breaks the scaling mechanism!
            is_fp8_scaled = "fp8_scaled" in unet_name_lower or "fp8-scaled" in unet_name_lower
            
            # CRITICAL FIX: Match ComfyUI's native behavior exactly!
            # ComfyUI ALWAYS passes model_options, even if it's an empty dict {}
            # We must do the same to ensure identical behavior (prevents noise generation)
            # From ComfyUI nodes.py line 914: model_options = {} (always initialized as empty dict)
            model_options = {}
            
            # Only set explicit dtype if:
            # 1. User explicitly chose a dtype (not "default")
            # 2. Model is NOT fp8_scaled (scaled models need auto-detection)
            if weight_dtype != "default" and not is_fp8_scaled:
                model_options = {}
                if weight_dtype == "fp8_e4m3fn":
                    model_options["dtype"] = torch.float8_e4m3fn
                    print(f"🔧 Using FP8 E4M3FN precision (explicit)")
                elif weight_dtype == "fp8_e5m2":
                    model_options["dtype"] = torch.float8_e5m2
                    print(f"🔧 Using FP8 E5M2 precision (explicit)")
            
            if is_fp8_scaled and weight_dtype != "default":
                print(f"⚠️  WARNING: fp8_scaled models require 'default' dtype for automatic scaling")
                print(f"   Ignoring explicit dtype setting - using ComfyUI auto-detection")
                # Clear model_options to let ComfyUI auto-detect (empty dict allows auto-detection)
                model_options = {}
            
            # Use ComfyUI's native loader - ALWAYS pass model_options (matches native behavior)
            # For fp8_scaled models with "default" dtype, model_options is {} (empty dict)
            # which allows ComfyUI to auto-detect the scaling mechanism
            model = comfy.sd.load_diffusion_model(unet_path, model_options=model_options)
            
            if is_fp8_scaled:
                print("🔧 ComfyUI auto-detected fp8_scaled precision and scaling")
            
            # Validate output
            if model is None:
                raise ValueError(f"Diffusion model loading returned None for: {unet_name}")
            
            print("✅ Diffusion model loaded successfully")
            self._log_memory_status()
            
            return (model,)
            
        except torch.cuda.OutOfMemoryError as e:
            print(f"\n❌ GPU Out of Memory Error")
            print(f"Error: {e}")
            self._log_memory_status()
            self._suggest_solutions()
            raise
        
        except Exception as e:
            print(f"\n❌ Diffusion model loading failed: {e}")
            print(f"Model: {unet_name}")
            print(f"Path: {unet_path}")
            print(f"Path exists: {os.path.exists(unet_path)}")
            raise
    
    def _log_system_info(self) -> None:
        """Log GPU and ROCm configuration (non-intrusive)"""
        try:
            if torch.cuda.is_available():
                device_name = torch.cuda.get_device_name(0)
                print(f"🖥️  GPU: {device_name}")
                
                # Check for AMD/ROCm
                is_amd = "AMD" in device_name or "Radeon" in device_name
                if is_amd:
                    print("🔧 ROCm backend detected")
                    # Check ROCm version if available
                    if hasattr(torch.version, 'hip') and torch.version.hip:
                        print(f"   ROCm version: {torch.version.hip}")
                else:
                    print("ℹ️  Non-AMD GPU detected")
            else:
                print("ℹ️  CUDA not available - using CPU")
        except Exception as e:
            print(f"⚠️  GPU detection failed: {e}")
    
    def _log_memory_status(self) -> None:
        """Log current memory usage (helpful for debugging OOM)"""
        try:
            if torch.cuda.is_available():
                total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                allocated = torch.cuda.memory_allocated(0) / 1024**3
                reserved = torch.cuda.memory_reserved(0) / 1024**3
                free = total - allocated
                
                print(f"📊 Memory: {allocated:.2f}GB used / {total:.2f}GB total ({free:.2f}GB free)")
                
                # Warn about high fragmentation
                fragmentation = reserved - allocated
                if fragmentation > 0.5:  # 500MB
                    print(f"⚠️  Memory fragmentation detected: {fragmentation:.2f}GB")
                    print(f"💡 Consider restarting ComfyUI to clear fragmentation")
        except Exception:
            pass  # Silent failure for diagnostics
    
    def _suggest_solutions(self) -> None:
        """Suggest solutions for OOM errors"""
        print("\n💡 Possible solutions:")
        print("   1. Run ComfyUI with --cache-none flag (disables model caching)")
        print("   2. Restart ComfyUI to clear all GPU memory")
        print("   3. Close other applications using GPU memory")
        print("   4. Use a quantized model (fp8) if available")
        print("   5. Load only one large model at a time")
        print("   6. Reduce batch size or image resolution")
        print("   7. Split workflow into multiple separate workflows")


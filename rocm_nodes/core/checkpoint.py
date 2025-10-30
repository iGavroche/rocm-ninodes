"""
Checkpoint loader node for ROCM Ninodes.

Contains checkpoint loading implementation:
- ROCMOptimizedCheckpointLoader: ROCm-optimized checkpoint loader

Modern PyTorch (2.7+) with ROCm has improved memory allocators that don't require
manual memory cleanup or expandable_segments configuration. This loader leverages
ComfyUI's native memory management while providing ROCm-specific diagnostics.
"""

import os
from typing import Tuple

import torch
import comfy.sd
import folder_paths


class ROCMOptimizedCheckpointLoader:
    """
    ROCm-optimized checkpoint loader for AMD GPUs (gfx1151)
    
    Features:
    - Delegates to ComfyUI's native checkpoint loading (memory-optimized)
    - Provides ROCm-specific diagnostics and validation
    - Detects quantized models for compatibility warnings
    - No aggressive memory cleanup (prevents fragmentation)
    
    Designed for:
    - PyTorch 2.7+ with ROCm 6.4+
    - Modern memory allocators (no expandable_segments needed)
    - gfx1151 architecture with unified memory
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ckpt_name": (folder_paths.get_filename_list("checkpoints"), {
                    "tooltip": "Checkpoint file to load"
                })
            },
            "optional": {
                "compatibility_mode": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Enable extra validation for quantized/unusual models"
                })
            }
        }
    
    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    RETURN_NAMES = ("MODEL", "CLIP", "VAE")
    FUNCTION = "load_checkpoint"
    CATEGORY = "ROCm Ninodes/Loaders"
    DESCRIPTION = "ROCm-optimized checkpoint loader (PyTorch 2.7+, ROCm 6.4+)"
    
    def load_checkpoint(self, ckpt_name: str, compatibility_mode: bool = False) -> Tuple:
        """
        Load checkpoint using ComfyUI's native loader with ROCm diagnostics.
        
        Modern PyTorch (2.7+) handles memory allocation efficiently, so we avoid
        manual cleanup that can cause fragmentation. Let PyTorch's allocator do its job.
        """
        # Detect quantized models from filename
        quantized_indicators = ['fp8', 'int8', 'int4', 'quantized', 'bnb', 'gguf']
        ckpt_name_lower = ckpt_name.lower()
        is_quantized = any(indicator in ckpt_name_lower for indicator in quantized_indicators)
        
        if is_quantized:
            print(f"🔍 Detected quantized checkpoint: {ckpt_name}")
            print("💡 Quantized models use specialized dtypes - preserving original precision")
        
        # Get checkpoint path
        ckpt_path = folder_paths.get_full_path_or_raise("checkpoints", ckpt_name)
        
        # Log diagnostics (non-intrusive)
        print(f"\n📁 Loading checkpoint: {ckpt_name}")
        self._log_system_info()
        print("⏳ Loading model components (may take 2-6 minutes)...")
        
        try:
            # Use ComfyUI's native loader - it handles memory efficiently
            # No manual memory cleanup: modern PyTorch allocators are smart enough
            out = comfy.sd.load_checkpoint_guess_config(
                ckpt_path,
                output_vae=True,
                output_clip=True,
                embedding_directory=folder_paths.get_folder_paths("embeddings")
            )
            
            # Validate output structure
            if len(out) < 3:
                raise ValueError(f"Checkpoint loading returned {len(out)} items, expected 3 (MODEL, CLIP, VAE)")
            
            model, clip, vae = out[:3]
            
            # Optional: Extra validation in compatibility mode
            if compatibility_mode:
                self._validate_components(model, clip, vae, ckpt_name)
            
            print("✅ Checkpoint loaded successfully")
            self._log_memory_status()
            
            return (model, clip, vae)
            
        except torch.cuda.OutOfMemoryError as e:
            print(f"\n❌ GPU Out of Memory Error")
            print(f"Error: {e}")
            self._log_memory_status()
            self._suggest_solutions()
            raise
        
        except Exception as e:
            print(f"\n❌ Checkpoint loading failed: {e}")
            print(f"Checkpoint: {ckpt_name}")
            print(f"Path: {ckpt_path}")
            print(f"Path exists: {os.path.exists(ckpt_path)}")
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
        except Exception:
            pass  # Silent failure for diagnostics
    
    def _validate_components(self, model, clip, vae, ckpt_name: str) -> None:
        """Extra validation for compatibility mode"""
        if clip is None:
            raise ValueError(f"CLIP model is None - checkpoint may be invalid: {ckpt_name}")
        
        if not hasattr(clip, 'encode'):
            raise ValueError(f"CLIP model missing 'encode' method - invalid CLIP: {ckpt_name}")
        
        print("✓ Component validation passed")
    
    def _suggest_solutions(self) -> None:
        """Suggest solutions for OOM errors"""
        print("\n💡 Possible solutions:")
        print("   1. Close other applications using GPU memory")
        print("   2. Restart ComfyUI to clear memory")
        print("   3. Use a smaller model or quantized version")
        print("   4. Check for other loaded models in ComfyUI")
        print("   5. Verify PyTorch/ROCm versions are up to date")


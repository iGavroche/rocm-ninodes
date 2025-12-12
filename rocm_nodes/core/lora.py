"""
LoRA loader node for ROCM Ninodes.

Contains LoRA loading implementation:
- ROCMLoRALoader: ROCm-optimized LoRA loader with memory management
"""

from typing import Tuple, Optional

import torch
import folder_paths
from comfy.lora import load_lora

# Import utilities
from ..utils.memory import gentle_memory_cleanup


class ROCMLoRALoader:
    """
    ROCM-optimized LoRA loader with aggressive memory management to prevent fragmentation.
    Specifically designed to handle LoRA loading operations that cause OOM errors.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "The model to apply LoRA to"}),
                "lora_name": (folder_paths.get_filename_list("loras"), {"tooltip": "LoRA file to load"}),
                "strength_model": ("FLOAT", {
                    "default": 1.0, 
                    "min": -10.0, 
                    "max": 10.0, 
                    "step": 0.01,
                    "tooltip": "Strength of LoRA effect on model"
                }),
                "strength_clip": ("FLOAT", {
                    "default": 1.0, 
                    "min": -10.0, 
                    "max": 10.0, 
                    "step": 0.01,
                    "tooltip": "Strength of LoRA effect on CLIP"
                }),
            },
            "optional": {
                "clip": ("CLIP", {"tooltip": "CLIP model to apply LoRA to"}),
            }
        }
    
    RETURN_TYPES = ("MODEL", "CLIP")
    RETURN_NAMES = ("MODEL", "CLIP")
    FUNCTION = "load_lora"
    CATEGORY = "ROCm Ninodes/Loaders"
    DESCRIPTION = "ROCM-optimized LoRA loader with aggressive memory management"
    
    def load_lora(self, model, lora_name, strength_model, strength_clip, clip=None):
        """
        Load LoRA with aggressive memory management to prevent fragmentation
        """
        print(f"[LOADING] ROCM LoRA Loader: Loading {lora_name} with strengths {strength_model}/{strength_clip}")
        
        # Pre-loading cleanup
        if torch.cuda.is_available():
            print("[INFO] Pre-loading memory cleanup...")
            gentle_memory_cleanup()
            
            allocated_memory = torch.cuda.memory_allocated(0)
            reserved_memory = torch.cuda.memory_reserved(0)
            fragmentation = reserved_memory - allocated_memory
            
            print(f"Memory before LoRA loading: {allocated_memory/1024**3:.2f}GB allocated, {reserved_memory/1024**3:.2f}GB reserved")
            print(f"Fragmentation: {fragmentation/1024**2:.1f}MB")
        
        try:
            # Load LoRA
            lora_path = folder_paths.get_full_path("loras", lora_name)
            if lora_path is None:
                raise FileNotFoundError(f"LoRA file not found: {lora_name}")
            
            print(f"[LOADING] Loading LoRA from: {lora_path}")
            
            if torch.cuda.is_available():
                gentle_memory_cleanup()
            
            # Load the LoRA
            # ComfyUI's load_lora requires 'to_load' parameter specifying which components to load
            to_load = {}
            if model is not None:
                to_load["model"] = True
            if clip is not None:
                to_load["clip"] = True
            
            # If no components specified, load both by default
            if not to_load:
                to_load = {"model": True, "clip": True}
            
            lora = load_lora(lora_path, to_load)
            
            if torch.cuda.is_available():
                gentle_memory_cleanup()
            
            # Apply LoRA to model
            model_lora, clip_lora = lora.apply_to_model(model, clip, strength_model, strength_clip)
            
            # Post-loading cleanup
            if torch.cuda.is_available():
                print("[INFO] Post-loading memory cleanup...")
                gentle_memory_cleanup()
                
                allocated_memory_after = torch.cuda.memory_allocated(0)
                reserved_memory_after = torch.cuda.memory_reserved(0)
                fragmentation_after = reserved_memory_after - allocated_memory_after
                
                print(f"Memory after LoRA loading: {allocated_memory_after/1024**3:.2f}GB allocated, {reserved_memory_after/1024**3:.2f}GB reserved")
                print(f"Fragmentation: {fragmentation_after/1024**2:.1f}MB")
            
            print(f"[SUCCESS] ROCM LoRA Loader: Successfully loaded {lora_name}")
            return (model_lora, clip_lora)
            
        except Exception as e:
            print(f"[ERROR] ROCM LoRA Loader: Failed to load {lora_name}: {e}")
            
            if torch.cuda.is_available():
                print("[WARNING] Emergency memory cleanup...")
                gentle_memory_cleanup()
            
            raise e


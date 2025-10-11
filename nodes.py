#!/usr/bin/env python3
"""
SIMPLE ROCM NODES - EMERGENCY PERFORMANCE FIX
Removed all optimization overhead that was causing slowdown
"""
import torch
import torch.nn.functional as F
import comfy.model_management as model_management
import comfy.utils
import time
import logging
from typing import Dict, Any, Tuple, Optional

class ROCMOptimizedVAEDecode:
    """
    Simple ROCM VAE Decode - No optimization overhead
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "samples": ("LATENT", {"tooltip": "The latent to be decoded."}),
                "vae": ("VAE", {"tooltip": "The VAE model used for decoding the latent."}),
                "tile_size": ("INT", {
                    "default": 768, 
                    "min": 256, 
                    "max": 2048, 
                    "step": 64,
                    "tooltip": "Tile size for decoding."
                }),
                "overlap": ("INT", {
                    "default": 96, 
                    "min": 32, 
                    "max": 512, 
                    "step": 16,
                    "tooltip": "Overlap between tiles."
                }),
                "use_rocm_optimizations": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Enable ROCm-specific optimizations."
                }),
                "precision_mode": (["auto", "fp32", "fp16", "mixed"], {
                    "default": "auto",
                    "tooltip": "Precision mode."
                }),
                "batch_optimization": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Enable batch processing optimizations."
                }),
                "video_chunk_size": ("INT", {
                    "default": 8, 
                    "min": 1, 
                    "max": 32, 
                    "step": 1,
                    "tooltip": "Video chunk size."
                }),
                "memory_optimization_enabled": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Enable memory optimizations."
                }),
                "adaptive_tiling": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Enable adaptive tiling."
                })
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("IMAGE",)
    FUNCTION = "decode"
    CATEGORY = "ROCM Optimized/VAE"
    
    def decode(self, vae, samples, tile_size=768, overlap=96, use_rocm_optimizations=True, 
               precision_mode="auto", batch_optimization=True, video_chunk_size=8, 
               memory_optimization_enabled=True, adaptive_tiling=True):
        """
        Simple VAE decode - No optimization overhead
        """
        try:
            samples_processed = samples["samples"]
            
            # Simple precision handling
            if precision_mode == "auto":
                optimal_dtype = vae.vae_dtype
            elif precision_mode == "fp16":
                optimal_dtype = torch.float16
            elif precision_mode == "fp32":
                optimal_dtype = torch.float32
            else:
                optimal_dtype = vae.vae_dtype
            
            # Calculate memory requirements
            memory_used = vae.memory_used_decode(samples_processed.shape, optimal_dtype)
            
            # Load VAE
            model_management.load_models_gpu([vae.patcher], memory_required=memory_used)
            
            # Check if we can do direct decode
            if len(samples_processed.shape) == 4:  # Image
                B, C, H, W = samples_processed.shape
                
                # Try direct decode first
                try:
                    with torch.amp.autocast('cuda', enabled=(optimal_dtype != torch.float32), dtype=optimal_dtype):
                        out = vae.first_stage_model.decode(samples_processed.to(optimal_dtype))
                    
                    pixel_samples = vae.process_output(out)
                    pixel_samples = pixel_samples.to(vae.output_device).float()
                    
                    return (pixel_samples,)
                    
                except Exception as e:
                    logging.warning(f"Direct decode failed, falling back to tiled: {e}")
            
            # Fallback to tiled decode
            try:
                pixel_samples = self._decode_tiled(vae, samples, tile_size, overlap)
                return (pixel_samples,)
                
            except Exception as e:
                logging.warning(f"Tiled decode failed, falling back to standard VAE: {e}")
                
                # Final fallback to standard VAE decode
                pixel_samples = vae.decode(samples_processed)
                pixel_samples = vae.process_output(pixel_samples)
                
                return (pixel_samples,)
                
        except Exception as e:
            logging.error(f"VAE decode failed: {e}")
            raise e
    
    def _decode_tiled(self, vae, samples, tile_size, overlap):
        """Simple tiled decode"""
        samples_processed = samples["samples"]
        
        # Handle video tensors (5D) vs image tensors (4D)
        if len(samples_processed.shape) == 5:  # Video: B, C, T, H, W
            B, C, T, H, W = samples_processed.shape
            # For video, process each frame separately
            frames = []
            for t in range(T):
                frame_samples = samples_processed[:, :, t, :, :]  # Remove temporal dimension for processing
                frame_output = self._decode_single_frame(vae, frame_samples, tile_size, overlap)
                frames.append(frame_output)
            return torch.stack(frames, dim=2)  # Stack along temporal dimension
        else:  # Image: B, C, H, W
            return self._decode_single_frame(vae, samples_processed, tile_size, overlap)
    
    def _decode_single_frame(self, vae, samples_processed, tile_size, overlap):
        """Decode a single frame (4D tensor)"""
        # Get compression ratios
        try:
            compression = vae.spacial_compression_decode()
            temporal_compression = vae.temporal_compression_decode()
        except:
            compression = 8
            temporal_compression = 1
        
        # Calculate tile dimensions
        tile_h = tile_size // compression
        tile_w = tile_size // compression
        
        # Calculate number of tiles needed
        B, C, H, W = samples_processed.shape
        tiles_h = (H + tile_h - 1) // tile_h
        tiles_w = (W + tile_w - 1) // tile_w
        
        # Pre-allocate output tensor
        output_shape = (B, 3, H * compression, W * compression)
        pixel_samples = torch.empty(output_shape, dtype=torch.float32, device=samples_processed.device)
        
        # Process tiles
        for i in range(tiles_h):
            for j in range(tiles_w):
                # Calculate tile boundaries with overlap
                start_h = i * tile_h
                end_h = min(start_h + tile_h + overlap, H)
                start_w = j * tile_w
                end_w = min(start_w + tile_w + overlap, W)
                
                # Extract tile
                tile_samples = samples_processed[:, :, start_h:end_h, start_w:end_w]
                
                # Decode tile
                with torch.amp.autocast('cuda', enabled=True, dtype=vae.vae_dtype):
                    tile_output = vae.first_stage_model.decode(tile_samples.to(vae.vae_dtype))
                
                # Process output
                tile_output = vae.process_output(tile_output)
                
                # Copy to output with proper positioning
                output_start_h = start_h * compression
                output_end_h = end_h * compression
                output_start_w = start_w * compression
                output_end_w = end_w * compression
                
                pixel_samples[:, :, output_start_h:output_end_h, output_start_w:output_end_w] = tile_output
        
        return pixel_samples

# Apply no-op instrumentation
def instrument_node(node_class):
    """No-op decorator - instrumentation disabled"""
    return node_class

# Create no-op instrumentation object
class NodeInstrumentation:
    def __init__(self, *args, **kwargs):
        pass
    
    def capture_inputs(self, *args, **kwargs):
        return ""
    
    def capture_outputs(self, *args, **kwargs):
        return ""
    
    def capture_performance(self, *args, **kwargs):
        return ""

# Create global instrumentation object
instrumentation = NodeInstrumentation()

# Apply instrumentation (no-op)
@instrument_node
class ROCMOptimizedVAEDecodeInstrumented(ROCMOptimizedVAEDecode):
    pass

# Node class mapping for ComfyUI
NODE_CLASS_MAPPINGS = {
    "ROCMOptimizedVAEDecode": ROCMOptimizedVAEDecodeInstrumented
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ROCMOptimizedVAEDecode": "ROCM Optimized VAE Decode"
}

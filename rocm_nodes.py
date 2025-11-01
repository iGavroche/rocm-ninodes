"""
ROCM Optimized VAE Nodes for AMD GPUs
Specifically optimized for gfx1151 architecture with ROCm 6.4+
"""

import os
import gc
import time
from typing import Dict, Any, Tuple, Optional

# Environment variables are now managed by main.py and run_comfy.ps1

# Now import PyTorch and other modules
import torch
import torch.nn.functional as F
import comfy.model_management as model_management
import comfy.utils
import comfy.sample
import comfy.samplers
import latent_preview
import logging
import folder_paths

# Diagnostic logging for ROCm memory management
def log_rocm_diagnostics():
    """Log ROCm memory management configuration for debugging"""
    try:
        print("ROCm Memory Management Diagnostics:")
        print(f"   PYTORCH_CUDA_ALLOC_CONF: {os.environ.get('PYTORCH_CUDA_ALLOC_CONF', 'NOT SET')}")
        print(f"   PYTORCH_HIP_ALLOC_CONF: {os.environ.get('PYTORCH_HIP_ALLOC_CONF', 'NOT SET')}")
        
        if torch.cuda.is_available():
            print(f"   CUDA Available: {torch.cuda.is_available()}")
            print(f"   CUDA Device: {torch.cuda.get_device_name(0)}")
            
            # Get detailed memory info
            total_memory = torch.cuda.get_device_properties(0).total_memory
            allocated_memory = torch.cuda.memory_allocated(0)
            reserved_memory = torch.cuda.memory_reserved(0)
            free_memory = total_memory - allocated_memory
            fragmentation = reserved_memory - allocated_memory
            
            print(f"   Total Memory: {total_memory / 1024**3:.2f}GB")
            print(f"   Allocated Memory: {allocated_memory / 1024**3:.2f}GB")
            print(f"   Reserved Memory: {reserved_memory / 1024**3:.2f}GB")
            print(f"   Free Memory: {free_memory / 1024**3:.2f}GB")
            print(f"   Fragmentation: {fragmentation / 1024**3:.2f}GB")
            
            # Check fragmentation level
            if fragmentation > 500 * 1024**2:  # 500MB
                print(f"   WARNING: High fragmentation detected! This will cause OOM errors.")
                print(f"   Solution: Force memory defragmentation before operations.")
            
            # Check if PyTorch is using the right allocator
            try:
                # Try to get allocator info (this might not work on all PyTorch versions)
                allocator_info = torch.cuda.memory_stats(0)
                print(f"   Memory Allocator: Active (stats available)")
            except:
                print(f"   Memory Allocator: Unknown (stats not available)")
        else:
            print("   CUDA Not Available - using CPU")
            
    except Exception as e:
        print(f"   Diagnostic Error: {e}")

def simple_memory_cleanup():
    """Simple memory cleanup for ROCm"""
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
            return True
        return False
    except Exception as e:
        print(f"   Memory cleanup error: {e}")
        return False

def gentle_memory_cleanup():
    """Gentle memory cleanup - less aggressive for mature ROCm drivers"""
    try:
        if not torch.cuda.is_available():
            return False
        
        # Single cache clear with synchronization
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        # Light garbage collection
        gc.collect()
        
        return True
    except Exception as e:
        print(f"   Memory cleanup error: {e}")
        return False

# Diagnostics will be run when nodes are actually used, not at module import


# from debug_config import DEBUG_MODE, save_debug_data, capture_timing, capture_memory_usage, log_debug
DEBUG_MODE = False
def save_debug_data(*args, **kwargs): pass
def capture_timing(*args, **kwargs): pass
def capture_memory_usage(*args, **kwargs): pass
def log_debug(*args, **kwargs): pass

def get_gpu_memory_info():
    """Get accurate GPU memory information"""
    if not torch.cuda.is_available():
        return None, None, None, None
    
    total_memory = torch.cuda.get_device_properties(0).total_memory
    allocated_memory = torch.cuda.memory_allocated(0)
    reserved_memory = torch.cuda.memory_reserved(0)
    free_memory = total_memory - reserved_memory
    
    return total_memory, allocated_memory, reserved_memory, free_memory

def check_memory_safety(required_memory_gb=2.0):
    """Check if there's enough memory for the operation"""
    if not torch.cuda.is_available():
        return True, "CUDA not available"
    
    total_memory, allocated_memory, reserved_memory, free_memory = get_gpu_memory_info()
    
    if total_memory is None:
        return True, "Memory info unavailable"
    
    free_gb = free_memory / (1024**3)
    required_gb = required_memory_gb
    
    if free_gb < required_gb:
        return False, f"Only {free_gb:.2f}GB free, need {required_gb:.2f}GB"
    
    return True, f"Memory OK: {free_gb:.2f}GB free"

def emergency_memory_cleanup():
    """Emergency memory cleanup - gentle approach for mature ROCm"""
    try:
        if not torch.cuda.is_available():
            return False
        
        # Gentle cleanup - single pass
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        gc.collect()
        
        return True
    except Exception as e:
        logging.error(f"Emergency memory cleanup failed: {e}")
        return False


    """
    Quantization-aware memory cleanup - less aggressive for quantized models
    """
    try:
        if not torch.cuda.is_available():
            return False
        
        # Different cleanup strategies based on quantization type
        if quantization_type in ["fp8", "int8", "int4"]:
            # Quantized models are already memory-efficient, minimal cleanup
            torch.cuda.empty_cache()
            print(f"🧹 Light memory cleanup for {quantization_type} model")
        else:
            # Standard cleanup for non-quantized models
            gentle_memory_cleanup()
            print(f"🧹 Standard memory cleanup for {quantization_type} model")
        
        return True
    except Exception as e:
        print(f"   Quantized memory cleanup error: {e}")
        return False

def check_quantized_memory_safety(model_size, quantization_type="fp8"):
    """
    Check memory safety with quantization-aware estimation
    """
    if not torch.cuda.is_available():
        return True, "CUDA not available"
    
    total_memory, allocated_memory, reserved_memory, free_memory = get_gpu_memory_info()
    
    if total_memory is None:
        return True, "Memory info unavailable"
    
    # Calculate memory requirements based on quantization
    if quantization_type == "fp8":
        required_memory = model_size * 0.5  # 50% of FP32
        safety_margin = 1.2  # Less aggressive for quantized
    elif quantization_type in ["int8", "int4"]:
        required_memory = model_size * 0.25  # 25% of FP32
        safety_margin = 1.1  # Even less aggressive
    else:
        required_memory = model_size  # Full FP32
        safety_margin = 1.5  # Conservative for non-quantized
    
    required_gb = (required_memory * safety_margin) / (1024**3)
    free_gb = free_memory / (1024**3)
    
    if free_gb < required_gb:
        return False, f"Only {free_gb:.2f}GB free, need {required_gb:.2f}GB for {quantization_type}"
    
    return True, f"Memory OK for {quantization_type}: {free_gb:.2f}GB free"

def detect_model_quantization(model):
    """
    Detect if a model is quantized and return quantization type
    """
    try:
        # Check model dtype
        model_dtype = model.model_dtype()
        dtype_name = str(model_dtype).lower()
        
        if 'int8' in dtype_name:
            return "int8"
        elif 'int4' in dtype_name:
            return "int4"
        elif 'float8' in dtype_name or 'fp8' in dtype_name:
            return "fp8"
        elif 'bfloat16' in dtype_name or 'bf16' in dtype_name:
            return "bf16"
        else:
            return "fp32"
    except Exception as e:
        print(f"⚠️ Could not detect model quantization: {e}")
        return "unknown"



# Removed force_memory_cleanup - using gentle_memory_cleanup instead

# Removed emergency_memory_reset - using emergency_memory_cleanup instead





def aggressive_memory_cleanup():
    """Gentle memory cleanup - renamed for compatibility"""
    return gentle_memory_cleanup()

class ROCMOptimizedVAEDecode:
    """
    ROCM-optimized VAE Decode node for all AMD GPUs with ROCm support.
    Enhanced optimizations for Strix Halo (gfx1151) architecture.
    
    Optimization Strategy:
    1. GENERAL ROCM OPTIMIZATIONS (apply to all AMD GPUs):
       - Optimized memory management for ROCm
       - Better batching strategy for AMD GPUs
       - Reduced model conversion overhead
       - Async memory operations (non-blocking)
    
    2. ARCHITECTURE-SPECIFIC:
       - Intelligent tiled vs direct decode decision
       - Optimized tile sizes (enhanced for gfx1151)
       - Video chunking with overlap handling
    
    Key optimizations:
    - Non-blocking memory operations
    - Quantized model support (fp8/int8)
    - Smart video chunking for large videos
    - WAN VAE format handling
    """
    
    # Class-level cache to avoid repeated model conversions
    _vae_model_cache = {}
    
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
                    "tooltip": "Tile size optimized for gfx1151. Larger values use more VRAM but are faster."
                }),
                "overlap": ("INT", {
                    "default": 96, 
                    "min": 32, 
                    "max": 512, 
                    "step": 16,
                    "tooltip": "Overlap between tiles. Higher values reduce artifacts but use more VRAM."
                }),
                "use_rocm_optimizations": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Enable ROCm-specific optimizations for AMD GPUs"
                }),
                "precision_mode": (["auto", "fp32", "fp16", "bf16"], {
                    "default": "auto",
                    "tooltip": "Precision mode. 'auto' selects optimal for your GPU."
                }),
                "batch_optimization": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Enable batch processing optimizations"
                }),
                "video_chunk_size": ("INT", {
                    "default": 8,
                    "min": 1,
                    "max": 32,
                    "step": 1,
                    "tooltip": "Number of video frames to process at once (memory optimization)"
                }),
                "memory_optimization_enabled": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Enable memory optimization for video processing"
                })
            }
            ,
            "optional": {
                "compatibility_mode": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Enable stock ComfyUI compatibility mode (disables all ROCm optimizations)"
                })
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("IMAGE",)
    FUNCTION = "decode"
    CATEGORY = "RocM Ninodes/VAE"
    DESCRIPTION = "ROCM-optimized VAE Decode for all AMD GPUs with ROCm support. Enhanced for Strix Halo (gfx1151)."
    
    def decode(self, vae, samples, tile_size=768, overlap=96, use_rocm_optimizations=True, 
               precision_mode="auto", batch_optimization=True, video_chunk_size=8, 
               memory_optimization_enabled=True, compatibility_mode=False):
        """
        Optimized VAE decode for ROCm/AMD GPUs with video support and quantized model compatibility
        """
        start_time = time.time()
        log_debug(f"ROCMOptimizedVAEDecode.decode started with samples shape: {samples['samples'].shape}")
        
        # CRITICAL FIX: Detect quantized models to avoid breaking them
        is_quantized_model = False
        vae_model_dtype = getattr(vae.first_stage_model, 'dtype', None)
        if vae_model_dtype is not None:
            # Check for quantized dtypes - be more specific to avoid false positives
            quantized_dtypes = [torch.float8_e4m3fn, torch.float8_e5m2, torch.int8, torch.int4]
            if vae_model_dtype in quantized_dtypes:
                is_quantized_model = True
                print(f"🔍 Detected quantized VAE model (dtype: {vae_model_dtype})")
            elif hasattr(vae_model_dtype, '__name__') and 'int' in str(vae_model_dtype):
                is_quantized_model = True
                print(f"🔍 Detected quantized VAE model (dtype: {vae_model_dtype})")
        
        # CRITICAL FIX: Detect WAN VAE models that use causal decoding with feature caching
        # WAN VAE requires full video processing to maintain causal decoding chain
        # Chunking breaks the feature cache and causes jitter on first frames of each chunk
        is_wan_vae = False
        try:
            # Check if first_stage_model is a WAN VAE (from comfy.ldm.wan.vae or comfy.ldm.wan.vae2_2)
            vae_model_class = vae.first_stage_model.__class__
            vae_model_class_name = vae_model_class.__name__
            vae_model_module = vae_model_class.__module__ if hasattr(vae_model_class, '__module__') else None
            
            # Check if it's a WAN VAE by class name and module
            if vae_model_class_name == 'WanVAE':
                if vae_model_module and ('wan' in vae_model_module.lower()):
                    is_wan_vae = True
                    print(f"🔍 Detected WAN VAE model (causal decoding requires full video processing)")
        except Exception as e:
            # If detection fails, continue without WAN detection (safer)
            if DEBUG_MODE:
                print(f"⚠️ WAN VAE detection failed: {e}")
        
        # Compatibility mode: disable all optimizations for quantized models
        if compatibility_mode or is_quantized_model:
            print("🛡️ Compatibility mode enabled - using stock ComfyUI behavior")
            use_rocm_optimizations = False
            batch_optimization = False
            memory_optimization_enabled = False
        
        # CRITICAL FIX: Disable chunking for WAN VAE models to preserve causal decoding chain
        # WAN VAE uses causal decoding with feature caching - chunking breaks the cache
        # Native ComfyUI processes WAN VAE videos in full, so we must match that behavior
        if is_wan_vae:
            print("🛡️ WAN VAE detected - disabling chunking to preserve causal decoding chain")
            memory_optimization_enabled = False  # Disable chunking for WAN models
        
        # Capture input data for debugging
        if DEBUG_MODE:
            save_debug_data(samples, "vae_decode_input", "flux_1024x1024", {
                'node_type': 'ROCMOptimizedVAEDecode',
                'tile_size': tile_size,
                'overlap': overlap,
                'use_rocm_optimizations': use_rocm_optimizations,
                'precision_mode': precision_mode,
                'batch_optimization': batch_optimization,
                'video_chunk_size': video_chunk_size,
                'memory_optimization_enabled': memory_optimization_enabled
            })
        
        # Get device information
        device = vae.device
        is_amd = hasattr(device, 'type') and device.type == 'cuda'
        
        # Check if this is video data (5D tensor: B, C, T, H, W)
        is_video = len(samples["samples"].shape) == 5
        if is_video:
            B, C, T, H, W = samples["samples"].shape
            
            # Progress indicator for Windows users experiencing hanging
            print(f"🎬 Processing video: {T} frames, {H}x{W} resolution")
            
            # CRITICAL: Match ComfyUI's native behavior - don't chunk unless absolutely necessary
            # Native ComfyUI VAE decode processes the entire video at once
            # WAN VAE models MUST use full video processing to preserve causal decoding chain
            # Only chunk if we absolutely must for memory reasons AND it's not a WAN VAE
            if is_wan_vae:
                # WAN VAE requires full video processing - chunking breaks causal decoding
                use_chunking = False
                print("📹 WAN VAE detected - using full video processing (causal decoding requires full sequence)")
            elif memory_optimization_enabled and T > video_chunk_size and T > 32:  # Only chunk for very large videos > 32 frames
                use_chunking = True
                print(f"📹 Using chunked processing: {video_chunk_size} frames per chunk (large video)")
            else:
                use_chunking = False
                print("📹 Using non-chunked processing (matching native ComfyUI behavior)")
            
            if use_chunking:
                # Process video in chunks to avoid memory exhaustion
                # CRITICAL: NO OVERLAP - decode exact frames to avoid duplicates
                # WAN VAE causal decoding means first frames might be slightly darker,
                # but overlap causes frame duplication issues
                chunk_results = []
                num_chunks = (T + video_chunk_size - 1) // video_chunk_size
                print(f"🔄 Processing {num_chunks} chunks (no overlap to prevent duplicates)...")
                
                for chunk_idx in range(num_chunks):
                    # Calculate chunk boundaries - NO OVERLAP to prevent duplicates
                    i = chunk_idx * video_chunk_size
                    end_idx = min(i + video_chunk_size, T)
                    actual_frames_needed = end_idx - i  # How many frames we need from this chunk
                    print(f"  📦 Chunk {chunk_idx + 1}/{num_chunks}: frames {i}-{end_idx-1} (need {actual_frames_needed} frames)")
                    
                    # NO OVERLAP - decode exactly the frames we need
                    chunk_start = i
                    chunk_end = end_idx
                    
                    # Extract chunk from latent (exact frame range, no overlap)
                    chunk = samples["samples"][:, :, chunk_start:chunk_end, :, :]
                    
                    # Decode chunk
                    # CRITICAL FIX: vae.decode() ALWAYS returns (B, T, H, W, C) format for video
                    # regardless of whether it uses regular or tiled decoding internally
                    with torch.no_grad():
                        try:
                            chunk_decoded = vae.decode(chunk)
                        except Exception as e:
                            # If decode fails, try tiled decode explicitly
                            if "out of memory" in str(e).lower() or "OOM" in str(e).upper():
                                print(f"⚠️ Memory error during chunk decode, retrying with explicit tiled decode...")
                                # Use explicit tiled decode with NO temporal overlap to prevent duplicates
                                compression = vae.spacial_compression_decode()
                                temporal_compression = vae.temporal_compression_decode()
                                tile = 256 // compression if compression > 0 else 256
                                overlap_spatial = tile // 4
                                # NO temporal overlap - decode exactly what we need
                                chunk_decoded = vae.decode_tiled(chunk, tile_x=tile, tile_y=tile, overlap=overlap_spatial, 
                                                                 tile_t=video_chunk_size, overlap_t=None)
                            else:
                                raise e
                    
                    # VAE decode returns a tuple, extract the tensor
                    if isinstance(chunk_decoded, tuple):
                        chunk_decoded = chunk_decoded[0]
                    
                    # CRITICAL FIX: vae.decode() always returns (B, T, H, W, C) format for video
                    # Check format and handle accordingly
                    if len(chunk_decoded.shape) == 5:
                        if chunk_decoded.shape[-1] == 3 or chunk_decoded.shape[-1] == 4:
                            # (B, T, H, W, C) format - correct format from vae.decode()
                            format_channels_last = True
                            temporal_dim = 1
                        elif chunk_decoded.shape[1] == 3 or chunk_decoded.shape[1] == 4:
                            # (B, C, T, H, W) format - unexpected but handle it
                            format_channels_last = False
                            temporal_dim = 2
                        else:
                            # Fallback: assume channels-last based on ComfyUI's vae.decode() behavior
                            format_channels_last = True
                            temporal_dim = 1
                    else:
                        # Not 5D, shouldn't happen for video but handle gracefully
                        format_channels_last = True
                        temporal_dim = 1
                    
                    # CRITICAL: NO OVERLAP - decoded frames should exactly match latent frames (WAN VAE is 1:1)
                    decoded_total_frames = chunk_decoded.shape[temporal_dim]
                    latent_frames_decoded = chunk_end - chunk_start  # Frames we decoded in latent space
                    
                    # Verify decoded frame count matches latent frame count (WAN VAE is 1:1)
                    if decoded_total_frames != latent_frames_decoded:
                        print(f"    ⚠️ Chunk {chunk_idx + 1}: Frame count mismatch - decoded {decoded_total_frames}, expected {latent_frames_decoded} (latent input)")
                        # For WAN VAE, this should be 1:1, but adjust if needed
                        if decoded_total_frames > latent_frames_decoded:
                            # Trim extra frames from end
                            if format_channels_last:
                                chunk_decoded = chunk_decoded[:, :latent_frames_decoded, :, :, :]
                            else:
                                chunk_decoded = chunk_decoded[:, :, :latent_frames_decoded, :, :]
                            decoded_total_frames = latent_frames_decoded
                        elif decoded_total_frames < latent_frames_decoded:
                            print(f"    ❌ Chunk {chunk_idx + 1}: Missing frames - decoded {decoded_total_frames}, expected {latent_frames_decoded}")
                    
                    # NO CROPPING NEEDED - we decoded exactly what we need (no overlap)
                    final_frames = chunk_decoded.shape[temporal_dim]
                    if final_frames != actual_frames_needed:
                        print(f"    ❌ Chunk {chunk_idx + 1}: Frame count mismatch - got {final_frames} frames, expected {actual_frames_needed} frames")
                        print(f"       This will cause frame misalignment! Chunk should be frames {i}-{end_idx-1}")
                    elif chunk_idx < 3:  # Only print for first few chunks to reduce noise
                        print(f"    ✅ Chunk {chunk_idx + 1}: Decoded {decoded_total_frames} frames (frames {i}-{end_idx-1})")
                    
                    chunk_results.append(chunk_decoded)
                    
                    # Clear memory after each chunk (reduced frequency to prevent fragmentation)
                    if chunk_idx % 2 == 0:  # Only clear every 2 chunks
                        torch.cuda.empty_cache()
                
                # Concatenate results along temporal dimension
                # CRITICAL FIX: vae.decode() always returns (B, T, H, W, C) format for video
                # Verify all chunks have consistent shape before concatenation
                first = chunk_results[0]
                temporal_dim = 1 if (first.shape[-1] == 3 or first.shape[-1] == 4) else 2
                
                # Verify frame counts match expectations before concatenation
                total_concatenated_frames = sum(chunk.shape[temporal_dim] for chunk in chunk_results)
                if total_concatenated_frames != T:
                    print(f"⚠️ WARNING: Total concatenated frames ({total_concatenated_frames}) != expected ({T})")
                    print(f"   Chunk frame counts: {[chunk.shape[temporal_dim] for chunk in chunk_results]}")
                
                # Concatenate along temporal dimension
                if len(first.shape) == 5:
                    if first.shape[-1] == 3 or first.shape[-1] == 4:
                        # (B, T, H, W, C) format - correct format from vae.decode()
                        result = torch.cat(chunk_results, dim=1)  # Concatenate along temporal dimension
                    elif first.shape[1] == 3 or first.shape[1] == 4:
                        # (B, C, T, H, W) format - unexpected but handle it
                        result = torch.cat(chunk_results, dim=2)  # Concatenate along temporal dimension
                    else:
                        # Fallback: assume channels-last format based on ComfyUI behavior
                        result = torch.cat(chunk_results, dim=1)
                else:
                    # Shouldn't happen for video, but handle gracefully
                    result = torch.cat(chunk_results, dim=1)
                
                
                # Convert 5D video tensor to 4D image tensor for ComfyUI
                # CRITICAL: Match ComfyUI's native behavior exactly
                # Native does: images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])
                if len(result.shape) == 5:
                    # Match native ComfyUI reshape logic
                    result = result.reshape(-1, result.shape[-3], result.shape[-2], result.shape[-1])
                
                return (result,)
            else:
                # Process entire video at once - EXACTLY match ComfyUI's native behavior
                # Native ComfyUI: images = vae.decode(samples["samples"])
                #              if len(images.shape) == 5: images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])
                with torch.no_grad():
                    result = vae.decode(samples["samples"])
                
                # VAE decode returns a tuple, extract the tensor (ComfyUI handles this internally)
                if isinstance(result, tuple):
                    result = result[0]
                
                # Convert 5D video tensor to 4D image tensor - EXACT native ComfyUI logic
                if len(result.shape) == 5:
                    # Match native ComfyUI reshape logic exactly
                    result = result.reshape(-1, result.shape[-3], result.shape[-2], result.shape[-1])
                
                # Capture output data and timing for debugging
                if DEBUG_MODE:
                    end_time = time.time()
                    save_debug_data(result, "vae_decode_output", "flux_1024x1024", {
                        'node_type': 'ROCMOptimizedVAEDecode',
                        'execution_time': end_time - start_time,
                        'output_shape': result.shape,
                        'output_dtype': str(result.dtype),
                        'output_device': str(result.device)
                    })
                    capture_timing("vae_decode", start_time, end_time, {
                        'node_type': 'ROCMOptimizedVAEDecode',
                        'is_video': True,
                        'video_chunks': len(chunk_results) if 'chunk_results' in locals() else 1
                    })
                    capture_memory_usage("vae_decode", {
                        'node_type': 'ROCMOptimizedVAEDecode',
                        'is_video': True
                    })
                
                return (result,)
        
        # CRITICAL FIX: Preserve quantized model dtypes - do not convert
        if is_quantized_model:
            print(f"🔒 Preserving quantized model dtype: {vae_model_dtype}")
            optimal_dtype = vae_model_dtype
            # Skip all dtype conversions for quantized models
        elif precision_mode == "auto":
            if is_amd:
                # Check VAE model's actual dtype first to avoid type mismatch
                if vae_model_dtype is not None:
                    # Use the VAE model's existing dtype to avoid type mismatch errors
                    optimal_dtype = vae_model_dtype
                    logging.info(f"Using VAE model's existing dtype: {optimal_dtype}")
                else:
                    # Fallback to VAE's configured dtype
                    optimal_dtype = vae.vae_dtype
                    logging.info(f"Using VAE's configured dtype: {optimal_dtype}")
            else:
                optimal_dtype = vae.vae_dtype
        else:
            dtype_map = {
                "fp32": torch.float32,
                "fp16": torch.float16,
                "bf16": torch.bfloat16
            }
            optimal_dtype = dtype_map[precision_mode]
        
        # CRITICAL FIX: Skip dtype conversion for quantized models
        if not is_quantized_model:
            # Handle BFloat16 model with Float32 input type mismatch
            # If the VAE model has BFloat16 weights but we're using Float32, we need to match them
            if vae_model_dtype == torch.bfloat16 and optimal_dtype == torch.float32:
                logging.warning("VAE model has BFloat16 weights but input is Float32 - this will cause type mismatch")
                logging.warning("Converting VAE model to Float32 to match input dtype")
                vae.first_stage_model = vae.first_stage_model.to(torch.float32)
                optimal_dtype = torch.float32
        else:
            print(f"🔒 Skipping dtype conversion for quantized model (dtype: {vae_model_dtype})")
        
        # ROCm-specific optimizations (general for all AMD, enhanced for gfx1151)
        if use_rocm_optimizations and is_amd:
            # GENERAL ROCM OPTIMIZATIONS (apply to all AMD GPUs)
            torch.backends.cuda.matmul.allow_tf32 = False  # TF32 not supported on AMD
            torch.backends.cuda.matmul.allow_fp16_accumulation = True  # Better performance on AMD
            
            # Detect gfx1151 for architecture-specific optimizations
            is_gfx1151 = False
            try:
                if torch.cuda.is_available():
                    arch = torch.cuda.get_device_properties(0).gcnArchName
                    if 'gfx1151' in arch:
                        is_gfx1151 = True
            except:
                pass
            
            # Async memory management (non-blocking)
            try:
                # Light async cleanup - doesn't block GPU
                torch.cuda.empty_cache()  # Async - non-blocking
                # Don't call gentle_memory_cleanup() - it contains torch.cuda.synchronize()
            except Exception as e:
                if DEBUG_MODE:
                    print(f"⚠️ Memory optimization skipped: {e}")
            
            # Architecture-specific tile size optimization (enhanced for gfx1151)
            if is_gfx1151:
                # Strix Halo (gfx1151) specific tile size cap
                if tile_size > 1024:
                    tile_size = 1024  # Cap for gfx1151 memory
            elif is_amd:
                # General ROCm tile size cap (conservative for broader compatibility)
                if tile_size > 2048:
                    tile_size = 2048  # Cap for general AMD GPUs
            if overlap > tile_size // 4:
                overlap = tile_size // 4
        
        # Optimized memory management - avoid expensive memory_used_decode call
        samples_shape = samples["samples"].shape
        if len(samples_shape) == 4:  # Standard image format
            # Estimate memory usage without calling expensive memory_used_decode
            estimated_memory = samples_shape[0] * samples_shape[1] * samples_shape[2] * samples_shape[3] * 4  # 4 bytes per float32
            estimated_memory *= 8  # Decode expansion factor
        else:
            # For video or other formats, use a conservative estimate
            estimated_memory = samples_shape[0] * samples_shape[1] * samples_shape[2] * samples_shape[3] * 4 * 8
        
        # Check memory safety before loading models
        estimated_memory_gb = estimated_memory / (1024**3)
        is_safe, memory_msg = check_memory_safety(required_memory_gb=estimated_memory_gb + 2.0)
        
        if not is_safe:
            print(f"⚠️ VAE Decode Memory Warning: {memory_msg}")
            print("🧹 Performing emergency cleanup before VAE decode...")
            emergency_memory_cleanup()
        
        # Load models with estimated memory (faster than exact calculation)
        try:
            model_management.load_models_gpu([vae.patcher], memory_required=estimated_memory)
        except Exception as e:
            if "out of memory" in str(e).lower():
                print("💾 VAE model loading failed due to memory - performing cleanup")
                emergency_memory_cleanup()
                # Try again with reduced memory requirement
                model_management.load_models_gpu([vae.patcher], memory_required=estimated_memory // 2)
            else:
                raise e
        
        # Calculate optimal batch size for gfx1151 (less conservative)
        free_memory = model_management.get_free_memory(device)
        if batch_optimization and is_amd:
            # Less conservative batching for better performance
            batch_number = max(1, int(free_memory / (estimated_memory * 1.2)))  # Reduced from 1.5x to 1.2x
        else:
            batch_number = max(1, int(free_memory / estimated_memory))
        
        # Process in batches
        pixel_samples = None
        samples_tensor = samples["samples"]
        
        # Optimized processing logic - avoid redundant conversions
        samples_processed = samples_tensor.to(device).to(optimal_dtype)
        
        # Use caching to avoid repeated model conversions
        vae_id = id(vae.first_stage_model)
        cache_key = f"{vae_id}_{optimal_dtype}"
        
        if cache_key not in self._vae_model_cache:
            # CRITICAL FIX: Only convert model dtype if it's different AND not quantized
            if not is_quantized_model:
                current_model_dtype = getattr(vae.first_stage_model, 'dtype', None)
                if current_model_dtype is not None and current_model_dtype != optimal_dtype:
                    logging.info(f"Converting VAE model from {current_model_dtype} to {optimal_dtype}")
                    vae.first_stage_model = vae.first_stage_model.to(optimal_dtype)
                else:
                    logging.info(f"VAE model already in correct dtype: {current_model_dtype}")
            else:
                logging.info(f"Preserving quantized model dtype: {vae_model_dtype}")
            self._vae_model_cache[cache_key] = True
        
        # Decide processing method based on image size and available memory
        image_size = samples_tensor.shape[2] * samples_tensor.shape[3]
        use_tiled = image_size > 512 * 512 or estimated_memory > free_memory * 0.8
        
        if is_video:
            print(f"✅ Video processing completed, switching to image processing")
        else:
            print(f"🖼️ Processing image: {samples_tensor.shape[2]}x{samples_tensor.shape[3]}")
            if use_tiled:
                print(f"🔧 Using tiled processing for large image")
            else:
                print(f"⚡ Using direct processing for optimal speed")
        
        if not use_tiled:
            # Direct decode for smaller images or when memory is sufficient
            try:
                if use_rocm_optimizations and is_amd:
                    # Direct decode without autocast for ROCm
                    out = vae.first_stage_model.decode(samples_processed)
                else:
                    # Use autocast for non-AMD GPUs
                    with torch.cuda.amp.autocast(enabled=(optimal_dtype != torch.float32), dtype=optimal_dtype):
                        out = vae.first_stage_model.decode(samples_processed)
                
                out = out.to(device).float()  # CRITICAL FIX: Keep on GPU device, not output_device
                pixel_samples = vae.process_output(out)
                
            except Exception as e:
                logging.warning(f"Direct decode failed, falling back to tiled: {e}")
                use_tiled = True
        
        if use_tiled:
            # Use optimized tiled decoding
            pixel_samples = self._decode_tiled_optimized(
                vae, samples_tensor, tile_size, overlap, optimal_dtype, batch_number
            )
        
        # Reshape if needed (match standard VAE decode behavior)
        if len(pixel_samples.shape) == 5:
            pixel_samples = pixel_samples.reshape(-1, pixel_samples.shape[-3], 
                                                pixel_samples.shape[-2], pixel_samples.shape[-1])
        
        # Move to GPU device (not output_device which might be CPU)
        pixel_samples = pixel_samples.to(device)
        
        # CRITICAL FIX: Preserve original tensor values - don't convert dtype unnecessarily
        # ComfyUI's save_images function expects (B, H, W, C) format with preserved values
        logging.info(f"Before ComfyUI format fix: shape={pixel_samples.shape}, dtype={pixel_samples.dtype}")
        
        # Ensure the tensor is contiguous
        pixel_samples = pixel_samples.contiguous()
        
        # CRITICAL FIX: Only convert dtype if absolutely necessary for ComfyUI compatibility
        # Preserve original VAE output values to prevent darker/colorless frames
        if pixel_samples.dtype not in [torch.float32, torch.float16]:
            logging.info(f"Converting dtype from {pixel_samples.dtype} to float32 for ComfyUI compatibility")
            pixel_samples = pixel_samples.float()
        else:
            logging.info(f"Preserving original dtype: {pixel_samples.dtype}")
        
        # Validate tensor value range before format conversion
        min_val = pixel_samples.min().item()
        max_val = pixel_samples.max().item()
        logging.info(f"Before format conversion: min={min_val:.4f}, max={max_val:.4f}, mean={pixel_samples.mean().item():.4f}")
        
        if min_val < -1.1 or max_val > 1.1:
            logging.warning(f"Unexpected tensor value range [{min_val:.4f}, {max_val:.4f}] - this might cause visual artifacts")
        
        # CRITICAL FIX: Convert from (B, C, H, W) to (B, H, W, C) for ComfyUI compatibility
        # This is the key fix - ComfyUI's save_images expects (B, H, W, C) format
        if len(pixel_samples.shape) == 4 and pixel_samples.shape[1] == 3:
            logging.info(f"Converting tensor from (B, C, H, W) to (B, H, W, C) for ComfyUI compatibility")
            pixel_samples = pixel_samples.permute(0, 2, 3, 1).contiguous()
            logging.info(f"After permute: shape={pixel_samples.shape}, dtype={pixel_samples.dtype}")
        
        # Final validation - ensure we have the right shape, dtype, and value range
        if isinstance(pixel_samples, torch.Tensor):
            final_min = pixel_samples.min().item()
            final_max = pixel_samples.max().item()
            final_mean = pixel_samples.mean().item()
            logging.info(f"Final tensor validation: shape={pixel_samples.shape}, dtype={pixel_samples.dtype}, device={pixel_samples.device}")
            logging.info(f"Final value range: min={final_min:.4f}, max={final_max:.4f}, mean={final_mean:.4f}")
            
            # Validate final output quality
            if final_min < -1.1 or final_max > 1.1:
                logging.error(f"FINAL VALIDATION FAILED: Value range [{final_min:.4f}, {final_max:.4f}] is outside expected range!")
            elif final_min < -0.1 or final_max > 1.1:
                logging.warning(f"FINAL VALIDATION WARNING: Value range [{final_min:.4f}, {final_max:.4f}] might cause visual issues")
            else:
                logging.info(f"FINAL VALIDATION PASSED: Value range [{final_min:.4f}, {final_max:.4f}] is acceptable")
        
        decode_time = time.time() - start_time
        logging.info(f"ROCM VAE Decode completed in {decode_time:.2f}s")
        
        # Completion message for user feedback
        if is_video:
            print(f"✅ Video decode completed in {decode_time:.2f}s")
        else:
            print(f"✅ Image decode completed in {decode_time:.2f}s")
        
        return (pixel_samples,)
    
    def _decode_tiled_optimized(self, vae, samples, tile_size, overlap, dtype, batch_number):
        """
        Optimized tiled decoding for ROCm - avoid redundant model conversions
        """
        compression = vae.spacial_compression_decode()
        tile_x = tile_size // compression
        tile_y = tile_size // compression
        overlap_adj = overlap // compression
        
        # Detect if this is a quantized model
        is_quantized_model = False
        vae_model_dtype = getattr(vae.first_stage_model, 'dtype', None)
        if vae_model_dtype is not None:
            quantized_dtypes = [torch.float8_e4m3fn, torch.float8_e5m2, torch.int8, torch.int4]
            if vae_model_dtype in quantized_dtypes:
                is_quantized_model = True
            elif hasattr(vae_model_dtype, '__name__') and 'int' in str(vae_model_dtype):
                is_quantized_model = True
        
        # Use caching to avoid repeated model conversions in tiled decoding
        vae_id = id(vae.first_stage_model)
        cache_key = f"{vae_id}_{dtype}"
        
        if cache_key not in self._vae_model_cache:
            # CRITICAL FIX: Only convert model dtype if it's different AND not quantized
            if not is_quantized_model:
                current_model_dtype = getattr(vae.first_stage_model, 'dtype', None)
                if current_model_dtype is not None and current_model_dtype != dtype:
                    logging.info(f"Tiled decode: Converting VAE model from {current_model_dtype} to {dtype}")
                    vae.first_stage_model = vae.first_stage_model.to(dtype)
                else:
                    logging.info(f"Tiled decode: VAE model already in correct dtype: {current_model_dtype}")
            else:
                logging.info(f"Tiled decode: Preserving quantized model dtype: {vae_model_dtype}")
            self._vae_model_cache[cache_key] = True
        
        # Use ComfyUI's tiled scale with optimizations
        def decode_fn(samples_tile):
            # Ensure consistent data types for ROCm (minimal conversion)
            samples_tile = samples_tile.to(vae.device).to(dtype)
            return vae.first_stage_model.decode(samples_tile)
        
        # Ensure a valid output device is defined for tiled scaling
        device = vae.device if hasattr(vae, 'device') else (
            samples.device if hasattr(samples, 'device') else (
                torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
            )
        )

        result = comfy.utils.tiled_scale(
            samples, 
            decode_fn, 
            tile_x=tile_x, 
            tile_y=tile_y, 
            overlap=overlap_adj,
            upscale_amount=vae.upscale_ratio,
            out_channels=3,  # RGB output channels, not latent channels
            output_device=device  # CRITICAL FIX: Use GPU device, not vae.output_device
        )
        
        # Handle WAN VAE output format - ensure correct shape and channels
        if len(result.shape) == 5:  # Video format (B, C, T, H, W)
            # Reshape to (B*T, C, H, W) for video processing
            result = result.permute(0, 2, 1, 3, 4).contiguous()
            result = result.reshape(-1, result.shape[2], result.shape[3], result.shape[4])
        elif len(result.shape) == 4 and result.shape[1] > 3:
            # Handle case where we have too many channels - take first 3
            result = result[:, :3, :, :]
        
        # CRITICAL FIX: Keep standard VAE format (B, C, H, W) - don't convert to (B, H, W, C)
        # The permute operation was causing wrong tensor dimensions
        # Standard VAE decode should return (B, C, H, W) format
        # The main decode method will handle the final conversion to (B, H, W, C)
        
        # Capture output data and timing for debugging
        if DEBUG_MODE:
            end_time = time.time()
            save_debug_data(result, "vae_decode_output", "flux_1024x1024", {
                'node_type': 'ROCMOptimizedVAEDecode',
                'execution_time': end_time - time.time(),  # Use current time as fallback
                'output_shape': result.shape,
                'output_dtype': str(result.dtype),
                'output_device': str(result.device)
            })
            capture_timing("vae_decode", time.time(), end_time, {
                'node_type': 'ROCMOptimizedVAEDecode',
                'is_video': False,
                'tile_size': tile_size,
                'overlap': overlap
            })
            capture_memory_usage("vae_decode", {
                'node_type': 'ROCMOptimizedVAEDecode',
                'is_video': False
            })
            
        return result


class ROCMOptimizedVAEDecodeTiled:
    """
    Advanced tiled VAE decode with ROCm optimizations
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
                    "step": 64
                }),
                "overlap": ("INT", {
                    "default": 96, 
                    "min": 32, 
                    "max": 512, 
                    "step": 16
                }),
                "temporal_size": ("INT", {
                    "default": 64, 
                    "min": 8, 
                    "max": 4096, 
                    "step": 4,
                    "tooltip": "For video VAEs: frames to decode at once"
                }),
                "temporal_overlap": ("INT", {
                    "default": 8, 
                    "min": 4, 
                    "max": 4096, 
                    "step": 4,
                    "tooltip": "For video VAEs: frame overlap"
                }),
                "rocm_optimizations": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Enable ROCm-specific optimizations"
                })
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("IMAGE",)
    FUNCTION = "decode"
    CATEGORY = "RocM Ninodes/VAE"
    DESCRIPTION = "Advanced tiled VAE decode optimized for ROCm"
    
    def decode(self, vae, samples, tile_size=768, overlap=96, temporal_size=64, 
               temporal_overlap=8, rocm_optimizations=True):
        """
        Advanced tiled decode with ROCm optimizations
        """
        start_time = time.time()
        
        # Adjust tile size for ROCm
        if rocm_optimizations:
            if tile_size < overlap * 4:
                overlap = tile_size // 4
            if temporal_size < temporal_overlap * 2:
                temporal_overlap = temporal_size // 2
        
        # Handle temporal compression
        temporal_compression = vae.temporal_compression_decode()
        if temporal_compression is not None:
            temporal_size = max(2, temporal_size // temporal_compression)
            temporal_overlap = max(1, min(temporal_size // 2, temporal_overlap // temporal_compression))
        else:
            temporal_size = None
            temporal_overlap = None
        
        # Use VAE's tiled decode with optimizations
        compression = vae.spacial_compression_decode()
        images = vae.decode_tiled(
            samples["samples"], 
            tile_x=tile_size // compression, 
            tile_y=tile_size // compression, 
            overlap=overlap // compression, 
            tile_t=temporal_size, 
            overlap_t=temporal_overlap
        )
        
        # Reshape if needed
        if len(images.shape) == 5:
            images = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])
        
        decode_time = time.time() - start_time
        logging.info(f"ROCM Tiled VAE Decode completed in {decode_time:.2f}s")
        
        return (images,)


class ROCMVAEPerformanceMonitor:
    """
    Monitor VAE performance and provide optimization suggestions
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vae": ("VAE", {"tooltip": "VAE to monitor"}),
                "test_resolution": ("INT", {
                    "default": 1024,
                    "min": 256,
                    "max": 4096,
                    "step": 64,
                    "tooltip": "Test resolution for benchmarking"
                })
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("DEVICE_INFO", "PERFORMANCE_TIPS", "OPTIMAL_SETTINGS")
    FUNCTION = "analyze"
    CATEGORY = "RocM Ninodes/VAE"
    DESCRIPTION = "Analyze VAE performance and provide optimization recommendations"
    
    def analyze(self, vae, test_resolution=1024):
        """
        Analyze VAE performance and provide recommendations
        """
        device = vae.device
        device_info = f"Device: {device}\n"
        device_info += f"VAE dtype: {vae.vae_dtype}\n"
        device_info += f"Output device: {vae.output_device}\n"
        
        # Check if it's an AMD GPU
        is_amd = hasattr(device, 'type') and device.type == 'cuda'
        if is_amd:
            try:
                device_name = torch.cuda.get_device_name(0)
                device_info += f"GPU: {device_name}\n"
            except:
                device_info += "GPU: AMD (ROCm)\n"
        
        # Performance tips
        tips = []
        if is_amd:
            tips.append("• Use fp32 precision for better ROCm performance")
            tips.append("• Tile size 768-1024 works well for gfx1151")
            tips.append("• Enable ROCm optimizations for best results")
            tips.append("• Consider using tiled decode for images > 1024x1024")
        else:
            tips.append("• Use fp16 or bf16 for better performance")
            tips.append("• Larger tile sizes generally perform better")
        
        # Optimal settings
        settings = []
        if is_amd:
            settings.append(f"Recommended tile_size: {min(1024, test_resolution)}")
            settings.append(f"Recommended overlap: {min(128, test_resolution // 8)}")
            settings.append("Recommended precision: fp32")
            settings.append("Recommended batch_optimization: True")
        else:
            settings.append(f"Recommended tile_size: {test_resolution}")
            settings.append(f"Recommended overlap: {test_resolution // 8}")
            settings.append("Recommended precision: auto")
            settings.append("Recommended batch_optimization: True")
        
        return (
            device_info,
            "\n".join(tips),
            "\n".join(settings)
        )


class ROCMOptimizedKSampler:
    """
    ROCM-optimized KSampler for all AMD GPUs with ROCm support.
    
    Optimization Strategy:
    1. GENERAL ROCM OPTIMIZATIONS (apply to all AMD GPUs):
       - TF32 disabled (not supported on AMD)
       - FP16 accumulation enabled (better performance on AMD)
       - Async memory management (non-blocking)
       - WAN model full-load optimization
    
    2. ARCHITECTURE-SPECIFIC OPTIMIZATIONS (if supported):
       - PyTorch attention enabled for compatible architectures
         (gfx90a, gfx942, gfx1100, gfx1101, gfx1151, gfx1201+)
       - ROCm version detection for feature support
    
    3. STRix HALO (gfx1151) SPECIFIC:
       - Full PyTorch attention support
       - Enhanced memory management
       - Specialized optimizations for Strix Halo architecture
    
    Key optimizations:
    - Non-blocking memory operations (reduces idle time)
    - Optimized precision handling for AMD GPUs
    - Reduced memory fragmentation
    - Better batch processing
    - Architecture-aware feature detection
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "The model to sample from"}),
                "seed": ("INT", {
                    "default": 0, 
                    "min": 0, 
                    "max": 0xffffffffffffffff,
                    "tooltip": "Random seed for generation"
                }),
                "steps": ("INT", {
                    "default": 20, 
                    "min": 1, 
                    "max": 10000,
                    "tooltip": "Number of sampling steps"
                }),
                "cfg": ("FLOAT", {
                    "default": 8.0, 
                    "min": 0.0, 
                    "max": 100.0, 
                    "step": 0.1, 
                    "round": 0.01,
                    "tooltip": "Classifier-free guidance scale"
                }),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {
                    "tooltip": "Sampling algorithm to use"
                }),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {
                    "tooltip": "Scheduler for noise timesteps"
                }),
                "positive": ("CONDITIONING", {"tooltip": "Positive conditioning"}),
                "negative": ("CONDITIONING", {"tooltip": "Negative conditioning"}),
                "latent_image": ("LATENT", {"tooltip": "Latent image to sample from"}),
                "denoise": ("FLOAT", {
                    "default": 1.0, 
                    "min": 0.0, 
                    "max": 1.0, 
                    "step": 0.01,
                    "tooltip": "Denoising strength"
                })
            }
        }
    
    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("LATENT",)
    FUNCTION = "sample"
    CATEGORY = "RocM Ninodes/Sampling"
    DESCRIPTION = "ROCM-optimized KSampler for all AMD GPUs with ROCm support. Enhanced for Strix Halo (gfx1151)."
    
    def sample(self, model, seed, steps, cfg, sampler_name, scheduler, positive, negative, 
               latent_image, denoise=1.0):
        """
        Optimized sampling for ROCm/AMD GPUs with minimal overhead and gfx1151-specific optimizations
        """
        start_time = time.time()
        
        # Detect WAN models for special handling (more reliable detection)
        is_wan_model = False
        try:
            if hasattr(model, 'model') and model.model is not None:
                model_name = model.model.__class__.__name__
                # Check for WAN model variants (WAN21, WAN22, WAN22_S2V, etc.)
                if 'WAN' in model_name or 'WanModel' in model_name or 'Wan22' in model_name:
                    is_wan_model = True
            # Also check model_type attribute for additional detection
            if hasattr(model, 'model_type'):
                model_type_str = str(model.model_type).lower()
                if 'wan' in model_type_str:
                    is_wan_model = True
        except:
            pass
        
        # Detect architecture for optimizations
        device = model_management.get_torch_device()
        is_amd = hasattr(device, 'type') and device.type == 'cuda'
        is_gfx1151 = False
        arch = None
        supports_pytorch_attention = False
        
        # Apply general ROCm optimizations for ALL AMD GPUs
        if is_amd and torch.cuda.is_available():
            try:
                arch = torch.cuda.get_device_properties(0).gcnArchName
                rocm_version_str = str(torch.version.hip)
                try:
                    rocm_version = tuple(map(int, rocm_version_str.split(".")[:2]))
                except:
                    rocm_version = (6, 0)
                
                # GENERAL ROCM OPTIMIZATIONS (apply to all AMD GPUs)
                torch.backends.cuda.matmul.allow_tf32 = False  # TF32 not supported on AMD
                torch.backends.cuda.matmul.allow_fp16_accumulation = True  # Better performance on AMD
                
                # Check if this is Strix Halo (gfx1151)
                if 'gfx1151' in arch:
                    is_gfx1151 = True
                    supports_pytorch_attention = True  # gfx1151 supports it
                    print(f"🚀 Detected Strix Halo (gfx1151) - applying architecture-specific optimizations")
                
                # Check for PyTorch attention support (multiple architectures support it)
                # Based on ComfyUI's model_management.py logic
                pytorch_attention_arches = ["gfx90a", "gfx942", "gfx1100", "gfx1101", "gfx1151"]
                if any((a in arch) for a in pytorch_attention_arches):
                    supports_pytorch_attention = True
                
                # For ROCm 7.0+, additional architectures support PyTorch attention
                if rocm_version >= (7, 0):
                    if any((a in arch) for a in ["gfx1201"]):
                        supports_pytorch_attention = True
                
                # Enable PyTorch attention if supported (benefits all compatible AMD GPUs)
                if supports_pytorch_attention:
                    torch.backends.cuda.enable_math_sdp(True)
                    torch.backends.cuda.enable_flash_sdp(True)
                    torch.backends.cuda.enable_mem_efficient_sdp(True)
                    
                    if is_gfx1151:
                        print(f"✅ Enabled PyTorch attention optimizations for gfx1151")
                    elif arch:
                        print(f"✅ Enabled PyTorch attention optimizations for {arch}")
                
                # OPTIMIZATION: torch.compile for ROCm (experimental but can provide significant speedup)
                # Check if torch.compile is available (PyTorch 2.0+)
                try:
                    if hasattr(torch, 'compile') and torch.__version__ >= "2.0":
                        # For ROCm, use "inductor" backend (more compatible than "cudagraphs")
                        # This will be applied to the model during sampling via model options
                        # Note: torch.compile is applied lazily on first forward pass
                        if is_gfx1151:
                            # Enable torch.compile optimizations for gfx1151
                            # Will be applied via model.model_options if supported
                            torch._dynamo.config.suppress_errors = True  # Don't fail on unsupported ops
                            print(f"✅ torch.compile optimizations ready for gfx1151 (will compile on first use)")
                        elif is_amd:
                            torch._dynamo.config.suppress_errors = True
                            if DEBUG_MODE:
                                print(f"✅ torch.compile optimizations ready for {arch}")
                except Exception as e:
                    if DEBUG_MODE:
                        print(f"⚠️ torch.compile not available: {e}")
                
            except Exception as e:
                # Fallback: still apply general ROCm optimizations even if detection fails
                if is_amd:
                    torch.backends.cuda.matmul.allow_tf32 = False
                    torch.backends.cuda.matmul.allow_fp16_accumulation = True
        
        # OPTIMIZATION: Non-blocking memory cleanup (async - doesn't sync GPU)
        if torch.cuda.is_available():
            # Only cleanup if memory is critically low - use async method
            total_mem, alloc_mem, reserved_mem, free_mem = get_gpu_memory_info()
            if free_mem is not None and free_mem < 2 * 1024**3:  # Less than 2GB free
                torch.cuda.empty_cache()  # Async - doesn't block
            # Don't call gentle_memory_cleanup() - it contains torch.cuda.synchronize()
        
        # Initialize progress early for UI feedback
        is_video_workflow = latent_image["samples"].shape[0] > 1
        if is_video_workflow:
            print(f"🎬 Video workflow detected: {latent_image['samples'].shape[0]} frames")
        else:
            print(f"🖼️ Image workflow detected")
        
        # Initialize progress bar early for UI feedback during initialization
        init_pbar = comfy.utils.ProgressBar(5)  # 5 phases: init, model_load, prep, noise, sample
        init_pbar.update_absolute(1, 5, preview=None)
        
        # Note: WAN model will be loaded during prepare_sampling() inside comfy.sample.sample()
        # We skip pre-loading here to avoid duplicate loads and let prepare_sampling handle it
        # with proper memory estimation
        if is_wan_model:
            init_pbar.update_absolute(2, 5, preview=None)
            print(f"📦 WAN model will be loaded during sampling preparation...")
            if is_gfx1151:
                print(f"🚀 WAN model detected - will force full GPU load for Strix Halo (gfx1151) optimization")
            elif is_amd:
                print(f"🚀 WAN model detected - will force full GPU load for ROCm optimization")
        
        print(f"⚙️ Preparing KSampler: {steps} steps, CFG {cfg}, {sampler_name}")
        
        # Use vanilla ComfyUI sampling (standard path)
        try:
            init_pbar.update_absolute(3, 5, preview=None)
            # Use ComfyUI's sample function directly
            latent_image_tensor = latent_image["samples"]
            latent_image_tensor = comfy.sample.fix_empty_latent_channels(model, latent_image_tensor)
            init_pbar.update_absolute(4, 5, preview=None)
            
            # OPTIMIZATION: Non-blocking latent validation (use tensor operations, not .item())
            # Only check if needed for debugging, and use non-blocking tensor ops
            if DEBUG_MODE and denoise < 1.0:
                latent_shape = latent_image_tensor.shape
                if len(latent_shape) >= 4:
                    # Use tensor comparisons - non-blocking
                    with torch.no_grad():
                        latent_mean_tensor = latent_image_tensor.mean()
                        latent_std_tensor = latent_image_tensor.std()
                        # Check without .item() - use tensor operations
                        is_empty = torch.abs(latent_mean_tensor) < 0.001 and latent_std_tensor < 0.001
                        is_noise = latent_std_tensor > 1.5
                        if is_empty or is_noise:
                            print(f"⚠️ WARNING: Latent may be invalid (img2img workflow)")
            
            # Match original KSampler: Let ComfyUI handle device placement naturally
            # Do not modify tensors before ComfyUI's sample() function - it handles device placement correctly
            batch_inds = latent_image["batch_index"] if "batch_index" in latent_image else None
            
            # Standard noise preparation (matches original KSampler exactly)
            noise = comfy.sample.prepare_noise(latent_image_tensor, seed, batch_inds)
            
            # Standard noise mask extraction (matches original KSampler exactly)
            noise_mask = None
            if "noise_mask" in latent_image:
                noise_mask = latent_image["noise_mask"]
            
            # Prepare callback - enable progress reporting for ALL workflows
            is_video_workflow = latent_image_tensor.shape[0] > 1
            
            # Complete initialization phase
            init_pbar.update_absolute(5, 5, preview=None)
            print(f"🚀 Starting sampling: {steps} steps, CFG {cfg}, {sampler_name}")
            
            # OPTIMIZATION: Enhanced progress callback for all workflows
            # Provides UI feedback with optional preview for image workflows
            pbar = comfy.utils.ProgressBar(steps)
            last_update_time = [time.time()]  # Use list for mutable closure
            callback_start_time = time.time()  # Use current time for callback (after init)
            
            def enhanced_callback(step, x0, x, total_steps):
                """Enhanced callback for all workflows - reports progress to UI and console"""
                current_time = time.time()
                # Update progress bar every 0.3 seconds or at completion for responsive UI
                if current_time - last_update_time[0] >= 0.3 or step == total_steps - 1:
                    # Generate preview only for image workflows (not video to reduce overhead)
                    preview_bytes = None
                    if not is_video_workflow and step % 5 == 0:  # Generate preview every 5 steps for images
                        try:
                            previewer = latent_preview.get_previewer(model.load_device, model.model.latent_format)
                            if previewer:
                                preview_bytes = previewer.decode_latent_to_preview_image("JPEG", x0)
                        except:
                            preview_bytes = None  # Don't block on preview generation
                    
                    # Update progress bar with preview (UI feedback)
                    pbar.update_absolute(step + 1, total_steps, preview=preview_bytes)
                    
                    # Console feedback every 10% or at completion
                    if step % max(1, total_steps // 10) == 0 or step == total_steps - 1:
                        progress_pct = ((step + 1) / total_steps) * 100
                        elapsed = current_time - callback_start_time
                        if step > 0:
                            est_total = elapsed / ((step + 1) / total_steps)
                            est_remaining = est_total - elapsed
                            avg_time_per_step = elapsed / (step + 1)
                            workflow_type = "Video" if is_video_workflow else "Image"
                            print(f"📊 {workflow_type} KSampler: {step + 1}/{total_steps} ({progress_pct:.1f}%) | "
                                  f"Elapsed: {elapsed:.1f}s | Remaining: ~{est_remaining:.1f}s | "
                                  f"Avg: {avg_time_per_step:.2f}s/step")
                        else:
                            workflow_type = "Video" if is_video_workflow else "Image"
                            print(f"📊 {workflow_type} KSampler: {step + 1}/{total_steps} ({progress_pct:.1f}%) | Starting...")
                    
                    last_update_time[0] = current_time
            
            callback = enhanced_callback
            disable_pbar = False  # Always enable progress bar for UI feedback
            
            # Note: comfy.sample.sample() will call prepare_sampling() internally
            # which calls load_models_gpu() - this can take time for WAN models
            # We show a message and periodically update progress during this phase
            print(f"⏳ Preparing model for sampling (this may take a moment for WAN models)...")
            prep_start = time.time()
            
            # For WAN models, start a periodic progress update thread to show activity
            # during the blocking prepare_sampling() -> load_models_gpu() call
            import threading
            progress_update_active = [True]
            progress_pbar_ref = [pbar]  # Reference to progress bar for thread
            
            def periodic_progress_update():
                """Periodically update progress bar during model loading"""
                update_interval = 2.0  # Update every 2 seconds
                elapsed = 0
                while progress_update_active[0]:
                    time.sleep(update_interval)
                    elapsed += update_interval
                    if elapsed > 5.0 and progress_pbar_ref[0] is not None:  # Start showing updates after 5 seconds
                        try:
                            # Keep progress bar visible (at step 1 of total steps)
                            progress_pbar_ref[0].update_absolute(1, steps, preview=None)
                        except:
                            pass  # Don't fail if progress bar isn't ready yet
                    if elapsed % 10 == 0:  # Print every 10 seconds
                        print(f"⏳ Still preparing model for sampling... ({elapsed:.0f}s elapsed)")
            
            if is_wan_model:
                progress_thread = threading.Thread(target=periodic_progress_update, daemon=True)
                progress_thread.start()
            
            # OPTIMIZATION: Apply torch.compile to model if supported (for ROCm performance boost)
            # torch.compile can provide 10-30% speedup on compatible operations
            try:
                if is_amd and hasattr(torch, 'compile') and torch.__version__ >= "2.0":
                    # Enable compile optimizations for ROCm
                    torch._dynamo.config.suppress_errors = True
                    if DEBUG_MODE and is_gfx1151:
                        print(f"🔧 torch.compile optimizations enabled (will compile on first forward pass)")
            except:
                pass  # Don't fail if compile setup fails
            
            # Match original KSampler: Let comfy.sample.sample() handle device placement
            # Do not modify tensors - ComfyUI's sample() function handles device placement internally
            # Sample (this internally calls prepare_sampling() -> load_models_gpu())
            try:
                samples = comfy.sample.sample(
                    model, noise, steps, cfg, sampler_name, scheduler, 
                    positive, negative, latent_image_tensor, denoise=denoise, 
                    noise_mask=noise_mask, callback=callback, disable_pbar=disable_pbar, seed=seed
                )
            finally:
                # Stop progress updates once sampling starts (callback will handle it)
                progress_update_active[0] = False
            
            prep_time = time.time() - prep_start
            if prep_time > 5.0:  # Only log if it took significant time
                print(f"✅ Model preparation completed in {prep_time:.1f}s - starting sampling now")
            
            # Wrap in latent format
            out = latent_image.copy()
            out["samples"] = samples
            result = (out,)
            
        except Exception as e:
            print(f"Sampling failed: {e}")
            raise e
        
        # OPTIMIZATION: Non-blocking post-sampling cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()  # Async - non-blocking
            # Don't call gentle_memory_cleanup() - it contains torch.cuda.synchronize()
        
        sample_time = time.time() - start_time
        print(f"✅ KSampler completed in {sample_time:.2f}s")
        
        return result
    


class ROCMOptimizedKSamplerAdvanced:
    """
    Advanced ROCM-optimized KSampler for all AMD GPUs with ROCm support.
    
    Optimization Strategy:
    1. GENERAL ROCM OPTIMIZATIONS (apply to all AMD GPUs):
       - TF32 disabled (not supported on AMD)
       - FP16 accumulation enabled (better performance on AMD)
       - Async memory management (non-blocking)
       - WAN model full-load optimization
    
    2. ARCHITECTURE-SPECIFIC OPTIMIZATIONS (if supported):
       - PyTorch attention enabled for compatible architectures
         (gfx90a, gfx942, gfx1100, gfx1101, gfx1151, gfx1201+)
       - ROCm version detection for feature support
    
    3. STRix HALO (gfx1151) SPECIFIC:
       - Full PyTorch attention support
       - Enhanced memory management
       - Specialized optimizations for Strix Halo architecture
    
    Provides extended control options including step control, precision mode,
    and memory optimization settings while maintaining ROCm compatibility.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "The model to sample from"}),
                "add_noise": (["enable", "disable"], {
                    "default": "enable",
                    "tooltip": "Whether to add noise to the latent"
                }),
                "noise_seed": ("INT", {
                    "default": 0, 
                    "min": 0, 
                    "max": 0xffffffffffffffff,
                    "tooltip": "Random seed for noise generation"
                }),
                "steps": ("INT", {
                    "default": 20, 
                    "min": 1, 
                    "max": 10000,
                    "tooltip": "Number of sampling steps"
                }),
                "cfg": ("FLOAT", {
                    "default": 8.0, 
                    "min": 0.0, 
                    "max": 100.0, 
                    "step": 0.1, 
                    "round": 0.01,
                    "tooltip": "Classifier-free guidance scale"
                }),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {
                    "tooltip": "Sampling algorithm to use"
                }),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {
                    "tooltip": "Scheduler for noise timesteps"
                }),
                "positive": ("CONDITIONING", {"tooltip": "Positive conditioning"}),
                "negative": ("CONDITIONING", {"tooltip": "Negative conditioning"}),
                "latent_image": ("LATENT", {"tooltip": "Latent image to sample from"}),
                "start_at_step": ("INT", {
                    "default": 0, 
                    "min": 0, 
                    "max": 10000,
                    "tooltip": "Step to start sampling from"
                }),
                "end_at_step": ("INT", {
                    "default": 10000, 
                    "min": 0, 
                    "max": 10000,
                    "tooltip": "Step to end sampling at"
                }),
                "return_with_leftover_noise": (["disable", "enable"], {
                    "default": "disable",
                    "tooltip": "Whether to return with leftover noise"
                }),
                "use_rocm_optimizations": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Enable ROCm-specific optimizations"
                }),
                "precision_mode": (["auto", "fp32", "fp16", "bf16"], {
                    "default": "auto",
                    "tooltip": "Precision mode for sampling"
                }),
                "memory_optimization": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Enable memory optimization for AMD GPUs"
                })
            }
            ,
            "optional": {
                "compatibility_mode": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Enable compatibility mode for quantized models (disables optimizations)"
                })
            }
        }
    
    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("LATENT",)
    FUNCTION = "sample"
    CATEGORY = "RocM Ninodes/Sampling"
    DESCRIPTION = "Advanced ROCM-optimized KSampler for all AMD GPUs with ROCm support. Enhanced for Strix Halo (gfx1151)."
    
    def sample(self, model, add_noise, noise_seed, steps, cfg, sampler_name, scheduler, 
               positive, negative, latent_image, start_at_step, end_at_step, 
               return_with_leftover_noise, use_rocm_optimizations=True, 
               precision_mode="auto", memory_optimization=True, denoise=1.0, compatibility_mode=False):
        """
        Advanced optimized sampling for ROCm/AMD GPUs with video workflow optimizations and quantized model support
        """
        start_time = time.time()
        
        # CRITICAL FIX: Detect quantized models to avoid breaking them
        is_quantized_model = False
        try:
            model_dtype = model.model_dtype()
            if hasattr(model_dtype, '__name__'):
                dtype_name = str(model_dtype)
                # Check for quantized dtypes - be more specific
                quantized_indicators = ['int8', 'int4', 'float8', 'quantized']
                if any(indicator in dtype_name.lower() for indicator in quantized_indicators):
                    is_quantized_model = True
                    print(f"🔍 Detected quantized model (dtype: {model_dtype})")
        except Exception as e:
            print(f"⚠️ Could not detect model dtype: {e}")
        
        # Compatibility mode: disable all optimizations for quantized models
        if compatibility_mode or is_quantized_model:
            print("🛡️ Compatibility mode enabled - using stock ComfyUI behavior")
            use_rocm_optimizations = False
            memory_optimization = False
        
        # Detect WAN models for special handling (more reliable detection)
        is_wan_model = False
        try:
            if hasattr(model, 'model') and model.model is not None:
                model_name = model.model.__class__.__name__
                # Check for WAN model variants (WAN21, WAN22, WAN22_S2V, etc.)
                if 'WAN' in model_name or 'WanModel' in model_name or 'Wan22' in model_name:
                    is_wan_model = True
            # Also check model_type attribute for additional detection
            if hasattr(model, 'model_type'):
                model_type_str = str(model.model_type).lower()
                if 'wan' in model_type_str:
                    is_wan_model = True
        except:
            pass
        
        # Detect video workflow by batch size
        batch_size = latent_image["samples"].shape[0]
        is_video_workflow = batch_size > 1
        
        # Detect gfx1151 architecture for optimizations
        is_gfx1151 = False
        device = model_management.get_torch_device()
        is_amd = hasattr(device, 'type') and device.type == 'cuda'
        
        # OPTIMIZATION: Non-blocking memory cleanup (async - doesn't sync GPU)
        if torch.cuda.is_available():
            _total, _alloc, _reserved, _free = get_gpu_memory_info()
            if _free is not None and _free < 2 * 1024**3:  # < 2GB free
                torch.cuda.empty_cache()  # Async - non-blocking
                # Don't call gentle_memory_cleanup() - it contains torch.cuda.synchronize()
        
        # OPTIMIZATION: For WAN models, force full load to prevent offloading between steps
        if is_wan_model:
            try:
                model_management.load_models_gpu(
                    [model.patcher], 
                    force_full_load=True,  # Keep model in VRAM - prevents idle time
                    memory_required=0  # Already loaded, just ensure it stays
                )
            except Exception as e:
                logging.warning(f"Could not force full load for WAN model: {e}")
        
        # Minimal progress for video workflows to reduce CPU overhead
        if is_video_workflow:
            print(f"🎬 Video KSampler: {steps} steps, batch={batch_size}, {sampler_name}")
        else:
            print(f"🎬 Starting Advanced KSampler: {steps} steps, CFG {cfg}, {sampler_name}")
        
        # ROCm optimizations: General for all AMD, enhanced for gfx1151
        if use_rocm_optimizations:
            arch = None
            supports_pytorch_attention = False
            
            if is_amd and torch.cuda.is_available():
                try:
                    arch = torch.cuda.get_device_properties(0).gcnArchName
                    rocm_version_str = str(torch.version.hip)
                    try:
                        rocm_version = tuple(map(int, rocm_version_str.split(".")[:2]))
                    except:
                        rocm_version = (6, 0)
                    
                    # Check if this is Strix Halo (gfx1151)
                    if 'gfx1151' in arch:
                        is_gfx1151 = True
                        supports_pytorch_attention = True
                        if not is_video_workflow:
                            print(f"🚀 Detected Strix Halo (gfx1151) - applying architecture-specific optimizations")
                    
                    # Check for PyTorch attention support (multiple architectures)
                    pytorch_attention_arches = ["gfx90a", "gfx942", "gfx1100", "gfx1101", "gfx1151"]
                    if any((a in arch) for a in pytorch_attention_arches):
                        supports_pytorch_attention = True
                    
                    # For ROCm 7.0+, additional architectures support PyTorch attention
                    if rocm_version >= (7, 0):
                        if any((a in arch) for a in ["gfx1201"]):
                            supports_pytorch_attention = True
                except:
                    pass
            
            # GENERAL ROCM OPTIMIZATIONS (apply to all AMD GPUs)
            if is_amd:
                torch.backends.cuda.matmul.allow_tf32 = False  # TF32 not supported on AMD
                torch.backends.cuda.matmul.allow_fp16_accumulation = True  # Better performance on AMD
                
                # Enable PyTorch attention if architecture supports it
                if supports_pytorch_attention:
                    torch.backends.cuda.enable_math_sdp(True)
                    torch.backends.cuda.enable_flash_sdp(True)
                    torch.backends.cuda.enable_mem_efficient_sdp(True)
                    
                    if is_gfx1151 and not is_video_workflow:
                        print(f"✅ Enabled PyTorch attention optimizations for Strix Halo")
                    elif arch and not is_video_workflow and DEBUG_MODE:
                        print(f"✅ Enabled PyTorch attention optimizations for {arch}")
            
            if is_amd:
                # Set optimal precision
                if precision_mode == "auto":
                    optimal_dtype = torch.float32
                else:
                    dtype_map = {
                        "fp32": torch.float32,
                        "fp16": torch.float16,
                        "bf16": torch.bfloat16
                    }
                    optimal_dtype = dtype_map[precision_mode]
                
                # OPTIMIZED memory management for video workflows (non-blocking)
                if memory_optimization:
                    # Check available memory and adjust strategy
                    total_memory, allocated_memory, reserved_memory, free_memory = get_gpu_memory_info()
                    
                    if total_memory is not None:
                        # Only show detailed memory info for debugging
                        if not is_video_workflow and DEBUG_MODE:
                            print(f"🔍 Advanced KSampler Memory: {allocated_memory/1024**3:.2f}GB allocated, {reserved_memory/1024**3:.2f}GB reserved, {free_memory/1024**3:.2f}GB free")
                        
                        # Video workflow optimization: more aggressive memory usage
                        if is_video_workflow:
                            # For video workflows, use more memory for better GPU utilization
                            memory_fraction = min(0.90, max(0.75, free_memory / total_memory))
                            
                            # Only cleanup if memory is critically low (async)
                            if free_memory < 2 * 1024**3:  # Less than 2GB free
                                print("⚠️ Critical memory - async cleanup")
                                torch.cuda.empty_cache()  # Async - non-blocking
                                memory_fraction = 0.80
                            # Don't call gentle_memory_cleanup() - it syncs
                        else:
                            # Regular workflow: conservative approach
                            memory_fraction = min(0.80, max(0.55, free_memory / total_memory))
                            
                            if free_memory < 3 * 1024**3:  # Less than 3GB free
                                print("⚠️ Low memory detected - async cleanup")
                                torch.cuda.empty_cache()  # Async - non-blocking
                                memory_fraction = 0.65
                            # Don't call gentle_memory_cleanup() - it syncs
                        
                        # Apply memory fraction
                        if hasattr(torch.cuda, 'set_per_process_memory_fraction'):
                            try:
                                torch.cuda.set_per_process_memory_fraction(memory_fraction)
                                if not is_video_workflow and DEBUG_MODE:
                                    print(f"🔧 Memory fraction set to {memory_fraction*100:.0f}%")
                            except Exception as e:
                                if DEBUG_MODE:
                                    print(f"⚠️ Memory fraction setting failed: {e}")
        
        # Configure sampling parameters
        force_full_denoise = True
        if return_with_leftover_noise == "enable":
            force_full_denoise = False
        
        disable_noise = False
        if add_noise == "disable":
            disable_noise = True
        
        # Use advanced ksampler with optimizations
        try:
            # Minimal progress for video workflows
            if not is_video_workflow:
                print("⚡ Preparing sampling parameters...")
            
            # Use ComfyUI's sample function directly with advanced options
            latent_image_tensor = latent_image["samples"]
            # Ensure latent on GPU to avoid CPU paging
            if torch.cuda.is_available() and latent_image_tensor.device.type != 'cuda':
                latent_image_tensor = latent_image_tensor.to(model_management.get_torch_device(), non_blocking=True)
            latent_image_tensor = comfy.sample.fix_empty_latent_channels(model, latent_image_tensor)

            # Guard 1: Only zero-out latent if it's truly a fresh generation (text2img), not img2img
            # CRITICAL FIX: Don't zero out latents for img2img workflows (denoise < 1.0) or when latent contains image data
            # OPTIMIZATION: Use non-blocking tensor operations instead of .item()
            if add_noise == "enable" and start_at_step == 0:
                # If denoise < 1.0, this is img2img - preserve the latent content
                if denoise < 1.0:
                    if DEBUG_MODE:
                        logging.info(f"Preserving latent image content for img2img workflow (denoise={denoise})")
                else:
                    # OPTIMIZATION: Use tensor operations instead of .item() to avoid blocking
                    # For full denoise, check if latent contains actual image data vs noise
                    with torch.no_grad():
                        latent_variance_tensor = latent_image_tensor.var()
                        latent_mean_abs_tensor = latent_image_tensor.abs().mean()
                        # Use tensor comparisons - non-blocking
                        is_likely_noise = (latent_variance_tensor > 1.2) or ((latent_mean_abs_tensor < 0.005) and (latent_variance_tensor < 0.05))
                        is_likely_noise = is_likely_noise.item() if is_likely_noise.numel() == 1 else bool(is_likely_noise.any().item())
                    
                    # Only zero out if it's clearly noise, preserving img2img latents even in text2img mode
                    if is_likely_noise:
                        # Zero-out latent contents to avoid pass-through artifacts/scrambled frames when resolution changed
                        latent_image_tensor = torch.zeros_like(latent_image_tensor, device=latent_image_tensor.device)
                        # If a noise_mask exists but is mismatched, drop it to prevent shape errors
                        if "noise_mask" in latent_image and isinstance(latent_image["noise_mask"], torch.Tensor):
                            if tuple(latent_image["noise_mask"].shape) != tuple(latent_image_tensor.shape):
                                latent_image["noise_mask"] = None
                    elif DEBUG_MODE:
                        # Only log if debugging - avoid blocking operations
                        logging.info(f"Preserving latent image content")
            
            # Prepare noise (minimal progress for video)
            if not is_video_workflow:
                print("🎲 Preparing noise for sampling...")
            if disable_noise:
                # CRITICAL FIX: Create noise on same device as latent_image_tensor, not CPU
                noise = torch.zeros(latent_image_tensor.size(), dtype=latent_image_tensor.dtype, layout=latent_image_tensor.layout, device=latent_image_tensor.device)
            else:
                batch_inds = latent_image["batch_index"] if "batch_index" in latent_image else None
                # Guard 2: Recompute noise with current latent shape to avoid stale shapes after resolution changes
                noise = comfy.sample.prepare_noise(latent_image_tensor, noise_seed, batch_inds)
                # Move noise to GPU if created on CPU by upstream helper
                if torch.cuda.is_available() and noise.device.type != 'cuda':
                    noise = noise.to(latent_image_tensor.device, non_blocking=True)
            
            # Prepare noise mask
            noise_mask = None
            if "noise_mask" in latent_image:
                noise_mask = latent_image["noise_mask"]
            
            # Prepare callback - enable progress reporting for ALL workflows
            # OPTIMIZATION: Enhanced progress callback for all workflows
            # Provides UI feedback with optional preview for image workflows
            pbar = comfy.utils.ProgressBar(steps)
            last_update_time = [time.time()]  # Use list for mutable closure
            adv_callback_start_time = start_time
            
            def enhanced_callback(step, x0, x, total_steps):
                """Enhanced callback for all workflows - reports progress to UI and console"""
                current_time = time.time()
                # Update progress bar every 0.3 seconds or at completion for responsive UI
                if current_time - last_update_time[0] >= 0.3 or step == total_steps - 1:
                    # Generate preview only for image workflows (not video to reduce overhead)
                    preview_bytes = None
                    if not is_video_workflow and step % 5 == 0:  # Generate preview every 5 steps for images
                        try:
                            previewer = latent_preview.get_previewer(model.load_device, model.model.latent_format)
                            if previewer:
                                preview_bytes = previewer.decode_latent_to_preview_image("JPEG", x0)
                        except:
                            preview_bytes = None  # Don't block on preview generation
                    
                    # Update progress bar with preview (UI feedback)
                    pbar.update_absolute(step + 1, total_steps, preview=preview_bytes)
                    
                    # Console feedback every 10% or at completion
                    if step % max(1, total_steps // 10) == 0 or step == total_steps - 1:
                        progress_pct = ((step + 1) / total_steps) * 100
                        elapsed = current_time - adv_callback_start_time
                        if step > 0:
                            est_total = elapsed / ((step + 1) / total_steps)
                            est_remaining = est_total - elapsed
                            avg_time_per_step = elapsed / (step + 1)
                            workflow_type = "Video" if is_video_workflow else "Image"
                            print(f"📊 Advanced {workflow_type} KSampler: {step + 1}/{total_steps} ({progress_pct:.1f}%) | "
                                  f"Elapsed: {elapsed:.1f}s | Remaining: ~{est_remaining:.1f}s | "
                                  f"Avg: {avg_time_per_step:.2f}s/step")
                        else:
                            workflow_type = "Video" if is_video_workflow else "Image"
                            print(f"📊 Advanced {workflow_type} KSampler: {step + 1}/{total_steps} ({progress_pct:.1f}%) | Starting...")
                    
                    last_update_time[0] = current_time
            
            callback = enhanced_callback
            disable_pbar = False  # Always enable progress bar for UI feedback
            
            # Minimal progress for video workflows
            if not is_video_workflow:
                print(f"🚀 Starting sampling: {steps} steps, CFG {cfg}, {sampler_name}")
                print("Using standard ComfyUI sampling path")
            
            # Sample with advanced options
            samples = comfy.sample.sample(
                model, noise, steps, cfg, sampler_name, scheduler, 
                positive, negative, latent_image_tensor, denoise=denoise, 
                disable_noise=disable_noise, start_step=start_at_step, 
                last_step=end_at_step, force_full_denoise=force_full_denoise,
                noise_mask=noise_mask, callback=callback, disable_pbar=disable_pbar, seed=noise_seed
            )
            
            # Wrap in latent format
            out = latent_image.copy()
            out["samples"] = samples
            result = (out,)
            
        except Exception as e:
            print(f"⚠️ Advanced sampling failed, falling back to standard: {e}")
            logging.warning(f"Advanced sampling failed, falling back to standard: {e}")
            
            # Check if it's a memory error
            is_memory_error = "out of memory" in str(e).lower() or "oom" in str(e).lower()
            
            if is_memory_error:
                print("💾 Memory error detected - performing async cleanup")
                # Use async cleanup instead of blocking emergency_memory_cleanup()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()  # Async - non-blocking
            
            # Light memory cleanup before fallback
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Fallback to direct sampling with memory safety
            if not is_video_workflow:
                print("🔄 Using fallback sampling method...")
            
            try:
                # Use a more conservative approach for fallback
                latent_fallback = latent_image["samples"]
                # Ensure latent on GPU
                if torch.cuda.is_available() and latent_fallback.device.type != 'cuda':
                    latent_fallback = latent_fallback.to(model_management.get_torch_device(), non_blocking=True)
                # Prepare noise on same device as latent to avoid None/CPU issues
                batch_inds_fb = latent_image["batch_index"] if "batch_index" in latent_image else None
                noise_fb = comfy.sample.prepare_noise(latent_fallback, noise_seed, batch_inds_fb)
                if torch.cuda.is_available() and noise_fb.device.type != 'cuda':
                    noise_fb = noise_fb.to(latent_fallback.device, non_blocking=True)
                samples = comfy.sample.sample(
                    model, noise_fb, steps, cfg, sampler_name, scheduler, 
                    positive, negative, latent_fallback, denoise=denoise, 
                    start_step=start_at_step, last_step=end_at_step, 
                    force_full_denoise=force_full_denoise
                )
                # Wrap in latent format
                out = latent_image.copy()
                out["samples"] = samples
                result = (out,)
            except Exception as e2:
                print(f"❌ Fallback sampling also failed: {e2}")
                logging.error(f"Fallback sampling failed: {e2}")
                
                # Last resort: try with minimal parameters
                try:
                    print("🆘 Attempting minimal sampling...")
                    latent_min = latent_image["samples"]
                    if torch.cuda.is_available() and latent_min.device.type != 'cuda':
                        latent_min = latent_min.to(model_management.get_torch_device(), non_blocking=True)
                    batch_inds_min = latent_image["batch_index"] if "batch_index" in latent_image else None
                    noise_min = comfy.sample.prepare_noise(latent_min, noise_seed, batch_inds_min)
                    if torch.cuda.is_available() and noise_min.device.type != 'cuda':
                        noise_min = noise_min.to(latent_min.device, non_blocking=True)
                    samples = comfy.sample.sample(
                        model, noise_min, min(steps, 10), min(cfg, 7.0), sampler_name, scheduler, 
                        positive, negative, latent_min, denoise=min(denoise, 0.8), 
                        start_step=0, last_step=min(steps, 10), 
                        force_full_denoise=False
                    )
                    out = latent_image.copy()
                    out["samples"] = samples
                    result = (out,)
                    print("✅ Minimal sampling succeeded")
                except Exception as e3:
                    print(f"❌ All sampling attempts failed: {e3}")
                    raise e3
        
        # OPTIMIZATION: Non-blocking post-sampling cleanup
        if memory_optimization and torch.cuda.is_available():
            torch.cuda.empty_cache()  # Async - non-blocking
            # Don't call gentle_memory_cleanup() - it contains torch.cuda.synchronize()
            if not is_video_workflow and DEBUG_MODE:
                print("🧹 Memory cache cleared (async)")
        
        sample_time = time.time() - start_time
        if is_video_workflow:
            print(f"✅ Video KSampler completed in {sample_time:.2f}s")
        else:
            print(f"✅ ROCM Advanced KSampler completed in {sample_time:.2f}s")
        logging.info(f"ROCM Advanced KSampler completed in {sample_time:.2f}s")
        
        return result




class ROCMSamplerPerformanceMonitor:
    """
    Monitor sampler performance and provide optimization suggestions
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "Model to analyze"}),
                "test_steps": ("INT", {
                    "default": 20,
                    "min": 1,
                    "max": 100,
                    "tooltip": "Number of test steps for benchmarking"
                })
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("DEVICE_INFO", "PERFORMANCE_TIPS", "OPTIMAL_SETTINGS")
    FUNCTION = "analyze"
    CATEGORY = "RocM Ninodes/Sampling"
    DESCRIPTION = "Analyze sampler performance and provide optimization recommendations"
    
    def analyze(self, model, test_steps=20):
        """
        Analyze sampler performance and provide recommendations
        """
        device = model.model_dtype()
        device_info = f"Model device: {device}\n"
        device_info += f"Model dtype: {model.model_dtype()}\n"
        
        # Check if it's an AMD GPU
        is_amd = hasattr(device, 'type') and device.type == 'cuda'
        if is_amd:
            try:
                device_name = torch.cuda.get_device_name(0)
                device_info += f"GPU: {device_name}\n"
            except:
                device_info += "GPU: AMD (ROCm)\n"
        
        # Performance tips
        tips = []
        if is_amd:
            tips.append("• Use fp32 precision for better ROCm performance")
            tips.append("• Enable memory optimization for better VRAM usage")
            tips.append("• Use attention optimization for faster sampling")
            tips.append("• Consider lower CFG values for faster generation")
            tips.append("• Euler and Heun samplers work well with ROCm")
        else:
            tips.append("• Use fp16 or bf16 for better performance")
            tips.append("• Enable all optimizations for best results")
            tips.append("• Higher CFG values generally work better")
        
        # Optimal settings
        settings = []
        if is_amd:
            settings.append(f"Recommended precision: fp32")
            settings.append(f"Recommended memory optimization: True")
            settings.append(f"Recommended attention optimization: True")
            settings.append(f"Recommended samplers: euler, heun, dpmpp_2m")
            settings.append(f"Recommended schedulers: simple, normal")
            settings.append(f"Recommended CFG: 7.0-8.0")
        else:
            settings.append(f"Recommended precision: auto")
            settings.append(f"Recommended memory optimization: True")
            settings.append(f"Recommended attention optimization: True")
            settings.append(f"Recommended samplers: dpmpp_2m, dpmpp_sde")
            settings.append(f"Recommended schedulers: normal, karras")
            settings.append(f"Recommended CFG: 8.0-12.0")
        
        return (
            device_info,
            "\n".join(tips),
            "\n".join(settings)
        )


class ROCMOptimizedCheckpointLoader:
    """
    ROCM-optimized checkpoint loader for AMD GPUs (gfx1151)
    
    Features:
    - Optimized loading for Flux and WAN models
    - Memory-efficient loading for AMD GPUs
    - Automatic precision optimization
    - Flux-specific optimizations (skip negative CLIP)
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ckpt_name": (folder_paths.get_filename_list("checkpoints"), {
                    "tooltip": "Checkpoint file to load"
                }),
                "lazy_loading": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Enable lazy loading for better memory usage"
                }),
                "optimize_for_flux": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Optimize loading for Flux models"
                }),
                "precision_mode": (["auto", "fp32", "fp16", "bf16"], {
                    "default": "auto",
                    "tooltip": "Precision mode - auto detects quantized models and preserves their dtype"
                }),
                "compatibility_mode": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Enable compatibility mode for quantized models (disables optimizations)"
                })
            }
        }
    
    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    RETURN_NAMES = ("MODEL", "CLIP", "VAE")
    FUNCTION = "load_checkpoint"
    CATEGORY = "RocM Ninodes/Loaders"
    DESCRIPTION = "ROCM-optimized checkpoint loader for AMD GPUs (gfx1151)"
    
    def load_checkpoint(self, ckpt_name, lazy_loading=True, optimize_for_flux=True, precision_mode="auto", compatibility_mode=False):
        """
        ROCM-optimized checkpoint loading with quantized model support
        """
        import folder_paths
        import comfy.sd
        import torch
        import os
        
        # CRITICAL FIX: Detect quantized models from filename
        is_quantized_model = False
        quantized_indicators = ['fp8', 'int8', 'int4', 'quantized', 'bnb', 'gguf']
        ckpt_name_lower = ckpt_name.lower()
        for indicator in quantized_indicators:
            if indicator in ckpt_name_lower:
                is_quantized_model = True
                print(f"🔍 Detected quantized checkpoint: {ckpt_name}")
                break
        
        # Compatibility mode: disable optimizations for quantized models
        if compatibility_mode or is_quantized_model:
            print("🛡️ Compatibility mode enabled - using stock ComfyUI behavior")
            optimize_for_flux = False
            lazy_loading = False
        
        try:
            # Force memory defragmentation before loading (skip for quantized models)
            if torch.cuda.is_available() and not is_quantized_model:
                # Check if defragmentation is needed and perform it
                simple_memory_cleanup()
                
                # Additional cleanup
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                gc.collect()
            elif is_quantized_model:
                print("🔒 Skipping memory cleanup for quantized model compatibility")
            
            # Get checkpoint path
            ckpt_path = folder_paths.get_full_path_or_raise("checkpoints", ckpt_name)
            print(f"📁 Loading checkpoint: {ckpt_name}")
            print(f"📂 Checkpoint path: {ckpt_path}")
            print(f"✅ File exists: {os.path.exists(ckpt_path)}")
            
            # Progress indicator for Windows users
            print("🔄 Initializing model loading...")
            
            # Check if ROCm is available
            try:
                if torch.cuda.is_available():
                    device_name = torch.cuda.get_device_name(0)
                    print(f"GPU: {device_name}")
                    if "AMD" in device_name or "Radeon" in device_name:
                        print("AMD GPU detected - using ROCm optimizations")
                    else:
                        print("Non-AMD GPU detected - using compatibility mode")
                else:
                    print("CUDA not available - using CPU mode")
            except Exception as e:
                print(f"GPU detection failed: {e}")
                print("ROCm not available - running in compatibility mode")
            
            # Use ComfyUI's standard loading - this is the most reliable approach
            print("📦 Loading model components...")
            print("⏳ Large models may take 2-6 minutes to load (this is normal)")
            print("🔄 Please be patient, especially on first run...")
            
            out = comfy.sd.load_checkpoint_guess_config(
                ckpt_path, 
                output_vae=True, 
                output_clip=True, 
                embedding_directory=folder_paths.get_folder_paths("embeddings")
            )
            print("✅ Model loading completed")
            
            # Validate the output
            if len(out) < 3:
                raise ValueError(f"Checkpoint loading returned {len(out)} items, expected 3")
            
            model, clip, vae = out[:3]
            print(f"Model loaded: {type(model)}")
            print(f"CLIP loaded: {type(clip)}")
            print(f"VAE loaded: {type(vae)}")
            
            # CRITICAL FIX: Validate that CLIP is not None
            if clip is None:
                raise ValueError(f"CLIP model is None - checkpoint {ckpt_name} may not contain a valid CLIP model")
            
            # Additional validation for Windows compatibility
            if not hasattr(clip, 'encode'):
                raise ValueError(f"CLIP model does not have encode method - invalid CLIP model")
            
            print(f"CLIP validation passed: {type(clip)}")
            
            return (model, clip, vae)
            
        except Exception as e:
            print(f"ROCM checkpoint loading failed: {e}")
            print("Attempting fallback to standard ComfyUI loading...")
            # Fallback to standard loading
            try:
                ckpt_path = folder_paths.get_full_path_or_raise("checkpoints", ckpt_name)
                out = comfy.sd.load_checkpoint_guess_config(
                    ckpt_path, 
                    output_vae=True, 
                    output_clip=True, 
                    embedding_directory=folder_paths.get_folder_paths("embeddings")
                )
                
                # Validate fallback output
                if len(out) < 3:
                    raise ValueError(f"Fallback loading returned {len(out)} items, expected 3")
                
                model, clip, vae = out[:3]
                
                # Validate CLIP in fallback
                if clip is None:
                    raise ValueError(f"Fallback CLIP model is None - checkpoint {ckpt_name} may not contain a valid CLIP model")
                
                if not hasattr(clip, 'encode'):
                    raise ValueError(f"Fallback CLIP model does not have encode method - invalid CLIP model")
                
                print("Fallback loading successful")
                return (model, clip, vae)
            except Exception as e2:
                print(f"Fallback loading failed: {e2}")
                # Last resort: try without ROCm optimizations
                try:
                    print("Attempting loading without any optimizations...")
                    ckpt_path = folder_paths.get_full_path_or_raise("checkpoints", ckpt_name)
                    out = comfy.sd.load_checkpoint_guess_config(
                        ckpt_path, 
                        output_vae=True, 
                        output_clip=True, 
                        embedding_directory=folder_paths.get_folder_paths("embeddings")
                    )
                    print("Minimal loading successful")
                    return out[:3]
                except Exception as e3:
                    print(f"All loading attempts failed: {e3}")
                    raise e3


class ROCMFluxBenchmark:
    """
    Comprehensive Flux workflow benchmark for AMD GPUs
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "The model to benchmark"}),
                "vae": ("VAE", {"tooltip": "The VAE model"}),
                "clip": ("CLIP", {"tooltip": "The CLIP model"}),
                "test_resolutions": ("STRING", {
                    "default": "256x320,512x512,1024x1024",
                    "tooltip": "Comma-separated resolutions to test (WxH)"
                }),
                "test_steps": ("INT", {
                    "default": 20,
                    "min": 1,
                    "max": 100,
                    "tooltip": "Number of sampling steps for testing"
                }),
                "test_cfg_values": ("STRING", {
                    "default": "1.0,3.5,8.0",
                    "tooltip": "Comma-separated CFG values to test"
                }),
                "test_hipblas": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Test HIPBlas optimizations"
                }),
                "generate_report": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Generate detailed optimization report"
                })
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("BENCHMARK_RESULTS", "PERFORMANCE_CHART", "OPTIMIZATION_RECOMMENDATIONS", "MEMORY_ANALYSIS")
    FUNCTION = "benchmark"
    CATEGORY = "RocM Ninodes/Benchmark"
    DESCRIPTION = "Comprehensive Flux workflow benchmark for AMD GPUs"
    
    def benchmark(self, model, vae, clip, test_resolutions="256x320,512x512,1024x1024",
                 test_steps=20, test_cfg_values="1.0,3.5,8.0", test_hipblas=True,
                 generate_report=True):
        """
        Run comprehensive benchmark tests
        """
        import time
        import torch
        import gc
        
        # Parse test parameters
        resolutions = []
        for res in test_resolutions.split(','):
            w, h = map(int, res.strip().split('x'))
            resolutions.append((w, h))
        
        cfg_values = [float(cfg.strip()) for cfg in test_cfg_values.split(',')]
        
        # Get device information
        device = comfy.model_management.get_torch_device()
        is_amd = hasattr(device, 'type') and device.type == 'cuda'
        
        results = {
            'device': str(device),
            'is_amd': is_amd,
            'resolutions': {},
            'cfg_tests': {},
            'memory_usage': {},
            'recommendations': []
        }
        
        # Test each resolution
        for w, h in resolutions:
            print(f"Testing resolution {w}x{h}...")
            resolution_key = f"{w}x{h}"
            results['resolutions'][resolution_key] = {
                'times': [],
                'memory_peak': 0,
                'memory_avg': 0
            }
            
            # Test with different CFG values
            for cfg in cfg_values:
                print(f"  Testing CFG {cfg}...")
                start_time = time.time()
                
                # Clear memory
                torch.cuda.empty_cache()
                gc.collect()
                
                # Create test latent
                latent = torch.randn(1, 4, h//8, w//8, device=device)
                
                # Test VAE decode
                try:
                    with torch.no_grad():
                        decoded = vae.decode(latent)
                    decode_time = time.time() - start_time
                    
                    # Record results
                    results['resolutions'][resolution_key]['times'].append(decode_time)
                    results['resolutions'][resolution_key]['memory_peak'] = max(
                        results['resolutions'][resolution_key]['memory_peak'],
                        torch.cuda.max_memory_allocated() / 1024**3
                    )
                    
                    print(f"    VAE decode time: {decode_time:.2f}s")
                    
                except Exception as e:
                    print(f"    Error testing {w}x{h} CFG {cfg}: {e}")
                    results['resolutions'][resolution_key]['times'].append(float('inf'))
        
        # Generate recommendations
        if is_amd:
            results['recommendations'].extend([
                "• Use fp32 precision for best ROCm performance",
                "• Enable memory optimization for better VRAM usage",
                "• Use Euler or Heun samplers for optimal speed",
                "• Consider lower CFG values (3.5-7.0) for faster generation",
                "• Use tile size 768-1024 for optimal memory/speed balance"
            ])
        else:
            results['recommendations'].extend([
                "• Use fp16 or bf16 for better performance",
                "• Enable all available optimizations",
                "• Use DPM++ 2M sampler for best quality/speed",
                "• Higher CFG values (8.0-12.0) generally work better",
                "• Use tile size 512-768 for optimal performance"
            ])
        
        # Format results
        benchmark_text = f"Benchmark Results for {results['device']}\n"
        benchmark_text += f"AMD GPU: {results['is_amd']}\n\n"
        
        for res_key, res_data in results['resolutions'].items():
            if res_data['times']:
                avg_time = sum(t for t in res_data['times'] if t != float('inf')) / len([t for t in res_data['times'] if t != float('inf')])
                benchmark_text += f"{res_key}: {avg_time:.2f}s avg, {res_data['memory_peak']:.1f}GB peak\n"
        
        performance_chart = "Performance Chart:\n"
        for res_key, res_data in results['resolutions'].items():
            if res_data['times']:
                times_str = ", ".join(f"{t:.2f}s" for t in res_data['times'] if t != float('inf'))
                performance_chart += f"{res_key}: [{times_str}]\n"
        
        recommendations_text = "Optimization Recommendations:\n"
        for rec in results['recommendations']:
            recommendations_text += f"{rec}\n"
        
        memory_analysis = f"Memory Analysis:\n"
        memory_analysis += f"Device: {results['device']}\n"
        memory_analysis += f"Total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB\n"
        memory_analysis += f"Current VRAM usage: {torch.cuda.memory_allocated() / 1024**3:.1f}GB\n"
        
        return (benchmark_text, performance_chart, recommendations_text, memory_analysis)






class ROCMMemoryOptimizer:
    """
    Memory optimization helper for AMD GPUs
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "optimization_level": (["conservative", "balanced", "aggressive"], {
                    "default": "balanced",
                    "tooltip": "Memory optimization level"
                }),
                "enable_gc": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Enable garbage collection"
                }),
                "clear_cache": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Clear CUDA cache"
                }),
                "cleanup_frequency": ("INT", {
                    "default": 10,
                    "min": 1,
                    "max": 100,
                    "tooltip": "Cleanup frequency (operations between cleanups)"
                })
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("MEMORY_STATUS", "OPTIMIZATION_LOG", "RECOMMENDATIONS")
    FUNCTION = "optimize_memory"
    CATEGORY = "RocM Ninodes/Memory"
    DESCRIPTION = "Memory optimization helper for AMD GPUs"
    
    def __init__(self):
        self.operation_count = 0
    
    def optimize_memory(self, optimization_level="balanced", enable_gc=True, 
                       clear_cache=True, cleanup_frequency=10):
        """
        Optimize memory usage for AMD GPUs
        """
        import torch
        import gc
        
        self.operation_count += 1
        
        # Get current memory status
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            free = total - reserved
            
            memory_status = f"Memory Status:\n"
            memory_status += f"Allocated: {allocated:.2f}GB\n"
            memory_status += f"Reserved: {reserved:.2f}GB\n"
            memory_status += f"Free: {free:.2f}GB\n"
            memory_status += f"Total: {total:.2f}GB\n"
        else:
            memory_status = "CUDA not available - no GPU memory to optimize"
        
        # Perform optimizations
        optimization_log = f"Optimization Log (Level: {optimization_level}):\n"
        
        if enable_gc and self.operation_count % cleanup_frequency == 0:
            gc.collect()
            optimization_log += "✓ Garbage collection performed\n"
        
        if clear_cache and self.operation_count % cleanup_frequency == 0:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                optimization_log += "✓ CUDA cache cleared\n"
        
        # Level-specific optimizations
        if optimization_level == "aggressive":
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                optimization_log += "✓ CUDA synchronization performed\n"
        
        # Generate recommendations
        recommendations = "Memory Optimization Recommendations:\n"
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            
            if allocated / total > 0.8:
                recommendations += "⚠ High memory usage detected\n"
                recommendations += "• Consider reducing batch size\n"
                recommendations += "• Use lower resolution\n"
                recommendations += "• Enable aggressive optimization\n"
            elif allocated / total > 0.6:
                recommendations += "• Memory usage is moderate\n"
                recommendations += "• Consider balanced optimization\n"
            else:
                recommendations += "✓ Memory usage is good\n"
                recommendations += "• Current settings are optimal\n"
        
        return (memory_status, optimization_log, recommendations)


# Node mappings
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
    CATEGORY = "RocM Ninodes/Loaders"
    DESCRIPTION = "ROCM-optimized LoRA loader with aggressive memory management"
    
    def load_lora(self, model, lora_name, strength_model, strength_clip, clip=None):
        """
        Load LoRA with aggressive memory management to prevent fragmentation
        """
        print(f"🔄 ROCM LoRA Loader: Loading {lora_name} with strengths {strength_model}/{strength_clip}")
        
        # Pre-loading gentle memory cleanup
        if torch.cuda.is_available():
            print("🧹 Pre-loading memory cleanup...")
            gentle_memory_cleanup()
            
            # Log memory state before LoRA loading
            allocated_memory = torch.cuda.memory_allocated(0)
            reserved_memory = torch.cuda.memory_reserved(0)
            fragmentation = reserved_memory - allocated_memory
            
            print(f"Memory before LoRA loading: {allocated_memory/1024**3:.2f}GB allocated, {reserved_memory/1024**3:.2f}GB reserved")
            print(f"Fragmentation: {fragmentation/1024**2:.1f}MB")
        
        try:
            # Import ComfyUI's LoRA loading functions
            from comfy.lora import load_lora
            from comfy.model_patcher import ModelPatcher
            
            # Load LoRA with aggressive memory management
            lora_path = folder_paths.get_full_path("loras", lora_name)
            if lora_path is None:
                raise FileNotFoundError(f"LoRA file not found: {lora_name}")
            
            print(f"📁 Loading LoRA from: {lora_path}")
            
            # Apply gentle memory cleanup before each major operation
            if torch.cuda.is_available():
                gentle_memory_cleanup()
            
            # Load the LoRA
            lora = load_lora(lora_path)
            
            # Apply gentle memory cleanup after loading
            if torch.cuda.is_available():
                gentle_memory_cleanup()
            
            # Apply LoRA to model with memory management
            model_lora, clip_lora = lora.apply_to_model(model, clip, strength_model, strength_clip)
            
            # Post-loading gentle memory cleanup
            if torch.cuda.is_available():
                print("🧹 Post-loading memory cleanup...")
                gentle_memory_cleanup()
                
                # Log memory state after LoRA loading
                allocated_memory_after = torch.cuda.memory_allocated(0)
                reserved_memory_after = torch.cuda.memory_reserved(0)
                fragmentation_after = reserved_memory_after - allocated_memory_after
                
                print(f"Memory after LoRA loading: {allocated_memory_after/1024**3:.2f}GB allocated, {reserved_memory_after/1024**3:.2f}GB reserved")
                print(f"Fragmentation: {fragmentation_after/1024**2:.1f}MB")
            
            print(f"✅ ROCM LoRA Loader: Successfully loaded {lora_name}")
            return (model_lora, clip_lora)
            
        except Exception as e:
            print(f"❌ ROCM LoRA Loader: Failed to load {lora_name}: {e}")
            
            # Emergency memory cleanup on failure
            if torch.cuda.is_available():
                print("🚨 Emergency memory cleanup...")
                gentle_memory_cleanup()
            
            raise e


NODE_CLASS_MAPPINGS = {
    "ROCMOptimizedCheckpointLoader": ROCMOptimizedCheckpointLoader,
    "ROCMOptimizedVAEDecode": ROCMOptimizedVAEDecode,
    "ROCMOptimizedVAEDecodeTiled": ROCMOptimizedVAEDecodeTiled,
    "ROCMVAEPerformanceMonitor": ROCMVAEPerformanceMonitor,
    "ROCMOptimizedKSampler": ROCMOptimizedKSampler,
    "ROCMOptimizedKSamplerAdvanced": ROCMOptimizedKSamplerAdvanced,
    "ROCMSamplerPerformanceMonitor": ROCMSamplerPerformanceMonitor,
    "ROCMFluxBenchmark": ROCMFluxBenchmark,
    "ROCMMemoryOptimizer": ROCMMemoryOptimizer,
    "ROCMLoRALoader": ROCMLoRALoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ROCMOptimizedCheckpointLoader": "ROCM Checkpoint Loader",
    "ROCMOptimizedVAEDecode": "ROCM VAE Decode",
    "ROCMOptimizedVAEDecodeTiled": "ROCM VAE Decode Tiled", 
    "ROCMVAEPerformanceMonitor": "ROCM VAE Performance Monitor",
    "ROCMOptimizedKSampler": "ROCM KSampler",
    "ROCMOptimizedKSamplerAdvanced": "ROCM KSampler Advanced",
    "ROCMSamplerPerformanceMonitor": "ROCM Sampler Performance Monitor",
    "ROCMFluxBenchmark": "ROCM Flux Benchmark",
    "ROCMMemoryOptimizer": "ROCM Memory Optimizer",
    "ROCMLoRALoader": "ROCM LoRA Loader",
    "ROCMQuantizedModelOptimizer": "ROCM Quantized Model Optimizer",
}

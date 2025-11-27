"""
VAE Decode nodes for ROCM Ninodes.

Contains all VAE-related node implementations:
- ROCMOptimizedVAEDecode: Main VAE decode node with ROCm optimizations
- ROCMOptimizedVAEDecodeTiled: Advanced tiled VAE decode
- ROCMVAEPerformanceMonitor: Performance monitoring and recommendations
"""

import time
import logging
from typing import Dict, Any, Tuple, Optional

import torch
import torch.nn.functional as F
import comfy.model_management as model_management
import comfy.utils
import comfy.sample
import comfy.samplers
import latent_preview
import folder_paths

# Import WAN VAE classes for detection
try:
    import comfy.ldm.wan.vae
    import comfy.ldm.wan.vae2_2
    WAN_VAE_AVAILABLE = True
except ImportError:
    WAN_VAE_AVAILABLE = False

# Import utilities from refactored modules
from ..utils.memory import (
    gentle_memory_cleanup,
    check_memory_safety,
    emergency_memory_cleanup,
    get_gpu_memory_info,
)
from ..utils.debug import (
    DEBUG_MODE,
    log_debug,
    save_debug_data,
    capture_timing,
    capture_memory_usage,
)
from ..utils.quantization import detect_model_quantization
from ..constants import DEFAULT_TILE_SIZE, DEFAULT_TILE_OVERLAP


class ROCMOptimizedVAEDecode:
    """
    ROCM-optimized VAE Decode node specifically tuned for gfx1151 architecture.
    
    Key optimizations:
    - Optimized memory management for ROCm
    - Better batching strategy for AMD GPUs
    - Reduced model conversion overhead
    - Intelligent tiled vs direct decode decision
    - Optimized tile sizes for gfx1151
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
                    "default": 81,
                    "min": 1,
                    "max": 200,
                    "step": 1,
                    "tooltip": "Number of video frames to process at once (memory optimization). Set to 81 or higher to process all frames without chunking."
                }),
                "memory_optimization_enabled": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Enable memory optimization for video processing"
                })
            },
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
    CATEGORY = "ROCm Ninodes/VAE"
    DESCRIPTION = "ROCM-optimized VAE Decode for AMD GPUs (gfx1151)"
    
    def decode(self, vae, samples, tile_size=768, overlap=96, use_rocm_optimizations=True, 
               precision_mode="auto", batch_optimization=True, video_chunk_size=81, 
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
            if WAN_VAE_AVAILABLE:
                # Primary detection: isinstance check with imported WAN VAE classes
                if isinstance(vae.first_stage_model, (comfy.ldm.wan.vae.WanVAE, comfy.ldm.wan.vae2_2.WanVAE)):
                    is_wan_vae = True
                    print(f"🔍 Detected WAN VAE model (causal decoding requires full video processing)")
                else:
                    # Fallback detection: check VAE attributes (latent_channels: 16 for WAN 2.1, 48 for WAN 2.2)
                    latent_channels = getattr(vae, 'latent_channels', None)
                    if latent_channels in [16, 48]:
                        # Additional check: WAN VAEs have temporal compression
                        temporal_compression = vae.temporal_compression_decode() if hasattr(vae, 'temporal_compression_decode') else None
                        if temporal_compression is not None:
                            is_wan_vae = True
                            print(f"🔍 Detected WAN VAE model (latent_channels={latent_channels}, causal decoding requires full video processing)")
        except Exception as e:
            # If detection fails, print error but continue (safer than crashing)
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
        
        # OPTIMIZATION: Apply ROCm backend settings early (before video processing)
        # These are global settings that improve performance without affecting causal decoding
        if use_rocm_optimizations and is_amd:
            torch.backends.cuda.matmul.allow_tf32 = False  # TF32 not supported on AMD
            torch.backends.cuda.matmul.allow_fp16_accumulation = True  # Better performance on AMD
            
            # Detect gfx1151 for architecture-specific optimizations
            is_gfx1151 = False
            try:
                if torch.cuda.is_available():
                    arch = torch.cuda.get_device_properties(0).gcnArchName
                    if 'gfx1151' in arch:
                        is_gfx1151 = True
                        if is_wan_vae:
                            print(f"🚀 gfx1151 detected - optimized for WAN VAE video processing")
            except:
                pass
        
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
            elif memory_optimization_enabled and T > video_chunk_size and T > 20:  # Only chunk for videos > 20 frames
                use_chunking = True
                print(f"📹 Using chunked processing: {video_chunk_size} frames per chunk")
            else:
                use_chunking = False
                print("📹 Using non-chunked processing for optimal speed")
            
            if use_chunking:
                # Process video in chunks to avoid memory exhaustion
                # NO OVERLAP - overlaps cause stuttering in videos
                chunk_results = []
                num_chunks = (T + video_chunk_size - 1) // video_chunk_size
                print(f"🔄 Processing {num_chunks} chunks (no overlap to prevent stuttering)...")
                
                for i in range(0, T, video_chunk_size):
                    chunk_idx = i // video_chunk_size + 1
                    end_idx = min(i + video_chunk_size, T)
                    print(f"  📦 Chunk {chunk_idx}/{num_chunks}: frames {i}-{end_idx-1}")
                    
                    # NO OVERLAP - decode exactly the frames we need
                    chunk_start = i
                    chunk_end = end_idx
                    
                    chunk = samples["samples"][:, :, chunk_start:chunk_end, :, :]
                    
                    # Decode chunk
                    with torch.no_grad():
                        chunk_decoded = vae.decode(chunk)
                    
                    # VAE decode returns a tuple, extract the tensor
                    if isinstance(chunk_decoded, tuple):
                        chunk_decoded = chunk_decoded[0]
                    
                    # No cropping needed - we decoded exactly the frames we need
                    chunk_results.append(chunk_decoded)
                    
                    # Clear memory after each chunk (reduced frequency to prevent fragmentation)
                    if i % (video_chunk_size * 2) == 0:
                        torch.cuda.empty_cache()
                
                # Concatenate results along temporal dimension
                first = chunk_results[0]
                if len(first.shape) == 5 and first.shape[1] == 3:
                    result = torch.cat(chunk_results, dim=2)
                elif len(first.shape) == 5 and first.shape[-1] == 3:
                    result = torch.cat(chunk_results, dim=1)
                else:
                    result = torch.cat(chunk_results, dim=2)
                
                # Convert 5D video tensor to 4D image tensor for ComfyUI (B*T, H, W, C)
                if len(result.shape) == 5:
                    if result.shape[1] == 3:
                        result = result.permute(0, 2, 3, 4, 1).contiguous()
                        B, T, H, W, C = result.shape
                        result = result.reshape(B * T, H, W, C)
                    elif result.shape[-1] == 3:
                        B, T, H, W, C = result.shape
                        result = result.reshape(B * T, H, W, C)
                    else:
                        result = result.permute(0, 2, 3, 4, 1).contiguous()
                        B, T, H, W, C = result.shape
                        result = result.reshape(B * T, H, W, C)
                
                return (result,)
            else:
                # Process entire video at once - EXACTLY match native ComfyUI behavior
                # Native ComfyUI: images = vae.decode(samples["samples"])
                #              if len(images.shape) == 5: images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])
                if is_wan_vae:
                    print(f"🎯 Processing entire WAN VAE video at once (causal decoding): {T} frames")
                    # OPTIMIZATION: Async memory cleanup before large WAN video processing (non-blocking)
                    if use_rocm_optimizations and is_amd:
                        torch.cuda.empty_cache()  # Async - non-blocking, won't interfere with causal decoding
                else:
                    print(f"🎯 Processing entire video at once: {T} frames")
                
                with torch.no_grad():
                    result = vae.decode(samples["samples"])
                
                # VAE decode returns a tuple, extract the tensor (ComfyUI handles this internally)
                if isinstance(result, tuple):
                    result = result[0]
                
                # Convert 5D video tensor to 4D image tensor - EXACT native ComfyUI logic
                # Native does: images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])
                if len(result.shape) == 5:
                    # Match native ComfyUI reshape logic exactly
                    result = result.reshape(-1, result.shape[-3], result.shape[-2], result.shape[-1])
                
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
        
        # CRITICAL FIX: Preserve quantized model dtypes
        if is_quantized_model:
            print(f"🔒 Preserving quantized model dtype: {vae_model_dtype}")
            optimal_dtype = vae_model_dtype
        elif precision_mode == "auto":
            if is_amd:
                if vae_model_dtype is not None:
                    optimal_dtype = vae_model_dtype
                    logging.info(f"Using VAE model's existing dtype: {optimal_dtype}")
                else:
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
        
        # Skip dtype conversion for quantized models
        if not is_quantized_model:
            if vae_model_dtype == torch.bfloat16 and optimal_dtype == torch.float32:
                logging.warning("VAE model has BFloat16 weights but input is Float32")
                logging.warning("Converting VAE model to Float32 to match input dtype")
                vae.first_stage_model = vae.first_stage_model.to(torch.float32)
                optimal_dtype = torch.float32
        else:
            print(f"🔒 Skipping dtype conversion for quantized model (dtype: {vae_model_dtype})")
        
        # ROCm-specific optimizations
        if use_rocm_optimizations and is_amd:
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cuda.matmul.allow_fp16_accumulation = True
            
            try:
                gentle_memory_cleanup()
                print("🧹 Memory cache cleared gently")
            except Exception as e:
                print(f"⚠️ Memory optimization skipped: {e}")
            
            # Optimize tile size for gfx1151
            if tile_size > 1024:
                tile_size = 1024
            if overlap > tile_size // 4:
                overlap = tile_size // 4
        
        # Optimized memory management
        samples_shape = samples["samples"].shape
        if len(samples_shape) == 4:
            estimated_memory = samples_shape[0] * samples_shape[1] * samples_shape[2] * samples_shape[3] * 4 * 8
        else:
            estimated_memory = samples_shape[0] * samples_shape[1] * samples_shape[2] * samples_shape[3] * 4 * 8
        
        # Check memory safety
        estimated_memory_gb = estimated_memory / (1024**3)
        is_safe, memory_msg = check_memory_safety(required_memory_gb=estimated_memory_gb + 2.0)
        
        if not is_safe:
            print(f"⚠️ VAE Decode Memory Warning: {memory_msg}")
            print("🧹 Performing emergency cleanup before VAE decode...")
            emergency_memory_cleanup()
        
        # Load models
        try:
            model_management.load_models_gpu([vae.patcher], memory_required=estimated_memory)
        except Exception as e:
            if "out of memory" in str(e).lower():
                print("💾 VAE model loading failed due to memory - performing cleanup")
                emergency_memory_cleanup()
                model_management.load_models_gpu([vae.patcher], memory_required=estimated_memory // 2)
            else:
                raise e
        
        # Calculate optimal batch size
        free_memory = model_management.get_free_memory(device)
        if batch_optimization and is_amd:
            batch_number = max(1, int(free_memory / (estimated_memory * 1.2)))
        else:
            batch_number = max(1, int(free_memory / estimated_memory))
        
        # Process samples
        samples_tensor = samples["samples"]
        samples_processed = samples_tensor.to(device).to(optimal_dtype)
        
        # Use caching to avoid repeated model conversions
        vae_id = id(vae.first_stage_model)
        cache_key = f"{vae_id}_{optimal_dtype}"
        
        if cache_key not in self._vae_model_cache:
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
        
        # Decide processing method
        image_size = samples_tensor.shape[2] * samples_tensor.shape[3]
        use_tiled = image_size > 512 * 512 or estimated_memory > free_memory * 0.8
        
        if not is_video:
            print(f"🖼️ Processing image: {samples_tensor.shape[2]}x{samples_tensor.shape[3]}")
            if use_tiled:
                print(f"🔧 Using tiled processing for large image")
            else:
                print(f"⚡ Using direct processing for optimal speed")
        
        if not use_tiled:
            try:
                # CRITICAL FIX: Use vae.decode() wrapper (same as ComfyUI native node)
                # This ensures proper formatting, device placement, and normalization
                # Calling first_stage_model.decode() directly bypasses important formatting steps!
                # ComfyUI native node: images = vae.decode(samples["samples"])
                pixel_samples = vae.decode(samples_processed)
                
                # vae.decode() returns a tuple for video, extract if needed
                if isinstance(pixel_samples, tuple):
                    pixel_samples = pixel_samples[0]
            except Exception as e:
                logging.warning(f"Direct decode failed, falling back to tiled: {e}")
                use_tiled = True
        
        if use_tiled:
            pixel_samples = self._decode_tiled_optimized(
                vae, samples_tensor, tile_size, overlap, optimal_dtype, batch_number
            )
        
        # CRITICAL FIX: vae.decode() already returns properly formatted output
        # ComfyUI native node only reshapes for video (5D -> 4D)
        # We don't need manual dtype/format conversion - vae.decode() handles it!
        if len(pixel_samples.shape) == 5:
            # Handle video output (same as ComfyUI native)
            pixel_samples = pixel_samples.reshape(-1, pixel_samples.shape[-3], 
                                                pixel_samples.shape[-2], pixel_samples.shape[-1])
        
        decode_time = time.time() - start_time
        logging.info(f"ROCM VAE Decode completed in {decode_time:.2f}s")
        
        if is_video:
            print(f"✅ Video decode completed in {decode_time:.2f}s")
        else:
            print(f"✅ Image decode completed in {decode_time:.2f}s")
        
        return (pixel_samples,)
    
    def _decode_tiled_optimized(self, vae, samples, tile_size, overlap, dtype, batch_number):
        """Optimized tiled decoding for ROCm"""
        compression = vae.spacial_compression_decode()
        tile_x = tile_size // compression
        tile_y = tile_size // compression
        overlap_adj = overlap // compression
        
        # Detect quantized model
        is_quantized_model = False
        vae_model_dtype = getattr(vae.first_stage_model, 'dtype', None)
        if vae_model_dtype is not None:
            quantized_dtypes = [torch.float8_e4m3fn, torch.float8_e5m2, torch.int8, torch.int4]
            if vae_model_dtype in quantized_dtypes:
                is_quantized_model = True
            elif hasattr(vae_model_dtype, '__name__') and 'int' in str(vae_model_dtype):
                is_quantized_model = True
        
        # Cache model conversions
        vae_id = id(vae.first_stage_model)
        cache_key = f"{vae_id}_{dtype}"
        
        if cache_key not in self._vae_model_cache:
            if not is_quantized_model:
                current_model_dtype = getattr(vae.first_stage_model, 'dtype', None)
                if current_model_dtype is not None and current_model_dtype != dtype:
                    logging.info(f"Tiled decode: Converting VAE model from {current_model_dtype} to {dtype}")
                    vae.first_stage_model = vae.first_stage_model.to(dtype)
            else:
                logging.info(f"Tiled decode: Preserving quantized model dtype: {vae_model_dtype}")
            self._vae_model_cache[cache_key] = True
        
        def decode_fn(samples_tile):
            # CRITICAL FIX: Use vae.decode() wrapper instead of first_stage_model.decode()
            # This ensures consistent output formatting with ComfyUI's native behavior
            samples_tile = samples_tile.to(vae.device).to(dtype)
            result = vae.decode(samples_tile)
            if isinstance(result, tuple):
                result = result[0]
            return result
        
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
            out_channels=3,
            output_device=device
        )
        
        # Handle WAN VAE output format
        if len(result.shape) == 5:
            result = result.permute(0, 2, 1, 3, 4).contiguous()
            result = result.reshape(-1, result.shape[2], result.shape[3], result.shape[4])
        elif len(result.shape) == 4 and result.shape[1] > 3:
            result = result[:, :3, :, :]
        
        if DEBUG_MODE:
            end_time = time.time()
            save_debug_data(result, "vae_decode_output", "flux_1024x1024", {
                'node_type': 'ROCMOptimizedVAEDecode',
                'execution_time': end_time - time.time(),
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
    """Advanced tiled VAE decode with ROCm optimizations"""
    
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
    CATEGORY = "ROCm Ninodes/VAE"
    DESCRIPTION = "Advanced tiled VAE decode optimized for ROCm"
    
    def decode(self, vae, samples, tile_size=768, overlap=96, temporal_size=64, 
               temporal_overlap=8, rocm_optimizations=True):
        """Advanced tiled decode with ROCm optimizations"""
        start_time = time.time()
        
        if rocm_optimizations:
            if tile_size < overlap * 4:
                overlap = tile_size // 4
            if temporal_size < temporal_overlap * 2:
                temporal_overlap = temporal_size // 2
        
        temporal_compression = vae.temporal_compression_decode()
        if temporal_compression is not None:
            temporal_size = max(2, temporal_size // temporal_compression)
            temporal_overlap = max(1, min(temporal_size // 2, temporal_overlap // temporal_compression))
        else:
            temporal_size = None
            temporal_overlap = None
        
        compression = vae.spacial_compression_decode()
        images = vae.decode_tiled(
            samples["samples"], 
            tile_x=tile_size // compression, 
            tile_y=tile_size // compression, 
            overlap=overlap // compression, 
            tile_t=temporal_size, 
            overlap_t=temporal_overlap
        )
        
        if len(images.shape) == 5:
            images = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])
        
        decode_time = time.time() - start_time
        logging.info(f"ROCM Tiled VAE Decode completed in {decode_time:.2f}s")
        
        return (images,)


class ROCMVAEPerformanceMonitor:
    """Monitor VAE performance and provide optimization suggestions"""
    
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
    CATEGORY = "ROCm Ninodes/VAE"
    DESCRIPTION = "Analyze VAE performance and provide optimization recommendations"
    
    def analyze(self, vae, test_resolution=1024):
        """Analyze VAE performance and provide recommendations"""
        device = vae.device
        device_info = f"Device: {device}\n"
        device_info += f"VAE dtype: {vae.vae_dtype}\n"
        device_info += f"Output device: {vae.output_device}\n"
        
        is_amd = hasattr(device, 'type') and device.type == 'cuda'
        if is_amd:
            try:
                device_name = torch.cuda.get_device_name(0)
                device_info += f"GPU: {device_name}\n"
            except:
                device_info += "GPU: AMD (ROCm)\n"
        
        tips = []
        if is_amd:
            tips.append("• Use fp32 precision for better ROCm performance")
            tips.append("• Tile size 768-1024 works well for gfx1151")
            tips.append("• Enable ROCm optimizations for best results")
            tips.append("• Consider using tiled decode for images > 1024x1024")
        else:
            tips.append("• Use fp16 or bf16 for better performance")
            tips.append("• Larger tile sizes generally perform better")
        
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


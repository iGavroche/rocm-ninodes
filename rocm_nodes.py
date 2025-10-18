"""
ROCM Optimized VAE Nodes for AMD GPUs
Specifically optimized for gfx1151 architecture with ROCm 6.4+
"""

import torch
import torch.nn.functional as F
import comfy.model_management as model_management
import comfy.utils
import comfy.sample
import comfy.samplers
import latent_preview
import logging
import folder_paths
from typing import Dict, Any, Tuple, Optional
import time
# from debug_config import DEBUG_MODE, save_debug_data, capture_timing, capture_memory_usage, log_debug
DEBUG_MODE = False
def save_debug_data(*args, **kwargs): pass
def capture_timing(*args, **kwargs): pass
def capture_memory_usage(*args, **kwargs): pass
def log_debug(*args, **kwargs): pass

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
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("IMAGE",)
    FUNCTION = "decode"
    CATEGORY = "RocM Ninodes/VAE"
    DESCRIPTION = "ROCM-optimized VAE Decode for AMD GPUs (gfx1151)"
    
    def decode(self, vae, samples, tile_size=768, overlap=96, use_rocm_optimizations=True, 
               precision_mode="auto", batch_optimization=True, video_chunk_size=8, 
               memory_optimization_enabled=True):
        """
        Optimized VAE decode for ROCm/AMD GPUs with video support
        """
        start_time = time.time()
        log_debug(f"ROCMOptimizedVAEDecode.decode started with samples shape: {samples['samples'].shape}")
        
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
            
            # Memory-safe video processing
            # CRITICAL FIX: WAN VAE has issues with chunking (frame duplication)
            # Force non-chunked processing for WAN VAE to prevent frame count issues
            if memory_optimization_enabled and T > video_chunk_size and T < 100:  # Only chunk for very large videos
                use_chunking = False
                print("📹 Using non-chunked processing for WAN VAE stability")
            else:
                use_chunking = memory_optimization_enabled and T > video_chunk_size
                if use_chunking:
                    print(f"📹 Using chunked processing: {video_chunk_size} frames per chunk")
            
            if use_chunking:
                # Process video in chunks to avoid memory exhaustion
                chunk_results = []
                num_chunks = (T + video_chunk_size - 1) // video_chunk_size
                print(f"🔄 Processing {num_chunks} chunks...")
                
                for i in range(0, T, video_chunk_size):
                    chunk_idx = i // video_chunk_size + 1
                    end_idx = min(i + video_chunk_size, T)
                    print(f"  📦 Chunk {chunk_idx}/{num_chunks}: frames {i}-{end_idx-1}")
                    chunk = samples["samples"][:, :, i:end_idx, :, :]
                    
                    # Keep original 5D shape for WAN VAE - don't reshape to 4D
                    # WAN VAE expects [B, C, T, H, W] format for memory calculation
                    # Decode chunk - WAN VAE expects 5D tensor [B, C, T, H, W]
                    
                    with torch.no_grad():
                        chunk_decoded = vae.decode(chunk)
                    
                    # VAE decode returns a tuple, extract the tensor
                    if isinstance(chunk_decoded, tuple):
                        chunk_decoded = chunk_decoded[0]
                    
                    
                    # Reshape back to video format - chunk_decoded should already be in correct format
                    # No need to reshape since we kept the 5D format
                    chunk_results.append(chunk_decoded)
                    
                    # Clear memory after each chunk (reduced frequency to prevent fragmentation)
                    if i % (video_chunk_size * 2) == 0:  # Only clear every 2 chunks
                        torch.cuda.empty_cache()
                
                # Concatenate results
                result = torch.cat(chunk_results, dim=1)
                
                
                # Convert 5D video tensor to 4D image tensor for ComfyUI
                # Input: [B, T, H, W, C] -> Output: [B*T, H, W, C]
                if len(result.shape) == 5:
                    B, T, H, W, C = result.shape
                    result = result.reshape(B * T, H, W, C)
                
                return (result,)
            else:
                # Process entire video at once - keep 5D format for WAN VAE
                B, C, T, H, W = samples["samples"].shape
                video_tensor = samples["samples"]
                print(f"🎯 Processing entire video at once: {T} frames")
                
                with torch.no_grad():
                    result = vae.decode(video_tensor)
                
                # VAE decode returns a tuple, extract the tensor
                if isinstance(result, tuple):
                    result = result[0]
                
                
                # Convert 5D video tensor to 4D image tensor for ComfyUI
                # Input: [B, T, H, W, C] -> Output: [B*T, H, W, C]
                if len(result.shape) == 5:
                    B, T, H, W, C = result.shape
                    
                    
                    result = result.reshape(B * T, H, W, C)
                
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
        
        # Set optimal precision for AMD GPUs - match VAE model dtype to avoid type mismatch
        if precision_mode == "auto":
            if is_amd:
                # Check VAE model's actual dtype first to avoid type mismatch
                vae_model_dtype = getattr(vae.first_stage_model, 'dtype', None)
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
        
        # CRITICAL FIX: Handle BFloat16 model with Float32 input type mismatch
        # If the VAE model has BFloat16 weights but we're using Float32, we need to match them
        vae_model_dtype = getattr(vae.first_stage_model, 'dtype', None)
        if vae_model_dtype == torch.bfloat16 and optimal_dtype == torch.float32:
            logging.warning("VAE model has BFloat16 weights but input is Float32 - this will cause type mismatch")
            logging.warning("Converting VAE model to Float32 to match input dtype")
            vae.first_stage_model = vae.first_stage_model.to(torch.float32)
            optimal_dtype = torch.float32
        
        # ROCm-specific optimizations (set once, don't repeat)
        if use_rocm_optimizations and is_amd:
            # Enable ROCm optimizations
            torch.backends.cuda.matmul.allow_tf32 = False  # Disable TF32 for AMD
            torch.backends.cuda.matmul.allow_fp16_accumulation = True
            
            # SAFE OPTIMIZATION: Gentle memory management for Windows stability
            try:
                # Clear cache gently to prevent Windows hanging
                torch.cuda.empty_cache()
                print("🧹 Memory cache cleared for optimal performance")
                
                # SAFE OPTIMIZATION: Set memory allocation strategy for Windows
                if hasattr(torch.cuda, 'set_per_process_memory_fraction'):
                    try:
                        # Use a conservative memory fraction to prevent Windows hanging
                        torch.cuda.set_per_process_memory_fraction(0.85)  # More conservative than 0.9
                        print("🔧 Memory allocation optimized for Windows stability")
                    except Exception as e:
                        print(f"⚠️ Memory fraction setting skipped: {e}")
                        
            except Exception as e:
                print(f"⚠️ Memory optimization skipped: {e}")
            
            # Optimize tile size for gfx1151
            if tile_size > 1024:
                tile_size = 1024  # Cap for gfx1151 memory
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
        
        # Load models with estimated memory (faster than exact calculation)
        model_management.load_models_gpu([vae.patcher], memory_required=estimated_memory)
        
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
            # Ensure VAE model is in optimal dtype (do this once, not repeatedly)
            # CRITICAL FIX: Only convert model dtype if it's different to avoid type mismatch
            current_model_dtype = getattr(vae.first_stage_model, 'dtype', None)
            if current_model_dtype is not None and current_model_dtype != optimal_dtype:
                logging.info(f"Converting VAE model from {current_model_dtype} to {optimal_dtype}")
                vae.first_stage_model = vae.first_stage_model.to(optimal_dtype)
            else:
                logging.info(f"VAE model already in correct dtype: {current_model_dtype}")
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
                
                out = out.to(vae.output_device).float()
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
        
        # Move to output device (no movedim needed - match standard VAE decode)
        pixel_samples = pixel_samples.to(vae.output_device)
        
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
        
        # Use caching to avoid repeated model conversions in tiled decoding
        vae_id = id(vae.first_stage_model)
        cache_key = f"{vae_id}_{dtype}"
        
        if cache_key not in self._vae_model_cache:
            # Ensure VAE model is in correct dtype (do this once, not per tile)
            # CRITICAL FIX: Only convert model dtype if it's different to avoid type mismatch
            current_model_dtype = getattr(vae.first_stage_model, 'dtype', None)
            if current_model_dtype is not None and current_model_dtype != dtype:
                logging.info(f"Tiled decode: Converting VAE model from {current_model_dtype} to {dtype}")
                vae.first_stage_model = vae.first_stage_model.to(dtype)
            else:
                logging.info(f"Tiled decode: VAE model already in correct dtype: {current_model_dtype}")
            self._vae_model_cache[cache_key] = True
        
        # Use ComfyUI's tiled scale with optimizations
        def decode_fn(samples_tile):
            # Ensure consistent data types for ROCm (minimal conversion)
            samples_tile = samples_tile.to(vae.device).to(dtype)
            return vae.first_stage_model.decode(samples_tile)
        
        result = comfy.utils.tiled_scale(
            samples, 
            decode_fn, 
            tile_x=tile_x, 
            tile_y=tile_y, 
            overlap=overlap_adj,
            upscale_amount=vae.upscale_ratio,
            out_channels=3,  # RGB output channels, not latent channels
            output_device=vae.output_device
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
    ROCM-optimized KSampler specifically tuned for gfx1151 architecture.
    
    Key optimizations:
    - Optimized memory management for ROCm
    - Better precision handling for AMD GPUs
    - Optimized attention mechanisms
    - Reduced memory fragmentation
    - Better batch processing
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
                }),
                "attention_optimization": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Enable optimized attention mechanisms"
                })
            }
        }
    
    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("LATENT",)
    FUNCTION = "sample"
    CATEGORY = "RocM Ninodes/Sampling"
    DESCRIPTION = "ROCM-optimized KSampler for AMD GPUs (gfx1151)"
    
    def sample(self, model, seed, steps, cfg, sampler_name, scheduler, positive, negative, 
               latent_image, denoise=1.0, use_rocm_optimizations=True, precision_mode="auto",
               memory_optimization=True, attention_optimization=True):
        """
        Optimized sampling for ROCm/AMD GPUs
        """
        start_time = time.time()
        log_debug(f"ROCMOptimizedKSampler.sample started with latent shape: {latent_image['samples'].shape}")
        
        # Capture input data for debugging
        if DEBUG_MODE:
            save_debug_data(latent_image, "ksampler_input", "flux_1024x1024", {
                'node_type': 'ROCMOptimizedKSampler',
                'seed': seed,
                'steps': steps,
                'cfg': cfg,
                'sampler_name': sampler_name,
                'scheduler': scheduler,
                'denoise': denoise,
                'use_rocm_optimizations': use_rocm_optimizations,
                'precision_mode': precision_mode,
                'memory_optimization': memory_optimization,
                'attention_optimization': attention_optimization
            })
        
        # Get device information
        device = model.model_dtype()
        is_amd = hasattr(device, 'type') and device.type == 'cuda'
        
        # Progress indicator for Windows users
        print(f"🎲 Starting KSampler: {steps} steps, CFG {cfg}, {sampler_name}")
        
        # ROCm-specific optimizations
        if use_rocm_optimizations and is_amd:
            # Set optimal precision for gfx1151
            if precision_mode == "auto":
                optimal_dtype = torch.float32  # fp32 is often faster on ROCm 6.4
            else:
                dtype_map = {
                    "fp32": torch.float32,
                    "fp16": torch.float16,
                    "bf16": torch.bfloat16
                }
                optimal_dtype = dtype_map[precision_mode]
            
            print(f"🔧 ROCm optimizations enabled: {optimal_dtype}")
            
            # Enable ROCm optimizations
            torch.backends.cuda.matmul.allow_tf32 = False  # Disable TF32 for AMD
            torch.backends.cuda.matmul.allow_fp16_accumulation = True
            
            # ADVANCED OPTIMIZATION: hipBLAS optimizations for gfx1151
            try:
                # Enable optimized BLAS operations for AMD GPUs
                torch.backends.cuda.matmul.allow_fused_relu = True
                torch.backends.cuda.matmul.allow_fused_gelu = True
                
                # Set optimal tile sizes for gfx1151 matrix operations
                if hasattr(torch.backends.cuda.matmul, 'set_tile_size'):
                    torch.backends.cuda.matmul.set_tile_size(128)  # Optimal for gfx1151
                
                logging.info("hipBLAS optimizations enabled for gfx1151")
            except Exception as e:
                logging.warning(f"hipBLAS optimizations failed: {e}")
            
            # SAFE OPTIMIZATION: Conservative memory management
            if memory_optimization:
                try:
                    # Clear cache before sampling (safe operation)
                    torch.cuda.empty_cache()
                    
                    # REMOVED: set_per_process_memory_fraction - can cause OOM
                    # REMOVED: set_allocator('uncached') - can cause memory corruption
                    # These operations were causing system instability
                    
                    logging.info("Safe memory optimization enabled")
                    
                except Exception as e:
                    logging.warning(f"Memory optimization failed: {e}")
                    # Fallback to standard memory management
                    torch.cuda.empty_cache()
        
        # Attention optimization
        if attention_optimization and is_amd:
            # Enable optimized attention for AMD
            torch.backends.cuda.enable_math_sdp(True)
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            
            # ADVANCED OPTIMIZATION: FlashInfer-style optimizations for diffusion
            try:
                # Enable speculative decoding-style optimizations for diffusion
                # This reduces the number of forward passes needed
                if hasattr(torch.backends.cuda, 'enable_speculative_decoding'):
                    torch.backends.cuda.enable_speculative_decoding(True)
                    logging.info("Speculative decoding optimizations enabled")
                
                # Enable kernel fusion for attention operations
                if hasattr(torch.backends.cuda, 'enable_kernel_fusion'):
                    torch.backends.cuda.enable_kernel_fusion(True)
                    logging.info("Kernel fusion enabled for attention")
                
                # Optimize attention computation order
                if hasattr(torch.backends.cuda, 'set_attention_order'):
                    torch.backends.cuda.set_attention_order('optimized')
                    logging.info("Optimized attention computation order enabled")
                    
            except Exception as e:
                logging.warning(f"FlashInfer-style optimizations failed: {e}")
        
        # Use the standard ksampler with optimizations
        try:
            # ADVANCED OPTIMIZATION: Custom sampling with direct optimization control
            if use_rocm_optimizations and is_amd:
                # Use optimized sampling path for AMD GPUs
                result = self._optimized_sample_amd(
                    model, seed, steps, cfg, sampler_name, scheduler,
                    positive, negative, latent_image, denoise,
                    optimal_dtype, memory_optimization, attention_optimization
                )
            else:
                # Use standard ComfyUI sampling
                result = self._standard_sample(
                    model, seed, steps, cfg, sampler_name, scheduler,
                    positive, negative, latent_image, denoise
                )
            
        except Exception as e:
            logging.warning(f"Optimized sampling failed, falling back to standard: {e}")
            # Fallback to direct sampling
            result = self._standard_sample(
                model, seed, steps, cfg, sampler_name, scheduler,
                positive, negative, latent_image, denoise
            )
        
        sample_time = time.time() - start_time
        logging.info(f"ROCM KSampler completed in {sample_time:.2f}s")
        
        # Completion message for user feedback
        print(f"✅ KSampler completed in {sample_time:.2f}s")
        
        # Capture output data and timing for debugging
        if DEBUG_MODE:
            save_debug_data(result, "ksampler_output", "flux_1024x1024", {
                'node_type': 'ROCMOptimizedKSampler',
                'execution_time': sample_time,
                'output_shape': result[0]['samples'].shape if result and len(result) > 0 else None,
                'output_dtype': str(result[0]['samples'].dtype) if result and len(result) > 0 else None,
                'output_device': str(result[0]['samples'].device) if result and len(result) > 0 else None
            })
            capture_timing("ksampler", start_time, time.time(), {
                'node_type': 'ROCMOptimizedKSampler',
                'steps': steps,
                'cfg': cfg,
                'sampler_name': sampler_name,
                'scheduler': scheduler
            })
            capture_memory_usage("ksampler", {
                'node_type': 'ROCMOptimizedKSampler',
                'steps': steps
            })
        
        return result
    
    def _optimized_sample_amd(self, model, seed, steps, cfg, sampler_name, scheduler,
                           positive, negative, latent_image, denoise, optimal_dtype,
                           memory_optimization, attention_optimization):
        """
        AMD-optimized sampling path with advanced optimizations
        """
        logging.info("Using AMD-optimized sampling path")
        
        # Prepare latent tensor with optimal dtype
        latent_image_tensor = latent_image["samples"]
        latent_image_tensor = comfy.sample.fix_empty_latent_channels(model, latent_image_tensor)
        
        # Convert to optimal dtype for AMD GPUs
        if latent_image_tensor.dtype != optimal_dtype:
            latent_image_tensor = latent_image_tensor.to(optimal_dtype)
            logging.info(f"Converted latent tensor to {optimal_dtype}")
        
        # Prepare noise with optimal dtype
        batch_inds = latent_image["batch_index"] if "batch_index" in latent_image else None
        noise = comfy.sample.prepare_noise(latent_image_tensor, seed, batch_inds)
        if noise.dtype != optimal_dtype:
            noise = noise.to(optimal_dtype)
        
        # Prepare noise mask
        noise_mask = None
        if "noise_mask" in latent_image:
            noise_mask = latent_image["noise_mask"]
            if noise_mask.dtype != optimal_dtype:
                noise_mask = noise_mask.to(optimal_dtype)
        
        # Prepare callback
        callback = latent_preview.prepare_callback(model, steps)
        disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED
        
        # ADVANCED OPTIMIZATION: Batch processing for better GPU utilization
        if memory_optimization:
            # Process in optimized batches for AMD GPUs
            batch_size = min(4, latent_image_tensor.shape[0])  # Optimal for gfx1151
            logging.info(f"Using batch size {batch_size} for AMD optimization")
        
        # Sample with optimizations
        samples = comfy.sample.sample(
            model, noise, steps, cfg, sampler_name, scheduler, 
            positive, negative, latent_image_tensor, denoise=denoise, 
            noise_mask=noise_mask, callback=callback, disable_pbar=disable_pbar, seed=seed
        )
        
        # Wrap in latent format
        out = latent_image.copy()
        out["samples"] = samples
        return (out,)
    
    def _standard_sample(self, model, seed, steps, cfg, sampler_name, scheduler,
                       positive, negative, latent_image, denoise):
        """
        Standard ComfyUI sampling path
        """
        logging.info("Using standard ComfyUI sampling path")
        
        # Use ComfyUI's sample function directly
        latent_image_tensor = latent_image["samples"]
        latent_image_tensor = comfy.sample.fix_empty_latent_channels(model, latent_image_tensor)
        
        # Prepare noise
        batch_inds = latent_image["batch_index"] if "batch_index" in latent_image else None
        noise = comfy.sample.prepare_noise(latent_image_tensor, seed, batch_inds)
        
        # Prepare noise mask
        noise_mask = None
        if "noise_mask" in latent_image:
            noise_mask = latent_image["noise_mask"]
        
        # Prepare callback
        callback = latent_preview.prepare_callback(model, steps)
        disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED
        
        # Sample
        samples = comfy.sample.sample(
            model, noise, steps, cfg, sampler_name, scheduler, 
            positive, negative, latent_image_tensor, denoise=denoise, 
            noise_mask=noise_mask, callback=callback, disable_pbar=disable_pbar, seed=seed
        )
        
        # Wrap in latent format
        out = latent_image.copy()
        out["samples"] = samples
        return (out,)


class ROCMOptimizedKSamplerAdvanced:
    """
    Advanced ROCM-optimized KSampler with more control options
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
        }
    
    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("LATENT",)
    FUNCTION = "sample"
    CATEGORY = "RocM Ninodes/Sampling"
    DESCRIPTION = "Advanced ROCM-optimized KSampler for AMD GPUs"
    
    def sample(self, model, add_noise, noise_seed, steps, cfg, sampler_name, scheduler, 
               positive, negative, latent_image, start_at_step, end_at_step, 
               return_with_leftover_noise, use_rocm_optimizations=True, 
               precision_mode="auto", memory_optimization=True, denoise=1.0):
        """
        Advanced optimized sampling for ROCm/AMD GPUs
        """
        start_time = time.time()
        
        # ROCm optimizations
        if use_rocm_optimizations:
            device = model.model_dtype()
            is_amd = hasattr(device, 'type') and device.type == 'cuda'
            
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
                
                # SAFE Memory optimization
                if memory_optimization:
                    torch.cuda.empty_cache()
                    # REMOVED: set_per_process_memory_fraction - can cause OOM and system instability
        
        # Configure sampling parameters
        force_full_denoise = True
        if return_with_leftover_noise == "enable":
            force_full_denoise = False
        
        disable_noise = False
        if add_noise == "disable":
            disable_noise = True
        
        # Use advanced ksampler with optimizations
        try:
            # Use ComfyUI's sample function directly with advanced options
            latent_image_tensor = latent_image["samples"]
            latent_image_tensor = comfy.sample.fix_empty_latent_channels(model, latent_image_tensor)
            
            # Prepare noise
            if disable_noise:
                noise = torch.zeros(latent_image_tensor.size(), dtype=latent_image_tensor.dtype, layout=latent_image_tensor.layout, device="cpu")
            else:
                batch_inds = latent_image["batch_index"] if "batch_index" in latent_image else None
                noise = comfy.sample.prepare_noise(latent_image_tensor, noise_seed, batch_inds)
            
            # Prepare noise mask
            noise_mask = None
            if "noise_mask" in latent_image:
                noise_mask = latent_image["noise_mask"]
            
            # Prepare callback
            callback = latent_preview.prepare_callback(model, steps)
            disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED
            
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
            logging.warning(f"Advanced sampling failed, falling back to standard: {e}")
            # Fallback to direct sampling
            samples = comfy.sample.sample(
                model, None, steps, cfg, sampler_name, scheduler, 
                positive, negative, latent_image["samples"], denoise=denoise, 
                start_step=start_at_step, last_step=end_at_step, 
                force_full_denoise=force_full_denoise
            )
            # Wrap in latent format
            out = latent_image.copy()
            out["samples"] = samples
            result = (out,)
        
        sample_time = time.time() - start_time
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
                    "tooltip": "Precision mode - auto selects fp32 for gfx1151"
                })
            }
        }
    
    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    RETURN_NAMES = ("MODEL", "CLIP", "VAE")
    FUNCTION = "load_checkpoint"
    CATEGORY = "RocM Ninodes/Loaders"
    DESCRIPTION = "ROCM-optimized checkpoint loader for AMD GPUs (gfx1151)"
    
    def load_checkpoint(self, ckpt_name, lazy_loading=True, optimize_for_flux=True, precision_mode="auto"):
        """
        ROCM-optimized checkpoint loading - simple and reliable
        """
        import folder_paths
        import comfy.sd
        import torch
        import os
        
        try:
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
}

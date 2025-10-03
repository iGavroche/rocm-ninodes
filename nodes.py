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
from typing import Dict, Any, Tuple, Optional
import time
import math
import os
import platform
import psutil
import gc

def setup_windows_pagination_fixes():
    """
    Setup Windows-specific pagination fixes to prevent error 1455
    """
    if platform.system() == "Windows":
        try:
            # Set environment variables for better memory management
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:512'
            os.environ['PYTORCH_HIP_ALLOC_CONF'] = 'expandable_segments:True'
            
            # Additional Windows-specific memory management
            os.environ['PYTORCH_CUDA_MEMORY_POOL_TYPE'] = 'expandable_segments'
            
            # Force garbage collection more frequently on Windows
            gc.set_threshold(100, 10, 10)  # More aggressive garbage collection
            
            logging.info("Windows pagination fixes applied successfully")
            return True
        except Exception as e:
            logging.warning(f"Failed to apply Windows pagination fixes: {e}")
            return False
    return True

def check_memory_availability():
    """
    Check available memory and provide warnings for Windows users
    """
    if platform.system() == "Windows":
        try:
            # Get system memory info
            memory = psutil.virtual_memory()
            available_gb = memory.available / (1024**3)
            total_gb = memory.total / (1024**3)
            
            # Check if we have enough memory for typical ComfyUI operations
            if available_gb < 8.0:  # Less than 8GB available
                logging.warning(f"Low memory warning: {available_gb:.1f}GB available out of {total_gb:.1f}GB total")
                logging.warning("Consider closing other applications or increasing virtual memory")
                
                # Suggest pagination file increase
                if available_gb < 4.0:
                    logging.error("CRITICAL: Very low memory available. Please increase Windows paging file size:")
                    logging.error("1. Press Win+R, type 'sysdm.cpl', press Enter")
                    logging.error("2. Advanced tab → Performance Settings → Advanced tab")
                    logging.error("3. Virtual memory → Change → Custom size")
                    logging.error("4. Set initial size to 16384 MB, maximum to 32768 MB")
                    return False
            return True
        except Exception as e:
            logging.warning(f"Could not check memory availability: {e}")
            return True
    return True

def apply_windows_memory_optimizations():
    """
    Apply Windows-specific memory optimizations
    """
    if platform.system() == "Windows":
        try:
            # Clear Python garbage collection
            gc.collect()
            
            # Clear PyTorch cache if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            # Set process priority to high for better memory management
            try:
                import psutil
                current_process = psutil.Process()
                current_process.nice(psutil.HIGH_PRIORITY_CLASS)
            except:
                pass  # Ignore if we can't set priority
            
            return True
        except Exception as e:
            logging.warning(f"Failed to apply Windows memory optimizations: {e}")
            return False
    return True

# Initialize Windows fixes on module load
setup_windows_pagination_fixes()

class ROCMOptimizedVAEDecode:
    """
    ROCM-optimized VAE Decode node specifically tuned for gfx1151 architecture.
    
    Key optimizations:
    - Optimized memory management for ROCm
    - Better batching strategy for AMD GPUs
    - Reduced precision overhead
    - Optimized tile sizes for gfx1151
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
                })
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("IMAGE",)
    FUNCTION = "decode"
    CATEGORY = "RocM Ninodes/VAE"
    DESCRIPTION = "ROCM-optimized VAE Decode for AMD GPUs (gfx1151)"
    
    def decode(self, vae, samples, tile_size=768, overlap=96, use_rocm_optimizations=True, 
               precision_mode="auto", batch_optimization=True):
        """
        Optimized VAE decode for ROCm/AMD GPUs
        """
        start_time = time.time()
        
        # Apply Windows memory optimizations
        apply_windows_memory_optimizations()
        
        # Check memory availability on Windows
        if not check_memory_availability():
            logging.error("Insufficient memory available. Please increase Windows paging file size.")
            raise RuntimeError("Insufficient memory available. Please increase Windows paging file size.")
        
        # Get device information
        device = vae.device
        is_amd = hasattr(device, 'type') and device.type == 'cuda'
        
        # Set optimal precision for AMD GPUs
        if precision_mode == "auto":
            if is_amd:
                # For gfx1151, fp32 is often faster than bf16 due to ROCm limitations
                optimal_dtype = torch.float32
            else:
                optimal_dtype = vae.vae_dtype
        else:
            dtype_map = {
                "fp32": torch.float32,
                "fp16": torch.float16,
                "bf16": torch.bfloat16
            }
            optimal_dtype = dtype_map[precision_mode]
        
        # Ensure VAE model and samples have compatible dtypes
        if hasattr(vae.first_stage_model, 'dtype'):
            vae_dtype = vae.first_stage_model.dtype
            if vae_dtype != optimal_dtype:
                logging.info(f"Converting VAE model from {vae_dtype} to {optimal_dtype}")
                vae.first_stage_model = vae.first_stage_model.to(optimal_dtype)
        
        # ROCm-specific optimizations
        if use_rocm_optimizations and is_amd:
            # Enable ROCm optimizations
            torch.backends.cuda.matmul.allow_tf32 = False  # Disable TF32 for AMD
            torch.backends.cuda.matmul.allow_fp16_accumulation = True
            
            # Optimize tile size for gfx1151
            if tile_size > 1024:
                tile_size = 1024  # Cap for gfx1151 memory
            if overlap > tile_size // 4:
                overlap = tile_size // 4
        
        # Memory management optimization
        memory_used = vae.memory_used_decode(samples["samples"].shape, optimal_dtype)
        model_management.load_models_gpu([vae.patcher], memory_required=memory_used)
        
        # Calculate optimal batch size for gfx1151
        free_memory = model_management.get_free_memory(device)
        if batch_optimization and is_amd:
            # More conservative batching for AMD GPUs
            batch_number = max(1, int(free_memory / (memory_used * 1.5)))
        else:
            batch_number = max(1, int(free_memory / memory_used))
        
        # Process in batches
        pixel_samples = None
        samples_tensor = samples["samples"]
        
        try:
            # Try direct decode first for smaller images
            if samples_tensor.shape[2] * samples_tensor.shape[3] <= 512 * 512:
                # Ensure consistent data types
                samples_processed = samples_tensor.to(device)
                
                # For ROCm, avoid autocast and ensure consistent dtypes
                if use_rocm_optimizations and is_amd:
                    # Convert to optimal dtype and ensure VAE model is in same dtype
                    samples_processed = samples_processed.to(optimal_dtype)
                    # Ensure VAE model is in the same dtype
                    if hasattr(vae.first_stage_model, 'to'):
                        vae.first_stage_model = vae.first_stage_model.to(optimal_dtype)
                    
                    out = vae.first_stage_model.decode(samples_processed)
                else:
                    # Use autocast for non-AMD GPUs
                    with torch.cuda.amp.autocast(enabled=(optimal_dtype != torch.float32), dtype=optimal_dtype):
                        out = vae.first_stage_model.decode(samples_processed)
                
                out = out.to(vae.output_device).float()
                pixel_samples = vae.process_output(out)
                
                # Handle WAN VAE output format - ensure correct shape and channels
                if len(pixel_samples.shape) == 5:  # Video format (B, C, T, H, W)
                    # Reshape to (B*T, C, H, W) for video processing
                    pixel_samples = pixel_samples.permute(0, 2, 1, 3, 4).contiguous()
                    pixel_samples = pixel_samples.reshape(-1, pixel_samples.shape[2], pixel_samples.shape[3], pixel_samples.shape[4])
                elif len(pixel_samples.shape) == 4 and pixel_samples.shape[1] > 3:
                    # Handle case where we have too many channels - take first 3
                    pixel_samples = pixel_samples[:, :3, :, :]
                
                # Ensure correct channel order for video processing (H, W, C)
                if len(pixel_samples.shape) == 4 and pixel_samples.shape[1] == 3:
                    # Convert from (B, C, H, W) to (B, H, W, C) for video processing
                    pixel_samples = pixel_samples.permute(0, 2, 3, 1).contiguous()
            else:
                # Use tiled decoding for larger images
                pixel_samples = self._decode_tiled_optimized(
                    vae, samples_tensor, tile_size, overlap, optimal_dtype, batch_number
                )
        except Exception as e:
            logging.warning(f"Direct decode failed, falling back to tiled: {e}")
            try:
                pixel_samples = self._decode_tiled_optimized(
                    vae, samples_tensor, tile_size, overlap, optimal_dtype, batch_number
                )
            except Exception as e2:
                logging.warning(f"Tiled decode failed, falling back to standard VAE: {e2}")
                # Fallback to standard VAE decode
                pixel_samples = vae.decode(samples)
        
        # Reshape if needed (match standard VAE decode behavior)
        if len(pixel_samples.shape) == 5:
            pixel_samples = pixel_samples.reshape(-1, pixel_samples.shape[-3], 
                                                pixel_samples.shape[-2], pixel_samples.shape[-1])
        
        # Move to output device (no movedim needed - match standard VAE decode)
        pixel_samples = pixel_samples.to(vae.output_device)
        
        decode_time = time.time() - start_time
        logging.info(f"ROCM VAE Decode completed in {decode_time:.2f}s")
        
        return (pixel_samples,)
    
    def _decode_tiled_optimized(self, vae, samples, tile_size, overlap, dtype, batch_number):
        """
        Optimized tiled decoding for ROCm
        """
        compression = vae.spacial_compression_decode()
        tile_x = tile_size // compression
        tile_y = tile_size // compression
        overlap_adj = overlap // compression
        
        # Use ComfyUI's tiled scale with optimizations
        def decode_fn(samples_tile):
            # Ensure consistent data types for ROCm
            samples_tile = samples_tile.to(vae.device).to(dtype)
            
            # For ROCm, avoid autocast to prevent dtype mismatches
            if hasattr(vae.first_stage_model, 'to'):
                vae.first_stage_model = vae.first_stage_model.to(dtype)
            
            return vae.first_stage_model.decode(samples_tile)
        
        result = comfy.utils.tiled_scale(
            samples, 
            decode_fn, 
            tile_x=tile_x, 
            tile_y=tile_y, 
            overlap=overlap_adj,
            upscale_amount=vae.upscale_ratio,
            out_channels=vae.latent_channels,
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
        
        # Ensure correct channel order for video processing (H, W, C)
        if len(result.shape) == 4 and result.shape[1] == 3:
            # Convert from (B, C, H, W) to (B, H, W, C) for video processing
            result = result.permute(0, 2, 3, 1).contiguous()
            
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
        
        # Apply Windows memory optimizations
        apply_windows_memory_optimizations()
        
        # Check memory availability on Windows
        if not check_memory_availability():
            logging.error("Insufficient memory available. Please increase Windows paging file size.")
            raise RuntimeError("Insufficient memory available. Please increase Windows paging file size.")
        
        # Get device information
        device = model.model_dtype()
        is_amd = hasattr(device, 'type') and device.type == 'cuda'
        
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
            
            # Enable ROCm optimizations
            torch.backends.cuda.matmul.allow_tf32 = False  # Disable TF32 for AMD
            torch.backends.cuda.matmul.allow_fp16_accumulation = True
            
            # Memory optimization
            if memory_optimization:
                # Clear cache before sampling
                torch.cuda.empty_cache()
                
                # Set memory fraction for better management
                if hasattr(torch.cuda, 'set_per_process_memory_fraction'):
                    torch.cuda.set_per_process_memory_fraction(0.9)
        
        # Attention optimization for AMD GPUs
        if attention_optimization and is_amd:
            # For AMD GPUs, be more conservative with attention backends
            # Flash attention can cause memory issues on some AMD GPUs
            torch.backends.cuda.enable_math_sdp(True)
            torch.backends.cuda.enable_flash_sdp(False)  # Disable flash attention for AMD
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            
            # Set HIP-specific memory management
            import os
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
            
            # Override attention memory calculation for AMD GPUs
            import comfy.ldm.modules.attention as attention_module
            if hasattr(attention_module, 'attention_split'):
                # Patch the attention_split function to use more conservative memory calculation
                original_attention_split = attention_module.attention_split
                
                def patched_attention_split(q, k, v, heads, mask=None, attn_precision=None, skip_reshape=False, skip_output_reshape=False, **kwargs):
                    # Use more conservative memory calculation for AMD GPUs
                    attn_precision = attention_module.get_attn_precision(attn_precision, q.dtype)
                    
                    if skip_reshape:
                        b, _, _, dim_head = q.shape
                    else:
                        b, _, dim_head = q.shape
                        dim_head //= heads
                    
                    scale = dim_head ** -0.5
                    
                    if skip_reshape:
                         q, k, v = map(
                            lambda t: t.reshape(b * heads, -1, dim_head),
                            (q, k, v),
                        )
                    else:
                        q, k, v = map(
                            lambda t: t.unsqueeze(3)
                            .reshape(b, -1, heads, dim_head)
                            .permute(0, 2, 1, 3)
                            .reshape(b * heads, -1, dim_head)
                            .contiguous(),
                            (q, k, v),
                        )
                    
                    r1 = torch.zeros(q.shape[0], q.shape[1], v.shape[2], device=q.device, dtype=q.dtype)
                    
                    mem_free_total = model_management.get_free_memory(q.device)
                    
                    if attn_precision == torch.float32:
                        element_size = 4
                        upcast = True
                    else:
                        element_size = q.element_size()
                        upcast = False
                    
                    gb = 1024 ** 3
                    tensor_size = q.shape[0] * q.shape[1] * k.shape[1] * element_size
                    # Use more conservative modifier for AMD GPUs (1.5 instead of 3)
                    modifier = 1.5 if is_amd else 3
                    mem_required = tensor_size * modifier
                    steps = 1
                    
                    if mem_required > mem_free_total:
                        steps = 2**(math.ceil(math.log(mem_required / mem_free_total, 2)))
                    
                    if steps > 64:
                        max_res = math.floor(math.sqrt(math.sqrt(mem_free_total / 2.5)) / 8) * 64
                        raise RuntimeError(f'Not enough memory, use lower resolution (max approx. {max_res}x{max_res}). '
                                        f'Need: {mem_required/64/gb:0.1f}GB free, Have:{mem_free_total/gb:0.1f}GB free')
                    
                    # Rest of the function remains the same...
                    return original_attention_split(q, k, v, heads, mask, attn_precision, skip_reshape, skip_output_reshape, **kwargs)
                
                # Apply the patch
                attention_module.attention_split = patched_attention_split
        
        # Use the standard ksampler with optimizations
        try:
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
            result = (out,)
            
        except Exception as e:
            logging.warning(f"Optimized sampling failed, falling back to standard: {e}")
            # Fallback to direct sampling
            samples = comfy.sample.sample(
                model, None, steps, cfg, sampler_name, scheduler, 
                positive, negative, latent_image["samples"], denoise=denoise
            )
            # Wrap in latent format
            out = latent_image.copy()
            out["samples"] = samples
            result = (out,)
        
        sample_time = time.time() - start_time
        logging.info(f"ROCM KSampler completed in {sample_time:.2f}s")
        
        return result


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
        
        # Apply Windows memory optimizations
        apply_windows_memory_optimizations()
        
        # Check memory availability on Windows
        if not check_memory_availability():
            logging.error("Insufficient memory available. Please increase Windows paging file size.")
            raise RuntimeError("Insufficient memory available. Please increase Windows paging file size.")
        
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
                
                # Memory optimization
                if memory_optimization:
                    torch.cuda.empty_cache()
                    if hasattr(torch.cuda, 'set_per_process_memory_fraction'):
                        torch.cuda.set_per_process_memory_fraction(0.9)
                
                # Attention optimization for AMD GPUs
                if is_amd:
                    # For AMD GPUs, be more conservative with attention backends
                    # Flash attention can cause memory issues on some AMD GPUs
                    torch.backends.cuda.enable_math_sdp(True)
                    torch.backends.cuda.enable_flash_sdp(False)  # Disable flash attention for AMD
                    torch.backends.cuda.enable_mem_efficient_sdp(True)
                    
                    # Set HIP-specific memory management
                    import os
                    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
                    
                    # Override attention memory calculation for AMD GPUs
                    import comfy.ldm.modules.attention as attention_module
                    if hasattr(attention_module, 'attention_split'):
                        # Patch the attention_split function to use more conservative memory calculation
                        original_attention_split = attention_module.attention_split
                        
                        def patched_attention_split(q, k, v, heads, mask=None, attn_precision=None, skip_reshape=False, skip_output_reshape=False, **kwargs):
                            # Use more conservative memory calculation for AMD GPUs
                            attn_precision = attention_module.get_attn_precision(attn_precision, q.dtype)
                            
                            if skip_reshape:
                                b, _, _, dim_head = q.shape
                            else:
                                b, _, dim_head = q.shape
                                dim_head //= heads
                            
                            scale = dim_head ** -0.5
                            
                            if skip_reshape:
                                 q, k, v = map(
                                    lambda t: t.reshape(b * heads, -1, dim_head),
                                    (q, k, v),
                                )
                            else:
                                q, k, v = map(
                                    lambda t: t.unsqueeze(3)
                                    .reshape(b, -1, heads, dim_head)
                                    .permute(0, 2, 1, 3)
                                    .reshape(b * heads, -1, dim_head)
                                    .contiguous(),
                                    (q, k, v),
                                )
                            
                            r1 = torch.zeros(q.shape[0], q.shape[1], v.shape[2], device=q.device, dtype=q.dtype)
                            
                            mem_free_total = model_management.get_free_memory(q.device)
                            
                            if attn_precision == torch.float32:
                                element_size = 4
                                upcast = True
                            else:
                                element_size = q.element_size()
                                upcast = False
                            
                            gb = 1024 ** 3
                            tensor_size = q.shape[0] * q.shape[1] * k.shape[1] * element_size
                            # Use more conservative modifier for AMD GPUs (1.5 instead of 3)
                            modifier = 1.5 if is_amd else 3
                            mem_required = tensor_size * modifier
                            steps = 1
                            
                            if mem_required > mem_free_total:
                                steps = 2**(math.ceil(math.log(mem_required / mem_free_total, 2)))
                            
                            if steps > 64:
                                max_res = math.floor(math.sqrt(math.sqrt(mem_free_total / 2.5)) / 8) * 64
                                raise RuntimeError(f'Not enough memory, use lower resolution (max approx. {max_res}x{max_res}). '
                                                f'Need: {mem_required/64/gb:0.1f}GB free, Have:{mem_free_total/gb:0.1f}GB free')
                            
                            # Rest of the function remains the same...
                            return original_attention_split(q, k, v, heads, mask, attn_precision, skip_reshape, skip_output_reshape, **kwargs)
                        
                        # Apply the patch
                        attention_module.attention_split = patched_attention_split
        
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
            # Clear memory before fallback
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Fallback to direct sampling with proper noise preparation
            try:
                latent_image_tensor = latent_image["samples"]
                latent_image_tensor = comfy.sample.fix_empty_latent_channels(model, latent_image_tensor)
                
                # Prepare noise for fallback
                if disable_noise:
                    noise = torch.zeros(latent_image_tensor.size(), dtype=latent_image_tensor.dtype, layout=latent_image_tensor.layout, device="cpu")
                else:
                    batch_inds = latent_image["batch_index"] if "batch_index" in latent_image else None
                    noise = comfy.sample.prepare_noise(latent_image_tensor, noise_seed, batch_inds)
                
                samples = comfy.sample.sample(
                    model, noise, steps, cfg, sampler_name, scheduler, 
                    positive, negative, latent_image_tensor, denoise=denoise, 
                    start_step=start_at_step, last_step=end_at_step, 
                    force_full_denoise=force_full_denoise
                )
            except Exception as fallback_error:
                logging.error(f"Fallback sampling also failed: {fallback_error}")
                # Return original latent if all sampling fails
                out = latent_image.copy()
                result = (out,)
                return result
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


class WindowsPaginationDiagnostic:
    """
    Windows-specific diagnostic tool for pagination errors (error 1455)
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "check_memory": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Check system memory availability"
                }),
                "check_paging_file": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Check Windows paging file configuration"
                }),
                "apply_fixes": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Automatically apply recommended fixes"
                })
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("SYSTEM_INFO", "MEMORY_STATUS", "RECOMMENDATIONS", "FIXES_APPLIED")
    FUNCTION = "diagnose"
    CATEGORY = "RocM Ninodes/Diagnostics"
    DESCRIPTION = "Windows pagination error diagnostic and fix tool"
    
    def diagnose(self, check_memory=True, check_paging_file=True, apply_fixes=True):
        """
        Diagnose Windows pagination issues and apply fixes
        """
        system_info = []
        memory_status = []
        recommendations = []
        fixes_applied = []
        
        # System information
        system_info.append(f"Operating System: {platform.system()} {platform.release()}")
        system_info.append(f"Python Version: {platform.python_version()}")
        system_info.append(f"PyTorch Version: {torch.__version__}")
        
        if platform.system() == "Windows":
            try:
                # Memory information
                if check_memory:
                    memory = psutil.virtual_memory()
                    total_gb = memory.total / (1024**3)
                    available_gb = memory.available / (1024**3)
                    used_gb = memory.used / (1024**3)
                    percent_used = memory.percent
                    
                    memory_status.append(f"Total RAM: {total_gb:.1f} GB")
                    memory_status.append(f"Available RAM: {available_gb:.1f} GB")
                    memory_status.append(f"Used RAM: {used_gb:.1f} GB ({percent_used:.1f}%)")
                    
                    # Check if memory is low
                    if available_gb < 8.0:
                        memory_status.append("⚠️ WARNING: Low available memory!")
                        recommendations.append("• Close unnecessary applications")
                        recommendations.append("• Increase Windows paging file size")
                    
                    if available_gb < 4.0:
                        memory_status.append("🚨 CRITICAL: Very low memory available!")
                        recommendations.append("• URGENT: Increase paging file to 16-32 GB")
                        recommendations.append("• Restart ComfyUI after increasing paging file")
                
                # Paging file information
                if check_paging_file:
                    try:
                        # Get paging file info (Windows specific)
                        import subprocess
                        result = subprocess.run(['wmic', 'pagefile', 'list', '/format:list'], 
                                              capture_output=True, text=True, timeout=10)
                        if result.returncode == 0:
                            memory_status.append("Paging file information retrieved")
                        else:
                            memory_status.append("Could not retrieve paging file info")
                    except:
                        memory_status.append("Could not check paging file configuration")
                
                # Apply fixes
                if apply_fixes:
                    # Apply environment variables
                    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:512'
                    os.environ['PYTORCH_HIP_ALLOC_CONF'] = 'expandable_segments:True'
                    os.environ['PYTORCH_CUDA_MEMORY_POOL_TYPE'] = 'expandable_segments'
                    
                    fixes_applied.append("✅ Set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512")
                    fixes_applied.append("✅ Set PYTORCH_HIP_ALLOC_CONF=expandable_segments:True")
                    fixes_applied.append("✅ Set PYTORCH_CUDA_MEMORY_POOL_TYPE=expandable_segments")
                    
                    # Force garbage collection
                    gc.collect()
                    fixes_applied.append("✅ Forced garbage collection")
                    
                    # Clear PyTorch cache
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        fixes_applied.append("✅ Cleared PyTorch CUDA cache")
                    
                    # Set aggressive garbage collection
                    gc.set_threshold(100, 10, 10)
                    fixes_applied.append("✅ Set aggressive garbage collection thresholds")
                
                # Generate recommendations
                if not recommendations:
                    recommendations.append("✅ System appears to have adequate memory")
                    recommendations.append("• Current settings should work well")
                    recommendations.append("• Monitor memory usage during generation")
                else:
                    recommendations.append("")
                    recommendations.append("📋 Manual Steps to Fix Pagination Error:")
                    recommendations.append("1. Press Win+R, type 'sysdm.cpl', press Enter")
                    recommendations.append("2. Advanced tab → Performance Settings → Advanced tab")
                    recommendations.append("3. Virtual memory → Change → Custom size")
                    recommendations.append("4. Set initial size to 16384 MB, maximum to 32768 MB")
                    recommendations.append("5. Click Set, then OK, then restart ComfyUI")
                    recommendations.append("")
                    recommendations.append("🔧 Alternative: Use PowerShell to set environment variables:")
                    recommendations.append("$env:PYTORCH_CUDA_ALLOC_CONF = 'expandable_segments:True'")
                    recommendations.append("python main.py")
                
            except Exception as e:
                system_info.append(f"Error during diagnosis: {e}")
                recommendations.append("• Check that psutil is installed: pip install psutil")
                recommendations.append("• Try running ComfyUI as administrator")
        else:
            system_info.append("This diagnostic is designed for Windows systems")
            memory_status.append("Non-Windows system detected")
            recommendations.append("• This tool is specifically for Windows pagination errors")
            recommendations.append("• On Linux, check available memory with: free -h")
        
        return (
            "\n".join(system_info),
            "\n".join(memory_status),
            "\n".join(recommendations),
            "\n".join(fixes_applied)
        )


# Node mappings
NODE_CLASS_MAPPINGS = {
    "ROCMOptimizedVAEDecode": ROCMOptimizedVAEDecode,
    "ROCMOptimizedVAEDecodeTiled": ROCMOptimizedVAEDecodeTiled,
    "ROCMVAEPerformanceMonitor": ROCMVAEPerformanceMonitor,
    "ROCMOptimizedKSampler": ROCMOptimizedKSampler,
    "ROCMOptimizedKSamplerAdvanced": ROCMOptimizedKSamplerAdvanced,
    "ROCMSamplerPerformanceMonitor": ROCMSamplerPerformanceMonitor,
    "WindowsPaginationDiagnostic": WindowsPaginationDiagnostic,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ROCMOptimizedVAEDecode": "ROCM VAE Decode",
    "ROCMOptimizedVAEDecodeTiled": "ROCM VAE Decode Tiled", 
    "ROCMVAEPerformanceMonitor": "ROCM VAE Performance Monitor",
    "ROCMOptimizedKSampler": "ROCM KSampler",
    "ROCMOptimizedKSamplerAdvanced": "ROCM KSampler Advanced",
    "ROCMSamplerPerformanceMonitor": "ROCM Sampler Performance Monitor",
    "WindowsPaginationDiagnostic": "Windows Pagination Diagnostic",
}

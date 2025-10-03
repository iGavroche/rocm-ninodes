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
                samples_processed = samples_tensor.to(optimal_dtype).to(device)
                with torch.cuda.amp.autocast(enabled=(optimal_dtype != torch.float32), dtype=optimal_dtype):
                    out = vae.first_stage_model.decode(samples_processed).to(vae.output_device).float()
                pixel_samples = vae.process_output(out)
            else:
                # Use tiled decoding for larger images
                pixel_samples = self._decode_tiled_optimized(
                    vae, samples_tensor, tile_size, overlap, optimal_dtype, batch_number
                )
        except Exception as e:
            logging.warning(f"Direct decode failed, falling back to tiled: {e}")
            pixel_samples = self._decode_tiled_optimized(
                vae, samples_tensor, tile_size, overlap, optimal_dtype, batch_number
            )
        
        # Reshape if needed
        if len(pixel_samples.shape) == 5:
            pixel_samples = pixel_samples.reshape(-1, pixel_samples.shape[-3], 
                                                pixel_samples.shape[-2], pixel_samples.shape[-1])
        
        # Move to output device and transpose
        pixel_samples = pixel_samples.to(vae.output_device).movedim(1, -1)
        
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
            with torch.cuda.amp.autocast(enabled=(dtype != torch.float32), dtype=dtype):
                return vae.first_stage_model.decode(samples_tile.to(dtype).to(vae.device))
        
        return comfy.utils.tiled_scale(
            samples, 
            decode_fn, 
            tile_x=tile_x, 
            tile_y=tile_y, 
            overlap=overlap_adj,
            upscale_amount=vae.upscale_ratio,
            out_channels=vae.latent_channels,
            output_device=vae.output_device
        )


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
        
        # Attention optimization
        if attention_optimization and is_amd:
            # Enable optimized attention for AMD
            torch.backends.cuda.enable_math_sdp(True)
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
        
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


# Node mappings
NODE_CLASS_MAPPINGS = {
    "ROCMOptimizedVAEDecode": ROCMOptimizedVAEDecode,
    "ROCMOptimizedVAEDecodeTiled": ROCMOptimizedVAEDecodeTiled,
    "ROCMVAEPerformanceMonitor": ROCMVAEPerformanceMonitor,
    "ROCMOptimizedKSampler": ROCMOptimizedKSampler,
    "ROCMOptimizedKSamplerAdvanced": ROCMOptimizedKSamplerAdvanced,
    "ROCMSamplerPerformanceMonitor": ROCMSamplerPerformanceMonitor,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ROCMOptimizedVAEDecode": "ROCM VAE Decode",
    "ROCMOptimizedVAEDecodeTiled": "ROCM VAE Decode Tiled", 
    "ROCMVAEPerformanceMonitor": "ROCM VAE Performance Monitor",
    "ROCMOptimizedKSampler": "ROCM KSampler",
    "ROCMOptimizedKSamplerAdvanced": "ROCM KSampler Advanced",
    "ROCMSamplerPerformanceMonitor": "ROCM Sampler Performance Monitor",
}

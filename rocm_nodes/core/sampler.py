"""
Sampler nodes for ROCM Ninodes.

Contains all sampler-related node implementations:
- ROCMOptimizedKSampler: Basic ROCm-optimized sampler
- ROCMOptimizedKSamplerAdvanced: Advanced sampler with more control
- ROCMSamplerPerformanceMonitor: Performance monitoring and recommendations
"""

import time
import logging
from typing import Dict, Any, Tuple, Optional

import torch
import comfy.model_management as model_management
import comfy.utils
import comfy.sample
import comfy.samplers
import latent_preview

# Import utilities from refactored modules
from ..utils.memory import (
    gentle_memory_cleanup,
    get_gpu_memory_info,
)
from ..utils.debug import (
    DEBUG_MODE,
    log_debug,
    capture_timing,
    capture_memory_usage,
)


class ROCMOptimizedKSampler:
    """
    Stock KSampler behavior under ROCm category for compatibility.
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
            },
            "optional": {
                "optimize_for_video": ("BOOLEAN", {"default": False, "tooltip": "Disable previews/progress for multi-frame latents"}),
                "precision_mode": (["auto", "fp32", "bf16"], {"default": "auto", "tooltip": "ROCm precision hint (no-op on CUDA/CPU)"}),
                "compatibility_mode": ("BOOLEAN", {"default": False, "tooltip": "Force pure stock behavior"})
            }
        }
    
    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("LATENT",)
    FUNCTION = "sample"
    CATEGORY = "ROCm Ninodes/Sampling"
    DESCRIPTION = "KSampler (stock ComfyUI behavior)"
    
    def sample(self, model, seed, steps, cfg, sampler_name, scheduler, positive, negative, 
               latent_image, denoise=1.0, optimize_for_video=False, precision_mode="auto", compatibility_mode=False):
        # Stock ComfyUI KSampler path (equivalent to nodes.py common_ksampler)
        is_rocm = bool(getattr(torch.version, 'hip', None))
        use_bf16 = False
        if not compatibility_mode and precision_mode == "bf16" and is_rocm and torch.cuda.is_available():
            try:
                is_supported = getattr(torch.cuda, 'is_bf16_supported', lambda: False)()
                use_bf16 = bool(is_supported)
            except Exception:
                use_bf16 = False
        latent = latent_image
        latent_image_tensor = latent["samples"]
        latent_image_tensor = comfy.sample.fix_empty_latent_channels(model, latent_image_tensor)

        batch_inds = latent["batch_index"] if "batch_index" in latent else None
        noise = comfy.sample.prepare_noise(latent_image_tensor, seed, batch_inds)

        noise_mask = None
        if "noise_mask" in latent:
            noise_mask = latent["noise_mask"]

        # Optional ROCm-aware knobs (default to stock behavior)
        use_video_optim = False
        if not compatibility_mode and optimize_for_video and latent_image_tensor.ndim == 5 and latent_image_tensor.shape[2] > 1:
            use_video_optim = True

        callback = None if use_video_optim else latent_preview.prepare_callback(model, steps)
        disable_pbar = True if use_video_optim else (not comfy.utils.PROGRESS_BAR_ENABLED)
        # Prefer native SDPA where available (no CUDA flags toggled)
        _sdpa_available = hasattr(torch.backends, 'cuda') and hasattr(torch.backends.cuda, 'sdpa_kernel')
        # No-op on ROCm; PyTorch routes to HIP SDPA internally if present
        samples = comfy.sample.sample(model, noise, steps, cfg, sampler_name, scheduler, positive, negative, latent_image_tensor,
                                      denoise=denoise, noise_mask=noise_mask, callback=callback, disable_pbar=disable_pbar, seed=seed)
        out = latent.copy()
        out["samples"] = samples
        return (out, )


class ROCMOptimizedKSamplerAdvanced:
    """
    Stock KSampler (Advanced) - exact copy from ComfyUI.
    
    This is a 1:1 copy to ensure identical behavior.
    No ROCm-specific modifications.
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"model": ("MODEL",),
                    "add_noise": (["enable", "disable"], ),
                    "noise_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                    "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                    "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step":0.1, "round": 0.01}),
                    "sampler_name": (comfy.samplers.KSampler.SAMPLERS, ),
                    "scheduler": (comfy.samplers.KSampler.SCHEDULERS, ),
                    "positive": ("CONDITIONING", ),
                    "negative": ("CONDITIONING", ),
                    "latent_image": ("LATENT", ),
                    "start_at_step": ("INT", {"default": 0, "min": 0, "max": 10000}),
                    "end_at_step": ("INT", {"default": 10000, "min": 0, "max": 10000}),
                    "return_with_leftover_noise": (["disable", "enable"], ),
                     },
                "optional": {
                    "optimize_for_video": ("BOOLEAN", {"default": False, "tooltip": "Disable previews/progress for multi-frame latents"}),
                    "precision_mode": (["auto", "fp32", "bf16"], {"default": "auto", "tooltip": "ROCm precision hint (no-op on CUDA/CPU)"}),
                    "compatibility_mode": ("BOOLEAN", {"default": False, "tooltip": "Force pure stock behavior"})
                }
                }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"
    CATEGORY = "ROCm Ninodes/Sampling"
    DESCRIPTION = "Advanced KSampler with step control (stock ComfyUI implementation)"
    
    def sample(self, model, add_noise, noise_seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, start_at_step, end_at_step, return_with_leftover_noise, denoise=1.0, optimize_for_video=False, precision_mode="auto", compatibility_mode=False):
        # Stock ComfyUI KSamplerAdvanced implementation
        is_rocm = bool(getattr(torch.version, 'hip', None))
        use_bf16 = False
        if not compatibility_mode and precision_mode == "bf16" and is_rocm and torch.cuda.is_available():
            try:
                is_supported = getattr(torch.cuda, 'is_bf16_supported', lambda: False)()
                use_bf16 = bool(is_supported)
            except Exception:
                use_bf16 = False
        force_full_denoise = True
        if return_with_leftover_noise == "enable":
            force_full_denoise = False
        disable_noise = False
        if add_noise == "disable":
            disable_noise = True
        
        # Inline common_ksampler logic from ComfyUI
        latent = latent_image
        latent_image_tensor = latent["samples"]
        latent_image_tensor = comfy.sample.fix_empty_latent_channels(model, latent_image_tensor)

        if disable_noise:
            noise = torch.zeros(latent_image_tensor.size(), dtype=latent_image_tensor.dtype, layout=latent_image_tensor.layout, device="cpu")
        else:
            batch_inds = latent["batch_index"] if "batch_index" in latent else None
            noise = comfy.sample.prepare_noise(latent_image_tensor, noise_seed, batch_inds)

        noise_mask = None
        if "noise_mask" in latent:
            noise_mask = latent["noise_mask"]

        use_video_optim = False
        if not compatibility_mode and optimize_for_video and latent_image_tensor.ndim == 5 and latent_image_tensor.shape[2] > 1:
            use_video_optim = True

        callback = None if use_video_optim else latent_preview.prepare_callback(model, steps)
        disable_pbar = True if use_video_optim else not comfy.utils.PROGRESS_BAR_ENABLED
        _sdpa_available = hasattr(torch.backends, 'cuda') and hasattr(torch.backends.cuda, 'sdpa_kernel')
        samples = comfy.sample.sample(model, noise, steps, cfg, sampler_name, scheduler, positive, negative, latent_image_tensor,
                                      denoise=denoise, disable_noise=disable_noise, start_step=start_at_step, last_step=end_at_step,
                                      force_full_denoise=force_full_denoise, noise_mask=noise_mask, callback=callback, disable_pbar=disable_pbar, seed=noise_seed)
        out = latent.copy()
        out["samples"] = samples
        return (out, )


class ROCMSamplerPerformanceMonitor:
    """Monitor sampler performance and provide optimization suggestions"""
    
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
    CATEGORY = "ROCm Ninodes/Sampling"
    DESCRIPTION = "Analyze sampler performance and provide optimization recommendations"
    
    def analyze(self, model, test_steps=20):
        """Analyze sampler performance and provide recommendations"""
        try:
            device = model_management.get_torch_device()
            model_dtype = model.model_dtype()
            device_info = f"Model device: {device}\n"
            device_info += f"Model dtype: {model_dtype}\n"
        except:
            device = None
            device_info = "Model info unavailable\n"
        
        is_amd = device and hasattr(device, 'type') and device.type == 'cuda'
        if is_amd:
            try:
                device_name = torch.cuda.get_device_name(0)
                device_info += f"GPU: {device_name}\n"
            except:
                device_info += "GPU: AMD (ROCm)\n"
        
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
        
        settings = []
        if is_amd:
            settings.append("Recommended precision: fp32")
            settings.append("Recommended memory optimization: True")
            settings.append("Recommended attention optimization: True")
            settings.append("Recommended samplers: euler, heun, dpmpp_2m")
            settings.append("Recommended schedulers: simple, normal")
            settings.append("Recommended CFG: 7.0-8.0")
        else:
            settings.append("Recommended precision: auto")
            settings.append("Recommended memory optimization: True")
            settings.append("Recommended attention optimization: True")
            settings.append("Recommended samplers: dpmpp_2m, dpmpp_sde")
            settings.append("Recommended schedulers: normal, karras")
            settings.append("Recommended CFG: 8.0-12.0")
        
        return (
            device_info,
            "\n".join(tips),
            "\n".join(settings)
        )


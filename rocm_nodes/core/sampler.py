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

from ..utils.memory import (
    gentle_memory_cleanup,
    emergency_memory_cleanup,
    get_gpu_memory_info,
)
from ..utils.debug import (
    DEBUG_MODE,
    log_debug,
    capture_timing,
    capture_memory_usage,
)
from ..utils.architecture import (
    detect_architecture,
    detect_model_sampling_type,
    apply_rocm_backend_settings,
)


class ROCMOptimizedKSampler:
    """
    Stock KSampler behavior under ROCm category for compatibility.
    Auto-detects GPU architecture and model type for optimal settings.
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
    DESCRIPTION = "KSampler with auto-detected architecture and model optimizations"

    def sample(self, model, seed, steps, cfg, sampler_name, scheduler, positive, negative,
               latent_image, denoise=1.0, optimize_for_video=False, precision_mode="auto", compatibility_mode=False):
        # ── Auto-detect architecture and model type ──────────────────────────
        arch_info = detect_architecture()
        model_info = detect_model_sampling_type(model)
        is_rocm = bool(getattr(torch.version, 'hip', None))

        if not compatibility_mode:
            apply_rocm_backend_settings(arch_info)

            # Aggressive cleanup for high-memory models (Ideogram4 etc.)
            if model_info["has_high_memory"]:
                print(f"💾 High-memory model detected (factor: {model_info['memory_usage_factor']}) — cleaning up")
                emergency_memory_cleanup()

            # Log model type
            if model_info["model_type"] == "flow":
                print(f"🌊 Flow-matching model detected ({model_info['latent_channels']}ch latent)")
            elif model_info["is_pixel_space"]:
                print(f"📷 Pixel-space model detected — no VAE decode needed downstream")
            elif model_info["has_high_memory"]:
                print(f"🔋 Large model detected — memory management enabled")

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
        is_video_workflow = False
        if not compatibility_mode and optimize_for_video and latent_image_tensor.ndim == 5 and latent_image_tensor.shape[2] > 1:
            use_video_optim = True
            is_video_workflow = True

        disable_pbar = use_video_optim

        pbar = comfy.utils.ProgressBar(steps)

        last_update_time = [time.time()]
        start_time = time.time()

        def enhanced_callback(step, x0, x, total_steps):
            current_time = time.time()

            if current_time - last_update_time[0] >= 0.3 or step == total_steps - 1:
                preview_bytes = None
                if not is_video_workflow and step % 5 == 0:
                    try:
                        previewer = latent_preview.get_previewer(model.load_device, model.model.latent_format)
                        if previewer:
                            preview_bytes = previewer.decode_latent_to_preview_image("JPEG", x0)
                    except:
                        preview_bytes = None

                try:
                    pbar.update_absolute(step + 1, total_steps, preview=preview_bytes)
                except:
                    pass
                last_update_time[0] = current_time

            progress_pct = ((step + 1) / total_steps) * 100
            elapsed = current_time - start_time
            if step > 0:
                est_total = elapsed / ((step + 1) / total_steps)
                est_remaining = est_total - elapsed
                avg_time_per_step = elapsed / (step + 1)
                workflow_type = "Video" if is_video_workflow else "Image"
                print(f"📊 {workflow_type} KSampler: Step {step + 1}/{total_steps} ({progress_pct:.1f}%) | "
                      f"Elapsed: {elapsed:.1f}s | Remaining: ~{est_remaining:.1f}s | "
                      f"Avg: {avg_time_per_step:.2f}s/step", flush=True)
            else:
                workflow_type = "Video" if is_video_workflow else "Image"
                print(f"📊 {workflow_type} KSampler: Step {step + 1}/{total_steps} ({progress_pct:.1f}%) | Starting...", flush=True)

        callback = enhanced_callback

        print(f"🚀 Starting sampling: {steps} steps, CFG {cfg}, {sampler_name} ({scheduler})", flush=True)
        samples = comfy.sample.sample(model, noise, steps, cfg, sampler_name, scheduler, positive, negative, latent_image_tensor,
                                      denoise=denoise, noise_mask=noise_mask, callback=callback, disable_pbar=False, seed=seed)
        print(f"✅ Sampling completed", flush=True)

        # Gentle cleanup after sampling (if model was large)
        if model_info["has_high_memory"]:
            gentle_memory_cleanup()

        out = latent.copy()
        out["samples"] = samples
        return (out,)


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
        arch_info = detect_architecture()
        model_info = detect_model_sampling_type(model)
        is_rocm = bool(getattr(torch.version, 'hip', None))

        if not compatibility_mode:
            apply_rocm_backend_settings(arch_info)

            if model_info["has_high_memory"]:
                print(f"💾 High-memory model detected (factor: {model_info['memory_usage_factor']}) — cleaning up")
                emergency_memory_cleanup()

            if model_info["model_type"] == "flow":
                print(f"🌊 Flow-matching model detected ({model_info['latent_channels']}ch latent)")
            elif model_info["is_pixel_space"]:
                print(f"📷 Pixel-space model detected — no VAE decode needed downstream")

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
        is_video_workflow = False
        if not compatibility_mode and optimize_for_video and latent_image_tensor.ndim == 5 and latent_image_tensor.shape[2] > 1:
            use_video_optim = True
            is_video_workflow = True

        disable_pbar = use_video_optim

        pbar = comfy.utils.ProgressBar(steps)

        last_update_time = [time.time()]
        start_time = time.time()

        def enhanced_callback(step, x0, x, total_steps):
            current_time = time.time()

            if current_time - last_update_time[0] >= 0.3 or step == total_steps - 1:
                preview_bytes = None
                if not is_video_workflow and step % 5 == 0:
                    try:
                        previewer = latent_preview.get_previewer(model.load_device, model.model.latent_format)
                        if previewer:
                            preview_bytes = previewer.decode_latent_to_preview_image("JPEG", x0)
                    except:
                        preview_bytes = None

                try:
                    pbar.update_absolute(step + 1, total_steps, preview=preview_bytes)
                except:
                    pass
                last_update_time[0] = current_time

            progress_pct = ((step + 1) / total_steps) * 100
            elapsed = current_time - start_time
            if step > 0:
                est_total = elapsed / ((step + 1) / total_steps)
                est_remaining = est_total - elapsed
                avg_time_per_step = elapsed / (step + 1)
                workflow_type = "Video" if is_video_workflow else "Image"
                print(f"📊 Advanced {workflow_type} KSampler: Step {step + 1}/{total_steps} ({progress_pct:.1f}%) | "
                      f"Elapsed: {elapsed:.1f}s | Remaining: ~{est_remaining:.1f}s | "
                      f"Avg: {avg_time_per_step:.2f}s/step", flush=True)
            else:
                workflow_type = "Video" if is_video_workflow else "Image"
                print(f"📊 Advanced {workflow_type} KSampler: Step {step + 1}/{total_steps} ({progress_pct:.1f}%) | Starting...", flush=True)

        callback = enhanced_callback

        print(f"🚀 Starting advanced sampling: {steps} steps (range: {start_at_step}-{end_at_step}), CFG {cfg}, {sampler_name} ({scheduler})", flush=True)
        samples = comfy.sample.sample(model, noise, steps, cfg, sampler_name, scheduler, positive, negative, latent_image_tensor,
                                      denoise=denoise, disable_noise=disable_noise, start_step=start_at_step, last_step=end_at_step,
                                      force_full_denoise=force_full_denoise, noise_mask=noise_mask, callback=callback, disable_pbar=False, seed=noise_seed)
        print(f"✅ Advanced sampling completed", flush=True)

        if model_info["has_high_memory"]:
            gentle_memory_cleanup()

        out = latent.copy()
        out["samples"] = samples
        return (out,)


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
        arch_info = detect_architecture()
        model_info = detect_model_sampling_type(model)

        device_info = f"Architecture: {arch_info['family']}\n"
        device_info += f"APU mode: {arch_info['is_apu']}\n"

        try:
            device = model_management.get_torch_device()
            model_dtype = model.model_dtype()
            device_info += f"Model device: {device}\n"
            device_info += f"Model dtype: {model_dtype}\n"
        except:
            device_info += "Model info unavailable\n"

        if torch.cuda.is_available():
            try:
                device_name = torch.cuda.get_device_name(0)
                device_info += f"GPU: {device_name}\n"
            except:
                device_info += "GPU: AMD (ROCm)\n"

        device_info += f"Sampling type: {model_info['model_type']}\n"
        device_info += f"Latent channels: {model_info['latent_channels']}\n"
        device_info += f"Memory factor: {model_info['memory_usage_factor']}x\n"

        tips = []
        settings = []

        is_amd = arch_info["family"] != "cpu"

        # Model-type tips
        if model_info["is_pixel_space"]:
            tips.append("• Pixel-space model — no VAE decode needed, lower overall memory")
            settings.append("No VAE decode configuration needed")
        elif model_info["model_type"] == "flow":
            tips.append("• Flow-matching model — use euler / dpmpp_sde / dpmpp_2m samplers")
            tips.append("• sgm_uniform scheduler recommended for flow matching")
            if model_info["latent_channels"] >= 128:
                tips.append(f"• Large latent format ({model_info['latent_channels']}ch) — ensure adequate memory")
        elif model_info["model_type"] == "eps":
            tips.append("• Standard noise-prediction model")
        elif model_info["model_type"] == "v_prediction":
            tips.append("• V-prediction model")
        else:
            tips.append("• Unknown model type — standard settings applied")

        # Architecture tips
        if is_amd:
            pref = arch_info.get("preferred_precision", "fp16")
            tips.append(f"• {pref.upper()} recommended for {arch_info['family']}")
            tips.append("• Enable ROCm optimizations for best results")
            if arch_info["is_apu"]:
                tips.append("• APU mode: disable smart-memory for unified memory")
            settings.append(f"Recommended precision: {pref}")
            settings.append("Recommended samplers: euler, dpmpp_2m, dpmpp_sde")
            settings.append(f"Recommended schedulers: sgm_uniform, normal")
        else:
            tips.append("• Use fp16 or bf16 for better performance")
            settings.append("Recommended precision: auto")
            settings.append("Recommended samplers: dpmpp_2m, dpmpp_sde")

        return (
            device_info,
            "\n".join(tips),
            "\n".join(settings)
        )

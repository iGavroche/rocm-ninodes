"""
VAE Decode nodes for ROCM Ninodes.

Contains all VAE-related node implementations:
- ROCMOptimizedVAEDecode: Main VAE decode node with ROCm optimizations
- ROCMOptimizedVAEDecodeTiled: Advanced tiled VAE decode
- ROCMVAEPerformanceMonitor: Performance monitoring and recommendations
"""

import time
import gc
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

try:
    import comfy.ldm.wan.vae
    import comfy.ldm.wan.vae2_2
    WAN_VAE_AVAILABLE = True
except ImportError:
    WAN_VAE_AVAILABLE = False

try:
    from comfy.ldm.lightricks.vae.causal_video_autoencoder import VideoVAE
    LTX_VAE_AVAILABLE = True
except ImportError:
    LTX_VAE_AVAILABLE = False

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
from ..utils.architecture import detect_architecture, apply_rocm_backend_settings
from ..constants import (
    DEFAULT_TILE_SIZE, DEFAULT_TILE_OVERLAP,
    MIN_LATENT_TILE_SIZE, MIN_LATENT_OVERLAP,
)


def _detect_vae_type(vae) -> str:
    """Classify the VAE model type.

    Returns:
        "pixel_space" | "ltxv_vae" | "wan_vae" | "standard"
    """
    latent_channels = getattr(vae, 'latent_channels', None)

    if latent_channels == 3:
        try:
            spatial_compression = vae.spacial_compression_decode()
            if spatial_compression == 1:
                return "pixel_space"
        except Exception:
            pass

    if latent_channels == 128:
        return "ltxv_vae"

    if not hasattr(vae, 'first_stage_model'):
        return "standard"
    sd = getattr(vae.first_stage_model, 'state_dict', None)
    if sd is not None:
        try:
            keys = sd().keys() if callable(sd) else sd.keys()
            if "decoder.up_blocks.0.res_blocks.0.conv1.conv.weight" in keys:
                return "ltxv_vae"
        except Exception:
            pass

    if WAN_VAE_AVAILABLE:
        try:
            if isinstance(vae.first_stage_model, (comfy.ldm.wan.vae.WanVAE, comfy.ldm.wan.vae2_2.WanVAE)):
                return "wan_vae"
        except Exception:
            pass

    if latent_channels in [16, 48]:
        try:
            temporal_compression = vae.temporal_compression_decode()
            if temporal_compression is not None:
                return "wan_vae"
        except Exception:
            pass

    return "standard"


def _select_precision(precision_mode: str, vae_type: str, vae, is_quantized: bool, vae_model_dtype, arch_info: dict) -> torch.dtype:
    """Select optimal compute precision based on user setting, VAE type, and architecture."""
    if is_quantized:
        return vae_model_dtype

    if precision_mode != "auto":
        dtype_map = {
            "fp32": torch.float32,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
        }
        return dtype_map[precision_mode]

    if is_quantized:
        return vae_model_dtype

    if vae_model_dtype is not None:
        return vae_model_dtype

    if arch_info["family"] != "cpu":
        pref = arch_info.get("preferred_precision", "fp16")
        dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
        return dtype_map.get(pref, torch.float16)

    return vae.vae_dtype


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
                    "tooltip": "Tile size. Larger values use more VRAM but are faster."
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
            },
            "optional": {
                "compatibility_mode": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Enable stock ComfyUI compatibility mode (disables all ROCm optimizations)"
                }),
                "enable_temporal_tiling": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "For LTX/WAN videos: decode in temporal chunks to reduce memory. Enable for long videos."
                }),
                "temporal_chunk_size": ("INT", {
                    "default": 16,
                    "min": 4,
                    "max": 256,
                    "step": 2,
                    "tooltip": "Temporal tile size in LATENT frames. 16 latent = ~121 output frames for LTX."
                }),
                "temporal_overlap": ("INT", {
                    "default": 2,
                    "min": 1,
                    "max": 8,
                    "step": 1,
                    "tooltip": "Overlap in LATENT frames between temporal tiles. Higher = smoother but more overhead."
                }),
                "last_frame_fix": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Repeat last latent frame before decode, then discard extra output frames. Fixes end artifacts."
                })
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("IMAGE",)
    FUNCTION = "decode"
    CATEGORY = "ROCm Ninodes/VAE"
    DESCRIPTION = "ROCM-optimized VAE Decode for AMD GPUs"

    def decode(self, vae, samples, tile_size=768, overlap=96, use_rocm_optimizations=True,
               precision_mode="auto", batch_optimization=True, compatibility_mode=False,
               enable_temporal_tiling=False, temporal_chunk_size=16,
               temporal_overlap=2, last_frame_fix=False):
        """
        Optimized VAE decode for ROCm/AMD GPUs with video support and quantized model compatibility
        """
        start_time = time.time()
        log_debug(f"ROCMOptimizedVAEDecode.decode started with samples shape: {samples['samples'].shape}")

        samples_tensor = samples["samples"]

        # ── Architecture detection ──────────────────────────────────────────
        arch_info = detect_architecture()
        is_amd = arch_info["family"] != "cpu"

        # ── VAE type detection ──────────────────────────────────────────────
        vae_type = _detect_vae_type(vae)
        log_debug(f"Detected VAE type: {vae_type} on arch: {arch_info['family']}")

        # ── Quantized model detection ───────────────────────────────────────
        is_quantized_model = False
        vae_model_dtype = getattr(vae.first_stage_model, 'dtype', None)
        if vae_model_dtype is not None:
            quantized_dtypes = [torch.float8_e4m3fn, torch.float8_e5m2, torch.int8, torch.int4]
            if vae_model_dtype in quantized_dtypes:
                is_quantized_model = True
                print(f"🔍 Detected quantized VAE model (dtype: {vae_model_dtype})")
            elif hasattr(vae_model_dtype, '__name__') and 'int' in str(vae_model_dtype):
                is_quantized_model = True
                print(f"🔍 Detected quantized VAE model (dtype: {vae_model_dtype})")

        # Compatibility mode
        if compatibility_mode or is_quantized_model:
            print("🛡️ Compatibility mode enabled - using stock ComfyUI behavior")
            use_rocm_optimizations = False
            batch_optimization = False

        # ═══════════════════════════════════════════════════════════════════
        # PIXEL-SPACE VAE PASSTHROUGH (z-image, z-image-turbo, ChromaRadiance)
        # ═══════════════════════════════════════════════════════════════════
        if vae_type == "pixel_space":
            print(f"📷 Pixel-space VAE detected — passthrough decode (no VAE processing needed)")
            with torch.no_grad():
                result = vae.decode(samples_tensor)
            if isinstance(result, tuple):
                result = result[0]
            if len(result.shape) == 5:
                result = result.reshape(-1, result.shape[-3],
                                        result.shape[-2], result.shape[-1])
            decode_time = time.time() - start_time
            print(f"✅ Pixel-space decode completed in {decode_time:.2f}s")
            return (result,)

        # ── ROCm backend settings ───────────────────────────────────────────
        if use_rocm_optimizations and is_amd:
            torch.backends.cuda.matmul.allow_fp16_accumulation = True

        # ── Debug capture ───────────────────────────────────────────────────
        if DEBUG_MODE:
            save_debug_data(samples, "vae_decode_input", "flux_1024x1024", {
                'node_type': 'ROCMOptimizedVAEDecode',
                'tile_size': tile_size,
                'overlap': overlap,
                'use_rocm_optimizations': use_rocm_optimizations,
                'precision_mode': precision_mode,
                'batch_optimization': batch_optimization,
                'vae_type': vae_type,
                'arch_family': arch_info['family'],
            })

        device = vae.device

        # ═══════════════════════════════════════════════════════════════════
        # VIDEO PATH (5D tensor: B, C, T, H, W)
        # ═══════════════════════════════════════════════════════════════════
        is_video = len(samples_tensor.shape) == 5
        if is_video:
            B, C, T, H, W = samples_tensor.shape
            print(f"🎬 Processing video: {T} frames, {H}x{W} resolution, VAE type: {vae_type}")

            # ── Output size estimation for large videos ──────────────────────
            out_frames, out_h, out_w = T, H, W
            try:
                temporal_comp = vae.temporal_compression_decode() or 1
                spatial_comp = vae.spacial_compression_decode() or 8
                out_frames = 1 + (T - 1) * temporal_comp
                out_h = H * spatial_comp
                out_w = W * spatial_comp
                est_gb = (B * out_frames * out_h * out_w * 3 * 4 * 3) / (1024**3)
                print(f"📊 Estimated output: {out_frames} frames at {out_h}x{out_w}, ~{est_gb:.1f}GB peak")
                if est_gb > 6:
                    print(f"⚠️ Large video decode — consider enabling temporal tiling")
            except Exception:
                pass

            # ── Convert model + input to efficient dtype for video ───────────
            video_dtype = None
            if use_rocm_optimizations and not is_quantized_model:
                video_dtype = _select_precision(
                    precision_mode, vae_type, vae, is_quantized_model,
                    vae_model_dtype, arch_info
                )
                working_dtypes = getattr(vae, 'working_dtypes', None)
                if working_dtypes and video_dtype not in working_dtypes:
                    video_dtype = working_dtypes[0]

                current_dtype = getattr(vae.first_stage_model, 'dtype', None)
                if current_dtype is not None and current_dtype != video_dtype:
                    # Causal video VAEs (LTX, WAN) need full precision for
                    # temporal state propagation — keep model weights native
                    if vae_type in ("ltxv_vae", "wan_vae") and video_dtype != current_dtype:
                        print(f"🔄 Causal VAE detected — keeping model in {current_dtype}, "
                              f"input in {video_dtype}", flush=True)
                    else:
                        try:
                            vae.first_stage_model = vae.first_stage_model.to(video_dtype)
                            print(f"💾 Converted VAE model from {current_dtype} to {video_dtype}", flush=True)
                        except Exception as e:
                            print(f"⚠️ Model dtype conversion skipped: {e}", flush=True)

                samples_processed = samples_tensor.to(device).to(video_dtype)
                print(f"🎯 Video input: {video_dtype}")
            else:
                samples_processed = samples_tensor

            # ── Aggressive cleanup before video decode ──────────────────────
            if use_rocm_optimizations and is_amd:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                gc.collect()

            # ── Temporal tiling path (for long videos) ───────────────────────
            if enable_temporal_tiling and vae_type in ("ltxv_vae", "wan_vae"):
                print(f"🧩 Temporal tiling enabled: chunk_size={temporal_chunk_size}, overlap={temporal_overlap}")
                result = self._decode_video_temporal_tiled(
                    vae, samples_processed, vae_type, temporal_chunk_size,
                    temporal_overlap, last_frame_fix, device, video_dtype or torch.float32
                )
            else:
                # ── Direct decode path (with chunked_io disabled for LTX) ────
                was_chunked = False
                if vae_type in ("ltxv_vae",):
                    was_chunked = getattr(vae.first_stage_model, 'comfy_has_chunked_io', False)
                    if was_chunked:
                        vae.first_stage_model.comfy_has_chunked_io = False
                        print("🔄 Disabled chunked IO to prevent giant pre-allocation")

                try:
                    with torch.no_grad():
                        result = vae.decode(samples_processed)
                except Exception as e:
                    if use_rocm_optimizations and video_dtype is not None and video_dtype != torch.float32:
                        print(f"⚠️ Video decode failed with {video_dtype}: {e}")
                        print(f"🔄 Retrying in fp32...")
                        torch.cuda.empty_cache()
                        try:
                            vae.first_stage_model = vae.first_stage_model.to(torch.float32)
                        except Exception:
                            pass
                        samples_retry = samples_tensor.to(device).to(torch.float32)
                        with torch.no_grad():
                            result = vae.decode(samples_retry)
                    elif vae_type in ("ltxv_vae",) and out_frames > 200:
                        raise RuntimeError(
                            f"Video VAE decode failed for {out_frames} frames at "
                            f"{out_h}x{out_w}. Try enabling temporal tiling "
                            f"(enable_temporal_tiling=True, temporal_chunk_size=16)."
                        ) from e
                    else:
                        raise RuntimeError(
                            f"Video VAE decode failed for {out_frames} frames at "
                            f"{out_h}x{out_w}. Try a shorter clip or reduce resolution."
                        ) from e
                finally:
                    if was_chunked:
                        vae.first_stage_model.comfy_has_chunked_io = True

            if isinstance(result, tuple):
                result = result[0]

            if len(result.shape) == 5:
                result = result.reshape(-1, result.shape[-3],
                                        result.shape[-2], result.shape[-1])

            if DEBUG_MODE:
                end_time = time.time()
                save_debug_data(result, "vae_decode_output", "flux_1024x1024", {
                    'node_type': 'ROCMOptimizedVAEDecode',
                    'execution_time': end_time - start_time,
                    'output_shape': result.shape,
                })
                capture_timing("vae_decode", start_time, end_time, {
                    'node_type': 'ROCMOptimizedVAEDecode',
                    'is_video': True,
                    'vae_type': vae_type,
                })

            decode_time = time.time() - start_time
            print(f"✅ Video decode completed in {decode_time:.2f}s")
            return (result,)

        # ═══════════════════════════════════════════════════════════════════
        # IMAGE PATH (4D tensor: B, C, H, W)
        # ═══════════════════════════════════════════════════════════════════

        # ── Precision selection ──────────────────────────────────────────────
        optimal_dtype = _select_precision(
            precision_mode, vae_type, vae, is_quantized_model, vae_model_dtype, arch_info
        )

        # ── Weight dtype mismatch fix ────────────────────────────────────────
        if not is_quantized_model:
            if vae_model_dtype == torch.bfloat16 and optimal_dtype == torch.float32:
                logging.warning("VAE model has BFloat16 weights but input is Float32 — converting")
                vae.first_stage_model = vae.first_stage_model.to(torch.float32)
                optimal_dtype = torch.float32

        # ── ROCm memory cleanup ─────────────────────────────────────────────
        if use_rocm_optimizations and is_amd:
            try:
                gentle_memory_cleanup()
                print("🧹 Memory cache cleared")
            except Exception as e:
                print(f"⚠️ Memory optimization skipped: {e}")

            if tile_size > arch_info["tile_size_max"]:
                tile_size = arch_info["tile_size_max"]
            if overlap > tile_size // 4:
                overlap = tile_size // 4

        # ── Memory estimation (fixed: uses all dims) ────────────────────────
        num_elements = 1
        for dim in samples_tensor.shape:
            num_elements *= dim
        bytes_per_element = 4 if optimal_dtype in (torch.float32,) else 2
        estimated_memory = num_elements * bytes_per_element * 2

        estimated_memory_gb = estimated_memory / (1024**3)
        is_safe, memory_msg = check_memory_safety(
            required_memory_gb=estimated_memory_gb + 2.0,
            is_apu=arch_info["is_apu"],
        )

        if not is_safe:
            print(f"⚠️ VAE Decode Memory Warning: {memory_msg}")
            print("🧹 Performing emergency cleanup before VAE decode...")
            emergency_memory_cleanup()

        # ── Load models ─────────────────────────────────────────────────────
        try:
            model_management.load_models_gpu([vae.patcher], memory_required=estimated_memory)
        except Exception as e:
            if "out of memory" in str(e).lower():
                print("💾 VAE model loading failed due to memory - performing cleanup")
                emergency_memory_cleanup()
                model_management.load_models_gpu([vae.patcher], memory_required=estimated_memory // 2)
            else:
                raise e

        # ── Batch number (capped for APU) ────────────────────────────────────
        free_memory = model_management.get_free_memory(device)
        if batch_optimization and is_amd:
            batch_number = max(1, int(free_memory / (estimated_memory * 1.2)))
        else:
            batch_number = max(1, int(free_memory / estimated_memory))
        if arch_info.get("batch_cap") is not None:
            batch_number = min(batch_number, arch_info["batch_cap"])

        # ── Prepare samples ──────────────────────────────────────────────────
        samples_processed = samples_tensor.to(device).to(optimal_dtype)

        # ── Model dtype cache ────────────────────────────────────────────────
        vae_id = id(vae.first_stage_model)
        cache_key = f"{vae_id}_{optimal_dtype}"
        if cache_key not in self._vae_model_cache:
            if not is_quantized_model:
                current_model_dtype = getattr(vae.first_stage_model, 'dtype', None)
                if current_model_dtype is not None and current_model_dtype != optimal_dtype:
                    logging.info(f"Converting VAE model from {current_model_dtype} to {optimal_dtype}")
                    vae.first_stage_model = vae.first_stage_model.to(optimal_dtype)
            self._vae_model_cache[cache_key] = True

        # ── Decide direct vs tiled ──────────────────────────────────────────
        image_size = samples_tensor.shape[2] * samples_tensor.shape[3]
        use_tiled = image_size > 512 * 512 or estimated_memory > free_memory * 0.8

        print(f"🖼️ Processing image: {samples_tensor.shape[2]}x{samples_tensor.shape[3]}")
        if use_tiled:
            print(f"🔧 Using tiled processing for large image")
        else:
            print(f"⚡ Using direct processing for optimal speed")

        if not use_tiled:
            try:
                pixel_samples = vae.decode(samples_processed)
                if isinstance(pixel_samples, tuple):
                    pixel_samples = pixel_samples[0]
            except Exception as e:
                logging.warning(f"Direct decode failed, falling back to tiled: {e}")
                use_tiled = True

        if use_tiled:
            pixel_samples = self._decode_tiled_optimized(
                vae, samples_tensor, tile_size, overlap, optimal_dtype, batch_number
            )

        if len(pixel_samples.shape) == 5:
            pixel_samples = pixel_samples.reshape(-1, pixel_samples.shape[-3],
                                                  pixel_samples.shape[-2], pixel_samples.shape[-1])

        decode_time = time.time() - start_time
        logging.info(f"ROCM VAE Decode completed in {decode_time:.2f}s")
        print(f"✅ Image decode completed in {decode_time:.2f}s")

        return (pixel_samples,)

    def _decode_tiled_optimized(self, vae, samples, tile_size, overlap, dtype, batch_number):
        """Optimized tiled decoding for ROCm with high-compression VAE support"""
        compression = vae.spacial_compression_decode()

        tile_latent = max(MIN_LATENT_TILE_SIZE, tile_size // compression)
        overlap_latent = max(MIN_LATENT_OVERLAP, overlap // compression)

        is_quantized_model = False
        vae_model_dtype = getattr(vae.first_stage_model, 'dtype', None)
        if vae_model_dtype is not None:
            quantized_dtypes = [torch.float8_e4m3fn, torch.float8_e5m2, torch.int8, torch.int4]
            if vae_model_dtype in quantized_dtypes:
                is_quantized_model = True
            elif hasattr(vae_model_dtype, '__name__') and 'int' in str(vae_model_dtype):
                is_quantized_model = True

        vae_id = id(vae.first_stage_model)
        cache_key = f"{vae_id}_{dtype}"
        if cache_key not in self._vae_model_cache:
            if not is_quantized_model:
                current_model_dtype = getattr(vae.first_stage_model, 'dtype', None)
                if current_model_dtype is not None and current_model_dtype != dtype:
                    logging.info(f"Tiled decode: Converting VAE model from {current_model_dtype} to {dtype}")
                    vae.first_stage_model = vae.first_stage_model.to(dtype)
            self._vae_model_cache[cache_key] = True

        def decode_fn(samples_tile):
            model_management.throw_exception_if_processing_interrupted()
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
            tile_x=tile_latent,
            tile_y=tile_latent,
            overlap=overlap_latent,
            upscale_amount=vae.upscale_ratio,
            out_channels=3,
            output_device=device
        )

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
                'output_device': str(result.device),
            })

        return result

    def _decode_video_temporal_tiled(self, vae, samples_tensor, vae_type, temporal_chunk_size,
                                     temporal_overlap, last_frame_fix, device, dtype):
        """Decode long videos by tiling the temporal dimension with overlap and blending.

        Uses the same approach as the LTXVideo custom node: each chunk overlaps its
        predecessor by temporal_overlap latent frames. The first output frame of each
        subsequent chunk is dropped (incomplete temporal context), and the next
        temporal_overlap * temporal_comp frames are linearly blended with the
        previous chunk's tail for seamless transitions.

        Args:
            vae: VAE object
            samples_tensor: 5D latent [B, C, T, H, W]
            temporal_chunk_size: Latent frames per chunk
            temporal_overlap: Overlap between chunks in latent frames
            last_frame_fix: Repeat last latent frame to fix end artifacts
            device: Target device
            dtype: Compute dtype

        Returns:
            BHWC tensor [total_frames, H, W, 3]
        """
        B, C, T, H, W = samples_tensor.shape
        temporal_comp = vae.temporal_compression_decode() or 8

        if last_frame_fix:
            last_frame = samples_tensor[:, :, -1:, :, :]
            samples_tensor = torch.cat([samples_tensor, last_frame], dim=2)
            T = T + 1
            print(f"  last_frame_fix enabled: padded to {T} latent frames")

        # ── Build temporal chunks ────────────────────────────────────────────
        # Using the LTXVideo chunk boundary formula:
        #   overlap_start accounts for one extra frame needed for causal context
        #   (the +1 term in max(1, chunk_start - overlap - 1))
        chunks = []
        chunk_start = 0
        while chunk_start < T:
            if chunk_start == 0:
                chunk_end = min(chunk_start + temporal_chunk_size, T)
                overlap_start = 0
            else:
                overlap_start = max(1, chunk_start - temporal_overlap - 1)
                extra = chunk_start - overlap_start
                chunk_end = min(chunk_start + temporal_chunk_size - extra, T)

            if chunk_end <= overlap_start:
                break
            chunks.append((overlap_start, chunk_end))
            chunk_start = chunk_end

        num_chunks = len(chunks)
        print(f"  Temporal tiling: {num_chunks} chunks, {T} latent frames")

        # ── Try to import progress bar ───────────────────────────────────────
        pbar = None
        if hasattr(comfy.utils, 'ProgressBar'):
            try:
                pbar = comfy.utils.ProgressBar(num_chunks)
            except Exception:
                pass

        # ── Decode each chunk with overlap + blend ──────────────────────────
        result_parts = []
        first_chunk_processed = False

        for chunk_idx, (c_start, c_end) in enumerate(chunks):
            model_management.throw_exception_if_processing_interrupted()
            chunk_frames = c_end - c_start
            chunk_latent = samples_tensor[:, :, c_start:c_end, :, :].to(device).to(dtype)

            with torch.no_grad():
                decoded = vae.decode(chunk_latent)
            if isinstance(decoded, tuple):
                decoded = decoded[0]

            # vae.decode() returns [B, out_T, H, W, C] with B=1
            decoded = decoded.squeeze(0)  # [out_T, H, W, C]

            if not first_chunk_processed:
                result_parts.append(decoded)
                first_chunk_processed = True
                out_T = decoded.shape[0]
                print(f"  Chunk 0: latent [{c_start}:{c_end}] ({chunk_frames}) → {out_T} output frames")
            else:
                out_T = decoded.shape[0]

                # Drop the first output frame — it has the most risk of temporal
                # artifacts since the first latent frame lacks backward context.
                decoded = decoded[1:]  # [out_T - 1, H, W, C]

                # Blend the overlap region with the previous chunk's tail
                blend_frames = min(temporal_overlap * temporal_comp, decoded.shape[0], result_parts[-1].shape[0])
                if blend_frames > 0:
                    prev_tail = result_parts[-1][-blend_frames:]
                    curr_head = decoded[:blend_frames]
                    w = torch.linspace(0, 1, blend_frames, device=decoded.device, dtype=decoded.dtype)
                    w = w.view(-1, 1, 1, 1)
                    result_parts[-1][-blend_frames:] = prev_tail * (1 - w) + curr_head * w

                # Append the clean tail of the current chunk
                result_parts.append(decoded[blend_frames:])

                print(f"  Chunk {chunk_idx}: latent [{c_start}:{c_end}] ({chunk_frames}) → "
                      f"{out_T} output, dropped 1, blended {blend_frames}")

            if pbar is not None:
                pbar.update_absolute(chunk_idx + 1)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        result = torch.cat(result_parts, dim=0)

        if last_frame_fix and result.shape[0] > temporal_comp:
            result = result[:-temporal_comp]
            print(f"  last_frame_fix: trimmed {temporal_comp} frames from end")

        print(f"  Temporal tiling complete: {result.shape[0]} total output frames")
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
        arch_info = detect_architecture()
        vae_type = _detect_vae_type(vae)

        device_info = f"Device: {device}\n"
        device_info += f"VAE dtype: {vae.vae_dtype}\n"
        device_info += f"Output device: {vae.output_device}\n"
        device_info += f"VAE type: {vae_type}\n"
        device_info += f"Architecture: {arch_info['family']}\n"
        device_info += f"APU mode: {arch_info['is_apu']}\n"

        if torch.cuda.is_available():
            try:
                device_name = torch.cuda.get_device_name(0)
                device_info += f"GPU: {device_name}\n"
                arch = torch.cuda.get_device_properties(0).gcnArchName
                device_info += f"Arch: {arch}\n"
            except:
                device_info += "GPU: AMD (ROCm)\n"

        tips = []
        settings = []

        if vae_type == "pixel_space":
            tips.append("• This VAE is a pixel-space passthrough (no actual decoding needed)")
            tips.append("• The model operates directly on RGB pixels")
            settings.append("No VAE decode configuration needed")
            settings.append("z-image / z-image-turbo models are supported natively")
        elif vae_type == "ltxv_vae":
            tips.append("• LTX Video VAE detected (128-channel, 32x spatial compression)")
            tips.append("• fp16 is strongly recommended for memory efficiency")
            tips.append("• Use tile_size 1024-2048 for reasonable tile sizes in latent space")
            tips.append("• Process full videos at once — causal decode chain")
            tips.append("• For videos >200 frames, enable temporal tiling (chunk_size=16, overlap=2)")
            if arch_info["is_apu"]:
                tips.append("• APU mode: batch capped to 8 to prevent over-allocation")
            settings.append(f"Recommended tile_size: {min(2048, test_resolution * 2)}")
            settings.append(f"Recommended overlap: {min(128, test_resolution // 4)}")
            settings.append(f"Recommended precision: fp16")
            settings.append(f"Recommended batch_optimization: True")
            settings.append(f"Temporal tiling available: enable_temporal_tiling=True, chunk_size=16")
            if arch_info["is_apu"]:
                settings.append("APU batch cap: 8")
        elif vae_type == "wan_vae":
            tips.append("• WAN VAE detected (causal temporal decoding)")
            tips.append("• Process full videos at once to avoid frame jitter")
            tips.append("• fp16 available for this VAE type")
            settings.append(f"Recommended tile_size: {min(1024, test_resolution)}")
            settings.append(f"Recommended overlap: {min(128, test_resolution // 8)}")
            settings.append(f"Recommended precision: {arch_info.get('preferred_precision', 'fp16')}")
            settings.append(f"Recommended batch_optimization: True")
        else:
            tips.append(f"• Standard VAE ({vae.latent_channels} channels)")
            tips.append(f"• {arch_info['preferred_precision'].upper()} recommended for {arch_info['family']}")
            tips.append(f"• Tile size {arch_info['tile_size_max'] // 2}-{arch_info['tile_size_max']} works well")
            tips.append("• Enable ROCm optimizations for best results")
            if arch_info["is_apu"]:
                tips.append("• APU mode: batch capped to 8")
            settings.append(f"Recommended tile_size: {min(arch_info['tile_size_max'], test_resolution)}")
            settings.append(f"Recommended overlap: {min(128, test_resolution // 8)}")
            settings.append(f"Recommended precision: {arch_info.get('preferred_precision', 'fp16')}")
            settings.append(f"Recommended batch_optimization: True")

        if arch_info["is_apu"]:
            tips.append("• Unified memory detected: disable smart-memory for best results")

        return (
            device_info,
            "\n".join(tips),
            "\n".join(settings)
        )

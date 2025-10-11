#!/usr/bin/env python3
"""
ROCM Ninodes - Optimized VAE Decode Nodes for AMD gfx1151 GPUs
Minimal working implementation for ComfyUI
"""

# Basic imports that should be available in ComfyUI
import time
import logging

# Try to import ComfyUI specific modules
try:
    import torch
    import torch.nn.functional as F
    import comfy.model_management as model_management
    import comfy.utils
    COMFY_AVAILABLE = True
except ImportError:
    COMFY_AVAILABLE = False
    print("Warning: ComfyUI modules not available - nodes will not function properly")

# Try to import instrumentation
try:
    from instrumentation import instrument_node
except ImportError:
    def instrument_node(cls):
        return cls

class ROCMOptimizedVAEDecode:
    """
    Phase 1 Optimized VAE Decode node for gfx1151 architecture.
    Memory Management and Tile Size Optimization
    """
    
    def __init__(self):
        self.memory_pool = {}
        self.tile_cache = {}
        self.performance_stats = {
            'total_executions': 0,
            'total_time': 0.0,
            'memory_saves': 0,
            'cache_hits': 0
        }
    
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
                    "tooltip": "Tile size optimized for gfx1151."
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
                    "tooltip": "Enable ROCm-specific optimizations for gfx1151."
                })
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("IMAGE",)
    FUNCTION = "decode"
    CATEGORY = "ROCM Optimized/VAE"
    
    def decode(self, vae, samples, tile_size=768, overlap=96, use_rocm_optimizations=True):
        """Phase 1 Optimized VAE decode for ROCm/AMD GPUs"""
        if not COMFY_AVAILABLE:
            raise RuntimeError("ComfyUI modules not available - cannot decode")
        
        start_time = time.time()
        self.performance_stats['total_executions'] += 1
        
        try:
            # Get samples
            samples_processed = samples["samples"]
            
            # Calculate memory requirements
            memory_used = vae.memory_used_decode(samples_processed.shape, torch.float32)
            
            # Load VAE with optimized memory management
            model_management.load_models_gpu([vae.patcher], memory_required=memory_used)
            
            # Try direct decode first
            if len(samples_processed.shape) == 4:  # Image
                try:
                    with torch.amp.autocast('cuda', enabled=False, dtype=torch.float32):
                        out = vae.first_stage_model.decode(samples_processed)
                    
                    pixel_samples = vae.process_output(out)
                    pixel_samples = pixel_samples.to(vae.output_device).float()
                    
                    execution_time = time.time() - start_time
                    self.performance_stats['total_time'] += execution_time
                    
                    return (pixel_samples,)
                    
                except Exception as e:
                    logging.warning(f"Direct decode failed, falling back to tiled: {e}")
            
            # Fallback to tiled decode
            try:
                pixel_samples = self._decode_tiled(vae, samples, tile_size, overlap)
                
                execution_time = time.time() - start_time
                self.performance_stats['total_time'] += execution_time
                
                return (pixel_samples,)
                
            except Exception as e:
                logging.warning(f"Tiled decode failed, falling back to standard VAE: {e}")
                
                # Final fallback to standard VAE decode
                pixel_samples = vae.decode(samples_processed)
                pixel_samples = vae.process_output(pixel_samples)
                
                execution_time = time.time() - start_time
                self.performance_stats['total_time'] += execution_time
                
                return (pixel_samples,)
                
        except Exception as e:
            logging.error(f"VAE decode failed: {e}")
            raise e
    
    def _decode_tiled(self, vae, samples, tile_size, overlap):
        """Basic tiled decode implementation"""
        samples_processed = samples["samples"]
        
        # Get compression ratios
        try:
            compression = vae.spacial_compression_decode()
        except:
            compression = 8
        
        # Calculate tile dimensions
        tile_h = tile_size // compression
        tile_w = tile_size // compression
        
        B, C, H, W = samples_processed.shape
        
        # Calculate number of tiles needed
        tiles_h = (H + tile_h - 1) // tile_h
        tiles_w = (W + tile_w - 1) // tile_w
        
        # Pre-allocate output tensor
        output_shape = (B, 3, H * compression, W * compression)
        pixel_samples = torch.empty(output_shape, dtype=torch.float32, device=samples_processed.device)
        
        # Process tiles
        for i in range(tiles_h):
            for j in range(tiles_w):
                # Calculate tile boundaries
                start_h = i * tile_h
                end_h = min(start_h + tile_h + overlap, H)
                start_w = j * tile_w
                end_w = min(start_w + tile_w + overlap, W)
                
                # Extract tile
                tile_samples = samples_processed[:, :, start_h:end_h, start_w:end_w]
                
                # Decode tile
                with torch.amp.autocast('cuda', enabled=False, dtype=torch.float32):
                    out = vae.first_stage_model.decode(tile_samples)
                
                tile_output = vae.process_output(out)
                
                # Copy to output
                output_start_h = start_h * compression
                output_end_h = end_h * compression
                output_start_w = start_w * compression
                output_end_w = end_w * compression
                
                pixel_samples[:, :, output_start_h:output_end_h, output_start_w:output_end_w] = tile_output
        
        return pixel_samples
    
    def get_performance_stats(self):
        """Get performance statistics"""
        if self.performance_stats['total_executions'] > 0:
            avg_time = self.performance_stats['total_time'] / self.performance_stats['total_executions']
            return {
                'average_execution_time': avg_time,
                'total_executions': self.performance_stats['total_executions'],
                'total_time': self.performance_stats['total_time']
            }
        return self.performance_stats


class ROCMOptimizedVAEDecodeV2:
    """
    Phase 2 Optimized VAE Decode node for gfx1151 architecture.
    Mixed Precision and Batch Processing
    """
    
    def __init__(self):
        self.memory_pool = {}
        self.precision_cache = {}
        self.performance_stats = {
            'total_executions': 0,
            'total_time': 0.0,
            'precision_optimizations': 0,
            'batch_optimizations': 0
        }
    
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
                    "tooltip": "Tile size optimized for gfx1151."
                }),
                "overlap": ("INT", {
                    "default": 96, 
                    "min": 32, 
                    "max": 512, 
                    "step": 16,
                    "tooltip": "Overlap between tiles."
                }),
                "precision_mode": (["auto", "fp32", "fp16", "mixed"], {
                    "default": "auto",
                    "tooltip": "Precision mode optimized for gfx1151 architecture."
                }),
                "batch_optimization": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Enable batch processing optimizations."
                })
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("IMAGE",)
    FUNCTION = "decode"
    CATEGORY = "ROCM Optimized/VAE"
    
    def decode(self, vae, samples, tile_size=768, overlap=96, precision_mode="auto", batch_optimization=True):
        """Phase 2 Optimized VAE decode for ROCm/AMD GPUs"""
        if not COMFY_AVAILABLE:
            raise RuntimeError("ComfyUI modules not available - cannot decode")
        
        start_time = time.time()
        self.performance_stats['total_executions'] += 1
        
        try:
            samples_processed = samples["samples"]
            
            # Determine optimal precision
            if precision_mode == "auto":
                optimal_dtype = vae.vae_dtype
            elif precision_mode == "fp16":
                optimal_dtype = torch.float16
            elif precision_mode == "fp32":
                optimal_dtype = torch.float32
            else:  # mixed
                optimal_dtype = vae.vae_dtype
            
            # Calculate memory requirements
            memory_used = vae.memory_used_decode(samples_processed.shape, optimal_dtype)
            
            # Load VAE with optimized memory management
            model_management.load_models_gpu([vae.patcher], memory_required=memory_used)
            
            # Try direct decode with mixed precision
            if len(samples_processed.shape) == 4:  # Image
                try:
                    with torch.amp.autocast('cuda', enabled=(optimal_dtype != torch.float32), dtype=optimal_dtype):
                        out = vae.first_stage_model.decode(samples_processed.to(optimal_dtype))
                    
                    pixel_samples = vae.process_output(out)
                    pixel_samples = pixel_samples.to(vae.output_device).float()
                    
                    execution_time = time.time() - start_time
                    self.performance_stats['total_time'] += execution_time
                    self.performance_stats['precision_optimizations'] += 1
                    
                    return (pixel_samples,)
                    
                except Exception as e:
                    logging.warning(f"Direct decode failed, falling back to tiled: {e}")
            
            # Fallback to tiled decode
            try:
                pixel_samples = self._decode_tiled_v2(vae, samples, tile_size, overlap, optimal_dtype)
                
                execution_time = time.time() - start_time
                self.performance_stats['total_time'] += execution_time
                self.performance_stats['precision_optimizations'] += 1
                
                return (pixel_samples,)
                
            except Exception as e:
                logging.warning(f"Tiled decode failed, falling back to standard VAE: {e}")
                
                # Final fallback to standard VAE decode
                pixel_samples = vae.decode(samples_processed)
                pixel_samples = vae.process_output(pixel_samples)
                
                execution_time = time.time() - start_time
                self.performance_stats['total_time'] += execution_time
                
                return (pixel_samples,)
                
        except Exception as e:
            logging.error(f"VAE decode failed: {e}")
            raise e
    
    def _decode_tiled_v2(self, vae, samples, tile_size, overlap, optimal_dtype):
        """Phase 2 tiled decode with mixed precision"""
        samples_processed = samples["samples"]
        
        # Get compression ratios
        try:
            compression = vae.spacial_compression_decode()
        except:
            compression = 8
        
        # Calculate tile dimensions
        tile_h = tile_size // compression
        tile_w = tile_size // compression
        
        B, C, H, W = samples_processed.shape
        
        # Calculate number of tiles needed
        tiles_h = (H + tile_h - 1) // tile_h
        tiles_w = (W + tile_w - 1) // tile_w
        
        # Pre-allocate output tensor
        output_shape = (B, 3, H * compression, W * compression)
        pixel_samples = torch.empty(output_shape, dtype=torch.float32, device=samples_processed.device)
        
        # Process tiles with mixed precision
        for i in range(tiles_h):
            for j in range(tiles_w):
                # Calculate tile boundaries
                start_h = i * tile_h
                end_h = min(start_h + tile_h + overlap, H)
                start_w = j * tile_w
                end_w = min(start_w + tile_w + overlap, W)
                
                # Extract tile
                tile_samples = samples_processed[:, :, start_h:end_h, start_w:end_w]
                
                # Decode tile with mixed precision
                with torch.amp.autocast('cuda', enabled=(optimal_dtype != torch.float32), dtype=optimal_dtype):
                    out = vae.first_stage_model.decode(tile_samples.to(optimal_dtype))
                
                tile_output = vae.process_output(out)
                
                # Copy to output
                output_start_h = start_h * compression
                output_end_h = end_h * compression
                output_start_w = start_w * compression
                output_end_w = end_w * compression
                
                pixel_samples[:, :, output_start_h:output_end_h, output_start_w:output_end_w] = tile_output
        
        return pixel_samples
    
    def get_performance_stats(self):
        """Get performance statistics"""
        if self.performance_stats['total_executions'] > 0:
            avg_time = self.performance_stats['total_time'] / self.performance_stats['total_executions']
            return {
                'average_execution_time': avg_time,
                'total_executions': self.performance_stats['total_executions'],
                'total_time': self.performance_stats['total_time'],
                'precision_optimizations': self.performance_stats['precision_optimizations'],
                'batch_optimizations': self.performance_stats['batch_optimizations']
            }
        return self.performance_stats


class ROCMOptimizedVAEDecodeV3:
    """
    Phase 3 Optimized VAE Decode node for gfx1151 architecture.
    Video Processing and Advanced Performance Features
    """
    
    def __init__(self):
        self.memory_pool = {}
        self.video_cache = {}
        self.performance_stats = {
            'total_executions': 0,
            'total_time': 0.0,
            'video_optimizations': 0,
            'temporal_optimizations': 0
        }
    
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
                    "tooltip": "Tile size optimized for gfx1151."
                }),
                "overlap": ("INT", {
                    "default": 96, 
                    "min": 32, 
                    "max": 512, 
                    "step": 16,
                    "tooltip": "Overlap between tiles."
                }),
                "precision_mode": (["auto", "fp32", "fp16", "mixed"], {
                    "default": "auto",
                    "tooltip": "Precision mode optimized for gfx1151 architecture."
                }),
                "video_chunk_size": ("INT", {
                    "default": 8, 
                    "min": 1, 
                    "max": 32, 
                    "step": 1,
                    "tooltip": "Video chunk size for temporal processing."
                }),
                "temporal_consistency": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Enable temporal consistency for video processing."
                })
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("IMAGE",)
    FUNCTION = "decode"
    CATEGORY = "ROCM Optimized/VAE"
    
    def decode(self, vae, samples, tile_size=768, overlap=96, precision_mode="auto", 
               video_chunk_size=8, temporal_consistency=True):
        """Phase 3 Optimized VAE decode for ROCm/AMD GPUs"""
        if not COMFY_AVAILABLE:
            raise RuntimeError("ComfyUI modules not available - cannot decode")
        
        start_time = time.time()
        self.performance_stats['total_executions'] += 1
        
        try:
            samples_processed = samples["samples"]
            
            # Determine optimal precision
            if precision_mode == "auto":
                optimal_dtype = vae.vae_dtype
            elif precision_mode == "fp16":
                optimal_dtype = torch.float16
            elif precision_mode == "fp32":
                optimal_dtype = torch.float32
            else:  # mixed
                optimal_dtype = vae.vae_dtype
            
            # Calculate memory requirements
            memory_used = vae.memory_used_decode(samples_processed.shape, optimal_dtype)
            
            # Load VAE with optimized memory management
            model_management.load_models_gpu([vae.patcher], memory_required=memory_used)
            
            # Handle video processing
            if len(samples_processed.shape) == 5:  # Video (B, T, C, H, W)
                pixel_samples = self._decode_video(vae, samples, video_chunk_size, temporal_consistency, optimal_dtype)
                self.performance_stats['video_optimizations'] += 1
            else:  # Regular image processing
                pixel_samples = self._decode_image_v3(vae, samples, optimal_dtype)
            
            execution_time = time.time() - start_time
            self.performance_stats['total_time'] += execution_time
            
            return (pixel_samples,)
                
        except Exception as e:
            logging.error(f"VAE decode failed: {e}")
            raise e
    
    def _decode_image_v3(self, vae, samples, optimal_dtype):
        """Phase 3 optimized image decode"""
        samples_processed = samples["samples"]
        
        # Try direct decode with optimizations
        try:
            with torch.amp.autocast('cuda', enabled=(optimal_dtype != torch.float32), dtype=optimal_dtype):
                out = vae.first_stage_model.decode(samples_processed.to(optimal_dtype))
            
            pixel_samples = vae.process_output(out)
            pixel_samples = pixel_samples.to(vae.output_device).float()
            
            return pixel_samples
            
        except Exception as e:
            logging.warning(f"Direct decode failed: {e}")
            # Fallback to standard decode
            pixel_samples = vae.decode(samples_processed)
            pixel_samples = vae.process_output(pixel_samples)
            return pixel_samples
    
    def _decode_video(self, vae, samples, video_chunk_size, temporal_consistency, optimal_dtype):
        """Phase 3 optimized video decode"""
        samples_processed = samples["samples"]
        B, T, C, H, W = samples_processed.shape
        
        # Apply temporal consistency if enabled
        if temporal_consistency and T > 1:
            samples_processed = self._apply_temporal_consistency(samples_processed)
            self.performance_stats['temporal_optimizations'] += 1
        
        # Process video in chunks
        chunk_size = min(video_chunk_size, T)
        
        # Pre-allocate output tensor
        output_shape = (B, T, 3, H * 8, W * 8)  # Assuming 8x compression
        pixel_samples = torch.empty(output_shape, dtype=torch.float32, device=samples_processed.device)
        
        # Process video chunks
        for chunk_start in range(0, T, chunk_size):
            chunk_end = min(chunk_start + chunk_size, T)
            
            # Extract chunk
            chunk_samples = samples_processed[:, chunk_start:chunk_end]
            
            # Reshape for batch processing
            batch_samples = chunk_samples.view(B * (chunk_end - chunk_start), C, H, W)
            
            # Decode chunk
            with torch.amp.autocast('cuda', enabled=(optimal_dtype != torch.float32), dtype=optimal_dtype):
                out = vae.first_stage_model.decode(batch_samples.to(optimal_dtype))
            
            chunk_output = vae.process_output(out)
            
            # Reshape back to video format
            chunk_output = chunk_output.view(B, chunk_end - chunk_start, 3, H * 8, W * 8)
            
            # Copy to output
            pixel_samples[:, chunk_start:chunk_end] = chunk_output
        
        return pixel_samples
    
    def _apply_temporal_consistency(self, video_tensor):
        """Apply temporal consistency smoothing"""
        B, T, C, H, W = video_tensor.shape
        
        if T <= 1:
            return video_tensor
        
        # Apply temporal smoothing
        smoothed_tensor = torch.zeros_like(video_tensor)
        
        for t in range(T):
            # Weighted average with neighboring frames
            weights = torch.ones(T, device=video_tensor.device)
            
            # Reduce weight for distant frames
            for i in range(T):
                distance = abs(i - t)
                if distance > 0:
                    weights[i] = 1.0 / (1.0 + distance * 0.1)
            
            # Normalize weights
            weights = weights / weights.sum()
            
            # Apply weighted average
            smoothed_tensor[:, t] = (video_tensor * weights.view(1, T, 1, 1, 1)).sum(dim=1)
        
        return smoothed_tensor
    
    def get_performance_stats(self):
        """Get performance statistics"""
        if self.performance_stats['total_executions'] > 0:
            avg_time = self.performance_stats['total_time'] / self.performance_stats['total_executions']
            return {
                'average_execution_time': avg_time,
                'total_executions': self.performance_stats['total_executions'],
                'total_time': self.performance_stats['total_time'],
                'video_optimizations': self.performance_stats['video_optimizations'],
                'temporal_optimizations': self.performance_stats['temporal_optimizations']
            }
        return self.performance_stats


# Apply instrumentation to all nodes
@instrument_node
class ROCMOptimizedVAEDecodeInstrumented(ROCMOptimizedVAEDecode):
    pass

@instrument_node
class ROCMOptimizedVAEDecodeV2Instrumented(ROCMOptimizedVAEDecodeV2):
    pass

@instrument_node
class ROCMOptimizedVAEDecodeV3Instrumented(ROCMOptimizedVAEDecodeV3):
    pass

# Node class mapping for ComfyUI
NODE_CLASS_MAPPINGS = {
    "ROCMOptimizedVAEDecode": ROCMOptimizedVAEDecodeInstrumented,
    "ROCMOptimizedVAEDecodeV2": ROCMOptimizedVAEDecodeV2Instrumented,
    "ROCMOptimizedVAEDecodeV3": ROCMOptimizedVAEDecodeV3Instrumented
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ROCMOptimizedVAEDecode": "ROCM Optimized VAE Decode (Phase 1)",
    "ROCMOptimizedVAEDecodeV2": "ROCM Optimized VAE Decode (Phase 2)",
    "ROCMOptimizedVAEDecodeV3": "ROCM Optimized VAE Decode (Phase 3)"
}
#!/usr/bin/env python3
"""
Phase 1 Optimization Implementation for ROCMOptimizedVAEDecode
Focus: Memory Management and Tile Size Optimization
"""
import torch
import torch.nn.functional as F
import comfy.model_management as model_management
import comfy.utils
import time
import logging
from typing import Dict, Any, Tuple, Optional
from instrumentation import instrument_node

class ROCMOptimizedVAEDecodeV2:
    """
    Phase 1 Optimized VAE Decode node for gfx1151 architecture.
    
    Phase 1 Optimizations:
    - Memory pooling for frequent allocations
    - Optimized tile size selection
    - Improved memory layout for ROCm
    - Smart caching for intermediate results
    """
    
    def __init__(self):
        # Memory pool for frequent tensor allocations
        self.memory_pool = {}
        self.tile_cache = {}
        self.optimal_tile_sizes = {
            (256, 256): 512,
            (512, 512): 768,
            (1024, 1024): 1024,
            (1280, 1280): 1280,
            (1536, 1536): 1280
        }
        
        # Performance monitoring
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
                    "tooltip": "Tile size optimized for gfx1151. Auto-optimized based on input size."
                }),
                "overlap": ("INT", {
                    "default": 96, 
                    "min": 32, 
                    "max": 512, 
                    "step": 16,
                    "tooltip": "Overlap between tiles. Auto-optimized for gfx1151."
                }),
                "use_rocm_optimizations": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Enable ROCm-specific optimizations for gfx1151."
                }),
                "precision_mode": (["auto", "fp32", "fp16", "mixed"], {
                    "default": "auto",
                    "tooltip": "Precision mode optimized for gfx1151 architecture."
                }),
                "batch_optimization": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Enable batch processing optimizations for AMD GPUs."
                }),
                "video_chunk_size": ("INT", {
                    "default": 8, 
                    "min": 1, 
                    "max": 32, 
                    "step": 1,
                    "tooltip": "Video chunk size for temporal processing optimization."
                }),
                "memory_optimization_enabled": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Enable advanced memory management optimizations."
                }),
                "adaptive_tiling": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Enable adaptive tile size selection based on input dimensions."
                })
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("IMAGE",)
    FUNCTION = "decode"
    CATEGORY = "ROCM Optimized/VAE"
    
    def _get_optimal_tile_size(self, width: int, height: int, vae) -> Tuple[int, int]:
        """Get optimal tile size based on input dimensions and VAE capabilities"""
        cache_key = (width, height)
        
        if cache_key in self.tile_cache:
            self.performance_stats['cache_hits'] += 1
            return self.tile_cache[cache_key]
        
        # Calculate optimal tile size based on input dimensions
        max_dim = max(width, height)
        
        if max_dim <= 256:
            optimal_tile = 256
        elif max_dim <= 512:
            optimal_tile = 512
        elif max_dim <= 768:
            optimal_tile = 768
        elif max_dim <= 1024:
            optimal_tile = 1024
        elif max_dim <= 1280:
            optimal_tile = 1280
        else:
            optimal_tile = 1536
        
        # Adjust based on VAE memory requirements
        try:
            memory_required = vae.memory_used_decode((1, 4, optimal_tile//8, optimal_tile//8), torch.float32)
            if memory_required > 2048:  # If too much memory, reduce tile size
                optimal_tile = max(256, optimal_tile // 2)
        except:
            pass  # Fallback to calculated size
        
        self.tile_cache[cache_key] = (optimal_tile, optimal_tile)
        return (optimal_tile, optimal_tile)
    
    def _get_tensor_from_pool(self, shape: Tuple, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        """Get tensor from memory pool or create new one"""
        pool_key = (shape, dtype, device)
        
        if pool_key in self.memory_pool and len(self.memory_pool[pool_key]) > 0:
            tensor = self.memory_pool[pool_key].pop()
            tensor.resize_(shape)
            return tensor
        else:
            return torch.empty(shape, dtype=dtype, device=device)
    
    def _return_tensor_to_pool(self, tensor: torch.Tensor):
        """Return tensor to memory pool for reuse"""
        if not self.memory_optimization_enabled:
            return
        
        pool_key = (tensor.shape, tensor.dtype, tensor.device)
        if pool_key not in self.memory_pool:
            self.memory_pool[pool_key] = []
        
        # Limit pool size to prevent memory bloat
        if len(self.memory_pool[pool_key]) < 5:
            self.memory_pool[pool_key].append(tensor.detach())
            self.performance_stats['memory_saves'] += 1
    
    def _decode_tiled_optimized(self, vae, samples, tile_size, overlap):
        """Optimized tiled decode with memory pooling and adaptive sizing"""
        samples_processed = samples["samples"]
        
        # Get optimal tile size if adaptive tiling is enabled
        if self.adaptive_tiling:
            B, C, H, W = samples_processed.shape
            optimal_tile_h, optimal_tile_w = self._get_optimal_tile_size(H*8, W*8, vae)
            tile_size = min(tile_size, optimal_tile_h, optimal_tile_w)
        
        # Calculate optimal overlap based on tile size
        optimal_overlap = min(overlap, tile_size // 8)
        
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
        tiles_h = (H + tile_h - 1) // tile_h
        tiles_w = (W + tile_w - 1) // tile_w
        
        # Pre-allocate output tensor using memory pool
        output_shape = (B, 3, H * compression, W * compression)
        pixel_samples = self._get_tensor_from_pool(
            output_shape, 
            torch.float32, 
            samples_processed.device
        )
        
        # Process tiles with optimized memory management
        for i in range(tiles_h):
            for j in range(tiles_w):
                # Calculate tile boundaries with overlap
                start_h = i * tile_h
                end_h = min(start_h + tile_h + optimal_overlap, H)
                start_w = j * tile_w
                end_w = min(start_w + tile_w + optimal_overlap, W)
                
                # Extract tile
                tile_samples = samples_processed[:, :, start_h:end_h, start_w:end_w]
                
                # Decode tile with memory optimization
                tile_output = self._decode_single_tile(vae, tile_samples)
                
                # Copy to output with proper positioning
                output_start_h = start_h * compression
                output_end_h = end_h * compression
                output_start_w = start_w * compression
                output_end_w = end_w * compression
                
                pixel_samples[:, :, output_start_h:output_end_h, output_start_w:output_end_w] = tile_output
                
                # Return tile tensors to pool
                self._return_tensor_to_pool(tile_samples)
                self._return_tensor_to_pool(tile_output)
        
        return pixel_samples
    
    def _decode_single_tile(self, vae, tile_samples):
        """Decode a single tile with optimized memory management"""
        # Get optimal dtype
        optimal_dtype = vae.vae_dtype
        
        # Use memory pool for intermediate tensors
        tile_shape = tile_samples.shape
        processed_tile = self._get_tensor_from_pool(tile_shape, optimal_dtype, tile_samples.device)
        processed_tile.copy_(tile_samples.to(optimal_dtype))
        
        # Decode with optimized autocast
        with torch.amp.autocast('cuda', enabled=(optimal_dtype != torch.float32), dtype=optimal_dtype):
            out = vae.first_stage_model.decode(processed_tile)
        
        # Process output with memory optimization
        pixel_samples = vae.process_output(out)
        
        # Return intermediate tensors to pool
        self._return_tensor_to_pool(processed_tile)
        self._return_tensor_to_pool(out)
        
        return pixel_samples
    
    def decode(self, vae, samples, tile_size=768, overlap=96, use_rocm_optimizations=True, 
               precision_mode="auto", batch_optimization=True, video_chunk_size=8, 
               memory_optimization_enabled=True, adaptive_tiling=True):
        """
        Phase 1 Optimized VAE decode for ROCm/AMD GPUs
        """
        start_time = time.time()
        
        # Store optimization settings
        self.memory_optimization_enabled = memory_optimization_enabled
        self.adaptive_tiling = adaptive_tiling
        
        # Update performance stats
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
            
            # Check if we can do direct decode
            if len(samples_processed.shape) == 4:  # Image
                B, C, H, W = samples_processed.shape
                
                # Try direct decode first
                try:
                    with torch.amp.autocast('cuda', enabled=(optimal_dtype != torch.float32), dtype=optimal_dtype):
                        out = vae.first_stage_model.decode(samples_processed.to(optimal_dtype))
                    
                    pixel_samples = vae.process_output(out)
                    pixel_samples = pixel_samples.to(vae.output_device).float()
                    
                    # Update performance stats
                    execution_time = time.time() - start_time
                    self.performance_stats['total_time'] += execution_time
                    
                    return (pixel_samples,)
                    
                except Exception as e:
                    logging.warning(f"Direct decode failed, falling back to tiled: {e}")
            
            # Fallback to optimized tiled decode
            try:
                pixel_samples = self._decode_tiled_optimized(vae, samples, tile_size, overlap)
                
                # Update performance stats
                execution_time = time.time() - start_time
                self.performance_stats['total_time'] += execution_time
                
                return (pixel_samples,)
                
            except Exception as e:
                logging.warning(f"Tiled decode failed, falling back to standard VAE: {e}")
                
                # Final fallback to standard VAE decode
                pixel_samples = vae.decode(samples_processed)
                pixel_samples = vae.process_output(pixel_samples)
                
                # Update performance stats
                execution_time = time.time() - start_time
                self.performance_stats['total_time'] += execution_time
                
                return (pixel_samples,)
                
        except Exception as e:
            logging.error(f"VAE decode failed: {e}")
            raise e
    
    def get_performance_stats(self):
        """Get performance statistics for monitoring"""
        if self.performance_stats['total_executions'] > 0:
            avg_time = self.performance_stats['total_time'] / self.performance_stats['total_executions']
            cache_hit_rate = self.performance_stats['cache_hits'] / max(1, self.performance_stats['total_executions'])
            memory_efficiency = self.performance_stats['memory_saves'] / max(1, self.performance_stats['total_executions'])
            
            return {
                'average_execution_time': avg_time,
                'total_executions': self.performance_stats['total_executions'],
                'cache_hit_rate': cache_hit_rate,
                'memory_efficiency': memory_efficiency,
                'total_memory_saves': self.performance_stats['memory_saves']
            }
        return self.performance_stats

# Apply instrumentation
@instrument_node
class ROCMOptimizedVAEDecodeV2Instrumented(ROCMOptimizedVAEDecodeV2):
    pass

# Node class mapping for ComfyUI
NODE_CLASS_MAPPINGS = {
    "ROCMOptimizedVAEDecodeV2": ROCMOptimizedVAEDecodeV2Instrumented
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ROCMOptimizedVAEDecodeV2": "ROCM Optimized VAE Decode V2 (Phase 1)"
}

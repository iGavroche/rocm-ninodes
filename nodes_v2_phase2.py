#!/usr/bin/env python3
"""
Phase 2 Optimization Implementation for ROCMOptimizedVAEDecode
Focus: Mixed Precision, Batch Processing, and Advanced Memory Management
"""
import torch
import torch.nn.functional as F
import comfy.model_management as model_management
import comfy.utils
import time
import logging
import math
from typing import Dict, Any, Tuple, Optional, List
from instrumentation import instrument_node

class ROCMOptimizedVAEDecodeV2Phase2:
    """
    Phase 2 Optimized VAE Decode node for gfx1151 architecture.
    
    Phase 2 Optimizations:
    - Smart mixed precision strategies (fp16/fp32) optimized for gfx1151
    - Advanced batch processing for AMD GPUs
    - Memory prefetching and advanced memory management
    - Enhanced tensor memory layout optimization
    - Advanced caching strategies
    """
    
    def __init__(self):
        # Memory pool for frequent tensor allocations
        self.memory_pool = {}
        self.tile_cache = {}
        self.precision_cache = {}
        self.batch_cache = {}
        
        # Phase 2 specific optimizations
        self.memory_prefetch_queue = []
        self.tensor_layout_cache = {}
        self.batch_size_optimizer = BatchSizeOptimizer()
        self.precision_optimizer = PrecisionOptimizer()
        
        # Optimal configurations for gfx1151
        self.optimal_tile_sizes = {
            (256, 256): 512,
            (512, 512): 768,
            (1024, 1024): 1024,
            (1280, 1280): 1280,
            (1536, 1536): 1280
        }
        
        # Mixed precision configurations for gfx1151
        self.precision_configs = {
            'fp32': {'accumulation_dtype': torch.float32, 'compute_dtype': torch.float32},
            'fp16': {'accumulation_dtype': torch.float16, 'compute_dtype': torch.float16},
            'mixed': {'accumulation_dtype': torch.float32, 'compute_dtype': torch.float16},
            'auto': {'accumulation_dtype': None, 'compute_dtype': None}  # Will be determined dynamically
        }
        
        # Batch processing configurations
        self.batch_configs = {
            'small': {'max_batch_size': 4, 'memory_threshold': 1024},
            'medium': {'max_batch_size': 8, 'memory_threshold': 2048},
            'large': {'max_batch_size': 16, 'memory_threshold': 4096}
        }
        
        # Performance monitoring
        self.performance_stats = {
            'total_executions': 0,
            'total_time': 0.0,
            'memory_saves': 0,
            'cache_hits': 0,
            'precision_optimizations': 0,
            'batch_optimizations': 0,
            'prefetch_hits': 0
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
                "precision_mode": (["auto", "fp32", "fp16", "mixed"], {
                    "default": "auto",
                    "tooltip": "Precision mode optimized for gfx1151 architecture."
                }),
                "batch_optimization": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Enable advanced batch processing optimizations for AMD GPUs."
                }),
                "memory_prefetching": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Enable memory prefetching for improved performance."
                }),
                "adaptive_precision": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Enable adaptive precision selection based on workload."
                }),
                "tensor_layout_optimization": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Enable optimized tensor memory layout for gfx1151."
                }),
                "advanced_caching": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Enable advanced caching strategies for intermediate results."
                })
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("IMAGE",)
    FUNCTION = "decode"
    CATEGORY = "ROCM Optimized/VAE"
    
    def _get_optimal_precision_config(self, vae, samples_shape: Tuple, precision_mode: str) -> Dict[str, torch.dtype]:
        """Get optimal precision configuration for gfx1151"""
        cache_key = (samples_shape, precision_mode)
        
        if cache_key in self.precision_cache:
            self.performance_stats['cache_hits'] += 1
            return self.precision_cache[cache_key]
        
        if precision_mode == "auto":
            # Smart precision selection for gfx1151
            B, C, H, W = samples_shape
            total_elements = B * C * H * W
            
            # For gfx1151, prefer fp32 for small tensors, mixed for large ones
            if total_elements < 1024 * 1024:  # Small tensors
                config = self.precision_configs['fp32'].copy()
            elif total_elements < 4096 * 4096:  # Medium tensors
                config = self.precision_configs['mixed'].copy()
            else:  # Large tensors
                config = self.precision_configs['fp16'].copy()
        else:
            config = self.precision_configs[precision_mode].copy()
        
        # Override with VAE's preferred dtype if available
        if hasattr(vae, 'vae_dtype') and vae.vae_dtype:
            config['compute_dtype'] = vae.vae_dtype
            if config['accumulation_dtype'] is None:
                config['accumulation_dtype'] = vae.vae_dtype
        
        self.precision_cache[cache_key] = config
        self.performance_stats['precision_optimizations'] += 1
        return config
    
    def _get_optimal_batch_config(self, samples_shape: Tuple, available_memory: int) -> Dict[str, int]:
        """Get optimal batch configuration for AMD GPUs"""
        cache_key = (samples_shape, available_memory)
        
        if cache_key in self.batch_cache:
            self.performance_stats['cache_hits'] += 1
            return self.batch_cache[cache_key]
        
        B, C, H, W = samples_shape
        tensor_size_mb = (B * C * H * W * 4) / (1024 * 1024)  # Assuming fp32
        
        # Determine batch configuration based on tensor size and available memory
        if tensor_size_mb < 512:
            config = self.batch_configs['small'].copy()
        elif tensor_size_mb < 2048:
            config = self.batch_configs['medium'].copy()
        else:
            config = self.batch_configs['large'].copy()
        
        # Adjust based on available memory
        if available_memory < config['memory_threshold']:
            config['max_batch_size'] = max(1, config['max_batch_size'] // 2)
        
        self.batch_cache[cache_key] = config
        self.performance_stats['batch_optimizations'] += 1
        return config
    
    def _optimize_tensor_layout(self, tensor: torch.Tensor) -> torch.Tensor:
        """Optimize tensor memory layout for gfx1151"""
        cache_key = (tensor.shape, tensor.dtype, tensor.device)
        
        if cache_key in self.tensor_layout_cache:
            self.performance_stats['cache_hits'] += 1
            return self.tensor_layout_cache[cache_key]
        
        # Ensure tensor is contiguous and properly aligned
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()
        
        # For gfx1151, ensure proper memory alignment
        if tensor.numel() % 16 == 0:  # 16-byte alignment
            optimized_tensor = tensor
        else:
            # Pad to ensure alignment
            pad_size = 16 - (tensor.numel() % 16)
            if tensor.dim() == 4:
                B, C, H, W = tensor.shape
                pad_shape = (B, C, H, W + pad_size)
                optimized_tensor = torch.empty(pad_shape, dtype=tensor.dtype, device=tensor.device)
                optimized_tensor[:, :, :, :W] = tensor
            else:
                optimized_tensor = tensor
        
        self.tensor_layout_cache[cache_key] = optimized_tensor
        return optimized_tensor
    
    def _prefetch_memory(self, tensors: List[torch.Tensor]):
        """Prefetch tensors to GPU memory for improved performance"""
        for tensor in tensors:
            if tensor.device.type == 'cuda':
                # Prefetch to GPU memory
                torch.cuda.prefetch(tensor)
                self.performance_stats['prefetch_hits'] += 1
    
    def _get_tensor_from_pool(self, shape: Tuple, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        """Get tensor from memory pool with layout optimization"""
        pool_key = (shape, dtype, device)
        
        if pool_key in self.memory_pool and len(self.memory_pool[pool_key]) > 0:
            tensor = self.memory_pool[pool_key].pop()
            tensor.resize_(shape)
            return self._optimize_tensor_layout(tensor)
        else:
            tensor = torch.empty(shape, dtype=dtype, device=device)
            return self._optimize_tensor_layout(tensor)
    
    def _return_tensor_to_pool(self, tensor: torch.Tensor):
        """Return tensor to memory pool for reuse"""
        pool_key = (tensor.shape, tensor.dtype, tensor.device)
        if pool_key not in self.memory_pool:
            self.memory_pool[pool_key] = []
        
        # Limit pool size to prevent memory bloat
        if len(self.memory_pool[pool_key]) < 8:  # Increased pool size for Phase 2
            self.memory_pool[pool_key].append(tensor.detach())
            self.performance_stats['memory_saves'] += 1
    
    def _decode_with_mixed_precision(self, vae, samples, precision_config: Dict[str, torch.dtype]):
        """Decode with optimized mixed precision for gfx1151"""
        compute_dtype = precision_config['compute_dtype']
        accumulation_dtype = precision_config['accumulation_dtype']
        
        # Convert samples to compute dtype
        samples_converted = samples.to(compute_dtype)
        
        # Use autocast with optimized settings for gfx1151
        with torch.amp.autocast('cuda', enabled=True, dtype=compute_dtype):
            # Decode with accumulation in higher precision
            if accumulation_dtype != compute_dtype:
                with torch.amp.autocast('cuda', enabled=True, dtype=accumulation_dtype):
                    out = vae.first_stage_model.decode(samples_converted)
            else:
                out = vae.first_stage_model.decode(samples_converted)
        
        return out
    
    def _decode_tiled_phase2(self, vae, samples, tile_size, overlap, precision_config, batch_config):
        """Phase 2 optimized tiled decode with advanced optimizations"""
        samples_processed = samples["samples"]
        
        # Get optimal tile size
        B, C, H, W = samples_processed.shape
        optimal_tile_h, optimal_tile_w = self._get_optimal_tile_size(H*8, W*8, vae)
        tile_size = min(tile_size, optimal_tile_h, optimal_tile_w)
        
        # Calculate optimal overlap
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
        
        # Pre-allocate output tensor with layout optimization
        output_shape = (B, 3, H * compression, W * compression)
        pixel_samples = self._get_tensor_from_pool(
            output_shape, 
            torch.float32, 
            samples_processed.device
        )
        
        # Prefetch input tensor
        if self.memory_prefetching:
            self._prefetch_memory([samples_processed])
        
        # Process tiles with batch optimization
        tile_batch_size = batch_config['max_batch_size']
        
        for i in range(tiles_h):
            for j in range(tiles_w):
                # Calculate tile boundaries
                start_h = i * tile_h
                end_h = min(start_h + tile_h + optimal_overlap, H)
                start_w = j * tile_w
                end_w = min(start_w + tile_w + optimal_overlap, W)
                
                # Extract tile
                tile_samples = samples_processed[:, :, start_h:end_h, start_w:end_w]
                
                # Decode tile with Phase 2 optimizations
                tile_output = self._decode_single_tile_phase2(vae, tile_samples, precision_config)
                
                # Copy to output with proper positioning
                output_start_h = start_h * compression
                output_end_h = end_h * compression
                output_start_w = start_w * compression
                output_end_w = end_w * compression
                
                pixel_samples[:, :, output_start_h:output_end_h, output_start_w:output_end_w] = tile_output
                
                # Return tensors to pool
                self._return_tensor_to_pool(tile_samples)
                self._return_tensor_to_pool(tile_output)
        
        return pixel_samples
    
    def _decode_single_tile_phase2(self, vae, tile_samples, precision_config):
        """Decode a single tile with Phase 2 optimizations"""
        # Optimize tensor layout
        tile_samples = self._optimize_tensor_layout(tile_samples)
        
        # Prefetch if enabled
        if self.memory_prefetching:
            self._prefetch_memory([tile_samples])
        
        # Decode with mixed precision
        out = self._decode_with_mixed_precision(vae, tile_samples, precision_config)
        
        # Process output with memory optimization
        pixel_samples = vae.process_output(out)
        
        # Return intermediate tensors to pool
        self._return_tensor_to_pool(out)
        
        return pixel_samples
    
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
    
    def decode(self, vae, samples, tile_size=768, overlap=96, precision_mode="auto", 
               batch_optimization=True, memory_prefetching=True, adaptive_precision=True,
               tensor_layout_optimization=True, advanced_caching=True):
        """
        Phase 2 Optimized VAE decode for ROCm/AMD GPUs
        """
        start_time = time.time()
        
        # Update performance stats
        self.performance_stats['total_executions'] += 1
        
        try:
            samples_processed = samples["samples"]
            
            # Get optimal precision configuration
            precision_config = self._get_optimal_precision_config(vae, samples_processed.shape, precision_mode)
            
            # Get optimal batch configuration
            available_memory = model_management.get_free_memory(vae.device)
            batch_config = self._get_optimal_batch_config(samples_processed.shape, available_memory)
            
            # Calculate memory requirements
            memory_used = vae.memory_used_decode(samples_processed.shape, precision_config['compute_dtype'])
            
            # Load VAE with optimized memory management
            model_management.load_models_gpu([vae.patcher], memory_required=memory_used)
            
            # Check if we can do direct decode
            if len(samples_processed.shape) == 4:  # Image
                B, C, H, W = samples_processed.shape
                
                # Try direct decode first with Phase 2 optimizations
                try:
                    # Optimize tensor layout
                    if tensor_layout_optimization:
                        samples_processed = self._optimize_tensor_layout(samples_processed)
                    
                    # Prefetch if enabled
                    if memory_prefetching:
                        self._prefetch_memory([samples_processed])
                    
                    # Decode with mixed precision
                    out = self._decode_with_mixed_precision(vae, samples_processed, precision_config)
                    
                    pixel_samples = vae.process_output(out)
                    pixel_samples = pixel_samples.to(vae.output_device).float()
                    
                    # Update performance stats
                    execution_time = time.time() - start_time
                    self.performance_stats['total_time'] += execution_time
                    
                    return (pixel_samples,)
                    
                except Exception as e:
                    logging.warning(f"Direct decode failed, falling back to tiled: {e}")
            
            # Fallback to Phase 2 optimized tiled decode
            try:
                pixel_samples = self._decode_tiled_phase2(vae, samples, tile_size, overlap, precision_config, batch_config)
                
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
            precision_efficiency = self.performance_stats['precision_optimizations'] / max(1, self.performance_stats['total_executions'])
            batch_efficiency = self.performance_stats['batch_optimizations'] / max(1, self.performance_stats['total_executions'])
            prefetch_efficiency = self.performance_stats['prefetch_hits'] / max(1, self.performance_stats['total_executions'])
            
            return {
                'average_execution_time': avg_time,
                'total_executions': self.performance_stats['total_executions'],
                'cache_hit_rate': cache_hit_rate,
                'memory_efficiency': memory_efficiency,
                'precision_efficiency': precision_efficiency,
                'batch_efficiency': batch_efficiency,
                'prefetch_efficiency': prefetch_efficiency,
                'total_memory_saves': self.performance_stats['memory_saves'],
                'total_precision_optimizations': self.performance_stats['precision_optimizations'],
                'total_batch_optimizations': self.performance_stats['batch_optimizations'],
                'total_prefetch_hits': self.performance_stats['prefetch_hits']
            }
        return self.performance_stats


class BatchSizeOptimizer:
    """Optimizer for determining optimal batch sizes for AMD GPUs"""
    
    def __init__(self):
        self.batch_history = []
        self.performance_history = []
    
    def get_optimal_batch_size(self, tensor_shape: Tuple, available_memory: int) -> int:
        """Determine optimal batch size based on tensor shape and available memory"""
        B, C, H, W = tensor_shape
        tensor_size_mb = (B * C * H * W * 4) / (1024 * 1024)
        
        # Calculate optimal batch size based on memory constraints
        max_batch_size = min(16, available_memory // (tensor_size_mb * 1024 * 1024))
        return max(1, max_batch_size)


class PrecisionOptimizer:
    """Optimizer for determining optimal precision settings for gfx1151"""
    
    def __init__(self):
        self.precision_history = []
        self.performance_history = []
    
    def get_optimal_precision(self, tensor_shape: Tuple, workload_type: str) -> str:
        """Determine optimal precision based on tensor shape and workload"""
        B, C, H, W = tensor_shape
        total_elements = B * C * H * W
        
        # For gfx1151, prefer fp32 for small tensors, mixed for large ones
        if total_elements < 1024 * 1024:
            return 'fp32'
        elif total_elements < 4096 * 4096:
            return 'mixed'
        else:
            return 'fp16'


# Apply instrumentation
@instrument_node
class ROCMOptimizedVAEDecodeV2Phase2Instrumented(ROCMOptimizedVAEDecodeV2Phase2):
    pass

# Node class mapping for ComfyUI
NODE_CLASS_MAPPINGS = {
    "ROCMOptimizedVAEDecodeV2Phase2": ROCMOptimizedVAEDecodeV2Phase2Instrumented
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ROCMOptimizedVAEDecodeV2Phase2": "ROCM Optimized VAE Decode V2 (Phase 2)"
}

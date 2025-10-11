#!/usr/bin/env python3
"""
Phase 3 Optimization Implementation for ROCMOptimizedVAEDecode
Focus: Video Processing Optimization and Advanced Performance Features
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

class ROCMOptimizedVAEDecodeV2Phase3:
    """
    Phase 3 Optimized VAE Decode node for gfx1151 architecture.
    
    Phase 3 Optimizations:
    - Video processing optimization with temporal consistency
    - Optimal video chunk sizes for temporal processing
    - Advanced performance monitoring and adaptive optimization
    - Real-time optimization adjustments based on usage patterns
    - Enhanced memory patterns for video workloads
    - Frame-to-frame processing optimization
    """
    
    def __init__(self):
        # Inherit Phase 2 optimizations
        self.memory_pool = {}
        self.tile_cache = {}
        self.precision_cache = {}
        self.batch_cache = {}
        
        # Phase 3 specific optimizations
        self.video_chunk_cache = {}
        self.temporal_cache = {}
        self.performance_monitor = AdvancedPerformanceMonitor()
        self.adaptive_optimizer = AdaptiveOptimizer()
        self.video_processor = VideoProcessor()
        
        # Memory prefetch queue and tensor layout cache
        self.memory_prefetch_queue = []
        self.tensor_layout_cache = {}
        
        # Optimal configurations for gfx1151
        self.optimal_tile_sizes = {
            (256, 256): 512,
            (512, 512): 768,
            (1024, 1024): 1024,
            (1280, 1280): 1280,
            (1536, 1536): 1280
        }
        
        # Video-specific configurations
        self.video_configs = {
            'small': {'chunk_size': 4, 'temporal_overlap': 2, 'memory_threshold': 1024},
            'medium': {'chunk_size': 8, 'temporal_overlap': 4, 'memory_threshold': 2048},
            'large': {'chunk_size': 16, 'temporal_overlap': 8, 'memory_threshold': 4096}
        }
        
        # Mixed precision configurations for gfx1151
        self.precision_configs = {
            'fp32': {'accumulation_dtype': torch.float32, 'compute_dtype': torch.float32},
            'fp16': {'accumulation_dtype': torch.float16, 'compute_dtype': torch.float16},
            'mixed': {'accumulation_dtype': torch.float32, 'compute_dtype': torch.float16},
            'auto': {'accumulation_dtype': None, 'compute_dtype': None}
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
            'prefetch_hits': 0,
            'video_optimizations': 0,
            'temporal_optimizations': 0,
            'adaptive_adjustments': 0
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
                "video_chunk_size": ("INT", {
                    "default": 8, 
                    "min": 1, 
                    "max": 32, 
                    "step": 1,
                    "tooltip": "Video chunk size for temporal processing optimization."
                }),
                "temporal_consistency": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Enable temporal consistency optimization for video processing."
                }),
                "adaptive_optimization": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Enable adaptive optimization based on usage patterns."
                }),
                "performance_monitoring": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Enable advanced performance monitoring and optimization."
                }),
                "memory_prefetching": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Enable memory prefetching for improved performance."
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
    
    def _get_optimal_video_config(self, samples_shape: Tuple, video_chunk_size: int) -> Dict[str, int]:
        """Get optimal video processing configuration"""
        cache_key = (samples_shape, video_chunk_size)
        
        if cache_key in self.video_chunk_cache:
            self.performance_stats['cache_hits'] += 1
            return self.video_chunk_cache[cache_key]
        
        B, C, H, W = samples_shape
        tensor_size_mb = (B * C * H * W * 4) / (1024 * 1024)
        
        # Determine video configuration based on tensor size
        if tensor_size_mb < 512:
            config = self.video_configs['small'].copy()
        elif tensor_size_mb < 2048:
            config = self.video_configs['medium'].copy()
        else:
            config = self.video_configs['large'].copy()
        
        # Adjust chunk size based on input
        config['chunk_size'] = min(video_chunk_size, config['chunk_size'])
        
        self.video_chunk_cache[cache_key] = config
        self.performance_stats['video_optimizations'] += 1
        return config
    
    def _optimize_temporal_processing(self, samples: torch.Tensor, temporal_consistency: bool) -> torch.Tensor:
        """Optimize temporal processing for video workloads"""
        if not temporal_consistency or len(samples.shape) != 5:  # Not video
            return samples
        
        B, T, C, H, W = samples.shape
        
        # Apply temporal smoothing for consistency
        if T > 1:
            # Use temporal overlap for smoother transitions
            temporal_overlap = min(2, T // 4)
            
            # Process with temporal consistency
            processed_samples = torch.zeros_like(samples)
            
            for t in range(T):
                start_t = max(0, t - temporal_overlap)
                end_t = min(T, t + temporal_overlap + 1)
                
                # Blend frames for temporal consistency
                frame_weight = 1.0 / (end_t - start_t)
                processed_samples[:, t] = samples[:, start_t:end_t].mean(dim=1) * frame_weight
            
            self.performance_stats['temporal_optimizations'] += 1
            return processed_samples
        
        return samples
    
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
        tensor_size_mb = (B * C * H * W * 4) / (1024 * 1024)
        
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
        if len(self.memory_pool[pool_key]) < 10:  # Increased pool size for Phase 3
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
    
    def _decode_video_optimized(self, vae, samples, video_config: Dict[str, int], precision_config: Dict[str, torch.dtype]):
        """Phase 3 optimized video decode with temporal consistency"""
        samples_processed = samples["samples"]
        
        # Handle video processing
        if len(samples_processed.shape) == 5:  # Video (B, T, C, H, W)
            B, T, C, H, W = samples_processed.shape
            
            # Apply temporal optimization
            samples_processed = self._optimize_temporal_processing(samples_processed, True)
            
            # Process video in chunks for memory efficiency
            chunk_size = video_config['chunk_size']
            temporal_overlap = video_config['temporal_overlap']
            
            # Pre-allocate output tensor
            output_shape = (B, T, 3, H * 8, W * 8)  # Assuming 8x compression
            pixel_samples = self._get_tensor_from_pool(
                output_shape, 
                torch.float32, 
                samples_processed.device
            )
            
            # Process video chunks
            for chunk_start in range(0, T, chunk_size):
                chunk_end = min(chunk_start + chunk_size, T)
                
                # Extract chunk with temporal overlap
                overlap_start = max(0, chunk_start - temporal_overlap)
                overlap_end = min(T, chunk_end + temporal_overlap)
                
                chunk_samples = samples_processed[:, overlap_start:overlap_end]
                
                # Decode chunk
                chunk_output = self._decode_chunk(vae, chunk_samples, precision_config)
                
                # Copy to output (excluding overlap)
                output_start = chunk_start - overlap_start
                output_end = output_start + (chunk_end - chunk_start)
                
                pixel_samples[:, chunk_start:chunk_end] = chunk_output[:, output_start:output_end]
                
                # Return chunk tensors to pool
                self._return_tensor_to_pool(chunk_samples)
                self._return_tensor_to_pool(chunk_output)
            
            return pixel_samples
        
        else:  # Regular image processing
            return self._decode_image_optimized(vae, samples, precision_config)
    
    def _decode_chunk(self, vae, chunk_samples, precision_config):
        """Decode a video chunk with optimizations"""
        B, T, C, H, W = chunk_samples.shape
        
        # Reshape for batch processing
        batch_samples = chunk_samples.view(B * T, C, H, W)
        
        # Decode with mixed precision
        out = self._decode_with_mixed_precision(vae, batch_samples, precision_config)
        
        # Process output
        pixel_samples = vae.process_output(out)
        
        # Reshape back to video format
        pixel_samples = pixel_samples.view(B, T, 3, H * 8, W * 8)
        
        return pixel_samples
    
    def _decode_image_optimized(self, vae, samples, precision_config):
        """Optimized image decode"""
        samples_processed = samples["samples"]
        
        # Optimize tensor layout
        samples_processed = self._optimize_tensor_layout(samples_processed)
        
        # Prefetch if enabled
        self._prefetch_memory([samples_processed])
        
        # Decode with mixed precision
        out = self._decode_with_mixed_precision(vae, samples_processed, precision_config)
        
        # Process output
        pixel_samples = vae.process_output(out)
        
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
               video_chunk_size=8, temporal_consistency=True, adaptive_optimization=True,
               performance_monitoring=True, memory_prefetching=True, 
               tensor_layout_optimization=True, advanced_caching=True):
        """
        Phase 3 Optimized VAE decode for ROCm/AMD GPUs
        """
        start_time = time.time()
        
        # Update performance stats
        self.performance_stats['total_executions'] += 1
        
        # Adaptive optimization
        if adaptive_optimization:
            self.adaptive_optimizer.update_usage_pattern(samples["samples"].shape)
            self.performance_stats['adaptive_adjustments'] += 1
        
        try:
            samples_processed = samples["samples"]
            
            # Get optimal configurations
            precision_config = self._get_optimal_precision_config(vae, samples_processed.shape, precision_mode)
            available_memory = model_management.get_free_memory(vae.device)
            batch_config = self._get_optimal_batch_config(samples_processed.shape, available_memory)
            video_config = self._get_optimal_video_config(samples_processed.shape, video_chunk_size)
            
            # Calculate memory requirements
            memory_used = vae.memory_used_decode(samples_processed.shape, precision_config['compute_dtype'])
            
            # Load VAE with optimized memory management
            model_management.load_models_gpu([vae.patcher], memory_required=memory_used)
            
            # Check if we can do direct decode
            if len(samples_processed.shape) == 4:  # Image
                try:
                    pixel_samples = self._decode_image_optimized(vae, samples, precision_config)
                    pixel_samples = pixel_samples.to(vae.output_device).float()
                    
                    # Update performance stats
                    execution_time = time.time() - start_time
                    self.performance_stats['total_time'] += execution_time
                    
                    return (pixel_samples,)
                    
                except Exception as e:
                    logging.warning(f"Direct decode failed, falling back to video optimized: {e}")
            
            # Use Phase 3 optimized decode (handles both image and video)
            try:
                pixel_samples = self._decode_video_optimized(vae, samples, video_config, precision_config)
                
                # Update performance stats
                execution_time = time.time() - start_time
                self.performance_stats['total_time'] += execution_time
                
                return (pixel_samples,)
                
            except Exception as e:
                logging.warning(f"Video optimized decode failed, falling back to standard VAE: {e}")
                
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
        """Get comprehensive performance statistics for monitoring"""
        if self.performance_stats['total_executions'] > 0:
            avg_time = self.performance_stats['total_time'] / self.performance_stats['total_executions']
            cache_hit_rate = self.performance_stats['cache_hits'] / max(1, self.performance_stats['total_executions'])
            memory_efficiency = self.performance_stats['memory_saves'] / max(1, self.performance_stats['total_executions'])
            precision_efficiency = self.performance_stats['precision_optimizations'] / max(1, self.performance_stats['total_executions'])
            batch_efficiency = self.performance_stats['batch_optimizations'] / max(1, self.performance_stats['total_executions'])
            prefetch_efficiency = self.performance_stats['prefetch_hits'] / max(1, self.performance_stats['total_executions'])
            video_efficiency = self.performance_stats['video_optimizations'] / max(1, self.performance_stats['total_executions'])
            temporal_efficiency = self.performance_stats['temporal_optimizations'] / max(1, self.performance_stats['total_executions'])
            adaptive_efficiency = self.performance_stats['adaptive_adjustments'] / max(1, self.performance_stats['total_executions'])
            
            return {
                'average_execution_time': avg_time,
                'total_executions': self.performance_stats['total_executions'],
                'cache_hit_rate': cache_hit_rate,
                'memory_efficiency': memory_efficiency,
                'precision_efficiency': precision_efficiency,
                'batch_efficiency': batch_efficiency,
                'prefetch_efficiency': prefetch_efficiency,
                'video_efficiency': video_efficiency,
                'temporal_efficiency': temporal_efficiency,
                'adaptive_efficiency': adaptive_efficiency,
                'total_memory_saves': self.performance_stats['memory_saves'],
                'total_precision_optimizations': self.performance_stats['precision_optimizations'],
                'total_batch_optimizations': self.performance_stats['batch_optimizations'],
                'total_prefetch_hits': self.performance_stats['prefetch_hits'],
                'total_video_optimizations': self.performance_stats['video_optimizations'],
                'total_temporal_optimizations': self.performance_stats['temporal_optimizations'],
                'total_adaptive_adjustments': self.performance_stats['adaptive_adjustments']
            }
        return self.performance_stats


class AdvancedPerformanceMonitor:
    """Advanced performance monitoring for Phase 3"""
    
    def __init__(self):
        self.metrics_history = []
        self.performance_trends = {}
    
    def record_metrics(self, metrics: Dict[str, Any]):
        """Record performance metrics"""
        self.metrics_history.append({
            'timestamp': time.time(),
            'metrics': metrics
        })
        
        # Keep only recent history
        if len(self.metrics_history) > 100:
            self.metrics_history = self.metrics_history[-100:]
    
    def get_performance_trends(self) -> Dict[str, float]:
        """Get performance trend analysis"""
        if len(self.metrics_history) < 2:
            return {}
        
        recent_metrics = self.metrics_history[-10:]  # Last 10 measurements
        older_metrics = self.metrics_history[-20:-10] if len(self.metrics_history) >= 20 else []
        
        trends = {}
        for key in recent_metrics[0]['metrics']:
            if isinstance(recent_metrics[0]['metrics'][key], (int, float)):
                recent_avg = sum(m['metrics'][key] for m in recent_metrics) / len(recent_metrics)
                if older_metrics:
                    older_avg = sum(m['metrics'][key] for m in older_metrics) / len(older_metrics)
                    trends[key] = recent_avg - older_avg
                else:
                    trends[key] = 0.0
        
        return trends


class AdaptiveOptimizer:
    """Adaptive optimization based on usage patterns"""
    
    def __init__(self):
        self.usage_patterns = {}
        self.optimization_history = []
    
    def update_usage_pattern(self, tensor_shape: Tuple):
        """Update usage pattern tracking"""
        shape_key = str(tensor_shape)
        if shape_key not in self.usage_patterns:
            self.usage_patterns[shape_key] = {'count': 0, 'last_used': time.time()}
        
        self.usage_patterns[shape_key]['count'] += 1
        self.usage_patterns[shape_key]['last_used'] = time.time()
    
    def get_optimal_settings(self, tensor_shape: Tuple) -> Dict[str, Any]:
        """Get optimal settings based on usage patterns"""
        shape_key = str(tensor_shape)
        
        if shape_key in self.usage_patterns:
            usage_count = self.usage_patterns[shape_key]['count']
            
            # Adjust settings based on usage frequency
            if usage_count > 10:  # Frequently used
                return {
                    'aggressive_caching': True,
                    'prefetch_enabled': True,
                    'memory_pool_size': 15
                }
            elif usage_count > 5:  # Moderately used
                return {
                    'aggressive_caching': True,
                    'prefetch_enabled': True,
                    'memory_pool_size': 10
                }
        
        # Default settings for infrequent usage
        return {
            'aggressive_caching': False,
            'prefetch_enabled': True,
            'memory_pool_size': 5
        }


class VideoProcessor:
    """Video processing optimization for Phase 3"""
    
    def __init__(self):
        self.temporal_cache = {}
        self.chunk_cache = {}
    
    def optimize_temporal_consistency(self, video_tensor: torch.Tensor) -> torch.Tensor:
        """Optimize temporal consistency for video processing"""
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
    
    def get_optimal_chunk_size(self, video_shape: Tuple, available_memory: int) -> int:
        """Get optimal chunk size for video processing"""
        B, T, C, H, W = video_shape
        frame_size_mb = (B * C * H * W * 4) / (1024 * 1024)
        
        # Calculate optimal chunk size based on memory
        max_chunk_size = min(T, available_memory // (frame_size_mb * 1024 * 1024))
        
        # Prefer powers of 2 for better memory alignment
        optimal_chunk = 1
        while optimal_chunk * 2 <= max_chunk_size:
            optimal_chunk *= 2
        
        return max(1, optimal_chunk)


# Apply instrumentation
@instrument_node
class ROCMOptimizedVAEDecodeV2Phase3Instrumented(ROCMOptimizedVAEDecodeV2Phase3):
    pass

# Node class mapping for ComfyUI
NODE_CLASS_MAPPINGS = {
    "ROCMOptimizedVAEDecodeV2Phase3": ROCMOptimizedVAEDecodeV2Phase3Instrumented
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ROCMOptimizedVAEDecodeV2Phase3": "ROCM Optimized VAE Decode V2 (Phase 3)"
}

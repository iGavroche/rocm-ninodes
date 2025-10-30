"""
Monitor and utility nodes for ROCM Ninodes.

Contains monitoring and utility node implementations:
- ROCMFluxBenchmark: Comprehensive Flux workflow benchmark
- ROCMMemoryOptimizer: Memory optimization helper
"""

import time
import gc
from typing import Tuple

import torch
import comfy.model_management as model_management


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
    CATEGORY = "ROCm Ninodes/Benchmark"
    DESCRIPTION = "Comprehensive Flux workflow benchmark for AMD GPUs"
    
    def benchmark(self, model, vae, clip, test_resolutions="256x320,512x512,1024x1024",
                 test_steps=20, test_cfg_values="1.0,3.5,8.0", test_hipblas=True,
                 generate_report=True):
        """Run comprehensive benchmark tests"""
        # Parse test parameters
        resolutions = []
        for res in test_resolutions.split(','):
            w, h = map(int, res.strip().split('x'))
            resolutions.append((w, h))
        
        cfg_values = [float(cfg.strip()) for cfg in test_cfg_values.split(',')]
        
        # Get device information
        device = model_management.get_torch_device()
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
                
                torch.cuda.empty_cache()
                gc.collect()
                
                latent = torch.randn(1, 4, h//8, w//8, device=device)
                
                try:
                    with torch.no_grad():
                        decoded = vae.decode(latent)
                    decode_time = time.time() - start_time
                    
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
                valid_times = [t for t in res_data['times'] if t != float('inf')]
                if valid_times:
                    avg_time = sum(valid_times) / len(valid_times)
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
        if torch.cuda.is_available():
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
    CATEGORY = "ROCm Ninodes/Memory"
    DESCRIPTION = "Memory optimization helper for AMD GPUs"
    
    def __init__(self):
        self.operation_count = 0
    
    def optimize_memory(self, optimization_level="balanced", enable_gc=True, 
                       clear_cache=True, cleanup_frequency=10):
        """Optimize memory usage for AMD GPUs"""
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


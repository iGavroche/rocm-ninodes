#!/usr/bin/env python3
"""
Automated benchmark suite for ROCM Ninodes
"""
import os
import sys
import time
import json
import argparse
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
from unittest.mock import Mock, patch, MagicMock


@dataclass
class BenchmarkResult:
    """Data class for benchmark results"""
    node_name: str
    test_name: str
    resolution: str
    time_seconds: float
    memory_mb: float
    success: bool
    error_message: str = ""
    timestamp: str = ""
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark runs"""
    test_resolutions: List[str] = None
    test_steps: List[int] = None
    test_cfg_values: List[float] = None
    test_precision_modes: List[str] = None
    test_rocm_optimizations: List[bool] = None
    test_memory_optimizations: List[bool] = None
    iterations_per_test: int = 3
    warmup_iterations: int = 1
    
    def __post_init__(self):
        if self.test_resolutions is None:
            self.test_resolutions = ["512x512", "1024x1024"]
        if self.test_steps is None:
            self.test_steps = [4, 20]
        if self.test_cfg_values is None:
            self.test_cfg_values = [1.0, 8.0]
        if self.test_precision_modes is None:
            self.test_precision_modes = ["auto", "fp32", "fp16"]
        if self.test_rocm_optimizations is None:
            self.test_rocm_optimizations = [True, False]
        if self.test_memory_optimizations is None:
            self.test_memory_optimizations = [True, False]


class BenchmarkRunner:
    """Main benchmark runner class"""
    
    def __init__(self, config: BenchmarkConfig = None):
        self.config = config or BenchmarkConfig()
        self.results: List[BenchmarkResult] = []
        self.mock_objects = self._setup_mock_objects()
    
    def _setup_mock_objects(self) -> Dict[str, Any]:
        """Setup mock objects for testing"""
        mock_vae = Mock()
        mock_vae.device = torch.device('cuda:0')
        mock_vae.output_device = torch.device('cuda:0')
        mock_vae.vae_dtype = torch.float32
        mock_vae.first_stage_model = Mock()
        mock_vae.first_stage_model.to.return_value = mock_vae.first_stage_model
        
        mock_model = Mock()
        mock_model.model_dtype.return_value = torch.device("cuda:0")
        
        mock_clip = Mock()
        
        # Mock latents
        mock_latent_image = {"samples": torch.randn(1, 4, 64, 64)}
        mock_latent_video = {"samples": torch.randn(1, 4, 16, 64, 64)}
        
        # Mock conditioning
        mock_conditioning = [{"pooled_output": torch.randn(1, 1024)}]
        
        return {
            'vae': mock_vae,
            'model': mock_model,
            'clip': mock_clip,
            'latent_image': mock_latent_image,
            'latent_video': mock_latent_video,
            'conditioning': mock_conditioning
        }
    
    def _measure_memory(self) -> float:
        """Measure current GPU memory usage in MB"""
        try:
            if torch.cuda.is_available():
                return torch.cuda.memory_allocated() / 1024 / 1024
            else:
                return 0.0
        except:
            return 0.0
    
    def _time_function(self, func, *args, **kwargs) -> Tuple[Any, float]:
        """Time a function execution"""
        start_time = time.time()
        start_memory = self._measure_memory()
        
        try:
            result = func(*args, **kwargs)
            success = True
            error_message = ""
        except Exception as e:
            result = None
            success = False
            error_message = str(e)
        
        end_time = time.time()
        end_memory = self._measure_memory()
        
        execution_time = end_time - start_time
        memory_used = max(0, end_memory - start_memory)
        
        return result, execution_time, memory_used, success, error_message
    
    def benchmark_vae_decode(self) -> List[BenchmarkResult]:
        """Benchmark VAE decode nodes"""
        results = []
        
        from nodes import ROCMOptimizedVAEDecode, ROCMOptimizedVAEDecodeSimple
        
        nodes_to_test = [
            ("ROCMOptimizedVAEDecode", ROCMOptimizedVAEDecode),
            ("ROCMOptimizedVAEDecodeSimple", ROCMOptimizedVAEDecodeSimple)
        ]
        
        for node_name, node_class in nodes_to_test:
            for resolution in self.config.test_resolutions:
                # Parse resolution
                width, height = map(int, resolution.split('x'))
                latent_height = height // 8
                latent_width = width // 8
                
                # Create test latent
                test_latent = {"samples": torch.randn(1, 4, latent_height, latent_width)}
                
                for precision_mode in self.config.test_precision_modes:
                    for rocm_opt in self.config.test_rocm_optimizations:
                        for mem_opt in self.config.test_memory_optimizations:
                            # Warmup
                            for _ in range(self.config.warmup_iterations):
                                try:
                                    node = node_class()
                                    with patch.object(self.mock_objects['vae'], 'decode') as mock_decode:
                                        mock_decode.return_value = torch.randn(1, 3, height, width)
                                        node.decode(
                                            self.mock_objects['vae'],
                                            test_latent,
                                            precision_mode=precision_mode,
                                            use_rocm_optimizations=rocm_opt,
                                            memory_optimization=mem_opt
                                        )
                                except:
                                    pass
                            
                            # Benchmark
                            times = []
                            memories = []
                            successes = []
                            errors = []
                            
                            for _ in range(self.config.iterations_per_test):
                                try:
                                    node = node_class()
                                    with patch.object(self.mock_objects['vae'], 'decode') as mock_decode:
                                        mock_decode.return_value = torch.randn(1, 3, height, width)
                                        
                                        result, exec_time, mem_used, success, error = self._time_function(
                                            node.decode,
                                            self.mock_objects['vae'],
                                            test_latent,
                                            precision_mode=precision_mode,
                                            use_rocm_optimizations=rocm_opt,
                                            memory_optimization=mem_opt
                                        )
                                        
                                        times.append(exec_time)
                                        memories.append(mem_used)
                                        successes.append(success)
                                        if error:
                                            errors.append(error)
                                
                                except Exception as e:
                                    times.append(0)
                                    memories.append(0)
                                    successes.append(False)
                                    errors.append(str(e))
                            
                            # Calculate averages
                            avg_time = sum(times) / len(times) if times else 0
                            avg_memory = sum(memories) / len(memories) if memories else 0
                            success_rate = sum(successes) / len(successes) if successes else 0
                            
                            result = BenchmarkResult(
                                node_name=node_name,
                                test_name=f"vae_decode_{resolution}_{precision_mode}_rocm{rocm_opt}_mem{mem_opt}",
                                resolution=resolution,
                                time_seconds=avg_time,
                                memory_mb=avg_memory,
                                success=success_rate > 0.5,
                                error_message="; ".join(errors) if errors else ""
                            )
                            
                            results.append(result)
                            print(f"✓ {node_name} {resolution} {precision_mode}: {avg_time:.3f}s, {avg_memory:.1f}MB")
        
        return results
    
    def benchmark_samplers(self) -> List[BenchmarkResult]:
        """Benchmark sampler nodes"""
        results = []
        
        from nodes import ROCMOptimizedKSampler, ROCMOptimizedKSamplerAdvanced
        
        nodes_to_test = [
            ("ROCMOptimizedKSampler", ROCMOptimizedKSampler),
            ("ROCMOptimizedKSamplerAdvanced", ROCMOptimizedKSamplerAdvanced)
        ]
        
        for node_name, node_class in nodes_to_test:
            for resolution in self.config.test_resolutions:
                # Parse resolution
                width, height = map(int, resolution.split('x'))
                latent_height = height // 8
                latent_width = width // 8
                
                # Create test latent
                test_latent = {"samples": torch.randn(1, 4, latent_height, latent_width)}
                
                for steps in self.config.test_steps:
                    for cfg in self.config.test_cfg_values:
                        for precision_mode in self.config.test_precision_modes:
                            for rocm_opt in self.config.test_rocm_optimizations:
                                # Warmup
                                for _ in range(self.config.warmup_iterations):
                                    try:
                                        node = node_class()
                                        with patch('nodes.comfy.sample') as mock_sample, \
                                             patch('nodes.latent_preview') as mock_latent_preview:
                                            
                                            mock_sample.fix_empty_latent_channels.return_value = test_latent["samples"]
                                            mock_sample.prepare_noise.return_value = torch.randn_like(test_latent["samples"])
                                            mock_sample.sample.return_value = torch.randn_like(test_latent["samples"])
                                            mock_latent_preview.prepare_callback.return_value = None
                                            
                                            if node_name == "ROCMOptimizedKSampler":
                                                node.sample(
                                                    self.mock_objects['model'],
                                                    12345, steps, cfg, "euler", "simple",
                                                    self.mock_objects['conditioning'],
                                                    self.mock_objects['conditioning'],
                                                    test_latent,
                                                    precision_mode=precision_mode,
                                                    use_rocm_optimizations=rocm_opt
                                                )
                                            else:  # Advanced
                                                node.sample(
                                                    self.mock_objects['model'],
                                                    "enable", 12345, steps, cfg, "euler", "simple",
                                                    self.mock_objects['conditioning'],
                                                    self.mock_objects['conditioning'],
                                                    test_latent,
                                                    0, steps, "disable",
                                                    precision_mode=precision_mode,
                                                    use_rocm_optimizations=rocm_opt
                                                )
                                    except:
                                        pass
                                
                                # Benchmark
                                times = []
                                memories = []
                                successes = []
                                errors = []
                                
                                for _ in range(self.config.iterations_per_test):
                                    try:
                                        node = node_class()
                                        with patch('nodes.comfy.sample') as mock_sample, \
                                             patch('nodes.latent_preview') as mock_latent_preview:
                                            
                                            mock_sample.fix_empty_latent_channels.return_value = test_latent["samples"]
                                            mock_sample.prepare_noise.return_value = torch.randn_like(test_latent["samples"])
                                            mock_sample.sample.return_value = torch.randn_like(test_latent["samples"])
                                            mock_latent_preview.prepare_callback.return_value = None
                                            
                                            if node_name == "ROCMOptimizedKSampler":
                                                result, exec_time, mem_used, success, error = self._time_function(
                                                    node.sample,
                                                    self.mock_objects['model'],
                                                    12345, steps, cfg, "euler", "simple",
                                                    self.mock_objects['conditioning'],
                                                    self.mock_objects['conditioning'],
                                                    test_latent,
                                                    precision_mode=precision_mode,
                                                    use_rocm_optimizations=rocm_opt
                                                )
                                            else:  # Advanced
                                                result, exec_time, mem_used, success, error = self._time_function(
                                                    node.sample,
                                                    self.mock_objects['model'],
                                                    "enable", 12345, steps, cfg, "euler", "simple",
                                                    self.mock_objects['conditioning'],
                                                    self.mock_objects['conditioning'],
                                                    test_latent,
                                                    0, steps, "disable",
                                                    precision_mode=precision_mode,
                                                    use_rocm_optimizations=rocm_opt
                                                )
                                            
                                            times.append(exec_time)
                                            memories.append(mem_used)
                                            successes.append(success)
                                            if error:
                                                errors.append(error)
                                    
                                    except Exception as e:
                                        times.append(0)
                                        memories.append(0)
                                        successes.append(False)
                                        errors.append(str(e))
                                
                                # Calculate averages
                                avg_time = sum(times) / len(times) if times else 0
                                avg_memory = sum(memories) / len(memories) if memories else 0
                                success_rate = sum(successes) / len(successes) if successes else 0
                                
                                result = BenchmarkResult(
                                    node_name=node_name,
                                    test_name=f"sampler_{resolution}_{steps}steps_{cfg}cfg_{precision_mode}_rocm{rocm_opt}",
                                    resolution=resolution,
                                    time_seconds=avg_time,
                                    memory_mb=avg_memory,
                                    success=success_rate > 0.5,
                                    error_message="; ".join(errors) if errors else ""
                                )
                                
                                results.append(result)
                                print(f"✓ {node_name} {resolution} {steps}steps {cfg}cfg: {avg_time:.3f}s, {avg_memory:.1f}MB")
        
        return results
    
    def benchmark_checkpoint_loader(self) -> List[BenchmarkResult]:
        """Benchmark checkpoint loader"""
        results = []
        
        from nodes import ROCMOptimizedCheckpointLoader
        
        node_name = "ROCMOptimizedCheckpointLoader"
        
        for precision_mode in self.config.test_precision_modes:
            for rocm_opt in self.config.test_rocm_optimizations:
                # Warmup
                for _ in range(self.config.warmup_iterations):
                    try:
                        node = ROCMOptimizedCheckpointLoader()
                        with patch('nodes.folder_paths') as mock_folder_paths, \
                             patch('nodes.comfy.sd') as mock_comfy_sd:
                            
                            mock_folder_paths.get_full_path_or_raise.return_value = "/path/to/test.safetensors"
                            mock_comfy_sd.load_checkpoint_guess_config.return_value = (
                                self.mock_objects['model'],
                                self.mock_objects['clip'],
                                self.mock_objects['vae']
                            )
                            
                            node.load_checkpoint(
                                "test.safetensors",
                                precision_mode=precision_mode,
                                use_rocm_optimizations=rocm_opt
                            )
                    except:
                        pass
                
                # Benchmark
                times = []
                memories = []
                successes = []
                errors = []
                
                for _ in range(self.config.iterations_per_test):
                    try:
                        node = ROCMOptimizedCheckpointLoader()
                        with patch('nodes.folder_paths') as mock_folder_paths, \
                             patch('nodes.comfy.sd') as mock_comfy_sd:
                            
                            mock_folder_paths.get_full_path_or_raise.return_value = "/path/to/test.safetensors"
                            mock_comfy_sd.load_checkpoint_guess_config.return_value = (
                                self.mock_objects['model'],
                                self.mock_objects['clip'],
                                self.mock_objects['vae']
                            )
                            
                            result, exec_time, mem_used, success, error = self._time_function(
                                node.load_checkpoint,
                                "test.safetensors",
                                precision_mode=precision_mode,
                                use_rocm_optimizations=rocm_opt
                            )
                            
                            times.append(exec_time)
                            memories.append(mem_used)
                            successes.append(success)
                            if error:
                                errors.append(error)
                    
                    except Exception as e:
                        times.append(0)
                        memories.append(0)
                        successes.append(False)
                        errors.append(str(e))
                
                # Calculate averages
                avg_time = sum(times) / len(times) if times else 0
                avg_memory = sum(memories) / len(memories) if memories else 0
                success_rate = sum(successes) / len(successes) if successes else 0
                
                result = BenchmarkResult(
                    node_name=node_name,
                    test_name=f"checkpoint_loader_{precision_mode}_rocm{rocm_opt}",
                    resolution="N/A",
                    time_seconds=avg_time,
                    memory_mb=avg_memory,
                    success=success_rate > 0.5,
                    error_message="; ".join(errors) if errors else ""
                )
                
                results.append(result)
                print(f"✓ {node_name} {precision_mode}: {avg_time:.3f}s, {avg_memory:.1f}MB")
        
        return results
    
    def benchmark_workflows(self) -> List[BenchmarkResult]:
        """Benchmark complete workflows"""
        results = []
        
        # Flux workflow benchmark
        flux_result = self._benchmark_flux_workflow()
        if flux_result:
            results.append(flux_result)
        
        # WAN workflow benchmark
        wan_result = self._benchmark_wan_workflow()
        if wan_result:
            results.append(wan_result)
        
        return results
    
    def _benchmark_flux_workflow(self) -> BenchmarkResult:
        """Benchmark Flux workflow"""
        from nodes import ROCMOptimizedCheckpointLoader, ROCMOptimizedKSampler, ROCMOptimizedVAEDecode
        
        try:
            # Warmup
            for _ in range(self.config.warmup_iterations):
                try:
                    with patch('nodes.folder_paths') as mock_folder_paths, \
                         patch('nodes.comfy.sd') as mock_comfy_sd, \
                         patch('nodes.comfy.sample') as mock_sample, \
                         patch('nodes.latent_preview') as mock_latent_preview:
                        
                        mock_folder_paths.get_full_path_or_raise.return_value = "/path/to/flux1-dev.safetensors"
                        mock_comfy_sd.load_checkpoint_guess_config.return_value = (
                            self.mock_objects['model'],
                            self.mock_objects['clip'],
                            self.mock_objects['vae']
                        )
                        
                        mock_sample.fix_empty_latent_channels.return_value = self.mock_objects['latent_image']["samples"]
                        mock_sample.prepare_noise.return_value = torch.randn_like(self.mock_objects['latent_image']["samples"])
                        mock_sample.sample.return_value = torch.randn_like(self.mock_objects['latent_image']["samples"])
                        mock_latent_preview.prepare_callback.return_value = None
                        
                        with patch.object(self.mock_objects['vae'], 'decode') as mock_decode:
                            mock_decode.return_value = torch.randn(1, 3, 1024, 1024)
                            
                            self._run_flux_workflow()
                except:
                    pass
            
            # Benchmark
            times = []
            memories = []
            successes = []
            errors = []
            
            for _ in range(self.config.iterations_per_test):
                try:
                    result, exec_time, mem_used, success, error = self._time_function(
                        self._run_flux_workflow
                    )
                    
                    times.append(exec_time)
                    memories.append(mem_used)
                    successes.append(success)
                    if error:
                        errors.append(error)
                
                except Exception as e:
                    times.append(0)
                    memories.append(0)
                    successes.append(False)
                    errors.append(str(e))
            
            # Calculate averages
            avg_time = sum(times) / len(times) if times else 0
            avg_memory = sum(memories) / len(memories) if memories else 0
            success_rate = sum(successes) / len(successes) if successes else 0
            
            return BenchmarkResult(
                node_name="FluxWorkflow",
                test_name="flux_complete_workflow",
                resolution="1024x1024",
                time_seconds=avg_time,
                memory_mb=avg_memory,
                success=success_rate > 0.5,
                error_message="; ".join(errors) if errors else ""
            )
        
        except Exception as e:
            return BenchmarkResult(
                node_name="FluxWorkflow",
                test_name="flux_complete_workflow",
                resolution="1024x1024",
                time_seconds=0,
                memory_mb=0,
                success=False,
                error_message=str(e)
            )
    
    def _benchmark_wan_workflow(self) -> BenchmarkResult:
        """Benchmark WAN workflow"""
        from nodes import ROCMOptimizedCheckpointLoader, ROCMOptimizedKSampler, ROCMOptimizedVAEDecode
        
        try:
            # Warmup
            for _ in range(self.config.warmup_iterations):
                try:
                    with patch('nodes.folder_paths') as mock_folder_paths, \
                         patch('nodes.comfy.sd') as mock_comfy_sd, \
                         patch('nodes.comfy.sample') as mock_sample, \
                         patch('nodes.latent_preview') as mock_latent_preview:
                        
                        mock_folder_paths.get_full_path_or_raise.return_value = "/path/to/wan_model.safetensors"
                        mock_comfy_sd.load_checkpoint_guess_config.return_value = (
                            self.mock_objects['model'],
                            self.mock_objects['clip'],
                            self.mock_objects['vae']
                        )
                        
                        mock_sample.fix_empty_latent_channels.return_value = self.mock_objects['latent_video']["samples"]
                        mock_sample.prepare_noise.return_value = torch.randn_like(self.mock_objects['latent_video']["samples"])
                        mock_sample.sample.return_value = torch.randn_like(self.mock_objects['latent_video']["samples"])
                        mock_latent_preview.prepare_callback.return_value = None
                        
                        with patch.object(self.mock_objects['vae'], 'decode') as mock_decode:
                            batch_size, channels, temporal, height, width = self.mock_objects['latent_video']["samples"].shape
                            mock_decode.return_value = torch.randn(batch_size * temporal, 3, height * 8, width * 8)
                            
                            self._run_wan_workflow()
                except:
                    pass
            
            # Benchmark
            times = []
            memories = []
            successes = []
            errors = []
            
            for _ in range(self.config.iterations_per_test):
                try:
                    result, exec_time, mem_used, success, error = self._time_function(
                        self._run_wan_workflow
                    )
                    
                    times.append(exec_time)
                    memories.append(mem_used)
                    successes.append(success)
                    if error:
                        errors.append(error)
                
                except Exception as e:
                    times.append(0)
                    memories.append(0)
                    successes.append(False)
                    errors.append(str(e))
            
            # Calculate averages
            avg_time = sum(times) / len(times) if times else 0
            avg_memory = sum(memories) / len(memories) if memories else 0
            success_rate = sum(successes) / len(successes) if successes else 0
            
            return BenchmarkResult(
                node_name="WANWorkflow",
                test_name="wan_complete_workflow",
                resolution="512x512x16",
                time_seconds=avg_time,
                memory_mb=avg_memory,
                success=success_rate > 0.5,
                error_message="; ".join(errors) if errors else ""
            )
        
        except Exception as e:
            return BenchmarkResult(
                node_name="WANWorkflow",
                test_name="wan_complete_workflow",
                resolution="512x512x16",
                time_seconds=0,
                memory_mb=0,
                success=False,
                error_message=str(e)
            )
    
    def _run_flux_workflow(self):
        """Run Flux workflow"""
        from nodes import ROCMOptimizedCheckpointLoader, ROCMOptimizedKSampler, ROCMOptimizedVAEDecode
        
        with patch('nodes.folder_paths') as mock_folder_paths, \
             patch('nodes.comfy.sd') as mock_comfy_sd, \
             patch('nodes.comfy.sample') as mock_sample, \
             patch('nodes.latent_preview') as mock_latent_preview:
            
            mock_folder_paths.get_full_path_or_raise.return_value = "/path/to/flux1-dev.safetensors"
            mock_comfy_sd.load_checkpoint_guess_config.return_value = (
                self.mock_objects['model'],
                self.mock_objects['clip'],
                self.mock_objects['vae']
            )
            
            mock_sample.fix_empty_latent_channels.return_value = self.mock_objects['latent_image']["samples"]
            mock_sample.prepare_noise.return_value = torch.randn_like(self.mock_objects['latent_image']["samples"])
            mock_sample.sample.return_value = torch.randn_like(self.mock_objects['latent_image']["samples"])
            mock_latent_preview.prepare_callback.return_value = None
            
            with patch.object(self.mock_objects['vae'], 'decode') as mock_decode:
                mock_decode.return_value = torch.randn(1, 3, 1024, 1024)
                
                # Load checkpoint
                loader = ROCMOptimizedCheckpointLoader()
                model, clip, vae = loader.load_checkpoint("flux1-dev.safetensors", optimize_for_flux=True)
                
                # Sample
                sampler = ROCMOptimizedKSampler()
                latent_result = sampler.sample(
                    model=model,
                    seed=12345,
                    steps=20,
                    cfg=1.0,
                    sampler_name="euler",
                    scheduler="simple",
                    positive=self.mock_objects['conditioning'],
                    negative=self.mock_objects['conditioning'],
                    latent_image=self.mock_objects['latent_image'],
                    denoise=1.0,
                    flux_optimization=True
                )
                
                # VAE decode
                vae_decoder = ROCMOptimizedVAEDecode()
                decoded_result = vae_decoder.decode(
                    vae=vae,
                    samples=latent_result[0],
                    flux_vae_optimization=True
                )
                
                return decoded_result
    
    def _run_wan_workflow(self):
        """Run WAN workflow"""
        from nodes import ROCMOptimizedCheckpointLoader, ROCMOptimizedKSampler, ROCMOptimizedVAEDecode
        
        with patch('nodes.folder_paths') as mock_folder_paths, \
             patch('nodes.comfy.sd') as mock_comfy_sd, \
             patch('nodes.comfy.sample') as mock_sample, \
             patch('nodes.latent_preview') as mock_latent_preview:
            
            mock_folder_paths.get_full_path_or_raise.return_value = "/path/to/wan_model.safetensors"
            mock_comfy_sd.load_checkpoint_guess_config.return_value = (
                self.mock_objects['model'],
                self.mock_objects['clip'],
                self.mock_objects['vae']
            )
            
            mock_sample.fix_empty_latent_channels.return_value = self.mock_objects['latent_video']["samples"]
            mock_sample.prepare_noise.return_value = torch.randn_like(self.mock_objects['latent_video']["samples"])
            mock_sample.sample.return_value = torch.randn_like(self.mock_objects['latent_video']["samples"])
            mock_latent_preview.prepare_callback.return_value = None
            
            with patch.object(self.mock_objects['vae'], 'decode') as mock_decode:
                batch_size, channels, temporal, height, width = self.mock_objects['latent_video']["samples"].shape
                mock_decode.return_value = torch.randn(batch_size * temporal, 3, height * 8, width * 8)
                
                # Load checkpoint
                loader = ROCMOptimizedCheckpointLoader()
                model, clip, vae = loader.load_checkpoint("wan_model.safetensors")
                
                # Sample
                sampler = ROCMOptimizedKSampler()
                latent_result = sampler.sample(
                    model=model,
                    seed=12345,
                    steps=4,
                    cfg=1.0,
                    sampler_name="euler",
                    scheduler="simple",
                    positive=self.mock_objects['conditioning'],
                    negative=self.mock_objects['conditioning'],
                    latent_image=self.mock_objects['latent_video'],
                    denoise=1.0
                )
                
                # VAE decode
                vae_decoder = ROCMOptimizedVAEDecode()
                decoded_result = vae_decoder.decode(
                    vae=vae,
                    samples=latent_result[0]
                )
                
                return decoded_result
    
    def run_all_benchmarks(self) -> List[BenchmarkResult]:
        """Run all benchmarks"""
        print("🚀 Starting ROCM Ninodes Benchmark Suite")
        print("=" * 50)
        
        all_results = []
        
        # VAE Decode benchmarks
        print("\n📊 Benchmarking VAE Decode nodes...")
        vae_results = self.benchmark_vae_decode()
        all_results.extend(vae_results)
        
        # Sampler benchmarks
        print("\n📊 Benchmarking Sampler nodes...")
        sampler_results = self.benchmark_samplers()
        all_results.extend(sampler_results)
        
        # Checkpoint loader benchmarks
        print("\n📊 Benchmarking Checkpoint Loader...")
        loader_results = self.benchmark_checkpoint_loader()
        all_results.extend(loader_results)
        
        # Workflow benchmarks
        print("\n📊 Benchmarking Complete Workflows...")
        workflow_results = self.benchmark_workflows()
        all_results.extend(workflow_results)
        
        self.results = all_results
        return all_results
    
    def generate_report(self, output_file: str = None) -> str:
        """Generate benchmark report"""
        if not self.results:
            return "No benchmark results available."
        
        # Group results by node
        node_results = {}
        for result in self.results:
            if result.node_name not in node_results:
                node_results[result.node_name] = []
            node_results[result.node_name].append(result)
        
        # Generate markdown report
        report = []
        report.append("# ROCM Ninodes Benchmark Report")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Summary table
        report.append("## Summary")
        report.append("")
        report.append("| Node | Tests | Avg Time (s) | Avg Memory (MB) | Success Rate |")
        report.append("|------|-------|--------------|-----------------|--------------|")
        
        for node_name, results in node_results.items():
            if results:
                avg_time = sum(r.time_seconds for r in results) / len(results)
                avg_memory = sum(r.memory_mb for r in results) / len(results)
                success_rate = sum(1 for r in results if r.success) / len(results) * 100
                
                report.append(f"| {node_name} | {len(results)} | {avg_time:.3f} | {avg_memory:.1f} | {success_rate:.1f}% |")
        
        report.append("")
        
        # Detailed results by node
        for node_name, results in node_results.items():
            report.append(f"## {node_name}")
            report.append("")
            
            # Sort by time
            results.sort(key=lambda x: x.time_seconds)
            
            report.append("| Test | Resolution | Time (s) | Memory (MB) | Success |")
            report.append("|------|------------|----------|-------------|---------|")
            
            for result in results:
                success_icon = "✅" if result.success else "❌"
                report.append(f"| {result.test_name} | {result.resolution} | {result.time_seconds:.3f} | {result.memory_mb:.1f} | {success_icon} |")
            
            report.append("")
        
        # Performance recommendations
        report.append("## Performance Recommendations")
        report.append("")
        
        # Find fastest configurations
        fastest_results = {}
        for node_name, results in node_results.items():
            if results:
                fastest = min(results, key=lambda x: x.time_seconds)
                fastest_results[node_name] = fastest
        
        if fastest_results:
            report.append("### Fastest Configurations")
            report.append("")
            for node_name, result in fastest_results.items():
                report.append(f"- **{node_name}**: {result.test_name} ({result.time_seconds:.3f}s)")
            report.append("")
        
        # Error analysis
        error_results = [r for r in self.results if not r.success and r.error_message]
        if error_results:
            report.append("### Error Analysis")
            report.append("")
            for result in error_results:
                report.append(f"- **{result.node_name}** ({result.test_name}): {result.error_message}")
            report.append("")
        
        report_text = "\n".join(report)
        
        # Save to file if specified
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report_text)
            print(f"📄 Report saved to: {output_file}")
        
        return report_text
    
    def save_results_json(self, output_file: str):
        """Save results as JSON"""
        results_data = {
            'timestamp': datetime.now().isoformat(),
            'config': asdict(self.config),
            'results': [asdict(result) for result in self.results]
        }
        
        with open(output_file, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        print(f"📊 Results saved to: {output_file}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='ROCM Ninodes Benchmark Suite')
    parser.add_argument('--output', '-o', default='benchmark_report.md', help='Output file for report')
    parser.add_argument('--json', '-j', default='benchmark_results.json', help='Output file for JSON results')
    parser.add_argument('--resolutions', nargs='+', default=['512x512', '1024x1024'], help='Test resolutions')
    parser.add_argument('--steps', nargs='+', type=int, default=[4, 20], help='Test steps')
    parser.add_argument('--cfg', nargs='+', type=float, default=[1.0, 8.0], help='Test CFG values')
    parser.add_argument('--precision', nargs='+', default=['auto', 'fp32', 'fp16'], help='Test precision modes')
    parser.add_argument('--iterations', type=int, default=3, help='Iterations per test')
    parser.add_argument('--warmup', type=int, default=1, help='Warmup iterations')
    
    args = parser.parse_args()
    
    # Create config
    config = BenchmarkConfig(
        test_resolutions=args.resolutions,
        test_steps=args.steps,
        test_cfg_values=args.cfg,
        test_precision_modes=args.precision,
        iterations_per_test=args.iterations,
        warmup_iterations=args.warmup
    )
    
    # Run benchmarks
    runner = BenchmarkRunner(config)
    results = runner.run_all_benchmarks()
    
    # Generate report
    report = runner.generate_report(args.output)
    runner.save_results_json(args.json)
    
    print("\n" + "=" * 50)
    print("🎉 Benchmark suite completed!")
    print(f"📊 Total tests: {len(results)}")
    print(f"✅ Successful: {sum(1 for r in results if r.success)}")
    print(f"❌ Failed: {sum(1 for r in results if not r.success)}")
    print(f"📄 Report: {args.output}")
    print(f"📊 JSON: {args.json}")


if __name__ == "__main__":
    main()



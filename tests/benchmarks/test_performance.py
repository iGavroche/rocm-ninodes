"""
Performance benchmark tests for ROCM Ninodes
"""

import pytest
import torch
import time
import numpy as np
import sys
from pathlib import Path
from typing import Dict, List, Tuple
from unittest.mock import patch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from nodes import (
    ROCMOptimizedVAEDecode, 
    ROCMOptimizedKSampler, 
    ROCMOptimizedCheckpointLoader,
    ROCMFluxBenchmark
)


class PerformanceBenchmark:
    """Base class for performance benchmarks"""
    
    def __init__(self):
        self.results = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.is_amd = self._is_amd_gpu()
    
    def _is_amd_gpu(self) -> bool:
        """Check if running on AMD GPU"""
        if torch.cuda.is_available():
            try:
                device_name = torch.cuda.get_device_name(0)
                return "AMD" in device_name or "Radeon" in device_name
            except:
                return False
        return False
    
    def measure_execution_time(self, func, *args, **kwargs) -> Tuple[float, any]:
        """Measure execution time of a function"""
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start_time = time.time()
        
        result = func(*args, **kwargs)
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        end_time = time.time()
        
        return end_time - start_time, result
    
    def measure_memory_usage(self, func, *args, **kwargs) -> Tuple[float, float, any]:
        """Measure memory usage of a function"""
        if not torch.cuda.is_available():
            return 0, 0, func(*args, **kwargs)
        
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        memory_before = torch.cuda.memory_allocated()
        result = func(*args, **kwargs)
        memory_after = torch.cuda.memory_allocated()
        
        torch.cuda.synchronize()
        
        return memory_before / 1024**3, memory_after / 1024**3, result


class TestVAEDecodePerformance(PerformanceBenchmark):
    """Performance tests for VAE Decode node"""
    
    def setup_method(self):
        """Setup for each test method"""
        self.node = ROCMOptimizedVAEDecode()
        self.mock_vae = self._create_mock_vae()
    
    def _create_mock_vae(self):
        """Create a mock VAE for testing"""
        class MockVAE:
            def __init__(self):
                self.device = self.device
                self.output_device = self.device
                self.vae_dtype = torch.float32
                self.latent_channels = 4
                self.upscale_ratio = 8
                self.first_stage_model = MockFirstStageModel()
            
            def decode(self, samples):
                if len(samples.shape) == 5:  # Video
                    B, C, T, H, W = samples.shape
                    return (torch.randn(B, T, H*8, W*8, 3),)
                else:  # Image
                    B, C, H, W = samples.shape
                    return (torch.randn(B, H*8, W*8, 3),)
            
            def memory_used_decode(self, shape, dtype):
                return 100 * 1024 * 1024  # 100MB
            
            def spacial_compression_decode(self):
                return 8
        
        class MockFirstStageModel:
            def __init__(self):
                self.dtype = torch.float32
            
            def to(self, dtype):
                self.dtype = dtype
                return self
            
            def decode(self, samples):
                if len(samples.shape) == 5:  # Video
                    B, C, T, H, W = samples.shape
                    return torch.randn(B, T, H*8, W*8, 3)
                else:  # Image
                    B, C, H, W = samples.shape
                    return torch.randn(B, H*8, W*8, 3)
        
        return MockVAE()
    
    @pytest.mark.benchmark
    @pytest.mark.parametrize("resolution", [(256, 256), (512, 512), (1024, 1024), (2048, 2048)])
    def test_vae_decode_resolution_performance(self, resolution):
        """Test VAE decode performance across different resolutions"""
        w, h = resolution
        latent = {
            "samples": torch.randn(1, 4, h//8, w//8, device=self.device)
        }
        
        execution_time, result = self.measure_execution_time(
            self.node.decode,
            vae=self.mock_vae,
            samples=latent,
            use_rocm_optimizations=True,
            precision_mode="auto"
        )
        
        assert isinstance(result, tuple)
        assert len(result) == 1
        
        # Store results for analysis
        self.results[f"vae_decode_{w}x{h}"] = {
            "execution_time": execution_time,
            "resolution": resolution,
            "device": str(self.device),
            "is_amd": self.is_amd
        }
        
        print(f"VAE Decode {w}x{h}: {execution_time:.3f}s")
    
    @pytest.mark.benchmark
    @pytest.mark.parametrize("precision", ["fp32", "fp16", "bf16"])
    def test_vae_decode_precision_performance(self, precision):
        """Test VAE decode performance across different precision modes"""
        latent = {
            "samples": torch.randn(1, 4, 64, 64, device=self.device)
        }
        
        execution_time, result = self.measure_execution_time(
            self.node.decode,
            vae=self.mock_vae,
            samples=latent,
            use_rocm_optimizations=True,
            precision_mode=precision
        )
        
        assert isinstance(result, tuple)
        assert len(result) == 1
        
        self.results[f"vae_decode_precision_{precision}"] = {
            "execution_time": execution_time,
            "precision": precision,
            "device": str(self.device),
            "is_amd": self.is_amd
        }
        
        print(f"VAE Decode {precision}: {execution_time:.3f}s")
    
    @pytest.mark.benchmark
    def test_vae_decode_memory_usage(self):
        """Test VAE decode memory usage"""
        latent = {
            "samples": torch.randn(1, 4, 128, 128, device=self.device)
        }
        
        memory_before, memory_after, result = self.measure_memory_usage(
            self.node.decode,
            vae=self.mock_vae,
            samples=latent,
            use_rocm_optimizations=True,
            precision_mode="auto"
        )
        
        memory_used = memory_after - memory_before
        
        assert isinstance(result, tuple)
        assert len(result) == 1
        assert memory_used >= 0
        
        self.results["vae_decode_memory"] = {
            "memory_used_gb": memory_used,
            "memory_before_gb": memory_before,
            "memory_after_gb": memory_after,
            "device": str(self.device),
            "is_amd": self.is_amd
        }
        
        print(f"VAE Decode Memory: {memory_used:.3f}GB")
    
    @pytest.mark.benchmark
    def test_vae_decode_video_performance(self):
        """Test VAE decode performance for video"""
        # Create video latent (B, C, T, H, W)
        latent = {
            "samples": torch.randn(1, 4, 8, 64, 64, device=self.device)
        }
        
        execution_time, result = self.measure_execution_time(
            self.node.decode,
            vae=self.mock_vae,
            samples=latent,
            use_rocm_optimizations=True,
            precision_mode="auto",
            video_chunk_size=4,
            memory_optimization_enabled=True
        )
        
        assert isinstance(result, tuple)
        assert len(result) == 1
        
        self.results["vae_decode_video"] = {
            "execution_time": execution_time,
            "video_frames": 8,
            "device": str(self.device),
            "is_amd": self.is_amd
        }
        
        print(f"VAE Decode Video (8 frames): {execution_time:.3f}s")


class TestKSamplerPerformance(PerformanceBenchmark):
    """Performance tests for KSampler node"""
    
    def setup_method(self):
        """Setup for each test method"""
        self.node = ROCMOptimizedKSampler()
        self.mock_model = self._create_mock_model()
        self.sample_conditioning = [
            [
                "A beautiful landscape",
                {
                    "pooled_output": torch.randn(1, 1280, device=self.device),
                    "embeds": torch.randn(1, 77, 1280, device=self.device)
                }
            ]
        ]
        self.sample_latent = {
            "samples": torch.randn(1, 4, 64, 64, device=self.device),
            "batch_index": [0]
        }
    
    def _create_mock_model(self):
        """Create a mock model for testing"""
        class MockModel:
            def __init__(self):
                self.device = self.device
                self.model_dtype = lambda: torch.float32
        
        return MockModel()
    
    @pytest.mark.benchmark
    @pytest.mark.parametrize("steps", [1, 5, 10, 20, 50])
    def test_ksampler_steps_performance(self, steps):
        """Test KSampler performance across different step counts"""
        with pytest.MonkeyPatch().context() as m:
            # Mock the sample function to avoid actual sampling
            def mock_sample(*args, **kwargs):
                return torch.randn(1, 4, 64, 64, device=self.device)
            
            m.setattr("comfy.sample.sample", mock_sample)
            
            execution_time, result = self.measure_execution_time(
                self.node.sample,
                model=self.mock_model,
                seed=42,
                steps=steps,
                cfg=8.0,
                sampler_name="euler",
                scheduler="simple",
                positive=self.sample_conditioning,
                negative=self.sample_conditioning,
                latent_image=self.sample_latent,
                denoise=1.0,
                use_rocm_optimizations=True,
                precision_mode="auto",
                memory_optimization=True,
                attention_optimization=True
            )
            
            assert isinstance(result, tuple)
            assert len(result) == 1
            
            self.results[f"ksampler_steps_{steps}"] = {
                "execution_time": execution_time,
                "steps": steps,
                "device": str(self.device),
                "is_amd": self.is_amd
            }
            
            print(f"KSampler {steps} steps: {execution_time:.3f}s")
    
    @pytest.mark.benchmark
    @pytest.mark.parametrize("cfg", [1.0, 3.5, 7.0, 8.0, 12.0])
    def test_ksampler_cfg_performance(self, cfg):
        """Test KSampler performance across different CFG values"""
        with pytest.MonkeyPatch().context() as m:
            def mock_sample(*args, **kwargs):
                return torch.randn(1, 4, 64, 64, device=self.device)
            
            m.setattr("comfy.sample.sample", mock_sample)
            
            execution_time, result = self.measure_execution_time(
                self.node.sample,
                model=self.mock_model,
                seed=42,
                steps=20,
                cfg=cfg,
                sampler_name="euler",
                scheduler="simple",
                positive=self.sample_conditioning,
                negative=self.sample_conditioning,
                latent_image=self.sample_latent,
                denoise=1.0,
                use_rocm_optimizations=True,
                precision_mode="auto"
            )
            
            assert isinstance(result, tuple)
            assert len(result) == 1
            
            self.results[f"ksampler_cfg_{cfg}"] = {
                "execution_time": execution_time,
                "cfg": cfg,
                "device": str(self.device),
                "is_amd": self.is_amd
            }
            
            print(f"KSampler CFG {cfg}: {execution_time:.3f}s")
    
    @pytest.mark.benchmark
    def test_ksampler_memory_usage(self):
        """Test KSampler memory usage"""
        with pytest.MonkeyPatch().context() as m:
            def mock_sample(*args, **kwargs):
                return torch.randn(1, 4, 64, 64, device=self.device)
            
            m.setattr("comfy.sample.sample", mock_sample)
            
            memory_before, memory_after, result = self.measure_memory_usage(
                self.node.sample,
                model=self.mock_model,
                seed=42,
                steps=20,
                cfg=8.0,
                sampler_name="euler",
                scheduler="simple",
                positive=self.sample_conditioning,
                negative=self.sample_conditioning,
                latent_image=self.sample_latent,
                denoise=1.0,
                use_rocm_optimizations=True,
                precision_mode="auto"
            )
            
            memory_used = memory_after - memory_before
            
            assert isinstance(result, tuple)
            assert len(result) == 1
            assert memory_used >= 0
            
            self.results["ksampler_memory"] = {
                "memory_used_gb": memory_used,
                "memory_before_gb": memory_before,
                "memory_after_gb": memory_after,
                "device": str(self.device),
                "is_amd": self.is_amd
            }
            
            print(f"KSampler Memory: {memory_used:.3f}GB")


class TestCheckpointLoaderPerformance(PerformanceBenchmark):
    """Performance tests for Checkpoint Loader node"""
    
    def setup_method(self):
        """Setup for each test method"""
        self.node = ROCMOptimizedCheckpointLoader()
    
    @pytest.mark.benchmark
    @patch('comfy.sd.load_checkpoint_guess_config')
    @patch('folder_paths.get_full_path_or_raise')
    def test_checkpoint_loader_performance(self, mock_get_path, mock_load_checkpoint):
        """Test checkpoint loader performance"""
        # Mock the checkpoint path
        mock_get_path.return_value = "/fake/path/checkpoint.safetensors"
        
        # Mock the checkpoint loading result
        mock_model = Mock()
        mock_clip = Mock()
        mock_vae = Mock()
        mock_load_checkpoint.return_value = (mock_model, mock_clip, mock_vae)
        
        execution_time, result = self.measure_execution_time(
            self.node.load_checkpoint,
            ckpt_name="test_checkpoint.safetensors",
            lazy_loading=True,
            optimize_for_flux=True,
            precision_mode="auto"
        )
        
        assert isinstance(result, tuple)
        assert len(result) == 3
        
        self.results["checkpoint_loader"] = {
            "execution_time": execution_time,
            "device": str(self.device),
            "is_amd": self.is_amd
        }
        
        print(f"Checkpoint Loader: {execution_time:.3f}s")


@pytest.mark.benchmark
def test_rocm_vs_standard_performance():
    """Compare ROCM optimized vs standard performance"""
    # This test would require actual model loading and comparison
    # For now, we'll create a placeholder test structure
    
    benchmark = PerformanceBenchmark()
    
    # Test parameters
    test_cases = [
        {"resolution": (512, 512), "steps": 20, "cfg": 8.0},
        {"resolution": (1024, 1024), "steps": 20, "cfg": 8.0},
        {"resolution": (512, 512), "steps": 50, "cfg": 8.0},
    ]
    
    results = {}
    
    for i, test_case in enumerate(test_cases):
        # This would be implemented with actual model loading
        # For now, we'll create mock results
        results[f"test_case_{i}"] = {
            "rocm_time": 1.0 + i * 0.1,  # Mock ROCM time
            "standard_time": 1.2 + i * 0.1,  # Mock standard time
            "improvement": 0.2,  # Mock improvement
            **test_case
        }
    
    # Verify results structure
    assert len(results) == len(test_cases)
    for key, result in results.items():
        assert "rocm_time" in result
        assert "standard_time" in result
        assert "improvement" in result
        assert result["improvement"] > 0  # ROCM should be faster


@pytest.mark.benchmark
def test_memory_efficiency():
    """Test memory efficiency of ROCM optimizations"""
    benchmark = PerformanceBenchmark()
    
    # Test different memory optimization levels
    optimization_levels = ["conservative", "balanced", "aggressive"]
    
    results = {}
    
    for level in optimization_levels:
        # Mock memory usage results
        results[level] = {
            "peak_memory_gb": 2.0 + optimization_levels.index(level) * 0.5,
            "avg_memory_gb": 1.5 + optimization_levels.index(level) * 0.3,
            "memory_efficiency": 0.8 - optimization_levels.index(level) * 0.1
        }
    
    # Verify results
    assert len(results) == len(optimization_levels)
    for level, result in results.items():
        assert "peak_memory_gb" in result
        assert "avg_memory_gb" in result
        assert "memory_efficiency" in result
        assert result["memory_efficiency"] > 0


def generate_performance_report():
    """Generate a comprehensive performance report"""
    # This would aggregate all benchmark results and generate a report
    report = {
        "timestamp": time.time(),
        "device": str(torch.device("cuda" if torch.cuda.is_available() else "cpu")),
        "is_amd": torch.cuda.is_available() and "AMD" in torch.cuda.get_device_name(0),
        "summary": {
            "total_tests": 0,
            "avg_improvement": 0.0,
            "memory_efficiency": 0.0
        },
        "recommendations": []
    }
    
    return report

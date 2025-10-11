"""
Unit tests for ROCMOptimizedKSampler nodes
"""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from nodes import ROCMOptimizedKSampler, ROCMOptimizedKSamplerAdvanced


class TestROCMOptimizedKSampler:
    """Test cases for ROCMOptimizedKSampler node"""
    
    def test_input_types(self):
        """Test that INPUT_TYPES returns correct structure"""
        input_types = ROCMOptimizedKSampler.INPUT_TYPES()
        
        assert "required" in input_types
        required = input_types["required"]
        
        # Check required inputs
        assert "model" in required
        assert "seed" in required
        assert "steps" in required
        assert "cfg" in required
        assert "sampler_name" in required
        assert "scheduler" in required
        assert "positive" in required
        assert "negative" in required
        assert "latent_image" in required
        assert "denoise" in required
        assert "use_rocm_optimizations" in required
        assert "precision_mode" in required
        assert "memory_optimization" in required
        assert "attention_optimization" in required
    
    def test_return_types(self):
        """Test that RETURN_TYPES is correct"""
        assert ROCMOptimizedKSampler.RETURN_TYPES == ("LATENT",)
        assert ROCMOptimizedKSampler.RETURN_NAMES == ("LATENT",)
        assert ROCMOptimizedKSampler.FUNCTION == "sample"
        assert ROCMOptimizedKSampler.CATEGORY == "RocM Ninodes/Sampling"
    
    @patch('comfy.sample.sample')
    def test_sample_basic(self, mock_sample, sample_model, sample_conditioning, sample_latent):
        """Test basic sampling functionality"""
        # Mock the sample function
        mock_sample.return_value = torch.randn(1, 4, 64, 64)
        
        node = ROCMOptimizedKSampler()
        
        result = node.sample(
            model=sample_model,
            seed=42,
            steps=20,
            cfg=8.0,
            sampler_name="euler",
            scheduler="simple",
            positive=sample_conditioning,
            negative=sample_conditioning,
            latent_image=sample_latent,
            denoise=1.0,
            use_rocm_optimizations=True,
            precision_mode="auto",
            memory_optimization=True,
            attention_optimization=True
        )
        
        assert isinstance(result, tuple)
        assert len(result) == 1
        assert isinstance(result[0], dict)
        assert "samples" in result[0]
        assert isinstance(result[0]["samples"], torch.Tensor)
    
    def test_precision_modes(self, sample_model, sample_conditioning, sample_latent):
        """Test different precision modes"""
        node = ROCMOptimizedKSampler()
        
        precision_modes = ["auto", "fp32", "fp16", "bf16"]
        
        for precision in precision_modes:
            with patch('comfy.sample.sample') as mock_sample:
                mock_sample.return_value = torch.randn(1, 4, 64, 64)
                
                result = node.sample(
                    model=sample_model,
                    seed=42,
                    steps=20,
                    cfg=8.0,
                    sampler_name="euler",
                    scheduler="simple",
                    positive=sample_conditioning,
                    negative=sample_conditioning,
                    latent_image=sample_latent,
                    precision_mode=precision,
                    use_rocm_optimizations=True
                )
                
                assert isinstance(result, tuple)
                assert len(result) == 1
    
    def test_rocm_optimizations(self, sample_model, sample_conditioning, sample_latent, is_amd_gpu):
        """Test ROCm optimizations"""
        node = ROCMOptimizedKSampler()
        
        with patch('comfy.sample.sample') as mock_sample:
            mock_sample.return_value = torch.randn(1, 4, 64, 64)
            
            # Test with ROCm optimizations enabled
            result = node.sample(
                model=sample_model,
                seed=42,
                steps=20,
                cfg=8.0,
                sampler_name="euler",
                scheduler="simple",
                positive=sample_conditioning,
                negative=sample_conditioning,
                latent_image=sample_latent,
                use_rocm_optimizations=True
            )
            
            assert isinstance(result, tuple)
            
            # Test with ROCm optimizations disabled
            result = node.sample(
                model=sample_model,
                seed=42,
                steps=20,
                cfg=8.0,
                sampler_name="euler",
                scheduler="simple",
                positive=sample_conditioning,
                negative=sample_conditioning,
                latent_image=sample_latent,
                use_rocm_optimizations=False
            )
            
            assert isinstance(result, tuple)
    
    def test_memory_optimization(self, sample_model, sample_conditioning, sample_latent):
        """Test memory optimization settings"""
        node = ROCMOptimizedKSampler()
        
        with patch('comfy.sample.sample') as mock_sample:
            mock_sample.return_value = torch.randn(1, 4, 64, 64)
            
            # Test with memory optimization enabled
            result = node.sample(
                model=sample_model,
                seed=42,
                steps=20,
                cfg=8.0,
                sampler_name="euler",
                scheduler="simple",
                positive=sample_conditioning,
                negative=sample_conditioning,
                latent_image=sample_latent,
                memory_optimization=True
            )
            
            assert isinstance(result, tuple)
            
            # Test with memory optimization disabled
            result = node.sample(
                model=sample_model,
                seed=42,
                steps=20,
                cfg=8.0,
                sampler_name="euler",
                scheduler="simple",
                positive=sample_conditioning,
                negative=sample_conditioning,
                latent_image=sample_latent,
                memory_optimization=False
            )
            
            assert isinstance(result, tuple)
    
    def test_attention_optimization(self, sample_model, sample_conditioning, sample_latent):
        """Test attention optimization settings"""
        node = ROCMOptimizedKSampler()
        
        with patch('comfy.sample.sample') as mock_sample:
            mock_sample.return_value = torch.randn(1, 4, 64, 64)
            
            # Test with attention optimization enabled
            result = node.sample(
                model=sample_model,
                seed=42,
                steps=20,
                cfg=8.0,
                sampler_name="euler",
                scheduler="simple",
                positive=sample_conditioning,
                negative=sample_conditioning,
                latent_image=sample_latent,
                attention_optimization=True
            )
            
            assert isinstance(result, tuple)
            
            # Test with attention optimization disabled
            result = node.sample(
                model=sample_model,
                seed=42,
                steps=20,
                cfg=8.0,
                sampler_name="euler",
                scheduler="simple",
                positive=sample_conditioning,
                negative=sample_conditioning,
                latent_image=sample_latent,
                attention_optimization=False
            )
            
            assert isinstance(result, tuple)
    
    def test_error_handling(self, sample_model, sample_conditioning, sample_latent):
        """Test error handling and fallbacks"""
        node = ROCMOptimizedKSampler()
        
        # Test with invalid model
        with pytest.raises(Exception):
            node.sample(
                model=None,
                seed=42,
                steps=20,
                cfg=8.0,
                sampler_name="euler",
                scheduler="simple",
                positive=sample_conditioning,
                negative=sample_conditioning,
                latent_image=sample_latent
            )
    
    def test_performance_timing(self, sample_model, sample_conditioning, sample_latent):
        """Test that performance timing works"""
        node = ROCMOptimizedKSampler()
        
        with patch('comfy.sample.sample') as mock_sample:
            mock_sample.return_value = torch.randn(1, 4, 64, 64)
            
            import time
            start_time = time.time()
            
            result = node.sample(
                model=sample_model,
                seed=42,
                steps=20,
                cfg=8.0,
                sampler_name="euler",
                scheduler="simple",
                positive=sample_conditioning,
                negative=sample_conditioning,
                latent_image=sample_latent,
                use_rocm_optimizations=True
            )
            
            end_time = time.time()
            execution_time = end_time - start_time
            
            assert execution_time > 0
            assert isinstance(result, tuple)


class TestROCMOptimizedKSamplerAdvanced:
    """Test cases for ROCMOptimizedKSamplerAdvanced node"""
    
    def test_input_types(self):
        """Test that INPUT_TYPES returns correct structure"""
        input_types = ROCMOptimizedKSamplerAdvanced.INPUT_TYPES()
        
        assert "required" in input_types
        required = input_types["required"]
        
        # Check required inputs
        assert "model" in required
        assert "add_noise" in required
        assert "noise_seed" in required
        assert "steps" in required
        assert "cfg" in required
        assert "sampler_name" in required
        assert "scheduler" in required
        assert "positive" in required
        assert "negative" in required
        assert "latent_image" in required
        assert "start_at_step" in required
        assert "end_at_step" in required
        assert "return_with_leftover_noise" in required
        assert "use_rocm_optimizations" in required
        assert "precision_mode" in required
        assert "memory_optimization" in required
    
    def test_return_types(self):
        """Test that RETURN_TYPES is correct"""
        assert ROCMOptimizedKSamplerAdvanced.RETURN_TYPES == ("LATENT",)
        assert ROCMOptimizedKSamplerAdvanced.RETURN_NAMES == ("LATENT",)
        assert ROCMOptimizedKSamplerAdvanced.FUNCTION == "sample"
        assert ROCMOptimizedKSamplerAdvanced.CATEGORY == "RocM Ninodes/Sampling"
    
    @patch('comfy.sample.sample')
    def test_sample_advanced(self, mock_sample, sample_model, sample_conditioning, sample_latent):
        """Test advanced sampling functionality"""
        # Mock the sample function
        mock_sample.return_value = torch.randn(1, 4, 64, 64)
        
        node = ROCMOptimizedKSamplerAdvanced()
        
        result = node.sample(
            model=sample_model,
            add_noise="enable",
            noise_seed=42,
            steps=20,
            cfg=8.0,
            sampler_name="euler",
            scheduler="simple",
            positive=sample_conditioning,
            negative=sample_conditioning,
            latent_image=sample_latent,
            start_at_step=0,
            end_at_step=20,
            return_with_leftover_noise="disable",
            use_rocm_optimizations=True,
            precision_mode="auto",
            memory_optimization=True
        )
        
        assert isinstance(result, tuple)
        assert len(result) == 1
        assert isinstance(result[0], dict)
        assert "samples" in result[0]
        assert isinstance(result[0]["samples"], torch.Tensor)
    
    def test_noise_control(self, sample_model, sample_conditioning, sample_latent):
        """Test noise control options"""
        node = ROCMOptimizedKSamplerAdvanced()
        
        with patch('comfy.sample.sample') as mock_sample:
            mock_sample.return_value = torch.randn(1, 4, 64, 64)
            
            # Test with noise enabled
            result = node.sample(
                model=sample_model,
                add_noise="enable",
                noise_seed=42,
                steps=20,
                cfg=8.0,
                sampler_name="euler",
                scheduler="simple",
                positive=sample_conditioning,
                negative=sample_conditioning,
                latent_image=sample_latent,
                start_at_step=0,
                end_at_step=20,
                return_with_leftover_noise="disable"
            )
            
            assert isinstance(result, tuple)
            
            # Test with noise disabled
            result = node.sample(
                model=sample_model,
                add_noise="disable",
                noise_seed=42,
                steps=20,
                cfg=8.0,
                sampler_name="euler",
                scheduler="simple",
                positive=sample_conditioning,
                negative=sample_conditioning,
                latent_image=sample_latent,
                start_at_step=0,
                end_at_step=20,
                return_with_leftover_noise="disable"
            )
            
            assert isinstance(result, tuple)
    
    def test_step_control(self, sample_model, sample_conditioning, sample_latent):
        """Test step control options"""
        node = ROCMOptimizedKSamplerAdvanced()
        
        with patch('comfy.sample.sample') as mock_sample:
            mock_sample.return_value = torch.randn(1, 4, 64, 64)
            
            # Test with custom start/end steps
            result = node.sample(
                model=sample_model,
                add_noise="enable",
                noise_seed=42,
                steps=20,
                cfg=8.0,
                sampler_name="euler",
                scheduler="simple",
                positive=sample_conditioning,
                negative=sample_conditioning,
                latent_image=sample_latent,
                start_at_step=5,
                end_at_step=15,
                return_with_leftover_noise="disable"
            )
            
            assert isinstance(result, tuple)
    
    def test_leftover_noise_control(self, sample_model, sample_conditioning, sample_latent):
        """Test leftover noise control options"""
        node = ROCMOptimizedKSamplerAdvanced()
        
        with patch('comfy.sample.sample') as mock_sample:
            mock_sample.return_value = torch.randn(1, 4, 64, 64)
            
            # Test with leftover noise enabled
            result = node.sample(
                model=sample_model,
                add_noise="enable",
                noise_seed=42,
                steps=20,
                cfg=8.0,
                sampler_name="euler",
                scheduler="simple",
                positive=sample_conditioning,
                negative=sample_conditioning,
                latent_image=sample_latent,
                start_at_step=0,
                end_at_step=20,
                return_with_leftover_noise="enable"
            )
            
            assert isinstance(result, tuple)
            
            # Test with leftover noise disabled
            result = node.sample(
                model=sample_model,
                add_noise="enable",
                noise_seed=42,
                steps=20,
                cfg=8.0,
                sampler_name="euler",
                scheduler="simple",
                positive=sample_conditioning,
                negative=sample_conditioning,
                latent_image=sample_latent,
                start_at_step=0,
                end_at_step=20,
                return_with_leftover_noise="disable"
            )
            
            assert isinstance(result, tuple)


@pytest.mark.parametrize("steps", [1, 5, 10, 20, 50])
def test_different_step_counts(steps, sample_model, sample_conditioning, sample_latent):
    """Test sampling with different step counts"""
    node = ROCMOptimizedKSampler()
    
    with patch('comfy.sample.sample') as mock_sample:
        mock_sample.return_value = torch.randn(1, 4, 64, 64)
        
        result = node.sample(
            model=sample_model,
            seed=42,
            steps=steps,
            cfg=8.0,
            sampler_name="euler",
            scheduler="simple",
            positive=sample_conditioning,
            negative=sample_conditioning,
            latent_image=sample_latent,
            use_rocm_optimizations=True
        )
        
        assert isinstance(result, tuple)
        assert len(result) == 1


@pytest.mark.parametrize("cfg", [1.0, 3.5, 7.0, 8.0, 12.0])
def test_different_cfg_values(cfg, sample_model, sample_conditioning, sample_latent):
    """Test sampling with different CFG values"""
    node = ROCMOptimizedKSampler()
    
    with patch('comfy.sample.sample') as mock_sample:
        mock_sample.return_value = torch.randn(1, 4, 64, 64)
        
        result = node.sample(
            model=sample_model,
            seed=42,
            steps=20,
            cfg=cfg,
            sampler_name="euler",
            scheduler="simple",
            positive=sample_conditioning,
            negative=sample_conditioning,
            latent_image=sample_latent,
            use_rocm_optimizations=True
        )
        
        assert isinstance(result, tuple)
        assert len(result) == 1


@pytest.mark.parametrize("sampler_name", ["euler", "heun", "dpmpp_2m", "dpmpp_sde"])
def test_different_samplers(sampler_name, sample_model, sample_conditioning, sample_latent):
    """Test sampling with different samplers"""
    node = ROCMOptimizedKSampler()
    
    with patch('comfy.sample.sample') as mock_sample:
        mock_sample.return_value = torch.randn(1, 4, 64, 64)
        
        result = node.sample(
            model=sample_model,
            seed=42,
            steps=20,
            cfg=8.0,
            sampler_name=sampler_name,
            scheduler="simple",
            positive=sample_conditioning,
            negative=sample_conditioning,
            latent_image=sample_latent,
            use_rocm_optimizations=True
        )
        
        assert isinstance(result, tuple)
        assert len(result) == 1

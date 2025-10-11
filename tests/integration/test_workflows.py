"""
Integration tests for ROCM Ninodes workflows
Tests complete workflows using Flux and WAN models
"""

import pytest
import torch
import json
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from nodes import (
    ROCMOptimizedCheckpointLoader,
    ROCMOptimizedVAEDecode,
    ROCMOptimizedKSampler,
    ROCMOptimizedKSamplerAdvanced
)


class TestFluxWorkflow:
    """Integration tests for Flux workflow"""
    
    def setup_method(self):
        """Setup for each test method"""
        self.workflow_file = Path(__file__).parent.parent.parent / "flux_dev_optimized.json"
        self.workflow_data = self._load_workflow_data()
        self.mock_models = self._create_mock_models()
    
    def _load_workflow_data(self) -> Dict[str, Any]:
        """Load Flux workflow data"""
        if self.workflow_file.exists():
            with open(self.workflow_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _create_mock_models(self) -> Dict[str, Any]:
        """Create mock models for testing"""
        class MockModel:
            def __init__(self):
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                self.model_dtype = lambda: torch.float32
        
        class MockVAE:
            def __init__(self):
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                self.output_device = self.device
                self.vae_dtype = torch.float32
                self.latent_channels = 4
                self.upscale_ratio = 8
                self.first_stage_model = MockFirstStageModel()
            
            def decode(self, samples):
                B, C, H, W = samples.shape
                return (torch.randn(B, H*8, W*8, 3),)
            
            def memory_used_decode(self, shape, dtype):
                return 100 * 1024 * 1024
            
            def spacial_compression_decode(self):
                return 8
        
        class MockCLIP:
            def __init__(self):
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        class MockFirstStageModel:
            def __init__(self):
                self.dtype = torch.float32
            
            def to(self, dtype):
                self.dtype = dtype
                return self
            
            def decode(self, samples):
                B, C, H, W = samples.shape
                return torch.randn(B, H*8, W*8, 3)
        
        return {
            "model": MockModel(),
            "vae": MockVAE(),
            "clip": MockCLIP()
        }
    
    @pytest.mark.integration
    @patch('comfy.sd.load_checkpoint_guess_config')
    @patch('folder_paths.get_full_path_or_raise')
    def test_flux_workflow_loading(self, mock_get_path, mock_load_checkpoint):
        """Test Flux workflow model loading"""
        # Mock checkpoint loading
        mock_get_path.return_value = "/fake/path/flux_dev.safetensors"
        mock_load_checkpoint.return_value = (
            self.mock_models["model"],
            self.mock_models["clip"],
            self.mock_models["vae"]
        )
        
        loader = ROCMOptimizedCheckpointLoader()
        
        result = loader.load_checkpoint(
            ckpt_name="flux_dev.safetensors",
            lazy_loading=True,
            optimize_for_flux=True,
            precision_mode="auto"
        )
        
        assert isinstance(result, tuple)
        assert len(result) == 3
        assert result[0] == self.mock_models["model"]
        assert result[1] == self.mock_models["clip"]
        assert result[2] == self.mock_models["vae"]
    
    @pytest.mark.integration
    def test_flux_workflow_text_encoding(self):
        """Test Flux workflow text encoding"""
        # This would test CLIP text encoding
        # For now, we'll create a mock test
        conditioning = [
            [
                "A portrait view of a beautiful blond girl with long hair and blue eyes looking in love",
                {
                    "pooled_output": torch.randn(1, 1280),
                    "embeds": torch.randn(1, 77, 1280)
                }
            ]
        ]
        
        assert isinstance(conditioning, list)
        assert len(conditioning) == 1
        assert isinstance(conditioning[0], list)
        assert len(conditioning[0]) == 2
        assert isinstance(conditioning[0][0], str)
        assert isinstance(conditioning[0][1], dict)
    
    @pytest.mark.integration
    def test_flux_workflow_sampling(self):
        """Test Flux workflow sampling"""
        sampler = ROCMOptimizedKSampler()
        
        # Create test data
        latent_image = {
            "samples": torch.randn(1, 4, 128, 128),
            "batch_index": [0]
        }
        
        positive_conditioning = [
            [
                "A portrait view of a beautiful blond girl with long hair and blue eyes looking in love",
                {
                    "pooled_output": torch.randn(1, 1280),
                    "embeds": torch.randn(1, 77, 1280)
                }
            ]
        ]
        
        negative_conditioning = [
            [
                "",
                {
                    "pooled_output": torch.randn(1, 1280),
                    "embeds": torch.randn(1, 77, 1280)
                }
            ]
        ]
        
        with patch('comfy.sample.sample') as mock_sample:
            mock_sample.return_value = torch.randn(1, 4, 128, 128)
            
            result = sampler.sample(
                model=self.mock_models["model"],
                seed=813485113100655,
                steps=20,
                cfg=8,
                sampler_name="euler",
                scheduler="simple",
                positive=positive_conditioning,
                negative=negative_conditioning,
                latent_image=latent_image,
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
    
    @pytest.mark.integration
    def test_flux_workflow_vae_decode(self):
        """Test Flux workflow VAE decode"""
        vae_decoder = ROCMOptimizedVAEDecode()
        
        # Create test latent
        latent = {
            "samples": torch.randn(1, 4, 128, 128)
        }
        
        result = vae_decoder.decode(
            vae=self.mock_models["vae"],
            samples=latent,
            tile_size=768,
            overlap=96,
            use_rocm_optimizations=True,
            precision_mode="auto",
            batch_optimization=True,
            flux_vae_optimization=True,
            adaptive_tile_sizing=True,
            fp8_latent_optimization=True
        )
        
        assert isinstance(result, tuple)
        assert len(result) == 1
        assert isinstance(result[0], torch.Tensor)
        assert len(result[0].shape) == 4  # B, H, W, C
    
    @pytest.mark.integration
    def test_flux_workflow_complete(self):
        """Test complete Flux workflow"""
        # This test would run the complete workflow
        # For now, we'll test the individual components
        
        # 1. Load checkpoint
        with patch('comfy.sd.load_checkpoint_guess_config') as mock_load_checkpoint:
            with patch('folder_paths.get_full_path_or_raise') as mock_get_path:
                mock_get_path.return_value = "/fake/path/flux_dev.safetensors"
                mock_load_checkpoint.return_value = (
                    self.mock_models["model"],
                    self.mock_models["clip"],
                    self.mock_models["vae"]
                )
                
                loader = ROCMOptimizedCheckpointLoader()
                model, clip, vae = loader.load_checkpoint(
                    ckpt_name="flux_dev.safetensors",
                    lazy_loading=True,
                    optimize_for_flux=True,
                    precision_mode="auto"
                )
        
        # 2. Create latent
        latent_image = {
            "samples": torch.randn(1, 4, 128, 128),
            "batch_index": [0]
        }
        
        # 3. Sample
        with patch('comfy.sample.sample') as mock_sample:
            mock_sample.return_value = torch.randn(1, 4, 128, 128)
            
            sampler = ROCMOptimizedKSampler()
            sampled_latent = sampler.sample(
                model=model,
                seed=813485113100655,
                steps=20,
                cfg=8,
                sampler_name="euler",
                scheduler="simple",
                positive=[["test prompt", {"pooled_output": torch.randn(1, 1280), "embeds": torch.randn(1, 77, 1280)}]],
                negative=[["", {"pooled_output": torch.randn(1, 1280), "embeds": torch.randn(1, 77, 1280)}]],
                latent_image=latent_image,
                denoise=1.0,
                use_rocm_optimizations=True,
                precision_mode="auto",
                memory_optimization=True,
                attention_optimization=True
            )
        
        # 4. Decode
        vae_decoder = ROCMOptimizedVAEDecode()
        final_image = vae_decoder.decode(
            vae=vae,
            samples=sampled_latent[0],
            tile_size=768,
            overlap=96,
            use_rocm_optimizations=True,
            precision_mode="auto",
            batch_optimization=True
        )
        
        # Verify final result
        assert isinstance(final_image, tuple)
        assert len(final_image) == 1
        assert isinstance(final_image[0], torch.Tensor)
        assert len(final_image[0].shape) == 4  # B, H, W, C


class TestWANWorkflow:
    """Integration tests for WAN video workflow"""
    
    def setup_method(self):
        """Setup for each test method"""
        self.workflow_file = Path(__file__).parent.parent.parent / "example_workflow_wan_video.json"
        self.workflow_data = self._load_workflow_data()
        self.mock_models = self._create_mock_models()
    
    def _load_workflow_data(self) -> Dict[str, Any]:
        """Load WAN workflow data"""
        if self.workflow_file.exists():
            with open(self.workflow_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _create_mock_models(self) -> Dict[str, Any]:
        """Create mock models for testing"""
        class MockModel:
            def __init__(self):
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                self.model_dtype = lambda: torch.float32
        
        class MockVAE:
            def __init__(self):
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
                return 100 * 1024 * 1024
            
            def spacial_compression_decode(self):
                return 8
        
        class MockCLIP:
            def __init__(self):
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
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
        
        return {
            "model": MockModel(),
            "vae": MockVAE(),
            "clip": MockCLIP()
        }
    
    @pytest.mark.integration
    def test_wan_workflow_video_processing(self):
        """Test WAN workflow video processing"""
        vae_decoder = ROCMOptimizedVAEDecode()
        
        # Create video latent (B, C, T, H, W)
        video_latent = {
            "samples": torch.randn(1, 4, 8, 64, 64)
        }
        
        result = vae_decoder.decode(
            vae=self.mock_models["vae"],
            samples=video_latent,
            tile_size=768,
            overlap=96,
            use_rocm_optimizations=True,
            precision_mode="auto",
            batch_optimization=True,
            video_chunk_size=4,
            memory_optimization_enabled=True
        )
        
        assert isinstance(result, tuple)
        assert len(result) == 1
        assert isinstance(result[0], torch.Tensor)
        # Should be reshaped to 4D for ComfyUI
        assert len(result[0].shape) == 4  # B*T, H, W, C
    
    @pytest.mark.integration
    def test_wan_workflow_advanced_sampling(self):
        """Test WAN workflow advanced sampling"""
        sampler = ROCMOptimizedKSamplerAdvanced()
        
        # Create test data
        latent_image = {
            "samples": torch.randn(1, 4, 64, 64),
            "batch_index": [0]
        }
        
        conditioning = [
            [
                "The girl blows the candles then smiles and winks",
                {
                    "pooled_output": torch.randn(1, 1280),
                    "embeds": torch.randn(1, 77, 1280)
                }
            ]
        ]
        
        with patch('comfy.sample.sample') as mock_sample:
            mock_sample.return_value = torch.randn(1, 4, 64, 64)
            
            result = sampler.sample(
                model=self.mock_models["model"],
                add_noise="enable",
                noise_seed=549535961820368,
                steps=4,
                cfg=1,
                sampler_name="euler",
                scheduler="simple",
                positive=conditioning,
                negative=conditioning,
                latent_image=latent_image,
                start_at_step=0,
                end_at_step=2,
                return_with_leftover_noise="enable",
                use_rocm_optimizations=True,
                precision_mode="auto",
                memory_optimization=True
            )
            
            assert isinstance(result, tuple)
            assert len(result) == 1
            assert isinstance(result[0], dict)
            assert "samples" in result[0]
    
    @pytest.mark.integration
    def test_wan_workflow_complete(self):
        """Test complete WAN workflow"""
        # This test would run the complete WAN workflow
        # For now, we'll test the key components
        
        # 1. Load models (mocked)
        with patch('comfy.sd.load_checkpoint_guess_config') as mock_load_checkpoint:
            with patch('folder_paths.get_full_path_or_raise') as mock_get_path:
                mock_get_path.return_value = "/fake/path/wan_model.safetensors"
                mock_load_checkpoint.return_value = (
                    self.mock_models["model"],
                    self.mock_models["clip"],
                    self.mock_models["vae"]
                )
                
                loader = ROCMOptimizedCheckpointLoader()
                model, clip, vae = loader.load_checkpoint(
                    ckpt_name="wan_model.safetensors",
                    lazy_loading=True,
                    optimize_for_flux=False,
                    precision_mode="auto"
                )
        
        # 2. Create video latent
        video_latent = {
            "samples": torch.randn(1, 4, 8, 64, 64)
        }
        
        # 3. Sample (mocked)
        with patch('comfy.sample.sample') as mock_sample:
            mock_sample.return_value = torch.randn(1, 4, 8, 64, 64)
            
            sampler = ROCMOptimizedKSamplerAdvanced()
            sampled_latent = sampler.sample(
                model=model,
                add_noise="enable",
                noise_seed=549535961820368,
                steps=4,
                cfg=1,
                sampler_name="euler",
                scheduler="simple",
                positive=[["test prompt", {"pooled_output": torch.randn(1, 1280), "embeds": torch.randn(1, 77, 1280)}]],
                negative=[["", {"pooled_output": torch.randn(1, 1280), "embeds": torch.randn(1, 77, 1280)}]],
                latent_image=video_latent,
                start_at_step=0,
                end_at_step=2,
                return_with_leftover_noise="enable",
                use_rocm_optimizations=True,
                precision_mode="auto",
                memory_optimization=True
            )
        
        # 4. Decode video
        vae_decoder = ROCMOptimizedVAEDecode()
        final_video = vae_decoder.decode(
            vae=vae,
            samples=sampled_latent[0],
            tile_size=768,
            overlap=96,
            use_rocm_optimizations=True,
            precision_mode="auto",
            batch_optimization=True,
            video_chunk_size=4,
            memory_optimization_enabled=True
        )
        
        # Verify final result
        assert isinstance(final_video, tuple)
        assert len(final_video) == 1
        assert isinstance(final_video[0], torch.Tensor)
        # Should be reshaped to 4D for ComfyUI
        assert len(final_video[0].shape) == 4  # B*T, H, W, C


@pytest.mark.integration
def test_workflow_compatibility():
    """Test compatibility between different workflow types"""
    # Test that nodes work with both Flux and WAN workflows
    
    # Test VAE decode with both image and video latents
    vae_decoder = ROCMOptimizedVAEDecode()
    
    # Image latent
    image_latent = {"samples": torch.randn(1, 4, 64, 64)}
    
    # Video latent
    video_latent = {"samples": torch.randn(1, 4, 8, 64, 64)}
    
    # Mock VAE
    class MockVAE:
        def __init__(self):
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
            return 100 * 1024 * 1024
        
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
    
    mock_vae = MockVAE()
    
    # Test image decode
    image_result = vae_decoder.decode(
        vae=mock_vae,
        samples=image_latent,
        use_rocm_optimizations=True
    )
    
    assert isinstance(image_result, tuple)
    assert len(image_result) == 1
    assert isinstance(image_result[0], torch.Tensor)
    assert len(image_result[0].shape) == 4  # B, H, W, C
    
    # Test video decode
    video_result = vae_decoder.decode(
        vae=mock_vae,
        samples=video_latent,
        use_rocm_optimizations=True,
        video_chunk_size=4,
        memory_optimization_enabled=True
    )
    
    assert isinstance(video_result, tuple)
    assert len(video_result) == 1
    assert isinstance(video_result[0], torch.Tensor)
    assert len(video_result[0].shape) == 4  # B*T, H, W, C


@pytest.mark.integration
def test_error_recovery():
    """Test error recovery in workflows"""
    # Test that workflows can recover from errors
    
    # Test checkpoint loader error recovery
    loader = ROCMOptimizedCheckpointLoader()
    
    with patch('comfy.sd.load_checkpoint_guess_config') as mock_load_checkpoint:
        with patch('folder_paths.get_full_path_or_raise') as mock_get_path:
            # First call fails, second succeeds
            mock_get_path.return_value = "/fake/path/checkpoint.safetensors"
            mock_load_checkpoint.side_effect = [
                Exception("First attempt failed"),
                (Mock(), Mock(), Mock())
            ]
            
            result = loader.load_checkpoint(
                ckpt_name="test_checkpoint.safetensors",
                lazy_loading=True,
                optimize_for_flux=True,
                precision_mode="auto"
            )
            
            assert isinstance(result, tuple)
            assert len(result) == 3

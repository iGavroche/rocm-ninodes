"""
Unit tests for ROCMOptimizedCheckpointLoader node
"""

import pytest
import torch
import os
from unittest.mock import Mock, patch, MagicMock
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from nodes import ROCMOptimizedCheckpointLoader


class TestROCMOptimizedCheckpointLoader:
    """Test cases for ROCMOptimizedCheckpointLoader node"""
    
    def test_input_types(self):
        """Test that INPUT_TYPES returns correct structure"""
        input_types = ROCMOptimizedCheckpointLoader.INPUT_TYPES()
        
        assert "required" in input_types
        required = input_types["required"]
        
        # Check required inputs
        assert "ckpt_name" in required
        assert "lazy_loading" in required
        assert "optimize_for_flux" in required
        assert "precision_mode" in required
    
    def test_return_types(self):
        """Test that RETURN_TYPES is correct"""
        assert ROCMOptimizedCheckpointLoader.RETURN_TYPES == ("MODEL", "CLIP", "VAE")
        assert ROCMOptimizedCheckpointLoader.RETURN_NAMES == ("MODEL", "CLIP", "VAE")
        assert ROCMOptimizedCheckpointLoader.FUNCTION == "load_checkpoint"
        assert ROCMOptimizedCheckpointLoader.CATEGORY == "RocM Ninodes/Loaders"
    
    @patch('comfy.sd.load_checkpoint_guess_config')
    @patch('folder_paths.get_full_path_or_raise')
    def test_load_checkpoint_basic(self, mock_get_path, mock_load_checkpoint):
        """Test basic checkpoint loading functionality"""
        # Mock the checkpoint path
        mock_get_path.return_value = "/fake/path/checkpoint.safetensors"
        
        # Mock the checkpoint loading result
        mock_model = Mock()
        mock_clip = Mock()
        mock_vae = Mock()
        mock_load_checkpoint.return_value = (mock_model, mock_clip, mock_vae)
        
        node = ROCMOptimizedCheckpointLoader()
        
        result = node.load_checkpoint(
            ckpt_name="test_checkpoint.safetensors",
            lazy_loading=True,
            optimize_for_flux=True,
            precision_mode="auto"
        )
        
        assert isinstance(result, tuple)
        assert len(result) == 3
        assert result[0] == mock_model
        assert result[1] == mock_clip
        assert result[2] == mock_vae
        
        # Verify the checkpoint loading was called correctly
        mock_get_path.assert_called_once_with("checkpoints", "test_checkpoint.safetensors")
        mock_load_checkpoint.assert_called_once()
    
    @patch('comfy.sd.load_checkpoint_guess_config')
    @patch('folder_paths.get_full_path_or_raise')
    def test_lazy_loading_options(self, mock_get_path, mock_load_checkpoint):
        """Test lazy loading options"""
        # Mock the checkpoint path
        mock_get_path.return_value = "/fake/path/checkpoint.safetensors"
        
        # Mock the checkpoint loading result
        mock_model = Mock()
        mock_clip = Mock()
        mock_vae = Mock()
        mock_load_checkpoint.return_value = (mock_model, mock_clip, mock_vae)
        
        node = ROCMOptimizedCheckpointLoader()
        
        # Test with lazy loading enabled
        result = node.load_checkpoint(
            ckpt_name="test_checkpoint.safetensors",
            lazy_loading=True,
            optimize_for_flux=True,
            precision_mode="auto"
        )
        
        assert isinstance(result, tuple)
        assert len(result) == 3
        
        # Test with lazy loading disabled
        result = node.load_checkpoint(
            ckpt_name="test_checkpoint.safetensors",
            lazy_loading=False,
            optimize_for_flux=True,
            precision_mode="auto"
        )
        
        assert isinstance(result, tuple)
        assert len(result) == 3
    
    @patch('comfy.sd.load_checkpoint_guess_config')
    @patch('folder_paths.get_full_path_or_raise')
    def test_flux_optimization_options(self, mock_get_path, mock_load_checkpoint):
        """Test Flux optimization options"""
        # Mock the checkpoint path
        mock_get_path.return_value = "/fake/path/checkpoint.safetensors"
        
        # Mock the checkpoint loading result
        mock_model = Mock()
        mock_clip = Mock()
        mock_vae = Mock()
        mock_load_checkpoint.return_value = (mock_model, mock_clip, mock_vae)
        
        node = ROCMOptimizedCheckpointLoader()
        
        # Test with Flux optimization enabled
        result = node.load_checkpoint(
            ckpt_name="test_checkpoint.safetensors",
            lazy_loading=True,
            optimize_for_flux=True,
            precision_mode="auto"
        )
        
        assert isinstance(result, tuple)
        assert len(result) == 3
        
        # Test with Flux optimization disabled
        result = node.load_checkpoint(
            ckpt_name="test_checkpoint.safetensors",
            lazy_loading=True,
            optimize_for_flux=False,
            precision_mode="auto"
        )
        
        assert isinstance(result, tuple)
        assert len(result) == 3
    
    @patch('comfy.sd.load_checkpoint_guess_config')
    @patch('folder_paths.get_full_path_or_raise')
    def test_precision_modes(self, mock_get_path, mock_load_checkpoint):
        """Test different precision modes"""
        # Mock the checkpoint path
        mock_get_path.return_value = "/fake/path/checkpoint.safetensors"
        
        # Mock the checkpoint loading result
        mock_model = Mock()
        mock_clip = Mock()
        mock_vae = Mock()
        mock_load_checkpoint.return_value = (mock_model, mock_clip, mock_vae)
        
        node = ROCMOptimizedCheckpointLoader()
        
        precision_modes = ["auto", "fp32", "fp16", "bf16"]
        
        for precision in precision_modes:
            result = node.load_checkpoint(
                ckpt_name="test_checkpoint.safetensors",
                lazy_loading=True,
                optimize_for_flux=True,
                precision_mode=precision
            )
            
            assert isinstance(result, tuple)
            assert len(result) == 3
    
    @patch('comfy.sd.load_checkpoint_guess_config')
    @patch('folder_paths.get_full_path_or_raise')
    def test_gpu_detection(self, mock_get_path, mock_load_checkpoint):
        """Test GPU detection and logging"""
        # Mock the checkpoint path
        mock_get_path.return_value = "/fake/path/checkpoint.safetensors"
        
        # Mock the checkpoint loading result
        mock_model = Mock()
        mock_clip = Mock()
        mock_vae = Mock()
        mock_load_checkpoint.return_value = (mock_model, mock_clip, mock_vae)
        
        node = ROCMOptimizedCheckpointLoader()
        
        with patch('torch.cuda.is_available', return_value=True):
            with patch('torch.cuda.get_device_name', return_value="AMD Radeon RX 7900 XTX"):
                result = node.load_checkpoint(
                    ckpt_name="test_checkpoint.safetensors",
                    lazy_loading=True,
                    optimize_for_flux=True,
                    precision_mode="auto"
                )
                
                assert isinstance(result, tuple)
                assert len(result) == 3
    
    @patch('comfy.sd.load_checkpoint_guess_config')
    @patch('folder_paths.get_full_path_or_raise')
    def test_error_handling_primary(self, mock_get_path, mock_load_checkpoint):
        """Test error handling in primary loading"""
        # Mock the checkpoint path
        mock_get_path.return_value = "/fake/path/checkpoint.safetensors"
        
        # Mock the checkpoint loading to raise an exception
        mock_load_checkpoint.side_effect = Exception("Primary loading failed")
        
        node = ROCMOptimizedCheckpointLoader()
        
        with pytest.raises(Exception, match="Primary loading failed"):
            node.load_checkpoint(
                ckpt_name="test_checkpoint.safetensors",
                lazy_loading=True,
                optimize_for_flux=True,
                precision_mode="auto"
            )
    
    @patch('comfy.sd.load_checkpoint_guess_config')
    @patch('folder_paths.get_full_path_or_raise')
    def test_error_handling_fallback(self, mock_get_path, mock_load_checkpoint):
        """Test error handling with fallback loading"""
        # Mock the checkpoint path
        mock_get_path.return_value = "/fake/path/checkpoint.safetensors"
        
        # Mock the checkpoint loading to fail first, then succeed
        mock_model = Mock()
        mock_clip = Mock()
        mock_vae = Mock()
        
        mock_load_checkpoint.side_effect = [
            Exception("Primary loading failed"),
            (mock_model, mock_clip, mock_vae)
        ]
        
        node = ROCMOptimizedCheckpointLoader()
        
        result = node.load_checkpoint(
            ckpt_name="test_checkpoint.safetensors",
            lazy_loading=True,
            optimize_for_flux=True,
            precision_mode="auto"
        )
        
        assert isinstance(result, tuple)
        assert len(result) == 3
        assert result[0] == mock_model
        assert result[1] == mock_clip
        assert result[2] == mock_vae
    
    @patch('comfy.sd.load_checkpoint_guess_config')
    @patch('folder_paths.get_full_path_or_raise')
    def test_error_handling_all_fail(self, mock_get_path, mock_load_checkpoint):
        """Test error handling when all loading attempts fail"""
        # Mock the checkpoint path
        mock_get_path.return_value = "/fake/path/checkpoint.safetensors"
        
        # Mock the checkpoint loading to always fail
        mock_load_checkpoint.side_effect = Exception("All loading attempts failed")
        
        node = ROCMOptimizedCheckpointLoader()
        
        with pytest.raises(Exception, match="All loading attempts failed"):
            node.load_checkpoint(
                ckpt_name="test_checkpoint.safetensors",
                lazy_loading=True,
                optimize_for_flux=True,
                precision_mode="auto"
            )
    
    @patch('comfy.sd.load_checkpoint_guess_config')
    @patch('folder_paths.get_full_path_or_raise')
    def test_invalid_checkpoint_name(self, mock_get_path, mock_load_checkpoint):
        """Test handling of invalid checkpoint names"""
        # Mock the checkpoint path to raise an exception
        mock_get_path.side_effect = FileNotFoundError("Checkpoint not found")
        
        node = ROCMOptimizedCheckpointLoader()
        
        with pytest.raises(FileNotFoundError, match="Checkpoint not found"):
            node.load_checkpoint(
                ckpt_name="nonexistent_checkpoint.safetensors",
                lazy_loading=True,
                optimize_for_flux=True,
                precision_mode="auto"
            )
    
    @patch('comfy.sd.load_checkpoint_guess_config')
    @patch('folder_paths.get_full_path_or_raise')
    def test_incomplete_loading_result(self, mock_get_path, mock_load_checkpoint):
        """Test handling of incomplete loading results"""
        # Mock the checkpoint path
        mock_get_path.return_value = "/fake/path/checkpoint.safetensors"
        
        # Mock the checkpoint loading to return incomplete result
        mock_load_checkpoint.return_value = (Mock(), Mock())  # Only 2 items instead of 3
        
        node = ROCMOptimizedCheckpointLoader()
        
        with pytest.raises(ValueError, match="Checkpoint loading returned 2 items, expected 3"):
            node.load_checkpoint(
                ckpt_name="test_checkpoint.safetensors",
                lazy_loading=True,
                optimize_for_flux=True,
                precision_mode="auto"
            )
    
    def test_performance_timing(self):
        """Test that performance timing works"""
        node = ROCMOptimizedCheckpointLoader()
        
        with patch('comfy.sd.load_checkpoint_guess_config') as mock_load_checkpoint:
            with patch('folder_paths.get_full_path_or_raise') as mock_get_path:
                # Mock the checkpoint path
                mock_get_path.return_value = "/fake/path/checkpoint.safetensors"
                
                # Mock the checkpoint loading result
                mock_model = Mock()
                mock_clip = Mock()
                mock_vae = Mock()
                mock_load_checkpoint.return_value = (mock_model, mock_clip, mock_vae)
                
                import time
                start_time = time.time()
                
                result = node.load_checkpoint(
                    ckpt_name="test_checkpoint.safetensors",
                    lazy_loading=True,
                    optimize_for_flux=True,
                    precision_mode="auto"
                )
                
                end_time = time.time()
                execution_time = end_time - start_time
                
                assert execution_time > 0
                assert isinstance(result, tuple)
                assert len(result) == 3


@pytest.mark.parametrize("ckpt_name", [
    "flux-dev.safetensors",
    "flux-schnell.safetensors", 
    "wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors",
    "wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors"
])
def test_different_checkpoint_types(ckpt_name):
    """Test loading different types of checkpoints"""
    node = ROCMOptimizedCheckpointLoader()
    
    with patch('comfy.sd.load_checkpoint_guess_config') as mock_load_checkpoint:
        with patch('folder_paths.get_full_path_or_raise') as mock_get_path:
            # Mock the checkpoint path
            mock_get_path.return_value = f"/fake/path/{ckpt_name}"
            
            # Mock the checkpoint loading result
            mock_model = Mock()
            mock_clip = Mock()
            mock_vae = Mock()
            mock_load_checkpoint.return_value = (mock_model, mock_clip, mock_vae)
            
            result = node.load_checkpoint(
                ckpt_name=ckpt_name,
                lazy_loading=True,
                optimize_for_flux=True,
                precision_mode="auto"
            )
            
            assert isinstance(result, tuple)
            assert len(result) == 3
            assert result[0] == mock_model
            assert result[1] == mock_clip
            assert result[2] == mock_vae

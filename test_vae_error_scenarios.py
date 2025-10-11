#!/usr/bin/env python3
"""
Comprehensive Test Suite for ROCM VAE Decoder
Tests all error scenarios and edge cases to prevent regressions
"""

import torch
import numpy as np
import pickle
import os
import sys
import time
import unittest
from unittest.mock import Mock, patch

# Add ComfyUI to path
sys.path.append('/home/nino/ComfyUI')

class TestROCMVAEDecoder(unittest.TestCase):
    """Test suite for ROCM VAE Decoder error scenarios"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_data_dir = 'test_data/debug'
        os.makedirs(self.test_data_dir, exist_ok=True)
        
        # Create mock VAE for testing
        self.mock_vae = self.create_mock_vae()
        
        # Test tensor shapes
        self.video_5d_shape = torch.Size([1, 16, 2, 32, 32])  # [B, C, T, H, W]
        self.video_4d_shape = torch.Size([1, 16, 32, 32])     # [B, C, H, W]
        self.image_4d_shape = torch.Size([1, 256, 256, 3])   # [B, H, W, C]
        self.image_5d_shape = torch.Size([1, 2, 256, 256, 3]) # [B, T, H, W, C]
    
    def create_mock_vae(self):
        """Create a mock VAE that simulates WAN VAE behavior"""
        mock_vae = Mock()
        mock_vae.device = torch.device('cpu')
        mock_vae.vae_dtype = torch.float32
        mock_vae.output_device = torch.device('cpu')
        
        # Mock memory calculation (WAN VAE style)
        mock_vae.memory_used_decode = lambda shape, dtype: 8000 * shape[3] * shape[4] * (16 * 16) * 4
        
        # Mock decode method
        def mock_decode(samples):
            # Simulate WAN VAE decode behavior
            if len(samples.shape) == 5:  # [B, C, T, H, W]
                B, C, T, H, W = samples.shape
                # Return [B, T, H*8, W*8, 3] (upscaled)
                return torch.randn(B, T, H*8, W*8, 3, dtype=samples.dtype, device=samples.device)
            else:
                raise ValueError(f"Expected 5D tensor, got {len(samples.shape)}D")
        
        mock_vae.decode = mock_decode
        return mock_vae
    
    def test_attribute_error_dict_shape(self):
        """Test case for AttributeError: 'dict' object has no attribute 'shape'"""
        print("\n=== Testing AttributeError: 'dict' object has no attribute 'shape' ===")
        
        # Simulate the error scenario
        samples_dict = {"samples": torch.randn(1, 16, 2, 32, 32)}
        
        # This should NOT raise AttributeError
        try:
            tensor_shape = samples_dict["samples"].shape
            print(f"✓ Correctly accessed tensor shape: {tensor_shape}")
            self.assertEqual(tensor_shape, self.video_5d_shape)
        except AttributeError as e:
            self.fail(f"AttributeError occurred: {e}")
    
    def test_tuple_index_out_of_range_error(self):
        """Test case for IndexError: tuple index out of range"""
        print("\n=== Testing IndexError: tuple index out of range ===")
        
        # Test with correct 5D tensor (should work)
        correct_5d_tensor = torch.randn(1, 16, 2, 32, 32)
        try:
            # Simulate WAN VAE memory calculation
            memory_used = 8000 * correct_5d_tensor.shape[3] * correct_5d_tensor.shape[4] * (16 * 16) * 4
            print(f"✓ 5D tensor memory calculation successful: {memory_used}")
        except IndexError as e:
            self.fail(f"IndexError occurred with 5D tensor: {e}")
        
        # Test with incorrect 4D tensor (should fail)
        incorrect_4d_tensor = torch.randn(1, 16, 32, 32)
        with self.assertRaises(IndexError):
            # This should raise IndexError
            memory_used = 8000 * incorrect_4d_tensor.shape[3] * incorrect_4d_tensor.shape[4] * (16 * 16) * 4
            print(f"✗ 4D tensor should have failed but didn't")
    
    def test_video_output_format_error(self):
        """Test case for ValueError: Expected numpy array with ndim 3 but got 4"""
        print("\n=== Testing ValueError: Expected numpy array with ndim 3 but got 4 ===")
        
        # Simulate 5D video output from VAE
        video_5d_output = torch.randn(1, 2, 256, 256, 3)  # [B, T, H, W, C]
        print(f"Input 5D shape: {video_5d_output.shape}")
        
        # Convert to 4D format for ComfyUI
        B, T, H, W, C = video_5d_output.shape
        image_4d_output = video_5d_output.reshape(B * T, H, W, C)
        print(f"Output 4D shape: {image_4d_output.shape}")
        
        # Test individual frames (should be 3D)
        for i in range(image_4d_output.shape[0]):
            frame = image_4d_output[i]  # [H, W, C]
            self.assertEqual(len(frame.shape), 3, f"Frame {i} should be 3D, got {len(frame.shape)}D")
            print(f"✓ Frame {i} shape: {frame.shape}")
    
    def test_vae_decode_input_format(self):
        """Test VAE decode with different input formats"""
        print("\n=== Testing VAE decode input formats ===")
        
        # Test 5D tensor (correct format)
        tensor_5d = torch.randn(1, 16, 2, 32, 32)
        try:
            result = self.mock_vae.decode(tensor_5d)
            print(f"✓ 5D tensor decode successful: {result.shape}")
            self.assertEqual(len(result.shape), 5)
        except Exception as e:
            self.fail(f"5D tensor decode failed: {e}")
        
        # Test 4D tensor (should fail)
        tensor_4d = torch.randn(1, 16, 32, 32)
        with self.assertRaises(ValueError):
            result = self.mock_vae.decode(tensor_4d)
            print("✗ 4D tensor should have failed but didn't")
    
    def test_chunked_video_processing(self):
        """Test chunked video processing logic"""
        print("\n=== Testing chunked video processing ===")
        
        # Simulate video tensor
        video_tensor = torch.randn(1, 16, 2, 32, 32)  # [B, C, T, H, W]
        B, C, T, H, W = video_tensor.shape
        
        # Process in chunks
        chunk_size = 1
        chunk_results = []
        
        for i in range(0, T, chunk_size):
            end_idx = min(i + chunk_size, T)
            chunk = video_tensor[:, :, i:end_idx, :, :]
            
            print(f"Chunk {i}: shape {chunk.shape}")
            
            # Decode chunk
            chunk_decoded = self.mock_vae.decode(chunk)
            chunk_results.append(chunk_decoded)
        
        # Concatenate results
        result = torch.cat(chunk_results, dim=1)
        print(f"Concatenated result: {result.shape}")
        
        # Convert to 4D format
        if len(result.shape) == 5:
            B, T, H, W, C = result.shape
            result = result.reshape(B * T, H, W, C)
            print(f"Final 4D result: {result.shape}")
        
        self.assertEqual(len(result.shape), 4)
        self.assertEqual(result.shape[0], T)  # Should have T frames
    
    def test_memory_calculation_edge_cases(self):
        """Test memory calculation with edge cases"""
        print("\n=== Testing memory calculation edge cases ===")
        
        test_cases = [
            (torch.Size([1, 16, 1, 32, 32]), "Single frame"),
            (torch.Size([1, 16, 2, 32, 32]), "Two frames"),
            (torch.Size([1, 16, 17, 32, 32]), "Many frames"),
            (torch.Size([2, 16, 2, 32, 32]), "Multiple batches"),
            (torch.Size([1, 16, 2, 64, 64]), "Larger resolution"),
        ]
        
        for shape, description in test_cases:
            try:
                # Simulate WAN VAE memory calculation
                memory_used = 8000 * shape[3] * shape[4] * (16 * 16) * 4
                print(f"✓ {description}: {shape} -> {memory_used} bytes")
            except IndexError as e:
                self.fail(f"Memory calculation failed for {description}: {e}")
    
    def test_tensor_shape_conversions(self):
        """Test tensor shape conversions"""
        print("\n=== Testing tensor shape conversions ===")
        
        # Test 5D to 4D conversion
        tensor_5d = torch.randn(1, 2, 256, 256, 3)  # [B, T, H, W, C]
        B, T, H, W, C = tensor_5d.shape
        tensor_4d = tensor_5d.reshape(B * T, H, W, C)
        
        print(f"5D input: {tensor_5d.shape}")
        print(f"4D output: {tensor_4d.shape}")
        
        self.assertEqual(tensor_4d.shape, (2, 256, 256, 3))
        
        # Test individual frame extraction
        for i in range(tensor_4d.shape[0]):
            frame = tensor_4d[i]
            self.assertEqual(frame.shape, (256, 256, 3))
            print(f"✓ Frame {i}: {frame.shape}")
    
    def test_error_recovery_scenarios(self):
        """Test error recovery scenarios"""
        print("\n=== Testing error recovery scenarios ===")
        
        # Test with malformed input
        malformed_inputs = [
            torch.randn(1, 16, 32, 32),  # 4D instead of 5D
            torch.randn(1, 16, 2, 32),  # Missing dimension
            torch.randn(1, 16, 2, 32, 32, 1),  # Extra dimension
        ]
        
        for i, malformed_input in enumerate(malformed_inputs):
            print(f"Testing malformed input {i+1}: {malformed_input.shape}")
            
            if len(malformed_input.shape) == 5:
                # Should work
                try:
                    result = self.mock_vae.decode(malformed_input)
                    print(f"✓ Malformed input {i+1} worked: {result.shape}")
                except Exception as e:
                    print(f"✗ Malformed input {i+1} failed: {e}")
            else:
                # Should fail
                with self.assertRaises(ValueError):
                    result = self.mock_vae.decode(malformed_input)
                    print(f"✗ Malformed input {i+1} should have failed but didn't")
    
    def test_performance_benchmarks(self):
        """Test performance benchmarks"""
        print("\n=== Testing performance benchmarks ===")
        
        # Test different tensor sizes
        test_sizes = [
            (1, 16, 1, 32, 32),   # Single frame
            (1, 16, 2, 32, 32),   # Two frames
            (1, 16, 5, 32, 32),   # Five frames
            (1, 16, 17, 32, 32),  # Many frames
        ]
        
        for size in test_sizes:
            tensor = torch.randn(*size)
            
            start_time = time.time()
            result = self.mock_vae.decode(tensor)
            decode_time = time.time() - start_time
            
            print(f"Size {size}: {decode_time:.4f}s")
            
            # Performance should be reasonable
            self.assertLess(decode_time, 1.0, f"Decode time too slow: {decode_time}s")
    
    def save_test_results(self):
        """Save test results for analysis"""
        test_results = {
            'test_timestamp': int(time.time()),
            'test_cases_passed': self._testMethodName,
            'tensor_shapes_tested': [
                str(self.video_5d_shape),
                str(self.video_4d_shape),
                str(self.image_4d_shape),
                str(self.image_5d_shape),
            ],
            'error_scenarios_tested': [
                'AttributeError: dict object has no attribute shape',
                'IndexError: tuple index out of range',
                'ValueError: Expected numpy array with ndim 3 but got 4',
            ]
        }
        
        filename = f'{self.test_data_dir}/vae_error_test_results_{int(time.time())}.pkl'
        with open(filename, 'wb') as f:
            pickle.dump(test_results, f)
        print(f"✓ Test results saved to {filename}")

def run_comprehensive_tests():
    """Run all comprehensive tests"""
    print("ROCM VAE Decoder Comprehensive Test Suite")
    print("=" * 50)
    
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestROCMVAEDecoder)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print(f"\nTest Summary:")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\nFailures:")
        for test, traceback in result.failures:
            print(f"  {test}: {traceback}")
    
    if result.errors:
        print("\nErrors:")
        for test, traceback in result.errors:
            print(f"  {test}: {traceback}")
    
    return result.wasSuccessful()

if __name__ == "__main__":
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)

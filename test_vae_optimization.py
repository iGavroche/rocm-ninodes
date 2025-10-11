#!/usr/bin/env python3
"""
VAE Decoder Optimization Test Suite
Tests different optimization strategies for the ROCM VAE decoder
"""

import pickle
import torch
import time
import sys
import os
import gc

# Add ComfyUI to path
sys.path.append('/home/nino/ComfyUI')

def load_test_data():
    """Load the saved test data"""
    import glob
    
    # Find the most recent debug file
    debug_files = glob.glob('test_data/debug/wan_vae_input_debug_*.pkl')
    if not debug_files:
        print("✗ No debug data found in test_data/debug/")
        print("Run the WAN workflow first to generate test data")
        return None
    
    # Use the most recent file
    latest_file = max(debug_files, key=os.path.getctime)
    print(f"✓ Loading debug data from: {latest_file}")
    
    try:
        with open(latest_file, 'rb') as f:
            debug_data = pickle.load(f)
        print("✓ Loaded debug data successfully")
        return debug_data
    except Exception as e:
        print(f"✗ Error loading debug data: {e}")
        return None

def create_mock_vae():
    """Create a mock VAE for testing (when real VAE is not available)"""
    class MockVAE:
        def __init__(self):
            self.vae_dtype = torch.float32
            
        def decode(self, samples):
            """Mock decode that simulates VAE behavior"""
            # Simulate some processing time
            time.sleep(0.01)
            
            # Convert latent to image-like tensor
            # Input: [B, C, H, W] where C=16 (latent channels)
            # Output: [B, 3, H*8, W*8] (RGB image)
            B, C, H, W = samples.shape
            output = torch.randn(B, 3, H*8, W*8, dtype=samples.dtype, device=samples.device)
            return output
            
        def memory_used_decode(self, shape, dtype):
            """Mock memory calculation"""
            return shape[0] * shape[1] * shape[2] * shape[3] * 4  # 4 bytes per float32
    
    return MockVAE()

def test_direct_decode(debug_data):
    """Test direct tensor decode (current working method)"""
    print("\n=== Testing Direct Tensor Decode ===")
    
    tensor = debug_data['chunk_reshaped_tensor']
    mock_vae = create_mock_vae()
    
    # Test multiple times for timing
    times = []
    for i in range(5):
        start_time = time.time()
        result = mock_vae.decode(tensor)
        decode_time = time.time() - start_time
        times.append(decode_time)
        print(f"  Run {i+1}: {decode_time:.4f}s")
    
    avg_time = sum(times) / len(times)
    print(f"  Average time: {avg_time:.4f}s")
    print(f"  Result shape: {result.shape}")
    
    return avg_time, result

def test_chunked_decode(debug_data, chunk_size=1):
    """Test chunked decoding strategy"""
    print(f"\n=== Testing Chunked Decode (chunk_size={chunk_size}) ===")
    
    tensor = debug_data['chunk_reshaped_tensor']
    mock_vae = create_mock_vae()
    
    B, C, H, W = tensor.shape
    
    # Simulate chunked processing
    times = []
    for i in range(5):
        start_time = time.time()
        
        # Process in chunks
        chunk_results = []
        for j in range(0, B, chunk_size):
            end_idx = min(j + chunk_size, B)
            chunk = tensor[j:end_idx]
            
            chunk_result = mock_vae.decode(chunk)
            chunk_results.append(chunk_result)
            
            # Simulate memory cleanup
            gc.collect()
        
        # Concatenate results
        result = torch.cat(chunk_results, dim=0)
        
        decode_time = time.time() - start_time
        times.append(decode_time)
        print(f"  Run {i+1}: {decode_time:.4f}s")
    
    avg_time = sum(times) / len(times)
    print(f"  Average time: {avg_time:.4f}s")
    print(f"  Result shape: {result.shape}")
    
    return avg_time, result

def test_memory_optimization(debug_data):
    """Test memory optimization strategies"""
    print("\n=== Testing Memory Optimization Strategies ===")
    
    tensor = debug_data['chunk_reshaped_tensor']
    mock_vae = create_mock_vae()
    
    strategies = [
        ("No optimization", lambda t: mock_vae.decode(t)),
        ("CPU offload", lambda t: mock_vae.decode(t.cpu()).cuda() if torch.cuda.is_available() else mock_vae.decode(t)),
        ("Half precision", lambda t: mock_vae.decode(t.half()).float()),
        ("Mixed precision", lambda t: mock_vae.decode(t.float())),
    ]
    
    results = {}
    for name, strategy in strategies:
        print(f"\n  Testing: {name}")
        times = []
        
        for i in range(3):
            start_time = time.time()
            try:
                result = strategy(tensor)
                decode_time = time.time() - start_time
                times.append(decode_time)
                print(f"    Run {i+1}: {decode_time:.4f}s")
            except Exception as e:
                print(f"    Run {i+1}: FAILED - {e}")
                times.append(float('inf'))
        
        avg_time = sum(t for t in times if t != float('inf')) / len([t for t in times if t != float('inf')])
        results[name] = avg_time
        print(f"    Average: {avg_time:.4f}s")
    
    return results

def test_different_precisions(debug_data):
    """Test different precision modes"""
    print("\n=== Testing Different Precision Modes ===")
    
    tensor = debug_data['chunk_reshaped_tensor']
    mock_vae = create_mock_vae()
    
    precisions = [
        ("fp32", torch.float32),
        ("fp16", torch.float16),
        ("bf16", torch.bfloat16),
    ]
    
    results = {}
    for name, dtype in precisions:
        print(f"\n  Testing: {name}")
        times = []
        
        for i in range(3):
            try:
                test_tensor = tensor.to(dtype)
                start_time = time.time()
                result = mock_vae.decode(test_tensor)
                decode_time = time.time() - start_time
                times.append(decode_time)
                print(f"    Run {i+1}: {decode_time:.4f}s")
            except Exception as e:
                print(f"    Run {i+1}: FAILED - {e}")
                times.append(float('inf'))
        
        avg_time = sum(t for t in times if t != float('inf')) / len([t for t in times if t != float('inf')])
        results[name] = avg_time
        print(f"    Average: {avg_time:.4f}s")
    
    return results

def generate_optimization_report(results):
    """Generate an optimization report"""
    print("\n" + "="*60)
    print("VAE DECODER OPTIMIZATION REPORT")
    print("="*60)
    
    print("\n📊 Performance Results:")
    for category, data in results.items():
        print(f"\n{category}:")
        if isinstance(data, dict):
            for name, time_val in data.items():
                print(f"  {name}: {time_val:.4f}s")
        else:
            print(f"  {data:.4f}s")
    
    print("\n🎯 Optimization Recommendations:")
    
    # Find best performing strategy
    if 'memory_optimization' in results:
        best_memory = min(results['memory_optimization'].items(), key=lambda x: x[1])
        print(f"  • Best memory strategy: {best_memory[0]} ({best_memory[1]:.4f}s)")
    
    if 'precision_test' in results:
        best_precision = min(results['precision_test'].items(), key=lambda x: x[1])
        print(f"  • Best precision: {best_precision[0]} ({best_precision[1]:.4f}s)")
    
    print(f"  • Direct decode: {results['direct_decode']:.4f}s")
    print(f"  • Chunked decode: {results['chunked_decode']:.4f}s")
    
    if results['direct_decode'] < results['chunked_decode']:
        print("  • Recommendation: Use direct decode for better performance")
    else:
        print("  • Recommendation: Use chunked decode for better memory usage")

def main():
    """Main test function"""
    print("VAE Decoder Optimization Test Suite")
    print("="*50)
    
    # Load test data
    debug_data = load_test_data()
    if debug_data is None:
        return
    
    print(f"\nTest tensor info:")
    print(f"  Shape: {debug_data['chunk_reshaped_shape']}")
    print(f"  Dtype: {debug_data['chunk_reshaped_dtype']}")
    print(f"  Device: {debug_data['chunk_reshaped_device']}")
    
    # Run tests
    results = {}
    
    # Test direct decode
    direct_time, direct_result = test_direct_decode(debug_data)
    results['direct_decode'] = direct_time
    
    # Test chunked decode
    chunked_time, chunked_result = test_chunked_decode(debug_data)
    results['chunked_decode'] = chunked_time
    
    # Test memory optimization
    memory_results = test_memory_optimization(debug_data)
    results['memory_optimization'] = memory_results
    
    # Test different precisions
    precision_results = test_different_precisions(debug_data)
    results['precision_test'] = precision_results
    
    # Generate report
    generate_optimization_report(results)
    
    # Save results with timestamp
    timestamp = int(time.time())
    os.makedirs('test_data/optimization', exist_ok=True)
    filename = f'test_data/optimization/vae_optimization_results_{timestamp}.pkl'
    with open(filename, 'wb') as f:
        pickle.dump(results, f)
    print(f"\n💾 Saved optimization results to {filename}")

if __name__ == "__main__":
    main()

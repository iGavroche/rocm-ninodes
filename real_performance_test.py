#!/usr/bin/env python3
"""
Real VAE Performance Test
Test with actual ComfyUI environment to identify performance issues
"""
import sys
import os
sys.path.insert(0, '/home/nino/ComfyUI')

import time
import subprocess

def test_real_performance():
    """Test real performance by running a simple ComfyUI workflow"""
    print("🔍 Real VAE Performance Test")
    print("=" * 40)
    
    # Check if we can import the nodes
    try:
        from custom_nodes.rocm_ninodes.nodes import ROCMOptimizedVAEDecodeInstrumented
        print("✅ Successfully imported ROCMOptimizedVAEDecodeInstrumented")
    except Exception as e:
        print(f"❌ Failed to import nodes: {e}")
        return
    
    # Check if ComfyUI modules are available
    try:
        import comfy.model_management
        import comfy.utils
        print("✅ ComfyUI modules available")
    except Exception as e:
        print(f"❌ ComfyUI modules not available: {e}")
        return
    
    # Check GPU availability
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA available: {torch.cuda.get_device_name()}")
            print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
        else:
            print("⚠️  CUDA not available - using CPU")
    except Exception as e:
        print(f"❌ Torch error: {e}")
        return
    
    print("\n📊 Performance Analysis:")
    print("The 173% slowdown (55s → 150s) could be caused by:")
    print("1. Missing optimizations from original implementation")
    print("2. Instrumentation overhead (though we disabled it)")
    print("3. Memory management issues")
    print("4. Tile size or precision problems")
    print("5. Batch processing issues")
    
    print("\n🔧 Next Steps:")
    print("1. Compare original vs simplified implementation")
    print("2. Check for missing critical optimizations")
    print("3. Test with real workflow data")
    print("4. Profile memory usage and GPU utilization")

if __name__ == "__main__":
    test_real_performance()

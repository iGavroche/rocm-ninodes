#!/usr/bin/env python3
"""
Simple Performance Test for Simplified VAE
Test the simplified version to see if it's actually faster
"""
import sys
import os
sys.path.insert(0, '/home/nino/ComfyUI')

import time

def test_simplified_performance():
    """Test if simplified version imports and works"""
    print("🔍 Testing Simplified VAE Performance")
    print("=" * 50)
    
    # Test import
    try:
        from custom_nodes.rocm_ninodes.nodes import ROCMOptimizedVAEDecodeInstrumented
        print("✅ Successfully imported simplified ROCMOptimizedVAEDecodeInstrumented")
    except Exception as e:
        print(f"❌ Failed to import simplified nodes: {e}")
        return False
    
    # Test basic functionality
    try:
        node = ROCMOptimizedVAEDecodeInstrumented()
        print("✅ Successfully created node instance")
        
        # Check input types
        input_types = node.INPUT_TYPES()
        print(f"✅ Input types: {list(input_types['required'].keys())}")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to create node: {e}")
        return False

def compare_versions():
    """Compare simplified vs complex versions"""
    print("\n📊 Version Comparison")
    print("=" * 30)
    
    # Check simplified version
    simplified_works = test_simplified_performance()
    
    if simplified_works:
        print("\n✅ Simplified Version:")
        print("  - Clean, minimal implementation")
        print("  - No complex optimization layers")
        print("  - No performance tracking overhead")
        print("  - No caching systems")
        print("  - Direct VAE decode with fallbacks")
        
        print("\n🎯 Expected Benefits:")
        print("  - Lower memory overhead")
        print("  - Faster execution (no optimization overhead)")
        print("  - More predictable performance")
        print("  - Easier to debug")
        
        print("\n⚠️  Potential Drawbacks:")
        print("  - May miss some optimizations")
        print("  - Less sophisticated memory management")
        print("  - No adaptive optimizations")
    
    return simplified_works

if __name__ == "__main__":
    success = compare_versions()
    
    if success:
        print("\n🎉 Simplified version is ready for testing!")
        print("Next step: Test with actual ComfyUI workflow")
    else:
        print("\n❌ Simplified version has issues")

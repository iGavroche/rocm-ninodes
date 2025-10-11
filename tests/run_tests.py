"""
Test runner for ROCM Ninodes test suite
"""
import sys
import os
import pytest
import subprocess
from pathlib import Path

def run_tests():
    """Run the complete test suite"""
    print("🧪 ROCM Ninodes Test Suite")
    print("=" * 50)
    
    # Ensure we're in the right directory
    test_dir = Path(__file__).parent
    os.chdir(test_dir.parent)
    
    # Run unit tests
    print("\n📋 Running Unit Tests...")
    unit_result = pytest.main([
        "tests/unit/",
        "-v",
        "--tb=short"
    ])
    
    # Run integration tests
    print("\n🔗 Running Integration Tests...")
    integration_result = pytest.main([
        "tests/integration/",
        "-v",
        "--tb=short"
    ])
    
    # Run benchmark tests
    print("\n⚡ Running Performance Benchmarks...")
    benchmark_result = pytest.main([
        "tests/benchmarks/",
        "-v",
        "--tb=short"
    ])
    
    # Summary
    print("\n📊 Test Results Summary")
    print("=" * 30)
    print(f"Unit Tests: {'✅ PASSED' if unit_result == 0 else '❌ FAILED'}")
    print(f"Integration Tests: {'✅ PASSED' if integration_result == 0 else '❌ FAILED'}")
    print(f"Benchmark Tests: {'✅ PASSED' if benchmark_result == 0 else '❌ FAILED'}")
    
    overall_result = unit_result + integration_result + benchmark_result
    
    if overall_result == 0:
        print("\n🎉 All tests passed!")
    else:
        print(f"\n⚠️  {overall_result} test failures detected")
    
    return overall_result == 0

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)

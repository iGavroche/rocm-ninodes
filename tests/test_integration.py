#!/usr/bin/env python3
"""
Main integration test runner
Combines Flux and WAN integration tests
"""

import pytest
import sys
import os

def main():
    """Run all integration tests"""
    print("ROCM Ninodes Integration Tests")
    print("=============================")
    
    # Run Flux integration tests
    print("\n1. Flux Integration Tests")
    print("-" * 30)
    flux_result = pytest.main(["-v", "test_integration_flux.py"])
    
    # Run WAN integration tests
    print("\n2. WAN Integration Tests")
    print("-" * 30)
    wan_result = pytest.main(["-v", "test_integration_wan.py"])
    
    # Summary
    print("\nIntegration Test Summary")
    print("========================")
    print(f"Flux tests: {'PASSED' if flux_result == 0 else 'FAILED'}")
    print(f"WAN tests: {'PASSED' if wan_result == 0 else 'FAILED'}")
    
    if flux_result == 0 and wan_result == 0:
        print("\n✓ All integration tests passed!")
        return 0
    else:
        print("\n✗ Some integration tests failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())

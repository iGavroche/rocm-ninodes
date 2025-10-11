#!/usr/bin/env python3
"""
Test runner for ROCM Ninodes
Runs all tests and generates reports
"""

import sys
import os
import subprocess
import json
import time
from pathlib import Path
from typing import Dict, List, Any

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from instrumentation import NodeInstrumentation


class TestRunner:
    """Test runner for ROCM Ninodes"""
    
    def __init__(self):
        self.test_dir = Path(__file__).parent
        self.project_root = self.test_dir.parent
        self.results = {}
        self.instrumentation = NodeInstrumentation()
    
    def run_unit_tests(self) -> Dict[str, Any]:
        """Run unit tests"""
        print("Running unit tests...")
        
        unit_tests = [
            "test_vae_decode.py",
            "test_ksampler.py", 
            "test_checkpoint_loader.py"
        ]
        
        results = {}
        
        for test_file in unit_tests:
            test_path = self.test_dir / "unit" / test_file
            if test_path.exists():
                print(f"  Running {test_file}...")
                
                # Run pytest for this specific test file
                cmd = [
                    sys.executable, "-m", "pytest", 
                    str(test_path),
                    "-v", "--tb=short"
                ]
                
                try:
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
                    results[test_file] = {
                        "returncode": result.returncode,
                        "stdout": result.stdout,
                        "stderr": result.stderr,
                        "success": result.returncode == 0
                    }
                    
                    if result.returncode == 0:
                        print(f"    ✓ {test_file} passed")
                    else:
                        print(f"    ✗ {test_file} failed")
                        print(f"    Error: {result.stderr}")
                        
                except subprocess.TimeoutExpired:
                    results[test_file] = {
                        "returncode": -1,
                        "stdout": "",
                        "stderr": "Test timed out",
                        "success": False
                    }
                    print(f"    ✗ {test_file} timed out")
                except Exception as e:
                    results[test_file] = {
                        "returncode": -1,
                        "stdout": "",
                        "stderr": str(e),
                        "success": False
                    }
                    print(f"    ✗ {test_file} error: {e}")
        
        return results
    
    def run_integration_tests(self) -> Dict[str, Any]:
        """Run integration tests"""
        print("Running integration tests...")
        
        integration_tests = [
            "test_workflows.py"
        ]
        
        results = {}
        
        for test_file in integration_tests:
            test_path = self.test_dir / "integration" / test_file
            if test_path.exists():
                print(f"  Running {test_file}...")
                
                cmd = [
                    sys.executable, "-m", "pytest", 
                    str(test_path),
                    "-v", "--tb=short", "-m", "integration"
                ]
                
                try:
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
                    results[test_file] = {
                        "returncode": result.returncode,
                        "stdout": result.stdout,
                        "stderr": result.stderr,
                        "success": result.returncode == 0
                    }
                    
                    if result.returncode == 0:
                        print(f"    ✓ {test_file} passed")
                    else:
                        print(f"    ✗ {test_file} failed")
                        print(f"    Error: {result.stderr}")
                        
                except subprocess.TimeoutExpired:
                    results[test_file] = {
                        "returncode": -1,
                        "stdout": "",
                        "stderr": "Test timed out",
                        "success": False
                    }
                    print(f"    ✗ {test_file} timed out")
                except Exception as e:
                    results[test_file] = {
                        "returncode": -1,
                        "stdout": "",
                        "stderr": str(e),
                        "success": False
                    }
                    print(f"    ✗ {test_file} error: {e}")
        
        return results
    
    def run_benchmark_tests(self) -> Dict[str, Any]:
        """Run benchmark tests"""
        print("Running benchmark tests...")
        
        benchmark_tests = [
            "test_performance.py"
        ]
        
        results = {}
        
        for test_file in benchmark_tests:
            test_path = self.test_dir / "benchmarks" / test_file
            if test_path.exists():
                print(f"  Running {test_file}...")
                
                cmd = [
                    sys.executable, "-m", "pytest", 
                    str(test_path),
                    "-v", "--tb=short", "-m", "benchmark"
                ]
                
                try:
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=1200)
                    results[test_file] = {
                        "returncode": result.returncode,
                        "stdout": result.stdout,
                        "stderr": result.stderr,
                        "success": result.returncode == 0
                    }
                    
                    if result.returncode == 0:
                        print(f"    ✓ {test_file} passed")
                    else:
                        print(f"    ✗ {test_file} failed")
                        print(f"    Error: {result.stderr}")
                        
                except subprocess.TimeoutExpired:
                    results[test_file] = {
                        "returncode": -1,
                        "stdout": "",
                        "stderr": "Test timed out",
                        "success": False
                    }
                    print(f"    ✗ {test_file} timed out")
                except Exception as e:
                    results[test_file] = {
                        "returncode": -1,
                        "stdout": "",
                        "stderr": str(e),
                        "success": False
                    }
                    print(f"    ✗ {test_file} error: {e}")
        
        return results
    
    def generate_test_report(self) -> Dict[str, Any]:
        """Generate comprehensive test report"""
        print("Generating test report...")
        
        # Collect all test results
        unit_results = self.run_unit_tests()
        integration_results = self.run_integration_tests()
        benchmark_results = self.run_benchmark_tests()
        
        # Calculate summary statistics
        total_tests = len(unit_results) + len(integration_results) + len(benchmark_results)
        passed_tests = sum(1 for r in unit_results.values() if r["success"]) + \
                      sum(1 for r in integration_results.values() if r["success"]) + \
                      sum(1 for r in benchmark_results.values() if r["success"])
        failed_tests = total_tests - passed_tests
        
        # Generate report
        report = {
            "timestamp": time.time(),
            "summary": {
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "failed_tests": failed_tests,
                "success_rate": (passed_tests / total_tests * 100) if total_tests > 0 else 0
            },
            "unit_tests": unit_results,
            "integration_tests": integration_results,
            "benchmark_tests": benchmark_results,
            "device_info": self._get_device_info(),
            "recommendations": self._generate_recommendations(unit_results, integration_results, benchmark_results)
        }
        
        # Save report
        report_file = self.project_root / "test_data" / "test_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Test report saved to {report_file}")
        
        return report
    
    def _get_device_info(self) -> Dict[str, Any]:
        """Get device information"""
        import torch
        
        device_info = {
            "cuda_available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "current_device": str(torch.cuda.current_device()) if torch.cuda.is_available() else None
        }
        
        if torch.cuda.is_available():
            try:
                device_info["device_name"] = torch.cuda.get_device_name(0)
                device_info["is_amd"] = "AMD" in device_info["device_name"] or "Radeon" in device_info["device_name"]
            except:
                device_info["device_name"] = "Unknown"
                device_info["is_amd"] = False
        
        return device_info
    
    def _generate_recommendations(self, unit_results: Dict, integration_results: Dict, benchmark_results: Dict) -> List[str]:
        """Generate recommendations based on test results"""
        recommendations = []
        
        # Check for failed tests
        failed_unit = [name for name, result in unit_results.items() if not result["success"]]
        failed_integration = [name for name, result in integration_results.items() if not result["success"]]
        failed_benchmark = [name for name, result in benchmark_results.items() if not result["success"]]
        
        if failed_unit:
            recommendations.append(f"Fix {len(failed_unit)} failed unit tests: {', '.join(failed_unit)}")
        
        if failed_integration:
            recommendations.append(f"Fix {len(failed_integration)} failed integration tests: {', '.join(failed_integration)}")
        
        if failed_benchmark:
            recommendations.append(f"Fix {len(failed_benchmark)} failed benchmark tests: {', '.join(failed_benchmark)}")
        
        # Check for performance issues
        if not failed_benchmark and benchmark_results:
            recommendations.append("All benchmark tests passed - performance looks good")
        
        # General recommendations
        recommendations.extend([
            "Run tests regularly to catch regressions",
            "Add more test cases for edge conditions",
            "Monitor performance benchmarks over time",
            "Update test data when models change"
        ])
        
        return recommendations
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all tests and generate report"""
        print("=" * 60)
        print("ROCM Ninodes Test Suite")
        print("=" * 60)
        
        start_time = time.time()
        
        # Generate test report
        report = self.generate_test_report()
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Print summary
        print("\n" + "=" * 60)
        print("TEST SUMMARY")
        print("=" * 60)
        print(f"Total tests: {report['summary']['total_tests']}")
        print(f"Passed: {report['summary']['passed_tests']}")
        print(f"Failed: {report['summary']['failed_tests']}")
        print(f"Success rate: {report['summary']['success_rate']:.1f}%")
        print(f"Total time: {total_time:.2f}s")
        
        if report['summary']['failed_tests'] > 0:
            print("\nFAILED TESTS:")
            for test_type, results in [("Unit", report['unit_tests']), 
                                     ("Integration", report['integration_tests']), 
                                     ("Benchmark", report['benchmark_tests'])]:
                for test_name, result in results.items():
                    if not result['success']:
                        print(f"  {test_type}: {test_name}")
        
        print("\nRECOMMENDATIONS:")
        for rec in report['recommendations']:
            print(f"  • {rec}")
        
        print("=" * 60)
        
        return report


def main():
    """Main entry point"""
    runner = TestRunner()
    
    # Check if specific test type was requested
    if len(sys.argv) > 1:
        test_type = sys.argv[1].lower()
        
        if test_type == "unit":
            runner.run_unit_tests()
        elif test_type == "integration":
            runner.run_integration_tests()
        elif test_type == "benchmark":
            runner.run_benchmark_tests()
        else:
            print(f"Unknown test type: {test_type}")
            print("Available types: unit, integration, benchmark")
            sys.exit(1)
    else:
        # Run all tests
        runner.run_all_tests()


if __name__ == "__main__":
    main()

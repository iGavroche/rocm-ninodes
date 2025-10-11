#!/usr/bin/env python3
"""
Analyze baseline performance data for ROCMOptimizedVAEDecode
"""
import json
import os
import statistics
from pathlib import Path

def analyze_baseline_performance():
    """Analyze the baseline performance of ROCMOptimizedVAEDecode"""
    benchmark_dir = Path("test_data/benchmarks")
    
    if not benchmark_dir.exists():
        print("No benchmark data found!")
        return
    
    # Collect all execution times
    execution_times = []
    memory_usage = []
    
    for benchmark_file in benchmark_dir.glob("ROCMOptimizedVAEDecode_*.json"):
        try:
            with open(benchmark_file, 'r') as f:
                data = json.load(f)
                execution_times.append(data.get('execution_time', 0))
                if 'memory_usage' in data:
                    memory_usage.append(data['memory_usage'])
        except Exception as e:
            print(f"Error reading {benchmark_file}: {e}")
    
    if not execution_times:
        print("No execution time data found!")
        return
    
    # Calculate statistics
    print("=== ROCMOptimizedVAEDecode Baseline Performance Analysis ===")
    print(f"Total samples: {len(execution_times)}")
    print(f"Average execution time: {statistics.mean(execution_times):.6f}s")
    print(f"Median execution time: {statistics.median(execution_times):.6f}s")
    print(f"Min execution time: {min(execution_times):.6f}s")
    print(f"Max execution time: {max(execution_times):.6f}s")
    print(f"Standard deviation: {statistics.stdev(execution_times):.6f}s")
    
    # Performance distribution
    fast_executions = [t for t in execution_times if t < 0.01]
    medium_executions = [t for t in execution_times if 0.01 <= t < 0.05]
    slow_executions = [t for t in execution_times if t >= 0.05]
    
    print(f"\nPerformance Distribution:")
    print(f"Fast (< 0.01s): {len(fast_executions)} ({len(fast_executions)/len(execution_times)*100:.1f}%)")
    print(f"Medium (0.01-0.05s): {len(medium_executions)} ({len(medium_executions)/len(execution_times)*100:.1f}%)")
    print(f"Slow (>= 0.05s): {len(slow_executions)} ({len(slow_executions)/len(execution_times)*100:.1f}%)")
    
    # Save baseline metrics
    baseline_metrics = {
        "node_name": "ROCMOptimizedVAEDecode",
        "total_samples": len(execution_times),
        "average_time": statistics.mean(execution_times),
        "median_time": statistics.median(execution_times),
        "min_time": min(execution_times),
        "max_time": max(execution_times),
        "std_dev": statistics.stdev(execution_times),
        "performance_distribution": {
            "fast": len(fast_executions),
            "medium": len(medium_executions),
            "slow": len(slow_executions)
        }
    }
    
    with open("test_data/optimization/baseline_metrics.json", "w") as f:
        json.dump(baseline_metrics, f, indent=2)
    
    print(f"\nBaseline metrics saved to test_data/optimization/baseline_metrics.json")
    
    return baseline_metrics

if __name__ == "__main__":
    analyze_baseline_performance()

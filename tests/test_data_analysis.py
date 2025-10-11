#!/usr/bin/env python3
"""
Isolated node testing using real captured data
Tests the core logic without ComfyUI dependencies
"""

import pytest
import torch
import numpy as np
import pickle
import os
import sys
import time
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

class TestDataAnalyzer:
    """Analyze captured data to understand node behavior"""
    
    def __init__(self, test_data_dir="test_data"):
        self.test_data_dir = test_data_dir
    
    def analyze_vae_performance(self):
        """Analyze VAE decode performance from captured data"""
        print("=== VAE Decode Performance Analysis ===")
        
        # Load input data
        inputs = self._load_files(f"{self.test_data_dir}/inputs", "ROCMOptimizedVAEDecode")
        outputs = self._load_files(f"{self.test_data_dir}/outputs", "ROCMOptimizedVAEDecode")
        benchmarks = self._load_files(f"{self.test_data_dir}/benchmarks", "ROCMOptimizedVAEDecode", json=True)
        
        print(f"Total inputs captured: {len(inputs)}")
        print(f"Total outputs captured: {len(outputs)}")
        print(f"Total benchmarks captured: {len(benchmarks)}")
        
        # Analyze input shapes
        input_shapes = {}
        for inp in inputs:
            shape = tuple(inp['inputs']['samples']['samples']['shape'])
            if shape not in input_shapes:
                input_shapes[shape] = 0
            input_shapes[shape] += 1
        
        print(f"\nInput shape distribution:")
        for shape, count in sorted(input_shapes.items()):
            print(f"  {shape}: {count} cases")
        
        # Analyze output shapes
        output_shapes = {}
        for out in outputs:
            shape = tuple(out['outputs'][0]['shape'])
            if shape not in output_shapes:
                output_shapes[shape] = 0
            output_shapes[shape] += 1
        
        print(f"\nOutput shape distribution:")
        for shape, count in sorted(output_shapes.items()):
            print(f"  {shape}: {count} cases")
        
        # Analyze execution times
        execution_times = [b['execution_time'] for b in benchmarks if b['execution_time'] > 0]
        if execution_times:
            print(f"\nExecution time analysis:")
            print(f"  Successful executions: {len(execution_times)}")
            print(f"  Average time: {np.mean(execution_times):.4f}s")
            print(f"  Min time: {np.min(execution_times):.4f}s")
            print(f"  Max time: {np.max(execution_times):.4f}s")
            print(f"  Std deviation: {np.std(execution_times):.4f}s")
        else:
            print(f"\nNo successful executions found!")
        
        # Analyze failure patterns
        failed_executions = len(benchmarks) - len(execution_times)
        if failed_executions > 0:
            print(f"\nFailure analysis:")
            print(f"  Failed executions: {failed_executions}")
            print(f"  Success rate: {len(execution_times)/len(benchmarks)*100:.1f}%")
        
        return {
            'input_shapes': input_shapes,
            'output_shapes': output_shapes,
            'execution_times': execution_times,
            'success_rate': len(execution_times)/len(benchmarks) if benchmarks else 0
        }
    
    def analyze_parameter_usage(self):
        """Analyze parameter usage patterns"""
        print("\n=== Parameter Usage Analysis ===")
        
        inputs = self._load_files(f"{self.test_data_dir}/inputs", "ROCMOptimizedVAEDecode")
        
        # Analyze tile sizes
        tile_sizes = {}
        overlap_values = {}
        precision_modes = {}
        optimization_settings = {}
        
        for inp in inputs:
            params = inp['inputs']
            
            # Tile size analysis
            tile_size = params.get('tile_size', 768)
            if tile_size not in tile_sizes:
                tile_sizes[tile_size] = 0
            tile_sizes[tile_size] += 1
            
            # Overlap analysis
            overlap = params.get('overlap', 96)
            if overlap not in overlap_values:
                overlap_values[overlap] = 0
            overlap_values[overlap] += 1
            
            # Precision mode analysis
            precision = params.get('precision_mode', 'auto')
            if precision not in precision_modes:
                precision_modes[precision] = 0
            precision_modes[precision] += 1
            
            # Optimization settings
            rocm_opt = params.get('use_rocm_optimizations', True)
            batch_opt = params.get('batch_optimization', True)
            memory_opt = params.get('memory_optimization_enabled', True)
            
            opt_key = f"rocm:{rocm_opt}_batch:{batch_opt}_memory:{memory_opt}"
            if opt_key not in optimization_settings:
                optimization_settings[opt_key] = 0
            optimization_settings[opt_key] += 1
        
        print(f"Tile size usage:")
        for size, count in sorted(tile_sizes.items()):
            print(f"  {size}: {count} cases")
        
        print(f"\nOverlap usage:")
        for overlap, count in sorted(overlap_values.items()):
            print(f"  {overlap}: {count} cases")
        
        print(f"\nPrecision mode usage:")
        for mode, count in sorted(precision_modes.items()):
            print(f"  {mode}: {count} cases")
        
        print(f"\nOptimization settings:")
        for setting, count in sorted(optimization_settings.items()):
            print(f"  {setting}: {count} cases")
        
        return {
            'tile_sizes': tile_sizes,
            'overlap_values': overlap_values,
            'precision_modes': precision_modes,
            'optimization_settings': optimization_settings
        }
    
    def identify_issues(self):
        """Identify issues from the captured data"""
        print("\n=== Issue Identification ===")
        
        inputs = self._load_files(f"{self.test_data_dir}/inputs", "ROCMOptimizedVAEDecode")
        outputs = self._load_files(f"{self.test_data_dir}/outputs", "ROCMOptimizedVAEDecode")
        benchmarks = self._load_files(f"{self.test_data_dir}/benchmarks", "ROCMOptimizedVAEDecode", json=True)
        
        issues = []
        
        # Check success rate
        execution_times = [b['execution_time'] for b in benchmarks if b['execution_time'] > 0]
        success_rate = len(execution_times) / len(benchmarks) if benchmarks else 0
        
        if success_rate < 0.5:
            issues.append(f"Low success rate: {success_rate:.1%}")
        
        # Check output size issues
        for out in outputs:
            shape = tuple(out['outputs'][0]['shape'])
            if shape[2] < 10 or shape[3] < 10:  # Very small outputs
                issues.append(f"Suspiciously small output: {shape}")
        
        # Check execution time issues
        if execution_times:
            avg_time = np.mean(execution_times)
            if avg_time < 0.001:
                issues.append(f"Very short execution times suggest early failures: {avg_time:.4f}s")
        
        # Check input/output ratio
        if len(outputs) < len(inputs) * 0.1:
            issues.append(f"Very few outputs compared to inputs: {len(outputs)}/{len(inputs)}")
        
        if issues:
            print("Issues identified:")
            for issue in issues:
                print(f"  - {issue}")
        else:
            print("No major issues identified")
        
        return issues
    
    def _load_files(self, directory, node_name, json=False):
        """Load files from directory matching node name"""
        files = []
        if not os.path.exists(directory):
            return files
            
        for filename in os.listdir(directory):
            if filename.startswith(node_name) and filename.endswith(('.pkl', '.json')):
                filepath = os.path.join(directory, filename)
                try:
                    if json:
                        with open(filepath, 'r') as f:
                            files.append(json.load(f))
                    else:
                        with open(filepath, 'rb') as f:
                            files.append(pickle.load(f))
                except Exception as e:
                    print(f"Warning: Could not load {filepath}: {e}")
        return files


class TestROCMNodeAnalysis:
    """Test ROCM nodes using real captured data analysis"""
    
    def test_analyze_captured_data(self):
        """Analyze all captured data to understand node behavior"""
        analyzer = TestDataAnalyzer()
        
        # Analyze performance
        perf_data = analyzer.analyze_vae_performance()
        
        # Analyze parameters
        param_data = analyzer.analyze_parameter_usage()
        
        # Identify issues
        issues = analyzer.identify_issues()
        
        # Store analysis results
        analysis_results = {
            'performance': perf_data,
            'parameters': param_data,
            'issues': issues,
            'timestamp': int(time.time())
        }
        
        # Save analysis
        with open('test_data/analysis_results.json', 'w') as f:
            json.dump(analysis_results, f, indent=2)
        
        print(f"\nAnalysis saved to test_data/analysis_results.json")
        
        # Assertions based on analysis
        assert perf_data['success_rate'] > 0, "No successful executions found"
        
        if issues:
            print(f"\nFound {len(issues)} issues that need attention")
            # Don't fail the test, just report issues
            for issue in issues:
                print(f"  - {issue}")
    
    def test_create_optimization_recommendations(self):
        """Create optimization recommendations based on analysis"""
        analyzer = TestDataAnalyzer()
        
        # Load analysis if available
        analysis_file = 'test_data/analysis_results.json'
        if os.path.exists(analysis_file):
            with open(analysis_file, 'r') as f:
                analysis = json.load(f)
        else:
            # Run analysis first
            perf_data = analyzer.analyze_vae_performance()
            param_data = analyzer.analyze_parameter_usage()
            issues = analyzer.identify_issues()
            analysis = {
                'performance': perf_data,
                'parameters': param_data,
                'issues': issues
            }
        
        recommendations = []
        
        # Performance recommendations
        if analysis['performance']['success_rate'] < 0.8:
            recommendations.append({
                'type': 'performance',
                'priority': 'high',
                'issue': 'Low success rate',
                'recommendation': 'Investigate and fix node failures',
                'details': f"Success rate: {analysis['performance']['success_rate']:.1%}"
            })
        
        # Parameter optimization recommendations
        tile_sizes = analysis['parameters']['tile_sizes']
        if len(tile_sizes) > 1:
            most_common_tile = max(tile_sizes.items(), key=lambda x: x[1])
            recommendations.append({
                'type': 'optimization',
                'priority': 'medium',
                'issue': 'Multiple tile sizes used',
                'recommendation': f'Consider optimizing for most common tile size: {most_common_tile[0]}',
                'details': f"Usage distribution: {dict(tile_sizes)}"
            })
        
        # Memory optimization recommendations
        if any('memory' in issue.lower() for issue in analysis['issues']):
            recommendations.append({
                'type': 'memory',
                'priority': 'high',
                'issue': 'Memory-related issues detected',
                'recommendation': 'Implement better memory management',
                'details': 'Check memory allocation and deallocation patterns'
            })
        
        print(f"\n=== Optimization Recommendations ===")
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. [{rec['priority'].upper()}] {rec['type'].upper()}")
            print(f"   Issue: {rec['issue']}")
            print(f"   Recommendation: {rec['recommendation']}")
            print(f"   Details: {rec['details']}")
            print()
        
        # Save recommendations
        with open('test_data/optimization_recommendations.json', 'w') as f:
            json.dump(recommendations, f, indent=2)
        
        print(f"Recommendations saved to test_data/optimization_recommendations.json")
        
        return recommendations


if __name__ == "__main__":
    import time
    # Run tests
    pytest.main([__file__, "-v", "-s"])

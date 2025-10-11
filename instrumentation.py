"""
Instrumentation module for ROCM Ninodes testing and optimization
Captures inputs, outputs, and performance metrics for all nodes
"""

import os
import pickle
import time
import json
import torch
import numpy as np
from typing import Any, Dict, List, Tuple, Optional
from datetime import datetime
import logging

class NodeInstrumentation:
    """
    Instrumentation class for capturing node data and performance metrics
    """
    
    def __init__(self, test_data_dir: str = "test_data"):
        self.test_data_dir = test_data_dir
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.captured_data = {}
        self.performance_metrics = {}
        
        # Create directories
        os.makedirs(f"{test_data_dir}/inputs", exist_ok=True)
        os.makedirs(f"{test_data_dir}/outputs", exist_ok=True)
        os.makedirs(f"{test_data_dir}/benchmarks", exist_ok=True)
        os.makedirs(f"{test_data_dir}/optimization", exist_ok=True)
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def capture_inputs(self, node_name: str, inputs: Dict[str, Any], 
                      node_id: str = None) -> str:
        """
        Capture and save node inputs for testing
        """
        timestamp = int(time.time() * 1000)
        node_id = node_id or f"{node_name}_{timestamp}"
        
        # Prepare input data for serialization
        serializable_inputs = self._prepare_for_serialization(inputs)
        
        # Save to file
        filename = f"{self.test_data_dir}/inputs/{node_name}_{node_id}_{timestamp}.pkl"
        with open(filename, 'wb') as f:
            pickle.dump({
                'node_name': node_name,
                'node_id': node_id,
                'timestamp': timestamp,
                'inputs': serializable_inputs,
                'session_id': self.session_id
            }, f)
        
        self.logger.info(f"Captured inputs for {node_name} -> {filename}")
        return filename
    
    def capture_outputs(self, node_name: str, outputs: Tuple[Any, ...], 
                       node_id: str = None) -> str:
        """
        Capture and save node outputs for testing
        """
        timestamp = int(time.time() * 1000)
        node_id = node_id or f"{node_name}_{timestamp}"
        
        # Prepare output data for serialization
        serializable_outputs = self._prepare_for_serialization(outputs)
        
        # Save to file
        filename = f"{self.test_data_dir}/outputs/{node_name}_{node_id}_{timestamp}.pkl"
        with open(filename, 'wb') as f:
            pickle.dump({
                'node_name': node_name,
                'node_id': node_id,
                'timestamp': timestamp,
                'outputs': serializable_outputs,
                'session_id': self.session_id
            }, f)
        
        self.logger.info(f"Captured outputs for {node_name} -> {filename}")
        return filename
    
    def capture_performance(self, node_name: str, start_time: float, 
                           end_time: float, memory_used: float = None,
                           node_id: str = None) -> str:
        """
        Capture performance metrics for a node
        """
        timestamp = int(time.time() * 1000)
        node_id = node_id or f"{node_name}_{timestamp}"
        
        metrics = {
            'node_name': node_name,
            'node_id': node_id,
            'timestamp': timestamp,
            'execution_time': end_time - start_time,
            'memory_used': memory_used,
            'session_id': self.session_id,
            'start_time': start_time,
            'end_time': end_time
        }
        
        # Save to file
        filename = f"{self.test_data_dir}/benchmarks/{node_name}_{node_id}_{timestamp}.json"
        with open(filename, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        self.logger.info(f"Captured performance for {node_name}: {metrics['execution_time']:.3f}s")
        return filename
    
    def _prepare_for_serialization(self, data: Any) -> Any:
        """
        Prepare data for serialization by converting tensors to numpy arrays
        """
        if isinstance(data, dict):
            return {k: self._prepare_for_serialization(v) for k, v in data.items()}
        elif isinstance(data, (list, tuple)):
            return type(data)(self._prepare_for_serialization(item) for item in data)
        elif isinstance(data, torch.Tensor):
            return {
                'tensor_data': data.detach().cpu().numpy(),
                'shape': list(data.shape),
                'dtype': str(data.dtype),
                'device': str(data.device)
            }
        elif hasattr(data, '__dict__'):
            # For objects with __dict__, try to serialize their attributes
            try:
                return {k: self._prepare_for_serialization(v) for k, v in data.__dict__.items()}
            except:
                return str(data)
        else:
            return data
    
    def load_test_data(self, node_name: str, data_type: str = "inputs") -> List[Dict]:
        """
        Load all test data for a specific node
        """
        data_dir = f"{self.test_data_dir}/{data_type}"
        test_files = []
        
        for filename in os.listdir(data_dir):
            if filename.startswith(node_name) and filename.endswith('.pkl'):
                filepath = os.path.join(data_dir, filename)
                with open(filepath, 'rb') as f:
                    test_files.append(pickle.load(f))
        
        return sorted(test_files, key=lambda x: x['timestamp'])
    
    def load_benchmark_data(self, node_name: str) -> List[Dict]:
        """
        Load benchmark data for a specific node
        """
        benchmark_dir = f"{self.test_data_dir}/benchmarks"
        benchmark_files = []
        
        for filename in os.listdir(benchmark_dir):
            if filename.startswith(node_name) and filename.endswith('.json'):
                filepath = os.path.join(benchmark_dir, filename)
                with open(filepath, 'r') as f:
                    benchmark_files.append(json.load(f))
        
        return sorted(benchmark_files, key=lambda x: x['timestamp'])
    
    def generate_test_report(self, node_name: str) -> Dict:
        """
        Generate a comprehensive test report for a node
        """
        inputs = self.load_test_data(node_name, "inputs")
        outputs = self.load_test_data(node_name, "outputs")
        benchmarks = self.load_benchmark_data(node_name)
        
        report = {
            'node_name': node_name,
            'session_id': self.session_id,
            'total_tests': len(inputs),
            'inputs_captured': len(inputs),
            'outputs_captured': len(outputs),
            'benchmarks_captured': len(benchmarks),
            'avg_execution_time': np.mean([b['execution_time'] for b in benchmarks]) if benchmarks else 0,
            'min_execution_time': np.min([b['execution_time'] for b in benchmarks]) if benchmarks else 0,
            'max_execution_time': np.max([b['execution_time'] for b in benchmarks]) if benchmarks else 0,
            'avg_memory_used': np.mean([b['memory_used'] for b in benchmarks if b['memory_used']]) if benchmarks else 0,
            'test_data_summary': {
                'input_shapes': [self._extract_shapes(inp['inputs']) for inp in inputs],
                'output_shapes': [self._extract_shapes(out['outputs']) for out in outputs]
            }
        }
        
        return report
    
    def _extract_shapes(self, data: Any) -> Dict:
        """
        Extract shape information from data
        """
        shapes = {}
        if isinstance(data, dict):
            for k, v in data.items():
                if isinstance(v, dict) and 'shape' in v:
                    shapes[k] = v['shape']
                elif isinstance(v, (list, tuple)):
                    shapes[k] = [len(v)]
        return shapes


# Global instrumentation instance
instrumentation = NodeInstrumentation()


def instrument_node(node_class):
    """
    Decorator to instrument a node class
    """
    # Get the main function name (usually 'decode', 'sample', etc.)
    function_name = None
    for attr_name in ['decode', 'sample', 'load', 'process']:
        if hasattr(node_class, attr_name):
            function_name = attr_name
            break
    
    if not function_name:
        print(f"Warning: Could not find main function for {node_class.__name__}")
        return node_class
    
    original_function = getattr(node_class, function_name)
    
    def instrumented_function(self, *args, **kwargs):
        # Capture inputs
        inputs = {**kwargs}
        input_filename = instrumentation.capture_inputs(
            node_class.__name__, inputs
        )
        
        # Start timing
        start_time = time.time()
        
        # Get memory before execution
        memory_before = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        
        try:
            # Execute original function
            result = original_function(self, *args, **kwargs)
            
            # End timing
            end_time = time.time()
            memory_after = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
            memory_used = (memory_after - memory_before) / 1024**3  # Convert to GB
            
            # Capture outputs
            output_filename = instrumentation.capture_outputs(
                node_class.__name__, result
            )
            
            # Capture performance
            performance_filename = instrumentation.capture_performance(
                node_class.__name__, start_time, end_time, memory_used
            )
            
            return result
            
        except Exception as e:
            # Still capture performance data even on error
            end_time = time.time()
            memory_after = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
            memory_used = (memory_after - memory_before) / 1024**3
            
            instrumentation.capture_performance(
                node_class.__name__, start_time, end_time, memory_used
            )
            
            raise e
    
    # Replace the function
    setattr(node_class, function_name, instrumented_function)
    return node_class


def create_test_fixtures():
    """
    Create test fixtures from captured data
    """
    fixtures_dir = "tests/fixtures"
    os.makedirs(fixtures_dir, exist_ok=True)
    
    # Get all available nodes
    from nodes import NODE_CLASS_MAPPINGS
    
    for node_name, node_class in NODE_CLASS_MAPPINGS.items():
        # Load test data
        inputs = instrumentation.load_test_data(node_name, "inputs")
        outputs = instrumentation.load_test_data(node_name, "outputs")
        
        if inputs and outputs:
            # Create fixture file
            fixture_data = {
                'node_name': node_name,
                'test_cases': []
            }
            
            # Match inputs with outputs by timestamp
            input_dict = {inp['timestamp']: inp for inp in inputs}
            output_dict = {out['timestamp']: out for out in outputs}
            
            for timestamp in sorted(input_dict.keys()):
                if timestamp in output_dict:
                    fixture_data['test_cases'].append({
                        'timestamp': timestamp,
                        'inputs': input_dict[timestamp]['inputs'],
                        'outputs': output_dict[timestamp]['outputs']
                    })
            
            # Save fixture
            fixture_file = f"{fixtures_dir}/{node_name}_fixtures.json"
            with open(fixture_file, 'w') as f:
                json.dump(fixture_data, f, indent=2)
            
            print(f"Created fixture for {node_name}: {len(fixture_data['test_cases'])} test cases")


if __name__ == "__main__":
    # Create test fixtures from existing data
    create_test_fixtures()

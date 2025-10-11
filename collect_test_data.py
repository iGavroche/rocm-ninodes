#!/usr/bin/env python3
"""
Data collection script for ROCM Ninodes
Runs workflows to collect test data for optimization
"""

import sys
import os
import json
import time
import torch
from pathlib import Path
from typing import Dict, Any, List

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from instrumentation import NodeInstrumentation, instrument_node
from nodes import NODE_CLASS_MAPPINGS


class TestDataCollector:
    """Collects test data by running workflows"""
    
    def __init__(self):
        self.instrumentation = NodeInstrumentation()
        self.test_data_dir = Path("test_data")
        self.workflows_dir = Path(".")
        self.collected_data = {}
        
        # Create directories
        self.test_data_dir.mkdir(exist_ok=True)
        (self.test_data_dir / "inputs").mkdir(exist_ok=True)
        (self.test_data_dir / "outputs").mkdir(exist_ok=True)
        (self.test_data_dir / "benchmarks").mkdir(exist_ok=True)
        (self.test_data_dir / "optimization").mkdir(exist_ok=True)
    
    def instrument_all_nodes(self):
        """Instrument all ROCM nodes for data collection"""
        print("Instrumenting nodes for data collection...")
        
        for node_name, node_class in NODE_CLASS_MAPPINGS.items():
            try:
                instrument_node(node_class)
                print(f"  ✓ Instrumented {node_name}")
            except Exception as e:
                print(f"  ✗ Failed to instrument {node_name}: {e}")
    
    def load_workflow(self, workflow_file: str) -> Dict[str, Any]:
        """Load workflow from JSON file"""
        workflow_path = self.workflows_dir / workflow_file
        if not workflow_path.exists():
            raise FileNotFoundError(f"Workflow file not found: {workflow_file}")
        
        with open(workflow_path, 'r') as f:
            return json.load(f)
    
    def create_mock_models(self) -> Dict[str, Any]:
        """Create mock models for testing"""
        class MockModel:
            def __init__(self):
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                self.model_dtype = lambda: torch.float32
        
        class MockVAE:
            def __init__(self):
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                self.output_device = self.device
                self.vae_dtype = torch.float32
                self.latent_channels = 4
                self.upscale_ratio = 8
                self.first_stage_model = MockFirstStageModel()
            
            def decode(self, samples):
                if len(samples.shape) == 5:  # Video
                    B, C, T, H, W = samples.shape
                    return (torch.randn(B, T, H*8, W*8, 3),)
                else:  # Image
                    B, C, H, W = samples.shape
                    return (torch.randn(B, H*8, W*8, 3),)
            
            def memory_used_decode(self, shape, dtype):
                return 100 * 1024 * 1024
            
            def spacial_compression_decode(self):
                return 8
        
        class MockCLIP:
            def __init__(self):
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        class MockFirstStageModel:
            def __init__(self):
                self.dtype = torch.float32
            
            def to(self, dtype):
                self.dtype = dtype
                return self
            
            def decode(self, samples):
                if len(samples.shape) == 5:  # Video
                    B, C, T, H, W = samples.shape
                    return torch.randn(B, T, H*8, W*8, 3)
                else:  # Image
                    B, C, H, W = samples.shape
                    return torch.randn(B, H*8, W*8, 3)
        
        return {
            "model": MockModel(),
            "vae": MockVAE(),
            "clip": MockCLIP()
        }
    
    def collect_flux_workflow_data(self):
        """Collect data from Flux workflow"""
        print("Collecting Flux workflow data...")
        
        try:
            # Load Flux workflow
            workflow = self.load_workflow("flux_dev_optimized.json")
            print(f"  Loaded Flux workflow with {len(workflow.get('nodes', []))} nodes")
            
            # Create mock models
            models = self.create_mock_models()
            
            # Test checkpoint loader
            from nodes import ROCMOptimizedCheckpointLoader
            loader = ROCMOptimizedCheckpointLoader()
            
            with self._mock_checkpoint_loading():
                model, clip, vae = loader.load_checkpoint(
                    ckpt_name="flux_dev.safetensors",
                    lazy_loading=True,
                    optimize_for_flux=True,
                    precision_mode="auto"
                )
            
            # Test VAE decode with different resolutions
            from nodes import ROCMOptimizedVAEDecode
            vae_decoder = ROCMOptimizedVAEDecode()
            
            resolutions = [(256, 256), (512, 512), (1024, 1024)]
            for w, h in resolutions:
                latent = {
                    "samples": torch.randn(1, 4, h//8, w//8)
                }
                
                result = vae_decoder.decode(
                    vae=vae,
                    samples=latent,
                    tile_size=768,
                    overlap=96,
                    use_rocm_optimizations=True,
                    precision_mode="auto",
                    batch_optimization=True,
                    flux_vae_optimization=True,
                    adaptive_tile_sizing=True,
                    fp8_latent_optimization=True
                )
                
                print(f"    Tested VAE decode {w}x{h}")
            
            # Test KSampler
            from nodes import ROCMOptimizedKSampler
            sampler = ROCMOptimizedKSampler()
            
            with self._mock_sampling():
                latent_image = {
                    "samples": torch.randn(1, 4, 128, 128),
                    "batch_index": [0]
                }
                
                conditioning = [
                    [
                        "A portrait view of a beautiful blond girl with long hair and blue eyes looking in love",
                        {
                            "pooled_output": torch.randn(1, 1280),
                            "embeds": torch.randn(1, 77, 1280)
                        }
                    ]
                ]
                
                result = sampler.sample(
                    model=model,
                    seed=813485113100655,
                    steps=20,
                    cfg=8,
                    sampler_name="euler",
                    scheduler="simple",
                    positive=conditioning,
                    negative=conditioning,
                    latent_image=latent_image,
                    denoise=1.0,
                    use_rocm_optimizations=True,
                    precision_mode="auto",
                    memory_optimization=True,
                    attention_optimization=True
                )
                
                print("    Tested KSampler")
            
            print("  ✓ Flux workflow data collection completed")
            
        except Exception as e:
            print(f"  ✗ Flux workflow data collection failed: {e}")
    
    def collect_wan_workflow_data(self):
        """Collect data from WAN workflow"""
        print("Collecting WAN workflow data...")
        
        try:
            # Load WAN workflow
            workflow = self.load_workflow("example_workflow_wan_video.json")
            print(f"  Loaded WAN workflow with {len(workflow.get('nodes', []))} nodes")
            
            # Create mock models
            models = self.create_mock_models()
            
            # Test checkpoint loader
            from nodes import ROCMOptimizedCheckpointLoader
            loader = ROCMOptimizedCheckpointLoader()
            
            with self._mock_checkpoint_loading():
                model, clip, vae = loader.load_checkpoint(
                    ckpt_name="wan_model.safetensors",
                    lazy_loading=True,
                    optimize_for_flux=False,
                    precision_mode="auto"
                )
            
            # Test VAE decode with video data
            from nodes import ROCMOptimizedVAEDecode
            vae_decoder = ROCMOptimizedVAEDecode()
            
            # Test different video configurations
            video_configs = [
                (320, 320, 17),  # Small video
                (432, 432, 29),  # Medium video
                (640, 640, 8),   # Large video
            ]
            
            for w, h, frames in video_configs:
                video_latent = {
                    "samples": torch.randn(1, 4, frames, h//8, w//8)
                }
                
                result = vae_decoder.decode(
                    vae=vae,
                    samples=video_latent,
                    tile_size=768,
                    overlap=96,
                    use_rocm_optimizations=True,
                    precision_mode="auto",
                    batch_optimization=True,
                    video_chunk_size=4,
                    memory_optimization_enabled=True
                )
                
                print(f"    Tested VAE decode video {w}x{h}x{frames}")
            
            # Test advanced KSampler
            from nodes import ROCMOptimizedKSamplerAdvanced
            sampler = ROCMOptimizedKSamplerAdvanced()
            
            with self._mock_sampling():
                latent_image = {
                    "samples": torch.randn(1, 4, 64, 64),
                    "batch_index": [0]
                }
                
                conditioning = [
                    [
                        "The girl blows the candles then smiles and winks",
                        {
                            "pooled_output": torch.randn(1, 1280),
                            "embeds": torch.randn(1, 77, 1280)
                        }
                    ]
                ]
                
                result = sampler.sample(
                    model=model,
                    add_noise="enable",
                    noise_seed=549535961820368,
                    steps=4,
                    cfg=1,
                    sampler_name="euler",
                    scheduler="simple",
                    positive=conditioning,
                    negative=conditioning,
                    latent_image=latent_image,
                    start_at_step=0,
                    end_at_step=2,
                    return_with_leftover_noise="enable",
                    use_rocm_optimizations=True,
                    precision_mode="auto",
                    memory_optimization=True
                )
                
                print("    Tested Advanced KSampler")
            
            print("  ✓ WAN workflow data collection completed")
            
        except Exception as e:
            print(f"  ✗ WAN workflow data collection failed: {e}")
    
    def _mock_checkpoint_loading(self):
        """Context manager for mocking checkpoint loading"""
        from unittest.mock import patch
        
        return patch.multiple(
            'nodes',
            **{
                'comfy.sd.load_checkpoint_guess_config': self._mock_load_checkpoint,
                'folder_paths.get_full_path_or_raise': self._mock_get_path
            }
        )
    
    def _mock_sampling(self):
        """Context manager for mocking sampling"""
        from unittest.mock import patch
        
        def mock_sample(*args, **kwargs):
            # Return a random tensor with appropriate shape
            if 'latent_image' in kwargs:
                latent_shape = kwargs['latent_image']['samples'].shape
                return torch.randn(latent_shape)
            return torch.randn(1, 4, 64, 64)
        
        return patch('comfy.sample.sample', mock_sample)
    
    def _mock_load_checkpoint(self, *args, **kwargs):
        """Mock checkpoint loading"""
        models = self.create_mock_models()
        return (models["model"], models["clip"], models["vae"])
    
    def _mock_get_path(self, folder, filename):
        """Mock path resolution"""
        return f"/fake/path/{filename}"
    
    def collect_performance_data(self):
        """Collect performance data for optimization"""
        print("Collecting performance data...")
        
        try:
            # Test different configurations
            configs = [
                {"precision": "fp32", "tile_size": 512, "optimizations": True},
                {"precision": "fp32", "tile_size": 768, "optimizations": True},
                {"precision": "fp32", "tile_size": 1024, "optimizations": True},
                {"precision": "fp16", "tile_size": 768, "optimizations": True},
                {"precision": "bf16", "tile_size": 768, "optimizations": True},
                {"precision": "fp32", "tile_size": 768, "optimizations": False},
            ]
            
            from nodes import ROCMOptimizedVAEDecode
            vae_decoder = ROCMOptimizedVAEDecode()
            models = self.create_mock_models()
            
            for i, config in enumerate(configs):
                print(f"  Testing configuration {i+1}/{len(configs)}: {config}")
                
                latent = {
                    "samples": torch.randn(1, 4, 64, 64)
                }
                
                start_time = time.time()
                
                result = vae_decoder.decode(
                    vae=models["vae"],
                    samples=latent,
                    tile_size=config["tile_size"],
                    overlap=96,
                    use_rocm_optimizations=config["optimizations"],
                    precision_mode=config["precision"],
                    batch_optimization=True
                )
                
                end_time = time.time()
                execution_time = end_time - start_time
                
                # Save performance data
                perf_data = {
                    "config": config,
                    "execution_time": execution_time,
                    "timestamp": time.time(),
                    "device": str(torch.device("cuda" if torch.cuda.is_available() else "cpu")),
                    "is_amd": self._is_amd_gpu()
                }
                
                perf_file = self.test_data_dir / "optimization" / f"performance_config_{i}.json"
                with open(perf_file, 'w') as f:
                    json.dump(perf_data, f, indent=2)
                
                print(f"    Execution time: {execution_time:.3f}s")
            
            print("  ✓ Performance data collection completed")
            
        except Exception as e:
            print(f"  ✗ Performance data collection failed: {e}")
    
    def _is_amd_gpu(self) -> bool:
        """Check if running on AMD GPU"""
        if torch.cuda.is_available():
            try:
                device_name = torch.cuda.get_device_name(0)
                return "AMD" in device_name or "Radeon" in device_name
            except:
                return False
        return False
    
    def generate_data_summary(self):
        """Generate summary of collected data"""
        print("Generating data summary...")
        
        summary = {
            "timestamp": time.time(),
            "device_info": {
                "cuda_available": torch.cuda.is_available(),
                "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                "is_amd": self._is_amd_gpu()
            },
            "collected_data": {
                "inputs": len(list((self.test_data_dir / "inputs").glob("*.pkl"))),
                "outputs": len(list((self.test_data_dir / "outputs").glob("*.pkl"))),
                "benchmarks": len(list((self.test_data_dir / "benchmarks").glob("*.json"))),
                "optimization": len(list((self.test_data_dir / "optimization").glob("*.json")))
            },
            "workflows_tested": [
                "flux_dev_optimized.json",
                "example_workflow_wan_video.json"
            ]
        }
        
        # Save summary
        summary_file = self.test_data_dir / "data_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"  Data summary saved to {summary_file}")
        print(f"  Inputs: {summary['collected_data']['inputs']}")
        print(f"  Outputs: {summary['collected_data']['outputs']}")
        print(f"  Benchmarks: {summary['collected_data']['benchmarks']}")
        print(f"  Optimization data: {summary['collected_data']['optimization']}")
    
    def run_data_collection(self):
        """Run complete data collection process"""
        print("=" * 60)
        print("ROCM Ninodes Data Collection")
        print("=" * 60)
        
        start_time = time.time()
        
        # Instrument nodes
        self.instrument_all_nodes()
        
        # Collect workflow data
        self.collect_flux_workflow_data()
        self.collect_wan_workflow_data()
        
        # Collect performance data
        self.collect_performance_data()
        
        # Generate summary
        self.generate_data_summary()
        
        end_time = time.time()
        total_time = end_time - start_time
        
        print("=" * 60)
        print(f"Data collection completed in {total_time:.2f}s")
        print("=" * 60)


def main():
    """Main entry point"""
    collector = TestDataCollector()
    collector.run_data_collection()


if __name__ == "__main__":
    main()

"""
Installation script for RocM-Nino
This script ensures the plugin is properly installed and configured
"""

import os
import sys
import json
from pathlib import Path

def check_comfyui_installation():
    """Check if ComfyUI is properly installed"""
    comfyui_path = Path(__file__).parent.parent.parent
    main_py = comfyui_path / "main.py"
    
    if not main_py.exists():
        print("❌ ComfyUI not found. Please ensure this plugin is in the correct location:")
        print("   ComfyUI/custom_nodes/ComfyUI-ROCM-Optimized-VAE/")
        return False
    
    print("✅ ComfyUI installation found")
    return True

def check_dependencies():
    """Check if required dependencies are available"""
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__} found")
        
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            print(f"✅ CUDA available - GPU: {device_name}")
            
            # Check if it's AMD
            if "AMD" in device_name or "Radeon" in device_name:
                print("✅ AMD GPU detected - ROCM optimizations will be active")
            else:
                print("⚠️  Non-AMD GPU detected - some optimizations may not apply")
        else:
            print("⚠️  CUDA not available - ROCM optimizations require GPU")
            
    except ImportError:
        print("⚠️  PyTorch not found - please ensure ComfyUI is properly installed")
        print("   ROCM optimizations will be available once PyTorch is installed")
    
    return True

def create_web_directory():
    """Create web directory for any frontend assets"""
    web_dir = Path(__file__).parent / "web"
    web_dir.mkdir(exist_ok=True)
    
    # Create a simple info file
    info_file = web_dir / "rocm-nino-info.js"
    info_file.write_text("""
// RocM-Nino Plugin Information
window.rocmNinoInfo = {
    name: "RocM-Nino",
    version: "1.0.0",
    description: "ROCM Optimized Nodes for ComfyUI",
    author: "Nino",
    nodes: [
        "ROCMOptimizedVAEDecode",
        "ROCMOptimizedVAEDecodeTiled", 
        "ROCMOptimizedKSampler",
        "ROCMOptimizedKSamplerAdvanced",
        "ROCMVAEPerformanceMonitor",
        "ROCMSamplerPerformanceMonitor"
    ]
};
""")
    
    print("✅ Web directory created")

def main():
    """Main installation function"""
    print("RocM-Nino: ROCM Optimized Nodes for ComfyUI")
    print("=" * 50)
    
    # Check ComfyUI installation
    if not check_comfyui_installation():
        return False
    
    # Check dependencies
    if not check_dependencies():
        print("\n❌ Installation failed due to missing dependencies")
        return False
    
    # Create web directory
    create_web_directory()
    
    print("\n✅ RocM-Nino plugin installed successfully!")
    print("\nNext steps:")
    print("1. Restart ComfyUI")
    print("2. Look for 'ROCM VAE Decode' and 'ROCM KSampler' nodes")
    print("3. Use the Performance Monitor nodes to optimize your setup")
    print("\nFor more information, visit: https://github.com/nino/rocm-nino")
    
    return True

if __name__ == "__main__":
    main()

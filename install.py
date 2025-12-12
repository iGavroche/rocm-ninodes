"""
Installation script for RocM-Nino
This script ensures the plugin is properly installed and configured
"""

import os
import sys
import json
from pathlib import Path

def safe_print(text):
    """
    Print text with emoji fallbacks for Windows console compatibility.
    Uses ASCII alternatives when UTF-8 encoding is not available.
    """
    # Check if we're on Windows and stdout encoding might not support emojis
    if sys.platform == "win32":
        encoding = getattr(sys.stdout, 'encoding', None)
        if encoding and encoding.lower() in ('cp1252', 'cp850', 'cp437'):
            # Windows console encodings that don't support emojis - replace them
            text = text.replace("✅", "[OK]")
            text = text.replace("❌", "[ERROR]")
            text = text.replace("⚠️", "[WARNING]")
        elif encoding is None:
            # Unknown encoding - be safe and replace emojis
            text = text.replace("✅", "[OK]")
            text = text.replace("❌", "[ERROR]")
            text = text.replace("⚠️", "[WARNING]")
    
    # Print the text (either original or with ASCII fallbacks)
    try:
        print(text, flush=True)
    except UnicodeEncodeError:
        # Final fallback: replace emojis and try again
        text = text.replace("✅", "[OK]")
        text = text.replace("❌", "[ERROR]")
        text = text.replace("⚠️", "[WARNING]")
        print(text, flush=True)

def check_comfyui_installation():
    """Check if ComfyUI is properly installed"""
    comfyui_path = Path(__file__).parent.parent.parent
    main_py = comfyui_path / "main.py"
    
    if not main_py.exists():
        safe_print("❌ ComfyUI not found. Please ensure this plugin is in the correct location:")
        safe_print("   ComfyUI/custom_nodes/ComfyUI-ROCM-Optimized-VAE/")
        return False
    
    safe_print("✅ ComfyUI installation found")
    return True

def check_dependencies():
    """Check if required dependencies are available"""
    try:
        import torch
        safe_print(f"✅ PyTorch {torch.__version__} found")
        
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            safe_print(f"✅ CUDA available - GPU: {device_name}")
            
            # Check if it's AMD
            if "AMD" in device_name or "Radeon" in device_name:
                safe_print("✅ AMD GPU detected - ROCM optimizations will be active")
            else:
                safe_print("⚠️  Non-AMD GPU detected - some optimizations may not apply")
        else:
            safe_print("⚠️  CUDA not available - ROCM optimizations require GPU")
            
    except ImportError:
        safe_print("⚠️  PyTorch not found - please ensure ComfyUI is properly installed")
        safe_print("   ROCM optimizations will be available once PyTorch is installed")
    
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
    
    safe_print("✅ Web directory created")

def main():
    """Main installation function"""
    safe_print("RocM-Nino: ROCM Optimized Nodes for ComfyUI")
    safe_print("=" * 50)
    
    # Check ComfyUI installation
    if not check_comfyui_installation():
        return False
    
    # Check dependencies
    if not check_dependencies():
        safe_print("\n❌ Installation failed due to missing dependencies")
        return False
    
    # Create web directory
    create_web_directory()
    
    safe_print("\n✅ RocM-Nino plugin installed successfully!")
    safe_print("\nNext steps:")
    safe_print("1. Restart ComfyUI")
    safe_print("2. Look for 'ROCM VAE Decode' and 'ROCM KSampler' nodes")
    safe_print("3. Use the Performance Monitor nodes to optimize your setup")
    safe_print("\nFor more information, visit: https://github.com/nino/rocm-nino")
    
    return True

if __name__ == "__main__":
    main()

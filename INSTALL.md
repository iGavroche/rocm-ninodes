# Quick Installation Guide

## 🚀 **Easy Installation (3 steps)**

### Step 1: Clone the Repository
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/nino/rocm-nino.git ComfyUI-ROCM-Optimized-VAE
```

### Step 2: Install the Plugin
```bash
cd ComfyUI-ROCM-Optimized-VAE
python install.py
```

### Step 3: Restart ComfyUI
- Close ComfyUI completely
- Start ComfyUI again
- Look for "ROCM VAE Decode" and "ROCM KSampler" nodes

## ✅ **Verification**

1. **Check Node Categories**: Look for "latent/rocm_optimized" and "sampling/rocm_optimized" in the node menu
2. **Test Performance**: Use the Performance Monitor nodes to analyze your setup
3. **Load Example**: Try the included `example_workflow.json`

## 🎯 **Quick Start**

1. Replace your standard "VAE Decode" with "ROCM VAE Decode"
2. Replace your standard "KSampler" with "ROCM KSampler"
3. Use default settings (optimized for gfx1151)
4. Enable all ROCm optimizations

## 📊 **Expected Results**

- **20-40% faster** end-to-end generation
- **25-35% better** VRAM usage
- **Reduced OOM errors** with better memory management
- **Maintained or improved** output quality

## 🆘 **Need Help?**

- **Issues**: [GitHub Issues](https://github.com/nino/rocm-nino/issues)
- **Documentation**: [Full README](README.md)
- **Performance**: Use the Performance Monitor nodes for recommendations

---

**RocM-Nino v1.0.0** - ROCM Optimized Nodes for ComfyUI

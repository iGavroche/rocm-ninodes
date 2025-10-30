# Troubleshooting Noise Output

## Problem: Getting Noise Instead of Images When Changing Dimensions

### Symptoms
- You change the desired output dimensions
- The sampler runs without errors
- But the output is pure noise instead of a proper image

### Root Cause
The latent tensor dimensions don't match your desired output dimensions. The sampler is working correctly, but it's processing an incorrectly-sized or uninitialized latent.

## Quick Fix

### Step 1: Use Fresh Empty Latent
1. Add/locate your **Empty Latent Image** node
2. Set the dimensions to match your desired output:
   - Width: Your target width (e.g., 1024)
   - Height: Your target height (e.g., 1024)
   - Batch size: Usually 1
3. Connect this to your KSampler's `latent_image` input

### Step 2: Don't Scale in VAE Decode
The VAE Decode should NOT change dimensions:
- It decodes whatever size latent it receives
- Latent is typically 1/8th the pixel dimensions
- Example: 128x128 latent → 1024x1024 image

### Step 3: Verify Workflow Chain
```
✓ Correct workflow:
Empty Latent (1024x1024) 
  → KSampler 
  → VAE Decode 
  → Image (1024x1024)

✗ Wrong workflow:
Empty Latent (512x512) 
  → KSampler 
  → VAE Decode (try to scale to 1024x1024) 
  → Noise!
```

## Common Scenarios

### Scenario 1: Reusing a Workflow with Different Dimensions

**Problem:** You load a workflow made for 512x512, but want 1024x1024

**Solution:**
1. Find the "Empty Latent Image" node
2. Change width from 512 → 1024
3. Change height from 512 → 1024
4. Re-queue the prompt (don't just hit "Queue" - regenerate everything)

### Scenario 2: Testing Multiple Resolutions

**Problem:** You want to test different resolutions in sequence

**Solution:** Create separate Empty Latent Image nodes for each resolution or use a workflow that properly regenerates the latent each time.

### Scenario 3: Using Image-to-Image

**Problem:** Starting from an existing image with wrong dimensions

**Solution:**
- If upscaling: Use a proper upscale node first
- If downscaling: Resize the image BEFORE encoding to latent
- Don't change dimensions in the middle of the latent processing

## Verification Checklist

Run through this checklist:

- [ ] Empty Latent Image dimensions match desired output
- [ ] Empty Latent is connected to KSampler
- [ ] KSampler is processing the fresh latent
- [ ] VAE Decode is NOT trying to change dimensions
- [ ] You're not reusing a cached/old latent
- [ ] The model is fully loaded (no interrupted loads)
- [ ] CLIP embeddings are generated properly

## Technical Details

### Why This Happens

1. **Latent Space**: ComfyUI works in "latent space" (compressed representation)
   - Latent dimensions = Pixel dimensions ÷ 8
   - 1024x1024 pixels = 128x128 latent
   - 512x512 pixels = 64x64 latent

2. **Dimension Mismatch**: If you:
   - Create 64x64 latent (512x512)
   - Try to decode as 128x128 (1024x1024)
   - Result: Noise, artifacts, or errors

3. **Empty Latent = Starting Point**: 
   - "Empty" latent is actually random noise
   - The sampler refines this noise into an image
   - If the noise dimensions are wrong, output will be wrong

### What the Sampler Does

```python
# Simplified view of what happens:
empty_latent = torch.randn(1, 4, H//8, W//8)  # Random noise at correct size
refined_latent = sampler.refine(empty_latent, model, conditioning, steps)
image = vae.decode(refined_latent)  # Decode to pixels

# If H//8 and W//8 don't match your target:
# → refined_latent has wrong shape
# → VAE can't decode properly  
# → You get noise
```

## Advanced Troubleshooting

### Check Latent Dimensions

Enable debug mode to see actual latent dimensions:
```bash
# Set environment variable
set ROCM_NINODES_DEBUG=1  # Windows
export ROCM_NINODES_DEBUG=1  # Linux/Mac

# Run ComfyUI
python main.py
```

Look for messages like:
```
📊 Latent tensor info: shape=torch.Size([1, 4, 64, 64]), ...
```

The last two dimensions should be:
- Width÷8 and Height÷8 for your target dimensions
- Example: 1024x1024 → should see [1, 4, 128, 128]

### Check CLIP Embeddings

Make sure your text prompts are being encoded:
- Check for CLIP loading messages
- Verify positive/negative conditioning is connected
- Try a simple prompt like "a photo of a cat" first

### Memory Issues

Sometimes dimensions that are too large cause memory issues that manifest as noise:
- Check ComfyUI console for OOM (Out of Memory) errors
- Try smaller dimensions first (512x512)
- If it works at 512x512 but not 1024x1024, it's a memory issue

## Still Having Issues?

### Collect Debug Info

1. **Workflow JSON**: Export your workflow
2. **Console Log**: Copy the ComfyUI console output
3. **Node Settings**: Screenshot your Empty Latent Image and KSampler nodes
4. **System Info**: Run the ROCm diagnostics node

### Test with Basic Workflow

Try this minimal workflow:
```
1. ROCm Checkpoint Loader
   └─ Load: flux1-dev.safetensors (or your model)

2. CLIP Text Encode (Positive)
   └─ Text: "a beautiful landscape"
   
3. CLIP Text Encode (Negative)
   └─ Text: "blurry, bad quality"

4. Empty Latent Image
   └─ Width: 512
   └─ Height: 512
   └─ Batch: 1

5. ROCm KSampler
   └─ Steps: 20
   └─ CFG: 7.0
   └─ Sampler: euler
   └─ Scheduler: normal
   └─ Denoise: 1.0

6. ROCm VAE Decode
   └─ (default settings)

7. Save Image
```

If this basic workflow produces noise, there's likely an issue with:
- Model file (corrupted download)
- ComfyUI installation
- GPU/ROCm driver issues

If this basic workflow works but your complex workflow doesn't:
- The issue is in your workflow structure
- Check each node's connections
- Verify dimension consistency throughout

## Prevention

### Best Practices

1. **Always regenerate latents** when changing dimensions
2. **Test at low resolutions first** (512x512) before trying high-res
3. **Use workflow templates** for different resolutions
4. **Save working workflows** before experimenting
5. **Check console output** for errors/warnings

### Workflow Organization

Create separate workflow files for different use cases:
```
workflows/
├── basic_512.json       # 512x512 generation
├── basic_1024.json      # 1024x1024 generation  
├── hires_2048.json      # High-res generation
└── img2img_1024.json    # Image-to-image workflow
```

This prevents dimension confusion when switching between projects.


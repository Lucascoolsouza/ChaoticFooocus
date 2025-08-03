# Disco Diffusion Debug Guide

## Current Issues and Solutions

### Issue: Preview showing default pipeline image instead of disco effects

The disco distortion is being applied, but you might not be seeing it in the preview for several reasons:

## Debugging Steps

### 1. Check Console Output
When you generate an image with disco enabled, you should see:

```
[Disco] AGGRESSIVE initial injection - scale=10.0, preset=psychedelic
[Disco] ===== NORMAL MODE =====
[Disco] AGGRESSIVE injection: psychedelic distortion with scale 10.0 x1.0
[Disco] Applying PSYCHEDELIC distortion at scale 100.00
[Disco] Grid stats - min=-2.0000, max=2.0000, mean=0.0000
[Disco] AGGRESSIVE psychedelic distortion applied with blend factor 0.500
[Disco] ⭐ SAVED DISCO INITIAL LATENT to disco_initial_latent_[timestamp].png ⭐
```

### 2. Check for Debug Images
The system now saves debug images to disk:
- `disco_initial_latent_[timestamp].png` - Shows the disco-distorted initial latent
- `debug_step_005.png`, `debug_step_010.png`, etc. - Shows disco effects during generation

### 3. Recommended Settings for Visible Effects

**For Maximum Visibility:**
- `disco_enabled`: True
- `disco_scale`: 15.0 or higher
- `disco_preset`: 'scientific' or 'psychedelic'
- Use simple prompts like "abstract colorful painting"

**Settings Explanation:**
- `disco_scale < 5`: Very subtle effects
- `disco_scale 5-10`: Moderate effects
- `disco_scale 10-20`: Strong effects
- `disco_scale > 20`: Extreme effects

### 4. Why You Might Not See Effects in Preview

1. **Preview Timing**: The UI preview might be showing intermediate steps before disco effects are fully applied
2. **Scale Too Low**: disco_scale < 5 produces very subtle effects
3. **Complex Prompts**: Detailed prompts can override disco effects
4. **Sampler Issues**: Some samplers might not preserve disco effects well

### 5. Troubleshooting Checklist

**✓ Check these in order:**

1. **Console Messages**: Do you see disco debug messages?
   - If NO: Check that disco_enabled=True and disco_scale > 0
   - If YES: Continue to next step

2. **Debug Images**: Are debug images being saved to disk?
   - Check for `disco_initial_latent_*.png` files
   - If NO: Check file permissions and disk space
   - If YES: Compare with final result

3. **Effect Intensity**: Is the disco_scale high enough?
   - Try disco_scale = 20.0 for testing
   - Use 'scientific' preset for maximum effect

4. **Prompt Interference**: Is your prompt too specific?
   - Try simple prompts like "colorful abstract art"
   - Avoid detailed descriptions that might override effects

### 6. Current Integration Points

The disco effects are applied at these points:

1. **Initial Latent** (100% intensity): Applied once before generation starts
2. **First 50% of Steps** (decreasing intensity): Applied every few steps during generation
3. **Preview Generation**: Disco-enhanced previews should be generated

### 7. Expected Behavior

**What should happen:**
1. Initial latent gets disco distortion applied
2. During first 50% of generation, additional disco effects are applied
3. Debug images are saved showing the effects
4. Final image should show disco characteristics (swirls, waves, distortions)

**What you should see in console:**
```
[Disco] AGGRESSIVE initial injection - scale=15.0, preset=psychedelic
[Disco] FIRST-HALF INJECTION at step 1/30 (3.3%) - intensity 1.00
[Disco] FIRST-HALF INJECTION at step 3/30 (10.0%) - intensity 0.95
[Disco] FIRST-HALF INJECTION at step 6/30 (20.0%) - intensity 0.85
...
[Disco] Reached 50% mark - stopping disco injection to let image settle
```

### 8. Quick Test

Run this in your environment to test disco functionality:

```python
from extras.disco_diffusion.pipeline_disco import inject_disco_distortion
import torch

# Create test tensor
test = torch.randn(1, 4, 32, 32)

# Apply extreme distortion
result = inject_disco_distortion(
    test, 
    disco_scale=25.0, 
    distortion_type='scientific',
    intensity_multiplier=2.0
)

# Check difference
diff = (result - test).abs().mean().item()
print(f"Distortion applied: {diff > 0.1}")
```

### 9. If Still Not Working

**Check these files exist and have content:**
- `extras/disco_diffusion/pipeline_disco.py`
- `extras/disco_diffusion/disco_integration.py`
- `extras/disco_diffusion/__init__.py`

**Verify parameters are being passed:**
- Check that async_worker.py is passing disco parameters
- Verify process_diffusion() receives disco parameters
- Confirm disco_enabled=True in the call

### 10. Alternative: Force Disco Test

Add this to the beginning of `inject_disco_distortion()` for testing:

```python
# FORCE EXTREME DISTORTION FOR TESTING
if not test_mode:
    disco_scale = max(disco_scale, 20.0)  # Force minimum scale
    intensity_multiplier = max(intensity_multiplier, 1.5)  # Force intensity
    print(f"[DISCO TEST] Forcing scale={disco_scale}, intensity={intensity_multiplier}")
```

This will ensure disco effects are always visible during testing.
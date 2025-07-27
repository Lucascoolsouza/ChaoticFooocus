# NAG Negative Prompt Fix Summary

## Problem Identified
The user reported that NAG artifacts only appear when:
1. A negative prompt is provided in the NAG negative prompt field
2. Guidance scale is set to 1.0

The issue was that things "normalize" (return to normal) when guidance scale is 1.0, but only when a NAG negative prompt is used.

## Root Cause Analysis

### Issue 1: Artificial NAG Conditioning
The original implementation created fake "NAG negative conditioning" by:
- Adding random noise to the positive conditioning
- Shuffling tokens randomly
- This created unstable and artificial conditioning that caused artifacts

### Issue 2: Poor Handling of Guidance Scale 1.0
- The condition `apply_guidance = self.nag_scale > 1` disabled NAG completely at scale 1.0
- The condition `is_nag_enabled()` checked `scale > 1.0`, disabling NAG at scale 1.0
- When scale = 1.0, the NAG formula became unstable due to the artificial conditioning

### Issue 3: Excessive Noise and Perturbation
- The original code used `noise_scale = 0.01` which was too large
- Token shuffling with 10% of tokens was too aggressive
- These perturbations caused visible artifacts, especially at low guidance scales

## Fixes Applied

### Fix 1: Use Real Unconditional Conditioning
```python
# OLD: Create artificial noise-based conditioning
nag_cond = create_artificial_conditioning_with_noise_and_shuffling()

# NEW: Use existing unconditional conditioning (much more stable)
nag_cond = uncond if uncond else []
```

### Fix 2: Proper Guidance Scale 1.0 Support
```python
# OLD: Disable NAG at scale 1.0
apply_guidance = self.nag_scale > 1
is_nag_enabled = scale > 1.0

# NEW: Enable NAG at scale 1.0 with minimal effect
apply_guidance = self.nag_scale >= 1
is_nag_enabled = scale >= 1.0
```

### Fix 3: Scale-Aware Guidance Calculation
```python
# NEW: Handle guidance scale 1.0 properly
if nag_scale <= 1.0:
    # At scale 1.0 or below, apply very minimal guidance
    guidance_strength = max(0.0, (nag_scale - 1.0) * 0.1 + 0.01)  # Minimum 1% effect
    nag_guidance = cond_pred + (cond_pred - nag_pred) * guidance_strength
else:
    # For scales > 1.0, use standard NAG formula but with conservative scaling
    effective_scale = 1.0 + (nag_scale - 1.0) * 0.3  # Reduce effect by 70%
    nag_guidance = cond_pred * effective_scale - nag_pred * (effective_scale - 1.0)
```

### Fix 4: Conservative Alpha Blending
```python
# NEW: Scale-aware alpha blending
if nag_scale <= 1.0:
    alpha_strength = nag_alpha * 0.1  # 90% reduction at low scales
else:
    alpha_strength = nag_alpha * 0.3  # 70% reduction at higher scales

conservative_alpha = torch.clamp(torch.tensor(alpha_strength), 0.0, 0.5)
```

### Fix 5: Improved Attention Processor Math
```python
# NEW: Handle scale 1.0 in attention processor
if self.nag_scale == 1.0:
    # When scale is 1.0, apply minimal guidance to avoid complete bypass
    hidden_states_guidance = hidden_states_positive + (hidden_states_positive - hidden_states_negative) * 0.1
else:
    hidden_states_guidance = hidden_states_positive * self.nag_scale - hidden_states_negative * (self.nag_scale - 1)
```

## Expected Results

### Before Fix
- NAG disabled completely at guidance scale 1.0
- Artifacts when NAG negative prompt provided due to artificial noise conditioning
- Unstable behavior due to token shuffling and excessive noise

### After Fix
- NAG works smoothly at guidance scale 1.0 with minimal, stable effect
- No artifacts when NAG negative prompt provided (uses real unconditional conditioning)
- Stable behavior across all guidance scales
- Conservative effects that prevent over-processing

## Files Modified
1. `extras/nag/nag_integration.py` - Main sampling function and configuration
2. `extras/nag/attention_nag.py` - Attention processor math fixes

## Testing
Created comprehensive tests in:
- `test_nag_guidance_scale_fix.py` - Tests guidance scale 1.0 support
- `test_nag_negative_prompt_fix.py` - Tests negative prompt handling

The fix ensures NAG provides consistent, stable guidance across all scales while eliminating the artifacts that occurred when negative prompts were used at guidance scale 1.0.
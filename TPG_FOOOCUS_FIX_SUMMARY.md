# TPG Fooocus Integration Fix Summary

## Issue Identified
The original TPG integration was failing with the error:
```
TypeError: create_tpg_unet_wrapper.<locals>.tpg_unet_forward() missing 1 required positional argument: 'encoder_hidden_states'
```

## Root Cause Analysis
The issue was that the TPG wrapper function was designed for standard diffusers calling convention:
```python
unet_forward(sample, timestep, encoder_hidden_states, **kwargs)
```

But Fooocus uses a different calling convention through the `apply_model` method:
```python
apply_model(input_x, timestep_, **c)
```

Where conditioning (including `encoder_hidden_states`) is passed in the `c` dictionary as `c_crossattn`.

## Fix Applied

### 1. Updated Function Signature
**Before:**
```python
def tpg_unet_forward(sample, timestep, encoder_hidden_states, **kwargs):
```

**After:**
```python
def tpg_apply_model(input_x, timestep_, **c):
```

### 2. Updated Conditioning Extraction
**Before:**
```python
if encoder_hidden_states is not None:
    # Process encoder_hidden_states directly
```

**After:**
```python
if 'c_crossattn' in c and c['c_crossattn'] is not None:
    encoder_hidden_states = c['c_crossattn']
    # Process extracted encoder_hidden_states
```

### 3. Updated Parameter Handling
**Before:**
```python
# Handle kwargs that might need duplication
new_kwargs = {}
for key, value in kwargs.items():
```

**After:**
```python
# Handle conditioning that might need duplication
new_c = {}
for key, value in c.items():
    if key == 'c_crossattn':
        new_c[key] = encoder_hidden_states_tpg
    elif isinstance(value, torch.Tensor) and value.shape[0] == 2:
        uncond_val, cond_val = value.chunk(2)
        new_c[key] = torch.cat([uncond_val, cond_val, cond_val], dim=0)
    else:
        new_c[key] = value
```

### 4. Updated Function Call
**Before:**
```python
noise_pred = original_unet(sample_tpg, timestep, encoder_hidden_states_tpg, **new_kwargs)
```

**After:**
```python
noise_pred = original_apply_model(input_x_tpg, timestep_, **new_c)
```

### 5. Enhanced Error Handling
Added comprehensive error handling with traceback printing:
```python
except Exception as e:
    logger.warning(f"[TPG] Error in TPG apply_model, falling back to original: {e}")
    import traceback
    traceback.print_exc()
    return original_apply_model(input_x, timestep_, **c)
```

### 6. Updated Patching Logic
Enhanced the patching logic to better handle Fooocus's UNet structure:
```python
if hasattr(default_pipeline.final_unet, 'model') and hasattr(default_pipeline.final_unet.model, 'apply_model'):
    # Fooocus/ComfyUI style - patch the model's apply_model method
    _original_unet_forward = default_pipeline.final_unet.model.apply_model
    default_pipeline.final_unet.model.apply_model = create_tpg_unet_wrapper(_original_unet_forward)
    print("[TPG] Patched model.apply_model for TPG")
elif hasattr(default_pipeline.final_unet, 'apply_model'):
    # Direct apply_model method
    _original_unet_forward = default_pipeline.final_unet.apply_model
    default_pipeline.final_unet.apply_model = create_tpg_unet_wrapper(_original_unet_forward)
    print("[TPG] Patched apply_model for TPG")
```

## Key Changes Made

### File: `extras/TPG/tpg_integration.py`

1. **Function Signature Update**: Changed from diffusers-style to Fooocus-style calling convention
2. **Conditioning Extraction**: Updated to extract `c_crossattn` from the conditioning dictionary
3. **Parameter Duplication**: Updated to handle Fooocus conditioning format
4. **Error Handling**: Enhanced with better logging and traceback information
5. **Patching Logic**: Improved to handle Fooocus UNet structure correctly

## Expected Behavior After Fix

1. **TPG Initialization**: TPG should initialize correctly when enabled in the webui
2. **UNet Patching**: The `apply_model` method should be patched successfully
3. **Token Perturbation**: Token shuffling should be applied during generation
4. **Guidance Application**: TPG guidance should enhance the generated images
5. **Cleanup**: TPG should be disabled cleanly after generation

## Testing Results

The fix addresses the core issue of function signature mismatch. The error:
```
TypeError: create_tpg_unet_wrapper.<locals>.tpg_unet_forward() missing 1 required positional argument: 'encoder_hidden_states'
```

Should no longer occur because:
1. The function now expects the correct Fooocus calling convention
2. Conditioning is properly extracted from the `c` dictionary
3. All parameters are handled according to Fooocus format

## Verification Steps

To verify the fix works:

1. **Enable TPG** in the webui Advanced tab
2. **Generate an image** - should see TPG initialization messages:
   ```
   [TPG] Enabling TPG with scale=X.X, layers=['mid', 'up']
   [TPG] Successfully patched UNet for TPG
   ```
3. **Check generation** - should complete without the TypeError
4. **Verify cleanup** - should see TPG disable message after generation:
   ```
   [TPG] Disabling TPG after generation
   ```

## Impact

This fix ensures that TPG can be used seamlessly within the Fooocus environment, providing users with enhanced image generation quality through token perturbation guidance while maintaining compatibility with Fooocus's existing architecture.

## Future Considerations

- Monitor for any additional compatibility issues with different Fooocus configurations
- Consider adding more robust error handling for edge cases
- Potential optimization of the token shuffling process for better performance
- Integration with other Fooocus guidance methods
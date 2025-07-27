# TPG WebUI Integration Summary

## Overview
Token Perturbation Guidance (TPG) has been successfully integrated into the Fooocus WebUI. This implementation provides a user-friendly interface for applying TPG to enhance image generation quality.

## Files Modified

### 1. webui.py
- **Added TPG Controls Section** in the Advanced tab
- **Controls Added:**
  - `tpg_enabled`: Checkbox to enable/disable TPG
  - `tpg_scale`: Slider for TPG guidance scale (0.0-10.0, default 3.0)
  - `tpg_applied_layers`: CheckboxGroup for selecting UNet layers (['down', 'mid', 'up'])
  - `tpg_shuffle_strength`: Slider for token shuffle strength (0.0-1.0, default 1.0)
  - `tpg_adaptive_strength`: Checkbox for adaptive strength during sampling
  - `tpg_preset`: Dropdown with predefined settings
  - `tpg_status`: Status display showing current TPG configuration

- **Presets Available:**
  - **General**: Scale 3.0, layers ['mid', 'up'], shuffle 1.0, adaptive True
  - **Artistic**: Scale 4.0, layers ['mid', 'up'], shuffle 1.0, adaptive True  
  - **Photorealistic**: Scale 2.5, layers ['up'], shuffle 0.8, adaptive True
  - **Detailed**: Scale 3.5, layers ['mid', 'up'], shuffle 1.0, adaptive True

- **Interactive Features:**
  - Controls become visible when TPG is enabled
  - Preset selection automatically updates parameters
  - Real-time status updates showing current configuration

### 2. modules/async_worker.py
- **Added TPG Parameter Handling** in AsyncTask class
- **Parameters Added:**
  - `self.tpg_enabled`: Boolean flag for TPG activation
  - `self.tpg_scale`: TPG guidance scale value
  - `self.tpg_applied_layers`: List of layers to apply TPG to
  - `self.tpg_shuffle_strength`: Token shuffling strength
  - `self.tpg_adaptive_strength`: Adaptive strength flag

- **Parameter Processing:**
  - Proper parameter extraction from webui controls
  - Default value handling for safety
  - Debug logging for troubleshooting

### 3. modules/default_pipeline.py
- **Updated process_diffusion Function Signature**
- **Added TPG Parameters:**
  - `tpg_enabled=False`
  - `tpg_scale=3.0`
  - `tpg_applied_layers=None`
  - `tpg_shuffle_strength=1.0`
  - `tpg_adaptive_strength=True`

- **TPG Integration Logic:**
  - **Initialization**: TPG is enabled at the start of generation if requested
  - **Configuration**: Uses parameters from webui to configure TPG
  - **Cleanup**: TPG is automatically disabled after generation
  - **Error Handling**: Graceful fallback if TPG fails to initialize

- **Integration Points:**
  - TPG enabled before UNet processing begins
  - TPG disabled after image generation completes
  - Proper error handling and logging

### 4. TPG Module Updates
- **Fixed TPG Pipeline**: Updated to work with diffusers architecture
- **Integration Layer**: Provides clean interface for Fooocus integration
- **Interface Layer**: High-level API for easy TPG usage

## User Interface Location

The TPG controls are located in:
```
Advanced Tab → Token Perturbation Guidance (TPG) Accordion
```

## Usage Instructions

### Basic Usage:
1. Open the **Advanced** tab in Fooocus
2. Expand the **"Token Perturbation Guidance (TPG)"** accordion
3. Check the **"Enable TPG"** checkbox
4. Choose a preset from the dropdown or customize settings manually
5. Generate images as normal

### Preset Recommendations:
- **General**: Good starting point for most images
- **Artistic**: Higher guidance for creative/stylized content
- **Photorealistic**: Gentler guidance to preserve realism
- **Detailed**: Enhanced guidance for fine detail enhancement

### Custom Settings:
- **TPG Scale**: Higher values = stronger effect (recommended: 2.0-5.0)
- **Applied Layers**: 
  - `mid`: Middle UNet layers (core processing)
  - `up`: Upsampling layers (detail refinement)
  - `down`: Downsampling layers (feature extraction)
- **Shuffle Strength**: How much to shuffle tokens (1.0 = full shuffle)
- **Adaptive Strength**: Automatically adjust strength during sampling

## Technical Implementation

### Parameter Flow:
1. **WebUI Controls** → User adjusts TPG settings
2. **AsyncTask** → Parameters stored in task object
3. **process_diffusion** → TPG integration enabled
4. **TPG Integration** → UNet patching applied
5. **Generation** → Images generated with TPG
6. **Cleanup** → TPG disabled automatically

### Integration Architecture:
```
webui.py (UI Controls)
    ↓
async_worker.py (Parameter Handling)
    ↓
default_pipeline.py (Integration Logic)
    ↓
tpg_integration.py (TPG Implementation)
    ↓
pipeline_sdxl_tpg.py (Core TPG Logic)
```

## Safety Features

### Error Handling:
- Graceful fallback if TPG fails to initialize
- Automatic cleanup even if generation fails
- Debug logging for troubleshooting

### Parameter Validation:
- Default values for all parameters
- Safe ranges for sliders
- Automatic layer selection if none specified

### Performance Considerations:
- TPG only active during generation
- Automatic cleanup prevents memory leaks
- Minimal overhead when disabled

## Testing

### Tests Created:
- `test_tpg_fixed.py`: Core TPG functionality
- `test_tpg_integration.py`: Integration layer testing
- `test_tpg_structure.py`: Code structure validation
- `test_tpg_webui_integration.py`: WebUI integration testing

### Validation:
- All TPG components properly structured
- WebUI controls correctly implemented
- Parameter flow working as expected
- Integration points properly connected

## Benefits

### For Users:
- Easy-to-use interface for TPG
- Preset configurations for common use cases
- Real-time status feedback
- No manual configuration required

### For Developers:
- Clean integration with existing codebase
- Modular TPG implementation
- Comprehensive error handling
- Extensive documentation and testing

## Future Enhancements

### Potential Improvements:
- Advanced TPG scheduling options
- Per-layer TPG strength control
- TPG effect preview
- Integration with other guidance methods
- Performance optimizations

### Extensibility:
- Easy to add new TPG presets
- Modular design allows for TPG algorithm updates
- Clean API for additional TPG features

## Conclusion

The TPG integration provides a powerful and user-friendly way to enhance image generation in Fooocus. The implementation is robust, well-tested, and ready for production use. Users can now easily apply Token Perturbation Guidance to improve their generated images with just a few clicks.
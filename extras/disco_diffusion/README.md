# Disco Diffusion Extension for Fooocus

This extension adds psychedelic disco diffusion effects to Fooocus, inspired by the original Disco Diffusion project. It creates trippy, fractal-like, and kaleidoscopic images with various visual transformations.

## Features

### Transform Effects
- **Spherical Distortion**: Creates curved, bubble-like distortions
- **Kaleidoscope**: Generates symmetrical, kaleidoscopic patterns
- **Fractal Zoom**: Applies recursive zoom effects for fractal-like imagery
- **Color Shift**: Manipulates hue, saturation, and brightness dynamically

### Presets
- **Psychedelic**: High saturation, multiple effects, vibrant colors
- **Fractal**: Focus on fractal zoom and recursive patterns
- **Kaleidoscope**: Emphasizes symmetrical kaleidoscopic effects
- **Dreamy**: Subtle effects with soft color transitions
- **Custom**: Full manual control over all parameters

### Animation & Movement
- **Zoom**: Continuous fractal zooming effects
- **Rotate**: Rotating kaleidoscope patterns
- **Translate**: Moving/shifting effects
- **None**: Static effects only

### Visual Controls
- **Disco Scale**: Overall strength of all effects (0.0 - 1.0)
- **Color Coherence**: How much to preserve original colors (0.0 = full transformation, 1.0 = original colors)
- **Saturation Boost**: Increase color saturation (0.5 - 2.0)
- **Contrast Boost**: Increase image contrast (0.5 - 2.0)
- **Symmetry Mode**: Apply horizontal, vertical, or radial symmetry
- **Fractal Octaves**: Complexity of fractal effects (1-6)

## Usage

1. **Enable the Extension**: In the Advanced tab, expand "Disco Diffusion (Psychedelic Effects)" and check "Enable Disco Diffusion"

2. **Choose a Preset**: Select from predefined presets or use "Custom" for full control:
   - `psychedelic`: Best for trippy, colorful images
   - `fractal`: Great for mathematical/geometric patterns
   - `kaleidoscope`: Perfect for symmetrical designs
   - `dreamy`: Subtle effects for artistic enhancement

3. **Adjust Parameters**: Fine-tune the effects using the various sliders and options

4. **Set Seed**: Use the Disco Seed field for reproducible effects (leave empty for random)

## Technical Details

### How It Works
The extension patches the sampling function to apply visual transformations during the diffusion process. Effects are applied at specific steps during generation, creating psychedelic transformations that evolve with the image.

### Transform Pipeline
1. **Spherical Distortion**: Applies coordinate transformation using polar mapping
2. **Kaleidoscope**: Uses polar coordinates and modular arithmetic for symmetry
3. **Fractal Zoom**: Implements recursive coordinate scaling
4. **Color Shift**: Converts RGB to HSV for hue/saturation manipulation

### Integration Points
- Patches `ldm_patched.modules.samplers.sampling_function`
- Integrates with Fooocus pipeline in `modules.default_pipeline.process_diffusion`
- Automatically activates/deactivates with generation lifecycle

## Examples

### Basic Usage
```python
# Enable disco with psychedelic preset
disco_enabled = True
disco_preset = 'psychedelic'
disco_scale = 0.7
```

### Custom Configuration
```python
# Custom disco setup
disco_enabled = True
disco_preset = 'custom'
disco_scale = 0.5
disco_transforms = ['spherical', 'kaleidoscope', 'color_shift']
disco_saturation_boost = 1.5
disco_symmetry_mode = 'radial'
```

## Tips for Best Results

1. **Start with Presets**: Use predefined presets first, then customize
2. **Moderate Scale**: Values between 0.3-0.7 usually work best
3. **Combine Effects**: Mix different transforms for unique results
4. **Use Seeds**: Set disco_seed for reproducible psychedelic effects
5. **Adjust Coherence**: Lower values for more transformation, higher for subtlety

## Troubleshooting

### Common Issues
- **No Effect Visible**: Increase disco_scale or check that disco_enabled is True
- **Too Intense**: Reduce disco_scale or increase disco_color_coherence
- **Performance Issues**: Reduce number of transforms or fractal_octaves

### Debug Information
The extension prints status messages to console:
```
[Disco] Activating Disco Diffusion with scale 0.5
[Disco] Successfully patched sampling function
[Disco] Disabling Disco Diffusion after generation
```

## File Structure
```
extras/disco_diffusion/
├── __init__.py              # Package initialization
├── pipeline_disco.py       # Core disco effects and transforms
├── disco_integration.py    # Integration with Fooocus pipeline
└── README.md               # This documentation
```

## Dependencies
- PyTorch (for tensor operations)
- Standard Fooocus dependencies
- No additional packages required

## Performance Notes
- Disco effects add computational overhead during sampling
- More complex transforms (fractal_zoom, kaleidoscope) are more expensive
- Effects are applied selectively at specific sampling steps to balance quality/performance

## Contributing
To add new transforms:
1. Add transform function to `DiscoTransforms` class in `pipeline_disco.py`
2. Update `disco_transforms` list in `modules/flags.py`
3. Add preset configurations in `get_disco_presets()`

## License
This extension follows the same license as Fooocus.
# True Disco Diffusion Extension for Fooocus

This extension implements the **real Disco Diffusion algorithm** with CLIP guidance and geometric transforms, based on the scientific principles of the original Disco Diffusion project. It uses CLIP-guided diffusion with geometric transformations to create truly psychedelic, fractal-like images.

## Features

### Scientific Algorithm Features

- **CLIP Guidance**: Uses CLIP model to guide generation based on text prompts (spherical distance loss)
- **Geometric Transforms**: Real 2D transformations (translate, rotate, zoom) applied during diffusion
- **Cutout Analysis**: Creates multiple random cutouts for CLIP analysis (fractal-like detail enhancement)
- **Total Variation Loss**: Smoothness constraint to prevent noise artifacts
- **Range Loss**: Keeps pixel values in reasonable bounds

### Scientific Presets

- **Psychedelic**: High CLIP guidance (1000), all transforms, moderate cutouts (16)
- **Fractal**: Very high guidance (1500), zoom+rotate, many cutouts (32), high TV loss
- **Kaleidoscope**: Medium guidance (800), rotate+translate, radial symmetry
- **Dreamy**: Low guidance (500), translate only, few cutouts (8), high TV smoothing
- **Scientific**: Maximum guidance (2000), all transforms, maximum cutouts (40)
- **Custom**: Full manual control over all scientific parameters

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

### 🚀 Quick Start
1. **Enable the Extension**: In the Advanced tab, expand "Disco Diffusion (Psychedelic Effects)" and check "Enable Disco Diffusion"

2. **Choose a Scientific Preset**: Select from research-based presets:
   - `scientific`: Maximum CLIP guidance (2000), 40 cutouts, full algorithm
   - `psychedelic`: High guidance (1000), all transforms, vibrant effects
   - `fractal`: Very high guidance (1500), zoom+rotate, fractal patterns
   - `kaleidoscope`: Medium guidance (800), radial symmetry
   - `dreamy`: Low guidance (500), subtle effects

3. **CLIP Installation**: For full functionality, CLIP will be downloaded automatically on first use
   - ✅ **With CLIP**: Full scientific algorithm with semantic guidance
   - 🔄 **Without CLIP**: Geometric transforms only (still creates psychedelic effects)

4. **Adjust Parameters**: Fine-tune using scientific parameters:
   - **disco_scale**: CLIP guidance strength (500-2500)
   - **cutn**: Number of cutouts for analysis (8-64)
   - **tv_scale**: Total variation smoothing (0-300)

5. **Generate**: Create scientifically-guided psychedelic images!

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
- **CLIP (optional but recommended)**: For full scientific functionality
  - Automatic download on first use
  - Manual install: `pip install git+https://github.com/openai/CLIP.git`
  - See `CLIP_INSTALLATION.md` for detailed instructions

## Installation Status
The extension works in two modes:
- **🧬 Scientific Mode** (with CLIP): Full algorithm with semantic guidance
- **🎨 Geometric Mode** (without CLIP): Psychedelic transforms only

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

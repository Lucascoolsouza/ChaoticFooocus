# Seamless Tiling Enhancement

The seamless tiling enhancement feature allows you to create images that tile seamlessly when repeated, making them perfect for textures, backgrounds, and patterns.

## How to Use

1. **Enable Enhancement**: Check the "Enhance" checkbox in the main interface
2. **Select Seamless Tiling**: In the Enhancement panel, choose "Seamless Tiling" from the "Upscale or Variation" dropdown
3. **Configure Options**:
   - **Method**: Choose how the seamless effect is created
   - **Edge Overlap Ratio**: Control how much edge blending is applied

## Seamless Tiling Methods

### Blend (Default)
- **Description**: Blends opposite edges of the image together
- **Best for**: General purpose textures and patterns
- **Effect**: Creates smooth transitions between tile edges

### Mirror
- **Description**: Uses mirroring technique with edge blending
- **Best for**: Symmetric patterns and organic textures
- **Effect**: Creates natural-looking continuity

### Offset
- **Description**: Shifts the image by half and blends with original
- **Best for**: Random patterns and noise textures
- **Effect**: Breaks up obvious repetition patterns

## Edge Overlap Ratio

Controls how much of the image edges are used for blending:
- **Lower values (0.05-0.10)**: Minimal blending, preserves more original detail
- **Medium values (0.10-0.20)**: Balanced blending (recommended)
- **Higher values (0.20-0.30)**: More aggressive blending, smoother transitions

## Tips for Best Results

1. **Image Content**: Works best with:
   - Textures (wood, stone, fabric)
   - Abstract patterns
   - Natural elements (clouds, water, terrain)

2. **Avoid**: Images with:
   - Clear directional elements
   - Text or logos
   - Distinct objects at edges

3. **Testing**: Use the preview feature to check how your image tiles before final generation

## Technical Details

The seamless tiling enhancement:
- Processes images after generation or upload
- Maintains original image dimensions
- Uses advanced edge blending algorithms
- Supports all image formats supported by Fooocus

## Integration with Other Features

Seamless tiling can be combined with:
- **Upscaling**: Apply seamless tiling to upscaled images
- **Style Transfer**: Create seamless stylized textures
- **Inpainting**: Fix seams in existing tiled images

## Examples

### Creating a Seamless Wood Texture
1. Generate or upload a wood texture image
2. Enable Enhancement → Seamless Tiling
3. Use "Blend" method with 0.15 overlap ratio
4. Result: Perfect for 3D modeling and game development

### Making Seamless Patterns
1. Create an abstract pattern
2. Use "Mirror" method for symmetric results
3. Adjust overlap ratio based on pattern complexity
4. Result: Ideal for web backgrounds and print design

## Troubleshooting

**Visible seams still present?**
- Increase the edge overlap ratio
- Try a different method
- Ensure image doesn't have strong directional elements

**Image looks too blurred?**
- Decrease the edge overlap ratio
- Use "Blend" method for sharper results
- Check original image quality

**Method not working as expected?**
- Different methods work better for different content types
- Experiment with all three methods
- Consider the nature of your source image
# Background Removal Enhancement

This feature adds background removal capability to the Fooocus enhance options using rembg 2.0.

## How to Use

1. **Enable Enhancement**: Check the "Enhance" checkbox in the main interface
2. **Select Remove Background**: In the enhance panel, under "Upscale or Variation", select "Remove Background"
3. **Choose Model**: Select your preferred background removal model from the dropdown that appears
4. **Upload Image**: Use the enhance input image upload or generate an image first
5. **Generate**: Click generate to process the image with background removal

## Available Models

- **u2net** (Default): General purpose background removal, good balance of speed and quality
- **u2netp**: Lightweight version of u2net, faster but slightly lower quality
- **u2net_human_seg**: Optimized for human subjects
- **silueta**: High accuracy model, slower but better results
- **isnet-general-use**: New general purpose model with improved accuracy

## Tips

- **u2net** works well for most general images
- Use **u2net_human_seg** for portraits and people photos
- Try **silueta** for complex subjects that need high precision
- **u2netp** is good when you need faster processing
- **isnet-general-use** provides the latest improvements in accuracy

## Processing Order

The background removal can be set to process:
- **Before First Enhancement**: Remove background first, then apply other enhancements
- **After Last Enhancement**: Apply other enhancements first, then remove background

## Technical Details

- Uses rembg 2.0 library for background removal
- No AI generation steps required (instant processing)
- Outputs PNG with transparent background
- Compatible with all other Fooocus enhancement features
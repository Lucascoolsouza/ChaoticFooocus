# Simplified pixel art sampler - only pixelation, no color bleeding
@torch.no_grad()
def sample_euler_pixel_art_clean(model, x, sigmas, extra_args=None, callback=None, disable=None, s_noise=1., pixel_scale=2):
    """Clean Euler sampler with only pixelation effects - no color bleeding or artifacts."""
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    
    print(f"Clean Pixel Art Euler sampler active - scale: {pixel_scale}")
    
    def apply_clean_pixelation(tensor, scale_factor, step_ratio):
        """Apply only pixelation effect - no color quantization or sharpening"""
        if scale_factor <= 1 or step_ratio < 0.8:  # Only in final steps
            return tensor
            
        b, c, h, w = tensor.shape
        
        # Simple pixelation
        new_h = max(32, h // scale_factor)
        new_w = max(32, w // scale_factor)
        
        # Downsample and upsample
        downsampled = F.interpolate(tensor, size=(new_h, new_w), mode='area')
        pixelated = F.interpolate(downsampled, size=(h, w), mode='nearest')
        
        # Light blend to avoid artifacts
        blend_factor = 0.4
        return blend_factor * pixelated + (1.0 - blend_factor) * tensor
    
    for i in trange(len(sigmas) - 1, disable=disable):
        step_ratio = i / (len(sigmas) - 1)
        
        # Standard Euler step
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        d = to_d(x, sigmas[i], denoised)
        
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
        
        dt = sigmas[i + 1] - sigmas[i]
        x = x + d * dt
        
        # Apply only clean pixelation in final steps
        if i >= len(sigmas) - 3:
            x = apply_clean_pixelation(x, pixel_scale, step_ratio)
    
    return x

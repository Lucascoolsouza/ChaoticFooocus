import cv2
import numpy as np
import torch
from PIL import Image
import logging

logger = logging.getLogger(__name__)

def apply_disco_post_processing(image, disco_scale=5.0, distortion_type='psychedelic', 
                               intensity=1.0, blend_factor=0.5):
    """
    Apply disco diffusion-style post-processing effects to a final image.
    This works on the actual RGB image, not latent space.
    
    Args:
        image: Input image as numpy array [H, W, C] in 0-255 range
        disco_scale: Strength of the disco effect (1-20 recommended)
        distortion_type: Type of distortion ('psychedelic', 'fractal', 'kaleidoscope', etc.)
        intensity: Overall intensity multiplier
        blend_factor: How much to blend with original (0.0 = no effect, 1.0 = full effect)
    
    Returns:
        Enhanced image as numpy array [H, W, C] in 0-255 range
    """
    if image is None or disco_scale <= 0:
        return image
    
    print(f"[Disco Post] Applying {distortion_type} effect - scale={disco_scale:.1f}, blend={blend_factor:.2f}")
    
    # Convert to float for processing
    image_float = image.astype(np.float32) / 255.0
    H, W, C = image_float.shape
    
    # Create coordinate grids
    y_coords = np.linspace(-1, 1, H)
    x_coords = np.linspace(-1, 1, W)
    y_grid, x_grid = np.meshgrid(y_coords, x_coords, indexing='ij')
    
    # Scale the effect - much more aggressive for visible results
    base_scale = disco_scale * intensity * 0.05  # Adjusted for image space
    
    # Apply different distortion types
    if distortion_type == 'psychedelic':
        # Psychedelic swirl and wave distortions
        radius = np.sqrt(x_grid**2 + y_grid**2)
        angle = np.arctan2(y_grid, x_grid)
        
        # Strong swirl distortion
        swirl_strength = base_scale * 3.0
        new_angle = angle + swirl_strength * np.exp(-radius * 1.5)
        
        # Wave distortions
        wave_freq = base_scale * 8.0
        wave_amp = base_scale * 0.8
        x_wave = x_grid + wave_amp * np.sin(wave_freq * y_grid)
        y_wave = y_grid + wave_amp * np.cos(wave_freq * x_grid)
        
        # Add secondary waves
        wave_freq2 = base_scale * 12.0
        wave_amp2 = base_scale * 0.4
        x_wave += wave_amp2 * np.sin(wave_freq2 * y_grid + np.pi/3)
        y_wave += wave_amp2 * np.cos(wave_freq2 * x_grid + np.pi/3)
        
        # Combine swirl and waves
        new_x = radius * np.cos(new_angle) + x_wave * 0.7
        new_y = radius * np.sin(new_angle) + y_wave * 0.7
        
    elif distortion_type == 'fractal':
        # Fractal-like recursive distortions
        scale1 = base_scale * 2.5
        scale2 = base_scale * 1.8
        scale3 = base_scale * 1.2
        
        new_x = x_grid + scale1 * np.sin(3 * x_grid) * np.cos(2 * y_grid)
        new_y = y_grid + scale1 * np.cos(3 * y_grid) * np.sin(2 * x_grid)
        
        # Add finer details
        new_x += scale2 * np.sin(7 * new_x) * np.cos(5 * new_y)
        new_y += scale2 * np.cos(7 * new_y) * np.sin(5 * new_x)
        
        new_x += scale3 * np.sin(13 * new_x) * np.cos(11 * new_y)
        new_y += scale3 * np.cos(13 * new_y) * np.sin(11 * new_x)
        
    elif distortion_type == 'kaleidoscope':
        # Kaleidoscope-like symmetrical distortions
        radius = np.sqrt(x_grid**2 + y_grid**2)
        angle = np.arctan2(y_grid, x_grid)
        
        # Create kaleidoscope effect
        n_mirrors = 8
        mirror_angle = 2 * np.pi / n_mirrors
        folded_angle = np.abs((angle % mirror_angle) - mirror_angle/2)
        
        # Strong radial modulation
        radial_mod = 1 + base_scale * 1.5 * np.sin(radius * 6)
        radial_mod2 = 1 + base_scale * 0.8 * np.cos(radius * 10)
        
        new_x = radius * np.cos(folded_angle) * radial_mod * radial_mod2
        new_y = radius * np.sin(folded_angle) * radial_mod * radial_mod2
        
        # Add spiral component
        spiral_strength = base_scale * 1.2
        spiral_angle = folded_angle + spiral_strength * radius
        new_x += base_scale * 0.6 * np.cos(spiral_angle)
        new_y += base_scale * 0.6 * np.sin(spiral_angle)
        
    elif distortion_type == 'wave':
        # Strong wave distortions
        wave_freq = base_scale * 6.0
        wave_amp = base_scale * 1.0
        
        new_x = x_grid + wave_amp * np.sin(wave_freq * y_grid)
        new_y = y_grid + wave_amp * np.sin(wave_freq * x_grid)
        
        # Add perpendicular waves
        wave_freq2 = base_scale * 9.0
        wave_amp2 = base_scale * 0.6
        new_x += wave_amp2 * np.cos(wave_freq2 * x_grid)
        new_y += wave_amp2 * np.cos(wave_freq2 * y_grid)
        
    elif distortion_type == 'scientific':
        # Maximum aggression - combine all effects
        radius = np.sqrt(x_grid**2 + y_grid**2)
        angle = np.arctan2(y_grid, x_grid)
        
        # Swirl component
        swirl_strength = base_scale * 4.0
        new_angle = angle + swirl_strength * np.exp(-radius * 1.2)
        
        # Wave components
        wave_freq = base_scale * 10.0
        wave_amp = base_scale * 1.2
        x_wave = wave_amp * np.sin(wave_freq * y_grid) * np.cos(wave_freq * x_grid)
        y_wave = wave_amp * np.cos(wave_freq * x_grid) * np.sin(wave_freq * y_grid)
        
        # Fractal components
        fractal_scale = base_scale * 2.0
        x_fractal = fractal_scale * np.sin(5 * x_grid) * np.cos(3 * y_grid)
        y_fractal = fractal_scale * np.cos(5 * y_grid) * np.sin(3 * x_grid)
        
        # Combine everything
        new_x = radius * np.cos(new_angle) + x_wave + x_fractal
        new_y = radius * np.sin(new_angle) + y_wave + y_fractal
        
    else:  # default psychedelic
        radius = np.sqrt(x_grid**2 + y_grid**2)
        angle = np.arctan2(y_grid, x_grid)
        
        swirl_strength = base_scale * 3.0
        new_angle = angle + swirl_strength * np.exp(-radius * 1.5)
        
        wave_amp = base_scale * 0.8
        wave_freq = base_scale * 6.0
        new_x = radius * np.cos(new_angle) + wave_amp * np.sin(wave_freq * y_grid)
        new_y = radius * np.sin(new_angle) + wave_amp * np.cos(wave_freq * x_grid)
    
    # Clamp coordinates to reasonable range
    new_x = np.clip(new_x, -1.5, 1.5)
    new_y = np.clip(new_y, -1.5, 1.5)
    
    # Convert to pixel coordinates
    pixel_x = ((new_x + 1) * 0.5 * (W - 1)).astype(np.float32)
    pixel_y = ((new_y + 1) * 0.5 * (H - 1)).astype(np.float32)
    
    # Apply distortion using OpenCV remap
    try:
        distorted = cv2.remap(image_float, pixel_x, pixel_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        
        # Blend with original
        result = (1.0 - blend_factor) * image_float + blend_factor * distorted
        
        # Add disco-style color enhancement
        if distortion_type in ['psychedelic', 'scientific']:
            # Enhance saturation and add color shifts
            hsv = cv2.cvtColor(result, cv2.COLOR_RGB2HSV)
            
            # Boost saturation
            saturation_boost = 1.0 + (base_scale * 8.0)
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation_boost, 0, 1)
            
            # Add subtle hue shifts for psychedelic effect
            hue_shift = base_scale * 0.1 * np.sin(radius * 4)
            hsv[:, :, 0] = (hsv[:, :, 0] + hue_shift) % 1.0
            
            result = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        
        # Enhance contrast slightly
        contrast_boost = 1.0 + (base_scale * 2.0)
        result = np.clip((result - 0.5) * contrast_boost + 0.5, 0, 1)
        
        # Convert back to uint8
        result = (result * 255).astype(np.uint8)
        
        print(f"[Disco Post] Applied {distortion_type} distortion successfully")
        return result
        
    except Exception as e:
        logger.error(f"Disco post-processing failed: {e}")
        return image

class DiscoPostProcessor:
    """Disco Diffusion post-processor for applying effects to final images."""
    
    def __init__(self):
        self.enabled = False
        self.disco_scale = 10.0
        self.distortion_type = 'psychedelic'
        self.intensity = 1.0
        self.blend_factor = 0.6
    
    def process(self, image):
        """Process an image with disco effects if enabled."""
        if not self.enabled or image is None or self.disco_scale <= 0:
            return image
            
        return apply_disco_post_processing(
            image,
            disco_scale=self.disco_scale,
            distortion_type=self.distortion_type,
            intensity=self.intensity,
            blend_factor=self.blend_factor
        )
    
    def configure(self, enabled=None, disco_scale=None, distortion_type=None, 
                  intensity=None, blend_factor=None):
        """Configure the disco post-processor settings."""
        if enabled is not None:
            self.enabled = enabled
        if disco_scale is not None:
            self.disco_scale = disco_scale
        if distortion_type is not None:
            self.distortion_type = distortion_type
        if intensity is not None:
            self.intensity = intensity
        if blend_factor is not None:
            self.blend_factor = blend_factor
        
        status = f"Disco Post-Processor: {'Enabled' if self.enabled else 'Disabled'}"
        if self.enabled:
            status += f", Scale: {self.disco_scale}, Type: {self.distortion_type}"
        
        return status

# Create a global instance
disco_post_processor = DiscoPostProcessor()

def update_disco_post_processor_settings(enabled, disco_scale, distortion_type, 
                                        intensity=1.0, blend_factor=0.6):
    """Update the disco post-processor settings."""
    return disco_post_processor.configure(
        enabled=enabled,
        disco_scale=disco_scale,
        distortion_type=distortion_type,
        intensity=intensity,
        blend_factor=blend_factor
    )
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
        disco_scale: Strength of the disco effect
        distortion_type: Type of distortion ('psychedelic', 'fractal', 'kaleidoscope', etc.)
        intensity: Overall intensity multiplier
        blend_factor: How much to blend with original (0.0 = no effect, 1.0 = full effect)
    
    Returns:
        Enhanced image as numpy array [H, W, C] in 0-255 range
    """
    if image is None or disco_scale <= 0:
        return image
    
    # Convert to float for processing
    image_float = image.astype(np.float32) / 255.0
    H, W, C = image_float.shape
    
    # Create coordinate grids
    y_coords = np.linspace(-1, 1, H)
    x_coords = np.linspace(-1, 1, W)
    y_grid, x_grid = np.meshgrid(y_coords, x_coords, indexing='ij')
    
    # Scale the effect
    base_scale = disco_scale * intensity * 0.1  # Scale down for image space
    
    # Apply different distortion types
    if distortion_type == 'psychedelic':
        # Psychedelic swirl and wave distortions
        radius = np.sqrt(x_grid**2 + y_grid**2)
        angle = np.arctan2(y_grid, x_grid)
        
        # Swirl distortion
        swirl_strength = base_scale * 0.5
        new_angle = angle + swirl_strength * np.exp(-radius * 2.0)
        
        # Wave distortions
        wave_freq = base_scale * 2.0
        wave_amp = base_scale * 0.2
        x_wave = x_grid + wave_amp * np.sin(wave_freq * y_grid)
        y_wave = y_grid + wave_amp * np.cos(wave_freq * x_grid)
        
        # Combine
        new_x = radius * np.cos(new_angle) + x_wave * 0.5
        new_y = radius * np.sin(new_angle) + y_wave * 0.5
        
    elif distortion_type == 'fractal':
        # Fractal-like recursive distortions
        scale1 = base_scale * 0.8
        scale2 = base_scale * 0.5
        scale3 = base_scale * 0.3
        
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
        n_mirrors = 6
        mirror_angle = 2 * np.pi / n_mirrors
        folded_angle = np.abs((angle % mirror_angle) - mirror_angle/2)
        
        # Radial modulation
        radial_mod = 1 + base_scale * 0.3 * np.sin(radius * 4)
        
        new_x = radius * np.cos(folded_angle) * radial_mod
        new_y = radius * np.sin(folded_angle) * radial_mod
        
    elif distortion_type == 'wave':
        # Simple wave distortions
        wave_freq = base_scale * 1.5
        wave_amp = base_scale * 0.3
        
        new_x = x_grid + wave_amp * np.sin(wave_freq * y_grid)
        new_y = y_grid + wave_amp * np.sin(wave_freq * x_grid)
        
    else:  # default psychedelic
        radius = np.sqrt(x_grid**2 + y_grid**2)
        angle = np.arctan2(y_grid, x_grid)
        
        swirl_strength = base_scale * 0.6
        new_angle = angle + swirl_strength * np.exp(-radius * 2.0)
        
        new_x = radius * np.cos(new_angle)
        new_y = radius * np.sin(new_angle)
    
    # Clamp coordinates
    new_x = np.clip(new_x, -1, 1)
    new_y = np.clip(new_y, -1)
    
    # Convert to pixel coordinates
    pixel_x = ((new_x + 1) * 0.5 * (W - 1)).astype(np.float32)
    pixel_y = ((new_y + 1) * 0.5 * (H - 1)).astype(np.float32)
    
    # Apply distortion using OpenCV remap
    try:
        distorted = cv2.remap(image_float, pixel_x, pixel_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        
        # Blend with original
        result = (1.0 - blend_factor) * image_float + blend_factor * distorted
        
        # Add some color enhancement for disco effect
        if distortion_type in ['psychedelic', 'scientific']:
            # Enhance saturation
            hsv = cv2.cvtColor(result, cv2.COLOR_RGB2HSV)
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * (1.0 + base_scale * 0.2), 0, 1)
            result = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        
        # Convert back to uint8
        result = np.clip(result, 0, 1)
        result = (result * 255).astype(np.uint8)
        
        return result
        
    except Exception as e:
        logger.error(f"Disco post-processing failed: {e}")
        return image

class DiscoDaemon:
    """Disco Diffusion post-processing daemon similar to DetailDaemon"""
    
    def __init__(self):
        self.enabled = False
        self.disco_scale = 5.0
        self.distortion_type = 'psychedelic'
        self.intensity = 1.0
        self.blend_factor = 0.5
        self.apply_to_final = True  # Apply to final image
        self.apply_to_preview = False  # Apply to preview images
    
    def process(self, image):
        """Process an image with disco effects if enabled"""
        if self.enabled and image is not None and self.disco_scale > 0:
            return apply_disco_post_processing(
                image,
                disco_scale=self.disco_scale,
                distortion_type=self.distortion_type,
                intensity=self.intensity,
                blend_factor=self.blend_factor
            )
        return image
    
    def configure(self, enabled=None, disco_scale=None, distortion_type=None, 
                  intensity=None, blend_factor=None, apply_to_final=None, apply_to_preview=None):
        """Configure the disco daemon settings"""
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
        if apply_to_final is not None:
            self.apply_to_final = apply_to_final
        if apply_to_preview is not None:
            self.apply_to_preview = apply_to_preview

# Create a global instance
disco_daemon = DiscoDaemon()

def update_disco_daemon_settings(enabled, disco_scale, distortion_type, intensity, blend_factor):
    """Update the disco daemon settings"""
    disco_daemon.configure(
        enabled=enabled,
        disco_scale=disco_scale,
        distortion_type=distortion_type,
        intensity=intensity,
        blend_factor=blend_factor
    )
    
    return f"Disco Daemon: {'Enabled' if enabled else 'Disabled'}, Scale: {disco_scale}, Type: {distortion_type}"

def apply_disco_to_image(image, disco_params):
    """Apply disco effects to an image using provided parameters"""
    if not disco_params.get('disco_enabled', False):
        return image
    
    return apply_disco_post_processing(
        image,
        disco_scale=disco_params.get('disco_scale', 5.0),
        distortion_type=disco_params.get('disco_preset', 'psychedelic'),
        intensity=disco_params.get('intensity_multiplier', 1.0),
        blend_factor=disco_params.get('blend_factor', 0.5)
    )
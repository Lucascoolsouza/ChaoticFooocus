import gradio as gr
import cv2
import numpy as np
import math

def enhance_psychedelic_details(image, intensity=0.45, color_shift=0.3, fractal_depth=0.6,
                               start=0.15, end=0.9, peak=0.5, bias=0.85, flow_multiplier=1.2,
                               wave_frequency=2.5, start_offset=0.1, end_offset=-0.05,
                               exponent=1.3, fade=0.15, mode='kaleidoscope', smooth=True,
                               chromatic_aberration=True, saturation_boost=0.4, hue_rotation=0.2,
                               contrast_waves=0.3, detail_recursion=3):
    """
    Apply psychedelic enhancement to an image with disco diffusion inspired effects.
    """
    if image is None:
        return image
    
    # Convert to float for processing
    image_float = image.astype(np.float32) / 255.0
    H, W, C = image_float.shape
    
    # Create coordinate grids for effects
    y_coords = np.linspace(0, 1, H)
    x_coords = np.linspace(0, 1, W)
    Y, X = np.meshgrid(y_coords, x_coords, indexing='ij')
    
    # Create base mask with peak intensity
    mask = np.ones((H, W), dtype=np.float32)
    
    # Apply start/end range with peak
    if start > 0 or end < 1:
        if mode in ['kaleidoscope', 'both']:
            # Radial mask for kaleidoscope effect
            center_y, center_x = H // 2, W // 2
            radius = np.sqrt((np.arange(H)[:, None] - center_y)**2 + (np.arange(W) - center_x)**2)
            max_radius = np.sqrt(center_y**2 + center_x**2)
            radial_coords = radius / max_radius
            
            mask *= np.where((radial_coords >= start) & (radial_coords <= end), 1.0, 0.0)
            
            # Add peak intensity at specified point
            peak_mask = np.exp(-((radial_coords - peak) ** 2) / (0.1 ** 2))
            mask = np.maximum(mask, peak_mask * 0.5)
            
        elif mode == 'fluid':
            # Flowing wave mask
            wave_mask = np.sin(Y * wave_frequency * np.pi) * np.cos(X * wave_frequency * np.pi)
            wave_mask = (wave_mask + 1) / 2  # Normalize to 0-1
            mask *= np.where((Y >= start) & (Y <= end), wave_mask, 0.0)
            
        elif mode == 'fractal':
            # Fractal-based mask
            fractal_mask = create_fractal_mask(H, W, fractal_depth)
            mask *= np.where((Y >= start) & (Y <= end), fractal_mask, 0.0)
    
    # Apply wave dynamics
    if wave_frequency > 0:
        wave_x = np.sin(X * wave_frequency * 2 * np.pi) * flow_multiplier
        wave_y = np.cos(Y * wave_frequency * 2 * np.pi) * flow_multiplier
        mask = mask * (1 + 0.2 * wave_x * wave_y)
    
    # Apply offsets with flow
    if start_offset != 0:
        shift_start = int(H * start_offset)
        mask = np.roll(mask, shift_start, axis=0)
    
    if end_offset != 0:
        shift_end = int(W * end_offset)
        mask = np.roll(mask, shift_end, axis=1)
    
    # Apply exponent curve for non-linear intensity
    if exponent != 1:
        mask = np.power(np.clip(mask, 0, 1), exponent)
    
    # Apply fade with psychedelic blur
    if fade > 0:
        fade_kernel_size = int(fade * 30) + 1
        if fade_kernel_size % 2 == 0:
            fade_kernel_size += 1
        mask = cv2.GaussianBlur(mask, (fade_kernel_size, fade_kernel_size), fade * 8)
    
    # Apply smoothing
    if smooth:
        mask = cv2.GaussianBlur(mask, (7, 7), 2.0)
    
    # Start with base image
    enhanced = image_float.copy()
    
    # Apply recursive detail enhancement
    for recursion in range(detail_recursion):
        recursion_intensity = intensity * (0.8 ** recursion)  # Diminishing intensity
        
        # Create detail enhancement with varying blur
        blur_sigma = 3.0 / max(recursion_intensity, 0.01)
        blurred = cv2.GaussianBlur(enhanced, (0, 0), blur_sigma)
        
        # Unsharp masking with bias
        detail_layer = (enhanced - blurred) * recursion_intensity
        detail_layer = detail_layer * bias + detail_layer * (1 - bias) * 0.3
        
        # Apply mask to detail layer
        if len(mask.shape) == 2:
            mask_3d = np.expand_dims(mask, axis=2)
        else:
            mask_3d = mask
        detail_layer = detail_layer * mask_3d
        
        enhanced = enhanced + detail_layer
    
    # Apply psychedelic color effects
    if color_shift > 0:
        enhanced = apply_color_shift(enhanced, color_shift, mask)
    
    if chromatic_aberration:
        enhanced = apply_chromatic_aberration(enhanced, intensity * 0.5)
    
    if saturation_boost > 0:
        enhanced = apply_saturation_boost(enhanced, saturation_boost, mask)
    
    if hue_rotation > 0:
        enhanced = apply_hue_rotation(enhanced, hue_rotation, Y, X)
    
    if contrast_waves > 0:
        enhanced = apply_contrast_waves(enhanced, contrast_waves, wave_frequency, Y, X)
    
    # Clip values to valid range
    enhanced = np.clip(enhanced, 0, 1)
    
    # Convert back to uint8
    enhanced = (enhanced * 255).astype(np.uint8)
    
    return enhanced

def create_fractal_mask(H, W, depth):
    """Create a fractal-based mask for detail enhancement"""
    mask = np.zeros((H, W), dtype=np.float32)
    
    for i in range(int(depth * 10)):
        freq = 2 ** (i * 0.5)
        y_coords = np.linspace(0, freq, H)
        x_coords = np.linspace(0, freq, W)
        Y, X = np.meshgrid(y_coords, x_coords, indexing='ij')
        
        fractal_layer = np.sin(Y * np.pi) * np.cos(X * np.pi)
        fractal_layer = (fractal_layer + 1) / 2
        
        mask += fractal_layer * (0.5 ** i)
    
    return np.clip(mask, 0, 1)

def apply_color_shift(image, shift_amount, mask):
    """Apply psychedelic color channel shifting"""
    if len(image.shape) != 3 or image.shape[2] != 3:
        return image
    
    shifted = image.copy()
    H, W = image.shape[:2]
    
    # Create different shifts for each channel
    shift_r = int(H * shift_amount * 0.02)
    shift_g = int(W * shift_amount * 0.015)
    shift_b = int(H * shift_amount * 0.025)
    
    # Apply shifts
    if shift_r != 0:
        shifted[:, :, 0] = np.roll(shifted[:, :, 0], shift_r, axis=0)
    if shift_g != 0:
        shifted[:, :, 1] = np.roll(shifted[:, :, 1], shift_g, axis=1)
    if shift_b != 0:
        shifted[:, :, 2] = np.roll(shifted[:, :, 2], shift_b, axis=0)
    
    # Blend with original based on mask
    mask_3d = np.expand_dims(mask, axis=2) if len(mask.shape) == 2 else mask
    return image * (1 - mask_3d * shift_amount) + shifted * mask_3d * shift_amount

def apply_chromatic_aberration(image, amount):
    """Apply chromatic aberration effect"""
    if len(image.shape) != 3 or image.shape[2] != 3:
        return image
    
    H, W = image.shape[:2]
    center_y, center_x = H // 2, W // 2
    
    # Create radial distortion
    y_coords = np.arange(H) - center_y
    x_coords = np.arange(W) - center_x
    Y, X = np.meshgrid(y_coords, x_coords, indexing='ij')
    
    radius = np.sqrt(Y**2 + X**2)
    max_radius = np.sqrt(center_y**2 + center_x**2)
    normalized_radius = radius / max_radius
    
    # Different distortion for each channel
    distortion_r = 1 + amount * 0.02 * normalized_radius
    distortion_b = 1 - amount * 0.02 * normalized_radius
    
    # Apply distortion (simplified version)
    result = image.copy()
    result[:, :, 0] *= distortion_r
    result[:, :, 2] *= distortion_b
    
    return np.clip(result, 0, 1)

def apply_saturation_boost(image, boost_amount, mask):
    """Apply saturation boost with mask"""
    # Convert to HSV
    hsv = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32) / 255.0
    
    # Boost saturation
    mask_3d = np.expand_dims(mask, axis=2) if len(mask.shape) == 2 else mask
    hsv[:, :, 1] = hsv[:, :, 1] * (1 + boost_amount * mask_3d[:, :, 0])
    hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 1)
    
    # Convert back to RGB
    rgb = cv2.cvtColor((hsv * 255).astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32) / 255.0
    return rgb

def apply_hue_rotation(image, rotation_amount, Y, X):
    """Apply dynamic hue rotation based on position"""
    # Convert to HSV
    hsv = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32) / 255.0
    
    # Create rotation pattern
    rotation_pattern = np.sin(Y * 4 * np.pi) * np.cos(X * 4 * np.pi) * rotation_amount
    
    # Apply hue rotation
    hsv[:, :, 0] = (hsv[:, :, 0] + rotation_pattern) % 1.0
    
    # Convert back to RGB
    rgb = cv2.cvtColor((hsv * 255).astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32) / 255.0
    return rgb

def apply_contrast_waves(image, wave_amount, frequency, Y, X):
    """Apply wave-based contrast modulation"""
    # Create wave pattern
    wave_pattern = np.sin(Y * frequency * 2 * np.pi) * np.cos(X * frequency * 2 * np.pi)
    wave_pattern = (wave_pattern + 1) / 2  # Normalize to 0-1
    
    # Apply contrast modulation
    contrast_factor = 1 + wave_amount * (wave_pattern - 0.5)
    contrast_factor = np.expand_dims(contrast_factor, axis=2)
    
    # Apply contrast
    enhanced = (image - 0.5) * contrast_factor + 0.5
    return np.clip(enhanced, 0, 1)


class PsychedelicDaemon:
    def __init__(self):
        self.enabled = False
        # Core psychedelic parameters
        self.intensity = 0.45
        self.color_shift = 0.3
        self.fractal_depth = 0.6
        
        # Disco diffusion inspired timing
        self.start = 0.15
        self.end = 0.9
        self.peak = 0.5
        
        # Psychedelic bias and flow
        self.bias = 0.85
        self.flow_multiplier = 1.2
        self.wave_frequency = 2.5
        
        # Offset controls
        self.start_offset = 0.1
        self.end_offset = -0.05
        
        # Exponential and fade controls
        self.exponent = 1.3
        self.fade = 0.15
        
        # Psychedelic modes
        self.mode = 'kaleidoscope'
        self.smooth = True
        self.chromatic_aberration = True
        
        # Advanced parameters
        self.saturation_boost = 0.4
        self.hue_rotation = 0.2
        self.contrast_waves = 0.3
        self.detail_recursion = 3
    
    def process(self, image):
        if self.enabled and image is not None:
            return enhance_psychedelic_details(
                image,
                intensity=self.intensity,
                color_shift=self.color_shift,
                fractal_depth=self.fractal_depth,
                start=self.start,
                end=self.end,
                peak=self.peak,
                bias=self.bias,
                flow_multiplier=self.flow_multiplier,
                wave_frequency=self.wave_frequency,
                start_offset=self.start_offset,
                end_offset=self.end_offset,
                exponent=self.exponent,
                fade=self.fade,
                mode=self.mode,
                smooth=self.smooth,
                chromatic_aberration=self.chromatic_aberration,
                saturation_boost=self.saturation_boost,
                hue_rotation=self.hue_rotation,
                contrast_waves=self.contrast_waves,
                detail_recursion=self.detail_recursion
            )
        return image
    
    def set_disco_preset(self):
        """Preset optimized for disco diffusion style effects"""
        self.intensity = 0.6
        self.color_shift = 0.5
        self.fractal_depth = 0.8
        self.bias = 0.9
        self.wave_frequency = 3.0
        self.saturation_boost = 0.6
        self.hue_rotation = 0.4
        self.mode = 'kaleidoscope'
        
    def set_fluid_preset(self):
        """Preset for fluid, flowing psychedelic effects"""
        self.intensity = 0.4
        self.color_shift = 0.25
        self.fractal_depth = 0.4
        self.bias = 0.75
        self.wave_frequency = 1.8
        self.flow_multiplier = 1.5
        self.mode = 'fluid'
        
    def set_fractal_preset(self):
        """Preset for fractal-heavy psychedelic effects"""
        self.intensity = 0.5
        self.fractal_depth = 0.9
        self.detail_recursion = 5
        self.bias = 0.8
        self.mode = 'fractal'

# Create a global instance
psychedelic_daemon = PsychedelicDaemon()

def update_psychedelic_daemon_settings(enabled, intensity, color_shift, fractal_depth, start, end, peak,
                                     bias, flow_multiplier, wave_frequency, saturation_boost, hue_rotation,
                                     contrast_waves, detail_recursion, chromatic_aberration, smooth, fade, mode):
    """Update the psychedelic daemon settings"""
    psychedelic_daemon.enabled = enabled
    psychedelic_daemon.intensity = intensity
    psychedelic_daemon.color_shift = color_shift
    psychedelic_daemon.fractal_depth = fractal_depth
    psychedelic_daemon.start = start
    psychedelic_daemon.end = end
    psychedelic_daemon.peak = peak
    psychedelic_daemon.bias = bias
    psychedelic_daemon.flow_multiplier = flow_multiplier
    psychedelic_daemon.wave_frequency = wave_frequency
    # Note: start_offset, end_offset, exponent not used in psychedelic daemon
    # These are detail_daemon specific parameters
    psychedelic_daemon.fade = fade
    psychedelic_daemon.mode = mode
    psychedelic_daemon.smooth = smooth
    psychedelic_daemon.chromatic_aberration = chromatic_aberration
    psychedelic_daemon.saturation_boost = saturation_boost
    psychedelic_daemon.hue_rotation = hue_rotation
    psychedelic_daemon.contrast_waves = contrast_waves
    psychedelic_daemon.detail_recursion = detail_recursion
    
    return f"Psychedelic Daemon: {'Enabled' if enabled else 'Disabled'}, Mode: {mode}, Intensity: {intensity}"

def apply_preset(preset_name):
    """Apply a preset configuration"""
    if preset_name == "disco":
        psychedelic_daemon.set_disco_preset()
    elif preset_name == "fluid":
        psychedelic_daemon.set_fluid_preset()
    elif preset_name == "fractal":
        psychedelic_daemon.set_fractal_preset()
    
    return f"Applied {preset_name} preset to Psychedelic Daemon"
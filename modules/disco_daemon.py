import torch
import torch.nn.functional as F
import numpy as np
import math
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

def apply_disco_distortion(latent_samples, disco_scale=5.0, distortion_type='psychedelic', 
                          intensity_multiplier=1.0, test_mode=False):
    """
    Apply disco diffusion-style distortions to latent space.
    
    Args:
        latent_samples: Input latent tensor [B,C,H,W]
        disco_scale: Overall strength of the distortion
        distortion_type: Type of distortion ('psychedelic', 'fractal', 'kaleidoscope', etc.)
        intensity_multiplier: Additional multiplier for intensity
        test_mode: If True, applies simple inversion for testing
    
    Returns:
        Distorted latent tensor
    """
    if test_mode:
        print(f"[Disco] TEST MODE: Inverting latents")
        return -latent_samples
        
    if disco_scale <= 0:
        return latent_samples
        
    try:
        device = latent_samples.device
        batch_size, channels, height, width = latent_samples.shape
        
        # Create coordinate grids for spatial transformations
        y_coords = torch.linspace(-1, 1, height, device=device)
        x_coords = torch.linspace(-1, 1, width, device=device)
        y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing='ij')
        
        # Scale the distortion
        base_scale = disco_scale * intensity_multiplier * 0.1  # More reasonable scaling
        
        # Apply different distortion types
        if distortion_type == 'psychedelic':
            # Psychedelic swirl and wave distortions
            radius = torch.sqrt(x_grid**2 + y_grid**2)
            angle = torch.atan2(y_grid, x_grid)
            
            # Swirl distortion
            swirl_strength = base_scale * 2.0
            new_angle = angle + swirl_strength * torch.exp(-radius * 2.0)
            
            # Wave distortions
            wave_freq = base_scale * 8.0
            wave_amp = base_scale * 0.3
            x_wave = x_grid + wave_amp * torch.sin(wave_freq * y_grid)
            y_wave = y_grid + wave_amp * torch.cos(wave_freq * x_grid)
            
            # Combine
            new_x = radius * torch.cos(new_angle) + x_wave * 0.5
            new_y = radius * torch.sin(new_angle) + y_wave * 0.5
            
        elif distortion_type == 'fractal':
            # Fractal-like recursive distortions
            scale1 = base_scale * 2.0
            scale2 = base_scale * 1.5
            scale3 = base_scale * 1.0
            
            new_x = x_grid + scale1 * torch.sin(3 * x_grid) * torch.cos(2 * y_grid)
            new_y = y_grid + scale1 * torch.cos(3 * y_grid) * torch.sin(2 * x_grid)
            
            # Add finer details
            new_x += scale2 * torch.sin(7 * new_x) * torch.cos(5 * new_y)
            new_y += scale2 * torch.cos(7 * new_y) * torch.sin(5 * new_x)
            
            new_x += scale3 * torch.sin(13 * new_x) * torch.cos(11 * new_y)
            new_y += scale3 * torch.cos(13 * new_y) * torch.sin(11 * new_x)
            
        elif distortion_type == 'kaleidoscope':
            # Kaleidoscope-like symmetrical distortions
            radius = torch.sqrt(x_grid**2 + y_grid**2)
            angle = torch.atan2(y_grid, x_grid)
            
            # Create kaleidoscope effect
            n_mirrors = 6
            mirror_angle = 2 * math.pi / n_mirrors
            folded_angle = torch.abs((angle % mirror_angle) - mirror_angle/2)
            
            # Radial modulation
            radial_mod = 1 + base_scale * 0.5 * torch.sin(radius * 6)
            
            new_x = radius * torch.cos(folded_angle) * radial_mod
            new_y = radius * torch.sin(folded_angle) * radial_mod
            
        elif distortion_type == 'wave':
            # Simple wave distortions
            wave_freq = base_scale * 6.0
            wave_amp = base_scale * 0.4
            
            new_x = x_grid + wave_amp * torch.sin(wave_freq * y_grid)
            new_y = y_grid + wave_amp * torch.sin(wave_freq * x_grid)
            
        else:  # default - psychedelic
            radius = torch.sqrt(x_grid**2 + y_grid**2)
            angle = torch.atan2(y_grid, x_grid)
            
            swirl_strength = base_scale * 2.0
            new_angle = angle + swirl_strength * torch.exp(-radius * 2.0)
            
            wave_amp = base_scale * 0.3
            wave_freq = base_scale * 6.0
            new_x = radius * torch.cos(new_angle) + wave_amp * torch.sin(wave_freq * y_grid)
            new_y = radius * torch.sin(new_angle) + wave_amp * torch.cos(wave_freq * x_grid)
        
        # Clamp coordinates to valid range
        new_x = torch.clamp(new_x, -1.5, 1.5)
        new_y = torch.clamp(new_y, -1.5, 1.5)
        
        # Apply grid sampling
        grid = torch.stack((new_x, new_y), dim=-1)
        grid = grid.unsqueeze(0).repeat(batch_size, 1, 1, 1)
        
        # Sample from the original latent using the distortion grid
        result = F.grid_sample(
            latent_samples, 
            grid, 
            mode='bilinear', 
            padding_mode='reflection',
            align_corners=False
        )
        
        # Blend with original
        blend_factor = min(0.6, intensity_multiplier * 0.4)
        result = latent_samples * (1.0 - blend_factor) + result * blend_factor
        
        print(f"[Disco] Applied {distortion_type} distortion (scale={disco_scale:.1f}, blend={blend_factor:.2f})")
        return result
        
    except Exception as e:
        logger.error(f"Disco distortion failed: {e}")
        return latent_samples

class DiscoDaemon:
    """Disco Diffusion daemon for applying psychedelic distortions to latents."""
    
    def __init__(self):
        self.enabled = False
        self.disco_scale = 5.0
        self.distortion_type = 'psychedelic'
        self.intensity_multiplier = 1.0
        self.test_mode = False
        
        # Animation parameters
        self.animation_mode = 'none'
        self.zoom_factor = 1.02
        self.rotation_speed = 0.1
        self.translation_x = 0.0
        self.translation_y = 0.0
        
        # Visual parameters
        self.color_coherence = 0.5
        self.saturation_boost = 1.2
        self.contrast_boost = 1.1
        self.symmetry_mode = 'none'
        self.fractal_octaves = 3
    
    def process_latent(self, latent_samples):
        """Process latent samples with disco distortion if enabled."""
        if not self.enabled or latent_samples is None:
            return latent_samples
            
        return apply_disco_distortion(
            latent_samples,
            disco_scale=self.disco_scale,
            distortion_type=self.distortion_type,
            intensity_multiplier=self.intensity_multiplier,
            test_mode=self.test_mode
        )
    
    def update_settings(self, **kwargs):
        """Update disco daemon settings."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        
        status = f"Disco Daemon: {'Enabled' if self.enabled else 'Disabled'}"
        if self.enabled:
            status += f", Scale: {self.disco_scale}, Type: {self.distortion_type}"
        
        return status

# Create a global instance
disco_daemon = DiscoDaemon()

def update_disco_daemon_settings(enabled, disco_scale, distortion_type, intensity_multiplier=1.0,
                                animation_mode='none', zoom_factor=1.02, rotation_speed=0.1,
                                translation_x=0.0, translation_y=0.0, color_coherence=0.5,
                                saturation_boost=1.2, contrast_boost=1.1, symmetry_mode='none',
                                fractal_octaves=3, test_mode=False):
    """Update the disco daemon settings."""
    return disco_daemon.update_settings(
        enabled=enabled,
        disco_scale=disco_scale,
        distortion_type=distortion_type,
        intensity_multiplier=intensity_multiplier,
        animation_mode=animation_mode,
        zoom_factor=zoom_factor,
        rotation_speed=rotation_speed,
        translation_x=translation_x,
        translation_y=translation_y,
        color_coherence=color_coherence,
        saturation_boost=saturation_boost,
        contrast_boost=contrast_boost,
        symmetry_mode=symmetry_mode,
        fractal_octaves=fractal_octaves,
        test_mode=test_mode
    )
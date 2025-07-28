# Disco Diffusion Extension for Fooocus
# Generates psychedelic, fractal-like images inspired by Disco Diffusion

import torch
import torch.nn.functional as F
import numpy as np
import math
import random
from typing import Optional, List, Tuple
import logging

logger = logging.getLogger(__name__)

class DiscoTransforms:
    """Collection of transforms for creating psychedelic effects"""
    
    @staticmethod
    def spherical_distortion(x, strength=0.5):
        """Apply spherical distortion effect"""
        b, c, h, w = x.shape
        device = x.device
        
        # Create coordinate grids
        y_coords = torch.linspace(-1, 1, h, device=device)
        x_coords = torch.linspace(-1, 1, w, device=device)
        yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')
        
        # Calculate distance from center
        r = torch.sqrt(xx**2 + yy**2)
        
        # Apply spherical distortion
        r_distorted = r * (1 + strength * r**2)
        
        # Convert back to coordinates
        mask = r > 0
        scale = torch.where(mask, r_distorted / r, torch.ones_like(r))
        
        xx_new = xx * scale
        yy_new = yy * scale
        
        # Create sampling grid
        grid = torch.stack([xx_new, yy_new], dim=-1).unsqueeze(0).repeat(b, 1, 1, 1)
        
        # Sample from original tensor
        return F.grid_sample(x, grid, mode='bilinear', padding_mode='reflection', align_corners=True)
    
    @staticmethod
    def kaleidoscope_effect(x, segments=6, rotation=0.0):
        """Create kaleidoscope effect"""
        b, c, h, w = x.shape
        device = x.device
        
        # Create coordinate grids
        y_coords = torch.linspace(-1, 1, h, device=device)
        x_coords = torch.linspace(-1, 1, w, device=device)
        yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')
        
        # Convert to polar coordinates
        r = torch.sqrt(xx**2 + yy**2)
        theta = torch.atan2(yy, xx) + rotation
        
        # Apply kaleidoscope effect
        segment_angle = 2 * math.pi / segments
        theta_mod = torch.fmod(theta, segment_angle)
        theta_reflected = torch.where(
            torch.fmod(torch.floor(theta / segment_angle), 2) == 0,
            theta_mod,
            segment_angle - theta_mod
        )
        
        # Convert back to cartesian
        xx_new = r * torch.cos(theta_reflected)
        yy_new = r * torch.sin(theta_reflected)
        
        # Create sampling grid
        grid = torch.stack([xx_new, yy_new], dim=-1).unsqueeze(0).repeat(b, 1, 1, 1)
        
        return F.grid_sample(x, grid, mode='bilinear', padding_mode='reflection', align_corners=True)
    
    @staticmethod
    def fractal_zoom(x, zoom_factor=1.2, center_x=0.0, center_y=0.0):
        """Apply fractal zoom effect"""
        b, c, h, w = x.shape
        device = x.device
        
        # Create coordinate grids
        y_coords = torch.linspace(-1, 1, h, device=device)
        x_coords = torch.linspace(-1, 1, w, device=device)
        yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')
        
        # Apply zoom and translation
        xx_new = (xx - center_x) / zoom_factor + center_x
        yy_new = (yy - center_y) / zoom_factor + center_y
        
        # Create sampling grid
        grid = torch.stack([xx_new, yy_new], dim=-1).unsqueeze(0).repeat(b, 1, 1, 1)
        
        return F.grid_sample(x, grid, mode='bilinear', padding_mode='reflection', align_corners=True)
    
    @staticmethod
    def color_shift(x, hue_shift=0.0, saturation_mult=1.0, brightness_mult=1.0):
        """Apply color transformations"""
        # Convert RGB to HSV
        r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
        
        max_val, _ = torch.max(x, dim=1, keepdim=True)
        min_val, _ = torch.min(x, dim=1, keepdim=True)
        delta = max_val - min_val
        
        # Hue calculation
        hue = torch.zeros_like(max_val)
        mask = delta > 0
        
        r_mask = (max_val == r) & mask
        g_mask = (max_val == g) & mask
        b_mask = (max_val == b) & mask
        
        hue[r_mask] = ((g - b) / delta)[r_mask] % 6
        hue[g_mask] = ((b - r) / delta + 2)[g_mask]
        hue[b_mask] = ((r - g) / delta + 4)[b_mask]
        hue = hue / 6
        
        # Saturation
        saturation = torch.where(max_val > 0, delta / max_val, torch.zeros_like(max_val))
        
        # Value (brightness)
        value = max_val
        
        # Apply transformations
        hue = (hue + hue_shift) % 1.0
        saturation = torch.clamp(saturation * saturation_mult, 0, 1)
        value = torch.clamp(value * brightness_mult, 0, 1)
        
        # Convert back to RGB
        c = value * saturation
        x_prime = (hue * 6) % 6
        x_val = c * (1 - torch.abs(x_prime % 2 - 1))
        m = value - c
        
        r_new = torch.zeros_like(hue)
        g_new = torch.zeros_like(hue)
        b_new = torch.zeros_like(hue)
        
        mask0 = (x_prime >= 0) & (x_prime < 1)
        mask1 = (x_prime >= 1) & (x_prime < 2)
        mask2 = (x_prime >= 2) & (x_prime < 3)
        mask3 = (x_prime >= 3) & (x_prime < 4)
        mask4 = (x_prime >= 4) & (x_prime < 5)
        mask5 = (x_prime >= 5) & (x_prime < 6)
        
        r_new[mask0] = c[mask0]
        g_new[mask0] = x_val[mask0]
        
        r_new[mask1] = x_val[mask1]
        g_new[mask1] = c[mask1]
        
        g_new[mask2] = c[mask2]
        b_new[mask2] = x_val[mask2]
        
        g_new[mask3] = x_val[mask3]
        b_new[mask3] = c[mask3]
        
        r_new[mask4] = x_val[mask4]
        b_new[mask4] = c[mask4]
        
        r_new[mask5] = c[mask5]
        b_new[mask5] = x_val[mask5]
        
        rgb_new = torch.cat([r_new + m, g_new + m, b_new + m], dim=1)
        
        return torch.clamp(rgb_new, 0, 1)

class DiscoSampler:
    """
    Disco Diffusion sampler that creates psychedelic effects during generation
    """
    
    def __init__(self, 
                 disco_enabled=False,
                 disco_scale=0.5,
                 disco_steps_schedule=None,
                 disco_transforms=None,
                 disco_seed=None,
                 disco_animation_mode='none',
                 disco_zoom_factor=1.02,
                 disco_rotation_speed=0.1,
                 disco_translation_x=0.0,
                 disco_translation_y=0.0,
                 disco_color_coherence=0.5,
                 disco_saturation_boost=1.2,
                 disco_contrast_boost=1.1,
                 disco_symmetry_mode='none',
                 disco_fractal_octaves=3,
                 disco_noise_schedule='linear'):
        
        self.disco_enabled = disco_enabled
        self.disco_scale = disco_scale
        self.disco_steps_schedule = disco_steps_schedule or [0.2, 0.4, 0.6, 0.8]
        self.disco_transforms = disco_transforms or ['spherical', 'kaleidoscope', 'color_shift']
        self.disco_seed = disco_seed
        self.disco_animation_mode = disco_animation_mode
        self.disco_zoom_factor = disco_zoom_factor
        self.disco_rotation_speed = disco_rotation_speed
        self.disco_translation_x = disco_translation_x
        self.disco_translation_y = disco_translation_y
        self.disco_color_coherence = disco_color_coherence
        self.disco_saturation_boost = disco_saturation_boost
        self.disco_contrast_boost = disco_contrast_boost
        self.disco_symmetry_mode = disco_symmetry_mode
        self.disco_fractal_octaves = disco_fractal_octaves
        self.disco_noise_schedule = disco_noise_schedule
        
        self.step_count = 0
        self.transform_state = {}
        self.original_sampling_function = None
        self.is_active = False
        
        # Initialize random state
        if self.disco_seed is not None:
            self.rng = random.Random(self.disco_seed)
        else:
            self.rng = random.Random()
    
    def activate(self, unet):
        """Activate Disco Diffusion by patching the sampling function"""
        if self.is_active or not self.disco_enabled:
            return
        
        print(f"[Disco] Activating Disco Diffusion with scale {self.disco_scale}")
        
        try:
            import ldm_patched.modules.samplers as samplers
            
            if not hasattr(self, '_original_sampling_function'):
                self._original_sampling_function = samplers.sampling_function
            
            samplers.sampling_function = self._create_disco_sampling_function(self._original_sampling_function)
            
            self.unet = unet
            self.is_active = True
            self.step_count = 0
            print("[Disco] Successfully patched sampling function")
            
        except Exception as e:
            print(f"[Disco] Failed to patch sampling function: {e}")
            return
    
    def deactivate(self):
        """Deactivate Disco Diffusion"""
        if not self.is_active:
            return
        
        print("[Disco] Deactivating Disco Diffusion")
        
        try:
            import ldm_patched.modules.samplers as samplers
            if hasattr(self, '_original_sampling_function'):
                samplers.sampling_function = self._original_sampling_function
                print("[Disco] Successfully restored original sampling function")
        except Exception as e:
            print(f"[Disco] Failed to restore sampling function: {e}")
        
        self.is_active = False
        self.step_count = 0
    
    def _create_disco_sampling_function(self, original_sampling_function):
        """Create Disco-modified sampling function"""
        def disco_sampling_function(model, x, timestep, uncond, cond, cond_scale, model_options={}, seed=None):
            try:
                # Get original prediction
                noise_pred = original_sampling_function(model, x, timestep, uncond, cond, cond_scale, model_options, seed)
                
                # Apply disco effects if enabled and at right step
                if self.disco_enabled and self._should_apply_disco_at_step():
                    noise_pred = self._apply_disco_effects(noise_pred, timestep)
                
                self.step_count += 1
                return noise_pred
                
            except Exception as e:
                print(f"[Disco] Error in disco sampling, falling back to original: {e}")
                return original_sampling_function(model, x, timestep, uncond, cond, cond_scale, model_options, seed)
        
        return disco_sampling_function
    
    def _should_apply_disco_at_step(self):
        """Determine if disco effects should be applied at current step"""
        if not self.disco_steps_schedule:
            return True
        
        # Convert step count to progress (0-1)
        # This is approximate since we don't know total steps
        progress = min(self.step_count / 50.0, 1.0)  # Assume ~50 steps max
        
        # Check if current progress matches any scheduled step
        for scheduled_step in self.disco_steps_schedule:
            if abs(progress - scheduled_step) < 0.05:  # 5% tolerance
                return True
        
        return False
    
    def _apply_disco_effects(self, x, timestep):
        """Apply disco diffusion effects to the tensor"""
        if x.dim() != 4 or x.shape[1] < 3:
            return x
        
        try:
            # Get current step progress
            progress = min(self.step_count / 50.0, 1.0)
            
            # Apply transforms based on configuration
            result = x.clone()
            
            if 'spherical' in self.disco_transforms:
                strength = self.disco_scale * (0.3 + 0.7 * progress)
                result = DiscoTransforms.spherical_distortion(result, strength)
            
            if 'kaleidoscope' in self.disco_transforms:
                segments = 6 + int(4 * progress)  # 6-10 segments
                rotation = self.disco_rotation_speed * self.step_count * 0.1
                result = DiscoTransforms.kaleidoscope_effect(result, segments, rotation)
            
            if 'fractal_zoom' in self.disco_transforms:
                zoom = 1.0 + (self.disco_zoom_factor - 1.0) * progress
                center_x = self.disco_translation_x * math.sin(self.step_count * 0.05)
                center_y = self.disco_translation_y * math.cos(self.step_count * 0.05)
                result = DiscoTransforms.fractal_zoom(result, zoom, center_x, center_y)
            
            if 'color_shift' in self.disco_transforms:
                hue_shift = 0.1 * math.sin(self.step_count * 0.1) * self.disco_scale
                sat_mult = self.disco_saturation_boost
                bright_mult = self.disco_contrast_boost
                result = DiscoTransforms.color_shift(result, hue_shift, sat_mult, bright_mult)
            
            # Apply symmetry if enabled
            if self.disco_symmetry_mode == 'horizontal':
                result = self._apply_horizontal_symmetry(result)
            elif self.disco_symmetry_mode == 'vertical':
                result = self._apply_vertical_symmetry(result)
            elif self.disco_symmetry_mode == 'radial':
                result = self._apply_radial_symmetry(result)
            
            # Blend with original based on color coherence
            coherence = self.disco_color_coherence
            result = coherence * x + (1 - coherence) * result
            
            return result
            
        except Exception as e:
            logger.warning(f"Disco effect application failed: {e}")
            return x
    
    def _apply_horizontal_symmetry(self, x):
        """Apply horizontal symmetry"""
        b, c, h, w = x.shape
        left_half = x[:, :, :, :w//2]
        right_half = torch.flip(left_half, dims=[3])
        return torch.cat([left_half, right_half], dim=3)
    
    def _apply_vertical_symmetry(self, x):
        """Apply vertical symmetry"""
        b, c, h, w = x.shape
        top_half = x[:, :, :h//2, :]
        bottom_half = torch.flip(top_half, dims=[2])
        return torch.cat([top_half, bottom_half], dim=2)
    
    def _apply_radial_symmetry(self, x):
        """Apply radial symmetry"""
        # Simple radial symmetry by rotating and averaging
        rotated = torch.rot90(x, k=2, dims=[2, 3])
        return (x + rotated) * 0.5

# Global disco sampler instance
disco_sampler = DiscoSampler()

def create_disco_noise_schedule(schedule_type='linear', steps=50):
    """Create noise schedule for disco effects"""
    if schedule_type == 'linear':
        return np.linspace(0, 1, steps)
    elif schedule_type == 'cosine':
        return 0.5 * (1 - np.cos(np.linspace(0, np.pi, steps)))
    elif schedule_type == 'exponential':
        return np.exp(np.linspace(0, 2, steps)) / np.exp(2)
    else:
        return np.linspace(0, 1, steps)

def get_disco_presets():
    """Get predefined disco effect presets"""
    return {
        'psychedelic': {
            'disco_scale': 0.7,
            'disco_transforms': ['spherical', 'kaleidoscope', 'color_shift'],
            'disco_saturation_boost': 1.5,
            'disco_contrast_boost': 1.3,
            'disco_rotation_speed': 0.2,
            'disco_symmetry_mode': 'none'
        },
        'fractal': {
            'disco_scale': 0.5,
            'disco_transforms': ['fractal_zoom', 'color_shift'],
            'disco_zoom_factor': 1.05,
            'disco_fractal_octaves': 4,
            'disco_saturation_boost': 1.2,
            'disco_symmetry_mode': 'radial'
        },
        'kaleidoscope': {
            'disco_scale': 0.8,
            'disco_transforms': ['kaleidoscope', 'color_shift'],
            'disco_rotation_speed': 0.3,
            'disco_saturation_boost': 1.4,
            'disco_symmetry_mode': 'radial'
        },
        'dreamy': {
            'disco_scale': 0.3,
            'disco_transforms': ['spherical', 'color_shift'],
            'disco_saturation_boost': 1.1,
            'disco_contrast_boost': 1.05,
            'disco_color_coherence': 0.8
        }
    }
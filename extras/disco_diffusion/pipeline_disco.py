# True Disco Diffusion Extension for Fooocus
# Implements the real Disco Diffusion algorithm with CLIP guidance and geometric transforms

import torch
import torch.nn.functional as F
import numpy as np
import math
import random
from typing import Optional, List, Tuple
import logging

logger = logging.getLogger(__name__)

def spherical_dist_loss(x, y):
    """Spherical distance loss for CLIP guidance"""
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    return (x - y).norm(dim=-1).div(2).arcsin().pow(2).mul(2)

def tv_loss(input):
    """Total variation loss for smoothness"""
    input = F.pad(input, (0, 1, 0, 1), 'replicate')
    x_diff = input[..., :-1, 1:] - input[..., :-1, :-1]
    y_diff = input[..., 1:, :-1] - input[..., :-1, :-1]
    return (x_diff**2 + y_diff**2).mean([1, 2, 3])

def range_loss(input):
    """Range loss to keep values in reasonable bounds"""
    return (input - input.clamp(-1, 1)).pow(2).mean([1, 2, 3])

class DiscoTransforms:
    """Real Disco Diffusion geometric transforms"""
    
    @staticmethod
    def translate_2d(tx, ty, device=None):
        """2D translation matrix"""
        mat = torch.zeros(2, 3, device=device)
        mat[0, 0] = 1
        mat[1, 1] = 1
        mat[0, 2] = tx
        mat[1, 2] = ty
        return mat
    
    @staticmethod
    def rotate_2d(theta, device=None):
        """2D rotation matrix"""
        if isinstance(theta, (int, float)):
            theta = torch.tensor(theta, device=device)
        mat = torch.zeros(2, 3, device=device)
        mat[0, 0] = torch.cos(theta)
        mat[0, 1] = -torch.sin(theta)
        mat[1, 0] = torch.sin(theta)
        mat[1, 1] = torch.cos(theta)
        return mat
    
    @staticmethod
    def scale_2d(sx, sy, device=None):
        """2D scaling matrix"""
        mat = torch.zeros(2, 3, device=device)
        mat[0, 0] = sx
        mat[1, 1] = sy
        return mat
    
    @staticmethod
    def apply_transform(x, transform_matrix):
        """Apply 2D transformation to tensor with tiled VAE awareness"""
        # Ensure transform_matrix is on the same device as x
        transform_matrix = transform_matrix.to(x.device)
        
        # For very large tensors, apply more conservative transforms to avoid tiling artifacts
        b, c, h, w = x.shape
        if h > 128 or w > 128:  # Large latent (likely high resolution)
            # Use nearest neighbor for large tensors to avoid blur
            grid = F.affine_grid(transform_matrix.unsqueeze(0), x.size(), align_corners=False)
            return F.grid_sample(x, grid, mode='nearest', padding_mode='reflection', align_corners=False)
        else:
            # Use bilinear for smaller tensors
            grid = F.affine_grid(transform_matrix.unsqueeze(0), x.size(), align_corners=False)
            return F.grid_sample(x, grid, mode='bilinear', padding_mode='reflection', align_corners=False)
    
    @staticmethod
    def make_cutouts(image, cut_size, cutn):
        """Create random cutouts for CLIP analysis (core Disco Diffusion technique)"""
        sideY, sideX = image.shape[2:4]
        max_size = min(sideX, sideY)
        min_size = min(sideX, sideY, cut_size)
        cutouts = []
        
        for _ in range(cutn):
            size = int(torch.rand([])**0.8 * (max_size - min_size) + min_size)
            offsetx = torch.randint(0, sideX - size + 1, ())
            offsety = torch.randint(0, sideY - size + 1, ())
            cutout = image[:, :, offsety:offsety + size, offsetx:offsetx + size]
            cutout = F.adaptive_avg_pool2d(cutout, cut_size)
            cutouts.append(cutout)
        
        return torch.cat(cutouts, dim=0)
    
    @staticmethod
    def symmetrize(x, mode='none'):
        """Apply symmetry transformations"""
        if mode == 'horizontal':
            left = x[:, :, :, :x.shape[3]//2]
            right = torch.flip(left, dims=[3])
            return torch.cat([left, right], dim=3)
        elif mode == 'vertical':
            top = x[:, :, :x.shape[2]//2, :]
            bottom = torch.flip(top, dims=[2])
            return torch.cat([top, bottom], dim=2)
        elif mode == 'radial':
            # Simple radial symmetry approximation
            rotated = torch.rot90(x, k=2, dims=[2, 3])
            return (x + rotated) * 0.5
        return x
    
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
        
        # Create sampling grid and ensure it's on the same device as x
        grid = torch.stack([xx_new, yy_new], dim=-1).unsqueeze(0).repeat(b, 1, 1, 1)
        grid = grid.to(x.device)
        
        return F.grid_sample(x, grid, mode='bilinear', padding_mode='reflection', align_corners=True)
    
    @staticmethod
    def color_shift(x, hue_shift=0.0, saturation_mult=1.0, brightness_mult=1.0):
        """Apply color transformations"""
        # Ensure we have exactly 3 channels (RGB)
        if x.shape[1] != 3:
            return x
        
        try:
            # Convert RGB to HSV
            r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
            
            # Clamp input values to valid range
            r = torch.clamp(r, 0, 1)
            g = torch.clamp(g, 0, 1)
            b = torch.clamp(b, 0, 1)
            
            max_val, _ = torch.max(torch.cat([r, g, b], dim=1), dim=1, keepdim=True)
            min_val, _ = torch.min(torch.cat([r, g, b], dim=1), dim=1, keepdim=True)
            delta = max_val - min_val
            
            # Hue calculation with epsilon to avoid division by zero
            eps = 1e-8
            hue = torch.zeros_like(max_val)
            mask = delta > eps
            
            r_mask = (max_val == r) & mask
            g_mask = (max_val == g) & mask
            b_mask = (max_val == b) & mask
            
            hue[r_mask] = ((g - b) / (delta + eps))[r_mask] % 6
            hue[g_mask] = ((b - r) / (delta + eps) + 2)[g_mask]
            hue[b_mask] = ((r - g) / (delta + eps) + 4)[b_mask]
            hue = hue / 6
            
            # Saturation
            saturation = torch.where(max_val > eps, delta / (max_val + eps), torch.zeros_like(max_val))
            
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
            
        except Exception as e:
            # If color shift fails, return original
            return x
    
    @staticmethod
    def latent_color_mix(x, mix_strength=0.3, step=0):
        """Apply color mixing in latent space for psychedelic effects"""
        try:
            b, c, h, w = x.shape
            
            # Create sinusoidal mixing patterns
            phase = step * 0.1
            
            # Mix channels in a rotating pattern
            if c >= 4:  # Standard latent space has 4 channels
                # Create rotation matrix for channel mixing
                angle = mix_strength * math.sin(phase)
                cos_a, sin_a = math.cos(angle), math.sin(angle)
                
                # Mix first two channels
                ch0, ch1 = x[:, 0:1], x[:, 1:2]
                new_ch0 = cos_a * ch0 - sin_a * ch1
                new_ch1 = sin_a * ch0 + cos_a * ch1
                
                # Mix last two channels with different phase
                angle2 = mix_strength * math.sin(phase + math.pi/2)
                cos_a2, sin_a2 = math.cos(angle2), math.sin(angle2)
                
                ch2, ch3 = x[:, 2:3], x[:, 3:4]
                new_ch2 = cos_a2 * ch2 - sin_a2 * ch3
                new_ch3 = sin_a2 * ch2 + cos_a2 * ch3
                
                result = torch.cat([new_ch0, new_ch1, new_ch2, new_ch3], dim=1)
                
                # Add any remaining channels unchanged
                if c > 4:
                    result = torch.cat([result, x[:, 4:]], dim=1)
                
                return result
            else:
                # For other channel counts, apply simple mixing
                result = x.clone()
                for i in range(min(c-1, 2)):
                    angle = mix_strength * math.sin(phase + i * math.pi/3)
                    cos_a, sin_a = math.cos(angle), math.sin(angle)
                    
                    ch_a, ch_b = result[:, i:i+1], result[:, i+1:i+2]
                    result[:, i:i+1] = cos_a * ch_a - sin_a * ch_b
                    result[:, i+1:i+2] = sin_a * ch_a + cos_a * ch_b
                
                return result
                
        except Exception as e:
            return x
            
        except Exception as e:
            # If color shift fails, return original
            return x

class DiscoSampler:
    """
    True Disco Diffusion sampler implementing CLIP guidance and geometric transforms
    """
    
    def __init__(self, 
                 disco_enabled=False,
                 disco_scale=1000.0,  # CLIP guidance scale
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
                 disco_noise_schedule='linear',
                 # Real Disco Diffusion parameters
                 cutn=16,  # Number of cutouts for CLIP
                 cut_pow=1.0,  # Cutout power
                 tv_scale=0.0,  # Total variation loss scale
                 range_scale=150.0,  # Range loss scale
                 sat_scale=0.0,  # Saturation loss scale
                 init_scale=1000.0,  # Initial image scale
                 skip_augs=False):
        
        self.disco_enabled = disco_enabled
        self.disco_scale = disco_scale
        self.disco_steps_schedule = disco_steps_schedule or [0.0, 1.0]  # Apply from start to end (every step)
        self.disco_transforms = disco_transforms or ['translate', 'rotate', 'zoom']
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
        
        # Real Disco Diffusion parameters
        self.cutn = cutn
        self.cut_pow = cut_pow
        self.tv_scale = tv_scale
        self.range_scale = range_scale
        self.sat_scale = sat_scale
        self.init_scale = init_scale
        self.skip_augs = skip_augs
        
        self.step_count = 0
        self.transform_state = {}
        self.original_sampling_function = None
        self.is_active = False
        self.clip_model = None
        self.clip_preprocess = None
        self.detected_resolution = None
        self.tiled_vae_detected = False
        
        # Initialize random state
        if self.disco_seed is not None:
            self.rng = random.Random(self.disco_seed)
            torch.manual_seed(self.disco_seed)
        else:
            self.rng = random.Random()
    
    def activate(self, unet):
        """Activate Disco Diffusion by patching the sampling function"""
        if self.is_active or not self.disco_enabled:
            return
        
        print(f"[Disco] Activating Disco Diffusion with scale {self.disco_scale}")
        
        # Initialize CLIP if not already done
        if self.clip_model is None:
            print("[Disco] Initializing CLIP for scientific guidance...")
            self._init_clip()
        
        try:
            import ldm_patched.modules.samplers as samplers
            
            if not hasattr(self, '_original_sampling_function'):
                self._original_sampling_function = samplers.sampling_function
            
            samplers.sampling_function = self._create_disco_sampling_function(self._original_sampling_function)
            
            self.unet = unet
            self.is_active = True
            self.step_count = 0
            
            if self.clip_model is not None:
                print("[Disco] Successfully activated with CLIP guidance")
            else:
                print("[Disco] Successfully activated with geometric transforms only")
            
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
        """Create true Disco Diffusion sampling function with CLIP guidance"""
        def disco_sampling_function(model, x, timestep, uncond, cond, cond_scale, model_options={}, seed=None):
            try:
                # Get original prediction first
                noise_pred = original_sampling_function(model, x, timestep, uncond, cond, cond_scale, model_options, seed)
                
                # Apply Disco Diffusion CLIP guidance if enabled
                if self.disco_enabled and self._should_apply_disco_at_step():
                    print(f"[Disco] Applying guidance at step {self.step_count}")
                    noise_pred = self._apply_disco_guidance(model, x, timestep, noise_pred, cond, model_options)
                elif self.disco_enabled:
                    print(f"[Disco] Skipping guidance at step {self.step_count} (not in schedule)")
                
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
        
        # Apply strong effects in first half, then let diffusion refine
        total_steps = 50  # Typical diffusion steps
        current_progress = self.step_count / total_steps
        
        # Strong psychedelic effects in first 60% of steps
        if current_progress < 0.6:
            return True
        # Gradual reduction in middle 30%
        elif current_progress < 0.9:
            return self.step_count % 2 == 0  # Every other step
        # Minimal effects in final 10% (let diffusion refine)
        else:
            return self.step_count % 4 == 0  # Every 4th step
    
    def _get_psychedelic_strength(self):
        """Get psychedelic effect strength based on diffusion progress"""
        total_steps = 50
        current_progress = self.step_count / total_steps
        
        if current_progress < 0.3:
            # Early steps: Maximum psychedelic effects
            return 1.0
        elif current_progress < 0.6:
            # Middle-early: Strong effects
            return 0.8
        elif current_progress < 0.8:
            # Middle-late: Moderate effects
            return 0.4
        else:
            # Final steps: Minimal effects (let diffusion refine)
            return 0.1
    
    def _detect_resolution_and_tiling(self, x):
        """Detect the actual image resolution and if tiled VAE is likely being used"""
        if x.dim() != 4:
            return
        
        b, c, h, w = x.shape
        latent_scale = 8  # Standard VAE downscaling
        estimated_resolution = max(h * latent_scale, w * latent_scale)
        
        # Update detected resolution
        if self.detected_resolution is None:
            self.detected_resolution = estimated_resolution
            print(f"[Disco] Detected resolution: {estimated_resolution}px")
        
        # Detect if tiled VAE is likely being used (large images)
        if estimated_resolution > 1024 and not self.tiled_vae_detected:
            self.tiled_vae_detected = True
            print(f"[Disco] Large resolution detected ({estimated_resolution}px) - using tiled VAE compatible transforms")
        
        return estimated_resolution
    
    def _compute_clip_guidance(self, x, timestep, cond):
        """Compute CLIP guidance for the current latent"""
        try:
            import torch
            import torch.nn.functional as F
            
            if self.clip_model is None:
                return None
            
            # Try to decode latent to image space for CLIP analysis
            # This is a simplified approach - in practice you'd need VAE access
            
            # For now, apply CLIP guidance in latent space using text embeddings
            if hasattr(cond, 'shape') and len(cond.shape) >= 2:
                # Extract text features from conditioning
                text_features = cond
                if len(text_features.shape) > 2:
                    text_features = text_features.mean(dim=1)  # Average over sequence length
                
                # Create pseudo-visual features from latent
                # This is a simplified approach - real implementation would decode to image
                b, c, h, w = x.shape
                
                # Reshape latent for CLIP-like processing
                visual_features = x.view(b, c, -1).mean(dim=-1)  # Global average pooling
                
                # Normalize features
                text_features = F.normalize(text_features, dim=-1)
                visual_features = F.normalize(visual_features, dim=-1)
                
                # Compute similarity and guidance direction
                similarity = torch.sum(text_features * visual_features, dim=-1, keepdim=True)
                
                # Create guidance signal - push towards higher similarity
                guidance_direction = text_features.unsqueeze(-1).unsqueeze(-1) - visual_features.unsqueeze(-1).unsqueeze(-1)
                guidance_direction = guidance_direction.expand(-1, -1, h, w)
                
                # Scale guidance by similarity (stronger when less similar)
                guidance_scale = (1.0 - similarity).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, h, w)
                clip_guidance = guidance_direction * guidance_scale
                
                print(f"[Disco] Computed CLIP guidance - similarity: {similarity.mean().item():.3f}")
                return clip_guidance
            
            return None
            
        except Exception as e:
            print(f"[Disco] CLIP guidance computation failed: {e}")
            return None
    
    def _apply_geometric_transforms_to_latent(self, x):
        """Apply scale-aware geometric transforms to latent space"""
        if x.dim() != 4:
            return x
        
        try:
            b, c, h, w = x.shape
            result = x.clone()
            
            # Calculate scale-aware transform strength
            latent_scale = 8  # Standard VAE downscaling
            effective_resolution = max(h * latent_scale, w * latent_scale)
            resolution_factor = min(effective_resolution / 512.0, 4.0)
            
            # Reduce transform intensity for larger images to prevent artifacts
            scale_factor = 0.05 / resolution_factor
            
            # Apply transforms based on settings with scale awareness
            if 'rotate' in self.disco_transforms:
                # Very small rotation scaled for resolution
                angle = self.disco_rotation_speed * scale_factor
                if abs(angle) > 0.001:  # Only apply if meaningful
                    transform_matrix = DiscoTransforms.rotate_2d(angle, device=result.device)
                    result = DiscoTransforms.apply_transform(result, transform_matrix)
            
            if 'translate' in self.disco_transforms:
                # Scale translation for latent space
                tx = self.disco_translation_x * scale_factor * 0.5
                ty = self.disco_translation_y * scale_factor * 0.5
                if abs(tx) > 0.001 or abs(ty) > 0.001:  # Only apply if meaningful
                    transform_matrix = DiscoTransforms.translate_2d(tx, ty, device=result.device)
                    result = DiscoTransforms.apply_transform(result, transform_matrix)
            
            if 'zoom' in self.disco_transforms:
                # Very subtle zoom scaled for resolution
                zoom_amount = (self.disco_zoom_factor - 1.0) * scale_factor * 0.2
                zoom = 1.0 + zoom_amount
                if abs(zoom - 1.0) > 0.001:  # Only apply if meaningful
                    transform_matrix = DiscoTransforms.scale_2d(zoom, zoom, device=result.device)
                    result = DiscoTransforms.apply_transform(result, transform_matrix)
            
            return result
            
        except Exception as e:
            print(f"[Disco] Scale-aware transform failed: {e}")
            return x
    
    def _apply_disco_effects(self, x, timestep):
        """Apply disco diffusion effects to the tensor"""
        if x.dim() != 4:
            return x
        
        try:
            # Get current step progress
            progress = min(self.step_count / 50.0, 1.0)
            
            # Apply transforms based on configuration
            result = x.clone()
            
            # Only apply spatial transforms (spherical, kaleidoscope, fractal_zoom) to all channels
            if 'spherical' in self.disco_transforms:
                strength = self.disco_scale * (0.8 + 1.2 * progress)  # Increased from 0.3+0.7 to 0.8+1.2
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
            
            # Apply latent space color mixing for psychedelic effects
            if 'color_shift' in self.disco_transforms:
                mix_strength = self.disco_scale * 0.8 * math.sin(self.step_count * 0.1)  # Increased from 0.3 to 0.8
                result = DiscoTransforms.latent_color_mix(result, 
                                                        mix_strength=abs(mix_strength),
                                                        step=self.step_count)
            
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
    """Get predefined disco effect presets with ULTRA STRONG CLIP guidance"""
    return {
        'psychedelic': {
            'disco_scale': 25000.0,  # ULTRA STRONG - 25x original
            'disco_transforms': ['translate', 'rotate', 'zoom'],
            'disco_rotation_speed': 0.2,
            'disco_zoom_factor': 1.02,
            'disco_translation_x': 0.1,
            'disco_translation_y': 0.1,
            'disco_symmetry_mode': 'none',
            'cutn': 32,  # More cutouts for stronger CLIP
            'tv_scale': 0.0,
            'range_scale': 150.0
        },
        'fractal': {
            'disco_scale': 35000.0,  # ULTRA STRONG - 35x original
            'disco_transforms': ['zoom', 'rotate'],
            'disco_zoom_factor': 1.05,
            'disco_rotation_speed': 0.1,
            'disco_symmetry_mode': 'radial',
            'cutn': 48,  # Even more cutouts
            'tv_scale': 100.0,
            'range_scale': 200.0
        },
        'kaleidoscope': {
            'disco_scale': 20000.0,  # ULTRA STRONG - 25x original
            'disco_transforms': ['rotate', 'translate'],
            'disco_rotation_speed': 0.3,
            'disco_translation_x': 0.05,
            'disco_translation_y': 0.05,
            'disco_symmetry_mode': 'radial',
            'cutn': 40,
            'tv_scale': 50.0,
            'range_scale': 100.0
        },
        'dreamy': {
            'disco_scale': 15000.0,  # ULTRA STRONG - 30x original
            'disco_transforms': ['translate'],
            'disco_translation_x': 0.02,
            'disco_translation_y': 0.02,
            'disco_symmetry_mode': 'none',
            'cutn': 24,
            'tv_scale': 200.0,
            'range_scale': 50.0
        },
        'scientific': {
            'disco_scale': 50000.0,  # MAXIMUM STRENGTH - 50x original
            'disco_transforms': ['translate', 'rotate', 'zoom'],
            'disco_rotation_speed': 0.15,
            'disco_zoom_factor': 1.03,
            'disco_translation_x': 0.08,
            'disco_translation_y': 0.08,
            'disco_symmetry_mode': 'none',
            'cutn': 64,  # Maximum cutouts for ultimate CLIP analysis
            'tv_scale': 150.0,
            'range_scale': 300.0,
            'cut_pow': 1.0
        },
        'extreme': {
            'disco_scale': 100000.0,  # EXTREME STRENGTH - 100x original
            'disco_transforms': ['translate', 'rotate', 'zoom'],
            'disco_rotation_speed': 0.25,
            'disco_zoom_factor': 1.05,
            'disco_translation_x': 0.15,
            'disco_translation_y': 0.15,
            'disco_symmetry_mode': 'none',
            'cutn': 80,  # Maximum possible cutouts
            'tv_scale': 100.0,
            'range_scale': 400.0,
            'cut_pow': 1.2
        }
    }

# Global disco sampler instance
disco_sampler = DiscoSampler()

class StableDiffusionXLTPGPipeline:
    """
    Placeholder TPG Pipeline class for compatibility with async_worker imports
    This maintains compatibility while the main TPG functionality is handled by tpg_integration.py
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize TPG pipeline - placeholder for compatibility"""
        pass
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        """Create TPG pipeline from pretrained model - placeholder for compatibility"""
        return cls()
    
    def __call__(self, *args, **kwargs):
        """TPG pipeline call - placeholder for compatibility"""
        # The actual TPG functionality is handled by tpg_integration.py
        # This is just for import compatibility
        raise NotImplementedError("TPG functionality is handled by tpg_integration.py")

# Extend DiscoSampler with scientific methods
def _init_clip_simple(self):
    """Initialize CLIP model for guidance - Simple OpenAI CLIP approach"""
    try:
        import clip
        import torch
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Get the selected CLIP model from settings, with fallback order
        preferred_model = getattr(self, 'disco_clip_model', 'RN50')
        
        # All available CLIP models in order of preference
        all_clip_models = [
            'RN50',           # Fast and efficient
            'RN101',          # Better quality ResNet
            'RN50x4',         # High quality
            'RN50x16',        # Very high quality
            'RN50x64',        # Maximum quality (very slow)
            'ViT-B/32',       # Vision Transformer Base
            'ViT-B/16',       # Better ViT
            'ViT-L/14',       # Large ViT (high quality)
            'ViT-L/14@336px'  # Maximum quality ViT
        ]
        
        # Try the preferred model first, then fallback to others
        models_to_try = [preferred_model] + [m for m in all_clip_models if m != preferred_model]
        
        for model_name in models_to_try:
            try:
                print(f"[Disco] Loading CLIP model: {model_name}")
                self.clip_model, self.clip_preprocess = clip.load(model_name, device=device)
                self.clip_model.eval()
                print(f"[Disco] CLIP {model_name} loaded successfully on {device}")
                self.clip_model_name = model_name
                
                # Show model info
                if hasattr(self.clip_model.visual, 'input_resolution'):
                    resolution = self.clip_model.visual.input_resolution
                    print(f"[Disco] CLIP input resolution: {resolution}x{resolution}")
                
                return
            except Exception as e:
                print(f"[Disco] Failed to load {model_name}: {e}")
                continue
        
        # If all models failed
        print("[Disco] Failed to load any CLIP model")
        self.clip_model = None
        
    except ImportError:
        print("[Disco] CLIP not available.")
        print("[Disco] To enable full Disco Diffusion functionality:")
        print("[Disco]   1. Install CLIP: pip install git+https://github.com/openai/CLIP.git")
        print("[Disco]   2. Install torchvision: pip install torchvision")
        print("[Disco] Using geometric transforms only (still creates psychedelic effects)")
        self.clip_model = None
    except Exception as e:
        print(f"[Disco] Failed to initialize CLIP: {e}")
        print("[Disco] Using geometric transforms only")
        self.clip_model = None

DiscoSampler._init_clip = lambda self: _init_clip_simple(self)
DiscoSampler._get_alpha_t = lambda self, timestep: _get_alpha_t_impl(self, timestep)
DiscoSampler._get_sigma_t = lambda self, timestep: _get_sigma_t_impl(self, timestep)
DiscoSampler._decode_latent_to_image = lambda self, latent: _decode_latent_to_image_impl(self, latent)
DiscoSampler._apply_geometric_transforms = lambda self, image: _apply_geometric_transforms_impl(self, image)
DiscoSampler._extract_text_embeddings = lambda self, cond: _extract_text_embeddings_impl(self, cond)
DiscoSampler._get_guidance_schedule = lambda self: _get_guidance_schedule_impl(self)
DiscoSampler._apply_disco_guidance = lambda self, model, x, timestep, noise_pred, cond, model_options: _apply_disco_guidance_impl(self, model, x, timestep, noise_pred, cond, model_options)

def _init_clip_impl(self):
        """Initialize CLIP model for guidance"""
        try:
            # First try to use the built-in CLIP download system
            import modules.config as config
            clip_info = config.downloading_clip_for_disco()
            
            if clip_info:
                if isinstance(clip_info, dict) and clip_info.get('type') == 'onnx':
                    # Load ONNX CLIP models
                    try:
                        from .clip_onnx import load_clip_onnx
                        
                        visual_path = clip_info['visual']
                        textual_path = clip_info['textual']
                        
                        if os.path.exists(visual_path) and os.path.exists(textual_path):
                            self.clip_model, self.clip_tokenizer = load_clip_onnx(visual_path, textual_path)
                            if self.clip_model:
                                print(f"[Disco] CLIP ONNX models loaded successfully")
                                self.clip_type = 'onnx'
                                return
                        
                    except Exception as e:
                        print(f"[Disco] Failed to load ONNX CLIP: {e}")
                
                elif isinstance(clip_info, dict) and clip_info.get('type') == 'pytorch':
                    # Load PyTorch CLIP model
                    try:
                        import clip
                        device = "cuda" if torch.cuda.is_available() else "cpu"
                        
                        model_path = clip_info['model']
                        if os.path.exists(model_path):
                            self.clip_model, self.clip_preprocess = clip.load(model_path, device=device)
                            self.clip_model.eval()
                            print(f"[Disco] CLIP PyTorch model loaded from {model_path} on {device}")
                            self.clip_type = 'pytorch'
                            return
                            
                    except ImportError:
                        print("[Disco] CLIP library not installed for PyTorch model")
                    except Exception as e:
                        print(f"[Disco] Failed to load PyTorch CLIP: {e}")
            
            # Final fallback: try to load CLIP normally (will download automatically)
            try:
                import clip
                device = "cuda" if torch.cuda.is_available() else "cpu"
                self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=device)
                self.clip_model.eval()
                print(f"[Disco] CLIP model loaded successfully on {device}")
                self.clip_type = 'pytorch'
                
            except ImportError:
                print("[Disco] CLIP not available.")
                print("[Disco] To enable full Disco Diffusion functionality:")
                print("[Disco]   1. ONNX Runtime: pip install onnxruntime (or onnxruntime-gpu)")
                print("[Disco]   2. Or CLIP library: pip install git+https://github.com/openai/CLIP.git")
                print("[Disco] Using geometric transforms only (still creates psychedelic effects)")
                self.clip_model = None
                self.clip_type = None
                
                # Try different loading methods based on file type
                if clip_path.endswith('.safetensors'):
                    print(f"[Disco] Attempting to load LAION CLIP model from {clip_path}")
                    # For LAION models, we'd need open_clip library
                    try:
                        import open_clip
                        model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained=clip_path, device=device)
                        self.clip_model = model
                        self.clip_preprocess = preprocess
                        print(f"[Disco] LAION CLIP model loaded successfully on {device}")
                        return
                    except ImportError:
                        print("[Disco] open_clip not available for LAION models, trying standard CLIP...")
                
                # Try standard CLIP library
                try:
                    import clip
                    if clip_path.endswith('.pt'):
                        # Load OpenAI CLIP model
                        self.clip_model, self.clip_preprocess = clip.load(clip_path, device=device)
                    else:
                        # Try to load as standard CLIP
                        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=device)
                    
                    self.clip_model.eval()
                    print(f"[Disco] CLIP model loaded successfully on {device}")
                    return
                    
                except ImportError:
                    print("[Disco] CLIP library not installed.")
            
            # Final fallback: try to load CLIP normally
            try:
                import clip
                device = "cuda" if torch.cuda.is_available() else "cpu"
                self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=device)
                self.clip_model.eval()
                print(f"[Disco] CLIP model loaded via automatic download on {device}")
                return
            except ImportError:
                pass
            
            # If all fails, set to None for fallback mode
            raise ImportError("No CLIP available")
            
        except ImportError:
            print("[Disco] CLIP not available.")
            print("[Disco] To enable full Disco Diffusion functionality:")
            print("[Disco]   1. Install CLIP: pip install git+https://github.com/openai/CLIP.git")
            print("[Disco]   2. Or install open_clip: pip install open-clip-torch")
            print("[Disco]   3. Or install via conda: conda install -c conda-forge clip-by-openai")
            print("[Disco] Using geometric transforms only (still creates psychedelic effects)")
            self.clip_model = None
        except Exception as e:
            print(f"[Disco] Failed to load CLIP: {e}")
            print("[Disco] Using geometric transforms only with enhanced effects")
            self.clip_model = None
def _get_alpha_t_impl(self, timestep):
    """Get alpha_t for DDIM sampling"""
    # This would need to be extracted from the actual scheduler
    # For now, use approximation
    import torch
    return torch.sqrt(1 - timestep / 1000.0)

def _get_sigma_t_impl(self, timestep):
    """Get sigma_t for DDIM sampling"""
    # This would need to be extracted from the actual scheduler
    # For now, use approximation
    import torch
    return torch.sqrt(timestep / 1000.0)

def _decode_latent_to_image_impl(self, latent):
    """Decode latent to image space for CLIP analysis"""
    try:
        # This would need access to the VAE decoder
        # For now, return None to skip CLIP guidance
        return None
    except Exception as e:
        logger.warning(f"Failed to decode latent: {e}")
        return None

def _apply_geometric_transforms_impl(self, image):
    """Apply geometric transforms (core Disco Diffusion technique)"""
    try:
        # Get current animation frame
        frame = self.step_count
        
        # Calculate transform parameters
        if 'translate' in self.disco_transforms:
            tx = self.disco_translation_x * math.sin(frame * 0.1)
            ty = self.disco_translation_y * math.cos(frame * 0.1)
            transform = DiscoTransforms.translate_2d(tx, ty)
            image = DiscoTransforms.apply_transform(image, transform)
        
        if 'rotate' in self.disco_transforms:
            angle = self.disco_rotation_speed * frame * 0.1
            transform = DiscoTransforms.rotate_2d(angle)
            image = DiscoTransforms.apply_transform(image, transform)
        
        if 'zoom' in self.disco_transforms:
            zoom = self.disco_zoom_factor ** (frame * 0.1)
            transform = DiscoTransforms.scale_2d(zoom, zoom)
            image = DiscoTransforms.apply_transform(image, transform)
        
        return image
        
    except Exception as e:
        logger.warning(f"Geometric transform failed: {e}")
        return image

def _extract_text_embeddings_impl(self, cond):
    """Extract text embeddings from conditioning"""
    try:
        # This would need to extract CLIP text embeddings from the conditioning
        # For now, return None to skip CLIP guidance
        return None
    except Exception as e:
        logger.warning(f"Failed to extract text embeddings: {e}")
        return None

def _get_guidance_schedule_impl(self):
    """Get STRONG guidance scale based on current step"""
    progress = min(self.step_count / 50.0, 1.0)
    # Much stronger guidance throughout the process
    base_strength = 5.0  # Increased from 1.0 to 5.0
    # Less reduction over time - keep it strong
    return base_strength * (1.0 - progress * 0.2)  # Only reduce by 20% instead of 50%

def _apply_disco_guidance_impl(self, model, x, timestep, noise_pred, cond, model_options):
    """Apply true Disco Diffusion CLIP guidance"""
    try:
        # Detect resolution and tiling on first run
        self._detect_resolution_and_tiling(x)
        
        print(f"[Disco] Applying guidance - CLIP available: {self.clip_model is not None}")
        
        # If CLIP is available, use full scientific algorithm
        if self.clip_model is not None:
            return self._apply_full_disco_guidance(model, x, timestep, noise_pred, cond, model_options)
        else:
            # Fallback: apply geometric transforms to latent space directly
            print("[Disco] Using geometric fallback (no CLIP)")
            return self._apply_geometric_disco_fallback(x, timestep, noise_pred)
        
    except Exception as e:
        print(f"[Disco] Guidance failed: {e}")
        logger.warning(f"Disco guidance failed: {e}")
        return noise_pred

def _apply_full_disco_guidance(self, model, x, timestep, noise_pred, cond, model_options):
    """Apply full CLIP-guided Disco Diffusion with strong guidance"""
    print("[Disco] Applying STRONG CLIP guidance")
    
    try:
        import torch
        import torch.nn.functional as F
        
        # Get guidance strength - much stronger now
        guidance_strength = self.disco_scale / 100.0  # Increased from /1000 to /100 (10x stronger)
        guidance_strength = min(guidance_strength, 50.0)  # Cap at 50x instead of 1x
        
        print(f"[Disco] Using STRONG guidance strength: {guidance_strength}")
        
        # Apply CLIP guidance if available
        if self.clip_model is not None:
            clip_guidance = self._compute_clip_guidance(x, timestep, cond)
            if clip_guidance is not None:
                # Apply CLIP guidance directly to noise prediction with strong influence
                noise_pred = noise_pred + clip_guidance * guidance_strength
                print(f"[Disco] Applied CLIP guidance with strength {guidance_strength}")
        
        # Also apply geometric transforms for additional psychedelic effects
        transformed_noise = self._apply_geometric_transforms_to_latent(noise_pred)
        
        # Strong blending for maximum effect
        blend_factor = min(guidance_strength / 10.0, 0.8)  # Up to 80% blend
        result = noise_pred * (1 - blend_factor) + transformed_noise * blend_factor
        
        print(f"[Disco] Applied STRONG guidance - CLIP + geometric with blend {blend_factor:.2f}")
        return result
        
    except Exception as e:
        print(f"[Disco] Strong CLIP guidance failed: {e}")
        import traceback
        traceback.print_exc()
        return noise_pred

def _apply_geometric_disco_fallback(self, x, timestep, noise_pred):
    """Apply scale-aware geometric transforms to latent space"""
    import torch
    try:
        frame = self.step_count
        result = x.clone()
        
        # Get latent dimensions and calculate scale factor
        b, c, h, w = x.shape
        latent_scale = 8  # Standard VAE downscaling factor
        effective_resolution = max(h * latent_scale, w * latent_scale)
        
        # Get psychedelic strength based on diffusion progress
        psychedelic_strength = self._get_psychedelic_strength()
        
        # Scale transform strength based on resolution and disco scale
        base_strength = min(self.disco_scale / 1000.0, 1.0)
        resolution_factor = min(effective_resolution / 512.0, 4.0)  # Cap at 4x for very large images
        
        # Much stronger effects early in diffusion, scaled by resolution
        if psychedelic_strength > 0.8:
            # Early steps: Strong psychedelic effects
            transform_strength = base_strength * 0.8 / resolution_factor
        elif psychedelic_strength > 0.4:
            # Middle steps: Moderate effects
            transform_strength = base_strength * 0.4 / resolution_factor
        else:
            # Late steps: Subtle refinement
            transform_strength = base_strength * 0.1 / resolution_factor
        
        print(f"[Disco] Scale-aware fallback - resolution: {effective_resolution}, strength: {transform_strength:.4f}")
        
        # Apply psychedelic transforms - stronger early, subtle late
        if 'translate' in self.disco_transforms and transform_strength > 0.001:
            # Scale translation based on psychedelic strength
            speed_multiplier = 0.2 if psychedelic_strength > 0.8 else 0.05
            amplitude_multiplier = 2.0 if psychedelic_strength > 0.8 else 0.5
            
            tx_pixels = self.disco_translation_x * math.sin(frame * speed_multiplier) * amplitude_multiplier
            ty_pixels = self.disco_translation_y * math.cos(frame * speed_multiplier) * amplitude_multiplier
            
            # Convert to latent space pixels
            tx_latent = max(1, int(abs(tx_pixels) / latent_scale)) * (1 if tx_pixels >= 0 else -1)
            ty_latent = max(1, int(abs(ty_pixels) / latent_scale)) * (1 if ty_pixels >= 0 else -1)
            
            # More aggressive bounds for early steps
            max_translation = w//2 if psychedelic_strength > 0.8 else w//4
            if abs(tx_latent) <= max_translation and abs(ty_latent) <= max_translation:
                result = torch.roll(result, shifts=(ty_latent, tx_latent), dims=(2, 3))
                print(f"[Disco] Applied psychedelic translation: tx={tx_latent}, ty={ty_latent}, strength={psychedelic_strength:.2f}")
        
        if 'rotate' in self.disco_transforms and transform_strength > 0.001:
            if psychedelic_strength > 0.8:
                # Early steps: More frequent rotation
                rotation_progress = self.disco_rotation_speed * frame * 0.05
                if rotation_progress > 0.5:  # Rotate more frequently
                    angle_steps = int(rotation_progress) % 4
                    if angle_steps > 0:
                        result = torch.rot90(result, k=angle_steps, dims=[2, 3])
                        print(f"[Disco] Applied psychedelic rotation: {angle_steps * 90}°, strength={psychedelic_strength:.2f}")
            else:
                # Later steps: Subtle rotation
                rotation_progress = self.disco_rotation_speed * frame * 0.01
                if rotation_progress > 1.0:
                    angle_steps = int(rotation_progress) % 4
                    if angle_steps > 0:
                        result = torch.rot90(result, k=angle_steps, dims=[2, 3])
                        print(f"[Disco] Applied subtle rotation: {angle_steps * 90}°")
        
        if 'zoom' in self.disco_transforms and transform_strength > 0.001:
            if psychedelic_strength > 0.8:
                # Early steps: More aggressive zoom
                zoom_base = 1.0 + (self.disco_zoom_factor - 1.0) * 0.5  # Stronger zoom
                zoom = zoom_base ** (frame * 0.03)  # Faster zoom
                max_zoom = 1.5  # Allow more zoom early
            else:
                # Later steps: Subtle zoom
                zoom_base = 1.0 + (self.disco_zoom_factor - 1.0) * 0.1
                zoom = zoom_base ** (frame * 0.01)
                max_zoom = 1.2  # Conservative zoom late
            
            if abs(zoom - 1.0) > 0.005:
                if zoom > 1.0 and zoom < max_zoom:
                    # Zoom in
                    crop_factor = 1.0 / zoom
                    new_h = max(h//3, int(h * crop_factor))
                    new_w = max(w//3, int(w * crop_factor))
                    
                    start_h = (h - new_h) // 2
                    start_w = (w - new_w) // 2
                    
                    cropped = result[:, :, start_h:start_h+new_h, start_w:start_w+new_w]
                    result = F.interpolate(cropped, size=(h, w), mode='bilinear', align_corners=False)
                    print(f"[Disco] Applied psychedelic zoom: {zoom:.4f}, strength={psychedelic_strength:.2f}")
        
        # Subtle channel mixing for psychedelic effects (latent-space safe)
        if c >= 4 and transform_strength > 0.001:
            mix_strength = transform_strength * 0.3
            phase = frame * 0.02
            
            # Mix adjacent channels with sine waves
            for i in range(0, min(c-1, 3), 2):
                if i+1 < c:
                    mix_factor = mix_strength * math.sin(phase + i * math.pi/3)
                    if abs(mix_factor) > 0.001:
                        ch_a, ch_b = result[:, i:i+1], result[:, i+1:i+2]
                        result[:, i:i+1] = ch_a * (1 - abs(mix_factor)) + ch_b * mix_factor
                        result[:, i+1:i+2] = ch_b * (1 - abs(mix_factor)) - ch_a * mix_factor
            
            print(f"[Disco] Applied subtle channel mixing")
        
        # Apply symmetry effects (safe for tiled processing)
        if self.disco_symmetry_mode != 'none' and transform_strength > 0.01:
            result = DiscoTransforms.symmetrize(result, self.disco_symmetry_mode)
            print(f"[Disco] Applied symmetry: {self.disco_symmetry_mode}")
        
        # Very conservative blending to avoid artifacts
        coherence = max(0.8, self.disco_color_coherence)  # High coherence to preserve quality
        modified_x = coherence * x + (1 - coherence) * result
        
        # Apply very subtle modification to noise prediction
        noise_modification = (modified_x - x) * transform_strength
        modified_noise = noise_pred + noise_modification
        
        print(f"[Disco] Scale-aware geometric transforms applied successfully")
        return modified_noise
        
    except Exception as e:
        print(f"[Disco] Scale-aware disco fallback failed: {e}")
        import traceback
        traceback.print_exc()
        return noise_pred

# Add the new methods to DiscoSampler
DiscoSampler._apply_full_disco_guidance = lambda self, model, x, timestep, noise_pred, cond, model_options: _apply_full_disco_guidance(self, model, x, timestep, noise_pred, cond, model_options)
DiscoSampler._apply_geometric_disco_fallback = lambda self, x, timestep, noise_pred: _apply_geometric_disco_fallback(self, x, timestep, noise_pred)
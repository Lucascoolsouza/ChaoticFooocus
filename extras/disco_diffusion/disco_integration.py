# Disco Diffusion Integration for Fooocus
# Handles integration with the main pipeline

import torch
import logging
from .pipeline_disco import disco_sampler, get_disco_presets

logger = logging.getLogger(__name__)

class DiscoIntegration:
    """Handles integration of Disco Diffusion with Fooocus pipeline"""
    
    def __init__(self):
        self.is_initialized = False
        self.current_settings = {}
    
    def initialize_disco(self, 
                        disco_enabled=False,
                        disco_scale=0.5,
                        disco_preset='custom',
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
                        disco_clip_model='RN50',
                        disco_noise_schedule='linear',
                        disco_steps_schedule=None):
        """Initialize disco diffusion with given parameters"""
        
        if not disco_enabled:
            disco_sampler.deactivate()
            return
        
        # Apply preset if not custom
        if disco_preset != 'custom':
            presets = get_disco_presets()
            if disco_preset in presets:
                preset_settings = presets[disco_preset]
                # Get base scale from preset and multiply by UI slider (0-1)
                base_disco_scale = preset_settings.get('disco_scale', 5000.0)  # Default high value
                disco_scale = base_disco_scale * disco_scale  # UI slider multiplies preset base
                disco_transforms = preset_settings.get('disco_transforms', disco_transforms)
                disco_zoom_factor = preset_settings.get('disco_zoom_factor', disco_zoom_factor)
                disco_rotation_speed = preset_settings.get('disco_rotation_speed', disco_rotation_speed)
                disco_saturation_boost = preset_settings.get('disco_saturation_boost', disco_saturation_boost)
                disco_contrast_boost = preset_settings.get('disco_contrast_boost', disco_contrast_boost)
                disco_symmetry_mode = preset_settings.get('disco_symmetry_mode', disco_symmetry_mode)
                disco_fractal_octaves = preset_settings.get('disco_fractal_octaves', disco_fractal_octaves)
                disco_color_coherence = preset_settings.get('disco_color_coherence', disco_color_coherence)
        
        # Set default transforms if none provided
        if disco_transforms is None:
            disco_transforms = ['spherical', 'color_shift']
        
        # Set default steps schedule if none provided
        if disco_steps_schedule is None:
            disco_steps_schedule = [0.0, 1.0]  # Apply from start to end (every step)
        
        # Update disco sampler settings
        disco_sampler.disco_enabled = disco_enabled
        disco_sampler.disco_scale = disco_scale
        disco_sampler.disco_transforms = disco_transforms
        disco_sampler.disco_seed = disco_seed
        disco_sampler.disco_animation_mode = disco_animation_mode
        disco_sampler.disco_zoom_factor = disco_zoom_factor
        disco_sampler.disco_rotation_speed = disco_rotation_speed
        disco_sampler.disco_translation_x = disco_translation_x
        disco_sampler.disco_translation_y = disco_translation_y
        disco_sampler.disco_color_coherence = disco_color_coherence
        disco_sampler.disco_saturation_boost = disco_saturation_boost
        disco_sampler.disco_contrast_boost = disco_contrast_boost
        disco_sampler.disco_symmetry_mode = disco_symmetry_mode
        disco_sampler.disco_fractal_octaves = disco_fractal_octaves
        disco_sampler.disco_clip_model = disco_clip_model
        disco_sampler.disco_noise_schedule = disco_noise_schedule
        disco_sampler.disco_steps_schedule = disco_steps_schedule
        
        # Re-initialize random state if seed changed
        if disco_seed is not None:
            import random
            disco_sampler.rng = random.Random(disco_seed)
        
        self.current_settings = {
            'disco_enabled': disco_enabled,
            'disco_scale': disco_scale,
            'disco_preset': disco_preset,
            'disco_transforms': disco_transforms,
            'disco_seed': disco_seed,
            'disco_animation_mode': disco_animation_mode,
            'disco_zoom_factor': disco_zoom_factor,
            'disco_rotation_speed': disco_rotation_speed,
            'disco_translation_x': disco_translation_x,
            'disco_translation_y': disco_translation_y,
            'disco_color_coherence': disco_color_coherence,
            'disco_saturation_boost': disco_saturation_boost,
            'disco_contrast_boost': disco_contrast_boost,
            'disco_symmetry_mode': disco_symmetry_mode,
            'disco_fractal_octaves': disco_fractal_octaves,
            'disco_noise_schedule': disco_noise_schedule,
            'disco_steps_schedule': disco_steps_schedule
        }
        
        self.is_initialized = True
        
        print(f"[Disco] Initialized with preset: {disco_preset}, scale: {disco_scale}, transforms: {disco_transforms}")
    
    def activate_for_generation(self, unet, vae=None):
        """Activate disco effects for generation"""
        if self.is_initialized and disco_sampler.disco_enabled:
            disco_sampler.activate(unet, vae)
            return True
        return False
    
    def deactivate_after_generation(self):
        """Deactivate disco effects after generation"""
        disco_sampler.deactivate()
    
    def get_current_settings(self):
        """Get current disco settings"""
        return self.current_settings.copy()
    
    def get_status_string(self):
        """Get status string for UI display"""
        if not self.is_initialized or not disco_sampler.disco_enabled:
            return "Disco Diffusion: Disabled"
        
        settings = self.current_settings
        preset = settings.get('disco_preset', 'custom')
        scale = settings.get('disco_scale', 0.5)
        transforms = settings.get('disco_transforms', [])
        
        transform_str = ", ".join(transforms) if transforms else "none"
        
        return f"Disco: {preset.title()} | Scale: {scale} | Effects: {transform_str}"

# Global integration instance
disco_integration = DiscoIntegration()

def apply_disco_to_pipeline(pipeline_module, disco_settings):
    """Apply disco diffusion settings to the pipeline"""
    try:
        # Initialize disco with settings
        disco_integration.initialize_disco(**disco_settings)
        
        # Activate for the current UNet if available
        if hasattr(pipeline_module, 'final_unet') and pipeline_module.final_unet is not None:
            disco_integration.activate_for_generation(pipeline_module.final_unet)
            return True
        
        return False
        
    except Exception as e:
        logger.error(f"Failed to apply disco to pipeline: {e}")
        return False

def cleanup_disco_from_pipeline():
    """Clean up disco effects from pipeline"""
    try:
        disco_integration.deactivate_after_generation()
    except Exception as e:
        logger.error(f"Failed to cleanup disco from pipeline: {e}")
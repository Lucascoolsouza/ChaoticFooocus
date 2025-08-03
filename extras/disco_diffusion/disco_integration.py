# Disco Diffusion Integration for Fooocus
# Handles the integration of disco effects into the main pipeline

import torch
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

class DiscoIntegration:
    """Manages disco diffusion integration with the main pipeline."""
    
    def __init__(self):
        self.enabled = False
        self.settings = {}
        
    def configure(self, **kwargs):
        """Configure disco settings."""
        self.settings.update(kwargs)
        self.enabled = kwargs.get('disco_enabled', False)
        
    def apply_to_latent(self, latent_samples, **kwargs):
        """Apply disco distortion to latent samples."""
        if not self.enabled:
            return latent_samples
            
        try:
            from .pipeline_disco import inject_disco_distortion
            
            return inject_disco_distortion(
                latent_samples,
                disco_scale=kwargs.get('disco_scale', 5.0),
                distortion_type=kwargs.get('disco_preset', 'psychedelic'),
                intensity_multiplier=kwargs.get('intensity_multiplier', 1.0),
                test_mode=kwargs.get('test_mode', False)
            )
        except Exception as e:
            logger.error(f"Failed to apply disco distortion: {e}")
            return latent_samples

# Global integration instance
disco_integration = DiscoIntegration()

def apply_disco_to_pipeline(pipeline, **kwargs):
    """Apply disco effects to the pipeline."""
    disco_integration.configure(**kwargs)
    return disco_integration

def cleanup_disco_from_pipeline(pipeline):
    """Clean up disco effects from the pipeline."""
    disco_integration.enabled = False
    disco_integration.settings.clear()
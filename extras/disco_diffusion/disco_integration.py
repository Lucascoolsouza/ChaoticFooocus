# Disco Diffusion Integration for Fooocus
# Handles integration with the main pipeline

import torch
import logging
from .pipeline_disco import disco_settings, run_clip_guidance_loop
import clip

logger = logging.getLogger(__name__)

class DiscoIntegration:
    """Handles integration of Disco Diffusion with Fooocus pipeline"""
    
    def __init__(self):
        self.is_initialized = False
        self.clip_model = None
        self.clip_preprocess = None

    def initialize_disco(self, **kwargs):
        """Initialize disco diffusion with given parameters"""
        disco_settings.update(**kwargs)
        self.is_initialized = True
        print(f"[Disco] Initialized with settings: {kwargs}")

    def _load_clip_model(self):
        if self.clip_model is None:
            try:
                device = "cuda" if torch.cuda.is_available() else "cpu"
                model_name = getattr(disco_settings, 'disco_clip_model', 'RN50')
                self.clip_model, self.clip_preprocess = clip.load(model_name, device=device)
                self.clip_model.eval()
                print(f"[Disco] CLIP model '{model_name}' loaded successfully on {device}.")
            except Exception as e:
                logger.error(f"Failed to load CLIP model: {e}", exc_info=True)
                self.clip_model = None

    def run_disco_guidance(self, latent, vae, text_prompt, async_task=None):
        """Run the pre-sampling CLIP guidance loop."""
        if not self.is_initialized or not getattr(disco_settings, 'disco_enabled', False):
            return latent

        self._load_clip_model()
        if self.clip_model is None:
            logger.error("Cannot run Disco guidance without a CLIP model.")
            return latent

        return run_clip_guidance_loop(
            latent=latent,
            vae=vae,
            clip_model=self.clip_model,
            clip_preprocess=self.clip_preprocess,
            text_prompt=text_prompt,
            async_task=async_task,
            steps=getattr(disco_settings, 'disco_guidance_steps', 100),
            disco_scale=disco_settings.disco_scale,
            cutn=getattr(disco_settings, 'cutn', 16),
            tv_scale=getattr(disco_settings, 'tv_scale', 150.0),
            range_scale=getattr(disco_settings, 'range_scale', 50.0)
        )

# Global integration instance
disco_integration = DiscoIntegration()

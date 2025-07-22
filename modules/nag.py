"""
NAG (Negative Attention Guidance) implementation for Stable Diffusion XL
Based on: https://github.com/sag-org/negative-attention-guidance
"""

import torch
import torch.nn.functional as F
from typing import Optional, Union, List, Dict, Any, Callable
import inspect

try:
    from diffusers import StableDiffusionXLPipeline
    from diffusers.utils import logging
    from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import (
        retrieve_timesteps,
        rescale_noise_cfg,
    )
    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False
    print("diffusers not available - NAG will use basic implementation")

logger = logging.get_logger(__name__) if DIFFUSERS_AVAILABLE else None


class NAGStableDiffusionXLPipeline(StableDiffusionXLPipeline if DIFFUSERS_AVAILABLE else object):
    """
    NAG-enabled Stable Diffusion XL Pipeline
    
    This pipeline extends the standard StableDiffusionXLPipeline with 
    Negative Attention Guidance (NAG) which helps improve prompt adherence 
    by using negative prompts more effectively.
    """
    
    def __init__(self, *args, **kwargs):
        if DIFFUSERS_AVAILABLE:
            super().__init__(*args, **kwargs)
        else:
            # Fallback initialization for basic compatibility
            self.vae = kwargs.get('vae')
            self.text_encoder = kwargs.get('text_encoder')
            self.text_encoder_2 = kwargs.get('text_encoder_2')
            self.tokenizer = kwargs.get('tokenizer')
            self.tokenizer_2 = kwargs.get('tokenizer_2')
            self.unet = kwargs.get('unet')
            self.scheduler = kwargs.get('scheduler')
            self.image_processor = kwargs.get('image_processor')
    
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        timesteps: List[int] = None,
        denoising_end: Optional[float] = None,
        guidance_scale: float = 5.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
        original_size: Optional[Tuple[int, int]] = None,
        crops_coords_top_left: Tuple[int, int] = (0, 0),
        target_size: Optional[Tuple[int, int]] = None,
        negative_original_size: Optional[Tuple[int, int]] = None,
        negative_crops_coords_top_left: Tuple[int, int] = (0, 0),
        negative_target_size: Optional[Tuple[int, int]] = None,
        clip_skip: Optional[int] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        # NAG-specific parameters
        nag_negative_prompt: Optional[Union[str, List[str]]] = None,
        nag_scale: float = 3.0,
        **kwargs
    ):
        """
        Generate images using NAG-enhanced Stable Diffusion XL
        
        Args:
            prompt: The positive prompt
            nag_negative_prompt: The negative prompt for NAG
            guidance_scale: Standard CFG guidance scale
            nag_scale: NAG guidance scale
            num_inference_steps: Number of denoising steps
            height: Image height
            width: Image width
        """
        
        # Encode prompts
        positive_embeds = self._encode_prompt(prompt)
        negative_embeds = self._encode_prompt(nag_negative_prompt) if nag_negative_prompt else None
        
        # Initialize latents
        latents = self._prepare_latents(height, width)
        
        # Set timesteps
        self.scheduler.set_timesteps(num_inference_steps)
        timesteps = self.scheduler.timesteps
        
        # Denoising loop with NAG
        for i, t in enumerate(timesteps):
            # Expand latents for CFG
            latent_model_input = torch.cat([latents] * 2) if guidance_scale > 1.0 else latents
            
            # Predict noise
            if guidance_scale > 1.0:
                # Standard CFG with positive and negative prompts
                prompt_embeds = torch.cat([negative_embeds, positive_embeds]) if negative_embeds is not None else torch.cat([torch.zeros_like(positive_embeds), positive_embeds])
                noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=prompt_embeds).sample
                
                # Apply CFG
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                
                # Apply NAG if enabled
                if nag_scale > 0 and negative_embeds is not None:
                    # Get noise prediction without conditioning (for NAG)
                    noise_pred_nag = self.unet(latents, t, encoder_hidden_states=torch.zeros_like(positive_embeds)).sample
                    
                    # Apply NAG correction
                    nag_correction = F.normalize(noise_pred_nag.abs(), p=2, dim=1) * (noise_pred - noise_pred_nag)
                    noise_pred = noise_pred - nag_scale * nag_correction
            else:
                noise_pred = self.unet(latents, t, encoder_hidden_states=positive_embeds).sample
            
            # Compute previous sample
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample
        
        # Decode latents to images
        images = self._decode_latents(latents)
        
        return images
    
    def _encode_prompt(self, prompt: str):
        """Encode text prompt to embeddings"""
        # Tokenize
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        
        # Get embeddings from first text encoder
        text_embeddings = self.text_encoder(text_inputs.input_ids.to(self.text_encoder.device))[0]
        
        # If we have a second text encoder (SDXL), use it too
        if self.text_encoder_2 is not None:
            text_inputs_2 = self.tokenizer_2(
                prompt,
                padding="max_length",
                max_length=self.tokenizer_2.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_embeddings_2 = self.text_encoder_2(text_inputs_2.input_ids.to(self.text_encoder_2.device))[0]
            
            # Concatenate embeddings for SDXL
            text_embeddings = torch.cat([text_embeddings, text_embeddings_2], dim=-1)
        
        return text_embeddings
    
    def _prepare_latents(self, height: int, width: int):
        """Initialize random latents"""
        shape = (1, self.unet.config.in_channels, height // 8, width // 8)
        latents = torch.randn(shape, device=self.unet.device, dtype=self.unet.dtype)
        
        # Scale latents by scheduler's init noise sigma
        latents = latents * self.scheduler.init_noise_sigma
        
        return latents
    
    def _decode_latents(self, latents):
        """Decode latents to images"""
        # Scale latents
        latents = 1 / self.vae.config.scaling_factor * latents
        
        # Decode
        images = self.vae.decode(latents).sample
        
        # Convert to PIL images
        images = (images / 2 + 0.5).clamp(0, 1)
        images = images.cpu().permute(0, 2, 3, 1).float().numpy()
        
        if self.image_processor:
            images = self.image_processor.numpy_to_pil(images)
        
        return images
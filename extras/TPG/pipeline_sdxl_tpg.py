# Implementation of StableDiffusionXLTPGPipeline (Token Perturbation Guidance)

import inspect
import sys
from typing import Type, Any, Callable, Dict, List, Optional, Tuple, Union

import math
import torch
import torch.nn.functional as F

import os
from accelerate.utils import set_seed

from transformers import (
    CLIPImageProcessor,
    CLIPTextModel,
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    CLIPVisionModelWithProjection,
)

from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.image_processor import PipelineImageInput, VaeImageProcessor
from diffusers.loaders import (
    FromSingleFileMixin,
    IPAdapterMixin,
    StableDiffusionXLLoraLoaderMixin,
    TextualInversionLoaderMixin,
)
from diffusers.models import AutoencoderKL, ImageProjection, UNet2DConditionModel
from diffusers.models.attention_processor import (
    Attention,
    AttnProcessor2_0,
    FusedAttnProcessor2_0,
    LoRAAttnProcessor2_0,
    LoRAXFormersAttnProcessor,
    XFormersAttnProcessor,
)
from diffusers.models.attention import BasicTransformerBlock
from diffusers.models.lora import adjust_lora_scale_text_encoder
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils import (
    USE_PEFT_BACKEND,
    deprecate,
    is_invisible_watermark_available,
    is_torch_xla_available,
    logging,
    replace_example_docstring,
    scale_lora_layers,
    unscale_lora_layers,
)
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline, StableDiffusionMixin
from diffusers.pipelines.stable_diffusion_xl.pipeline_output import StableDiffusionXLPipelineOutput
from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import (
    StableDiffusionXLPipeline,
    retrieve_timesteps,
    rescale_noise_cfg,
)

logger = logging.get_logger(__name__)

class StableDiffusionXLTPGPipeline(StableDiffusionXLPipeline):
    """
    Simplified TPG Pipeline that inherits from StableDiffusionXLPipeline
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._tpg_scale = 0.0
        self._tpg_applied_layers_index = None
    
    @property
    def tpg_scale(self):
        return self._tpg_scale
    
    @property
    def do_token_perturbation_guidance(self):
        return self._tpg_scale > 0
    
    @property
    def tpg_applied_layers_index(self):
        if self._tpg_applied_layers_index is None:
            return ["d6", "d7", "d8", "d9", "d10", "d11", "d12", "d13", "d14", "d15", "d16", "d17", "d18", "d19", "d20", "d21", "d22", "d23"]
        return self._tpg_applied_layers_index
    
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
        tpg_scale: float = 0.0,
        tpg_applied_layers_index: List[str] = None,
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
        **kwargs,
    ):
        """
        Simplified TPG implementation that just modifies the guidance calculation
        """
        
        # Set TPG parameters
        self._tpg_scale = tpg_scale
        self._tpg_applied_layers_index = tpg_applied_layers_index
        
        # If TPG is not enabled, use the parent pipeline
        if not self.do_token_perturbation_guidance:
            return super().__call__(
                prompt=prompt,
                prompt_2=prompt_2,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                timesteps=timesteps,
                denoising_end=denoising_end,
                guidance_scale=guidance_scale,
                negative_prompt=negative_prompt,
                negative_prompt_2=negative_prompt_2,
                num_images_per_prompt=num_images_per_prompt,
                eta=eta,
                generator=generator,
                latents=latents,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
                output_type=output_type,
                return_dict=return_dict,
                cross_attention_kwargs=cross_attention_kwargs,
                guidance_rescale=guidance_rescale,
                original_size=original_size,
                crops_coords_top_left=crops_coords_top_left,
                target_size=target_size,
                negative_original_size=negative_original_size,
                negative_crops_coords_top_left=negative_crops_coords_top_left,
                negative_target_size=negative_target_size,
                clip_skip=clip_skip,
                callback_on_step_end=callback_on_step_end,
                callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
                **kwargs,
            )
        
        # Store original UNet forward method
        original_unet_forward = None
        if hasattr(self.unet, 'forward'):
            original_unet_forward = self.unet.forward
        elif hasattr(self.unet, '__call__'):
            original_unet_forward = self.unet.__call__
        
        def tpg_unet_forward(sample, timestep, encoder_hidden_states, **kwargs):
            """Modified UNet forward that handles TPG guidance"""
            batch_size = sample.shape[0]
            
            if self.do_token_perturbation_guidance and batch_size == 2:
                # For TPG, duplicate the conditional part with token shuffling
                uncond_sample, cond_sample = sample.chunk(2)
                sample = torch.cat([uncond_sample, cond_sample, cond_sample], dim=0)
                
                if encoder_hidden_states is not None:
                    uncond_embeds, cond_embeds = encoder_hidden_states.chunk(2)
                    # Apply token shuffling to the third copy
                    cond_embeds_shuffled = self._shuffle_tokens(cond_embeds)
                    encoder_hidden_states = torch.cat([uncond_embeds, cond_embeds, cond_embeds_shuffled], dim=0)
                
                # Handle other kwargs that might need duplication
                if 'added_cond_kwargs' in kwargs:
                    added_cond_kwargs = kwargs['added_cond_kwargs']
                    new_added_cond_kwargs = {}
                    for key, value in added_cond_kwargs.items():
                        if isinstance(value, torch.Tensor) and value.shape[0] == 2:
                            uncond_val, cond_val = value.chunk(2)
                            new_added_cond_kwargs[key] = torch.cat([uncond_val, cond_val, cond_val], dim=0)
                        else:
                            new_added_cond_kwargs[key] = value
                    kwargs['added_cond_kwargs'] = new_added_cond_kwargs
                
                # Call original UNet
                if hasattr(self.unet, 'model') and hasattr(self.unet.model, 'apply_model'):
                    # ComfyUI style
                    noise_pred = self.unet.model.apply_model(sample, timestep, c_crossattn=encoder_hidden_states, **kwargs)
                else:
                    # Standard diffusers style
                    noise_pred = original_unet_forward(sample, timestep, encoder_hidden_states=encoder_hidden_states, **kwargs)
                
                # Apply TPG guidance
                if noise_pred.shape[0] == 3:
                    noise_pred_uncond, noise_pred_cond, noise_pred_tpg = noise_pred.chunk(3)
                    
                    # First apply CFG
                    noise_pred_cfg = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
                    
                    # Then apply TPG
                    noise_pred_final = noise_pred_cfg + tpg_scale * (noise_pred_cond - noise_pred_tpg)
                    
                    # Return only the final prediction (batch size 1)
                    return noise_pred_final.unsqueeze(0) if noise_pred_final.dim() == 3 else noise_pred_final[:1]
                else:
                    return noise_pred
            else:
                # Call original UNet without TPG
                if hasattr(self.unet, 'model') and hasattr(self.unet.model, 'apply_model'):
                    return self.unet.model.apply_model(sample, timestep, c_crossattn=encoder_hidden_states, **kwargs)
                else:
                    return original_unet_forward(sample, timestep, encoder_hidden_states=encoder_hidden_states, **kwargs)
        
        # Temporarily replace UNet forward method
        if hasattr(self.unet, 'forward'):
            self.unet.forward = tpg_unet_forward
        elif hasattr(self.unet, '__call__'):
            self.unet.__call__ = tpg_unet_forward
        
        try:
            # Call the parent pipeline
            result = super().__call__(
                prompt=prompt,
                prompt_2=prompt_2,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                timesteps=timesteps,
                denoising_end=denoising_end,
                guidance_scale=guidance_scale,
                negative_prompt=negative_prompt,
                negative_prompt_2=negative_prompt_2,
                num_images_per_prompt=num_images_per_prompt,
                eta=eta,
                generator=generator,
                latents=latents,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
                output_type=output_type,
                return_dict=return_dict,
                cross_attention_kwargs=cross_attention_kwargs,
                guidance_rescale=guidance_rescale,
                original_size=original_size,
                crops_coords_top_left=crops_coords_top_left,
                target_size=target_size,
                negative_original_size=negative_original_size,
                negative_crops_coords_top_left=negative_crops_coords_top_left,
                negative_target_size=negative_target_size,
                clip_skip=clip_skip,
                callback_on_step_end=callback_on_step_end,
                callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
                **kwargs,
            )
            
            return result
        
        finally:
            # Restore original UNet forward method
            if original_unet_forward is not None:
                if hasattr(self.unet, 'forward'):
                    self.unet.forward = original_unet_forward
                elif hasattr(self.unet, '__call__'):
                    self.unet.__call__ = original_unet_forward
    
    def _shuffle_tokens(self, x):
        """
        Randomly shuffle the order of input tokens.
        
        Args:
            x (torch.Tensor): Input tensor with shape (batch_size, num_tokens, channels)
        
        Returns:
            torch.Tensor: Shuffled tensor with the same shape
        """
        try:
            if len(x.shape) >= 2:
                b, n = x.shape[:2]
                permutation = torch.randperm(n, device=x.device)
                return x[:, permutation]
            return x
        except Exception as e:
            logger.warning(f"Token shuffling failed: {e}")
            return x
#!/usr/bin/env python3
"""
Stable Diffusion XL TPG (Token Perturbation Guidance) Pipeline
Based on the paper: "Token Perturbation Guidance for Improved Diffusion Generation"
"""

import logging
import torch
import torch.nn.functional as F
from typing import Any, Callable, Dict, List, Optional, Union
from diffusers import StableDiffusionXLPipeline
from diffusers.utils import (
    is_invisible_watermark_available,
    logging as diffusers_logging,
)
from diffusers.image_processor import VaeImageProcessor

logger = logging.getLogger(__name__)

class StableDiffusionXLTPGPipeline(StableDiffusionXLPipeline):
    """
    Stable Diffusion XL Pipeline with Token Perturbation Guidance (TPG)
    
    TPG works by:
    1. Computing attention maps during the denoising process
    2. Perturbing token embeddings to create degraded predictions
    3. Using the difference to guide the generation away from degraded outputs
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._tpg_scale = 0.0
        self._tpg_applied_layers = []
        self._original_attn_processors = {}
        
    @property
    def tpg_scale(self):
        """TPG guidance scale"""
        return self._tpg_scale
    
    @tpg_scale.setter
    def tpg_scale(self, value):
        self._tpg_scale = value
    
    @property
    def do_token_perturbation_guidance(self):
        """Whether TPG is enabled"""
        return self._tpg_scale > 0
    
    def _get_attention_processors(self):
        """Get all attention processors from the UNet"""
        processors = {}
        
        def fn_recursive_retrieve(name, module, processors):
            if hasattr(module, "get_processor"):
                processors[f"{name}.processor"] = module.get_processor(return_deprecated_lora=True)
            
            for sub_name, child in module.named_children():
                fn_recursive_retrieve(f"{name}.{sub_name}", child, processors)
            
            return processors
        
        for name, module in self.unet.named_children():
            fn_recursive_retrieve(name, module, processors)
        
        return processors
    
    def _set_attention_processors(self, processors):
        """Set attention processors in the UNet"""
        def fn_recursive_attn_processor(name, module, processor):
            if hasattr(module, "set_processor"):
                if not isinstance(processor, dict):
                    module.set_processor(processor)
                else:
                    module.set_processor(processor.pop(f"{name}.processor"))
            
            for sub_name, child in module.named_children():
                fn_recursive_attn_processor(f"{name}.{sub_name}", child, processor)
        
        for name, module in self.unet.named_children():
            fn_recursive_attn_processor(name, module, processors)
    
    def enable_tpg(self, tpg_scale: float = 3.0, tpg_applied_layers: List[str] = None):
        """
        Enable Token Perturbation Guidance
        
        Args:
            tpg_scale: Guidance scale for TPG (higher = stronger effect)
            tpg_applied_layers: List of layer names to apply TPG to (None = all layers)
        """
        if tpg_applied_layers is None:
            # Default to applying TPG to middle and up layers
            tpg_applied_layers = ["mid", "up"]
        
        self._tpg_scale = tpg_scale
        self._tpg_applied_layers = tpg_applied_layers
        
        # Store original attention processors
        self._original_attn_processors = self._get_attention_processors()
        
        # Create TPG attention processors
        tpg_processors = {}
        for name, processor in self._original_attn_processors.items():
            # Check if this layer should have TPG applied
            should_apply_tpg = any(layer_type in name for layer_type in tpg_applied_layers)
            
            if should_apply_tpg:
                tpg_processors[name] = TPGAttentionProcessor(processor, perturbation_scale=self._tpg_scale)
            else:
                tpg_processors[name] = processor
        
        # Set the TPG processors
        self._set_attention_processors(tpg_processors)
        
        print(f"[TPG DEBUG] TPG enabled with scale {tpg_scale} on layers: {tpg_applied_layers}")
        for name, processor in tpg_processors.items():
            if isinstance(processor, TPGAttentionProcessor):
                print(f"[TPG DEBUG]   - Layer '{name}' set with TPGAttentionProcessor (perturbation_scale={processor.perturbation_scale})")
            else:
                print(f"[TPG DEBUG]   - Layer '{name}' set with original processor")
        logger.info(f"TPG enabled with scale {tpg_scale} on layers: {tpg_applied_layers}")
    
    def disable_tpg(self):
        """Disable Token Perturbation Guidance"""
        if self._original_attn_processors:
            self._set_attention_processors(self._original_attn_processors)
            self._original_attn_processors = {}
        
        self._tpg_scale = 0.0
        self._tpg_applied_layers = []
        
        logger.info("TPG disabled")
    
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
        tpg_applied_layers: List[str] = None,
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
        original_size: Optional[tuple[int, int]] = None,
        crops_coords_top_left: tuple[int, int] = (0, 0),
        target_size: Optional[tuple[int, int]] = None,
        negative_original_size: Optional[tuple[int, int]] = None,
        negative_crops_coords_top_left: tuple[int, int] = (0, 0),
        negative_target_size: Optional[tuple[int, int]] = None,
        clip_skip: Optional[int] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        **kwargs,
    ):
        """
        Generate images using Stable Diffusion XL with optional TPG
        """
        
        # Enable TPG if requested
        if tpg_scale > 0:
            print(f"[TPG DEBUG] Enabling TPG with scale={tpg_scale}, layers={tpg_applied_layers}")
            self.enable_tpg(tpg_scale=tpg_scale, tpg_applied_layers=tpg_applied_layers)
        
        # Store original UNet forward method
        original_unet_forward = None
        if hasattr(self.unet, 'forward'):
            original_unet_forward = self.unet.forward
        elif hasattr(self.unet, '__call__'):
            original_unet_forward = self.unet.__call__
        
        def tpg_unet_forward(sample, timestep, encoder_hidden_states, **kwargs):
            """Modified UNet forward that handles TPG guidance"""
            if tpg_scale > 0 and encoder_hidden_states.shape[0] == 2:
                # Duplicate the conditional part for TPG
                uncond_embeds, cond_embeds = encoder_hidden_states.chunk(2)
                encoder_hidden_states = torch.cat([uncond_embeds, cond_embeds, cond_embeds], dim=0)
                
                # Duplicate latents accordingly
                if sample.shape[0] == 2:
                    uncond_sample, cond_sample = sample.chunk(2)
                    sample = torch.cat([uncond_sample, cond_sample, cond_sample], dim=0)
                
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
                    noise_pred_uncond, noise_pred_cond, noise_pred_perturb = noise_pred.chunk(3)
                    
                    # First apply CFG
                    noise_pred_cfg = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
                    
                    # Then apply TPG
                    noise_pred_final = noise_pred_cfg + tpg_scale * (noise_pred_cond - noise_pred_perturb)
                    
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
            
            # Always disable TPG after generation to clean up
            if tpg_scale > 0:
                print("[TPG DEBUG] Disabling TPG after generation.")
                self.disable_tpg()


class TPGAttentionProcessor:
    """
    Attention processor that implements Token Perturbation Guidance
    """
    
    def __init__(self, original_processor, perturbation_scale: float = 1.0):
        self.original_processor = original_processor
        self.perturbation_scale = perturbation_scale
        print(f"[TPG DEBUG] TPGAttentionProcessor initialized with perturbation_scale={self.perturbation_scale}")
    
    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        scale: float = 1.0,
    ):
        """
        Apply token perturbation guidance
        """
        # Get the batch size and check if we're doing guidance
        batch_size = hidden_states.shape[0]
        
        # For TPG, we expect batch_size to be 3 (unconditional + conditional + perturbed)
        # We'll apply perturbation to the third part
        
        if batch_size >= 3 and self.perturbation_scale > 0:
            # Split into unconditional, conditional, and perturbed parts
            if batch_size == 3:
                hidden_states_uncond, hidden_states_cond, hidden_states_perturb = hidden_states.chunk(3)
                if encoder_hidden_states is not None:
                    encoder_hidden_states_uncond, encoder_hidden_states_cond, encoder_hidden_states_perturb = encoder_hidden_states.chunk(3)
                else:
                    encoder_hidden_states_uncond = encoder_hidden_states_cond = encoder_hidden_states_perturb = None
            else:
                # Handle batch_size > 3 by taking first as uncond, second as cond, rest as perturb
                hidden_states_uncond = hidden_states[:1]
                hidden_states_cond = hidden_states[1:2]
                hidden_states_perturb = hidden_states[2:]
                if encoder_hidden_states is not None:
                    encoder_hidden_states_uncond = encoder_hidden_states[:1]
                    encoder_hidden_states_cond = encoder_hidden_states[1:2]
                    encoder_hidden_states_perturb = encoder_hidden_states[2:]
                else:
                    encoder_hidden_states_uncond = encoder_hidden_states_cond = encoder_hidden_states_perturb = None
            
            # Process unconditional normally
            out_uncond = self.original_processor(
                attn, hidden_states_uncond, encoder_hidden_states_uncond, attention_mask, temb, scale
            )
            
            # Process conditional normally
            out_cond = self.original_processor(
                attn, hidden_states_cond, encoder_hidden_states_cond, attention_mask, temb, scale
            )
            
            # Process perturbed with token perturbation
            out_perturb = self._process_with_perturbation(
                attn, hidden_states_perturb, encoder_hidden_states_perturb, attention_mask, temb, scale
            )
            
            # Combine outputs
            return torch.cat([out_uncond, out_cond, out_perturb], dim=0)
        
        else:
            # Standard processing without TPG
            return self.original_processor(
                attn, hidden_states, encoder_hidden_states, attention_mask, temb, scale
            )
    
    def _process_with_perturbation(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        scale: float = 1.0,
    ):
        """
        Process attention with token perturbation applied
        """
        print(f"[TPG DEBUG] _process_with_perturbation called with perturbation_scale={self.perturbation_scale}")
        
        # Apply token perturbation to encoder_hidden_states (text embeddings)
        if encoder_hidden_states is not None and self.perturbation_scale > 0:
            print(f"[TPG DEBUG]   Applying token perturbation")
            print(f"[TPG DEBUG]   Encoder hidden states BEFORE perturbation (mean, std): {encoder_hidden_states.mean():.4f}, {encoder_hidden_states.std():.4f}")
            
            # Method 1: Token shuffling - randomly permute token order
            if encoder_hidden_states.shape[1] > 1:  # Only shuffle if we have more than 1 token
                batch_size, seq_len, hidden_dim = encoder_hidden_states.shape
                
                # Create random permutation for each batch item
                perturbed_encoder_hidden_states = encoder_hidden_states.clone()
                for b in range(batch_size):
                    # Generate random permutation
                    perm = torch.randperm(seq_len, device=encoder_hidden_states.device)
                    # Apply perturbation with scaling
                    if self.perturbation_scale < 1.0:
                        # Partial shuffling - interpolate between original and shuffled
                        shuffled = encoder_hidden_states[b, perm]
                        perturbed_encoder_hidden_states[b] = (
                            (1 - self.perturbation_scale) * encoder_hidden_states[b] + 
                            self.perturbation_scale * shuffled
                        )
                    else:
                        # Full shuffling
                        perturbed_encoder_hidden_states[b] = encoder_hidden_states[b, perm]
                
                encoder_hidden_states = perturbed_encoder_hidden_states
            
            print(f"[TPG DEBUG]   Encoder hidden states AFTER perturbation (mean, std): {encoder_hidden_states.mean():.4f}, {encoder_hidden_states.std():.4f}")
        
        # Process with perturbed encoder hidden states
        result = self.original_processor(
            attn, hidden_states, encoder_hidden_states, attention_mask, temb, scale
        )
        
        return result
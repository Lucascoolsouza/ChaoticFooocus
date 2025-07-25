#!/usr/bin/env python3
"""
Stable Diffusion XL PAG (Perturbed Attention Guidance) Pipeline
Based on the paper: "Self-Rectifying Diffusion Sampling with Perturbed-Attention Guidance"
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

class StableDiffusionXLPAGPipeline(StableDiffusionXLPipeline):
    """
    Stable Diffusion XL Pipeline with Perturbed Attention Guidance (PAG)
    
    PAG works by:
    1. Computing attention maps during the denoising process
    2. Perturbing these attention maps to create degraded predictions
    3. Using the difference to guide the generation away from degraded outputs
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._pag_scale = 0.0
        self._pag_applied_layers = []
        self._original_attn_processors = {}
        
    @property
    def pag_scale(self):
        """PAG guidance scale"""
        return self._pag_scale
    
    @pag_scale.setter
    def pag_scale(self, value):
        self._pag_scale = value
    
    @property
    def do_perturbed_attention_guidance(self):
        """Whether PAG is enabled"""
        return self._pag_scale > 0
    
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
    
    def enable_pag(self, pag_scale: float = 3.0, pag_applied_layers: List[str] = None):
        """
        Enable Perturbed Attention Guidance
        
        Args:
            pag_scale: Guidance scale for PAG (higher = stronger effect)
            pag_applied_layers: List of layer names to apply PAG to (None = all layers)
        """
        if pag_applied_layers is None:
            # Default to applying PAG to middle and up layers
            pag_applied_layers = ["mid", "up"]
        
        self._pag_scale = pag_scale
        self._pag_applied_layers = pag_applied_layers
        
        # Store original attention processors
        self._original_attn_processors = self._get_attention_processors()
        
        # Create PAG attention processors
        pag_processors = {}
        for name, processor in self._original_attn_processors.items():
            # Check if this layer should have PAG applied
            should_apply_pag = any(layer_type in name for layer_type in pag_applied_layers)
            
            if should_apply_pag:
                pag_processors[name] = PAGAttentionProcessor(processor, perturbation_scale=self._pag_scale)
            else:
                pag_processors[name] = processor
        
        # Set the PAG processors
        self._set_attention_processors(pag_processors)
        
        print(f"[PAG DEBUG] PAG enabled with scale {pag_scale} on layers: {pag_applied_layers}")
        for name, processor in pag_processors.items():
            if isinstance(processor, PAGAttentionProcessor):
                print(f"[PAG DEBUG]   - Layer '{name}' set with PAGAttentionProcessor (perturbation_scale={processor.perturbation_scale})")
            else:
                print(f"[PAG DEBUG]   - Layer '{name}' set with original processor")
        logger.info(f"PAG enabled with scale {pag_scale} on layers: {pag_applied_layers}")
    
    def disable_pag(self):
        """Disable Perturbed Attention Guidance"""
        if self._original_attn_processors:
            self._set_attention_processors(self._original_attn_processors)
            self._original_attn_processors = {}
        
        self._pag_scale = 0.0
        self._pag_applied_layers = []
        
        logger.info("PAG disabled")
    
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
        pag_scale: float = 0.0,
        pag_applied_layers: List[str] = None,
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
        Generate images using Stable Diffusion XL with optional PAG
        """
        
        # Enable PAG if requested
        if pag_scale > 0:
            print(f"[PAG DEBUG] Enabling PAG with scale={pag_scale}, layers={pag_applied_layers}")
            self.enable_pag(pag_scale=pag_scale, pag_applied_layers=pag_applied_layers)
        
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
            # Always disable PAG after generation to clean up
            if pag_scale > 0:
                print("[PAG DEBUG] Disabling PAG after generation.")
                self.disable_pag()


class PAGAttentionProcessor:
    """
    Attention processor that implements Perturbed Attention Guidance
    """
    
    def __init__(self, original_processor, perturbation_scale: float = 1.0):
        self.original_processor = original_processor
        self.perturbation_scale = perturbation_scale
        print(f"[PAG DEBUG] PAGAttentionProcessor initialized with perturbation_scale={self.perturbation_scale}")
    
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
        Apply perturbed attention guidance
        """
        # Get the batch size and check if we're doing guidance
        batch_size = hidden_states.shape[0]
        print(f"[PAG DEBUG] PAGAttentionProcessor.__call__ - batch_size={batch_size}")
        
        # For PAG, we expect batch_size to be 2 (unconditional + conditional)
        # or 3 (unconditional + conditional + perturbed) when PAG is active
        
        if batch_size == 3:
            print("[PAG DEBUG]   Processing with 3 chunks (uncond, cond, perturb)")
            # Split into unconditional, conditional, and perturbed
            hidden_states_uncond, hidden_states_cond, hidden_states_perturb = hidden_states.chunk(3)
            
            if encoder_hidden_states is not None:
                encoder_hidden_states_uncond, encoder_hidden_states_cond, encoder_hidden_states_perturb = encoder_hidden_states.chunk(3)
            else:
                encoder_hidden_states_uncond = encoder_hidden_states_cond = encoder_hidden_states_perturb = None
            
            # Process unconditional
            out_uncond = self.original_processor(
                attn, hidden_states_uncond, encoder_hidden_states_uncond, attention_mask, temb, scale
            )
            
            # Process conditional
            out_cond = self.original_processor(
                attn, hidden_states_cond, encoder_hidden_states_cond, attention_mask, temb, scale
            )
            
            # Process perturbed (with attention perturbation)
            out_perturb = self._process_with_perturbation(
                attn, hidden_states_perturb, encoder_hidden_states_perturb, attention_mask, temb, scale
            )
            
            # Combine outputs
            return torch.cat([out_uncond, out_cond, out_perturb], dim=0)
        
        else:
            print("[PAG DEBUG]   Processing without PAG (batch_size != 3)")
            # Standard processing without PAG
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
        Process attention with perturbation applied
        """
        print(f"[PAG DEBUG] _process_with_perturbation called with perturbation_scale={self.perturbation_scale}")
        # Store original forward method
        original_get_attention_scores = attn.get_attention_scores
        
        def perturbed_get_attention_scores(query, key, attention_mask=None):
            # Get normal attention scores
            attention_scores = original_get_attention_scores(query, key, attention_mask)
            
            # Apply perturbation to attention scores
            # Simple perturbation: add noise or modify the attention pattern
            if self.perturbation_scale > 0:
                print(f"[PAG DEBUG]   Applying perturbation with scale={self.perturbation_scale}")
                # Method 1: Add noise to attention scores
                noise = torch.randn_like(attention_scores) * self.perturbation_scale * 0.1
                attention_scores = attention_scores + noise
                
                # Method 2: Alternatively, you could blur or modify the attention pattern
                # attention_scores = self._blur_attention(attention_scores)
            
            return attention_scores
        
        # Temporarily replace the attention scoring function
        attn.get_attention_scores = perturbed_get_attention_scores
        
        try:
            # Process with perturbed attention
            result = self.original_processor(
                attn, hidden_states, encoder_hidden_states, attention_mask, temb, scale
            )
        finally:
            # Restore original function
            attn.get_attention_scores = original_get_attention_scores
        
        return result
    
    def _blur_attention(self, attention_scores):
        """
        Apply blur to attention scores as a form of perturbation
        """
        # Simple blur using average pooling
        batch_size, num_heads, seq_len, seq_len = attention_scores.shape
        
        # Apply a simple blur kernel
        kernel_size = 3
        padding = kernel_size // 2
        
        # Reshape for conv2d
        attention_scores_reshaped = attention_scores.view(batch_size * num_heads, 1, seq_len, seq_len)
        
        # Create blur kernel
        blur_kernel = torch.ones(1, 1, kernel_size, kernel_size, device=attention_scores.device)
        blur_kernel = blur_kernel / (kernel_size * kernel_size)
        
        # Apply blur
        blurred = F.conv2d(attention_scores_reshaped, blur_kernel, padding=padding)
        
        # Reshape back
        blurred = blurred.view(batch_size, num_heads, seq_len, seq_len)
        
        return blurred
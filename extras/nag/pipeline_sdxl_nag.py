# NAG (Normalized Attention Guidance) Pipeline for Fooocus
# Based on TPG pipeline structure for consistency

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import math
import torch
import torch.nn as nn
import logging

from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.image_processor import PipelineImageInput
from diffusers.utils import (
    deprecate,
    is_torch_xla_available,
)
from diffusers.pipelines.stable_diffusion_xl.pipeline_output import StableDiffusionXLPipelineOutput

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm
    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False

from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import (
    StableDiffusionXLPipeline,
    retrieve_timesteps,
    rescale_noise_cfg,
)

from .attention_nag import NAGAttnProcessor2_0

import numpy as np
from PIL import Image, ImageDraw

logger = logging.getLogger(__name__)

class NAGSampler:
    """
    NAG (Normalized Attention Guidance) sampler that integrates with Fooocus ksampler
    Similar to TPG but uses attention normalization instead of token perturbation
    """
    
    def __init__(self, nag_scale=1.0, nag_tau=2.5, nag_alpha=0.5):
        self.nag_scale = nag_scale
        self.nag_tau = nag_tau
        self.nag_alpha = nag_alpha
        self.original_sampling_function = None
        self.is_active = False
        self.original_attn_processors = {}
    
    def activate(self, unet):
        """Activate NAG by patching attention processors"""
        if self.is_active:
            return
        
        print(f"[NAG] Activating NAG with scale {self.nag_scale}")
        
        try:
            self.unet = unet
            
            # Store original attention processors
            if hasattr(unet, 'attn_processors'):
                self.original_attn_processors = unet.attn_processors.copy()
            elif hasattr(unet, 'model') and hasattr(unet.model, 'attn_processors'):
                self.original_attn_processors = unet.model.attn_processors.copy()
            
            # Set NAG attention processors
            self._set_nag_attn_processor()
            
            self.is_active = True
            print("[NAG] Successfully patched attention processors")
            
        except Exception as e:
            print(f"[NAG] Failed to patch attention processors: {e}")
            return
    
    def deactivate(self):
        """Deactivate NAG by restoring original attention processors"""
        if not self.is_active:
            return
        
        print("[NAG] Deactivating NAG")
        
        try:
            # Restore original attention processors
            if self.original_attn_processors and hasattr(self, 'unet'):
                if hasattr(self.unet, 'set_attn_processor'):
                    self.unet.set_attn_processor(self.original_attn_processors)
                elif hasattr(self.unet, 'model') and hasattr(self.unet.model, 'set_attn_processor'):
                    self.unet.model.set_attn_processor(self.original_attn_processors)
                print("[NAG] Successfully restored original attention processors")
        except Exception as e:
            print(f"[NAG] Failed to restore attention processors: {e}")
        
        self.is_active = False
    
    def _set_nag_attn_processor(self):
        """Set NAG attention processors"""
        if not hasattr(self, 'unet'):
            return
        
        # Handle different UNet structures (Diffusers vs ldm_patched)
        unet_to_modify = None
        if hasattr(self.unet, 'set_attn_processor'):
            unet_to_modify = self.unet
        elif hasattr(self.unet, 'model') and hasattr(self.unet.model, 'set_attn_processor'):
            unet_to_modify = self.unet.model
        
        if unet_to_modify is not None:
            attn_procs = {}
            existing_processors = getattr(unet_to_modify, 'attn_processors', {})
            
            for name, origin_attn_processor in existing_processors.items():
                if "attn2" in name:  # Cross-attention layers
                    attn_procs[name] = NAGAttnProcessor2_0(
                        nag_scale=self.nag_scale, 
                        nag_tau=self.nag_tau, 
                        nag_alpha=self.nag_alpha
                    )
                else:
                    attn_procs[name] = origin_attn_processor
            
            unet_to_modify.set_attn_processor(attn_procs)
        else:
            print("[NAG] Warning: Could not set attention processors - UNet structure not recognized")

# Global NAG sampler instance
nag_sampler = NAGSampler()

def safe_decode(latents, vae, width=512, height=512):
    """Safe VAE decoding with error handling"""
    try:
        with torch.no_grad():
            # Handle different VAE types (ComfyUI vs Diffusers)
            if hasattr(vae, 'decode'):
                vae_decode_fn = vae.decode
            elif hasattr(vae, 'model') and hasattr(vae.model, 'decode'):
                vae_decode_fn = vae.model.decode
            else:
                raise ValueError("VAE doesn't have a decode method")
            
            # Ensure latents are on the correct device
            if hasattr(vae, 'device'):
                device = vae.device
            elif hasattr(vae, 'model') and hasattr(vae.model, 'parameters'):
                device = next(vae.model.parameters()).device
            else:
                device = latents.device
            
            latents = latents.to(device)
            
            # Use standard SDXL scaling factor
            scaling_factor = 0.13025
            if hasattr(vae, 'config') and hasattr(vae.config, 'scaling_factor'):
                scaling_factor = vae.config.scaling_factor
            
            # Scale latents
            scaled_latents = latents / scaling_factor
            
            # Decode latents
            decoded = vae_decode_fn(scaled_latents, return_dict=False)[0]
            
            # Ensure proper format [B, C, H, W]
            if decoded.dim() == 4 and decoded.shape[1] == 3:
                pass  # Already in correct format
            elif decoded.dim() == 3:
                decoded = decoded.unsqueeze(0)  # Add batch dimension
            
            # Clamp to valid range and convert to [0, 1]
            decoded = torch.clamp((decoded + 1.0) / 2.0, 0.0, 1.0)
            
            # Convert to PIL Image
            decoded_np = decoded.squeeze(0).permute(1, 2, 0).cpu().numpy()
            decoded_np = (decoded_np * 255).astype(np.uint8)
            
            return Image.fromarray(decoded_np)
        
    except Exception as e:
        print(f"[safe_decode] Error during decode: {e}")
        # Return a red error image
        error_img = Image.new("RGB", (width, height), color="red")
        return error_img

class NAGStableDiffusionXLPipeline(StableDiffusionXLPipeline):
    """
    SDXL Pipeline with NAG support, structured similar to TPG pipeline
    """
    
    @property
    def do_normalized_attention_guidance(self):
        return self._nag_scale > 1

    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1

    def maybe_convert_prompt(self, prompt, tokenizer):
        """Override to bypass textual inversion logic"""
        return prompt

    def encode_prompt(self, prompt, prompt_2=None, device=None, num_images_per_prompt=1, 
                     do_classifier_free_guidance=True, negative_prompt=None, negative_prompt_2=None, 
                     prompt_embeds=None, negative_prompt_embeds=None, pooled_prompt_embeds=None, 
                     negative_pooled_prompt_embeds=None, lora_scale=None, clip_skip=None):
        """Override encode_prompt to handle pre-computed embeddings"""
        
        if prompt_embeds is not None:
            # Use the pre-computed embeddings directly
            batch_size = prompt_embeds.shape[0]
            
            # Store the shape for later use with NAG embeddings
            self._current_prompt_embeds_shape = prompt_embeds.shape
            
            # Ensure embeddings are on the correct device
            prompt_embeds = prompt_embeds.to(device)
            if pooled_prompt_embeds is not None:
                pooled_prompt_embeds = pooled_prompt_embeds.to(device)
            
            # Handle negative embeddings
            if negative_prompt_embeds is not None:
                negative_prompt_embeds = negative_prompt_embeds.to(device)
            if negative_pooled_prompt_embeds is not None:
                negative_pooled_prompt_embeds = negative_pooled_prompt_embeds.to(device)
            
            return prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds
        
        # Handle simple prompts (like NAG negative prompts)
        if isinstance(prompt, str) and prompt.strip() != "":
            try:
                inputs = self.tokenizer(
                    prompt,
                    padding="max_length",
                    truncation=True,
                    max_length=self.tokenizer.model_max_length,
                    return_tensors="pt"
                ).to(device)

                output = self.text_encoder(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    output_hidden_states=True,
                    return_dict=True,
                )

                prompt_embeds = output.hidden_states[-2]
                pooled_prompt_embeds = output.pooled_output

                return prompt_embeds, None, pooled_prompt_embeds, None

            except Exception as e:
                print(f"[NAG] Error encoding NAG negative prompt '{prompt}': {e}")
                print("[NAG] Falling back to zero embeddings for NAG negative prompt")
                
                # Fall back to zero embeddings
                batch_size = num_images_per_prompt
                seq_len = 77  # Default CLIP sequence length
                embed_dim = 2048  # Default SDXL embedding dimension
                pooled_dim = 1280  # Standard SDXL pooled dimension

                if hasattr(self, '_current_prompt_embeds_shape') and self._current_prompt_embeds_shape is not None:
                    seq_len = self._current_prompt_embeds_shape[1]
                    embed_dim = self._current_prompt_embeds_shape[2]

                dummy_embeds = torch.zeros((batch_size, seq_len, embed_dim), device=device, 
                                         dtype=torch.float16 if device.type == 'cuda' else torch.float32)
                dummy_pooled = torch.zeros((batch_size, pooled_dim), device=device, 
                                         dtype=torch.float16 if device.type == 'cuda' else torch.float32)

                return dummy_embeds, None, dummy_pooled, None
                
        elif isinstance(prompt, str) and prompt.strip() == "":
            # Create zero embeddings for empty strings
            batch_size = num_images_per_prompt
            seq_len = 77
            embed_dim = 2048
            pooled_dim = 1280
            
            if hasattr(self, '_current_prompt_embeds_shape') and self._current_prompt_embeds_shape is not None:
                seq_len = self._current_prompt_embeds_shape[1]
                embed_dim = self._current_prompt_embeds_shape[2]
            
            dummy_embeds = torch.zeros((batch_size, seq_len, embed_dim), device=device, 
                                     dtype=torch.float16 if device.type == 'cuda' else torch.float32)
            dummy_pooled = torch.zeros((batch_size, pooled_dim), device=device, 
                                     dtype=torch.float16 if device.type == 'cuda' else torch.float32)
            
            return dummy_embeds, None, dummy_pooled, None
        
        # Fallback to parent method
        return super().encode_prompt(prompt, prompt_2, device, num_images_per_prompt, 
                                   do_classifier_free_guidance, negative_prompt, negative_prompt_2, 
                                   prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, 
                                   negative_pooled_prompt_embeds, lora_scale, clip_skip)

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        timesteps: List[int] = None,
        sigmas: List[float] = None,
        denoising_end: Optional[float] = None,
        guidance_scale: float = 5.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        pooled_prompt_embeds: Optional[torch.Tensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.Tensor] = None,
        ip_adapter_image: Optional[PipelineImageInput] = None,
        ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
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
        callback_on_step_end: Optional[Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        # NAG specific parameters
        nag_scale: float = 1.0,
        nag_tau: float = 5.0,
        nag_alpha: float = 0.5,
        nag_negative_prompt: str = None,
        nag_negative_prompt_embeds: Optional[torch.Tensor] = None,
        nag_end: float = 1.0,
        **kwargs,
    ):
        """
        NAG Pipeline call method - structured similar to TPG
        """
        
        callback = kwargs.pop("callback", None)
        callback_steps = kwargs.pop("callback_steps", None)

        if callback is not None:
            deprecate("callback", "1.0.0", "Passing `callback` as an input argument to `__call__` is deprecated")
        if callback_steps is not None:
            deprecate("callback_steps", "1.0.0", "Passing `callback_steps` as an input argument to `__call__` is deprecated")

        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        # Default height and width
        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        # Enable memory optimizations
        if hasattr(self.vae, "enable_tiling"):
            self.vae.enable_tiling()
        elif hasattr(self.vae, "model") and hasattr(self.vae.model, "enable_tiling"):
            self.vae.model.enable_tiling()
        
        if hasattr(self.unet, "enable_attention_slicing"):
            self.unet.enable_attention_slicing()
        elif hasattr(self.unet, "model") and hasattr(self.unet.model, "enable_attention_slicing"):
            self.unet.model.enable_attention_slicing()

        original_size = original_size or (height, width)
        target_size = target_size or (height, width)

        # Check inputs
        self.check_inputs(
            prompt, prompt_2, height, width, callback_steps, negative_prompt, negative_prompt_2,
            prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds,
            ip_adapter_image, ip_adapter_image_embeds, callback_on_step_end_tensor_inputs,
        )

        self._guidance_scale = guidance_scale
        self._guidance_rescale = guidance_rescale
        self._clip_skip = clip_skip
        self._cross_attention_kwargs = cross_attention_kwargs
        self._denoising_end = denoising_end
        self._interrupt = False
        self._nag_scale = nag_scale

        # Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        # Handle device detection
        if hasattr(self.unet, 'model') and hasattr(self.unet.model, 'parameters'):
            device = next(iter(self.unet.model.parameters())).device
        elif hasattr(self.unet, 'parameters'):
            device = next(iter(self.unet.parameters())).device
        else:
            device = self._execution_device

        # Encode input prompt
        lora_scale = (
            self.cross_attention_kwargs.get("scale", None) if self.cross_attention_kwargs is not None else None
        )

        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            do_classifier_free_guidance=self.do_classifier_free_guidance or self.do_normalized_attention_guidance,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            lora_scale=lora_scale,
            clip_skip=self.clip_skip,
        )

        # Handle NAG negative prompt embeddings
        if self.do_normalized_attention_guidance:
            nag_negative_pooled_prompt_embeds = None
            
            if nag_negative_prompt_embeds is None:
                # Priority 1: Use existing negative prompt embeddings
                if self.do_classifier_free_guidance and negative_prompt_embeds is not None:
                    nag_negative_prompt_embeds = negative_prompt_embeds
                    nag_negative_pooled_prompt_embeds = negative_pooled_prompt_embeds
                    print("[NAG] Using existing CFG negative prompt embeddings for NAG")
                    
                # Priority 2: Encode NAG negative prompt
                elif nag_negative_prompt is not None and nag_negative_prompt.strip() != "":
                    print(f"[NAG] Encoding NAG negative prompt: '{nag_negative_prompt}'")
                    try:
                        nag_negative_prompt_embeds, _, nag_negative_pooled_prompt_embeds, _ = self.encode_prompt(
                            prompt=nag_negative_prompt,
                            device=device,
                            num_images_per_prompt=num_images_per_prompt,
                            do_classifier_free_guidance=False,
                            lora_scale=lora_scale,
                            clip_skip=self.clip_skip,
                        )
                    except Exception as e:
                        print(f"[NAG] Failed to encode NAG negative prompt: {e}")
                        nag_negative_prompt_embeds = None
                        nag_negative_pooled_prompt_embeds = None
                        
                # Priority 3: Use regular negative prompt
                elif negative_prompt is not None and negative_prompt.strip() != "":
                    print(f"[NAG] Using regular negative prompt for NAG: '{negative_prompt}'")
                    if negative_prompt_embeds is not None:
                        nag_negative_prompt_embeds = negative_prompt_embeds
                        nag_negative_pooled_prompt_embeds = negative_pooled_prompt_embeds
                    else:
                        try:
                            nag_negative_prompt_embeds, _, nag_negative_pooled_prompt_embeds, _ = self.encode_prompt(
                                prompt=negative_prompt,
                                device=device,
                                num_images_per_prompt=num_images_per_prompt,
                                do_classifier_free_guidance=False,
                                lora_scale=lora_scale,
                                clip_skip=self.clip_skip,
                            )
                        except Exception as e:
                            print(f"[NAG] Failed to encode regular negative prompt: {e}")
                            nag_negative_prompt_embeds = None
                            nag_negative_pooled_prompt_embeds = None
                else:
                    print("[NAG] No negative prompt available, NAG will be less effective")
                    nag_negative_prompt_embeds = None
                    nag_negative_pooled_prompt_embeds = None

        # Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, timesteps, sigmas
        )

        # Prepare latent variables
        num_channels_latents = 4  # Standard SDXL latent channels
        
        if latents is not None and latents.std().item() > 0.01:
            latents = latents.to(device=device, dtype=prompt_embeds.dtype)
        else:
            latents = self.prepare_latents(
                batch_size * num_images_per_prompt,
                num_channels_latents,
                height,
                width,
                prompt_embeds.dtype,
                device,
                generator,
                latents,
            )

        # Prepare extra step kwargs
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # Prepare added time ids & embeddings
        add_text_embeds = pooled_prompt_embeds
        if self.text_encoder_2 is None:
            text_encoder_projection_dim = int(pooled_prompt_embeds.shape[-1])
        else:
            text_encoder_projection_dim = 1280  # Standard SDXL

        add_time_ids = self._get_add_time_ids(
            original_size,
            crops_coords_top_left,
            target_size,
            dtype=prompt_embeds.dtype,
            text_encoder_projection_dim=text_encoder_projection_dim,
        )
        
        if negative_original_size is not None and negative_target_size is not None:
            negative_add_time_ids = self._get_add_time_ids(
                negative_original_size,
                negative_crops_coords_top_left,
                negative_target_size,
                dtype=prompt_embeds.dtype,
                text_encoder_projection_dim=text_encoder_projection_dim,
            )
        else:
            negative_add_time_ids = add_time_ids

        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            add_text_embeds = torch.cat([negative_pooled_prompt_embeds, add_text_embeds], dim=0)
            add_time_ids = torch.cat([negative_add_time_ids, add_time_ids], dim=0)

        if self.do_normalized_attention_guidance and nag_negative_prompt_embeds is not None:
            # Concatenate for NAG
            prompt_embeds = torch.cat([prompt_embeds, nag_negative_prompt_embeds], dim=0)
            
            num_existing_batches = add_text_embeds.shape[0]
            num_nag_batches = nag_negative_prompt_embeds.shape[0]
            
            if nag_negative_pooled_prompt_embeds is not None:
                replicated_add_text_embeds = nag_negative_pooled_prompt_embeds
            else:
                replicated_add_text_embeds = add_text_embeds[:num_nag_batches] if num_existing_batches >= num_nag_batches else add_text_embeds.repeat(math.ceil(num_nag_batches / num_existing_batches), 1, 1)[:num_nag_batches]
            
            replicated_add_time_ids = add_time_ids[:num_nag_batches] if num_existing_batches >= num_nag_batches else add_time_ids.repeat(math.ceil(num_nag_batches / num_existing_batches), 1)[:num_nag_batches]

            add_text_embeds = torch.cat([add_text_embeds, replicated_add_text_embeds], dim=0)
            add_time_ids = torch.cat([add_time_ids, replicated_add_time_ids], dim=0)

        elif self.do_normalized_attention_guidance and nag_negative_prompt_embeds is None:
            print("[NAG] Warning: NAG is enabled but no negative embeddings available, disabling NAG")
            self._nag_scale = 1.0

        prompt_embeds = prompt_embeds.to(device)
        add_text_embeds = add_text_embeds.to(device)
        add_time_ids = add_time_ids.to(device).repeat(batch_size * num_images_per_prompt, 1)

        if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
            image_embeds = self.prepare_ip_adapter_image_embeds(
                ip_adapter_image,
                ip_adapter_image_embeds,
                device,
                batch_size * num_images_per_prompt,
                self.do_classifier_free_guidance,
            )

        # Denoising loop
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)

        # Apply denoising_end
        if (
            self.denoising_end is not None
            and isinstance(self.denoising_end, float)
            and self.denoising_end > 0
            and self.denoising_end < 1
        ):
            discrete_timestep_cutoff = int(round(1000 - (self.denoising_end * 1000)))
            num_inference_steps = len(list(filter(lambda ts: ts >= discrete_timestep_cutoff, timesteps)))
            timesteps = timesteps[:num_inference_steps]

        timestep_cond = None

        # Set up NAG attention processors
        if self.do_normalized_attention_guidance:
            origin_attn_procs = getattr(self.unet, 'attn_processors', {})
            nag_sampler.nag_scale = nag_scale
            nag_sampler.nag_tau = nag_tau
            nag_sampler.nag_alpha = nag_alpha
            nag_sampler.activate(self.unet)
            attn_procs_recovered = False

        self._num_timesteps = len(timesteps)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                # Expand latents for guidance
                latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents

                if self.do_normalized_attention_guidance and nag_negative_prompt_embeds is not None:
                    target_batch_size = prompt_embeds.shape[0]
                    current_latent_batch_size = latent_model_input.shape[0]

                    if target_batch_size > current_latent_batch_size:
                        num_latents_to_add = target_batch_size - current_latent_batch_size
                        latent_model_input = torch.cat([latent_model_input, latents[0:1].repeat(num_latents_to_add, 1, 1, 1)], dim=0)

                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # Check if we should disable NAG for this timestep
                if t < math.floor((1 - nag_end) * 999) and self.do_normalized_attention_guidance and not attn_procs_recovered:
                    if hasattr(self.unet, 'set_attn_processor'):
                        self.unet.set_attn_processor(origin_attn_procs)
                    elif hasattr(self.unet, 'model') and hasattr(self.unet.model, 'set_attn_processor'):
                        self.unet.model.set_attn_processor(origin_attn_procs)
                    prompt_embeds = prompt_embeds[:len(latent_model_input)]
                    attn_procs_recovered = True

                # Predict noise residual
                added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}
                if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
                    added_cond_kwargs["image_embeds"] = image_embeds

                # Handle different UNet interfaces
                if hasattr(self.unet, 'model') and hasattr(self.unet.model, 'apply_model'):
                    # ComfyUI wrapped model
                    comfy_kwargs = {}
                    if "text_embeds" in added_cond_kwargs and "time_ids" in added_cond_kwargs:
                        pooled_output = added_cond_kwargs["text_embeds"]
                        time_ids = added_cond_kwargs["time_ids"]
                        
                        # Ensure batch sizes match
                        target_batch_size = latent_model_input.shape[0]
                        if pooled_output.shape[0] != target_batch_size:
                            if pooled_output.shape[0] < target_batch_size:
                                repeat_factor = target_batch_size // pooled_output.shape[0]
                                remainder = target_batch_size % pooled_output.shape[0]
                                pooled_output = torch.cat([pooled_output.repeat(repeat_factor, 1)] + 
                                                        ([pooled_output[:remainder]] if remainder > 0 else []), dim=0)
                            else:
                                pooled_output = pooled_output[:target_batch_size]
                        
                        if time_ids.shape[0] != target_batch_size:
                            if time_ids.shape[0] < target_batch_size:
                                repeat_factor = target_batch_size // time_ids.shape[0]
                                remainder = target_batch_size % time_ids.shape[0]
                                time_ids = torch.cat([time_ids.repeat(repeat_factor, 1)] + 
                                                   ([time_ids[:remainder]] if remainder > 0 else []), dim=0)
                            else:
                                time_ids = time_ids[:target_batch_size]
                        
                        # Extract dimensions from time_ids
                        if time_ids.shape[-1] >= 6:
                            height = int(time_ids[0, 0].item())
                            width = int(time_ids[0, 1].item())
                            crop_h = int(time_ids[0, 2].item())
                            crop_w = int(time_ids[0, 3].item())
                            target_height = int(time_ids[0, 4].item())
                            target_width = int(time_ids[0, 5].item())
                            
                            comfy_kwargs.update({
                                "pooled_output": pooled_output,
                                "width": width,
                                "height": height,
                                "crop_w": crop_w,
                                "crop_h": crop_h,
                                "target_width": target_width,
                                "target_height": target_height,
                                "device": latent_model_input.device,
                            })
                    
                    # Ensure prompt_embeds batch size matches
                    target_batch_size = latent_model_input.shape[0]
                    if prompt_embeds.shape[0] != target_batch_size:
                        if prompt_embeds.shape[0] < target_batch_size:
                            repeat_factor = target_batch_size // prompt_embeds.shape[0]
                            remainder = target_batch_size % prompt_embeds.shape[0]
                            prompt_embeds = torch.cat([prompt_embeds.repeat(repeat_factor, 1, 1)] + 
                                                    ([prompt_embeds[:remainder]] if remainder > 0 else []), dim=0)
                        else:
                            prompt_embeds = prompt_embeds[:target_batch_size]
                    
                    noise_pred = self.unet.model.apply_model(
                        latent_model_input,
                        t,
                        c_crossattn=prompt_embeds,
                        **comfy_kwargs,
                    )
                else:
                    # Standard Diffusers UNet
                    noise_pred = self.unet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=prompt_embeds,
                        timestep_cond=timestep_cond,
                        cross_attention_kwargs=self.cross_attention_kwargs,
                        added_cond_kwargs=added_cond_kwargs,
                        return_dict=False,
                    )[0]

                # Perform guidance
                if self.do_classifier_free_guidance:
                    if noise_pred.shape[0] >= 2:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)
                    else:
                        print(f"[NAG] Warning: Expected at least 2 noise predictions for CFG, got {noise_pred.shape[0]}")
                
                # Handle NAG guidance
                if self.do_normalized_attention_guidance and noise_pred.shape[0] >= 3:
                    noise_pred_uncond, noise_pred_text, noise_pred_nag = noise_pred.chunk(3)
                    
                    # Apply CFG first
                    noise_pred_cfg = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)
                    
                    # Apply NAG guidance
                    noise_pred = noise_pred_cfg + self._nag_scale * (noise_pred_text - noise_pred_nag)

                if self.do_classifier_free_guidance and self.guidance_rescale > 0.0:
                    noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=self.guidance_rescale)

                # Compute previous noisy sample
                latents_dtype = latents.dtype
                
                # Handle different scheduler types
                if type(self.scheduler).__name__ == 'SchedulerWrapper':
                    # Simple denoising step for Fooocus SchedulerWrapper
                    step_ratio = (1000 - t) / 1000.0
                    step_size = 0.05 * (1.0 - step_ratio * 0.7)
                    latents = latents - step_size * noise_pred
                else:
                    # Use real scheduler
                    scheduler_output = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)
                    if hasattr(scheduler_output, 'prev_sample'):
                        latents = scheduler_output.prev_sample
                    elif isinstance(scheduler_output, tuple):
                        latents = scheduler_output[0]
                    else:
                        latents = scheduler_output

                # Callback handling
                if callback is not None:
                    try:
                        preview_img = safe_decode(latents[:1], self.vae, width=width, height=height)
                        preview_np = np.array(preview_img)
                        callback(i, t, preview_np)
                    except Exception as e:
                        print(f"[Preview Callback] Failed at step {i}: {e}")

                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        latents = latents.to(latents_dtype)

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)
                    add_text_embeds = callback_outputs.pop("add_text_embeds", add_text_embeds)
                    negative_pooled_prompt_embeds = callback_outputs.pop("negative_pooled_prompt_embeds", negative_pooled_prompt_embeds)
                    add_time_ids = callback_outputs.pop("add_time_ids", add_time_ids)

                # Progress bar update
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

                if XLA_AVAILABLE:
                    xm.mark_step()

        # Cleanup NAG processors
        if self.do_normalized_attention_guidance:
            nag_sampler.deactivate()

        # Return latents for Fooocus integration
        if not return_dict:
            if latents.shape[0] > 1:
                latents = latents[:1]  # Take only the first sample
            return (latents,)
        
        # For other use cases, decode and return images
        final_image = safe_decode(latents, self.vae, width=width, height=height)
        self.maybe_free_model_hooks()

        return StableDiffusionXLPipelineOutput(images=[final_image])
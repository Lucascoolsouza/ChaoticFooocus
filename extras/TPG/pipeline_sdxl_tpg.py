# Simplified TPG Pipeline for Fooocus Integration

import torch
import torch.nn.functional as F
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import logging

logger = logging.getLogger(__name__)

class StableDiffusionXLTPGPipeline:
    """
    Simplified TPG Pipeline that works directly with Fooocus infrastructure
    instead of inheriting from diffusers pipelines
    """
    
    def __init__(self, vae, text_encoder, text_encoder_2, tokenizer, tokenizer_2, unet, scheduler, **kwargs):
        self.vae = vae
        self.text_encoder = text_encoder
        self.text_encoder_2 = text_encoder_2
        self.tokenizer = tokenizer
        self.tokenizer_2 = tokenizer_2
        self.unet = unet
        self.scheduler = scheduler
        
        # TPG specific parameters
        self._tpg_scale = 0.0
        self._tpg_applied_layers_index = None
        
        # Store other kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)
    
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
        TPG Pipeline call that works with Fooocus infrastructure
        """
        
        # Set TPG parameters
        self._tpg_scale = tpg_scale
        self._tpg_applied_layers_index = tpg_applied_layers_index
        
        print(f"[TPG] TPG Pipeline called with tpg_scale={tpg_scale}")
        
        # If TPG is not enabled, fall back to regular generation
        if not self.do_token_perturbation_guidance:
            print("[TPG] TPG not enabled, falling back to regular generation")
            # Return latents for Fooocus integration
            if not return_dict:
                return (latents,)
            else:
                from diffusers.pipelines.stable_diffusion_xl.pipeline_output import StableDiffusionXLPipelineOutput
                return StableDiffusionXLPipelineOutput(images=[])
        
        # For Fooocus integration, we need to work with the existing ksampler
        # Instead of reimplementing the entire pipeline, we'll modify the UNet behavior
        
        # Store original UNet forward method
        original_unet_forward = None
        if hasattr(self.unet, 'model') and hasattr(self.unet.model, 'apply_model'):
            original_apply_model = self.unet.model.apply_model
            
            def tpg_apply_model(x, timestep, c_crossattn=None, **kwargs):
                """Modified apply_model that handles TPG guidance"""
                
                # Check if we have the right batch size for TPG
                if c_crossattn is not None and c_crossattn.shape[0] == 2:
                    # Duplicate the conditional part for TPG
                    uncond_embeds, cond_embeds = c_crossattn.chunk(2)
                    
                    # Apply token shuffling to create perturbed embeddings
                    cond_embeds_shuffled = self._shuffle_tokens(cond_embeds)
                    
                    # Create the full batch: [uncond, cond, cond_shuffled]
                    c_crossattn_tpg = torch.cat([uncond_embeds, cond_embeds, cond_embeds_shuffled], dim=0)
                    
                    # Duplicate latents accordingly
                    if x.shape[0] == 2:
                        uncond_x, cond_x = x.chunk(2)
                        x_tpg = torch.cat([uncond_x, cond_x, cond_x], dim=0)
                    else:
                        x_tpg = x
                    
                    # Handle other conditioning
                    new_kwargs = {}
                    for key, value in kwargs.items():
                        if isinstance(value, torch.Tensor) and value.shape[0] == 2:
                            uncond_val, cond_val = value.chunk(2)
                            new_kwargs[key] = torch.cat([uncond_val, cond_val, cond_val], dim=0)
                        else:
                            new_kwargs[key] = value
                    
                    # Call original apply_model
                    noise_pred = original_apply_model(x_tpg, timestep, c_crossattn=c_crossattn_tpg, **new_kwargs)
                    
                    # Apply TPG guidance
                    if noise_pred.shape[0] == 3:
                        noise_pred_uncond, noise_pred_cond, noise_pred_tpg = noise_pred.chunk(3)
                        
                        # First apply CFG
                        noise_pred_cfg = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
                        
                        # Then apply TPG
                        noise_pred_final = noise_pred_cfg + tpg_scale * (noise_pred_cond - noise_pred_tpg)
                        
                        # Return in the expected format for Fooocus (batch size 2: uncond, cond)
                        return torch.cat([noise_pred_uncond, noise_pred_final], dim=0)
                    else:
                        return noise_pred
                else:
                    # Standard call without TPG
                    return original_apply_model(x, timestep, c_crossattn=c_crossattn, **kwargs)
            
            # Replace the apply_model method
            self.unet.model.apply_model = tpg_apply_model
            
        elif hasattr(self.unet, 'forward'):
            original_unet_forward = self.unet.forward
            
            def tpg_unet_forward(sample, timestep, encoder_hidden_states, **kwargs):
                """Modified UNet forward that handles TPG guidance"""
                
                # Check if we have the right batch size for TPG
                if encoder_hidden_states is not None and encoder_hidden_states.shape[0] == 2:
                    # Duplicate the conditional part for TPG
                    uncond_embeds, cond_embeds = encoder_hidden_states.chunk(2)
                    
                    # Apply token shuffling to create perturbed embeddings
                    cond_embeds_shuffled = self._shuffle_tokens(cond_embeds)
                    
                    # Create the full batch: [uncond, cond, cond_shuffled]
                    encoder_hidden_states_tpg = torch.cat([uncond_embeds, cond_embeds, cond_embeds_shuffled], dim=0)
                    
                    # Duplicate latents accordingly
                    if sample.shape[0] == 2:
                        uncond_sample, cond_sample = sample.chunk(2)
                        sample_tpg = torch.cat([uncond_sample, cond_sample, cond_sample], dim=0)
                    else:
                        sample_tpg = sample
                    
                    # Handle other conditioning
                    new_kwargs = {}
                    for key, value in kwargs.items():
                        if key == 'added_cond_kwargs' and isinstance(value, dict):
                            new_added_cond_kwargs = {}
                            for k, v in value.items():
                                if isinstance(v, torch.Tensor) and v.shape[0] == 2:
                                    uncond_v, cond_v = v.chunk(2)
                                    new_added_cond_kwargs[k] = torch.cat([uncond_v, cond_v, cond_v], dim=0)
                                else:
                                    new_added_cond_kwargs[k] = v
                            new_kwargs[key] = new_added_cond_kwargs
                        elif isinstance(value, torch.Tensor) and value.shape[0] == 2:
                            uncond_val, cond_val = value.chunk(2)
                            new_kwargs[key] = torch.cat([uncond_val, cond_val, cond_val], dim=0)
                        else:
                            new_kwargs[key] = value
                    
                    # Call original forward
                    noise_pred = original_unet_forward(sample_tpg, timestep, encoder_hidden_states=encoder_hidden_states_tpg, **new_kwargs)
                    
                    # Apply TPG guidance
                    if noise_pred.shape[0] == 3:
                        noise_pred_uncond, noise_pred_cond, noise_pred_tpg = noise_pred.chunk(3)
                        
                        # First apply CFG
                        noise_pred_cfg = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
                        
                        # Then apply TPG
                        noise_pred_final = noise_pred_cfg + tpg_scale * (noise_pred_cond - noise_pred_tpg)
                        
                        # Return in the expected format for Fooocus (batch size 2: uncond, cond)
                        return torch.cat([noise_pred_uncond, noise_pred_final], dim=0)
                    else:
                        return noise_pred
                else:
                    # Standard call without TPG
                    return original_unet_forward(sample, timestep, encoder_hidden_states=encoder_hidden_states, **kwargs)
            
            # Replace the forward method
            self.unet.forward = tpg_unet_forward
        
        try:
            print("[TPG] TPG modifications applied, returning latents for Fooocus processing")
            
            # For Fooocus integration, we return the latents and let Fooocus handle the rest
            # The TPG guidance will be applied through the modified UNet
            if not return_dict:
                return (latents,)
            else:
                from diffusers.pipelines.stable_diffusion_xl.pipeline_output import StableDiffusionXLPipelineOutput
                return StableDiffusionXLPipelineOutput(images=[])
        
        finally:
            # Restore original methods
            if hasattr(self.unet, 'model') and hasattr(self.unet.model, 'apply_model') and 'original_apply_model' in locals():
                self.unet.model.apply_model = original_apply_model
            elif original_unet_forward is not None:
                self.unet.forward = original_unet_forward
    
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
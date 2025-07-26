# PAG Sampler Integration for Fooocus

import torch
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)

class PAGSampler:
    """
    PAG (Perturbed Attention Guidance) sampler that integrates with Fooocus ksampler
    """
    
    def __init__(self, pag_scale=0.0, pag_applied_layers=None):
        self.pag_scale = pag_scale
        self.pag_applied_layers = pag_applied_layers or ["mid", "up"]
        self.original_unet_forward = None
        self.is_active = False
    
    def activate(self, unet):
        """Activate PAG by modifying the UNet forward method"""
        if self.is_active:
            return
        
        print(f"[PAG] Activating PAG with scale {self.pag_scale}")
        
        # Store original forward method
        if hasattr(unet, 'model') and hasattr(unet.model, 'apply_model'):
            self.original_unet_forward = unet.model.apply_model
            unet.model.apply_model = self._create_pag_apply_model(unet.model.apply_model)
        elif hasattr(unet, 'forward'):
            self.original_unet_forward = unet.forward
            unet.forward = self._create_pag_forward(unet.forward)
        
        self.unet = unet
        self.is_active = True
    
    def deactivate(self):
        """Deactivate PAG by restoring the original UNet forward method"""
        if not self.is_active or self.original_unet_forward is None:
            return
        
        print("[PAG] Deactivating PAG")
        
        # Restore original forward method
        if hasattr(self.unet, 'model') and hasattr(self.unet.model, 'apply_model'):
            self.unet.model.apply_model = self.original_unet_forward
        elif hasattr(self.unet, 'forward'):
            self.unet.forward = self.original_unet_forward
        
        self.original_unet_forward = None
        self.is_active = False
    
    def _create_pag_apply_model(self, original_apply_model):
        """Create PAG-modified apply_model for ComfyUI-style UNet"""
        def pag_apply_model(x, timestep, c_crossattn=None, **kwargs):
            # Check if we have the right batch size for PAG
            if c_crossattn is not None and c_crossattn.shape[0] == 2:
                # For PAG, we need to create a perturbed version of the conditioning
                uncond_embeds, cond_embeds = c_crossattn.chunk(2)
                
                # Create perturbed conditioning by applying attention perturbation
                cond_embeds_perturbed = self._apply_attention_perturbation(cond_embeds)
                
                # Create the full batch: [uncond, cond, cond_perturbed]
                c_crossattn_pag = torch.cat([uncond_embeds, cond_embeds, cond_embeds_perturbed], dim=0)
                
                # Duplicate latents accordingly
                if x.shape[0] == 2:
                    uncond_x, cond_x = x.chunk(2)
                    x_pag = torch.cat([uncond_x, cond_x, cond_x], dim=0)
                else:
                    x_pag = x
                
                # Handle other conditioning
                new_kwargs = {}
                for key, value in kwargs.items():
                    if isinstance(value, torch.Tensor) and value.shape[0] == 2:
                        uncond_val, cond_val = value.chunk(2)
                        new_kwargs[key] = torch.cat([uncond_val, cond_val, cond_val], dim=0)
                    else:
                        new_kwargs[key] = value
                
                # Call original apply_model
                noise_pred = original_apply_model(x_pag, timestep, c_crossattn=c_crossattn_pag, **new_kwargs)
                
                # Apply PAG guidance
                if noise_pred.shape[0] == 3:
                    noise_pred_uncond, noise_pred_cond, noise_pred_pag = noise_pred.chunk(3)
                    
                    # Apply PAG: enhance the difference between conditional and perturbed
                    noise_pred_enhanced = noise_pred_cond + self.pag_scale * (noise_pred_cond - noise_pred_pag)
                    
                    # Return in the expected format for Fooocus (batch size 2: uncond, enhanced_cond)
                    return torch.cat([noise_pred_uncond, noise_pred_enhanced], dim=0)
                else:
                    return noise_pred
            else:
                # Standard call without PAG
                return original_apply_model(x, timestep, c_crossattn=c_crossattn, **kwargs)
        
        return pag_apply_model
    
    def _create_pag_forward(self, original_forward):
        """Create PAG-modified forward for standard diffusers UNet"""
        def pag_forward(sample, timestep, encoder_hidden_states, **kwargs):
            # Check if we have the right batch size for PAG
            if encoder_hidden_states is not None and encoder_hidden_states.shape[0] == 2:
                # For PAG, we need to create a perturbed version of the conditioning
                uncond_embeds, cond_embeds = encoder_hidden_states.chunk(2)
                
                # Create perturbed conditioning by applying attention perturbation
                cond_embeds_perturbed = self._apply_attention_perturbation(cond_embeds)
                
                # Create the full batch: [uncond, cond, cond_perturbed]
                encoder_hidden_states_pag = torch.cat([uncond_embeds, cond_embeds, cond_embeds_perturbed], dim=0)
                
                # Duplicate latents accordingly
                if sample.shape[0] == 2:
                    uncond_sample, cond_sample = sample.chunk(2)
                    sample_pag = torch.cat([uncond_sample, cond_sample, cond_sample], dim=0)
                else:
                    sample_pag = sample
                
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
                noise_pred = original_forward(sample_pag, timestep, encoder_hidden_states=encoder_hidden_states_pag, **new_kwargs)
                
                # Apply PAG guidance
                if noise_pred.shape[0] == 3:
                    noise_pred_uncond, noise_pred_cond, noise_pred_pag = noise_pred.chunk(3)
                    
                    # Apply PAG: enhance the difference between conditional and perturbed
                    noise_pred_enhanced = noise_pred_cond + self.pag_scale * (noise_pred_cond - noise_pred_pag)
                    
                    # Return in the expected format for Fooocus (batch size 2: uncond, enhanced_cond)
                    return torch.cat([noise_pred_uncond, noise_pred_enhanced], dim=0)
                else:
                    return noise_pred
            else:
                # Standard call without PAG
                return original_forward(sample, timestep, encoder_hidden_states=encoder_hidden_states, **kwargs)
        
        return pag_forward
    
    def _apply_attention_perturbation(self, embeddings):
        """
        Apply attention perturbation for PAG by adding controlled noise to attention patterns
        """
        try:
            # Simple perturbation: add noise to the embeddings
            # This simulates perturbed attention by introducing controlled randomness
            perturbation_strength = 0.1 * self.pag_scale
            noise = torch.randn_like(embeddings) * perturbation_strength
            perturbed = embeddings + noise
            
            # Optional: apply blur to simulate attention degradation
            if self.pag_scale > 2.0:
                # Apply slight blur for stronger PAG
                if len(perturbed.shape) == 3:  # [batch, seq_len, dim]
                    # Simple smoothing across sequence dimension
                    kernel = torch.ones(3, device=perturbed.device) / 3.0
                    for b in range(perturbed.shape[0]):
                        for d in range(perturbed.shape[2]):
                            perturbed[b, :, d] = F.conv1d(
                                perturbed[b, :, d].unsqueeze(0).unsqueeze(0),
                                kernel.unsqueeze(0).unsqueeze(0),
                                padding=1
                            ).squeeze()
            
            return perturbed
        except Exception as e:
            logger.warning(f"Attention perturbation failed: {e}")
            return embeddings

# Global PAG sampler instance
pag_sampler = PAGSampler()
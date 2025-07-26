# NAG Sampler Integration for Fooocus

import torch
import logging

logger = logging.getLogger(__name__)

class NAGSampler:
    """
    NAG (Normalized Attention Guidance) sampler that integrates with Fooocus ksampler
    """
    
    def __init__(self, nag_scale=1.0, nag_tau=2.5, nag_alpha=0.5):
        self.nag_scale = nag_scale
        self.nag_tau = nag_tau
        self.nag_alpha = nag_alpha
        self.original_unet_forward = None
        self.is_active = False
    
    def activate(self, unet):
        """Activate NAG by modifying the UNet forward method"""
        if self.is_active:
            return
        
        print(f"[NAG] Activating NAG with scale {self.nag_scale}")
        
        # Store original forward method
        if hasattr(unet, 'model') and hasattr(unet.model, 'apply_model'):
            self.original_unet_forward = unet.model.apply_model
            unet.model.apply_model = self._create_nag_apply_model(unet.model.apply_model)
        elif hasattr(unet, 'forward'):
            self.original_unet_forward = unet.forward
            unet.forward = self._create_nag_forward(unet.forward)
        
        self.unet = unet
        self.is_active = True
    
    def deactivate(self):
        """Deactivate NAG by restoring the original UNet forward method"""
        if not self.is_active or self.original_unet_forward is None:
            return
        
        print("[NAG] Deactivating NAG")
        
        # Restore original forward method
        if hasattr(self.unet, 'model') and hasattr(self.unet.model, 'apply_model'):
            self.unet.model.apply_model = self.original_unet_forward
        elif hasattr(self.unet, 'forward'):
            self.unet.forward = self.original_unet_forward
        
        self.original_unet_forward = None
        self.is_active = False
    
    def _create_nag_apply_model(self, original_apply_model):
        """Create NAG-modified apply_model for ComfyUI-style UNet"""
        def nag_apply_model(x, timestep, c_crossattn=None, **kwargs):
            # Check if we have the right batch size for NAG
            if c_crossattn is not None and c_crossattn.shape[0] == 2:
                # For NAG, we need to create a degraded version of the conditioning
                uncond_embeds, cond_embeds = c_crossattn.chunk(2)
                
                # Create degraded conditioning by applying attention normalization
                cond_embeds_degraded = self._apply_attention_degradation(cond_embeds)
                
                # Create the full batch: [uncond, cond, cond_degraded]
                c_crossattn_nag = torch.cat([uncond_embeds, cond_embeds, cond_embeds_degraded], dim=0)
                
                # Duplicate latents accordingly
                if x.shape[0] == 2:
                    uncond_x, cond_x = x.chunk(2)
                    x_nag = torch.cat([uncond_x, cond_x, cond_x], dim=0)
                else:
                    x_nag = x
                
                # Handle other conditioning
                new_kwargs = {}
                for key, value in kwargs.items():
                    if isinstance(value, torch.Tensor) and value.shape[0] == 2:
                        uncond_val, cond_val = value.chunk(2)
                        new_kwargs[key] = torch.cat([uncond_val, cond_val, cond_val], dim=0)
                    else:
                        new_kwargs[key] = value
                
                # Call original apply_model
                noise_pred = original_apply_model(x_nag, timestep, c_crossattn=c_crossattn_nag, **new_kwargs)
                
                # Apply NAG guidance
                if noise_pred.shape[0] == 3:
                    noise_pred_uncond, noise_pred_cond, noise_pred_nag = noise_pred.chunk(3)
                    
                    # Apply NAG: enhance the difference between conditional and degraded
                    noise_pred_enhanced = noise_pred_cond + self.nag_scale * (noise_pred_cond - noise_pred_nag)
                    
                    # Return in the expected format for Fooocus (batch size 2: uncond, enhanced_cond)
                    return torch.cat([noise_pred_uncond, noise_pred_enhanced], dim=0)
                else:
                    return noise_pred
            else:
                # Standard call without NAG
                return original_apply_model(x, timestep, c_crossattn=c_crossattn, **kwargs)
        
        return nag_apply_model
    
    def _create_nag_forward(self, original_forward):
        """Create NAG-modified forward for standard diffusers UNet"""
        def nag_forward(sample, timestep, encoder_hidden_states, **kwargs):
            # Check if we have the right batch size for NAG
            if encoder_hidden_states is not None and encoder_hidden_states.shape[0] == 2:
                # For NAG, we need to create a degraded version of the conditioning
                uncond_embeds, cond_embeds = encoder_hidden_states.chunk(2)
                
                # Create degraded conditioning by applying attention normalization
                cond_embeds_degraded = self._apply_attention_degradation(cond_embeds)
                
                # Create the full batch: [uncond, cond, cond_degraded]
                encoder_hidden_states_nag = torch.cat([uncond_embeds, cond_embeds, cond_embeds_degraded], dim=0)
                
                # Duplicate latents accordingly
                if sample.shape[0] == 2:
                    uncond_sample, cond_sample = sample.chunk(2)
                    sample_nag = torch.cat([uncond_sample, cond_sample, cond_sample], dim=0)
                else:
                    sample_nag = sample
                
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
                noise_pred = original_forward(sample_nag, timestep, encoder_hidden_states=encoder_hidden_states_nag, **new_kwargs)
                
                # Apply NAG guidance
                if noise_pred.shape[0] == 3:
                    noise_pred_uncond, noise_pred_cond, noise_pred_nag = noise_pred.chunk(3)
                    
                    # Apply NAG: enhance the difference between conditional and degraded
                    noise_pred_enhanced = noise_pred_cond + self.nag_scale * (noise_pred_cond - noise_pred_nag)
                    
                    # Return in the expected format for Fooocus (batch size 2: uncond, enhanced_cond)
                    return torch.cat([noise_pred_uncond, noise_pred_enhanced], dim=0)
                else:
                    return noise_pred
            else:
                # Standard call without NAG
                return original_forward(sample, timestep, encoder_hidden_states=encoder_hidden_states, **kwargs)
        
        return nag_forward
    
    def _apply_attention_degradation(self, embeddings):
        """
        Apply attention degradation for NAG by normalizing attention patterns
        """
        try:
            # Simple degradation: reduce the magnitude of embeddings
            # This simulates degraded attention by making the conditioning less strong
            degraded = embeddings * (1.0 - self.nag_alpha)
            
            # Add some noise to further degrade the conditioning
            if self.nag_tau > 0:
                noise = torch.randn_like(embeddings) * (self.nag_tau * 0.01)
                degraded = degraded + noise
            
            return degraded
        except Exception as e:
            logger.warning(f"Attention degradation failed: {e}")
            return embeddings

# Global NAG sampler instance
nag_sampler = NAGSampler()
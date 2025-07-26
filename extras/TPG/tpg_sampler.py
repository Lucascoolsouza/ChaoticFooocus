# TPG Sampler Integration for Fooocus

import torch
import logging

logger = logging.getLogger(__name__)

class TPGSampler:
    """
    TPG (Token Perturbation Guidance) sampler that integrates with Fooocus ksampler
    """
    
    def __init__(self, tpg_scale=3.0, tpg_applied_layers_index=None):
        self.tpg_scale = tpg_scale
        self.tpg_applied_layers_index = tpg_applied_layers_index or []
        self.original_unet_forward = None
        self.is_active = False
    
    def activate(self, unet):
        """Activate TPG by modifying the UNet forward method"""
        if self.is_active:
            return
        
        print(f"[TPG] Activating TPG with scale {self.tpg_scale}")
        
        # Store original forward method
        if hasattr(unet, 'model') and hasattr(unet.model, 'apply_model'):
            self.original_unet_forward = unet.model.apply_model
            unet.model.apply_model = self._create_tpg_apply_model(unet.model.apply_model)
        elif hasattr(unet, 'forward'):
            self.original_unet_forward = unet.forward
            unet.forward = self._create_tpg_forward(unet.forward)
        
        self.unet = unet
        self.is_active = True
    
    def deactivate(self):
        """Deactivate TPG by restoring the original UNet forward method"""
        if not self.is_active or self.original_unet_forward is None:
            return
        
        print("[TPG] Deactivating TPG")
        
        # Restore original forward method
        if hasattr(self.unet, 'model') and hasattr(self.unet.model, 'apply_model'):
            self.unet.model.apply_model = self.original_unet_forward
        elif hasattr(self.unet, 'forward'):
            self.unet.forward = self.original_unet_forward
        
        self.original_unet_forward = None
        self.is_active = False
    
    def _create_tpg_apply_model(self, original_apply_model):
        """Create TPG-modified apply_model for ComfyUI-style UNet"""
        def tpg_apply_model(x, timestep, c_crossattn=None, **kwargs):
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
                    
                    # Apply TPG: enhance the difference between conditional and perturbed
                    noise_pred_enhanced = noise_pred_cond + self.tpg_scale * (noise_pred_cond - noise_pred_tpg)
                    
                    # Return in the expected format for Fooocus (batch size 2: uncond, enhanced_cond)
                    return torch.cat([noise_pred_uncond, noise_pred_enhanced], dim=0)
                else:
                    return noise_pred
            else:
                # Standard call without TPG
                return original_apply_model(x, timestep, c_crossattn=c_crossattn, **kwargs)
        
        return tpg_apply_model
    
    def _create_tpg_forward(self, original_forward):
        """Create TPG-modified forward for standard diffusers UNet"""
        def tpg_forward(sample, timestep, encoder_hidden_states, **kwargs):
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
                noise_pred = original_forward(sample_tpg, timestep, encoder_hidden_states=encoder_hidden_states_tpg, **new_kwargs)
                
                # Apply TPG guidance
                if noise_pred.shape[0] == 3:
                    noise_pred_uncond, noise_pred_cond, noise_pred_tpg = noise_pred.chunk(3)
                    
                    # Apply TPG: enhance the difference between conditional and perturbed
                    noise_pred_enhanced = noise_pred_cond + self.tpg_scale * (noise_pred_cond - noise_pred_tpg)
                    
                    # Return in the expected format for Fooocus (batch size 2: uncond, enhanced_cond)
                    return torch.cat([noise_pred_uncond, noise_pred_enhanced], dim=0)
                else:
                    return noise_pred
            else:
                # Standard call without TPG
                return original_forward(sample, timestep, encoder_hidden_states=encoder_hidden_states, **kwargs)
        
        return tpg_forward
    
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

# Global TPG sampler instance
tpg_sampler = TPGSampler()
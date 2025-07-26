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
        """Activate PAG by patching the sampling function"""
        if self.is_active:
            return
        
        print(f"[PAG] Activating PAG with scale {self.pag_scale}")
        
        # Import the sampling module
        try:
            import ldm_patched.modules.samplers as samplers
            
            # Store original sampling function if not already stored
            if not hasattr(self, '_original_sampling_function'):
                self._original_sampling_function = samplers.sampling_function
            
            # Replace with PAG-enhanced version
            samplers.sampling_function = self._create_pag_sampling_function(self._original_sampling_function)
            
            self.unet = unet
            self.is_active = True
            print("[PAG] Successfully patched sampling function")
            
        except Exception as e:
            print(f"[PAG] Failed to patch sampling function: {e}")
            return
    
    def deactivate(self):
        """Deactivate PAG by restoring the original sampling function"""
        if not self.is_active:
            return
        
        print("[PAG] Deactivating PAG")
        
        # Restore original sampling function
        try:
            import ldm_patched.modules.samplers as samplers
            if hasattr(self, '_original_sampling_function'):
                samplers.sampling_function = self._original_sampling_function
                print("[PAG] Successfully restored original sampling function")
        except Exception as e:
            print(f"[PAG] Failed to restore sampling function: {e}")
        
        self.is_active = False
    
    def _create_pag_sampling_function(self, original_sampling_function):
        """Create PAG-modified sampling function"""
        def pag_sampling_function(model, x, timestep, uncond, cond, cond_scale, model_options={}, seed=None):
            # Check if we should apply PAG
            if len(cond) > 0 and len(uncond) > 0:
                try:
                    # Create PAG conditioning by perturbing attention
                    pag_cond = []
                    for c in cond:
                        new_c = c.copy()
                        # Apply attention perturbation to text embeddings
                        if 'model_conds' in new_c:
                            for key, model_cond in new_c['model_conds'].items():
                                if hasattr(model_cond, 'cond') and isinstance(model_cond.cond, torch.Tensor):
                                    # Perturb attention for PAG
                                    perturbed_cond = self._apply_attention_perturbation(model_cond.cond)
                                    # Create a new model_cond with perturbed attention
                                    import copy
                                    new_model_cond = copy.deepcopy(model_cond)
                                    new_model_cond.cond = perturbed_cond
                                    new_c['model_conds'][key] = new_model_cond
                        pag_cond.append(new_c)
                    
                    # Get predictions for all three conditions
                    noise_pred_uncond = original_sampling_function(model, x, timestep, uncond, [], cond_scale, model_options, seed)
                    noise_pred_cond = original_sampling_function(model, x, timestep, [], cond, cond_scale, model_options, seed)
                    noise_pred_pag = original_sampling_function(model, x, timestep, [], pag_cond, cond_scale, model_options, seed)
                    
                    # Apply PAG guidance
                    noise_pred_enhanced = noise_pred_cond + self.pag_scale * (noise_pred_cond - noise_pred_pag)
                    
                    # Combine unconditional and enhanced conditional
                    return noise_pred_uncond + cond_scale * (noise_pred_enhanced - noise_pred_uncond)
                    
                except Exception as e:
                    print(f"[PAG] Error in PAG sampling, falling back to original: {e}")
                    return original_sampling_function(model, x, timestep, uncond, cond, cond_scale, model_options, seed)
            else:
                # Standard call without PAG
                return original_sampling_function(model, x, timestep, uncond, cond, cond_scale, model_options, seed)
        
        return pag_sampling_function
    
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
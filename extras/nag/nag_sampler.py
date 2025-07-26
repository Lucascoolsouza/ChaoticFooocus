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
        """Activate NAG by patching the sampling function"""
        if self.is_active:
            return
        
        print(f"[NAG] Activating NAG with scale {self.nag_scale}")
        
        # Import the sampling module
        try:
            import ldm_patched.modules.samplers as samplers
            
            # Store original sampling function if not already stored
            if not hasattr(self, '_original_sampling_function'):
                self._original_sampling_function = samplers.sampling_function
            
            # Replace with NAG-enhanced version
            samplers.sampling_function = self._create_nag_sampling_function(self._original_sampling_function)
            
            self.unet = unet
            self.is_active = True
            print("[NAG] Successfully patched sampling function")
            
        except Exception as e:
            print(f"[NAG] Failed to patch sampling function: {e}")
            return
    
    def deactivate(self):
        """Deactivate NAG by restoring the original sampling function"""
        if not self.is_active:
            return
        
        print("[NAG] Deactivating NAG")
        
        # Restore original sampling function
        try:
            import ldm_patched.modules.samplers as samplers
            if hasattr(self, '_original_sampling_function'):
                samplers.sampling_function = self._original_sampling_function
                print("[NAG] Successfully restored original sampling function")
        except Exception as e:
            print(f"[NAG] Failed to restore sampling function: {e}")
        
        self.is_active = False
    
    def _create_nag_sampling_function(self, original_sampling_function):
        """Create NAG-modified sampling function"""
        def nag_sampling_function(model, x, timestep, uncond, cond, cond_scale, model_options={}, seed=None):
            # Check if we should apply NAG
            if len(cond) > 0 and len(uncond) > 0:
                try:
                    # Create NAG conditioning by degrading attention
                    nag_cond = []
                    for c in cond:
                        new_c = c.copy()
                        # Apply attention degradation to text embeddings
                        if 'model_conds' in new_c:
                            for key, model_cond in new_c['model_conds'].items():
                                if hasattr(model_cond, 'cond') and isinstance(model_cond.cond, torch.Tensor):
                                    # Degrade attention for NAG
                                    degraded_cond = self._apply_attention_degradation(model_cond.cond)
                                    # Create a new model_cond with degraded attention
                                    import copy
                                    new_model_cond = copy.deepcopy(model_cond)
                                    new_model_cond.cond = degraded_cond
                                    new_c['model_conds'][key] = new_model_cond
                        nag_cond.append(new_c)
                    
                    # Get predictions for all three conditions
                    noise_pred_uncond = original_sampling_function(model, x, timestep, uncond, [], cond_scale, model_options, seed)
                    noise_pred_cond = original_sampling_function(model, x, timestep, [], cond, cond_scale, model_options, seed)
                    noise_pred_nag = original_sampling_function(model, x, timestep, [], nag_cond, cond_scale, model_options, seed)
                    
                    # Apply NAG guidance
                    noise_pred_enhanced = noise_pred_cond + self.nag_scale * (noise_pred_cond - noise_pred_nag)
                    
                    # Combine unconditional and enhanced conditional
                    return noise_pred_uncond + cond_scale * (noise_pred_enhanced - noise_pred_uncond)
                    
                except Exception as e:
                    print(f"[NAG] Error in NAG sampling, falling back to original: {e}")
                    return original_sampling_function(model, x, timestep, uncond, cond, cond_scale, model_options, seed)
            else:
                # Standard call without NAG
                return original_sampling_function(model, x, timestep, uncond, cond, cond_scale, model_options, seed)
        
        return nag_sampling_function
    
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
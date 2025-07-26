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
        """Activate TPG by patching the sampling function"""
        if self.is_active:
            return
        
        print(f"[TPG] Activating TPG with scale {self.tpg_scale}")
        
        # Import the sampling module
        try:
            import ldm_patched.modules.samplers as samplers
            
            # Store original sampling function if not already stored
            if not hasattr(self, '_original_sampling_function'):
                self._original_sampling_function = samplers.sampling_function
            
            # Replace with TPG-enhanced version
            samplers.sampling_function = self._create_tpg_sampling_function(self._original_sampling_function)
            
            self.unet = unet
            self.is_active = True
            print("[TPG] Successfully patched sampling function")
            
        except Exception as e:
            print(f"[TPG] Failed to patch sampling function: {e}")
            return
    
    def deactivate(self):
        """Deactivate TPG by restoring the original sampling function"""
        if not self.is_active:
            return
        
        print("[TPG] Deactivating TPG")
        
        # Restore original sampling function
        try:
            import ldm_patched.modules.samplers as samplers
            if hasattr(self, '_original_sampling_function'):
                samplers.sampling_function = self._original_sampling_function
                print("[TPG] Successfully restored original sampling function")
        except Exception as e:
            print(f"[TPG] Failed to restore sampling function: {e}")
        
        self.is_active = False
    
    def _create_tpg_sampling_function(self, original_sampling_function):
        """Create TPG-modified sampling function"""
        def tpg_sampling_function(model, x, timestep, uncond, cond, cond_scale, model_options={}, seed=None):
            # Check if we should apply TPG
            if len(cond) > 0 and len(uncond) > 0:
                try:
                    # Create TPG conditioning by shuffling tokens
                    tpg_cond = []
                    for c in cond:
                        new_c = c.copy()
                        # Apply token shuffling to text embeddings
                        if 'model_conds' in new_c:
                            for key, model_cond in new_c['model_conds'].items():
                                if hasattr(model_cond, 'cond') and isinstance(model_cond.cond, torch.Tensor):
                                    # Shuffle tokens for TPG
                                    shuffled_cond = self._shuffle_tokens(model_cond.cond)
                                    # Create a new model_cond with shuffled tokens
                                    import copy
                                    new_model_cond = copy.deepcopy(model_cond)
                                    new_model_cond.cond = shuffled_cond
                                    new_c['model_conds'][key] = new_model_cond
                        tpg_cond.append(new_c)
                    
                    # Get predictions for all three conditions
                    noise_pred_uncond = original_sampling_function(model, x, timestep, uncond, [], cond_scale, model_options, seed)
                    noise_pred_cond = original_sampling_function(model, x, timestep, [], cond, cond_scale, model_options, seed)
                    noise_pred_tpg = original_sampling_function(model, x, timestep, [], tpg_cond, cond_scale, model_options, seed)
                    
                    # Apply TPG guidance
                    noise_pred_enhanced = noise_pred_cond + self.tpg_scale * (noise_pred_cond - noise_pred_tpg)
                    
                    # Combine unconditional and enhanced conditional
                    return noise_pred_uncond + cond_scale * (noise_pred_enhanced - noise_pred_uncond)
                    
                except Exception as e:
                    print(f"[TPG] Error in TPG sampling, falling back to original: {e}")
                    return original_sampling_function(model, x, timestep, uncond, cond, cond_scale, model_options, seed)
            else:
                # Standard call without TPG
                return original_sampling_function(model, x, timestep, uncond, cond, cond_scale, model_options, seed)
        
        return tpg_sampling_function
    

    
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
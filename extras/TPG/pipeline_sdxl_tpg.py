# TPG Sampler Integration for Fooocus

import torch
import torch.nn.functional as F
import logging
import math

logger = logging.getLogger(__name__)

class TPGSampler:
    """
    TPG (Token Perturbation Guidance) sampler that integrates with Fooocus ksampler
    """
    
    def __init__(self, tpg_scale=0.0, tpg_applied_layers=None):
        self.tpg_scale = tpg_scale
        self.tpg_applied_layers = tpg_applied_layers or ["mid", "up"]
        self.original_sampling_function = None
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
                    # Create TPG conditioning by perturbing tokens
                    tpg_cond = []
                    for c in cond:
                        new_c = c.copy()
                        # Apply token perturbation to text embeddings
                        if 'model_conds' in new_c:
                            for key, model_cond in new_c['model_conds'].items():
                                if hasattr(model_cond, 'cond') and isinstance(model_cond.cond, torch.Tensor):
                                    # Perturb tokens for TPG
                                    perturbed_cond = self._apply_token_perturbation(model_cond.cond)
                                    # Create a new model_cond with perturbed tokens
                                    import copy
                                    new_model_cond = copy.deepcopy(model_cond)
                                    new_model_cond.cond = perturbed_cond
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
    
    def _apply_token_perturbation(self, embeddings):
        """
        Apply token perturbation for TPG by shuffling token order
        """
        try:
            # Method 1: Token shuffling - randomly permute token order
            if embeddings.shape[1] > 1:  # Only shuffle if we have more than 1 token
                batch_size, seq_len, hidden_dim = embeddings.shape
                
                # Create random permutation for each batch item
                perturbed_embeddings = embeddings.clone()
                for b in range(batch_size):
                    # Generate random permutation
                    perm = torch.randperm(seq_len, device=embeddings.device)
                    # Apply perturbation with scaling
                    if self.tpg_scale < 3.0:
                        # Partial shuffling - interpolate between original and shuffled
                        shuffled = embeddings[b, perm]
                        perturbation_strength = min(1.0, self.tpg_scale / 3.0)
                        perturbed_embeddings[b] = (
                            (1 - perturbation_strength) * embeddings[b] + 
                            perturbation_strength * shuffled
                        )
                    else:
                        # Full shuffling
                        perturbed_embeddings[b] = embeddings[b, perm]
                
                return perturbed_embeddings
            
            # Method 2: Add controlled noise if shuffling isn't applicable
            perturbation_strength = 0.1 * min(1.0, self.tpg_scale / 3.0)
            noise = torch.randn_like(embeddings) * perturbation_strength
            return embeddings + noise
            
        except Exception as e:
            logger.warning(f"Token perturbation failed: {e}")
            return embeddings

# Global TPG sampler instance
tpg_sampler = TPGSampler()

def shuffle_tokens(x, step=None, seed_offset=0, shuffle_strength=1.0):
    """Shuffle tokens for TPG - creates different shuffling at each step
    
    Args:
        x: Token embeddings to shuffle
        step: Current sampling step (for step-dependent shuffling)
        seed_offset: Offset for reproducible randomness
        shuffle_strength: How much to shuffle (0.0 = no shuffle, 1.0 = full shuffle)
    """
    try:
        if len(x.shape) >= 2:
            b, n = x.shape[:2]
            
            if shuffle_strength <= 0:
                return x
            
            # Create different shuffling for each step
            if step is not None:
                # Use step-based seed for reproducible but different shuffling each step
                generator = torch.Generator(device=x.device)
                generator.manual_seed(hash((step + seed_offset)) % (2**32))
                
                if shuffle_strength < 1.0:
                    # Partial shuffling: only shuffle a portion of tokens
                    num_to_shuffle = max(1, int(n * shuffle_strength))
                    indices_to_shuffle = torch.randperm(n, device=x.device, generator=generator)[:num_to_shuffle]
                    shuffled_indices = torch.randperm(num_to_shuffle, device=x.device, generator=generator)
                    
                    result = x.clone()
                    result[:, indices_to_shuffle] = x[:, indices_to_shuffle[shuffled_indices]]
                    return result
                else:
                    # Full shuffling
                    permutation = torch.randperm(n, device=x.device, generator=generator)
                    return x[:, permutation]
            else:
                # Random shuffling if no step provided
                if shuffle_strength < 1.0:
                    num_to_shuffle = max(1, int(n * shuffle_strength))
                    indices_to_shuffle = torch.randperm(n, device=x.device)[:num_to_shuffle]
                    shuffled_indices = torch.randperm(num_to_shuffle, device=x.device)
                    
                    result = x.clone()
                    result[:, indices_to_shuffle] = x[:, indices_to_shuffle[shuffled_indices]]
                    return result
                else:
                    permutation = torch.randperm(n, device=x.device)
                    return x[:, permutation]
                
        return x
    except Exception as e:
        print(f"[TPG] Token shuffling error: {e}")
        return x
    

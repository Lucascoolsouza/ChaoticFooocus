#!/usr/bin/env python3
"""
TPG Integration for Fooocus
Integrates Token Perturbation Guidance with Fooocus's existing pipeline infrastructure
"""

import torch
import logging
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)

# Global TPG configuration
_tpg_config = {
    'enabled': False,
    'scale': 3.0,
    'applied_layers': ['mid', 'up'],
    'shuffle_strength': 1.0,
    'adaptive_strength': True
}

# Global reference to the TPG pipeline instance
_tpg_pipeline = None
_original_unet_forward = None

def set_tpg_config(enabled: bool = False, scale: float = 3.0, applied_layers: List[str] = None, 
                   shuffle_strength: float = 1.0, adaptive_strength: bool = True):
    """Set global TPG configuration"""
    global _tpg_config
    if applied_layers is None:
        applied_layers = ['mid', 'up']
    
    _tpg_config.update({
        'enabled': enabled,
        'scale': scale,
        'applied_layers': applied_layers,
        'shuffle_strength': shuffle_strength,
        'adaptive_strength': adaptive_strength
    })
    
    if enabled:
        print(f"[TPG] TPG enabled with scale={scale}, layers={applied_layers}")
        print(f"[TPG] Shuffle strength={shuffle_strength}, adaptive={adaptive_strength}")
    else:
        print("[TPG] TPG disabled")

def get_tpg_config():
    """Get current TPG configuration"""
    return _tpg_config.copy()

def is_tpg_enabled():
    """Check if TPG is enabled"""
    return _tpg_config.get('enabled', False) and _tpg_config.get('scale', 0) > 0

def shuffle_tokens(x, step=None, seed_offset=0, shuffle_strength=None):
    """
    Shuffle tokens for TPG - creates different shuffling at each step
    
    Args:
        x: Token embeddings to shuffle [batch, seq_len, hidden_dim]
        step: Current sampling step (for step-dependent shuffling)
        seed_offset: Offset for reproducible randomness
        shuffle_strength: How much to shuffle (0.0 = no shuffle, 1.0 = full shuffle)
    """
    try:
        if shuffle_strength is None:
            shuffle_strength = _tpg_config.get('shuffle_strength', 1.0)
        
        if len(x.shape) < 2 or shuffle_strength <= 0:
            return x
        
        b, n = x.shape[:2]
        
        # Create different shuffling for each step
        if step is not None:
            # Use step-based seed for reproducible but different shuffling each step
            generator = torch.Generator(device=x.device)
            generator.manual_seed(hash((step + seed_offset)) % (2**32))
            
            # Adaptive shuffle strength based on sampling progress
            if _tpg_config.get('adaptive_strength', True) and step is not None:
                # Stronger shuffling early, weaker later
                progress = step / 50.0  # Assume ~50 steps
                adaptive_strength = shuffle_strength * (1.0 - 0.5 * min(1.0, progress))
            else:
                adaptive_strength = shuffle_strength
            
            if adaptive_strength < 1.0:
                # Partial shuffling: only shuffle a portion of tokens
                num_to_shuffle = max(1, int(n * adaptive_strength))
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
                
    except Exception as e:
        logger.warning(f"[TPG] Token shuffling error: {e}")
        return x

def apply_tpg_to_conditioning(cond_list, step=None):
    """
    Apply TPG token perturbation to conditioning
    
    Args:
        cond_list: List of conditioning dictionaries
        step: Current sampling step
    
    Returns:
        List of conditioning dictionaries with TPG applied
    """
    if not is_tpg_enabled():
        return cond_list
    
    try:
        tpg_cond = []
        for c in cond_list:
            new_c = c.copy()
            if 'model_conds' in new_c:
                for key, model_cond in new_c['model_conds'].items():
                    if hasattr(model_cond, 'cond') and isinstance(model_cond.cond, torch.Tensor):
                        # Create shuffled version
                        import copy
                        new_model_cond = copy.deepcopy(model_cond)
                        new_model_cond.cond = shuffle_tokens(
                            model_cond.cond,
                            step=step,
                            seed_offset=hash(str(model_cond.cond.shape)) % 1000
                        )
                        new_c['model_conds'][key] = new_model_cond
            tpg_cond.append(new_c)
        
        return tpg_cond
    
    except Exception as e:
        logger.warning(f"[TPG] Error applying TPG to conditioning: {e}")
        return cond_list

def create_tpg_sampling_function(original_sampling_function):
    """
    Create a TPG-enhanced sampling function that applies token perturbation guidance
    This wraps the sampling_function which has the signature:
    sampling_function(model, x, timestep, uncond, cond, cond_scale, model_options={}, seed=None)
    """
    def tpg_sampling_function(model, x, timestep, uncond, cond, cond_scale, model_options={}, seed=None):
        # Check if TPG should be applied
        if not is_tpg_enabled() or len(cond) == 0:
            return original_sampling_function(model, x, timestep, uncond, cond, cond_scale, model_options, seed)
        
        try:
            tpg_scale = _tpg_config.get('scale', 3.0)
            
            # Create perturbed conditioning by shuffling tokens
            tpg_cond = []
            for c in cond:
                new_c = c.copy()
                if 'model_conds' in new_c:
                    for key, model_cond in new_c['model_conds'].items():
                        if hasattr(model_cond, 'cond') and isinstance(model_cond.cond, torch.Tensor):
                            # Create shuffled version
                            import copy
                            new_model_cond = copy.deepcopy(model_cond)
                            
                            # Apply token shuffling
                            step = int(timestep.mean().item()) if hasattr(timestep, 'mean') else None
                            new_model_cond.cond = shuffle_tokens(
                                model_cond.cond,
                                step=step,
                                shuffle_strength=_tpg_config.get('shuffle_strength', 1.0)
                            )
                            new_c['model_conds'][key] = new_model_cond
                tpg_cond.append(new_c)
            
            # Get predictions for all conditions
            # 1. Standard CFG prediction
            cfg_result = original_sampling_function(model, x, timestep, uncond, cond, cond_scale, model_options, seed)
            
            # Import calc_cond_uncond_batch locally to avoid circular imports
            from ldm_patched.modules.samplers import calc_cond_uncond_batch
            
            # 2. Get conditional prediction without CFG
            cond_pred, _ = calc_cond_uncond_batch(model, cond, None, x, timestep, model_options)
            
            # 3. Get perturbed prediction
            tpg_pred, _ = calc_cond_uncond_batch(model, tpg_cond, None, x, timestep, model_options)
            
            # Apply TPG guidance
            # Enhance the CFG result using the difference between normal and perturbed conditional predictions
            tpg_enhanced = cfg_result + tpg_scale * (cond_pred - tpg_pred)
            
            return tpg_enhanced
            
        except Exception as e:
            logger.warning(f"[TPG] Error in TPG sampling, falling back to original: {e}")
            import traceback
            traceback.print_exc()
            return original_sampling_function(model, x, timestep, uncond, cond, cond_scale, model_options, seed)
    
    return tpg_sampling_function

def patch_sampling_for_tpg():
    """
    Patch the sampling function to include TPG support
    """
    if not is_tpg_enabled():
        return False
    
    try:
        import ldm_patched.modules.samplers as samplers
        
        global _original_unet_forward
        
        # Store original if not already stored
        if _original_unet_forward is None:
            _original_unet_forward = samplers.sampling_function
            samplers.sampling_function = create_tpg_sampling_function(_original_unet_forward)
            print("[TPG] Patched sampling_function for TPG")
        
        print("[TPG] Successfully patched sampling function for TPG")
        return True
        
    except Exception as e:
        logger.error(f"[TPG] Failed to patch sampling function: {e}")
        import traceback
        traceback.print_exc()
        return False

def unpatch_sampling_for_tpg():
    """
    Restore the original sampling function
    """
    try:
        import ldm_patched.modules.samplers as samplers
        
        global _original_unet_forward
        
        if _original_unet_forward is not None:
            samplers.sampling_function = _original_unet_forward
            _original_unet_forward = None
            print("[TPG] Successfully restored original sampling function")
            return True
        else:
            print("[TPG] No original sampling function found to restore")
            return False
            
    except Exception as e:
        logger.error(f"[TPG] Failed to restore sampling function: {e}")
        import traceback
        traceback.print_exc()
        return False

def enable_tpg(scale: float = 3.0, applied_layers: List[str] = None, 
               shuffle_strength: float = 1.0, adaptive_strength: bool = True):
    """
    Enable TPG with the specified parameters
    """
    set_tpg_config(
        enabled=True,
        scale=scale,
        applied_layers=applied_layers,
        shuffle_strength=shuffle_strength,
        adaptive_strength=adaptive_strength
    )
    
    return patch_sampling_for_tpg()

def disable_tpg():
    """
    Disable TPG and restore original sampling function
    """
    set_tpg_config(enabled=False)
    return unpatch_sampling_for_tpg()

# Context manager for temporary TPG usage
class TPGContext:
    """Context manager for temporary TPG activation"""
    
    def __init__(self, scale: float = 3.0, applied_layers: List[str] = None,
                 shuffle_strength: float = 1.0, adaptive_strength: bool = True):
        self.scale = scale
        self.applied_layers = applied_layers
        self.shuffle_strength = shuffle_strength
        self.adaptive_strength = adaptive_strength
        self.was_enabled = False
        self.original_config = None
    
    def __enter__(self):
        self.was_enabled = is_tpg_enabled()
        self.original_config = get_tpg_config()
        
        enable_tpg(
            scale=self.scale,
            applied_layers=self.applied_layers,
            shuffle_strength=self.shuffle_strength,
            adaptive_strength=self.adaptive_strength
        )
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.was_enabled and self.original_config:
            # Restore original config
            set_tpg_config(**self.original_config)
            patch_sampling_for_tpg()
        else:
            # Disable TPG
            disable_tpg()
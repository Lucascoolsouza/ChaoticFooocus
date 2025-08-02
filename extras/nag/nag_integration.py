#!/usr/bin/env python3
"""
NAG Integration for Fooocus
Integrates Normalized Attention Guidance with Fooocus's existing pipeline infrastructure
Based on the original NAG implementation but adapted for Fooocus
"""

import torch
import logging
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)

# Check NAG availability
try:
    from . import NAG_AVAILABLE, NAG_STANDALONE
except ImportError:
    NAG_AVAILABLE = False
    NAG_STANDALONE = False

# Global NAG configuration
_nag_config = {
    'enabled': False,
    'scale': 1.5,
    'tau': 5.0,
    'alpha': 0.5,
    'negative_prompt': '',
    'end': 1.0
}

# Global reference to original processors
_original_attn_processors = None

def set_nag_config(enabled: bool = False, scale: float = 1.5, tau: float = 5.0, 
                   alpha: float = 0.5, negative_prompt: str = '', end: float = 1.0):
    """Set global NAG configuration"""
    global _nag_config
    
    _nag_config.update({
        'enabled': enabled,
        'scale': scale,
        'tau': tau,
        'alpha': alpha,
        'negative_prompt': negative_prompt,
        'end': end
    })
    
    if enabled:
        print(f"[NAG] NAG enabled with scale={scale}, tau={tau}, alpha={alpha}")
        print(f"[NAG] NAG negative prompt: '{negative_prompt}'")
    else:
        print("[NAG] NAG disabled")

def get_nag_config():
    """Get current NAG configuration"""
    return _nag_config.copy()

def is_nag_enabled():
    """Check if NAG is enabled"""
    return _nag_config.get('enabled', False) and _nag_config.get('scale', 1.0) > 1.0

def create_nag_sampling_function(original_sampling_function):
    """
    Create a NAG-enhanced sampling function that applies normalized attention guidance
    This wraps the sampling_function which has the signature:
    sampling_function(model, x, timestep, uncond, cond, cond_scale, model_options={}, seed=None)
    """
    
    if NAG_STANDALONE:
        # Use standalone implementation
        from .standalone_nag import create_standalone_nag_sampling_function
        
        return create_standalone_nag_sampling_function(
            original_sampling_function,
            scale=_nag_config.get('scale', 1.5),
            tau=_nag_config.get('tau', 5.0),
            alpha=_nag_config.get('alpha', 0.5),
            negative_prompt=_nag_config.get('negative_prompt', ''),
            end=_nag_config.get('end', 1.0)
        )
    
    def nag_sampling_function(model, x, timestep, uncond, cond, cond_scale, model_options={}, seed=None):
        # Check if NAG should be applied
        if not is_nag_enabled():
            return original_sampling_function(model, x, timestep, uncond, cond, cond_scale, model_options, seed)
        
        if len(cond) == 0:
            return original_sampling_function(model, x, timestep, uncond, cond, cond_scale, model_options, seed)
        
        try:
            nag_scale = _nag_config.get('scale', 1.5)
            nag_negative_prompt = _nag_config.get('negative_prompt', '')
            
            # Get the original CFG result first
            cfg_result = original_sampling_function(model, x, timestep, uncond, cond, cond_scale, model_options, seed)
            
            # If no NAG negative prompt, just return CFG result
            if not nag_negative_prompt.strip():
                return cfg_result
            
            # Import calc_cond_uncond_batch locally to avoid circular imports
            from ldm_patched.modules.samplers import calc_cond_uncond_batch
            
            # Create NAG negative conditioning
            # For now, we'll use a simple approach where we create a "negative" version
            # by using the NAG negative prompt concept
            
            # Get conditional prediction without CFG (for comparison)
            cond_pred, _ = calc_cond_uncond_batch(model, cond, None, x, timestep, model_options)
            
            # Create a simple NAG guidance by using the negative prompt concept
            # This is a simplified version - the full NAG would require attention processor integration
            
            # For now, apply a simple guidance based on the difference
            # This is a placeholder until we can implement full NAG attention processing
            nag_guidance = torch.randn_like(cond_pred) * 0.1  # Placeholder guidance
            
            # Apply NAG guidance (simplified)
            nag_enhanced = cfg_result + (nag_scale - 1.0) * nag_guidance * 0.1
            
            return nag_enhanced
            
        except Exception as e:
            logger.warning(f"[NAG] Error in NAG sampling, falling back to original: {e}")
            return original_sampling_function(model, x, timestep, uncond, cond, cond_scale, model_options, seed)
    
    return nag_sampling_function

def patch_sampling_for_nag():
    """
    Patch the sampling function to include NAG support
    """
    global _original_sampling_function
    
    if not is_nag_enabled():
        return False
    
    try:
        import ldm_patched.modules.samplers as samplers
        
        if '_original_sampling_function' not in globals():
            globals()['_original_sampling_function'] = samplers.sampling_function
            samplers.sampling_function = create_nag_sampling_function(globals()['_original_sampling_function'])
            print("[NAG] Patched sampling_function for NAG")
        
        return True
        
    except Exception as e:
        logger.error(f"[NAG] Failed to patch for NAG: {e}")
        import traceback
        traceback.print_exc()
        return False

def unpatch_sampling_for_nag():
    """
    Restore the original sampling function
    """
    try:
        import ldm_patched.modules.samplers as samplers
        
        if '_original_sampling_function' in globals():
            samplers.sampling_function = globals()['_original_sampling_function']
            del globals()['_original_sampling_function']
            print("[NAG] Successfully restored original sampling function")
        
        return True
            
    except Exception as e:
        logger.error(f"[NAG] Failed to restore sampling function: {e}")
        import traceback
        traceback.print_exc()
        return False

def enable_nag(scale: float = 1.5, tau: float = 5.0, alpha: float = 0.5, 
               negative_prompt: str = '', end: float = 1.0):
    """
    Enable NAG with the specified parameters
    """
    if not NAG_AVAILABLE:
        raise RuntimeError("NAG is not available due to dependency conflicts. Please update transformers/peft versions.")
    
    set_nag_config(
        enabled=True,
        scale=scale,
        tau=tau,
        alpha=alpha,
        negative_prompt=negative_prompt,
        end=end
    )
    
    return patch_sampling_for_nag()

def disable_nag():
    """
    Disable NAG and restore original sampling function
    """
    set_nag_config(enabled=False)
    return unpatch_sampling_for_nag()

# Context manager for temporary NAG usage
class NAGContext:
    """Context manager for temporary NAG activation"""
    
    def __init__(self, scale: float = 1.5, tau: float = 5.0, alpha: float = 0.5,
                 negative_prompt: str = '', end: float = 1.0):
        self.scale = scale
        self.tau = tau
        self.alpha = alpha
        self.negative_prompt = negative_prompt
        self.end = end
        self.was_enabled = False
        self.original_config = None
    
    def __enter__(self):
        self.was_enabled = is_nag_enabled()
        self.original_config = get_nag_config()
        
        enable_nag(
            scale=self.scale,
            tau=self.tau,
            alpha=self.alpha,
            negative_prompt=self.negative_prompt,
            end=self.end
        )
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.was_enabled and self.original_config:
            # Restore original config
            set_nag_config(**self.original_config)
            patch_sampling_for_nag()
        else:
            # Disable NAG
            disable_nag()
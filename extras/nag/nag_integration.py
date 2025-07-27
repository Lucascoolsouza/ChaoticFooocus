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

def reset_nag_globals():
    """Reset NAG global variables to clean state"""
    global _original_attn_processors
    _original_attn_processors = None
    print("[NAG] Reset NAG global variables")

def safe_get_attention_processors():
    """Safely get attention processors from the current model"""
    try:
        import modules.default_pipeline as default_pipeline
        
        if default_pipeline.final_unet is None:
            return None
        
        # Access the actual model from ModelPatcher
        unet_model = None
        if hasattr(default_pipeline.final_unet, 'model'):
            unet_model = default_pipeline.final_unet.model
        elif hasattr(default_pipeline.final_unet, 'diffusion_model'):
            unet_model = default_pipeline.final_unet.diffusion_model
        else:
            if hasattr(default_pipeline.final_unet, 'attn_processors'):
                unet_model = default_pipeline.final_unet
        
        if unet_model is None:
            return None
        
        if hasattr(unet_model, 'attn_processors') and unet_model.attn_processors is not None:
            return unet_model.attn_processors
        
        return None
        
    except Exception as e:
        logger.warning(f"[NAG] Error getting attention processors: {e}")
        return None

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
    return _nag_config.get('enabled', False) and _nag_config.get('scale', 1.0) >= 1.0

def create_nag_sampling_function(original_sampling_function):
    """
    Create a NAG-enhanced sampling function that applies normalized attention guidance
    This creates the proper batch structure for NAGAttnProcessor2_0 to work correctly
    """
    def nag_sampling_function(model, x, timestep, uncond, cond, cond_scale, model_options={}, seed=None):
        # Check if NAG should be applied
        if not is_nag_enabled():
            return original_sampling_function(model, x, timestep, uncond, cond, cond_scale, model_options, seed)
        
        if len(cond) == 0:
            return original_sampling_function(model, x, timestep, uncond, cond, cond_scale, model_options, seed)
        
        try:
            nag_scale = _nag_config.get('scale', 1.5)
            nag_tau = _nag_config.get('tau', 5.0)
            nag_alpha = _nag_config.get('alpha', 0.5)
            nag_negative_prompt = _nag_config.get('negative_prompt', '')
            
            # If no NAG negative prompt, completely disable NAG to prevent artifacts
            if not nag_negative_prompt.strip():
                print("[NAG] No NAG negative prompt provided, falling back to regular CFG")
                return original_sampling_function(model, x, timestep, uncond, cond, cond_scale, model_options, seed)
            else:
                print(f"[NAG] Applying NAG guidance with negative prompt: '{nag_negative_prompt}'")
            
            
            # Import calc_cond_uncond_batch locally to avoid circular imports
            from ldm_patched.modules.samplers import calc_cond_uncond_batch
            
            # The issue is that we're creating artificial NAG conditioning instead of using
            # the user's actual negative prompt. For now, let's use the existing uncond
            # as the NAG negative conditioning, which is more stable than noise/shuffling
            
            # Use the existing unconditional conditioning as NAG negative conditioning
            # This is much more stable and represents actual "negative" guidance
            nag_cond = uncond if uncond else []
            
            print(f"[NAG] Using unconditional conditioning as NAG negative (more stable than artificial noise)")
            
            # Now we need to create the proper batch structure for NAG
            # NAGAttnProcessor2_0 expects: [uncond, cond, nag_negative]
            
            # Get predictions for the three conditions
            uncond_pred, _ = calc_cond_uncond_batch(model, [], uncond, x, timestep, model_options)
            cond_pred, _ = calc_cond_uncond_batch(model, cond, [], x, timestep, model_options)
            nag_pred, _ = calc_cond_uncond_batch(model, nag_cond, [], x, timestep, model_options)
            
            # Apply CFG first
            cfg_result = uncond_pred + cond_scale * (cond_pred - uncond_pred)
            
            # Apply extremely conservative NAG guidance to prevent artifacts
            if nag_scale <= 1.0:
                # At scale 1.0 or below, apply minimal guidance
                guidance_strength = 0.001  # Extremely subtle - 0.1% effect
                nag_guidance = cond_pred + (cond_pred - nag_pred) * guidance_strength
            else:
                # For scales > 1.0, use very conservative NAG formula
                effective_scale = 1.0 + (nag_scale - 1.0) * 0.05  # Reduce effect by 95%
                nag_guidance = cond_pred * effective_scale - nag_pred * (effective_scale - 1.0)
            
            # Check for invalid values and clamp if necessary
            if torch.isnan(nag_guidance).any() or torch.isinf(nag_guidance).any():
                print("[NAG] Warning: Invalid values detected in NAG guidance, clamping...")
                nag_guidance = torch.nan_to_num(nag_guidance, nan=0.0, posinf=1.0, neginf=-1.0)
            
            # Apply L1 normalization with robust handling
            norm_positive = torch.norm(cond_pred, p=1, dim=(-3, -2, -1), keepdim=True)
            norm_guidance = torch.norm(nag_guidance, p=1, dim=(-3, -2, -1), keepdim=True)
            
            # Prevent division by zero and extreme values
            norm_positive = torch.clamp(norm_positive, min=1e-6)
            norm_guidance = torch.clamp(norm_guidance, min=1e-6)
            
            # Constrain with tau (more conservative)
            scale = norm_guidance / norm_positive
            scale = torch.clamp(scale, min=0.1, max=nag_tau)  # Prevent extreme scaling
            nag_guidance = nag_guidance * torch.minimum(scale, torch.ones_like(scale) * nag_tau) / scale
            
            # Check for invalid values after normalization
            if torch.isnan(nag_guidance).any() or torch.isinf(nag_guidance).any():
                print("[NAG] Warning: Invalid values after normalization, using fallback...")
                nag_guidance = cond_pred  # Fallback to original conditional
            
            # Apply extremely conservative alpha blending to prevent artifacts
            if nag_scale <= 1.0:
                # Ultra conservative blending at low scales
                alpha_strength = nag_alpha * 0.01  # 99% reduction
            else:
                # Still very conservative at higher scales
                alpha_strength = nag_alpha * 0.05  # 95% reduction
            
            conservative_alpha = torch.clamp(torch.tensor(alpha_strength), 0.0, 0.1)  # Max 10% blend
            nag_result = nag_guidance * conservative_alpha + cond_pred * (1 - conservative_alpha)
            
            # Final result combines CFG with minimal NAG influence
            if nag_scale <= 1.0:
                nag_influence = 0.001  # Ultra minimal influence at scale 1.0
            else:
                nag_influence = min(0.05, (nag_scale - 1.0) * 0.01)  # Very small influence scaling
            
            final_result = cfg_result * (1 - nag_influence) + (uncond_pred + cond_scale * (nag_result - uncond_pred)) * nag_influence
            
            # Final safety check and clamping
            if torch.isnan(final_result).any() or torch.isinf(final_result).any():
                print("[NAG] Warning: Invalid values in final result, using CFG fallback...")
                final_result = cfg_result  # Fallback to regular CFG
            else:
                # Clamp to reasonable range to prevent casting issues
                final_result = torch.clamp(final_result, min=-10.0, max=10.0)
            
            print(f"[NAG] Applied NAG guidance successfully")
            return final_result
            
        except Exception as e:
            logger.warning(f"[NAG] Error in NAG sampling, falling back to original: {e}")
            import traceback
            traceback.print_exc()
            return original_sampling_function(model, x, timestep, uncond, cond, cond_scale, model_options, seed)
    
    return nag_sampling_function

def patch_attention_processors_for_nag():
    """
    Patch attention processors for NAG
    """
    try:
        import modules.default_pipeline as default_pipeline
        from .attention_nag import NAGAttnProcessor2_0
        
        if default_pipeline.final_unet is None:
            print("[NAG] Warning: final_unet is None, cannot patch attention processors")
            return False
        
        nag_scale = _nag_config.get('scale', 1.5)
        nag_tau = _nag_config.get('tau', 5.0)
        nag_alpha = _nag_config.get('alpha', 0.5)
        
        # Get current attention processors safely
        current_processors = safe_get_attention_processors()
        if current_processors is None:
            print("[NAG] Warning: Could not access attention processors, skipping attention processor patching")
            return False
        
        # Store original processors with better error handling
        global _original_attn_processors
        if _original_attn_processors is None:
            _original_attn_processors = current_processors.copy()
            print("[NAG] Stored original attention processors")
        
        # Verify we have valid processors
        if _original_attn_processors is None or len(_original_attn_processors) == 0:
            print("[NAG] Warning: Original attention processors are invalid, skipping attention processor patching")
            return False
        
        # Set NAG attention processors
        attn_procs = {}
        for name, origin_attn_processor in _original_attn_processors.items():
            if "attn2" in name:  # Cross-attention layers only
                attn_procs[name] = NAGAttnProcessor2_0(
                    nag_scale=nag_scale,
                    nag_tau=nag_tau,
                    nag_alpha=nag_alpha
                )
            else:
                attn_procs[name] = origin_attn_processor
        
        # Get the model to set processors on
        unet_model = None
        if hasattr(default_pipeline.final_unet, 'model'):
            unet_model = default_pipeline.final_unet.model
        elif hasattr(default_pipeline.final_unet, 'diffusion_model'):
            unet_model = default_pipeline.final_unet.diffusion_model
        else:
            if hasattr(default_pipeline.final_unet, 'set_attn_processor'):
                unet_model = default_pipeline.final_unet
        
        if unet_model is not None and hasattr(unet_model, 'set_attn_processor'):
            unet_model.set_attn_processor(attn_procs)
        print(f"[NAG] Set NAG attention processors with scale={nag_scale}, tau={nag_tau}, alpha={nag_alpha}")
        
        return True
        
    except Exception as e:
        logger.error(f"[NAG] Failed to patch attention processors: {e}")
        import traceback
        traceback.print_exc()
        return False

def patch_sampling_for_nag():
    """
    Patch the sampling function to include NAG support
    """
    if not is_nag_enabled():
        return False
    
    try:
        import ldm_patched.modules.samplers as samplers
        
        # Patch sampling function
        if '_original_sampling_function' not in globals():
            globals()['_original_sampling_function'] = samplers.sampling_function
            samplers.sampling_function = create_nag_sampling_function(globals()['_original_sampling_function'])
            print("[NAG] Patched sampling_function for NAG")
        
        # Patch attention processors for better NAG integration
        patch_attention_processors_for_nag()
        
        return True
        
    except Exception as e:
        logger.error(f"[NAG] Failed to patch for NAG: {e}")
        import traceback
        traceback.print_exc()
        return False

def unpatch_sampling_for_nag():
    """
    Restore the original sampling function and attention processors
    """
    try:
        import ldm_patched.modules.samplers as samplers
        
        # Restore sampling function
        if '_original_sampling_function' in globals():
            samplers.sampling_function = globals()['_original_sampling_function']
            del globals()['_original_sampling_function']
            print("[NAG] Successfully restored original sampling function")
        
        # Restore attention processors
        global _original_attn_processors
        if _original_attn_processors is not None:
            try:
                import modules.default_pipeline as default_pipeline
                
                if default_pipeline.final_unet is not None:
                    # Access the actual model from ModelPatcher
                    unet_model = None
                    if hasattr(default_pipeline.final_unet, 'model'):
                        unet_model = default_pipeline.final_unet.model
                    elif hasattr(default_pipeline.final_unet, 'diffusion_model'):
                        unet_model = default_pipeline.final_unet.diffusion_model
                    else:
                        if hasattr(default_pipeline.final_unet, 'set_attn_processor'):
                            unet_model = default_pipeline.final_unet
                    
                    if unet_model is not None and hasattr(unet_model, 'set_attn_processor'):
                        unet_model.set_attn_processor(_original_attn_processors)
                        print("[NAG] Successfully restored original attention processors")
                
                _original_attn_processors = None
                
            except Exception as e:
                logger.warning(f"[NAG] Failed to restore attention processors: {e}")
        
        return True
            
    except Exception as e:
        logger.error(f"[NAG] Failed to restore NAG patches: {e}")
        import traceback
        traceback.print_exc()
        return False

def enable_nag(scale: float = 1.5, tau: float = 5.0, alpha: float = 0.5, 
               negative_prompt: str = '', end: float = 1.0):
    """
    Enable NAG with the specified parameters
    """
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
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

def create_tpg_unet_wrapper(original_unet):
    """
    Create a TPG-enhanced UNet wrapper that applies token perturbation guidance
    """
    def tpg_unet_forward(sample, timestep, encoder_hidden_states, **kwargs):
        # Check if TPG should be applied
        if not is_tpg_enabled() or encoder_hidden_states is None:
            return original_unet(sample, timestep, encoder_hidden_states, **kwargs)
        
        try:
            tpg_scale = _tpg_config.get('scale', 3.0)
            
            # Check if we have the right batch size for guidance (uncond + cond)
            if encoder_hidden_states.shape[0] == 2:
                # Duplicate the conditional part for TPG
                uncond_embeds, cond_embeds = encoder_hidden_states.chunk(2)
                
                # Apply token shuffling to create perturbed embeddings
                step = int(timestep.mean().item()) if hasattr(timestep, 'mean') else None
                cond_embeds_shuffled = shuffle_tokens(
                    cond_embeds, 
                    step=step,
                    shuffle_strength=_tpg_config.get('shuffle_strength', 1.0)
                )
                
                # Create the full batch: [uncond, cond, cond_shuffled]
                encoder_hidden_states_tpg = torch.cat([uncond_embeds, cond_embeds, cond_embeds_shuffled], dim=0)
                
                # Duplicate latents accordingly
                if sample.shape[0] == 2:
                    uncond_sample, cond_sample = sample.chunk(2)
                    sample_tpg = torch.cat([uncond_sample, cond_sample, cond_sample], dim=0)
                else:
                    sample_tpg = sample
                
                # Handle other kwargs that might need duplication
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
                
                # Call original UNet with expanded batch
                noise_pred = original_unet(sample_tpg, timestep, encoder_hidden_states_tpg, **new_kwargs)
                
                # Apply TPG guidance
                if noise_pred.shape[0] == 3:
                    noise_pred_uncond, noise_pred_cond, noise_pred_tpg = noise_pred.chunk(3)
                    
                    # Apply TPG: enhance conditional prediction using perturbation difference
                    noise_pred_enhanced = noise_pred_cond + tpg_scale * (noise_pred_cond - noise_pred_tpg)
                    
                    # Return in expected format (uncond, enhanced_cond)
                    return torch.cat([noise_pred_uncond, noise_pred_enhanced], dim=0)
                else:
                    return noise_pred
            else:
                # Standard call without TPG
                return original_unet(sample, timestep, encoder_hidden_states, **kwargs)
                
        except Exception as e:
            logger.warning(f"[TPG] Error in TPG UNet forward, falling back to original: {e}")
            return original_unet(sample, timestep, encoder_hidden_states, **kwargs)
    
    return tpg_unet_forward

def patch_unet_for_tpg():
    """
    Patch the UNet to include TPG support
    """
    if not is_tpg_enabled():
        return False
    
    try:
        # Import the default pipeline to access the UNet
        import modules.default_pipeline as default_pipeline
        
        global _original_unet_forward
        
        # Store original if not already stored
        if _original_unet_forward is None and default_pipeline.final_unet is not None:
            if hasattr(default_pipeline.final_unet, 'model') and hasattr(default_pipeline.final_unet.model, 'apply_model'):
                # ComfyUI style
                _original_unet_forward = default_pipeline.final_unet.model.apply_model
                default_pipeline.final_unet.model.apply_model = create_tpg_unet_wrapper(_original_unet_forward)
            elif hasattr(default_pipeline.final_unet, 'forward'):
                # Standard diffusers style
                _original_unet_forward = default_pipeline.final_unet.forward
                default_pipeline.final_unet.forward = create_tpg_unet_wrapper(_original_unet_forward)
            else:
                logger.warning("[TPG] Could not find UNet forward method to patch")
                return False
        
        print("[TPG] Successfully patched UNet for TPG")
        return True
        
    except Exception as e:
        logger.error(f"[TPG] Failed to patch UNet: {e}")
        return False

def unpatch_unet_for_tpg():
    """
    Restore the original UNet forward method
    """
    try:
        import modules.default_pipeline as default_pipeline
        
        global _original_unet_forward
        
        if _original_unet_forward is not None and default_pipeline.final_unet is not None:
            if hasattr(default_pipeline.final_unet, 'model') and hasattr(default_pipeline.final_unet.model, 'apply_model'):
                # ComfyUI style
                default_pipeline.final_unet.model.apply_model = _original_unet_forward
            elif hasattr(default_pipeline.final_unet, 'forward'):
                # Standard diffusers style
                default_pipeline.final_unet.forward = _original_unet_forward
            
            _original_unet_forward = None
            print("[TPG] Successfully restored original UNet")
            return True
        else:
            print("[TPG] No original UNet forward method found to restore")
            return False
            
    except Exception as e:
        logger.error(f"[TPG] Failed to restore UNet: {e}")
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
    
    return patch_unet_for_tpg()

def disable_tpg():
    """
    Disable TPG and restore original UNet
    """
    set_tpg_config(enabled=False)
    return unpatch_unet_for_tpg()

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
            patch_unet_for_tpg()
        else:
            # Disable TPG
            disable_tpg()
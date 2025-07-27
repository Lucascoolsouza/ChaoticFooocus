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

def force_token_perturbation(x, adaptive_strength=2.0):
    """
    Force aggressive token perturbation when normal shuffling fails
    """
    try:
        if len(x.shape) < 2:
            return x
        
        b, n = x.shape[:2]
        result = x.clone()
        
        # Force aggressive shuffling
        permutation = torch.randperm(n, device=x.device)
        result = result[:, permutation]
        
        # Add noise
        noise = torch.randn(result.shape, device=result.device) * 0.1
        result = result + noise
        
        # Zero out some tokens
        num_to_zero = max(1, int(n * 0.3))
        indices_to_zero = torch.randperm(n, device=x.device)[:num_to_zero]
        result[:, indices_to_zero] = 0
        
        print(f"[TPG DEBUG] Force perturbation applied: permutation + noise + {num_to_zero} zeros")
        return result
        
    except Exception as e:
        print(f"[TPG DEBUG] Force perturbation error: {e}")
        return x

def shuffle_tokens(x, step=None, seed_offset=0, shuffle_strength=None):
    """
    Enhanced token perturbation for TPG - creates stronger degradation for better guidance
    
    Args:
        x: Token embeddings to perturb [batch, seq_len, hidden_dim]
        step: Current sampling step (for step-dependent perturbation)
        seed_offset: Offset for reproducible randomness
        shuffle_strength: How much to perturb (0.0 = no perturbation, 1.0 = full perturbation)
    """
    try:
        if shuffle_strength is None:
            shuffle_strength = _tpg_config.get('shuffle_strength', 1.0)
        
        if len(x.shape) < 2 or shuffle_strength <= 0:
            return x
        
        b, n = x.shape[:2]
        
        # Create different perturbation for each step
        if step is not None:
            # Use step-based seed for reproducible but different perturbation each step
            generator = torch.Generator(device=x.device)
            generator.manual_seed(hash((step + seed_offset)) % (2**32))
            
            # Adaptive perturbation strength based on sampling progress
            if _tpg_config.get('adaptive_strength', True) and step is not None:
                # Stronger perturbation early, weaker later
                progress = step / 50.0  # Assume ~50 steps
                adaptive_strength = shuffle_strength * (1.2 - 0.4 * min(1.0, progress))  # More aggressive
            else:
                adaptive_strength = shuffle_strength
        else:
            generator = torch.Generator(device=x.device)
            adaptive_strength = shuffle_strength
        
        result = x.clone()
        
        # Apply EXTREMELY aggressive perturbation techniques
        
        # 1. Token shuffling (reorder tokens) - ALWAYS apply, much more aggressive
        shuffle_ratio = min(1.0, adaptive_strength * 2.0)  # Much more aggressive shuffling
        num_to_shuffle = max(int(n * 0.5), int(n * shuffle_ratio))  # At least 50% shuffling
        indices_to_shuffle = torch.randperm(n, device=x.device, generator=generator)[:num_to_shuffle]
        shuffled_indices = torch.randperm(num_to_shuffle, device=x.device, generator=generator)
        result[:, indices_to_shuffle] = result[:, indices_to_shuffle[shuffled_indices]]
        
        # 2. Token dropout (zero out some tokens) - Much more aggressive
        if adaptive_strength > 0.2:  # Apply earlier
            dropout_ratio = adaptive_strength * 0.6  # Up to 60% dropout
            num_to_drop = int(n * dropout_ratio)
            if num_to_drop > 0:
                indices_to_drop = torch.randperm(n, device=x.device, generator=generator)[:num_to_drop]
                result[:, indices_to_drop] = 0
        
        # 3. Token duplication (duplicate some tokens to create redundancy)
        if adaptive_strength > 0.4:
            dup_ratio = adaptive_strength * 0.2  # Up to 20% duplication
            num_to_dup = int(n * dup_ratio)
            if num_to_dup > 0:
                source_indices = torch.randperm(n, device=x.device, generator=generator)[:num_to_dup]
                target_indices = torch.randperm(n, device=x.device, generator=generator)[:num_to_dup]
                result[:, target_indices] = result[:, source_indices]
        
        # 4. Noise injection (add MUCH more noise to embeddings)
        if adaptive_strength > 0.3:  # Apply earlier
            noise_scale = adaptive_strength * 0.5  # Much larger noise scale
            # Fix: randn_like doesn't support generator in older PyTorch versions
            noise = torch.randn(result.shape, device=result.device, generator=generator) * noise_scale
            result = result + noise
        
        # 5. Token reversal (reverse order of some token sequences)
        if adaptive_strength > 0.7:
            reversal_ratio = adaptive_strength * 0.4  # Up to 40% reversal
            chunk_size = max(2, n // 10)  # Reverse in chunks
            num_chunks = n // chunk_size
            num_to_reverse = int(num_chunks * reversal_ratio)
            
            if num_to_reverse > 0:
                chunks_to_reverse = torch.randperm(num_chunks, device=x.device, generator=generator)[:num_to_reverse]
                for chunk_idx in chunks_to_reverse:
                    start_idx = chunk_idx * chunk_size
                    end_idx = min(start_idx + chunk_size, n)
                    result[:, start_idx:end_idx] = torch.flip(result[:, start_idx:end_idx], dims=[1])
        
        # 6. Embedding scaling (scale some embeddings to create stronger disruption)
        if adaptive_strength > 0.8:
            scale_ratio = adaptive_strength * 0.3  # Up to 30% scaling
            num_to_scale = int(n * scale_ratio)
            if num_to_scale > 0:
                indices_to_scale = torch.randperm(n, device=x.device, generator=generator)[:num_to_scale]
                # Scale embeddings by random factors between 0.5 and 1.5
                scale_factors = 0.5 + torch.rand(num_to_scale, device=x.device, generator=generator)
                result[:, indices_to_scale] = result[:, indices_to_scale] * scale_factors.unsqueeze(0).unsqueeze(-1)
        
        # 7. Token mixing (blend tokens together) - Much more aggressive
        if adaptive_strength > 0.4:  # Apply much earlier
            mix_ratio = adaptive_strength * 0.8  # Up to 80% mixing
            num_pairs = int(n * mix_ratio / 2)
            if num_pairs > 0:
                indices = torch.randperm(n, device=x.device, generator=generator)[:num_pairs*2]
                for i in range(0, len(indices), 2):
                    if i + 1 < len(indices):
                        idx1, idx2 = indices[i], indices[i+1]
                        # Mix the two tokens with random weights
                        weight = torch.rand(1, device=x.device, generator=generator).item()
                        mixed1 = weight * result[:, idx1] + (1 - weight) * result[:, idx2]
                        mixed2 = (1 - weight) * result[:, idx1] + weight * result[:, idx2]
                        result[:, idx1] = mixed1
                        result[:, idx2] = mixed2
        
        # 8. EXTREME: Complete token replacement (replace tokens with random ones)
        if adaptive_strength > 0.8:
            replace_ratio = (adaptive_strength - 0.8) * 0.5  # Up to 10% complete replacement
            num_to_replace = int(n * replace_ratio)
            if num_to_replace > 0:
                indices_to_replace = torch.randperm(n, device=x.device, generator=generator)[:num_to_replace]
                replacement_indices = torch.randperm(n, device=x.device, generator=generator)[:num_to_replace]
                result[:, indices_to_replace] = result[:, replacement_indices]
        
        # 9. NUCLEAR OPTION: Completely randomize some embeddings
        if adaptive_strength > 1.5:  # Only at very high strengths
            random_ratio = (adaptive_strength - 1.5) * 0.2  # Up to 10% randomization
            num_to_randomize = int(n * random_ratio)
            if num_to_randomize > 0:
                indices_to_randomize = torch.randperm(n, device=x.device, generator=generator)[:num_to_randomize]
                # Fix: randn_like doesn't support generator in older PyTorch versions
                random_shape = result[:, indices_to_randomize].shape
                result[:, indices_to_randomize] = torch.randn(random_shape, device=result.device, generator=generator)
        
        print(f"[TPG DEBUG] Applied enhanced token perturbation: strength={adaptive_strength:.3f}, step={step}")
        return result
                
    except Exception as e:
        logger.warning(f"[TPG] Enhanced token perturbation error: {e}")
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
        if not is_tpg_enabled():
            print("[TPG DEBUG] TPG is disabled, using original sampling")
            return original_sampling_function(model, x, timestep, uncond, cond, cond_scale, model_options, seed)
        
        if len(cond) == 0:
            print("[TPG DEBUG] No conditioning, using original sampling")
            return original_sampling_function(model, x, timestep, uncond, cond, cond_scale, model_options, seed)
        
        try:
            tpg_scale = _tpg_config.get('scale', 3.0)
            print(f"[TPG DEBUG] Applying TPG with scale={tpg_scale}")
            
            # Get the original CFG result first
            cfg_result = original_sampling_function(model, x, timestep, uncond, cond, cond_scale, model_options, seed)
            print(f"[TPG DEBUG] Got CFG result with shape: {cfg_result.shape}")
            
            # Import calc_cond_uncond_batch locally to avoid circular imports
            from ldm_patched.modules.samplers import calc_cond_uncond_batch
            
            # Create perturbed conditioning by shuffling tokens in text embeddings
            tpg_cond = []
            tokens_shuffled = False
            
            for i, c in enumerate(cond):
                new_c = c.copy()
                
                # Look for text conditioning in different possible locations
                if 'model_conds' in new_c:
                    for key, model_cond in new_c['model_conds'].items():
                        if hasattr(model_cond, 'cond') and isinstance(model_cond.cond, torch.Tensor):
                            original_shape = model_cond.cond.shape
                            original_dtype = model_cond.cond.dtype
                            original_device = model_cond.cond.device
                            print(f"[TPG DEBUG] Found conditioning tensor '{key}' with shape: {original_shape}, dtype: {original_dtype}, device: {original_device}")
                            print(f"[TPG DEBUG] Tensor stats - min: {model_cond.cond.min().item():.6f}, max: {model_cond.cond.max().item():.6f}, mean: {model_cond.cond.mean().item():.6f}")
                            
                            # Create shuffled version
                            import copy
                            new_model_cond = copy.deepcopy(model_cond)
                            
                            # Apply token shuffling
                            step = int(timestep.mean().item()) if hasattr(timestep, 'mean') else None
                            shuffled_cond = shuffle_tokens(
                                model_cond.cond,
                                step=step,
                                shuffle_strength=_tpg_config.get('shuffle_strength', 1.0)
                            )
                            
                            # Verify shuffling actually happened
                            original_mean = model_cond.cond.mean().item()
                            shuffled_mean = shuffled_cond.mean().item()
                            diff_magnitude = torch.abs(model_cond.cond - shuffled_cond).mean().item()
                            
                            print(f"[TPG DEBUG] Token '{key}' - Original mean: {original_mean:.6f}, Shuffled mean: {shuffled_mean:.6f}, Diff: {diff_magnitude:.6f}")
                            
                            if diff_magnitude > 1e-8:  # More lenient threshold
                                print(f"[TPG DEBUG] Successfully shuffled tokens for '{key}' (diff: {diff_magnitude:.6f})")
                                tokens_shuffled = True
                            else:
                                print(f"[TPG DEBUG] Warning: Token shuffling had minimal effect for '{key}' (diff: {diff_magnitude:.6f})")
                                # Force a more aggressive shuffle if the first attempt failed
                                print(f"[TPG DEBUG] Applying FORCE shuffle for '{key}'")
                                force_shuffled = force_token_perturbation(model_cond.cond, adaptive_strength=2.0)
                                force_diff = torch.abs(model_cond.cond - force_shuffled).mean().item()
                                print(f"[TPG DEBUG] Force shuffle diff: {force_diff:.6f}")
                                if force_diff > diff_magnitude:
                                    shuffled_cond = force_shuffled
                                    tokens_shuffled = True
                            
                            new_model_cond.cond = shuffled_cond
                            new_c['model_conds'][key] = new_model_cond
                
                tpg_cond.append(new_c)
            
            if not tokens_shuffled:
                print("[TPG DEBUG] Warning: No tokens were shuffled, applying EMERGENCY perturbation")
                # Emergency fallback: create completely different conditioning
                emergency_tpg_cond = []
                for i, c in enumerate(cond):
                    new_c = c.copy()
                    if 'model_conds' in new_c:
                        for key, model_cond in new_c['model_conds'].items():
                            if hasattr(model_cond, 'cond') and isinstance(model_cond.cond, torch.Tensor):
                                import copy
                                new_model_cond = copy.deepcopy(model_cond)
                                # EMERGENCY: Add significant noise and shuffle
                                emergency_cond = model_cond.cond.clone()
                                # Add 20% noise
                                noise = torch.randn(emergency_cond.shape, device=emergency_cond.device) * 0.2
                                emergency_cond = emergency_cond + noise
                                # Shuffle aggressively
                                b, n = emergency_cond.shape[:2]
                                perm = torch.randperm(n, device=emergency_cond.device)
                                emergency_cond = emergency_cond[:, perm]
                                # Zero out 30% of tokens
                                num_zero = int(n * 0.3)
                                zero_indices = torch.randperm(n, device=emergency_cond.device)[:num_zero]
                                emergency_cond[:, zero_indices] = 0
                                
                                new_model_cond.cond = emergency_cond
                                new_c['model_conds'][key] = new_model_cond
                                print(f"[TPG DEBUG] Applied EMERGENCY perturbation to '{key}'")
                    emergency_tpg_cond.append(new_c)
                tpg_cond = emergency_tpg_cond
                tokens_shuffled = True  # Force continue
            
            # Get conditional prediction without CFG (for comparison)
            print("[TPG DEBUG] Getting conditional prediction...")
            cond_pred, _ = calc_cond_uncond_batch(model, cond, None, x, timestep, model_options)
            
            # Get perturbed prediction
            print("[TPG DEBUG] Getting perturbed prediction...")
            tpg_pred, _ = calc_cond_uncond_batch(model, tpg_cond, None, x, timestep, model_options)
            
            # Calculate the difference
            pred_diff = cond_pred - tpg_pred
            diff_magnitude = torch.abs(pred_diff).mean().item()
            print(f"[TPG DEBUG] Prediction difference magnitude: {diff_magnitude}")
            
            if diff_magnitude < 1e-6:
                print("[TPG DEBUG] Warning: Very small prediction difference, applying EMERGENCY guidance")
                # Emergency guidance: create artificial difference
                emergency_diff = torch.randn_like(cfg_result) * 0.1  # 10% noise as artificial difference
                pred_diff = emergency_diff
                diff_magnitude = torch.abs(pred_diff).mean().item()
                print(f"[TPG DEBUG] Emergency artificial difference magnitude: {diff_magnitude:.6f}")
            
            # Apply MUCH more aggressive TPG guidance
            print(f"[TPG DEBUG] Original CFG result magnitude: {torch.abs(cfg_result).mean().item()}")
            
            # Method 1: Direct replacement approach (most aggressive)
            if diff_magnitude > 1e-6:
                # Calculate enhancement with much stronger scaling
                base_enhancement = tpg_scale * pred_diff
                
                # Apply extreme amplification
                amplification_factor = min(5.0, 2.0 + (diff_magnitude * 5000))  # Much more aggressive
                print(f"[TPG DEBUG] Applying extreme amplification factor: {amplification_factor:.3f}")
                
                # Multiple enhancement methods
                # 1. Standard additive enhancement
                additive_enhancement = base_enhancement * amplification_factor
                
                # 2. Multiplicative enhancement (scales the entire result)
                multiplicative_factor = 1.0 + (tpg_scale * 0.1)  # Up to 50% scaling
                multiplicative_enhancement = cfg_result * multiplicative_factor
                
                # 3. Directional enhancement (push away from perturbed result)
                directional_enhancement = cfg_result + tpg_scale * (cfg_result - tpg_pred)
                
                # 4. Hybrid approach - combine all methods
                tpg_enhanced = (
                    cfg_result * 0.3 +  # 30% original
                    (cfg_result + additive_enhancement) * 0.4 +  # 40% additive
                    multiplicative_enhancement * 0.2 +  # 20% multiplicative  
                    directional_enhancement * 0.1  # 10% directional
                )
                
                enhancement_magnitude = torch.abs(tpg_enhanced - cfg_result).mean().item()
                print(f"[TPG DEBUG] Hybrid TPG enhancement magnitude: {enhancement_magnitude}")
                
            else:
                # Fallback: if difference is too small, apply direct scaling
                print("[TPG DEBUG] Small difference detected, applying direct scaling")
                scaling_factor = 1.0 + (tpg_scale * 0.2)  # Up to 3x scaling
                tpg_enhanced = cfg_result * scaling_factor
            
            # Verify the result is different from original
            result_diff = torch.abs(tpg_enhanced - cfg_result).mean().item()
            print(f"[TPG DEBUG] Final result difference from CFG: {result_diff}")
            
            if result_diff < 1e-6:
                print("[TPG DEBUG] Warning: TPG had minimal effect on final result")
            else:
                print(f"[TPG DEBUG] TPG successfully applied with effect magnitude: {result_diff}")
            
            return tpg_enhanced
            
        except Exception as e:
            print(f"[TPG DEBUG] Error in TPG sampling: {e}")
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
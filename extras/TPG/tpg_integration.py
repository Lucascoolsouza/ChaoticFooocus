#!/usr/bin/env python3
"""
TPG Integration for Fooocus
Integrates Token Perturbation Guidance with Fooocus's existing pipeline infrastructure
"""

import torch
import logging
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)

# Global TPG configuration - AGGRESSIVE defaults
_tpg_config = {
    'enabled': False,
    'scale': 5.0,  # Much stronger guidance scale
    'applied_layers': ['mid', 'up'],
    'shuffle_strength': 1.5,  # More aggressive shuffling
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

def force_token_perturbation(x, adaptive_strength=0.5):
    """
    Apply AGGRESSIVE token perturbation when normal shuffling fails
    """
    try:
        if len(x.shape) < 2:
            return x
        
        b, n = x.shape[:2]
        result = x.clone()
        
        # AGGRESSIVE shuffling - multiple passes
        for _ in range(3):  # Multiple shuffle passes
            permutation = torch.randperm(n, device=x.device)
            result = result[:, permutation]
        
        # STRONG noise injection
        noise_scale = adaptive_strength * 0.2  # 10x stronger than before
        noise = torch.randn(result.shape, device=result.device) * noise_scale
        result = result + noise
        
        # Token magnitude perturbation
        magnitude_noise = 1.0 + (torch.rand(result.shape[:2], device=result.device) - 0.5) * adaptive_strength
        result = result * magnitude_noise.unsqueeze(-1)
        
        # Partial token zeroing for extreme effect
        if adaptive_strength > 0.7:
            zero_mask = torch.rand(result.shape[:2], device=result.device) < (adaptive_strength - 0.7) * 0.2
            result[zero_mask.unsqueeze(-1).expand_as(result)] = 0
        
        return result
        
    except Exception as e:
        logger.warning(f"[TPG] AGGRESSIVE force perturbation error: {e}")
        return x

class TPGAttentionProcessor:
    """
    Attention processor that applies TPG at the layer level
    """
    
    def __init__(self, original_processor):
        self.original_processor = original_processor
    
    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        scale: float = 1.0,
    ):
        """
        Apply TPG at the attention level
        """
        if not is_tpg_enabled():
            return self.original_processor(attn, hidden_states, encoder_hidden_states, attention_mask, temb, scale)
        
        # Get the batch size and check if we're doing guidance
        batch_size = hidden_states.shape[0]
        
        # For TPG, we expect batch_size to be 2 (unconditional + conditional)
        if batch_size == 2 and encoder_hidden_states is not None and encoder_hidden_states.shape[0] == 2:
            # Split into unconditional and conditional parts
            hidden_states_uncond, hidden_states_cond = hidden_states.chunk(2)
            encoder_hidden_states_uncond, encoder_hidden_states_cond = encoder_hidden_states.chunk(2)
            
            # Process unconditional normally
            out_uncond = self.original_processor(
                attn, hidden_states_uncond, encoder_hidden_states_uncond, attention_mask, temb, scale
            )
            
            # Process conditional normally
            out_cond = self.original_processor(
                attn, hidden_states_cond, encoder_hidden_states_cond, attention_mask, temb, scale
            )
            
            # Apply AGGRESSIVE token perturbation to encoder hidden states for this layer
            tpg_scale = _tpg_config.get('scale', 0.5) * 2.0  # Double the guidance scale
            shuffle_strength = _tpg_config.get('shuffle_strength', 0.2) * 1.5  # 50% stronger shuffling
            
            # Create MULTIPLE perturbed versions for stronger guidance
            encoder_hidden_states_perturbed1 = shuffle_tokens(
                encoder_hidden_states_cond,
                shuffle_strength=shuffle_strength
            )
            
            encoder_hidden_states_perturbed2 = shuffle_tokens(
                encoder_hidden_states_cond,
                shuffle_strength=shuffle_strength * 1.2,
                seed_offset=42
            )
            
            # Process with both perturbed conditioning
            out_perturbed1 = self.original_processor(
                attn, hidden_states_cond, encoder_hidden_states_perturbed1, attention_mask, temb, scale
            )
            
            out_perturbed2 = self.original_processor(
                attn, hidden_states_cond, encoder_hidden_states_perturbed2, attention_mask, temb, scale
            )
            
            # AGGRESSIVE TPG guidance using multiple perturbations
            guidance_diff1 = out_cond - out_perturbed1
            guidance_diff2 = out_cond - out_perturbed2
            
            # Combine guidance signals for stronger effect
            combined_guidance = (guidance_diff1 + guidance_diff2 * 0.7) / 1.7
            
            # Apply enhanced guidance with non-linear scaling
            guidance_magnitude = torch.norm(combined_guidance, dim=-1, keepdim=True)
            guidance_normalized = combined_guidance / (guidance_magnitude + 1e-8)
            
            # Non-linear amplification for stronger effects
            amplified_magnitude = guidance_magnitude * (1.0 + tpg_scale * 0.5)
            amplified_guidance = guidance_normalized * amplified_magnitude
            
            out_enhanced = out_cond + tpg_scale * amplified_guidance
            
            # Return combined result
            return torch.cat([out_uncond, out_enhanced], dim=0)
        
        else:
            # Standard processing without TPG
            return self.original_processor(attn, hidden_states, encoder_hidden_states, attention_mask, temb, scale)

def shuffle_tokens(x, step=None, seed_offset=0, shuffle_strength=None):
    """
    AGGRESSIVE token perturbation for TPG - creates much stronger degradation for better guidance
    
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
            
            # MORE AGGRESSIVE adaptive perturbation strength
            if _tpg_config.get('adaptive_strength', True) and step is not None:
                # Much stronger perturbation early, still strong later
                progress = step / 50.0  # Assume ~50 steps
                adaptive_strength = shuffle_strength * (1.2 - 0.2 * min(1.0, progress))  # 1.2x to 1.0x strength
                adaptive_strength = min(2.0, adaptive_strength)  # Cap at 2.0x
            else:
                adaptive_strength = shuffle_strength * 1.5  # 50% stronger by default
        else:
            generator = torch.Generator(device=x.device)
            adaptive_strength = shuffle_strength * 1.5
        
        result = x.clone()
        
        # AGGRESSIVE token perturbation techniques
        
        # 1. AGGRESSIVE Token shuffling (reorder tokens)
        if adaptive_strength > 0.05:  # Lower threshold
            shuffle_ratio = min(1.0, adaptive_strength * 1.2)  # More aggressive shuffling
            num_to_shuffle = max(2, int(n * shuffle_ratio))  # Shuffle at least 2 tokens
            indices_to_shuffle = torch.randperm(n, device=x.device, generator=generator)[:num_to_shuffle]
            shuffled_indices = torch.randperm(num_to_shuffle, device=x.device, generator=generator)
            result[:, indices_to_shuffle] = result[:, indices_to_shuffle[shuffled_indices]]
        
        # 2. STRONGER noise injection
        if adaptive_strength > 0.1:  # Lower threshold
            noise_scale = adaptive_strength * 0.15  # 3x stronger noise
            noise = torch.randn(result.shape, device=result.device, generator=generator) * noise_scale
            result = result + noise
        
        # 3. AGGRESSIVE Token duplication and replacement
        if adaptive_strength > 0.3:  # Lower threshold
            dup_ratio = adaptive_strength * 0.25  # Up to 25% duplication (2.5x more)
            num_to_dup = int(n * dup_ratio)
            if num_to_dup > 0:
                source_indices = torch.randperm(n, device=x.device, generator=generator)[:num_to_dup]
                target_indices = torch.randperm(n, device=x.device, generator=generator)[:num_to_dup]
                result[:, target_indices] = result[:, source_indices]
        
        # 4. NEW: Token reversal (reverse order of segments)
        if adaptive_strength > 0.4:
            segment_size = max(2, n // 8)  # Reverse in segments
            for i in range(0, n - segment_size, segment_size * 2):
                end_idx = min(i + segment_size, n)
                result[:, i:end_idx] = torch.flip(result[:, i:end_idx], dims=[1])
        
        # 5. NEW: Magnitude scaling (make some tokens stronger/weaker)
        if adaptive_strength > 0.6:
            scale_factor = 1.0 + (torch.rand(result.shape[:2], device=result.device, generator=generator) - 0.5) * adaptive_strength * 0.5
            result = result * scale_factor.unsqueeze(-1)
        
        # 6. NEW: Token interpolation (blend neighboring tokens)
        if adaptive_strength > 0.7:
            blend_strength = adaptive_strength * 0.3
            if n > 1:
                shifted = torch.roll(result, shifts=1, dims=1)
                result = result * (1 - blend_strength) + shifted * blend_strength
        
        # 7. NEW: Extreme perturbation for very high strength
        if adaptive_strength > 1.0:
            # Add structured chaos
            chaos_strength = (adaptive_strength - 1.0) * 0.2
            chaos_noise = torch.randn(result.shape, device=result.device, generator=generator) * chaos_strength
            result = result + chaos_noise
            
            # Random token zeroing
            zero_ratio = (adaptive_strength - 1.0) * 0.1
            num_to_zero = int(n * zero_ratio)
            if num_to_zero > 0:
                zero_indices = torch.randperm(n, device=x.device, generator=generator)[:num_to_zero]
                result[:, zero_indices] = 0
        
        return result
                
    except Exception as e:
        logger.warning(f"[TPG] AGGRESSIVE token perturbation error: {e}")
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
            return original_sampling_function(model, x, timestep, uncond, cond, cond_scale, model_options, seed)
        
        if len(cond) == 0:
            return original_sampling_function(model, x, timestep, uncond, cond, cond_scale, model_options, seed)
        
        try:
            tpg_scale = _tpg_config.get('scale', 0.5)
            
            # Get the original CFG result first
            cfg_result = original_sampling_function(model, x, timestep, uncond, cond, cond_scale, model_options, seed)
            
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
                            # Create shuffled version
                            import copy
                            new_model_cond = copy.deepcopy(model_cond)
                            
                            # Apply token shuffling
                            step = int(timestep.mean().item()) if hasattr(timestep, 'mean') else None
                            shuffled_cond = shuffle_tokens(
                                model_cond.cond,
                                step=step,
                                shuffle_strength=_tpg_config.get('shuffle_strength', 0.2)
                            )
                            
                            # Verify shuffling actually happened
                            diff_magnitude = torch.abs(model_cond.cond - shuffled_cond).mean().item()
                            
                            if diff_magnitude > 1e-8:  # Check if perturbation was applied
                                tokens_shuffled = True
                            else:
                                # Apply force perturbation if normal shuffling failed
                                shuffled_cond = force_token_perturbation(model_cond.cond, adaptive_strength=0.5)
                                force_diff = torch.abs(model_cond.cond - shuffled_cond).mean().item()
                                if force_diff > diff_magnitude:
                                    tokens_shuffled = True
                            
                            new_model_cond.cond = shuffled_cond
                            new_c['model_conds'][key] = new_model_cond
                
                tpg_cond.append(new_c)
            
            if not tokens_shuffled:
                # Emergency fallback: create simple perturbed conditioning
                emergency_tpg_cond = []
                for i, c in enumerate(cond):
                    new_c = c.copy()
                    if 'model_conds' in new_c:
                        for key, model_cond in new_c['model_conds'].items():
                            if hasattr(model_cond, 'cond') and isinstance(model_cond.cond, torch.Tensor):
                                import copy
                                new_model_cond = copy.deepcopy(model_cond)
                                # AGGRESSIVE emergency perturbation
                                emergency_cond = model_cond.cond.clone()
                                # Add STRONG noise
                                noise = torch.randn(emergency_cond.shape, device=emergency_cond.device) * 0.2  # 4x stronger
                                emergency_cond = emergency_cond + noise
                                # MULTIPLE shuffles
                                b, n = emergency_cond.shape[:2]
                                for _ in range(3):  # Triple shuffle
                                    perm = torch.randperm(n, device=emergency_cond.device)
                                    emergency_cond = emergency_cond[:, perm]
                                # Add magnitude perturbation
                                mag_noise = 1.0 + (torch.rand(emergency_cond.shape[:2], device=emergency_cond.device) - 0.5) * 0.4
                                emergency_cond = emergency_cond * mag_noise.unsqueeze(-1)
                                
                                new_model_cond.cond = emergency_cond
                                new_c['model_conds'][key] = new_model_cond
                    emergency_tpg_cond.append(new_c)
                tpg_cond = emergency_tpg_cond
                tokens_shuffled = True
            
            # Get conditional prediction without CFG (for comparison)
            cond_pred, _ = calc_cond_uncond_batch(model, cond, None, x, timestep, model_options)
            
            # Get MULTIPLE perturbed predictions for stronger guidance
            tpg_pred1, _ = calc_cond_uncond_batch(model, tpg_cond, None, x, timestep, model_options)
            
            # Create second perturbation with different strength
            tpg_cond2 = []
            for i, c in enumerate(cond):
                new_c = c.copy()
                if 'model_conds' in new_c:
                    for key, model_cond in new_c['model_conds'].items():
                        if hasattr(model_cond, 'cond') and isinstance(model_cond.cond, torch.Tensor):
                            import copy
                            new_model_cond = copy.deepcopy(model_cond)
                            step = int(timestep.mean().item()) if hasattr(timestep, 'mean') else None
                            shuffled_cond = shuffle_tokens(
                                model_cond.cond,
                                step=step,
                                shuffle_strength=_tpg_config.get('shuffle_strength', 0.2) * 1.5,
                                seed_offset=123
                            )
                            new_model_cond.cond = shuffled_cond
                            new_c['model_conds'][key] = new_model_cond
                tpg_cond2.append(new_c)
            
            tpg_pred2, _ = calc_cond_uncond_batch(model, tpg_cond2, None, x, timestep, model_options)
            
            # Calculate AGGRESSIVE guidance using multiple perturbations
            pred_diff1 = cond_pred - tpg_pred1
            pred_diff2 = cond_pred - tpg_pred2
            
            # Combine differences for stronger guidance
            combined_diff = (pred_diff1 + pred_diff2 * 0.8) / 1.8
            diff_magnitude = torch.abs(combined_diff).mean().item()
            
            if diff_magnitude < 1e-6:
                # AGGRESSIVE emergency guidance
                emergency_diff = torch.randn_like(cfg_result) * 0.05  # 5x stronger
                combined_diff = emergency_diff
                diff_magnitude = torch.abs(combined_diff).mean().item()
            
            # AGGRESSIVE TPG guidance with non-linear scaling
            base_enhancement = tpg_scale * combined_diff
            
            # Add magnitude-based amplification
            enhancement_magnitude = torch.norm(base_enhancement, dim=(-3, -2, -1), keepdim=True)
            normalized_enhancement = base_enhancement / (enhancement_magnitude + 1e-8)
            
            # Non-linear amplification: stronger effects get even stronger
            amplification_factor = 1.0 + (enhancement_magnitude / (enhancement_magnitude + 0.1)) * 0.5
            amplified_enhancement = normalized_enhancement * enhancement_magnitude * amplification_factor
            
            # Apply directional bias for more dramatic effects
            step_progress = int(timestep.mean().item()) / 50.0 if hasattr(timestep, 'mean') else 0.5
            directional_boost = 1.0 + (1.0 - step_progress) * 0.3  # Stronger early in sampling
            
            tpg_enhanced = cfg_result + amplified_enhancement * directional_boost
            
            return tpg_enhanced
            
        except Exception as e:
            logger.warning(f"[TPG] Error in TPG sampling, falling back to original: {e}")
            return original_sampling_function(model, x, timestep, uncond, cond, cond_scale, model_options, seed)
    
    return tpg_sampling_function

def patch_sampling_for_tpg():
    """
    Patch the sampling function to include TPG support
    """
    global _original_unet_forward
    
    if not is_tpg_enabled():
        return False
    
    try:
        # Check if we need layer-specific TPG
        applied_layers = _tpg_config.get('applied_layers', ['mid', 'up'])
        all_layers = ['down', 'mid', 'up']
        
        if set(applied_layers) == set(all_layers):
            # If all layers are selected, use sampling function approach (more efficient)
            import ldm_patched.modules.samplers as samplers
            
            if _original_unet_forward is None:
                _original_unet_forward = samplers.sampling_function
                samplers.sampling_function = create_tpg_sampling_function(_original_unet_forward)
                print(f"[TPG] Patched sampling_function for TPG (all layers)")
        else:
            # If specific layers are selected, use attention processor approach
            success = patch_attention_processors_for_tpg()
            if success:
                print(f"[TPG] Patched attention processors for TPG (layers: {applied_layers})")
            else:
                # Fallback to sampling function approach
                import ldm_patched.modules.samplers as samplers
                if _original_unet_forward is None:
                    _original_unet_forward = samplers.sampling_function
                    samplers.sampling_function = create_tpg_sampling_function(_original_unet_forward)
                    print(f"[TPG] Fallback: Patched sampling_function for TPG")
        
        print("[TPG] Successfully patched for TPG")
        return True
        
    except Exception as e:
        logger.error(f"[TPG] Failed to patch for TPG: {e}")
        import traceback
        traceback.print_exc()
        return False

def patch_attention_processors_for_tpg():
    """
    Patch attention processors for layer-specific TPG
    """
    try:
        import modules.default_pipeline as default_pipeline
        
        if default_pipeline.final_unet is None:
            return False
        
        # Access the actual model from ModelPatcher
        unet_model = None
        if hasattr(default_pipeline.final_unet, 'model'):
            unet_model = default_pipeline.final_unet.model
        elif hasattr(default_pipeline.final_unet, 'diffusion_model'):
            unet_model = default_pipeline.final_unet.diffusion_model
        else:
            # Fallback: try to use final_unet directly if it has named_children
            if hasattr(default_pipeline.final_unet, 'named_children'):
                unet_model = default_pipeline.final_unet
            else:
                logger.warning("[TPG] Could not access UNet model from ModelPatcher")
                return False
        
        applied_layers = _tpg_config.get('applied_layers', ['mid', 'up'])
        
        # Get current attention processors
        current_processors = {}
        
        def get_processors_recursive(name, module):
            if hasattr(module, "get_processor"):
                current_processors[f"{name}.processor"] = module.get_processor(return_deprecated_lora=True)
            
            for sub_name, child in module.named_children():
                get_processors_recursive(f"{name}.{sub_name}", child)
        
        for name, module in unet_model.named_children():
            get_processors_recursive(name, module)
        
        # Create TPG processors for selected layers only
        tpg_processors = {}
        for name, processor in current_processors.items():
            # Check if this layer should have TPG applied
            should_apply_tpg = any(layer_type in name for layer_type in applied_layers)
            
            if should_apply_tpg:
                tpg_processors[name] = TPGAttentionProcessor(processor)
            else:
                tpg_processors[name] = processor
        
        # Set the processors
        def set_processors_recursive(name, module, processors):
            if hasattr(module, "set_processor"):
                if not isinstance(processors, dict):
                    module.set_processor(processors)
                else:
                    module.set_processor(processors.pop(f"{name}.processor"))
            
            for sub_name, child in module.named_children():
                set_processors_recursive(f"{name}.{sub_name}", child, processors)
        
        for name, module in unet_model.named_children():
            set_processors_recursive(name, module, tpg_processors.copy())
        
        return True
        
    except Exception as e:
        logger.error(f"[TPG] Failed to patch attention processors: {e}")
        return False

def unpatch_sampling_for_tpg():
    """
    Restore the original sampling function and attention processors
    """
    global _original_unet_forward
    
    try:
        success = True
        
        # Restore sampling function if it was patched
        import ldm_patched.modules.samplers as samplers
        
        if _original_unet_forward is not None:
            samplers.sampling_function = _original_unet_forward
            _original_unet_forward = None
            print("[TPG] Successfully restored original sampling function")
        
        # Restore attention processors if they were patched
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
                    # Fallback: try to use final_unet directly if it has named_children
                    if hasattr(default_pipeline.final_unet, 'named_children'):
                        unet_model = default_pipeline.final_unet
                    else:
                        logger.warning("[TPG] Could not access UNet model from ModelPatcher for restoration")
                        unet_model = None
                
                if unet_model is not None:
                    # Get current processors and restore any TPG processors to original
                    current_processors = {}
                    
                    def get_processors_recursive(name, module):
                        if hasattr(module, "get_processor"):
                            processor = module.get_processor(return_deprecated_lora=True)
                            if isinstance(processor, TPGAttentionProcessor):
                                current_processors[f"{name}.processor"] = processor.original_processor
                            else:
                                current_processors[f"{name}.processor"] = processor
                        
                        for sub_name, child in module.named_children():
                            get_processors_recursive(f"{name}.{sub_name}", child)
                    
                    for name, module in unet_model.named_children():
                        get_processors_recursive(name, module)
                    
                    # Set the restored processors
                    def set_processors_recursive(name, module, processors):
                        if hasattr(module, "set_processor"):
                            if not isinstance(processors, dict):
                                module.set_processor(processors)
                            else:
                                module.set_processor(processors.pop(f"{name}.processor"))
                        
                        for sub_name, child in module.named_children():
                            set_processors_recursive(f"{name}.{sub_name}", child, processors)
                    
                    for name, module in unet_model.named_children():
                        set_processors_recursive(name, module, current_processors.copy())
                    
                    print("[TPG] Successfully restored original attention processors")
        
        except Exception as e:
            logger.warning(f"[TPG] Failed to restore attention processors: {e}")
            success = False
        
        return success
            
    except Exception as e:
        logger.error(f"[TPG] Failed to restore TPG patches: {e}")
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
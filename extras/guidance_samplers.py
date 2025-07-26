# Guidance-enhanced samplers for Fooocus
# TPG, NAG, and PAG implementations integrated with k_diffusion sampling

import torch
import math
from tqdm.auto import trange

# Global guidance configuration
_guidance_config = {
    'tpg_scale': 0.0,
    'nag_scale': 1.0,
    'pag_scale': 0.0
}

def set_guidance_config(tpg_scale=0.0, nag_scale=1.0, pag_scale=0.0):
    """Set global guidance configuration"""
    global _guidance_config
    _guidance_config.update({
        'tpg_scale': tpg_scale,
        'nag_scale': nag_scale,
        'pag_scale': pag_scale
    })
    print(f"[GUIDANCE] Config updated: TPG={tpg_scale}, NAG={nag_scale}, PAG={pag_scale}")

def get_guidance_config():
    """Get current guidance configuration"""
    return _guidance_config.copy()

def shuffle_tokens(x, step=None, seed_offset=0):
    """Shuffle tokens for TPG - creates different shuffling at each step"""
    try:
        if len(x.shape) >= 2:
            b, n = x.shape[:2]
            
            # Create different shuffling for each step
            if step is not None:
                # Use step-based seed for reproducible but different shuffling each step
                generator = torch.Generator(device=x.device)
                generator.manual_seed(hash((step + seed_offset)) % (2**32))
                permutation = torch.randperm(n, device=x.device, generator=generator)
            else:
                # Random shuffling if no step provided
                permutation = torch.randperm(n, device=x.device)
                
            return x[:, permutation]
        return x
    except Exception:
        return x

def apply_attention_degradation(embeddings, degradation_strength=0.5):
    """Apply attention degradation for NAG - reduces attention weights to weaken conditioning"""
    try:
        # NAG works by degrading the attention mechanism, not just the embeddings
        # This creates a "negative" version that can be used for guidance
        degraded = embeddings * (1.0 - degradation_strength)
        
        # Add controlled noise to break attention patterns
        if degradation_strength > 0:
            noise_scale = degradation_strength * 0.1
            noise = torch.randn_like(embeddings) * noise_scale
            degraded = degraded + noise
            
        return degraded
    except Exception:
        return embeddings

def apply_attention_perturbation(embeddings):
    """Apply attention perturbation for PAG"""
    try:
        # Add controlled noise perturbation
        noise = torch.randn_like(embeddings) * 0.15
        return embeddings + noise
    except Exception:
        return embeddings

def to_d(x, sigma, denoised):
    """Converts a denoiser output to a Karras ODE derivative."""
    from ldm_patched.k_diffusion import utils
    return (x - denoised) / utils.append_dims(sigma, x.ndim)

@torch.no_grad()
def sample_euler_tpg(model, x, sigmas, extra_args=None, callback=None, disable=None, 
                     tpg_scale=None, s_churn=0., s_tmin=0., s_tmax=float('inf'), s_noise=1.):
    """Euler method with TPG (Token Perturbation Guidance)"""
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    
    # Get TPG scale from global config if not provided
    if tpg_scale is None:
        tpg_scale = _guidance_config.get('tpg_scale', 3.0)
    
    print(f"[TPG] Using TPG-enhanced Euler sampler with scale {tpg_scale}")
    
    for i in trange(len(sigmas) - 1, disable=disable):
        gamma = min(s_churn / (len(sigmas) - 1), 2 ** 0.5 - 1) if s_tmin <= sigmas[i] <= s_tmax else 0.
        sigma_hat = sigmas[i] * (gamma + 1)
        if gamma > 0:
            eps = torch.randn_like(x) * s_noise
            x = x + eps * (sigma_hat ** 2 - sigmas[i] ** 2) ** 0.5
        
        # Standard denoising
        denoised = model(x, sigma_hat * s_in, **extra_args)
        
        # Apply TPG if we have conditioning
        if 'cond' in extra_args and len(extra_args['cond']) > 0:
            try:
                # Create TPG conditioning by shuffling tokens
                tpg_extra_args = extra_args.copy()
                tpg_cond = []
                
                for c in extra_args['cond']:
                    new_c = c.copy()
                    if 'model_conds' in new_c:
                        for key, model_cond in new_c['model_conds'].items():
                            if hasattr(model_cond, 'cond') and isinstance(model_cond.cond, torch.Tensor):
                                # Create shuffled version
                                import copy
                                new_model_cond = copy.deepcopy(model_cond)
                                new_model_cond.cond = shuffle_tokens(model_cond.cond)
                                new_c['model_conds'][key] = new_model_cond
                    tpg_cond.append(new_c)
                
                tpg_extra_args['cond'] = tpg_cond
                
                # Get TPG prediction
                denoised_tpg = model(x, sigma_hat * s_in, **tpg_extra_args)
                
                # Apply TPG guidance
                denoised = denoised + tpg_scale * (denoised - denoised_tpg)
                
            except Exception as e:
                print(f"[TPG] Error applying TPG, using standard denoising: {e}")
        
        d = to_d(x, sigma_hat, denoised)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigma_hat, 'denoised': denoised})
        dt = sigmas[i + 1] - sigma_hat
        x = x + d * dt
    
    return x

@torch.no_grad()
def sample_euler_nag(model, x, sigmas, extra_args=None, callback=None, disable=None,
                     nag_scale=None, s_churn=0., s_tmin=0., s_tmax=float('inf'), s_noise=1.):
    """Euler method with NAG (Negative Attention Guidance)
    
    NAG restores effective negative prompting in few-step models by degrading
    positive conditioning, allowing negative prompts to have stronger influence.
    This enables direct suppression of visual, semantic, and stylistic attributes.
    """
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    
    # Get NAG scale from global config if not provided
    if nag_scale is None:
        nag_scale = _guidance_config.get('nag_scale', 1.5)
    
    print(f"[NAG] Using NAG-enhanced Euler sampler with scale {nag_scale}")
    print("[NAG] NAG restores effective negative prompting for better controllability")
    
    for i in trange(len(sigmas) - 1, disable=disable):
        gamma = min(s_churn / (len(sigmas) - 1), 2 ** 0.5 - 1) if s_tmin <= sigmas[i] <= s_tmax else 0.
        sigma_hat = sigmas[i] * (gamma + 1)
        if gamma > 0:
            eps = torch.randn_like(x) * s_noise
            x = x + eps * (sigma_hat ** 2 - sigmas[i] ** 2) ** 0.5
        
        # Standard denoising with CFG
        denoised = model(x, sigma_hat * s_in, **extra_args)
        
        # Apply NAG if we have both positive and negative conditioning
        if ('cond' in extra_args and len(extra_args['cond']) > 0 and 
            'uncond' in extra_args and len(extra_args['uncond']) > 0 and
            nag_scale > 1.0):
            try:
                # NAG Strategy: Create a "null" or degraded positive conditioning
                # This allows negative prompts to have stronger relative influence
                
                # Method 1: Degrade positive conditioning strength
                nag_extra_args = extra_args.copy()
                nag_cond = []
                
                for c in extra_args['cond']:
                    new_c = c.copy()
                    if 'model_conds' in new_c:
                        for key, model_cond in new_c['model_conds'].items():
                            if hasattr(model_cond, 'cond') and isinstance(model_cond.cond, torch.Tensor):
                                import copy
                                new_model_cond = copy.deepcopy(model_cond)
                                # Strong degradation to create "null" conditioning
                                new_model_cond.cond = apply_attention_degradation(
                                    model_cond.cond, 
                                    degradation_strength=0.8  # Very strong degradation
                                )
                                new_c['model_conds'][key] = new_model_cond
                    nag_cond.append(new_c)
                
                nag_extra_args['cond'] = nag_cond
                
                # Get prediction with degraded positive conditioning
                # This simulates what happens when positive prompts have less influence
                denoised_nag = model(x, sigma_hat * s_in, **nag_extra_args)
                
                # NAG guidance: The difference shows what the positive conditioning adds
                # By amplifying this difference, we enhance the relative strength of negative prompts
                positive_contribution = denoised - denoised_nag
                
                # Apply NAG: Reduce positive contribution, enhancing negative prompt effectiveness
                nag_strength = (nag_scale - 1.0)
                denoised = denoised + nag_strength * positive_contribution
                
                if i % 10 == 0:  # Log occasionally
                    print(f"[NAG] Step {i}: Applied NAG guidance (strength: {nag_strength:.2f})")
                
            except Exception as e:
                print(f"[NAG] Error applying NAG, using standard denoising: {e}")
        
        d = to_d(x, sigma_hat, denoised)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigma_hat, 'denoised': denoised})
        dt = sigmas[i + 1] - sigma_hat
        x = x + d * dt
    
    return x

@torch.no_grad()
def sample_euler_pag(model, x, sigmas, extra_args=None, callback=None, disable=None,
                     pag_scale=None, s_churn=0., s_tmin=0., s_tmax=float('inf'), s_noise=1.):
    """Euler method with PAG (Perturbed Attention Guidance)"""
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    
    # Get PAG scale from global config if not provided
    if pag_scale is None:
        pag_scale = _guidance_config.get('pag_scale', 3.0)
    
    print(f"[PAG] Using PAG-enhanced Euler sampler with scale {pag_scale}")
    
    for i in trange(len(sigmas) - 1, disable=disable):
        gamma = min(s_churn / (len(sigmas) - 1), 2 ** 0.5 - 1) if s_tmin <= sigmas[i] <= s_tmax else 0.
        sigma_hat = sigmas[i] * (gamma + 1)
        if gamma > 0:
            eps = torch.randn_like(x) * s_noise
            x = x + eps * (sigma_hat ** 2 - sigmas[i] ** 2) ** 0.5
        
        # Standard denoising
        denoised = model(x, sigma_hat * s_in, **extra_args)
        
        # Apply PAG if we have conditioning
        if 'cond' in extra_args and len(extra_args['cond']) > 0:
            try:
                # Create PAG conditioning by perturbing attention
                pag_extra_args = extra_args.copy()
                pag_cond = []
                
                for c in extra_args['cond']:
                    new_c = c.copy()
                    if 'model_conds' in new_c:
                        for key, model_cond in new_c['model_conds'].items():
                            if hasattr(model_cond, 'cond') and isinstance(model_cond.cond, torch.Tensor):
                                # Create perturbed version
                                import copy
                                new_model_cond = copy.deepcopy(model_cond)
                                new_model_cond.cond = apply_attention_perturbation(model_cond.cond)
                                new_c['model_conds'][key] = new_model_cond
                    pag_cond.append(new_c)
                
                pag_extra_args['cond'] = pag_cond
                
                # Get PAG prediction
                denoised_pag = model(x, sigma_hat * s_in, **pag_extra_args)
                
                # Apply PAG guidance
                denoised = denoised + pag_scale * (denoised - denoised_pag)
                
            except Exception as e:
                print(f"[PAG] Error applying PAG, using standard denoising: {e}")
        
        d = to_d(x, sigma_hat, denoised)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigma_hat, 'denoised': denoised})
        dt = sigmas[i + 1] - sigma_hat
        x = x + d * dt
    
    return x

# Combined guidance sampler
@torch.no_grad()
def sample_euler_guidance(model, x, sigmas, extra_args=None, callback=None, disable=None,
                         tpg_scale=None, nag_scale=None, pag_scale=None,
                         s_churn=0., s_tmin=0., s_tmax=float('inf'), s_noise=1.):
    """Euler method with combined TPG, NAG, and PAG guidance"""
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    
    # Get guidance scales from global config if not provided
    if tpg_scale is None:
        tpg_scale = _guidance_config.get('tpg_scale', 0.0)
    if nag_scale is None:
        nag_scale = _guidance_config.get('nag_scale', 1.0)
    if pag_scale is None:
        pag_scale = _guidance_config.get('pag_scale', 0.0)
    
    # Count active guidance methods
    active_methods = []
    if tpg_scale > 0:
        active_methods.append(f"TPG({tpg_scale})")
    if nag_scale > 1.0:
        active_methods.append(f"NAG({nag_scale})")
    if pag_scale > 0:
        active_methods.append(f"PAG({pag_scale})")
    
    if active_methods:
        print(f"[GUIDANCE] Using combined guidance: {', '.join(active_methods)}")
    
    for i in trange(len(sigmas) - 1, disable=disable):
        gamma = min(s_churn / (len(sigmas) - 1), 2 ** 0.5 - 1) if s_tmin <= sigmas[i] <= s_tmax else 0.
        sigma_hat = sigmas[i] * (gamma + 1)
        if gamma > 0:
            eps = torch.randn_like(x) * s_noise
            x = x + eps * (sigma_hat ** 2 - sigmas[i] ** 2) ** 0.5
        
        # Standard denoising
        denoised = model(x, sigma_hat * s_in, **extra_args)
        
        # Apply guidance if we have conditioning and any guidance is enabled
        if ('cond' in extra_args and len(extra_args['cond']) > 0 and 
            (tpg_scale > 0 or nag_scale > 1.0 or pag_scale > 0)):
            
            try:
                guidance_sum = torch.zeros_like(denoised)
                guidance_count = 0
                
                # Apply TPG
                if tpg_scale > 0:
                    tpg_extra_args = extra_args.copy()
                    tpg_cond = []
                    for c in extra_args['cond']:
                        new_c = c.copy()
                        if 'model_conds' in new_c:
                            for key, model_cond in new_c['model_conds'].items():
                                if hasattr(model_cond, 'cond') and isinstance(model_cond.cond, torch.Tensor):
                                    import copy
                                    new_model_cond = copy.deepcopy(model_cond)
                                    new_model_cond.cond = shuffle_tokens(model_cond.cond)
                                    new_c['model_conds'][key] = new_model_cond
                        tpg_cond.append(new_c)
                    tpg_extra_args['cond'] = tpg_cond
                    denoised_tpg = model(x, sigma_hat * s_in, **tpg_extra_args)
                    guidance_sum += tpg_scale * (denoised - denoised_tpg)
                    guidance_count += 1
                
                # Apply NAG - enhances negative prompting effectiveness
                if nag_scale > 1.0 and 'uncond' in extra_args and len(extra_args['uncond']) > 0:
                    nag_extra_args = extra_args.copy()
                    nag_cond = []
                    for c in extra_args['cond']:
                        new_c = c.copy()
                        if 'model_conds' in new_c:
                            for key, model_cond in new_c['model_conds'].items():
                                if hasattr(model_cond, 'cond') and isinstance(model_cond.cond, torch.Tensor):
                                    import copy
                                    new_model_cond = copy.deepcopy(model_cond)
                                    new_model_cond.cond = apply_attention_degradation(
                                        model_cond.cond, 
                                        degradation_strength=0.7
                                    )
                                    new_c['model_conds'][key] = new_model_cond
                        nag_cond.append(new_c)
                    nag_extra_args['cond'] = nag_cond
                    denoised_nag = model(x, sigma_hat * s_in, **nag_extra_args)
                    # NAG enhances the difference to restore negative prompting effectiveness
                    guidance_sum += (nag_scale - 1.0) * (denoised - denoised_nag)
                    guidance_count += 1
                
                # Apply PAG
                if pag_scale > 0:
                    pag_extra_args = extra_args.copy()
                    pag_cond = []
                    for c in extra_args['cond']:
                        new_c = c.copy()
                        if 'model_conds' in new_c:
                            for key, model_cond in new_c['model_conds'].items():
                                if hasattr(model_cond, 'cond') and isinstance(model_cond.cond, torch.Tensor):
                                    import copy
                                    new_model_cond = copy.deepcopy(model_cond)
                                    new_model_cond.cond = apply_attention_perturbation(model_cond.cond)
                                    new_c['model_conds'][key] = new_model_cond
                        pag_cond.append(new_c)
                    pag_extra_args['cond'] = pag_cond
                    denoised_pag = model(x, sigma_hat * s_in, **pag_extra_args)
                    guidance_sum += pag_scale * (denoised - denoised_pag)
                    guidance_count += 1
                
                # Apply combined guidance
                if guidance_count > 0:
                    denoised = denoised + guidance_sum / guidance_count
                
            except Exception as e:
                print(f"[GUIDANCE] Error applying guidance, using standard denoising: {e}")
        
        d = to_d(x, sigma_hat, denoised)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigma_hat, 'denoised': denoised})
        dt = sigmas[i + 1] - sigma_hat
        x = x + d * dt
    
    return x
# Guidance-enhanced samplers for Fooocus
# TPG, NAG, and PAG implementations integrated with k_diffusion sampling

import torch
import math
from tqdm.auto import trange

def shuffle_tokens(x):
    """Shuffle tokens for TPG"""
    try:
        if len(x.shape) >= 2:
            b, n = x.shape[:2]
            permutation = torch.randperm(n, device=x.device)
            return x[:, permutation]
        return x
    except Exception:
        return x

def apply_attention_degradation(embeddings):
    """Apply attention degradation for NAG"""
    try:
        # Simple degradation: reduce magnitude and add noise
        degraded = embeddings * 0.7
        noise = torch.randn_like(embeddings) * 0.1
        return degraded + noise
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
                     tpg_scale=3.0, s_churn=0., s_tmin=0., s_tmax=float('inf'), s_noise=1.):
    """Euler method with TPG (Token Perturbation Guidance)"""
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    
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
                     nag_scale=1.5, s_churn=0., s_tmin=0., s_tmax=float('inf'), s_noise=1.):
    """Euler method with NAG (Negative Attention Guidance)"""
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    
    print(f"[NAG] Using NAG-enhanced Euler sampler with scale {nag_scale}")
    
    for i in trange(len(sigmas) - 1, disable=disable):
        gamma = min(s_churn / (len(sigmas) - 1), 2 ** 0.5 - 1) if s_tmin <= sigmas[i] <= s_tmax else 0.
        sigma_hat = sigmas[i] * (gamma + 1)
        if gamma > 0:
            eps = torch.randn_like(x) * s_noise
            x = x + eps * (sigma_hat ** 2 - sigmas[i] ** 2) ** 0.5
        
        # Standard denoising
        denoised = model(x, sigma_hat * s_in, **extra_args)
        
        # Apply NAG if we have conditioning
        if 'cond' in extra_args and len(extra_args['cond']) > 0:
            try:
                # Create NAG conditioning by degrading attention
                nag_extra_args = extra_args.copy()
                nag_cond = []
                
                for c in extra_args['cond']:
                    new_c = c.copy()
                    if 'model_conds' in new_c:
                        for key, model_cond in new_c['model_conds'].items():
                            if hasattr(model_cond, 'cond') and isinstance(model_cond.cond, torch.Tensor):
                                # Create degraded version
                                import copy
                                new_model_cond = copy.deepcopy(model_cond)
                                new_model_cond.cond = apply_attention_degradation(model_cond.cond)
                                new_c['model_conds'][key] = new_model_cond
                    nag_cond.append(new_c)
                
                nag_extra_args['cond'] = nag_cond
                
                # Get NAG prediction
                denoised_nag = model(x, sigma_hat * s_in, **nag_extra_args)
                
                # Apply NAG guidance
                denoised = denoised + nag_scale * (denoised - denoised_nag)
                
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
                     pag_scale=3.0, s_churn=0., s_tmin=0., s_tmax=float('inf'), s_noise=1.):
    """Euler method with PAG (Perturbed Attention Guidance)"""
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    
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
                         tpg_scale=0.0, nag_scale=1.0, pag_scale=0.0,
                         s_churn=0., s_tmin=0., s_tmax=float('inf'), s_noise=1.):
    """Euler method with combined TPG, NAG, and PAG guidance"""
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    
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
                
                # Apply NAG
                if nag_scale > 1.0:
                    nag_extra_args = extra_args.copy()
                    nag_cond = []
                    for c in extra_args['cond']:
                        new_c = c.copy()
                        if 'model_conds' in new_c:
                            for key, model_cond in new_c['model_conds'].items():
                                if hasattr(model_cond, 'cond') and isinstance(model_cond.cond, torch.Tensor):
                                    import copy
                                    new_model_cond = copy.deepcopy(model_cond)
                                    new_model_cond.cond = apply_attention_degradation(model_cond.cond)
                                    new_c['model_conds'][key] = new_model_cond
                        nag_cond.append(new_c)
                    nag_extra_args['cond'] = nag_cond
                    denoised_nag = model(x, sigma_hat * s_in, **nag_extra_args)
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
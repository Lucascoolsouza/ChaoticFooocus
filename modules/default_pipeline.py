import modules.core as core
import os
import torch
import numpy as np # Added import for numpy
from PIL import Image # Added import for PIL Image
import sys
import os

# Debug functions (disco_diffusion removed)
DEBUG_AVAILABLE = False

# Fallback debug functions
def debug_latent_pass(latent, name="latent"):
    if latent is None:
        print(f"[{name}] None")
        return latent
    print(f"[{name}] shape={tuple(latent.shape)}, device={latent.device}, dtype={latent.dtype}")
    return latent

def preview_latent(*args, **kwargs):
    return None

# Debug utilities removed with disco_diffusion

import modules.patch
import modules.config
import modules.flags
import ldm_patched.modules.model_management
import ldm_patched.modules.latent_formats
import modules.inpaint_worker
import extras.vae_interpose as vae_interpose
from extras.expansion import FooocusExpansion

from ldm_patched.modules.model_base import SDXL, SDXLRefiner
from modules.sample_hijack import clip_separate
from modules.util import get_file_from_folder_list, get_enabled_loras
# Guidance samplers are now integrated into k_diffusion sampling
# No need for separate sampler classes

def apply_psychedelic_latent_effects(latent, progress, params):
    """Apply psychedelic effects directly to latent space during diffusion"""
    if latent is None or params.get('intensity', 0) <= 0:
        return latent
    
    try:
        import torch
        
        # Get latent dimensions
        B, C, H, W = latent.shape
        device = latent.device
        
        # Calculate effect intensity based on progress and peak
        peak = params.get('peak', 0.5)
        intensity = params.get('intensity', 0.45)
        
        # Create distance from peak for intensity modulation - MUCH stronger
        peak_distance = abs(progress - peak)
        peak_intensity = intensity * 3.0 * torch.exp(torch.tensor(-peak_distance * 2))  # 3x stronger, wider peak
        
        # Create coordinate grids for spatial effects
        y_coords = torch.linspace(0, 1, H, device=device)
        x_coords = torch.linspace(0, 1, W, device=device)
        Y, X = torch.meshgrid(y_coords, x_coords, indexing='ij')
        
        # Apply mode-specific effects
        mode = params.get('mode', 'kaleidoscope')
        wave_freq = params.get('wave_frequency', 2.5)
        flow_mult = params.get('flow_multiplier', 1.2)
        
        if mode == 'kaleidoscope':
            # Radial swirl effect in latent space
            center_y, center_x = H // 2, W // 2
            radius = torch.sqrt((torch.arange(H, device=device)[:, None] - center_y)**2 + 
                              (torch.arange(W, device=device) - center_x)**2)
            max_radius = torch.sqrt(torch.tensor(center_y**2 + center_x**2, device=device))
            normalized_radius = radius / max_radius
            
            # Create swirl pattern - MUCH stronger
            angle = torch.atan2(torch.arange(H, device=device)[:, None] - center_y,
                              torch.arange(W, device=device) - center_x)
            swirl_strength = peak_intensity * 0.5 * normalized_radius  # 5x stronger swirl
            
            # Apply channel mixing based on radial position - MUCH stronger
            for c in range(C):
                channel_offset = c * 2.0 * 3.14159 / C  # Distribute channels around circle
                channel_swirl = torch.sin(angle + channel_offset + swirl_strength * wave_freq)
                latent[:, c] = latent[:, c] * (1 + channel_swirl * peak_intensity * 0.3)  # 6x stronger modulation
                
        elif mode == 'fluid':
            # Flowing wave patterns
            wave_x = torch.sin(X * wave_freq * 2 * 3.14159 + progress * 3.14159) * flow_mult
            wave_y = torch.cos(Y * wave_freq * 2 * 3.14159 + progress * 3.14159) * flow_mult
            
            # Apply wave-based channel modulation - MUCH stronger
            for c in range(C):
                phase_offset = c * 3.14159 / 2  # Different phase for each channel
                wave_pattern = torch.sin(wave_x + wave_y + phase_offset)
                latent[:, c] = latent[:, c] * (1 + wave_pattern * peak_intensity * 0.2)  # 7x stronger waves
                
        elif mode == 'fractal':
            # Fractal-like recursive patterns
            fractal_pattern = torch.zeros_like(Y)
            for i in range(3):  # Limited iterations for performance
                freq = 2 ** (i * 0.5)
                fractal_wave = torch.sin(Y * freq * 3.14159) * torch.cos(X * freq * 3.14159)
                fractal_pattern += fractal_wave * (0.5 ** i)
            
            fractal_pattern = (fractal_pattern + 1) / 2  # Normalize
            
            # Apply fractal modulation to channels - MUCH stronger
            for c in range(C):
                latent[:, c] = latent[:, c] * (1 + fractal_pattern * peak_intensity * 0.25)  # 6x stronger fractals
                
        else:  # 'both' or combined effects
            # Combination of radial and wave effects
            center_y, center_x = H // 2, W // 2
            radius = torch.sqrt((torch.arange(H, device=device)[:, None] - center_y)**2 + 
                              (torch.arange(W, device=device) - center_x)**2)
            max_radius = torch.sqrt(torch.tensor(center_y**2 + center_x**2, device=device))
            normalized_radius = radius / max_radius
            
            # Combine radial and wave patterns
            radial_pattern = torch.exp(-((normalized_radius - 0.5) ** 2) / (0.3 ** 2))
            wave_pattern = torch.sin(Y * wave_freq * 3.14159) * torch.cos(X * wave_freq * 3.14159)
            combined_pattern = radial_pattern * wave_pattern
            
            # Apply to all channels with slight phase differences - MUCH stronger
            for c in range(C):
                phase = c * 3.14159 / C
                channel_pattern = combined_pattern * torch.cos(torch.tensor(phase))
                latent[:, c] = latent[:, c] * (1 + channel_pattern * peak_intensity * 0.2)  # 7x stronger combined
        
        # Apply bias to enhance effects
        bias = params.get('bias', 0.85)
        if bias != 1.0:
            # Enhance the modified latent based on bias
            original_mean = torch.mean(latent, dim=[2, 3], keepdim=True)
            latent = latent * bias + original_mean * (1 - bias)
        
        return latent
        
    except Exception as e:
        print(f"[Psychedelic Daemon] Error in latent effects: {e}")
        return latent


model_base = core.StableDiffusionModel()
model_refiner = core.StableDiffusionModel()

final_expansion = None
final_unet = None
final_clip = None
final_vae = None
final_refiner_unet = None
final_refiner_vae = None

loaded_ControlNets = {}


@torch.no_grad()
@torch.inference_mode()
def refresh_controlnets(model_paths):
    global loaded_ControlNets
    cache = {}
    for p in model_paths:
        if p is not None:
            if p in loaded_ControlNets:
                cache[p] = loaded_ControlNets[p]
            else:
                cache[p] = core.load_controlnet(p)
    loaded_ControlNets = cache
    return


@torch.no_grad()
@torch.inference_mode()
def assert_model_integrity():
    error_message = None

    if not isinstance(model_base.unet_with_lora.model, SDXL):
        error_message = 'You have selected base model other than SDXL. This is not supported yet.'

    if error_message is not None:
        raise NotImplementedError(error_message)

    return True


@torch.no_grad()
@torch.inference_mode()
def refresh_base_model(name, vae_name=None, artistic_strength=0.0):
    global model_base

    filename = get_file_from_folder_list(name, modules.config.paths_checkpoints)

    vae_filename = None
    if vae_name is not None and vae_name != modules.flags.default_vae:
        vae_filename = get_file_from_folder_list(vae_name, modules.config.path_vae)

    if model_base.filename == filename and model_base.vae_filename == vae_filename:
        return

    model_base = core.load_model(filename, vae_filename, artistic_strength=artistic_strength)
    print(f'Base model loaded: {model_base.filename}')
    print(f'VAE loaded: {model_base.vae_filename}')
    return


@torch.no_grad()
@torch.inference_mode()
def refresh_refiner_model(name):
    global model_refiner

    filename = get_file_from_folder_list(name, modules.config.paths_checkpoints)

    if model_refiner.filename == filename:
        return

    model_refiner = core.StableDiffusionModel()

    if name == 'None':
        print(f'Refiner unloaded.')
        return

    model_refiner = core.load_model(filename)
    print(f'Refiner model loaded: {model_refiner.filename}')

    if isinstance(model_refiner.unet.model, SDXL):
        model_refiner.clip = None
        model_refiner.vae = None
    elif isinstance(model_refiner.unet.model, SDXLRefiner):
        model_refiner.clip = None
        model_refiner.vae = None
    else:
        model_refiner.clip = None

    return


@torch.no_grad()
@torch.inference_mode()
def synthesize_refiner_model():
    global model_base, model_refiner

    print('Synthetic Refiner Activated')
    model_refiner = core.StableDiffusionModel(
        unet=model_base.unet,
        vae=model_base.vae,
        clip=model_base.clip,
        clip_vision=model_base.clip_vision,
        filename=model_base.filename
    )
    model_refiner.vae = None
    model_refiner.clip = None
    model_refiner.clip_vision = None

    return


@torch.no_grad()
@torch.inference_mode()
def refresh_loras(loras, base_model_additional_loras=None):
    global model_base, model_refiner

    if not isinstance(base_model_additional_loras, list):
        base_model_additional_loras = []

    model_base.refresh_loras(loras + base_model_additional_loras)
    model_refiner.refresh_loras(loras)

    return


@torch.no_grad()
@torch.inference_mode()
def clip_encode_single(clip, text, verbose=False):
    cached = clip.fcs_cond_cache.get(text, None)
    if cached is not None:
        if verbose:
            print(f'[CLIP Cached] {text}')
        return cached
    tokens = clip.tokenize(text)
    result = clip.encode_from_tokens(tokens, return_pooled=True)
    clip.fcs_cond_cache[text] = result
    if verbose:
        print(f'[CLIP Encoded] {text}')
    return result


@torch.no_grad()
@torch.inference_mode()
def clone_cond(conds):
    results = []

    for c, p in conds:
        p = p["pooled_output"]

        if isinstance(c, torch.Tensor):
            c = c.clone()

        if isinstance(p, torch.Tensor):
            p = p.clone()

        results.append([c, {"pooled_output": p}])

    return results


@torch.no_grad()
@torch.inference_mode()
def clip_encode(texts, pool_top_k=1):
    global final_clip

    if final_clip is None:
        return None
    if not isinstance(texts, list):
        return None
    if len(texts) == 0:
        return None

    cond_list = []
    pooled_acc = 0

    for i, text in enumerate(texts):
        cond, pooled = clip_encode_single(final_clip, text)
        cond_list.append(cond)
        if i < pool_top_k:
            pooled_acc += pooled

    return [[torch.cat(cond_list, dim=1), {"pooled_output": pooled_acc}]]


@torch.no_grad()
@torch.inference_mode()
def set_clip_skip(clip_skip: int):
    global final_clip

    if final_clip is None:
        return

    final_clip.clip_layer(-abs(clip_skip))
    return

@torch.no_grad()
@torch.inference_mode()
def clear_all_caches():
    final_clip.fcs_cond_cache = {}


@torch.no_grad()
@torch.inference_mode()
def prepare_text_encoder(async_call=True):
    if async_call:
        # TODO: make sure that this is always called in an async way so that users cannot feel it.
        pass
    assert_model_integrity()
    ldm_patched.modules.model_management.load_models_gpu([final_clip.patcher, final_expansion.patcher])
    return


@torch.no_grad()
@torch.inference_mode()
def refresh_everything(refiner_model_name, base_model_name, loras,
                       base_model_additional_loras=None, use_synthetic_refiner=False, vae_name=None, artistic_strength=0.0):
    global final_unet, final_clip, final_vae, final_refiner_unet, final_refiner_vae, final_expansion

    final_unet = None
    final_clip = None
    final_vae = None
    final_refiner_unet = None
    final_refiner_vae = None

    if use_synthetic_refiner and refiner_model_name == 'None':
        print('Synthetic Refiner Activated')
        refresh_base_model(base_model_name, vae_name, artistic_strength=artistic_strength)
        synthesize_refiner_model()
    else:
        refresh_refiner_model(refiner_model_name)
        refresh_base_model(base_model_name, vae_name, artistic_strength=artistic_strength)

    refresh_loras(loras, base_model_additional_loras=base_model_additional_loras)
    assert_model_integrity()

    final_unet = model_base.unet_with_lora
    final_clip = model_base.clip_with_lora
    final_vae = model_base.vae

    final_refiner_unet = model_refiner.unet_with_lora
    final_refiner_vae = model_refiner.vae

    if final_expansion is None:
        final_expansion = FooocusExpansion()

    prepare_text_encoder(async_call=True)
    clear_all_caches()
    return


refresh_everything(
    refiner_model_name=modules.config.default_refiner_model_name,
    base_model_name=modules.config.default_base_model_name,
    loras=get_enabled_loras(modules.config.default_loras),
    vae_name=modules.config.default_vae,
    artistic_strength=0.0,
)


@torch.no_grad()
@torch.inference_mode()
def vae_parse(latent):
    if final_refiner_vae is None:
        return latent

    result = vae_interpose.parse(latent["samples"])
    return {'samples': result}


@torch.no_grad()
@torch.inference_mode()
def calculate_sigmas_all(sampler, model, scheduler, steps):
    from ldm_patched.modules.samplers import calculate_sigmas_scheduler

    discard_penultimate_sigma = False
    if sampler in ['dpm_2', 'dpm_2_ancestral']:
        steps += 1
        discard_penultimate_sigma = True

    sigmas = calculate_sigmas_scheduler(model, scheduler, steps)

    if discard_penultimate_sigma:
        sigmas = torch.cat([sigmas[:-2], sigmas[-1:]])
    return sigmas


@torch.no_grad()
@torch.inference_mode()
def calculate_sigmas(sampler, model, scheduler, steps, denoise):
    if denoise is None or denoise > 0.9999:
        sigmas = calculate_sigmas_all(sampler, model, scheduler, steps)
    else:
        new_steps = int(steps / denoise)
        sigmas = calculate_sigmas_all(sampler, model, scheduler, new_steps)
        sigmas = sigmas[-(steps + 1):]
    return sigmas


@torch.no_grad()
@torch.inference_mode()
def get_candidate_vae(steps, switch, denoise=1.0, refiner_swap_method='joint'):
    assert refiner_swap_method in ['joint', 'separate', 'vae']

    if final_refiner_vae is not None and final_refiner_unet is not None:
        if denoise > 0.9:
            return final_vae, final_refiner_vae
        else:
            if denoise > (float(steps - switch) / float(steps)) ** 0.834:  # karras 0.834
                return final_vae, None
            else:
                return final_refiner_vae, None

    return final_vae, final_refiner_vae


@torch.no_grad()
@torch.inference_mode()
def process_diffusion(positive_cond, negative_cond, steps, switch, width, height, image_seed, callback, sampler_name, scheduler_name, latent=None, denoise=1.0, tiled=False, cfg_scale=7.0, refiner_swap_method='joint', disable_preview=False, original_prompt=None, original_negative_prompt=None, detail_daemon_enabled=False, detail_daemon_amount=0.25, detail_daemon_start=0.2, detail_daemon_end=0.8, detail_daemon_bias=0.71, detail_daemon_base_multiplier=0.85, detail_daemon_start_offset=0, detail_daemon_end_offset=-0.15, detail_daemon_exponent=1, detail_daemon_fade=0, detail_daemon_mode='both', detail_daemon_smooth=True, psychedelic_daemon_enabled=False, psychedelic_daemon_intensity=0.45, psychedelic_daemon_color_shift=0.3, psychedelic_daemon_fractal_depth=0.6, psychedelic_daemon_start=0.15, psychedelic_daemon_end=0.9, psychedelic_daemon_peak=0.5, psychedelic_daemon_bias=0.85, psychedelic_daemon_flow_multiplier=1.2, psychedelic_daemon_wave_frequency=2.5, psychedelic_daemon_saturation_boost=0.4, psychedelic_daemon_hue_rotation=0.2, psychedelic_daemon_contrast_waves=0.3, psychedelic_daemon_detail_recursion=3, psychedelic_daemon_chromatic_aberration=True, psychedelic_daemon_smooth=True, psychedelic_daemon_fade=0.15, psychedelic_daemon_mode='kaleidoscope', tpg_enabled=False, tpg_scale=3.0, tpg_applied_layers=None, tpg_shuffle_strength=1.0, tpg_adaptive_strength=True, drunk_enabled=False, drunk_attn_noise=0.0, drunk_layer_dropout=0.0, drunk_prompt_noise=0.0, drunk_cognitive_echo=0.0, drunk_dynamic_guidance=0.0, drunk_applied_layers=None, nag_enabled=False, nag_scale=1.5, nag_tau=5.0, nag_alpha=0.5, nag_negative_prompt="", nag_end=1.0, lfl_enabled=False, lfl_reference_image=None, lfl_aesthetic_strength=0.3, lfl_blend_mode='adaptive', force_grid_checkbox=False, async_task=None):
    print(f"[PROCESS_DIFFUSION ENTRY]")

    force_grid_unet_context = None # Initialize to None for cleanup
    imgs = [] # Initialize imgs to an empty list

    try:
        # Force Grid UNet Integration
        force_grid_unet_context = None
        if force_grid_checkbox:
            try:
                from extensions.force_grid_unet import ForceGridUNetContext
                # Calculate grid size based on image dimensions
                # For larger images, use bigger grids
                if width >= 1024 and height >= 1024:
                    grid_size = (3, 3)  # 3x3 grid for large images
                elif width >= 768 or height >= 768:
                    grid_size = (2, 2)  # 2x2 grid for medium images
                else:
                    grid_size = (2, 2)  # 2x2 grid for smaller images
                
                # Get the UNet model from the pipeline
                unet_model = final_unet
                
                force_grid_unet_context = ForceGridUNetContext(
                    unet_model=unet_model,
                    grid_size=grid_size,
                    blend_strength=0.15  # Moderate blend strength
                )
                force_grid_unet_context.__enter__()
                print(f"[Force Grid UNet] Enabled with {grid_size} grid for {width}x{height} image")
            except Exception as e:
                print(f"[Force Grid UNet] Error enabling: {e}")
                import traceback
                traceback.print_exc()
                force_grid_checkbox = False
        
        if steps == 0:
            # If steps is 0, no diffusion is performed. Return the initial latent or an empty list.
            if latent is not None:
                # Decode the latent if it's provided and return the image
                imgs = core.pytorch_to_numpy(core.decode_vae(final_vae, latent))
            return imgs
        
        if latent is None:
            initial_latent = core.generate_empty_latent(width=width, height=height, batch_size=1)
        else:
            initial_latent = latent
        
        # TPG Integration
        if tpg_enabled and tpg_scale > 0:
            try:
                from extras.TPG.tpg_integration import enable_tpg
                if tpg_applied_layers is None:
                    tpg_applied_layers = ['mid', 'up']
                
                print(f"[TPG] Enabling TPG with scale={tpg_scale}, layers={tpg_applied_layers}")
                enable_tpg(
                    scale=tpg_scale,
                    applied_layers=tpg_applied_layers,
                    shuffle_strength=tpg_shuffle_strength,
                    adaptive_strength=tpg_adaptive_strength
                )
            except Exception as e:
                print(f"[TPG] Error enabling TPG: {e}")
                import traceback
                traceback.print_exc()
        
        # NAG Integration
        if nag_enabled and nag_scale > 1.0:
            try:
                from extras.nag.nag_integration import enable_nag, NAG_AVAILABLE, NAG_STANDALONE
                
                if not NAG_AVAILABLE:
                    print(f"[NAG] NAG disabled due to dependency version conflict.")
                    print(f"[NAG] This is usually caused by incompatible transformers/peft versions.")
                    print(f"[NAG] Suggested fix: pip install transformers>=4.44.0 peft>=0.12.0")
                    print(f"[NAG] Continuing without NAG...")
                else:
                    mode_str = "standalone mode" if NAG_STANDALONE else "full mode"
                    print(f"[NAG] Enabling NAG in {mode_str} with scale={nag_scale}, tau={nag_tau}, alpha={nag_alpha}")
                    print(f"[NAG] NAG negative prompt: '{nag_negative_prompt}'")
                    
                    enable_nag(
                        scale=nag_scale,
                        tau=nag_tau,
                        alpha=nag_alpha,
                        negative_prompt=nag_negative_prompt,
                        end=nag_end
                    )
            except ImportError as e:
                if any(cache in str(e) for cache in ["EncoderDecoderCache", "HybridCache"]) or "transformers" in str(e):
                    print(f"[NAG] NAG disabled due to dependency version conflict.")
                    print(f"[NAG] Suggested fix: pip install transformers>=4.44.0 peft>=0.12.0")
                else:
                    print(f"[NAG] Import error: {e}")
            except RuntimeError as e:
                print(f"[NAG] {e}")
            except Exception as e:
                print(f"[NAG] Error enabling NAG: {e}")
                import traceback
                traceback.print_exc()
        
        # DRUNKUNet Integration
        if drunk_enabled:
            try:
                print(f"[DRUNKUNet] Enabling DRUNKUNet with parameters:")
                print(f"[DRUNKUNet] - Attention Noise: {drunk_attn_noise}")
                print(f"[DRUNKUNet] - Layer Dropout: {drunk_layer_dropout}")
                print(f"[DRUNKUNet] - Prompt Noise: {drunk_prompt_noise}")
                print(f"[DRUNKUNet] - Cognitive Echo: {drunk_cognitive_echo}")
                print(f"[DRUNKUNet] - Dynamic Guidance: {drunk_dynamic_guidance}")
                print(f"[DRUNKUNet] - Applied Layers: {drunk_applied_layers}")
                
                # Import and configure DRUNKUNet sampler
                from extras.drunkunet.drunkieunet_pipelinesdxl import drunkunet_sampler
                
                # Configure the sampler with current parameters
                drunkunet_sampler.attn_noise_strength = drunk_attn_noise
                drunkunet_sampler.layer_dropout_prob = drunk_layer_dropout
                drunkunet_sampler.prompt_noise_strength = drunk_prompt_noise
                drunkunet_sampler.cognitive_echo_strength = drunk_cognitive_echo
                drunkunet_sampler.drunk_applied_layers = drunk_applied_layers or ['mid', 'up']
                
                # Configure dynamic guidance if enabled
                if drunk_dynamic_guidance:
                    drunkunet_sampler.dynamic_guidance_params = {
                        'base': cfg_scale,
                        'amplitude': 2.0,
                        'frequency': 0.1,
                        'phase': 0
                    }
                else:
                    drunkunet_sampler.dynamic_guidance_params = {}
                
                # Activate DRUNKUNet with the UNet model
                drunkunet_sampler.activate(final_unet)
                print(f"[DRUNKUNet] Successfully activated DRUNKUNet sampler")
                
            except Exception as e:
                print(f"[DRUNKUNet] Error enabling DRUNKUNet: {e}")
                import traceback
                traceback.print_exc()
        
        # Disco Diffusion Integration
        # Disco Diffusion will be applied as post-processing after image generation
        
        target_unet, target_vae, target_refiner_unet, target_refiner_vae, target_clip         = final_unet, final_vae, final_refiner_unet, final_refiner_vae, final_clip


        assert refiner_swap_method in ['joint', 'separate', 'vae']

        if final_refiner_vae is not None and final_refiner_unet is not None:
            # Refiner Use Different VAE (then it is SD15)
            if denoise > 0.9:
                refiner_swap_method = 'vae'
            else:
                refiner_swap_method = 'joint'
                if denoise > (float(steps - switch) / float(steps)) ** 0.834:  # karras 0.834
                    target_unet, target_vae, target_refiner_unet, target_refiner_vae \
                        = final_unet, final_vae, None, None
                    print(f'[Sampler] only use Base because of partial denoise.')
                else:
                    positive_cond = clip_separate(positive_cond, target_model=final_refiner_unet.model, target_clip=final_clip)
                    negative_cond = clip_separate(negative_cond, target_model=final_refiner_unet.model, target_clip=final_clip)
                    target_unet, target_vae, target_refiner_unet, target_refiner_vae \
                        = final_refiner_unet, final_refiner_vae, None, None
                    print(f'[Sampler] only use Refiner because of partial denoise.')

        print(f'[Sampler] refiner_swap_method = {refiner_swap_method}')

        if latent is None:
            initial_latent = core.generate_empty_latent(width=width, height=height, batch_size=1)
        else:
            initial_latent = latent

        # Disco will be applied as post-processing after image generation

        minmax_sigmas = calculate_sigmas(sampler=sampler_name, scheduler=scheduler_name, model=final_unet.model, steps=steps, denoise=denoise)
        
        # Apply Detail Daemon sigma manipulation if enabled
        if detail_daemon_enabled:
            print(f'[Detail Daemon] Manipulating sigmas with amount {detail_daemon_amount}')
            
            # Create a schedule mask based on start/end parameters
            num_steps = len(minmax_sigmas)
            step_indices = torch.arange(num_steps, dtype=torch.float32)
            normalized_steps = step_indices / (num_steps - 1) if num_steps > 1 else torch.zeros_like(step_indices)
            
            # Apply start/end range
            mask = torch.ones_like(normalized_steps)
            if detail_daemon_start > 0 or detail_daemon_end < 1:
                mask = torch.where((normalized_steps >= detail_daemon_start) & (normalized_steps <= detail_daemon_end), 1.0, 0.0)
            
            # Apply offsets (simplified for now)
            if detail_daemon_start_offset != 0:
                shift_amount = int(num_steps * detail_daemon_start_offset)
                if shift_amount != 0:
                    mask = torch.roll(mask, shift_amount)
            
            # Apply exponent curve
            if detail_daemon_exponent != 1:
                mask = torch.pow(mask, detail_daemon_exponent)
            
            # Apply fade (gaussian blur approximation)
            if detail_daemon_fade > 0:
                # Simple smoothing - in practice you'd want proper gaussian blur
                kernel_size = max(3, int(detail_daemon_fade * 10))
                if kernel_size % 2 == 0:
                    kernel_size += 1
                # Simple moving average as approximation
                padding = kernel_size // 2
                mask_padded = torch.nn.functional.pad(mask.unsqueeze(0).unsqueeze(0), (padding, padding), mode='reflect')
                mask = torch.nn.functional.avg_pool1d(mask_padded, kernel_size, stride=1, padding=0).squeeze()
            
            # Calculate detail multiplier based on amount
            # Lower multiplier = less denoising = more detail
            # Use configurable base multiplier for detail enhancement
            base_multiplier = detail_daemon_base_multiplier
            detail_multiplier = torch.lerp(torch.ones_like(mask), torch.full_like(mask, base_multiplier), detail_daemon_amount * mask)
            detail_multiplier = detail_multiplier * detail_daemon_bias + detail_multiplier * (1 - detail_daemon_bias) * 0.5
            
            # Apply the multiplier to sigmas
            original_sigmas = minmax_sigmas.clone()
            minmax_sigmas = minmax_sigmas * detail_multiplier.to(minmax_sigmas.device)
            
            print(f'[Detail Daemon] Applied sigma multiplier range: {detail_multiplier.min():.3f} - {detail_multiplier.max():.3f}')
        
        # Apply Psychedelic Daemon sigma manipulation if enabled
        if psychedelic_daemon_enabled:
            print(f'[Psychedelic Daemon] Applying psychedelic effects with intensity {psychedelic_daemon_intensity}')
            
            # Create a more complex schedule mask for psychedelic effects
            num_steps = len(minmax_sigmas)
            step_indices = torch.arange(num_steps, dtype=torch.float32)
            normalized_steps = step_indices / (num_steps - 1) if num_steps > 1 else torch.zeros_like(step_indices)
            
            # Create psychedelic mask based on mode
            if psychedelic_daemon_mode == 'kaleidoscope':
                # Radial-like pattern for kaleidoscope effect
                center = 0.5
                radial_distance = torch.abs(normalized_steps - center)
                mask = torch.exp(-((radial_distance - (psychedelic_daemon_peak - center)) ** 2) / (0.2 ** 2))
                mask = mask * torch.sin(normalized_steps * psychedelic_daemon_wave_frequency * 3.14159) * 0.8 + 0.6  # Stronger waves
                
            elif psychedelic_daemon_mode == 'fluid':
                # Flowing wave pattern
                wave1 = torch.sin(normalized_steps * psychedelic_daemon_wave_frequency * 3.14159)
                wave2 = torch.cos(normalized_steps * psychedelic_daemon_wave_frequency * 2 * 3.14159)
                mask = (wave1 * wave2 + 1) / 2 * psychedelic_daemon_flow_multiplier
                
            elif psychedelic_daemon_mode == 'fractal':
                # Fractal-like recursive pattern
                mask = torch.zeros_like(normalized_steps)
                for i in range(int(psychedelic_daemon_fractal_depth * 5)):
                    freq = 2 ** (i * 0.5)
                    fractal_wave = torch.sin(normalized_steps * freq * 3.14159)
                    mask += fractal_wave * (0.5 ** i)
                mask = (mask + 1) / 2
                
            else:  # 'both' or default
                # Combination of effects
                radial_distance = torch.abs(normalized_steps - 0.5)
                radial_mask = torch.exp(-((radial_distance - (psychedelic_daemon_peak - 0.5)) ** 2) / (0.15 ** 2))
                wave_mask = torch.sin(normalized_steps * psychedelic_daemon_wave_frequency * 3.14159) * 0.6 + 0.8  # Stronger combined waves
                mask = radial_mask * wave_mask
            
            # Apply start/end range
            if psychedelic_daemon_start > 0 or psychedelic_daemon_end < 1:
                range_mask = torch.where((normalized_steps >= psychedelic_daemon_start) & 
                                       (normalized_steps <= psychedelic_daemon_end), 1.0, 0.0)
                mask = mask * range_mask
            
            # Apply peak enhancement - MUCH stronger
            peak_enhancement = torch.exp(-((normalized_steps - psychedelic_daemon_peak) ** 2) / (0.15 ** 2))
            mask = torch.maximum(mask, peak_enhancement * 0.8)  # Much stronger peak enhancement
            
            # Apply fade (gaussian-like smoothing)
            if psychedelic_daemon_fade > 0:
                kernel_size = max(3, int(psychedelic_daemon_fade * 20))
                if kernel_size % 2 == 0:
                    kernel_size += 1
                padding = kernel_size // 2
                mask_padded = torch.nn.functional.pad(mask.unsqueeze(0).unsqueeze(0), (padding, padding), mode='reflect')
                mask = torch.nn.functional.avg_pool1d(mask_padded, kernel_size, stride=1, padding=0).squeeze()
            
            # Apply smoothing if enabled
            if psychedelic_daemon_smooth:
                mask_padded = torch.nn.functional.pad(mask.unsqueeze(0).unsqueeze(0), (2, 2), mode='reflect')
                mask = torch.nn.functional.avg_pool1d(mask_padded, 5, stride=1, padding=0).squeeze()
            
            # Calculate psychedelic multiplier with MUCH more aggressive effects
            # Use intensity to control the strength of sigma manipulation
            base_multiplier = 0.4  # Much more aggressive base for strong psychedelic effects
            psychedelic_multiplier = torch.lerp(torch.ones_like(mask), 
                                              torch.full_like(mask, base_multiplier), 
                                              psychedelic_daemon_intensity * mask * 2.0)  # Double the intensity
            
            # Apply bias for more dramatic effects with stronger modulation
            psychedelic_multiplier = psychedelic_multiplier * psychedelic_daemon_bias + \
                                   psychedelic_multiplier * (1 - psychedelic_daemon_bias) * 0.1  # More extreme bias
            
            # Apply the multiplier to sigmas
            minmax_sigmas = minmax_sigmas * psychedelic_multiplier.to(minmax_sigmas.device)
            
            print(f'[Psychedelic Daemon] Applied sigma multiplier range: {psychedelic_multiplier.min():.3f} - {psychedelic_multiplier.max():.3f}')
        
        positive_sigmas = minmax_sigmas[minmax_sigmas > 0]
        if positive_sigmas.numel() == 0:
            raise ValueError("No positive sigma values found. This indicates an issue with the sampler or model configuration.")
        sigma_min, sigma_max = positive_sigmas.min(), minmax_sigmas.max()
        sigma_min = float(sigma_min.cpu().numpy())
        sigma_max = float(sigma_max.cpu().numpy())
        print(f'[Sampler] sigma_min = {sigma_min}, sigma_max = {sigma_max}')

        modules.patch.BrownianTreeNoiseSamplerPatched.global_init(
            initial_latent['samples'].to(ldm_patched.modules.model_management.get_torch_device()),
            sigma_min, sigma_max, seed=image_seed, cpu=False)

        # Aggressive Aesthetic Replication (LFL) Integration with UNet Hooks
        aesthetic_replicator = None
        lfl_hooked = False
        if lfl_enabled:
            try:
                from modules.neural_echo_sampler import (
                    setup_aesthetic_replication_for_task,
                    hook_unet_for_aesthetic_replication,
                    unhook_unet_aesthetic_replication,
                    set_aesthetic_timestep
                )
                
                # Create a mock task object with the parameters
                class MockTask:
                    def __init__(self):
                        self.lfl_enabled = lfl_enabled
                        self.lfl_reference_image = lfl_reference_image
                        self.lfl_aesthetic_strength = lfl_aesthetic_strength
                        self.lfl_blend_mode = lfl_blend_mode
                
                mock_task = MockTask()
                aesthetic_replicator = setup_aesthetic_replication_for_task(mock_task, final_vae)
                
                if aesthetic_replicator:
                    print(f"[LFL] AGGRESSIVE Aesthetic Replication initialized: strength={lfl_aesthetic_strength}, mode={lfl_blend_mode}")
                    
                    # Hook the UNet for aggressive aesthetic replication
                    lfl_hooked = hook_unet_for_aesthetic_replication(final_unet.model, initial_latent['samples'])
                    if lfl_hooked:
                        print("[LFL] UNet successfully hooked for aggressive aesthetic replication")
                    else:
                        print("[LFL] Warning: Failed to hook UNet, falling back to callback method")
                else:
                    print("[LFL] Failed to initialize Aesthetic Replication")
                    
            except Exception as e:
                print(f"[LFL] Error initializing Aesthetic Replication: {e}")
                import traceback
                traceback.print_exc()
                aesthetic_replicator = None

        # Note: Detail daemon callback approach is too expensive (VAE decode on every step)
        # We'll apply detail daemon as post-processing instead

        decoded_latent = None

        # Create enhanced callback with Neural Echo, LFL, and Psychedelic effects
        enhanced_callback = callback
        
        def create_enhanced_callback(original_callback, aesthetic_replicator, lfl_hooked, psychedelic_enabled, psychedelic_params):
            def enhanced_callback(step, x0, x, total_steps, preview_image=None):
                # Update timestep for adaptive aesthetic blending
                if lfl_hooked:
                    try:
                        timestep = int(1000 * (1.0 - step / total_steps))
                        set_aesthetic_timestep(timestep)
                    except Exception as e:
                        pass
                
                # Apply aesthetic replication if enabled
                enhanced_x0 = x0
                if aesthetic_replicator is not None and not lfl_hooked:
                    try:
                        enhanced_x0 = aesthetic_replicator(x, x0)
                    except Exception as e:
                        print(f"[LFL] Error in aesthetic replication callback: {e}")
                        enhanced_x0 = x0
                
                # Apply psychedelic effects during diffusion if enabled
                if psychedelic_enabled and enhanced_x0 is not None:
                    try:
                        # Calculate current progress (0 to 1)
                        progress = step / total_steps if total_steps > 0 else 0
                        
                        # Check if we're in the active range
                        if psychedelic_params['start'] <= progress <= psychedelic_params['end']:
                            # Apply psychedelic latent manipulation
                            enhanced_x0 = apply_psychedelic_latent_effects(
                                enhanced_x0, progress, psychedelic_params
                            )
                    except Exception as e:
                        print(f"[Psychedelic Daemon] Error in callback: {e}")
                
                # Call original callback
                if original_callback is not None:
                    original_callback(step, enhanced_x0, x, total_steps, preview_image)
            
            return enhanced_callback
        
        # Prepare psychedelic parameters for callback
        psychedelic_params = {
            'intensity': psychedelic_daemon_intensity,
            'start': psychedelic_daemon_start,
            'end': psychedelic_daemon_end,
            'peak': psychedelic_daemon_peak,
            'mode': psychedelic_daemon_mode,
            'wave_frequency': psychedelic_daemon_wave_frequency,
            'flow_multiplier': psychedelic_daemon_flow_multiplier,
            'bias': psychedelic_daemon_bias
        } if psychedelic_daemon_enabled else {}
        
        # Create the enhanced callback
        enhanced_callback = create_enhanced_callback(
            callback, aesthetic_replicator, lfl_hooked, 
            psychedelic_daemon_enabled, psychedelic_params
        )

        # Use sampler (with or without guidance)
        ksampler_imgs = core.ksampler(
            model=final_unet,
            positive=positive_cond,
            negative=negative_cond,
            latent=initial_latent,
            seed=image_seed,
            steps=steps,
            cfg=cfg_scale,
            sampler_name=sampler_name,
            scheduler=scheduler_name,
            denoise=denoise,
            disable_preview=disable_preview,
            refiner=final_refiner_unet,
            refiner_switch=switch,
            sigmas=minmax_sigmas,
            callback_function=enhanced_callback
        )['samples']
        
        # Convert latents to images
        if ksampler_imgs is not None:
            latent_dict = {'samples': ksampler_imgs}
            imgs = core.decode_vae(target_vae, latent_dict)
            imgs = core.pytorch_to_numpy(imgs)
        else:
            imgs = []
        
        # Psychedelic Daemon is applied during diffusion via sigma manipulation and callback
        # No post-processing needed as it's integrated into the sampling process
        
        # Disco post-processing removed
        
        # TPG Cleanup
        if tpg_enabled and tpg_scale > 0:
            try:
                from extras.TPG.tpg_integration import disable_tpg
                print("[TPG] Disabling TPG after generation")
                disable_tpg()
            except Exception as e:
                print(f"[TPG] Error disabling TPG: {e}")
        
        # NAG Cleanup
        if nag_enabled and nag_scale > 1.0:
            try:
                from extras.nag.nag_integration import disable_nag
                print("[NAG] Disabling NAG after generation")
                disable_nag()
            except Exception as e:
                print(f"[NAG] Error disabling NAG: {e}")
        
        # Disco Diffusion removed - no cleanup needed
        
        return imgs
    finally:
        # Cleanup DRUNKUNet if it was enabled
        if drunk_enabled:
            try:
                from extras.drunkunet.drunkieunet_pipelinesdxl import drunkunet_sampler
                drunkunet_sampler.deactivate()
                print("[DRUNKUNet] Deactivated after generation.")
            except Exception as e:
                print(f"[DRUNKUNet] Error deactivating DRUNKUNet: {e}")
                import traceback
                traceback.print_exc()
        
        if force_grid_checkbox and force_grid_unet_context is not None:
            try:
                force_grid_unet_context.__exit__(None, None, None)
                print("[Force Grid UNet] Disabled after generation.")
            except Exception as e:
                print(f"[Force Grid] Error disabling Force Grid: {e}")
        
        # Cleanup LFL UNet hooks
        if lfl_hooked:
            try:
                from modules.neural_echo_sampler import unhook_unet_aesthetic_replication
                unhook_unet_aesthetic_replication()
                print("[LFL] UNet hooks cleaned up after generation.")
            except Exception as e:
                print(f"[LFL] Error cleaning up UNet hooks: {e}")
                import traceback
                traceback.print_exc()
    

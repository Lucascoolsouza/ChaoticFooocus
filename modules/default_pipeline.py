import modules.core as core
import os
import torch
import numpy as np # Added import for numpy
from PIL import Image # Added import for PIL Image
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
def process_diffusion(positive_cond, negative_cond, steps, switch, width, height, image_seed, callback, sampler_name, scheduler_name, latent=None, denoise=1.0, tiled=False, cfg_scale=7.0, refiner_swap_method='joint', disable_preview=False, original_prompt=None, original_negative_prompt=None, detail_daemon_enabled=False, detail_daemon_amount=0.25, detail_daemon_start=0.2, detail_daemon_end=0.8, detail_daemon_bias=0.71, detail_daemon_base_multiplier=0.85, detail_daemon_start_offset=0, detail_daemon_end_offset=-0.15, detail_daemon_exponent=1, detail_daemon_fade=0, detail_daemon_mode='both', detail_daemon_smooth=True, tpg_enabled=False, tpg_scale=3.0, tpg_applied_layers=None, tpg_shuffle_strength=1.0, tpg_adaptive_strength=True, drunk_enabled=False, drunk_attn_noise=0.0, drunk_layer_dropout=0.0, drunk_prompt_noise=0.0, drunk_cognitive_echo=0.0, drunk_dynamic_guidance=0.0, drunk_applied_layers=None, nag_enabled=False, nag_scale=1.5, nag_tau=5.0, nag_alpha=0.5, nag_negative_prompt="", nag_end=1.0, disco_enabled=False, disco_scale=0.5, disco_preset='custom', disco_transforms=None, disco_seed=None, disco_animation_mode='none', disco_zoom_factor=1.02, disco_rotation_speed=0.1, disco_translation_x=0.0, disco_translation_y=0.0, disco_color_coherence=0.5, disco_saturation_boost=1.2, disco_contrast_boost=1.1, disco_symmetry_mode='none', disco_fractal_octaves=3, disco_clip_model='RN50', disco_guidance_steps=100, disco_cutn=16, disco_tv_scale=150.0, disco_range_scale=50.0, lfl_enabled=False, lfl_echo_strength=0.05, lfl_decay_factor=0.9, lfl_max_memory=20, force_grid_checkbox=False, async_task=None):
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
                from extras.nag.nag_integration import enable_nag, NAG_AVAILABLE
                
                if not NAG_AVAILABLE:
                    print(f"[NAG] NAG disabled due to dependency version conflict.")
                    print(f"[NAG] This is usually caused by incompatible transformers/peft versions.")
                    print(f"[NAG] Suggested fix: pip install transformers>=4.44.0 peft>=0.12.0")
                    print(f"[NAG] Continuing without NAG...")
                else:
                    print(f"[NAG] Enabling NAG with scale={nag_scale}, tau={nag_tau}, alpha={nag_alpha}")
                    print(f"[NAG] NAG negative prompt: '{nag_negative_prompt}'")
                    
                    enable_nag(
                        scale=nag_scale,
                        tau=nag_tau,
                        alpha=nag_alpha,
                        negative_prompt=nag_negative_prompt,
                        end=nag_end
                    )
            except ImportError as e:
                if "EncoderDecoderCache" in str(e) or "transformers" in str(e):
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

        # Disco Diffusion: Inject distortion into initial latent (mid-generation approach)
        if disco_enabled and disco_scale > 0:
            try:
                from extras.disco_diffusion.pipeline_disco import inject_disco_distortion
                print(f"[Disco] Injecting distortion into initial latent with scale={disco_scale}")
                
                # Apply distortion to the initial latent
                initial_latent['samples'] = inject_disco_distortion(
                    initial_latent['samples'],
                    disco_scale=disco_scale,
                    distortion_type=disco_preset if disco_preset != 'custom' else 'psychedelic'
                )
                print("[Disco] Distortion injection completed")
                
            except Exception as e:
                print(f"[Disco] Error injecting distortion: {e}")

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

        # Neural Echo Sampler (LFL) Integration
        neural_echo_sampler = None
        if lfl_enabled:
            try:
                from modules.neural_echo_sampler import initialize_neural_echo
                neural_echo_sampler = initialize_neural_echo(
                    echo_strength=lfl_echo_strength,
                    decay_factor=lfl_decay_factor,
                    max_memory=int(lfl_max_memory)
                )
                # Reset the sampler for this generation
                neural_echo_sampler.reset()
                print(f"[LFL] Neural Echo Sampler initialized: strength={lfl_echo_strength}, decay={lfl_decay_factor}, memory={int(lfl_max_memory)}")
            except Exception as e:
                print(f"[LFL] Error initializing Neural Echo Sampler: {e}")
                neural_echo_sampler = None

        # Note: Detail daemon callback approach is too expensive (VAE decode on every step)
        # We'll apply detail daemon as post-processing instead

        decoded_latent = None

        # Create Neural Echo enhanced callback if LFL is enabled
        enhanced_callback = callback
        if neural_echo_sampler is not None:
            def neural_echo_callback(step, x0, x, total_steps, preview_image=None):
                # Apply neural echo to the denoised latent (x0)
                try:
                    enhanced_x0 = neural_echo_sampler(x, x0)
                    # Call the original callback with the enhanced latent
                    if callback is not None:
                        callback(step, enhanced_x0, x, total_steps, preview_image)
                except Exception as e:
                    print(f"[LFL] Error in neural echo callback: {e}")
                    # Fallback to original callback
                    if callback is not None:
                        callback(step, x0, x, total_steps, preview_image)
            enhanced_callback = neural_echo_callback

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
        
        # Disco Diffusion: Distortion was already applied during generation
        if disco_enabled and disco_scale > 0:
            print(f"[Disco] Distortion effects applied during generation with scale={disco_scale}")
        
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
        
        # Cleanup Disco Diffusion
        if disco_enabled:
            try:
                from extras.disco_diffusion.disco_integration import disco_integration
                print("[Disco] Disabling Disco Diffusion after generation")
                disco_integration.deactivate_after_generation()
            except Exception as e:
                print(f"[Disco] Error disabling Disco Diffusion: {e}")
        
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
                import traceback
                traceback.print_exc()
    

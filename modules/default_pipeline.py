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
<<<<<<< HEAD
# Guidance samplers are now integrated into k_diffusion sampling
# No need for separate sampler classes
=======
from extras.nag.pipeline_sdxl_nag import NAGStableDiffusionXLPipeline, safe_decode
from extras.TPG.pipeline_sdxl_tpg import StableDiffusionXLTPGPipeline
from extras.PAG.pipeline_sdxl_pag import StableDiffusionXLPAGPipeline
>>>>>>> parent of a9a7293 (SAMPLERS)


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
def refresh_base_model(name, vae_name=None):
    global model_base

    filename = get_file_from_folder_list(name, modules.config.paths_checkpoints)

    vae_filename = None
    if vae_name is not None and vae_name != modules.flags.default_vae:
        vae_filename = get_file_from_folder_list(vae_name, modules.config.path_vae)

    if model_base.filename == filename and model_base.vae_filename == vae_filename:
        return

    model_base = core.load_model(filename, vae_filename)
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
                       base_model_additional_loras=None, use_synthetic_refiner=False, vae_name=None):
    global final_unet, final_clip, final_vae, final_refiner_unet, final_refiner_vae, final_expansion

    final_unet = None
    final_clip = None
    final_vae = None
    final_refiner_unet = None
    final_refiner_vae = None

    if use_synthetic_refiner and refiner_model_name == 'None':
        print('Synthetic Refiner Activated')
        refresh_base_model(base_model_name, vae_name)
        synthesize_refiner_model()
    else:
        refresh_refiner_model(refiner_model_name)
        refresh_base_model(base_model_name, vae_name)

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
def process_diffusion(positive_cond, negative_cond, steps, switch, width, height, image_seed, callback, sampler_name, scheduler_name, latent=None, denoise=1.0, tiled=False, cfg_scale=7.0, refiner_swap_method='joint', disable_preview=False, nag_scale=1.0, nag_tau=2.5, nag_alpha=0.5, nag_negative_prompt=None, nag_end=1.0, original_prompt=None, original_negative_prompt=None, detail_daemon_enabled=False, detail_daemon_amount=0.25, detail_daemon_start=0.2, detail_daemon_end=0.8, detail_daemon_bias=0.71, detail_daemon_base_multiplier=0.85, detail_daemon_start_offset=0, detail_daemon_end_offset=-0.15, detail_daemon_exponent=1, detail_daemon_fade=0, detail_daemon_mode='both', detail_daemon_smooth=True, tpg_enabled=False, tpg_scale=3.0, tpg_applied_layers_index=None, dag_enabled=False, dag_scale=0.0, dag_applied_layers=None):
    print(f"[PROCESS_DIFFUSION ENTRY] dag_enabled: {dag_enabled}, tpg_enabled: {tpg_enabled}, nag_scale: {nag_scale}")
    imgs = [] # Initialize imgs to an empty list
    if steps == 0:
        # If steps is 0, no diffusion is performed. Return the initial latent or an empty list.
        if latent is not None:
            # Decode the latent if it's provided and return the image
            imgs = core.pytorch_to_numpy(core.decode_vae(final_vae, latent))
        return imgs
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

    # Note: Detail daemon callback approach is too expensive (VAE decode on every step)
    # We'll apply detail daemon as post-processing instead

    decoded_latent = None

    if nag_scale > 1.0 or tpg_enabled:
        if tpg_enabled:
            print(f"[TPG] TPG is active with tpg_scale={tpg_scale}, tpg_applied_layers_index={tpg_applied_layers_index}")
            if isinstance(tpg_applied_layers_index, str):
                tpg_applied_layers_index = [s.strip() for s in tpg_applied_layers_index.split(',') if s.strip()]
            else:
                tpg_applied_layers_index = [] # Ensure it's a list if not a string

        if nag_scale > 1.0:
            print(f"[NAG] NAG is active with nag_scale={nag_scale}, nag_tau={nag_tau}, nag_alpha={nag_alpha}, nag_negative_prompt='{nag_negative_prompt}', nag_end={nag_end}")
            # Use nag_negative_prompt if provided, otherwise use the extracted negative_prompt_str
            if nag_negative_prompt is not None and nag_negative_prompt.strip() != "":
                final_nag_negative_prompt = nag_negative_prompt.strip()
                print(f"[NAG] Using provided NAG negative prompt: '{final_nag_negative_prompt}'")
            elif original_negative_prompt is not None and original_negative_prompt.strip() != "":
                final_nag_negative_prompt = original_negative_prompt.strip()
                print(f"[NAG] Using original negative prompt for NAG: '{final_nag_negative_prompt}'")
            else:
                # If no negative prompt is available, use a default strong negative prompt for NAG
                final_nag_negative_prompt = "blurry, low quality, distorted, deformed, ugly, bad anatomy, worst quality"
                print(f"[NAG] Using default NAG negative prompt: '{final_nag_negative_prompt}'")
        else:
            final_nag_negative_prompt = None # Ensure it's None if NAG is not active

        # Dynamically add a 'config' attribute to model_base.vae if it doesn't exist
        if not hasattr(model_base.vae, 'config'):
            class VAEConfig:
                def __init__(self):
                    self.block_out_channels = [128, 256, 512, 512]
            model_base.vae.config = VAEConfig()

        # Dynamically add a 'config' attribute to model_base.unet if it doesn't exist
        if not hasattr(model_base.unet, 'config'):
            class UNetConfig:
                def __init__(self):
                    self.sample_size = 128
            model_base.unet.config = UNetConfig()

        # Extract embeddings from positive_cond and negative_cond
        prompt_embeds_unpadded = positive_cond[0][0]
        pooled_prompt_embeds = positive_cond[0][1]["pooled_output"]

        negative_prompt_embeds_unpadded = negative_cond[0][0]
        negative_pooled_prompt_embeds = negative_cond[0][1]["pooled_output"]

        # Determine max sequence length and pad
        max_length = max(prompt_embeds_unpadded.shape[1], negative_prompt_embeds_unpadded.shape[1])

        prompt_embeds = torch.nn.functional.pad(prompt_embeds_unpadded, (0, 0, 0, max_length - prompt_embeds_unpadded.shape[1]))
        negative_prompt_embeds = torch.nn.functional.pad(negative_prompt_embeds_unpadded, (0, 0, 0, max_length - negative_prompt_embeds_unpadded.shape[1]))

        # Ensure all necessary components are on the correct device before passing to NAGStableDiffusionXLPipeline
        device = ldm_patched.modules.model_management.get_torch_device()

        # Dynamically add a 'config' attribute to model_base.vae if it doesn't exist
        if not hasattr(model_base.vae, 'config'):
            class VAEConfig:
                def __init__(self):
                    self.block_out_channels = [128, 256, 512, 512]
            model_base.vae.config = VAEConfig()

        # Dynamically add a 'config' attribute to model_base.unet if it doesn't exist
        if not hasattr(model_base.unet, 'config'):
            class UNetConfig:
                def __init__(self):
                    self.sample_size = 128
            model_base.unet.config = UNetConfig()

        # Move components to the correct device
        vae_on_device = model_base.vae  # VAE uses its own device management via patcher
        text_encoder_l_on_device = model_base.clip_with_lora.cond_stage_model.clip_l.to(device)
        text_encoder_g_on_device = model_base.clip_with_lora.cond_stage_model.clip_g.to(device)
        tokenizer_l_on_device = model_base.clip_with_lora.tokenizer.clip_l
        tokenizer_g_on_device = model_base.clip_with_lora.tokenizer.clip_g
        unet_on_device = model_base.unet_with_lora  # UNet with LoRAs applied - uses its own device management via patcher
        # Create a compatible scheduler wrapper for diffusers
        class SchedulerWrapper:
            def __init__(self, original_sampling):
                self.original_sampling = original_sampling
                self.config = type('Config', (), {
                    'num_train_timesteps': 1000,
                    'beta_start': 0.00085,
                    'beta_end': 0.012,
                    'beta_schedule': 'scaled_linear',
                    'prediction_type': 'epsilon',
                    'sample_max_value': 1.0,
                    'timestep_spacing': 'leading',
                    'steps_offset': 1
                })()
                self.timesteps = None
                self.init_noise_sigma = 1.0  # Standard initial noise sigma
                self.order = 1  # Scheduler order
                self.num_inference_steps = None
                
            def set_timesteps(self, num_inference_steps, device=None):
                # Create a simple timestep schedule
                import torch
                self.num_inference_steps = num_inference_steps
                self.timesteps = torch.linspace(999, 0, num_inference_steps, dtype=torch.long)
                if device is not None:
                    self.timesteps = self.timesteps.to(device)
            
            def step(self, model_output, timestep, sample, **kwargs):
                # Simple step function - just return the sample for now
                # The actual sampling will be handled by the custom ksampler
                return type('SchedulerOutput', (), {'prev_sample': sample})()
            
            def scale_model_input(self, sample, timestep):
                # Standard scaling - just return the sample as-is
                return sample
            
            def add_noise(self, original_samples, noise, timesteps):
                # Standard noise addition
                import torch
                return original_samples + noise
            
            @property
            def sigmas(self):
                # Return a simple sigma schedule
                import torch
                if self.timesteps is not None:
                    return torch.ones_like(self.timesteps, dtype=torch.float32)
                return torch.tensor([1.0])
            
            def __len__(self):
                return self.num_inference_steps or 50
        
        scheduler_on_device = SchedulerWrapper(model_base.unet_with_lora.model.model_sampling)

        # Add dtype property to text encoders if they don't have it
        if not hasattr(text_encoder_l_on_device, 'dtype'):
            text_encoder_l_on_device.dtype = torch.float16 if ldm_patched.modules.model_management.should_use_fp16() else torch.float32
        if not hasattr(text_encoder_g_on_device, 'dtype'):
            text_encoder_g_on_device.dtype = torch.float16 if ldm_patched.modules.model_management.should_use_fp16() else torch.float32

        # Wrap text encoders to handle diffusers-specific parameters
        def wrap_text_encoder(encoder):
            class TextEncoderWrapper:
                def __init__(self, original_encoder):
                    self.original_encoder = original_encoder
                    # Copy all attributes from the original encoder
                    for attr_name in dir(original_encoder):
                        if not attr_name.startswith('_'):
                            setattr(self, attr_name, getattr(original_encoder, attr_name))
                    
                    # Add a mock config attribute with projection_dim
                    class Config:
                        def __init__(self, projection_dim):
                            self.projection_dim = projection_dim
                    
                    # Attempt to get projection_dim from original_encoder.config, otherwise use a default
                    proj_dim = getattr(getattr(original_encoder, 'config', None), 'projection_dim', 1280) # Default for SDXL CLIP G
                    self.config = Config(proj_dim)

                def __call__(self, input_ids, output_hidden_states=False, **kwargs):
                    # Call the original encoder, ignoring unsupported parameters
                    # Make sure input_ids is properly formatted
                    if hasattr(input_ids, 'to'):
                        # If it's a tensor, ensure it's on CPU for the custom encoder
                        input_ids = input_ids.cpu()
                    return self.original_encoder(input_ids)
                
                def forward(self, input_ids, output_hidden_states=False, **kwargs):
                    # Call the original encoder, ignoring unsupported parameters
                    # Make sure input_ids is properly formatted
                    if hasattr(input_ids, 'to'):
                        # If it's a tensor, ensure it's on CPU for the custom encoder
                        input_ids = input_ids.cpu()
                    return self.original_encoder(input_ids)
            
            return TextEncoderWrapper(encoder)
        
        text_encoder_l_on_device = wrap_text_encoder(text_encoder_l_on_device)
        text_encoder_g_on_device = wrap_text_encoder(text_encoder_g_on_device)

        # Add config attribute to UNet if it doesn't have it (for diffusers compatibility)
        if not hasattr(unet_on_device, 'config'):
            class UNetConfig:
                def __init__(self):
                    # Basic UNet parameters
                    self.sample_size = 128  # Standard SDXL sample size
                    self.in_channels = 4    # Standard SDXL latent channels
                    self.out_channels = 4   # Standard SDXL output channels
                    
                    # Time embedding parameters
                    self.time_cond_proj_dim = None  # Not used in SDXL
                    self.time_embedding_type = "positional"
                    self.time_embedding_dim = None
                    self.time_embedding_act_fn = None
                    
                    # SDXL-specific parameters
                    self.addition_time_embed_dim = 256  # SDXL addition time embedding dimension
                    self.addition_embed_type = "text_time"  # SDXL addition embedding type
                    self.addition_embed_type_num_heads = 64  # SDXL addition embedding heads
                    
                    # Cross attention parameters
                    self.cross_attention_dim = 2048  # SDXL cross attention dimension
                    self.encoder_hid_dim = None
                    self.encoder_hid_dim_type = None
                    
                    # Architecture parameters
                    self.down_block_types = ["DownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D"]
                    self.up_block_types = ["CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "UpBlock2D"]
                    self.block_out_channels = [320, 640, 1280]
                    self.layers_per_block = 2
                    self.attention_head_dim = [5, 10, 20]
                    self.num_attention_heads = None
                    self.use_linear_projection = True
                    
                    # Conditioning parameters
                    self.class_embed_type = None
                    self.num_class_embeds = None
                    self.projection_class_embeddings_input_dim = 2816  # SDXL projection dimension
                    
                    # Other parameters
                    self.flip_sin_to_cos = True
                    self.freq_shift = 0
                    self.norm_num_groups = 32
                    self.norm_eps = 1e-5
                    self.resnet_time_scale_shift = "default"
                    self.transformer_layers_per_block = 1
                    
            unet_on_device.config = UNetConfig()

        # Add add_embedding attribute to UNet if it doesn't have it (for SDXL compatibility)
        if not hasattr(unet_on_device, 'add_embedding'):
            class AddEmbedding:
                def __init__(self):
                    self.linear_1 = type('Linear1', (), {'in_features': 2816})()  # Standard SDXL add embedding dimension
            unet_on_device.add_embedding = AddEmbedding()

        # Add attention processor methods to UNet if they don't exist (for NAG compatibility)
        if not hasattr(unet_on_device, 'set_attn_processor'):
            def set_attn_processor(attn_processors):
                # Mock method - in this codebase, attention processors are handled differently
                # Just store them for potential future use
                unet_on_device._attn_processors = attn_processors
            unet_on_device.set_attn_processor = set_attn_processor
        
        if not hasattr(unet_on_device, 'attn_processors'):
            # Create a mock attn_processors property
            unet_on_device.attn_processors = {}

        # Make UNet callable if it isn't already (for diffusers compatibility)
        if not callable(unet_on_device):
            def unet_call(sample, timestep, encoder_hidden_states, **kwargs):
                # Call the underlying model through the patcher system
                # This is a simplified call - the actual implementation would need proper handling
                # For now, we'll use the existing ksampler system instead
                return unet_on_device.model(sample, timestep, encoder_hidden_states, **kwargs)
            unet_on_device.__call__ = unet_call

        # Add missing attributes and methods to tokenizers for diffusers compatibility
        def add_tokenizer_compatibility(tokenizer):
            if not hasattr(tokenizer, 'tokenize'):
                def tokenize_method(prompt):
                    # Simple tokenization - just split by spaces for compatibility
                    return prompt.split() if isinstance(prompt, str) else []
                tokenizer.tokenize = tokenize_method
            
            if not hasattr(tokenizer, 'model_max_length'):
                tokenizer.model_max_length = 77  # Standard CLIP tokenizer max length
            
            if not callable(tokenizer):
                # Create a wrapper class that makes the tokenizer callable
                class CallableTokenizerWrapper:
                    def __init__(self, original_tokenizer):
                        self.original_tokenizer = original_tokenizer
                        # Copy all attributes from the original tokenizer
                        for attr_name in dir(original_tokenizer):
                            if not attr_name.startswith('_'):
                                setattr(self, attr_name, getattr(original_tokenizer, attr_name))
                    
                    def __call__(self, prompt, padding=True, truncation=True, max_length=None, return_tensors=None):
                        import torch
                        # Mock tokenizer call that returns the expected structure
                        max_len = max_length or getattr(self, 'model_max_length', 77)
                        
                        # Create a mock result that looks like what diffusers expects
                        class MockTokenizerOutput:
                            def __init__(self, input_ids):
                                self.input_ids = input_ids
                        
                        # Create mock input_ids as tensors with proper shape
                        if isinstance(prompt, str):
                            # Use a more realistic token sequence
                            words = prompt.split()
                            # Create token IDs: start token (49406), word tokens, end token (49407), padding (49407)
                            input_ids = [49406]  # Start token
                            for i, word in enumerate(words[:max_len-2]):  # Leave room for start/end tokens
                                input_ids.append(1000 + (i % 1000))  # Mock word tokens
                            input_ids.append(49407)  # End token
                            
                            # Pad to max_length
                            while len(input_ids) < max_len:
                                input_ids.append(49407)  # Padding token
                        else:
                            # Default: start token + padding
                            input_ids = [49406] + [49407] * (max_len - 1)
                        
                        # Convert to tensor with proper batch dimension
                        input_ids_tensor = torch.tensor([input_ids], dtype=torch.long)
                        
                        return MockTokenizerOutput(input_ids_tensor)
                
                # Replace the tokenizer with the wrapper
                return CallableTokenizerWrapper(tokenizer)
            
            return tokenizer
        
        tokenizer_l_on_device = add_tokenizer_compatibility(tokenizer_l_on_device)
        tokenizer_g_on_device = add_tokenizer_compatibility(tokenizer_g_on_device)

<<<<<<< HEAD
        # Handle guidance by selecting appropriate sampler
        # Check if guidance is active either through parameters OR through sampler selection
        guidance_samplers = ['euler_tpg', 'euler_nag', 'euler_dag', 'euler_guidance']
        sampler_guidance_active = sampler_name in guidance_samplers
        parameter_guidance_active = (tpg_enabled and tpg_scale > 0) or (nag_scale > 1.0) or (dag_enabled and dag_scale > 0)
        guidance_active = parameter_guidance_active or sampler_guidance_active
        
        if guidance_active:
            # If user selected a guidance sampler but didn't set parameters, use defaults
            if sampler_guidance_active and not parameter_guidance_active:
                print(f"[GUIDANCE] User selected guidance sampler '{sampler_name}' - using default parameters")
                if sampler_name == 'euler_tpg':
                    tpg_enabled, tpg_scale = True, 3.0
                elif sampler_name == 'euler_nag':
                    nag_scale = 1.5
                elif sampler_name == 'euler_dag':
                    dag_enabled, dag_scale = True, 2.5
                elif sampler_name == 'euler_guidance':
                    tpg_enabled, tpg_scale = True, 3.0
                    nag_scale = 1.5
                    dag_enabled, dag_scale = True, 2.5
            
            # Set global guidance configuration
            try:
                from ldm_patched.k_diffusion.sampling import set_guidance_config
                set_guidance_config(
                    tpg_scale=tpg_scale if tpg_enabled else 0.0,
                    nag_scale=nag_scale,
                    dag_scale=dag_scale if dag_enabled else 0.0
                )
                print(f"[GUIDANCE] Configuration set: TPG={tpg_scale if tpg_enabled else 0.0}, NAG={nag_scale}, DAG={dag_scale if dag_enabled else 0.0}")
            except ImportError:
                print("[GUIDANCE] Warning: guidance_samplers not available")
            
            # Determine which guidance sampler to use
            if sampler_guidance_active:
                # User explicitly selected a guidance sampler - use it
                guidance_sampler = sampler_name
                print(f"[GUIDANCE] Using user-selected guidance sampler: {sampler_name}")
            elif (tpg_enabled and tpg_scale > 0) and (nag_scale > 1.0) and (dag_enabled and dag_scale > 0):
                # All three guidance methods - use combined sampler
                print(f"[GUIDANCE] Using combined guidance: TPG({tpg_scale}), NAG({nag_scale}), DAG({dag_scale})")
                guidance_sampler = "euler_guidance"
            elif tpg_enabled and tpg_scale > 0:
                print(f"[TPG] Using TPG sampler with scale {tpg_scale}")
                guidance_sampler = "euler_tpg"
            elif nag_scale > 1.0:
                print(f"[NAG] Using NAG sampler with scale {nag_scale}")
                guidance_sampler = "euler_nag"
            elif dag_enabled and dag_scale > 0:
                print(f"[DAG] Using DAG sampler with scale {dag_scale}")
                guidance_sampler = "euler_dag"
            else:
                guidance_sampler = sampler_name
        else:
            guidance_sampler = sampler_name
        
        # Use sampler (with or without guidance)
=======
        # Select pipeline based on TPG/NAG/PAG enabled
        print(f"[DEFAULT_PIPELINE DEBUG] pag_enabled: {pag_enabled}, tpg_enabled: {tpg_enabled}, nag_scale: {nag_scale}")
        if tpg_enabled and tpg_scale > 0:
            pipe_class = StableDiffusionXLTPGPipeline
            print(f"[TPG] Using StableDiffusionXLTPGPipeline with scale {tpg_scale}")
        elif pag_enabled and pag_scale > 0:
            pipe_class = StableDiffusionXLPAGPipeline
            print(f"[PAG] Using StableDiffusionXLPAGPipeline with scale {pag_scale}")
        elif nag_scale > 1.0:
            pipe_class = NAGStableDiffusionXLPipeline
            print(f"[NAG] Using NAGStableDiffusionXLPipeline with scale {nag_scale}")
        else:
            # Fallback to regular ksampler if no special guidance is enabled
            print("[DEFAULT] No special guidance enabled, using regular ksampler")
            # Skip the complex pipeline setup and use regular ksampler
            pass  # This will fall through to the regular ksampler below

        # Instantiate the selected pipeline with the components on the correct device
        pipe = pipe_class(
            vae=vae_on_device,
            text_encoder=text_encoder_l_on_device,
            text_encoder_2=text_encoder_g_on_device,
            tokenizer=tokenizer_l_on_device,
            tokenizer_2=tokenizer_g_on_device,
            unet=unet_on_device,
            scheduler=scheduler_on_device
        )
        print(f"Pipeline instantiated. Its internal device should now be correctly set.")
        
        # Define a wrapper for the callback to decode latents for preview
        def pipe_callback(pipe, step, timestep, callback_kwargs):
            import numpy as np # Added import for numpy
            if callback is not None and not disable_preview:
                latents_for_preview = callback_kwargs["latents"]
                
                # Ensure VAE is on the correct device for decoding
                vae_device = pipe.vae.device
                latents_on_vae_device = latents_for_preview.to(vae_device)

                # Unscale and decode for preview
                scaling_factor = getattr(pipe.vae.config, 'scaling_factor', 0.13025)
                decoded_latents_tensor = pipe.vae.decode(latents_on_vae_device / scaling_factor)
                
                # Convert to PIL Image, then to uint8 HWC numpy array
                preview_img = safe_decode(latents_for_preview[:1], pipe.vae, width=width, height=height)
                preview_np = preview_img

                # Call the original callback with the decoded latent
                # The original callback expects (step, x0, x, total_steps, y)
                # Here, x0 is the decoded image tensor, x is the latent, total_steps is steps, y is not used
                callback(step, preview_np, latents_for_preview, steps, decoded_latents_tensor)
                return {"latents": latents_for_preview}

        # Prepare common pipeline arguments
        pipeline_args = {
            "prompt": original_prompt,
            "negative_prompt": original_negative_prompt,
            "prompt_embeds": prompt_embeds,
            "pooled_prompt_embeds": pooled_prompt_embeds,
            "negative_prompt_embeds": negative_prompt_embeds,
            "negative_pooled_prompt_embeds": negative_pooled_prompt_embeds,
            "height": height,
            "width": width,
            "num_inference_steps": steps,
            "guidance_scale": cfg_scale,
            "generator": torch.Generator(device="cpu").manual_seed(image_seed),
            "latents": initial_latent["samples"],
            "callback_on_step_end": pipe_callback,
            "callback_on_step_end_tensor_inputs": ["latents"],
            "return_dict": False,
        }

        # Add NAG specific arguments if NAG is enabled
        if nag_scale > 1.0:
            pipeline_args.update({
                "nag_scale": nag_scale,
                "nag_tau": nag_tau,
                "nag_alpha": nag_alpha,
                "nag_negative_prompt": final_nag_negative_prompt,
                "nag_end": nag_end,
            })

        # Add TPG specific arguments if TPG is enabled
        if tpg_enabled:
            pipeline_args.update({
                "tpg_scale": tpg_scale,
                "tpg_applied_layers_index": tpg_applied_layers_index,
            })
        
        # Add PAG specific arguments if PAG is enabled
        if pag_enabled:
            pipeline_args.update({
                "pag_scale": pag_scale,
                "pag_applied_layers": pag_applied_layers,
            })

        # Call the selected pipeline
        output = pipe(**pipeline_args)
        
        print(f"Using {'TPG' if tpg_enabled else ('PAG' if pag_enabled else 'NAG')} pipeline for generation")
        print(f"[DEBUG] Pipeline output type: {type(output)}")
        print(f"[DEBUG] Pipeline output content (first 100 chars): {str(output)[:100]}")
        
        # ---------- after pipeline call ----------
        # Pipeline returns a tuple with latents when return_dict=False
        if isinstance(output, tuple):
            latents = output[0]  # Extract latents from tuple
            print(f"[DEBUG] Extracted latents type: {type(latents)}")
            print(f"[DEBUG] Extracted latents shape: {latents.shape}")
            latent_dict = {'samples': latents}
            imgs = core.pytorch_to_numpy(core.decode_vae(target_vae, latent_dict))
        elif isinstance(output, Image.Image):
            # already decoded → convert to numpy HWC uint8 for Fooocus
            imgs = [np.asarray(output.convert("RGB"), dtype=np.uint8)] # skip decode & pytorch_to_numpy
        elif isinstance(output, np.ndarray) and output.ndim == 3:
            # returned uint8 HWC numpy
            imgs = [output]
        else:
            # old path: still latents (B, C, H, W) tensor
            latent_dict = {'samples': output}
            # Use tiled VAE decoding for better memory efficiency
            imgs = core.pytorch_to_numpy(core.decode_vae(target_vae, latent_dict, tiled=True))
        
        print(f"[DEBUG] imgs after processing pipeline output: {type(imgs)}")
        if len(imgs) > 0:
            print(f"[DEBUG] First image in imgs: {type(imgs[0])}, shape: {imgs[0].shape}")
        print("Final deliverable:", type(imgs[0]), getattr(imgs[0], 'size', '-'))
        
        # Skip the regular ksampler since we used NAG/TPG/PAG pipeline
    
    # Use regular ksampler if no special guidance is enabled
    if not (nag_scale > 1.0 or (tpg_enabled and tpg_scale > 0) or (pag_enabled and pag_scale > 0)):
        print("[DEFAULT] Using regular ksampler")
>>>>>>> parent of a9a7293 (SAMPLERS)
        ksampler_imgs = core.ksampler(
            model=final_unet,
            positive=positive_cond,
            negative=negative_cond,
            latent=initial_latent,
            seed=image_seed,
            steps=steps,
            cfg=cfg_scale,
            sampler_name=guidance_sampler,
            scheduler=scheduler_name,
            denoise=denoise,
            disable_preview=disable_preview,
            refiner=final_refiner_unet,
            refiner_switch=switch,
            sigmas=minmax_sigmas,
            callback_function=callback
        )['samples']
        
        # Convert latents to images
        if ksampler_imgs is not None:
            latent_dict = {'samples': ksampler_imgs}
            imgs = core.decode_vae(target_vae, latent_dict)
            imgs = core.pytorch_to_numpy(imgs)
        else:
            imgs = []
        
        return imgs

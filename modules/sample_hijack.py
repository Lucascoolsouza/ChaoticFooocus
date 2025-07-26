import torch
import ldm_patched.modules.samplers
import ldm_patched.modules.model_management

from collections import namedtuple
from ldm_patched.contrib.external_align_your_steps import AlignYourStepsScheduler
from ldm_patched.contrib.external_custom_sampler import SDTurboScheduler
from ldm_patched.k_diffusion import sampling as k_diffusion_sampling
# Import guidance configuration functions
from ldm_patched.modules.samplers import normal_scheduler, simple_scheduler, ddim_scheduler
from ldm_patched.modules.model_base import SDXLRefiner, SDXL
from ldm_patched.modules.conds import CONDRegular
from ldm_patched.modules.sample import get_additional_models, get_models_from_cond, cleanup_additional_models
from ldm_patched.modules.samplers import resolve_areas_and_cond_masks, wrap_model, calculate_start_end_timesteps, \
    create_cond_with_same_area_if_none, pre_run_control, apply_empty_x_to_equal_area, encode_model_conds

# Import guidance configuration functions
current_refiner = None
refiner_switch_step = -1


@torch.no_grad()
@torch.inference_mode()
def clip_separate_inner(c, p, target_model=None, target_clip=None):
    if target_model is None or isinstance(target_model, SDXLRefiner):
        c = c[..., -1280:].clone()
    elif isinstance(target_model, SDXL):
        c = c.clone()
    else:
        p = None
        c = c[..., :768].clone()

        final_layer_norm = target_clip.cond_stage_model.clip_l.transformer.text_model.final_layer_norm

        final_layer_norm_origin_device = final_layer_norm.weight.device
        final_layer_norm_origin_dtype = final_layer_norm.weight.dtype

        c_origin_device = c.device
        c_origin_dtype = c.dtype

        final_layer_norm.to(device='cpu', dtype=torch.float32)
        c = c.to(device='cpu', dtype=torch.float32)

        c = torch.chunk(c, int(c.size(1)) // 77, 1)
        c = [final_layer_norm(ci) for ci in c]
        c = torch.cat(c, dim=1)

        final_layer_norm.to(device=final_layer_norm_origin_device, dtype=final_layer_norm_origin_dtype)
        c = c.to(device=c_origin_device, dtype=c_origin_dtype)
    return c, p


@torch.no_grad()
@torch.inference_mode()
def clip_separate(cond, target_model=None, target_clip=None):
    results = []

    for c, px in cond:
        p = px.get('pooled_output', None)
        c, p = clip_separate_inner(c, p, target_model=target_model, target_clip=target_clip)
        p = {} if p is None else {'pooled_output': p.clone()}
        results.append([c, p])

    return results


@torch.no_grad()
@torch.inference_mode()
def clip_separate_after_preparation(cond, target_model=None, target_clip=None):
    results = []

    for x in cond:
        p = x.get('pooled_output', None)
        c = x['model_conds']['c_crossattn'].cond

        c, p = clip_separate_inner(c, p, target_model=target_model, target_clip=target_clip)

        result = {'model_conds': {'c_crossattn': CONDRegular(c)}}

        if p is not None:
            result['pooled_output'] = p.clone()

        results.append(result)

    return results


@torch.no_grad()
@torch.inference_mode()
def sample_hacked(model, noise, positive, negative, cfg, device, sampler, sigmas, model_options={}, latent_image=None, denoise_mask=None, callback=None, disable_pbar=False, seed=None):
    global current_refiner

    positive = positive[:]
    negative = negative[:]

    resolve_areas_and_cond_masks(positive, noise.shape[2], noise.shape[3], device)
    resolve_areas_and_cond_masks(negative, noise.shape[2], noise.shape[3], device)

    model_wrap = wrap_model(model)

    calculate_start_end_timesteps(model, negative)
    calculate_start_end_timesteps(model, positive)

    if latent_image is not None:
        latent_image = model.process_latent_in(latent_image)

    if hasattr(model, 'extra_conds'):
        positive = encode_model_conds(model.extra_conds, positive, noise, device, "positive", latent_image=latent_image, denoise_mask=denoise_mask)
        negative = encode_model_conds(model.extra_conds, negative, noise, device, "negative", latent_image=latent_image, denoise_mask=denoise_mask)

    #make sure each cond area has an opposite one with the same area
    for c in positive:
        create_cond_with_same_area_if_none(negative, c)
    for c in negative:
        create_cond_with_same_area_if_none(positive, c)

    # pre_run_control(model, negative + positive)
    pre_run_control(model, positive)  # negative is not necessary in Fooocus, 0.5s faster.

    apply_empty_x_to_equal_area(list(filter(lambda c: c.get('control_apply_to_uncond', False) == True, positive)), negative, 'control', lambda cond_cnets, x: cond_cnets[x])
    apply_empty_x_to_equal_area(positive, negative, 'gligen', lambda cond_cnets, x: cond_cnets[x])

    extra_args = {"cond":positive, "uncond":negative, "cond_scale": cfg, "model_options": model_options, "seed":seed}

    if sampler == "negative_focus":
        # Extract neg_text_emb from the negative conditioning
        # Assuming negative is a list of dictionaries, and the first one contains the main negative prompt
        if negative and 'model_conds' in negative[0] and 'c_crossattn' in negative[0]['model_conds']:
            extra_args["neg_text_emb"] = negative[0]['model_conds']['c_crossattn'].cond
        else:
            # Handle case where neg_text_emb might not be available or structured differently
            # For now, raise an error or set a default/empty tensor
            print("Warning: neg_text_emb not found for negative_focus sampler. Using empty tensor.")
            extra_args["neg_text_emb"] = torch.empty(1, 1, 768) # Placeholder, adjust dimensions as needed

    if sampler == "token_shuffle":
        extra_args["cond"] = positive[0]['model_conds']['c_crossattn'].cond
        extra_args["shuffle_start"] = 0.5
        extra_args["shuffle_prob"] = 0.3

    if sampler == "diverse_attention":
        extra_args["attn_dropout"] = 0.1  # Default value from sample_diverse_attention
        extra_args["attn_temp"] = 0.7     # Default value from sample_diverse_attention
        extra_args["diversity_start"] = 0.4 # Default value from sample_diverse_attention

    if sampler == "dpmpp_unipc_restart":
        extra_args["restart_list"] = None # Default value from sample_dpmpp_unipc_restart
        extra_args["unipc_order"] = 3
        extra_args["unipc_rtol"] = 0.05
        extra_args["unipc_atol"] = 0.0078
        extra_args["unipc_h_init"] = 0.05
        extra_args["unipc_pcoeff"] = 0.0
        extra_args["unipc_icoeff"] = 1.0
        extra_args["unipc_dcoeff"] = 0.0
        extra_args["unipc_accept_safety"] = 0.81

    if current_refiner is not None and hasattr(current_refiner.model, 'extra_conds'):
        positive_refiner = clip_separate_after_preparation(positive, target_model=current_refiner.model)
        negative_refiner = clip_separate_after_preparation(negative, target_model=current_refiner.model)

        positive_refiner = encode_model_conds(current_refiner.model.extra_conds, positive_refiner, noise, device, "positive", latent_image=latent_image, denoise_mask=denoise_mask)
        negative_refiner = encode_model_conds(current_refiner.model.extra_conds, negative_refiner, noise, device, "negative", latent_image=latent_image, denoise_mask=denoise_mask)

    def refiner_switch():
        cleanup_additional_models(set(get_models_from_cond(positive, "control") + get_models_from_cond(negative, "control")))

        extra_args["cond"] = positive_refiner
        extra_args["uncond"] = negative_refiner

        # clear ip-adapter for refiner
        extra_args['model_options'] = {k: {} if k == 'transformer_options' else v for k, v in extra_args['model_options'].items()}

        models, inference_memory = get_additional_models(positive_refiner, negative_refiner, current_refiner.model_dtype())
        ldm_patched.modules.model_management.load_models_gpu(
            [current_refiner] + models,
            model.memory_required([noise.shape[0] * 2] + list(noise.shape[1:])) + inference_memory)

        model_wrap.inner_model = current_refiner.model
        print('Refiner Swapped')
        return

    def callback_wrap(step, x0, x, total_steps):
        if step == refiner_switch_step and current_refiner is not None:
            refiner_switch()
        if callback is not None:
            # residual_noise_preview = x - x0
            # residual_noise_preview /= residual_noise_preview.std()
            # residual_noise_preview *= x0.std()
            callback(step, x0, x, total_steps)

    if sampler == "negative_focus":
        neg_text_emb = extra_args.pop("neg_text_emb", None)
        samples = sampler.sample(model_wrap, sigmas, extra_args, callback_wrap, noise, latent_image, denoise_mask, disable_pbar, neg_text_emb=neg_text_emb)
    else:
        samples = sampler.sample(model_wrap, sigmas, extra_args, callback_wrap, noise, latent_image, denoise_mask, disable_pbar)
    return model.process_latent_out(samples.to(torch.float32))


@torch.no_grad()
@torch.inference_mode()
def calculate_sigmas_scheduler_hacked(model, scheduler_name, steps):
    sigma_min = float(model.model_sampling.sigma_min)
    sigma_max = float(model.model_sampling.sigma_max)

    if scheduler_name == "normal":
        sigmas = normal_scheduler(model, steps)

    elif scheduler_name == "karras":
        sigmas = k_diffusion_sampling.get_sigmas_karras(n=steps, sigma_min=float(model.model_sampling.sigma_min), sigma_max=float(model.model_sampling.sigma_max))

    elif scheduler_name == "exponential":
        sigmas = k_diffusion_sampling.get_sigmas_exponential(
            n=steps, sigma_min=sigma_min, sigma_max=sigma_max
        )

    elif scheduler_name == "sgm_uniform":
        sigmas = normal_scheduler(model, steps, sgm=True)

    elif scheduler_name == "simple":
        sigmas = simple_scheduler(model, steps)

    elif scheduler_name == "ddim_uniform":
        sigmas = ddim_scheduler(model, steps)

    elif scheduler_name == "turbo":
        sigmas = SDTurboScheduler().get_sigmas(model=model, steps=steps, denoise=1.0)[0]

    elif scheduler_name == "align_your_steps":
        model_type = 'SDXL' if isinstance(
            model.latent_format,
            ldm_patched.modules.latent_formats.SDXL
        ) else 'SD1'
        sigmas = AlignYourStepsScheduler().get_sigmas(
            model_type=model_type, steps=steps, denoise=1.0
        )[0]
              
    else:
        raise TypeError(f"error invalid scheduler: {scheduler_name}")

    return sigmas



ldm_patched.modules.samplers.calculate_sigmas_scheduler = calculate_sigmas_scheduler_hacked
ldm_patched.modules.samplers.sample = sample_hacked

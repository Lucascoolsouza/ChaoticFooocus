from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import math

import torch
import torch.nn as nn

from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.image_processor import PipelineImageInput
from diffusers.utils import (
    deprecate,
    is_torch_xla_available,
)
from diffusers.pipelines.stable_diffusion_xl.pipeline_output import StableDiffusionXLPipelineOutput

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False
from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import (
    StableDiffusionXLPipeline,
    retrieve_timesteps,
    rescale_noise_cfg,
)

from .attention_nag import NAGAttnProcessor2_0

import numpy as np
from PIL import Image, ImageDraw

def safe_decode(latents, vae, width=512, height=512):
    try:
        with torch.no_grad():
            latents = latents.to(vae.device)
            decoded = vae.decode(latents)

            if hasattr(vae, "post_quant_conv"):
                vae.post_quant_conv = vae.post_quant_conv.to(decoded.device, decoded.dtype)
                decoded = vae.post_quant_conv(decoded)
            else:
                decoded = decoded[:, :3] if decoded.size(1) >= 3 else decoded.mean(dim=1, keepdim=True).repeat(1, 3, 1, 1)

            decoded = torch.clamp((decoded + 1) * 0.5, 0, 1)
            decoded_np = (decoded[0].permute(1, 2, 0) * 255).cpu().numpy().astype(np.uint8)
            return Image.fromarray(decoded_np, mode='RGB')

    except Exception as e:
        print(f"[safe_decode] ❌ Decode failed: {e}")
        # Return error image
        img = Image.new("RGB", (width, height), color="red")
        draw = ImageDraw.Draw(img)
        draw.text((10, 10), f"Decode Error", fill="white")
        return img



class NAGStableDiffusionXLPipeline(StableDiffusionXLPipeline):
    @property
    def do_normalized_attention_guidance(self):
        return self._nag_scale > 1

    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1

    def _set_nag_attn_processor(self, nag_scale, nag_tau=2.5, nag_alpha=0.5):
        if self.do_normalized_attention_guidance:
            attn_procs = {}
            for name, origin_attn_processor in getattr(self.unet, 'attn_processors', {}).items():
                if "attn2" in name:
                    attn_procs[name] = NAGAttnProcessor2_0(nag_scale=nag_scale, nag_tau=nag_tau, nag_alpha=nag_alpha)
                else:
                    attn_procs[name] = origin_attn_processor
            self.unet.set_attn_processor(attn_procs)

    def maybe_convert_prompt(self, prompt, tokenizer):
        # Override to bypass textual inversion logic that requires tokenizer.tokenize()
        # Just return the prompt as-is since we're using pre-computed embeddings anyway
        return prompt

    def encode_prompt(self, prompt, prompt_2=None, device=None, num_images_per_prompt=1, do_classifier_free_guidance=True, negative_prompt=None, negative_prompt_2=None, prompt_embeds=None, negative_prompt_embeds=None, pooled_prompt_embeds=None, negative_pooled_prompt_embeds=None, lora_scale=None, clip_skip=None):
        # Override encode_prompt to bypass tokenization issues
        # Since we're passing pre-computed embeddings, just return them
        if prompt_embeds is not None:
            # Use the pre-computed embeddings directly
            batch_size = prompt_embeds.shape[0]
            
            # Store the shape for later use with NAG embeddings
            self._current_prompt_embeds_shape = prompt_embeds.shape
            
            # Ensure embeddings are on the correct device
            prompt_embeds = prompt_embeds.to(device)
            if pooled_prompt_embeds is not None:
                pooled_prompt_embeds = pooled_prompt_embeds.to(device)
            
            # Handle negative embeddings
            if negative_prompt_embeds is not None:
                negative_prompt_embeds = negative_prompt_embeds.to(device)
            if negative_pooled_prompt_embeds is not None:
                negative_pooled_prompt_embeds = negative_pooled_prompt_embeds.to(device)
            
            return prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds
        
        # Handle simple prompts (like NAG negative prompts) by properly encoding them
        if isinstance(prompt, str) and prompt.strip() != "":
            # For NAG negative prompts, we need to encode them safely
            try:
                # Try to use parent method for proper text encoding
                return super().encode_prompt(prompt, prompt_2, device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt, negative_prompt_2, prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds, lora_scale, clip_skip)
            except Exception as e:
                print(f"[NAG] Error encoding NAG negative prompt '{prompt}': {e}")
                print("[NAG] Falling back to zero embeddings for NAG negative prompt")
                # Fall back to zero embeddings if encoding fails
                import torch
                batch_size = num_images_per_prompt
                
                # Try to get sequence length from stored shape
                seq_len = 77  # Default CLIP sequence length
                embed_dim = 2048  # Default SDXL embedding dimension
                
                if hasattr(self, '_current_prompt_embeds_shape') and self._current_prompt_embeds_shape is not None:
                    seq_len = self._current_prompt_embeds_shape[1]
                    embed_dim = self._current_prompt_embeds_shape[2]
                
                embed_dim = 2048  # Standard SDXL embedding dimension
                pooled_dim = 1280  # Standard SDXL pooled dimension
                
                # Create zero embeddings (neutral)
                dummy_embeds = torch.zeros((batch_size, seq_len, embed_dim), device=device, dtype=torch.float16 if device.type == 'cuda' else torch.float32)
                dummy_pooled = torch.zeros((batch_size, pooled_dim), device=device, dtype=torch.float16 if device.type == 'cuda' else torch.float32)
                
                return dummy_embeds, None, dummy_pooled, None
        elif isinstance(prompt, str) and prompt.strip() == "":
            import torch
            # Create zero embeddings only for empty strings
            batch_size = num_images_per_prompt
            
            # Try to get sequence length from stored shape
            seq_len = 77  # Default CLIP sequence length
            embed_dim = 2048  # Default SDXL embedding dimension
            
            if hasattr(self, '_current_prompt_embeds_shape') and self._current_prompt_embeds_shape is not None:
                seq_len = self._current_prompt_embeds_shape[1]
                embed_dim = self._current_prompt_embeds_shape[2]
            
            embed_dim = 2048  # Standard SDXL embedding dimension
            pooled_dim = 1280  # Standard SDXL pooled dimension
            
            # Create zero embeddings (neutral)
            dummy_embeds = torch.zeros((batch_size, seq_len, embed_dim), device=device, dtype=torch.float16 if device.type == 'cuda' else torch.float32)
            dummy_pooled = torch.zeros((batch_size, pooled_dim), device=device, dtype=torch.float16 if device.type == 'cuda' else torch.float32)
            
            return dummy_embeds, None, dummy_pooled, None
        
        # Fallback to parent method if no pre-computed embeddings
        return super().encode_prompt(prompt, prompt_2, device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt, negative_prompt_2, prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds, lora_scale, clip_skip)

    @torch.no_grad()
    def __call__(
            self,
            prompt: Union[str, List[str]] = None,
            prompt_2: Optional[Union[str, List[str]]] = None,
            height: Optional[int] = None,
            width: Optional[int] = None,
            num_inference_steps: int = 50,
            timesteps: List[int] = None,
            sigmas: List[float] = None,
            denoising_end: Optional[float] = None,
            guidance_scale: float = 5.0,
            negative_prompt: Optional[Union[str, List[str]]] = None,
            negative_prompt_2: Optional[Union[str, List[str]]] = None,
            num_images_per_prompt: Optional[int] = 1,
            eta: float = 0.0,
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            latents: Optional[torch.Tensor] = None,
            prompt_embeds: Optional[torch.Tensor] = None,
            negative_prompt_embeds: Optional[torch.Tensor] = None,
            pooled_prompt_embeds: Optional[torch.Tensor] = None,
            negative_pooled_prompt_embeds: Optional[torch.Tensor] = None,
            ip_adapter_image: Optional[PipelineImageInput] = None,
            ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
            output_type: Optional[str] = "pil",
            return_dict: bool = True,
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,
            guidance_rescale: float = 0.0,
            original_size: Optional[Tuple[int, int]] = None,
            crops_coords_top_left: Tuple[int, int] = (0, 0),
            target_size: Optional[Tuple[int, int]] = None,
            negative_original_size: Optional[Tuple[int, int]] = None,
            negative_crops_coords_top_left: Tuple[int, int] = (0, 0),
            negative_target_size: Optional[Tuple[int, int]] = None,
            clip_skip: Optional[int] = None,
            callback_on_step_end: Optional[
                Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
            ] = None,
            callback_on_step_end_tensor_inputs: List[str] = ["latents"],

            nag_scale: float = 1.0,
            nag_tau: float = 2.5,
            nag_alpha: float = 0.5,
            nag_negative_prompt: str = None,
            nag_negative_prompt_embeds: Optional[torch.Tensor] = None,
            nag_end: float = 1.0,

            **kwargs,
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts to be sent to the `tokenizer_2` and `text_encoder_2`. If not defined, `prompt` is
                used in both text-encoders
            height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The height in pixels of the generated image. This is set to 1024 by default for the best results.
                Anything below 512 pixels won't work well for
                [stabilityai/stable-diffusion-xl-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)
                and checkpoints that are not specifically fine-tuned on low resolutions.
            width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The width in pixels of the generated image. This is set to 1024 by default for the best results.
                Anything below 512 pixels won't work well for
                [stabilityai/stable-diffusion-xl-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)
                and checkpoints that are not specifically fine-tuned on low resolutions.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            timesteps (`List[int]`, *optional*):
                Custom timesteps to use for the denoising process with schedulers which support a `timesteps` argument
                in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is
                passed will be used. Must be in descending order.
            sigmas (`List[float]`, *optional*):
                Custom sigmas to use for the denoising process with schedulers which support a `sigmas` argument in
                their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is passed
                will be used.
            denoising_end (`float`, *optional*):
                When specified, determines the fraction (between 0.0 and 1.0) of the total denoising process to be
                completed before it is intentionally prematurely terminated. As a result, the returned sample will
                still retain a substantial amount of noise as determined by the discrete timesteps selected by the
                scheduler. The denoising_end parameter should ideally be utilized when this pipeline forms a part of a
                "Mixture of Denoisers" multi-pipeline setup, as elaborated in [**Refining the Image
                Output**](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/stable_diffusion_xl#refining-the-image-output)
            guidance_scale (`float`, *optional*, defaults to 5.0):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            negative_prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation to be sent to `tokenizer_2` and
                `text_encoder_2`. If not defined, `negative_prompt` is used in both text-encoders
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.Tensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            pooled_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting.
                If not provided, pooled text embeddings will be generated from `prompt` input argument.
            negative_pooled_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, pooled negative_prompt_embeds will be generated from `negative_prompt`
                input argument.
            ip_adapter_image: (`PipelineImageInput`, *optional*): Optional image input to work with IP Adapters.
            ip_adapter_image_embeds (`List[torch.Tensor]`, *optional*):
                Pre-generated image embeddings for IP-Adapter. It should be a list of length same as number of
                IP-adapters. Each element should be a tensor of shape `(batch_size, num_images, emb_dim)`. It should
                contain the negative image embedding if `do_classifier_free_guidance` is set to `True`. If not
                provided, embeddings are computed from the `ip_adapter_image` input argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion_xl.StableDiffusionXLPipelineOutput`] instead
                of a plain tuple.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            guidance_rescale (`float`, *optional*, defaults to 0.0):
                Guidance rescale factor proposed by [Common Diffusion Noise Schedules and Sample Steps are
                Flawed](https://arxiv.org/pdf/2305.08891.pdf) `guidance_scale` is defined as `φ` in equation 16. of
                [Common Diffusion Noise Schedules and Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf).
                Guidance rescale factor should fix overexposure when using zero terminal SNR.
            original_size (`Tuple[int]`, *optional*, defaults to (1024, 1024)):
                If `original_size` is not the same as `target_size` the image will appear to be down- or upsampled.
                `original_size` defaults to `(height, width)` if not specified. Part of SDXL's micro-conditioning as
                explained in section 2.2 of
                [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952).
            crops_coords_top_left (`Tuple[int]`, *optional*, defaults to (0, 0)):
                `crops_coords_top_left` can be used to generate an image that appears to be "cropped" from the position
                `crops_coords_top_left` downwards. Favorable, well-centered images are usually achieved by setting
                `crops_coords_top_left` to (0, 0). Part of SDXL's micro-conditioning as explained in section 2.2 of
                [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952).
            target_size (`Tuple[int]`, *optional*, defaults to (1024, 1024)):
                For most cases, `target_size` should be set to the desired height and width of the generated image. If
                not specified it will default to `(height, width)`. Part of SDXL's micro-conditioning as explained in
                section 2.2 of [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952).
            negative_original_size (`Tuple[int]`, *optional*, defaults to (1024, 1024)):
                To negatively condition the generation process based on a specific image resolution. Part of SDXL's
                micro-conditioning as explained in section 2.2 of
                [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952). For more
                information, refer to this issue thread: https://github.com/huggingface/diffusers/issues/4208.
            negative_crops_coords_top_left (`Tuple[int]`, *optional*, defaults to (0, 0)):
                To negatively condition the generation process based on a specific crop coordinates. Part of SDXL's
                micro-conditioning as explained in section 2.2 of
                [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952). For more
                information, refer to this issue thread: https://github.com/huggingface/diffusers/issues/4208.
            negative_target_size (`Tuple[int]`, *optional*, defaults to (1024, 1024)):
                To negatively condition the generation process based on a target image resolution. It should be as same
                as the `target_size` for most cases. Part of SDXL's micro-conditioning as explained in section 2.2 of
                [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952). For more
                information, refer to this issue thread: https://github.com/huggingface/diffusers/issues/4208.
            callback_on_step_end (`Callable`, `PipelineCallback`, `MultiPipelineCallbacks`, *optional*):
                A function or a subclass of `PipelineCallback` or `MultiPipelineCallbacks` that is called at the end of
                each denoising step during the inference. with the following arguments: `callback_on_step_end(self:
                DiffusionPipeline, step: int, timestep: int, callback_kwargs: Dict)`. `callback_kwargs` will include a
                list of all tensors as specified by `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.

        Examples:

        Returns:
            [`~pipelines.stable_diffusion_xl.StableDiffusionXLPipelineOutput`] or `tuple`:
            [`~pipelines.stable_diffusion_xl.StableDiffusionXLPipelineOutput`] if `return_dict` is True, otherwise a
            `tuple`. When returning a tuple, the first element is a list with the generated images.
        """

        callback = kwargs.pop("callback", None)
        callback_steps = kwargs.pop("callback_steps", None)

        if callback is not None:
            deprecate(
                "callback",
                "1.0.0",
                "Passing `callback` as an input argument to `__call__` is deprecated, consider use `callback_on_step_end`",
            )
        if callback_steps is not None:
            deprecate(
                "callback_steps",
                "1.0.0",
                "Passing `callback_steps` as an input argument to `__call__` is deprecated, consider use `callback_on_step_end`",
            )

        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        # 0. Default height and width to unet
        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        # Enable VAE tiling and attention slicing for memory optimization
        if hasattr(self.vae, "enable_tiling"):
            self.vae.enable_tiling()
        if hasattr(self.unet, "enable_attention_slicing"):
            self.unet.enable_attention_slicing()

        original_size = original_size or (height, width)
        target_size = target_size or (height, width)

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            prompt_2,
            height,
            width,
            callback_steps,
            negative_prompt,
            negative_prompt_2,
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
            ip_adapter_image,
            ip_adapter_image_embeds,
            callback_on_step_end_tensor_inputs,
        )

        self._guidance_scale = guidance_scale
        self._guidance_rescale = guidance_rescale
        self._clip_skip = clip_skip
        self._cross_attention_kwargs = cross_attention_kwargs
        self._denoising_end = denoising_end
        self._interrupt = False
        self._nag_scale = nag_scale

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = next(iter(self.unet.model.parameters())).device

        # 3. Encode input prompt
        lora_scale = (
            self.cross_attention_kwargs.get("scale", None) if self.cross_attention_kwargs is not None else None
        )

        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            do_classifier_free_guidance=self.do_classifier_free_guidance or self.do_normalized_attention_guidance,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            lora_scale=lora_scale,
            clip_skip=self.clip_skip,
        )
        if self.do_normalized_attention_guidance:
            if nag_negative_prompt_embeds is None:
                # First priority: use existing negative prompt embeddings if available
                if self.do_classifier_free_guidance and negative_prompt_embeds is not None:
                    nag_negative_prompt_embeds = negative_prompt_embeds
                    print("[NAG] Using existing CFG negative prompt embeddings for NAG")
                # Second priority: try to encode the NAG negative prompt if provided
                elif nag_negative_prompt is not None and nag_negative_prompt.strip() != "":
                    print(f"[NAG] Encoding NAG negative prompt: '{nag_negative_prompt}'")
                    try:
                        nag_negative_prompt_embeds = self.encode_prompt(
                            prompt=nag_negative_prompt,
                            device=device,
                            num_images_per_prompt=num_images_per_prompt,
                            do_classifier_free_guidance=False,
                            lora_scale=lora_scale,
                            clip_skip=self.clip_skip,
                        )[0]
                    except Exception as e:
                        print(f"[NAG] Failed to encode NAG negative prompt: {e}")
                        print("[NAG] Using existing negative prompt embeddings as fallback")
                        if negative_prompt_embeds is not None:
                            nag_negative_prompt_embeds = negative_prompt_embeds
                        else:
                            print("[NAG] No negative embeddings available, NAG will be less effective")
                            nag_negative_prompt_embeds = None
                # Third priority: try to encode the regular negative prompt
                elif negative_prompt is not None and negative_prompt.strip() != "":
                    print(f"[NAG] Using regular negative prompt for NAG: '{negative_prompt}'")
                    if negative_prompt_embeds is not None:
                        nag_negative_prompt_embeds = negative_prompt_embeds
                    else:
                        nag_negative_prompt = negative_prompt
                        try:
                            nag_negative_prompt_embeds = self.encode_prompt(
                                prompt=nag_negative_prompt,
                                device=device,
                                num_images_per_prompt=num_images_per_prompt,
                                do_classifier_free_guidance=False,
                                lora_scale=lora_scale,
                                clip_skip=self.clip_skip,
                            )[0]
                        except Exception as e:
                            print(f"[NAG] Failed to encode regular negative prompt: {e}")
                            nag_negative_prompt_embeds = None
                else:
                    print("[NAG] No negative prompt available, NAG will be less effective")
                    nag_negative_prompt_embeds = None

        # 4. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, timesteps, sigmas
        )

        # 5. Prepare latent variables
        num_channels_latents = 4  # Standard SDXL latent channels
        print(f"[NAG DEBUG] Input latents parameter: {latents.shape if latents is not None else None}")
        if latents is not None:
            print(f"[NAG DEBUG] Input latents mean: {latents.mean().item():.6f}, std: {latents.std().item():.6f}")
        
        # Try to use the input latents directly if they look reasonable
        if latents is not None and latents.std().item() > 0.01:
            print(f"[NAG DEBUG] Using input latents directly (they look reasonable)")
            latents = latents.to(device=device, dtype=prompt_embeds.dtype)
        else:
            print(f"[NAG DEBUG] Input latents are None or zero, calling prepare_latents")
            latents = self.prepare_latents(
                batch_size * num_images_per_prompt,
                num_channels_latents,
                height,
                width,
                prompt_embeds.dtype,
                device,
                generator,
                latents,
            )
        print(f"[NAG DEBUG] Final prepared latents shape: {latents.shape}, mean: {latents.mean().item():.6f}, std: {latents.std().item():.6f}")

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7. Prepare added time ids & embeddings
        add_text_embeds = pooled_prompt_embeds
        if self.text_encoder_2 is None:
            text_encoder_projection_dim = int(pooled_prompt_embeds.shape[-1])
        else:
            # Use a default projection dimension for SDXL
            text_encoder_projection_dim = 1280  # Standard SDXL text encoder 2 projection dimension

        add_time_ids = self._get_add_time_ids(
            original_size,
            crops_coords_top_left,
            target_size,
            dtype=prompt_embeds.dtype,
            text_encoder_projection_dim=text_encoder_projection_dim,
        )
        if negative_original_size is not None and negative_target_size is not None:
            negative_add_time_ids = self._get_add_time_ids(
                negative_original_size,
                negative_crops_coords_top_left,
                negative_target_size,
                dtype=prompt_embeds.dtype,
                text_encoder_projection_dim=text_encoder_projection_dim,
            )
        else:
            negative_add_time_ids = add_time_ids

        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            add_text_embeds = torch.cat([negative_pooled_prompt_embeds, add_text_embeds], dim=0)
            add_time_ids = torch.cat([negative_add_time_ids, add_time_ids], dim=0)

        if self.do_normalized_attention_guidance and nag_negative_prompt_embeds is not None:
            # Concatenate prompt_embeds, add_text_embeds, and add_time_ids for NAG
            prompt_embeds = torch.cat([prompt_embeds, nag_negative_prompt_embeds], dim=0)
            # For add_text_embeds and add_time_ids, we need to replicate the existing ones
            # to match the batch size of the concatenated prompt_embeds
            num_existing_batches = add_text_embeds.shape[0]
            num_nag_batches = nag_negative_prompt_embeds.shape[0]
            
            # Replicate existing add_text_embeds and add_time_ids for NAG
            replicated_add_text_embeds = add_text_embeds[:num_nag_batches] if num_existing_batches >= num_nag_batches else add_text_embeds.repeat(math.ceil(num_nag_batches / num_existing_batches), 1, 1)[:num_nag_batches]
            replicated_add_time_ids = add_time_ids[:num_nag_batches] if num_existing_batches >= num_nag_batches else add_time_ids.repeat(math.ceil(num_nag_batches / num_existing_batches), 1)[:num_nag_batches]

            add_text_embeds = torch.cat([add_text_embeds, replicated_add_text_embeds], dim=0)
            add_time_ids = torch.cat([add_time_ids, replicated_add_time_ids], dim=0)

        elif self.do_normalized_attention_guidance and nag_negative_prompt_embeds is None:
            print("[NAG] Warning: NAG is enabled but no negative embeddings available, disabling NAG")
            self._nag_scale = 1.0  # Disable NAG if no negative embeddings

        prompt_embeds = prompt_embeds.to(device)
        add_text_embeds = add_text_embeds.to(device)
        add_time_ids = add_time_ids.to(device).repeat(batch_size * num_images_per_prompt, 1)

        if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
            image_embeds = self.prepare_ip_adapter_image_embeds(
                ip_adapter_image,
                ip_adapter_image_embeds,
                device,
                batch_size * num_images_per_prompt,
                self.do_classifier_free_guidance,
            )

        # 8. Denoising loop
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)

        # 8.1 Apply denoising_end
        if (
                self.denoising_end is not None
                and isinstance(self.denoising_end, float)
                and self.denoising_end > 0
                and self.denoising_end < 1
        ):
            discrete_timestep_cutoff = int(
                round(
                    1000  # Standard diffusion timesteps
                    - (self.denoising_end * 1000)
                )
            )
            num_inference_steps = len(list(filter(lambda ts: ts >= discrete_timestep_cutoff, timesteps)))
            timesteps = timesteps[:num_inference_steps]

        # 9. Optionally get Guidance Scale Embedding
        timestep_cond = None
        # Skip guidance scale embedding for custom UNet - not typically used in SDXL
        timestep_cond = None

        if self.do_normalized_attention_guidance:
            origin_attn_procs = getattr(self.unet, 'attn_processors', {})
            self._set_nag_attn_processor(nag_scale, nag_tau, nag_alpha)
            attn_procs_recovered = False

        self._num_timesteps = len(timesteps)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents

                if self.do_normalized_attention_guidance and nag_negative_prompt_embeds is not None:
                    # Ensure latent_model_input matches the batch size of prompt_embeds
                    target_batch_size = prompt_embeds.shape[0]
                    current_latent_batch_size = latent_model_input.shape[0]

                    if target_batch_size > current_latent_batch_size:
                        num_latents_to_add = target_batch_size - current_latent_batch_size
                        # Repeat the first latent to match the required number of additional latents
                        latent_model_input = torch.cat([latent_model_input, latents[0:1].repeat(num_latents_to_add, 1, 1, 1)], dim=0)

                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # Check if we should disable NAG for this timestep
                if t < math.floor((1 - nag_end) * 999) and self.do_normalized_attention_guidance and not attn_procs_recovered:
                    self.unet.set_attn_processor(origin_attn_procs)
                    prompt_embeds = prompt_embeds[:len(latent_model_input)]
                    attn_procs_recovered = True

                # predict the noise residual
                added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}
                if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
                    added_cond_kwargs["image_embeds"] = image_embeds

                # Use ComfyUI model interface
                if hasattr(self.unet, 'model') and hasattr(self.unet.model, 'apply_model'):
                    # ComfyUI wrapped model - convert Diffusers conditioning to ComfyUI format
                    # For SDXL, we need to pass the additional conditioning as separate parameters
                    comfy_kwargs = {}
                    if "text_embeds" in added_cond_kwargs and "time_ids" in added_cond_kwargs:
                        # Convert Diffusers SDXL conditioning to ComfyUI format
                        pooled_output = added_cond_kwargs["text_embeds"]
                        time_ids = added_cond_kwargs["time_ids"]
                        
                        # Ensure batch sizes match latent_model_input
                        target_batch_size = latent_model_input.shape[0]
                        if pooled_output.shape[0] != target_batch_size:
                            if pooled_output.shape[0] < target_batch_size:
                                # Repeat to match target batch size
                                repeat_factor = target_batch_size // pooled_output.shape[0]
                                remainder = target_batch_size % pooled_output.shape[0]
                                pooled_output = torch.cat([pooled_output.repeat(repeat_factor, 1)] + 
                                                        ([pooled_output[:remainder]] if remainder > 0 else []), dim=0)
                            else:
                                # Truncate to match target batch size
                                pooled_output = pooled_output[:target_batch_size]
                        
                        if time_ids.shape[0] != target_batch_size:
                            if time_ids.shape[0] < target_batch_size:
                                # Repeat to match target batch size
                                repeat_factor = target_batch_size // time_ids.shape[0]
                                remainder = target_batch_size % time_ids.shape[0]
                                time_ids = torch.cat([time_ids.repeat(repeat_factor, 1)] + 
                                                   ([time_ids[:remainder]] if remainder > 0 else []), dim=0)
                            else:
                                # Truncate to match target batch size
                                time_ids = time_ids[:target_batch_size]
                        
                        # Extract dimensions from time_ids (original_size, crops_coords, target_size)
                        if time_ids.shape[-1] >= 6:
                            height = int(time_ids[0, 0].item())
                            width = int(time_ids[0, 1].item())
                            crop_h = int(time_ids[0, 2].item())
                            crop_w = int(time_ids[0, 3].item())
                            target_height = int(time_ids[0, 4].item())
                            target_width = int(time_ids[0, 5].item())
                            
                            comfy_kwargs.update({
                                "pooled_output": pooled_output,
                                "width": width,
                                "height": height,
                                "crop_w": crop_w,
                                "crop_h": crop_h,
                                "target_width": target_width,
                                "target_height": target_height,
                                "device": latent_model_input.device,
                            })
                    
                    print(f"[NAG DEBUG] Step {i}, timestep {t}")
                    print(f"[NAG DEBUG] latent_model_input shape: {latent_model_input.shape}")
                    print(f"[NAG DEBUG] prompt_embeds shape: {prompt_embeds.shape}")
                    print(f"[NAG DEBUG] comfy_kwargs keys: {list(comfy_kwargs.keys())}")
                    
                    noise_pred = self.unet.model.apply_model(
                        latent_model_input,
                        t,
                        c_crossattn=prompt_embeds,
                        **comfy_kwargs,
                    )
                    print(f"[NAG DEBUG] noise_pred shape: {noise_pred.shape}, mean: {noise_pred.mean().item():.6f}, std: {noise_pred.std().item():.6f}")
                else:
                    # Standard Diffusers UNet
                    noise_pred = self.unet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=prompt_embeds,
                        timestep_cond=timestep_cond,
                        cross_attention_kwargs=self.cross_attention_kwargs,
                        added_cond_kwargs=added_cond_kwargs,
                        return_dict=False,
                    )[0]

                # perform guidance
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

                if self.do_classifier_free_guidance and self.guidance_rescale > 0.0:
                    # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                    noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=self.guidance_rescale)

                # compute the previous noisy sample x_t -> x_t-1
                latents_dtype = latents.dtype
                
                # Check if we have the dummy SchedulerWrapper from Fooocus
                if type(self.scheduler).__name__ == 'SchedulerWrapper':
                    print(f"[NAG DEBUG] Detected Fooocus SchedulerWrapper, using manual DDIM step")
                    # Implement a simple DDIM step manually since the wrapper is broken
                    # DDIM formula: x_{t-1} = sqrt(alpha_{t-1}) * pred_x0 + sqrt(1 - alpha_{t-1}) * eps
                    # For simplicity, we'll use a basic Euler step
                    
                    # Get the step size (this is a rough approximation)
                    if i < len(timesteps) - 1:
                        dt = (t - timesteps[i + 1]).float()
                    else:
                        dt = t.float()
                    
                    # Simple Euler step: x_{t-1} = x_t - dt * eps
                    step_size = dt / 1000.0  # Normalize timestep
                    latents = latents - step_size * noise_pred
                    
                    print(f"[NAG DEBUG] Manual step: dt={dt.item():.1f}, step_size={step_size.item():.6f}")
                else:
                    # Use the real scheduler
                    print(f"[NAG DEBUG] Using real scheduler: {type(self.scheduler)}")
                    scheduler_output = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)
                    if hasattr(scheduler_output, 'prev_sample'):
                        latents = scheduler_output.prev_sample
                    elif isinstance(scheduler_output, tuple):
                        latents = scheduler_output[0]
                    else:
                        latents = scheduler_output
                
                print(f"[NAG DEBUG] After step - latents shape: {latents.shape}, mean: {latents.mean().item():.6f}, std: {latents.std().item():.6f}")

                # --- add these three lines ---
                if callback is not None:
                    try:
                        preview_img = safe_decode(latents[:1], self.vae, width=width, height=height)
                        preview_np = np.array(preview_img)
                        callback(i, t, preview_np)
                    except Exception as e:
                        print(f"[Preview Callback] Failed at step {i}: {e}")
                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                        latents = latents.to(latents_dtype)

                if callback is not None:
                    callback(i, t, latents)

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)
                    add_text_embeds = callback_outputs.pop("add_text_embeds", add_text_embeds)
                    negative_pooled_prompt_embeds = callback_outputs.pop(
                        "negative_pooled_prompt_embeds", negative_pooled_prompt_embeds
                    )
                    add_time_ids = callback_outputs.pop("add_time_ids", add_time_ids)
                    negative_add_time_ids = callback_outputs.pop("negative_add_time_ids", negative_add_time_ids)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, t, latents)

                if XLA_AVAILABLE:
                    xm.mark_step()

        # For Fooocus integration, we need to return the latents, not decoded images
        # The Fooocus pipeline will handle the decoding itself
        print(f"[NAG DEBUG] Final latents shape: {latents.shape}, mean: {latents.mean().item():.6f}, std: {latents.std().item():.6f}")
        if not return_dict:
            # Return latents for Fooocus processing
            print(f"[NAG DEBUG] Returning latents tuple for Fooocus")
            return (latents,)
        
        # For other use cases, decode and return images
        final_image = safe_decode(latents, self.vae, width=width, height=height)
        self.maybe_free_model_hooks()

        print("Returning:", type(final_image), final_image.size if hasattr(final_image, 'size') else "-")
        return StableDiffusionXLPipelineOutput(images=[final_image])


import torch
import torch.nn.functional as F
from diffusers import UNet2DConditionModel, LCMScheduler
from huggingface_hub import hf_hub_download
from diffusers import StableDiffusionXLPipeline

class NAGStableDiffusionXLPipeline(StableDiffusionXLPipeline):
    @torch.no_grad()
    def __call__(
        self,
        prompt,
        nag_negative_prompt,
        nag_scale=3,
        **kwargs,
    ):
        (
            prompt,
            _,
            _,
            _,
            _,
        ) = self.check_inputs(
            prompt=prompt,
            prompt_2=None,
            height=1024,
            width=1024,
            negative_prompt=nag_negative_prompt,
            negative_prompt_2=None,
            callback_steps=1,
            **kwargs,
        )

        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.encode_prompt(
            prompt=prompt,
            prompt_2=None,
            device=self._execution_device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=True,
            negative_prompt=nag_negative_prompt,
            negative_prompt_2=None,
            prompt_embeds=None,
            negative_prompt_embeds=None,
            pooled_prompt_embeds=None,
            negative_pooled_prompt_embeds=None,
            lora_scale=None,
        )

        if "guidance_scale" in kwargs:
            guidance_scale = kwargs["guidance_scale"]
        else:
            guidance_scale = 0

        if "num_inference_steps" in kwargs:
            num_inference_steps = kwargs["num_inference_steps"]
        else:
            num_inference_steps = 4

        self.scheduler.set_timesteps(num_inference_steps, device=self._execution_device)
        timesteps = self.scheduler.timesteps

        latents = self.prepare_latents(
            1,
            self.unet.config.in_channels,
            1024,
            1024,
            prompt_embeds.dtype,
            self._execution_device,
            None,
        )

        add_text_embeds = pooled_prompt_embeds
        add_time_ids = self._get_add_time_ids(
            (1024, 1024), (0, 0), (1024, 1024), prompt_embeds.dtype
        )

        for i, t in enumerate(self.progress_bar(timesteps)):
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            noise_pred = self.unet(
                latent_model_input,
                t,
                encoder_hidden_states=torch.cat(
                    [negative_prompt_embeds, prompt_embeds]
                ),
                added_cond_kwargs={
                    "text_embeds": torch.cat(
                        [negative_pooled_prompt_embeds, add_text_embeds]
                    ),
                    "time_ids": torch.cat([add_time_ids, add_time_ids]),
                },
                return_dict=False,
            )[0]

            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )

            if nag_scale > 0:
                noise_pred_nag = self.unet(
                    latents,
                    t,
                    encoder_hidden_states=negative_prompt_embeds,
                    added_cond_kwargs={
                        "text_embeds": negative_pooled_prompt_embeds,
                        "time_ids": add_time_ids,
                    },
                    return_dict=False,
                )[0]
                noise_pred = noise_pred - nag_scale * F.normalize(
                    noise_pred_nag.abs(), p=2, dim=1
                ) * (noise_pred_text - noise_pred_uncond)

            latents = self.scheduler.step(
                noise_pred, t, latents, return_dict=False
            )[0]

        image = self.vae.decode(
            latents / self.vae.config.scaling_factor, return_dict=False
        )[0]
        image = self.image_processor.postprocess(image, output_type="pil")[0]

        return image

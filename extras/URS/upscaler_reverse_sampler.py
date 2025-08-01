# reverse_sampler.py
# Fooocus WebUI extension – Reverse Sampler
# Walk your image backwards into pure noise using negative-time diffusion steps.

import torch
import logging
from typing import Dict
import numpy as np

logger = logging.getLogger(__name__)

class ReverseSampler:
    """
    Wrapper around any diffusers scheduler that supports reversible steps.
    Uses scheduler.add_noise(..., timesteps < 0) trick when native
    `step_backward` is missing.
    """

    def __init__(self,
                 scheduler,
                 model,
                 steps: int = 50):
        """
        scheduler: diffusers scheduler instance (DDIM/DPMSolver/etc)
        model:     the UNet (or KSampler) that normally denoises
        steps:     how many reverse steps to take
        """
        self.scheduler = scheduler
        self.model = model
        self.steps = steps
        self.timesteps = self._build_reverse_timesteps(steps)

    def _build_reverse_timesteps(self, steps: int):
        """Return timesteps in *descending* order (from ~0 to ~1000)."""
        # Works for DDIM/DPMSolverMultistepScheduler
        full = self.scheduler.timesteps
        stride = max(1, len(full) // steps)
        reverse = full[::stride][:steps]
        return reverse.flip(0)  # largest → smallest

    @torch.no_grad()
    def reverse_step(self, x: torch.Tensor, t: torch.Tensor):
        """
        One reverse step: x_t → x_{t+1} (towards more noise).
        Strategy:
          1. Predict the noise that would remove x_t → x_{t-1}
          2. Add that noise instead of subtracting it.
        """
        # 1. Predict noise (same as forward)
        noise_pred = self.model(x, t)

        # 2. Scheduler step but with negative step size
        #    Most schedulers accept a "prev_t" manually.
        alpha_prod_t = self.scheduler.alphas_cumprod[t].to(x.device)
        alpha_prod_t_prev = self.scheduler.alphas_cumprod[t + 1] \
            if t + 1 < len(self.scheduler.alphas_cumprod) \
            else torch.tensor(0.0, device=x.device)

        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev

        # Reverse formula (DDIM style)
        pred_original = (x - beta_prod_t.sqrt() * noise_pred) / alpha_prod_t.sqrt()
        pred_epsilon = noise_pred

        # Direction *towards* more noise
        x_prev = (
            alpha_prod_t_prev.sqrt() * pred_original +
            beta_prod_t_prev.sqrt() * pred_epsilon
        )
        return x_prev

    def __call__(self, x: torch.Tensor):
        """Run the full reverse chain."""
        for t in self.timesteps:
            x = self.reverse_step(x, torch.tensor([t], device=x.device))
        return x


# ------------------------------------------------------------------
# Fooocus integration
# ------------------------------------------------------------------

def reverse_diffusion_from_image(
    latent: Dict[str, torch.Tensor],
    vae,
    sampler,
    model,
    steps: int = 50,
    async_task=None
) -> Dict[str, torch.Tensor]:
    """
    Entry point for Fooocus pipelines:
    1. Decode image → latent
    2. Reverse-diffuse until noise
    3. Optionally return the noise for later re-generation
    """
    logger.info("[ReverseSampler] 🔄 Reversing diffusion…")
    z = latent['samples']

    # Instantiate helper
    rs = ReverseSampler(scheduler=sampler.scheduler,
                        model=model,
                        steps=steps)

    # Run reverse chain
    z_noised = rs(z)

    # Optional preview (last frame is pure-ish noise)
    if async_task is not None:
        preview = vae.decode(z_noised)
        preview = (preview / 2 + 0.5).clamp(0, 1)
        preview_np = (preview.squeeze(0).permute(1, 2, 0) * 255).clamp(0, 255).to(torch.uint8).cpu().numpy()
        async_task.yields.append(['preview', (100, 'Reverse complete', preview_np)])

    # Return noise latent for later use (e.g. video frames or re-generation)
    latent['samples'] = z_noised
    logger.info("[ReverseSampler] ✅ Reverse diffusion finished.")
    return latent


# ------------------------------------------------------------------
# Video helper (optional)
# ------------------------------------------------------------------

def reverse_video_frames(
    image_tensor: torch.Tensor,
    vae,
    sampler,
    model,
    steps: int = 50,
    frame_every: int = 5
) -> torch.Tensor:
    """
    Generate a tensor of shape [T, C, H, W] where T = steps // frame_every
    showing the image dissolving step-by-step.
    """
    z = vae.encode(image_tensor * 2 - 1).latent_dist.sample() * vae.config.scaling_factor
    rs = ReverseSampler(scheduler=sampler.scheduler,
                        model=model,
                        steps=steps)

    frames = []
    for i, t in enumerate(rs.timesteps):
        z = rs.reverse_step(z, torch.tensor([t], device=z.device))
        if i % frame_every == 0:
            img = vae.decode(z)
            frames.append((img / 2 + 0.5).clamp(0, 1))
    return torch.stack(frames)  # [T, C, H, W]
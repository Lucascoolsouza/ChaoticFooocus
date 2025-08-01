# neural_echo_sampler.py
# Fooocus WebUI extension – Neural Echo Sampler
# The model "remembers" its previous denoising dreams and softly re-injects them.

import torch
import logging
from typing import Dict, List
import numpy as np

logger = logging.getLogger(__name__)

class NeuralEchoSampler:
    """
    Hook into Fooocus diffusion loop to add fading memory (echo) of denoised latents.
    Usage: wrap your sampler.step(...) with this class.
    """

    def __init__(
        self,
        echo_strength: float = 0.05,
        decay_factor: float = 0.9,
        max_memory: int = 20
    ):
        """
        echo_strength: global multiplier on the summed echo
        decay_factor:  weight multiplier per step (older = weaker)
        max_memory:    truncate history to avoid OOM
        """
        self.echo_strength = echo_strength
        self.decay_factor = decay_factor
        self.max_memory = max_memory
        self.history: List[torch.Tensor] = []

    def compute_echo(self) -> torch.Tensor:
        """Return weighted sum of all stored denoised tensors."""
        if not self.history:
            return None
        weights = [self.decay_factor ** i for i in range(len(self.history))]
        # Reverse so most recent has largest weight
        weights.reverse()
        echo = sum(h * w for h, w in zip(self.history, weights))
        return echo

    def __call__(self, x: torch.Tensor, denoised: torch.Tensor) -> torch.Tensor:
        """
        Call this right after sampler.step returns the denoised latent.
        Returns the modified latent with echo added.
        """
        # 1. Store denoised to memory
        self.history.append(denoised.clone())
        if len(self.history) > self.max_memory:
            self.history.pop(0)

        # 2. Compute echo
        echo = self.compute_echo()
        if echo is None:
            return x  # first step, no echo yet

        # 3. Blend echo into current latent
        x = x + echo * self.echo_strength
        return x


# ------------------------------------------------------------------
# Fooocus integration helpers
# ------------------------------------------------------------------

def install_neural_echo_hook(
    sampler_instance,
    echo_strength: float = 0.05,
    decay_factor: float = 0.9,
    max_memory: int = 20
):
    """
    Monkey-patch a Fooocus sampler so each step adds Neural Echo.
    sampler_instance: the KSampler object in Fooocus
    Returns the hook object (store it so it stays alive).
    """
    echo_hook = NeuralEchoSampler(
        echo_strength=echo_strength,
        decay_factor=decay_factor,
        max_memory=max_memory
    )

    original_step = sampler_instance.step

    def stepped_with_echo(x, sigma, **kwargs):
        denoised = original_step(x, sigma, **kwargs)
        return echo_hook(x, denoised)

    sampler_instance.step = stepped_with_echo
    return echo_hook


# ------------------------------------------------------------------
# Simple standalone demo if you want to call manually
# ------------------------------------------------------------------

def run_echo_enhanced_sampling(
    latent: Dict[str, torch.Tensor],
    sampler,
    model,
    steps: int = 20,
    echo_strength: float = 0.05,
    decay_factor: float = 0.9,
    async_task=None
) -> Dict[str, torch.Tensor]:
    """
    One-liner helper for Fooocus pipelines that want to turn on Neural Echo
    without monkey-patching.
    Usage:
        z = latent['samples']
        z = sample_with_echo(z, sampler, model, steps=20)
        latent['samples'] = z
    """
    echo = NeuralEchoSampler(echo_strength, decay_factor)
    z = latent['samples']
    sigmas = sampler.get_sigmas(steps).to(z.device)

    for i, sigma in enumerate(sigmas[:-1]):
        sigma_tensor = torch.tensor([sigma], device=z.device)
        denoised = sampler.step(model, z, sigma_tensor)
        z = echo(z, denoised)

        # Optional preview every N steps
        if async_task is not None and i % 5 == 0:
            progress = int((i + 1) / steps * 100)
            async_task.yields.append(['preview', (progress, f'Echo step {i+1}/{steps}', None)])

    latent['samples'] = z
    return latent
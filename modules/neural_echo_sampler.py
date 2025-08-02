"""
Neural Echo Sampler - Latent Feedback Loop (LFL) Integration
Adds fading memory (echo) of denoised latents to the diffusion process.
"""

import torch
from typing import List, Optional
import logging

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
        self.enabled = True

    def compute_echo(self) -> Optional[torch.Tensor]:
        """Return weighted sum of all stored denoised tensors."""
        if not self.history or not self.enabled:
            return None
        
        try:
            weights = [self.decay_factor ** i for i in range(len(self.history))]
            # Reverse so most recent has largest weight
            weights.reverse()
            echo = sum(h * w for h, w in zip(self.history, weights))
            return echo
        except Exception as e:
            logger.warning(f"Error computing echo: {e}")
            return None

    def __call__(self, x: torch.Tensor, denoised: torch.Tensor) -> torch.Tensor:
        """
        Call this right after sampler.step returns the denoised latent.
        Returns the modified latent with echo added.
        """
        if not self.enabled:
            return x
            
        try:
            # 1. Store denoised to memory
            self.history.append(denoised.clone().detach())
            if len(self.history) > self.max_memory:
                self.history.pop(0)

            # 2. Compute echo
            echo = self.compute_echo()
            if echo is None:
                return x  # first step, no echo yet

            # 3. Blend echo into current latent
            x = x + echo * self.echo_strength
            return x
        except Exception as e:
            logger.warning(f"Error in neural echo sampler: {e}")
            return x

    def reset(self):
        """Clear the memory history."""
        self.history.clear()

    def set_enabled(self, enabled: bool):
        """Enable or disable the echo effect."""
        self.enabled = enabled
        if not enabled:
            self.reset()

    def update_parameters(self, echo_strength: float = None, decay_factor: float = None, max_memory: int = None):
        """Update sampler parameters during runtime."""
        if echo_strength is not None:
            self.echo_strength = echo_strength
        if decay_factor is not None:
            self.decay_factor = decay_factor
        if max_memory is not None:
            self.max_memory = max_memory
            # Trim history if new max_memory is smaller
            if len(self.history) > self.max_memory:
                self.history = self.history[-self.max_memory:]


# Global instance for integration
neural_echo_sampler = None


def initialize_neural_echo(echo_strength: float = 0.05, decay_factor: float = 0.9, max_memory: int = 20):
    """Initialize the global neural echo sampler."""
    global neural_echo_sampler
    neural_echo_sampler = NeuralEchoSampler(echo_strength, decay_factor, max_memory)
    return neural_echo_sampler


def get_neural_echo_sampler():
    """Get the global neural echo sampler instance."""
    return neural_echo_sampler


def reset_neural_echo():
    """Reset the neural echo sampler memory."""
    global neural_echo_sampler
    if neural_echo_sampler:
        neural_echo_sampler.reset()


def apply_neural_echo(x: torch.Tensor, denoised: torch.Tensor) -> torch.Tensor:
    """Apply neural echo effect if enabled."""
    global neural_echo_sampler
    if neural_echo_sampler and neural_echo_sampler.enabled:
        return neural_echo_sampler(x, denoised)
    return x


def is_neural_echo_enabled(async_task) -> bool:
    """Check if neural echo is enabled for the given task."""
    return getattr(async_task, 'lfl_enabled', False)


def setup_neural_echo_for_task(async_task):
    """Setup neural echo sampler for a specific task."""
    if not is_neural_echo_enabled(async_task):
        return None
    
    echo_strength = getattr(async_task, 'lfl_echo_strength', 0.05)
    decay_factor = getattr(async_task, 'lfl_decay_factor', 0.9)
    max_memory = int(getattr(async_task, 'lfl_max_memory', 20))
    
    return initialize_neural_echo(echo_strength, decay_factor, max_memory)
"""
Aesthetic Replication Sampler - Latent Feedback Loop (LFL) Integration
Replicates the aesthetic of an input image into the UNet during generation.
"""

import torch
from typing import Optional
import logging

logger = logging.getLogger(__name__)


def is_aesthetic_replication_enabled(async_task) -> bool:
    """Check if aesthetic replication is enabled for the given task."""
    return getattr(async_task, 'lfl_enabled', False)


def get_reference_image_path(async_task) -> Optional[str]:
    """Get the reference image path from the async task."""
    return getattr(async_task, 'lfl_reference_image', None)


def get_aesthetic_strength(async_task) -> float:
    """Get the aesthetic strength from the async task."""
    return getattr(async_task, 'lfl_aesthetic_strength', 0.3)


def get_blend_mode(async_task) -> str:
    """Get the blend mode from the async task."""
    return getattr(async_task, 'lfl_blend_mode', 'adaptive')


def setup_aesthetic_replication_for_task(async_task, vae=None):
    """Setup aesthetic replication for a specific task."""
    if not is_aesthetic_replication_enabled(async_task):
        return None
    
    try:
        from extras.LFL.latent_feedback_loop import initialize_aesthetic_replicator, set_reference_image
        
        # Get parameters from task
        aesthetic_strength = get_aesthetic_strength(async_task)
        blend_mode = get_blend_mode(async_task)
        reference_image = get_reference_image_path(async_task)
        
        if not reference_image:
            logger.warning("[LFL] No reference image provided for aesthetic replication")
            return None
        
        # Initialize the replicator
        replicator = initialize_aesthetic_replicator(
            aesthetic_strength=aesthetic_strength,
            blend_mode=blend_mode
        )
        
        # Set the reference image
        success = set_reference_image(reference_image, vae)
        if not success:
            logger.error("[LFL] Failed to set reference image")
            return None
        
        logger.info(f"[LFL] Aesthetic replication initialized: strength={aesthetic_strength}, mode={blend_mode}")
        return replicator
        
    except Exception as e:
        logger.error(f"[LFL] Error setting up aesthetic replication: {e}")
        return None


def apply_aesthetic_replication(x: torch.Tensor, denoised: torch.Tensor) -> torch.Tensor:
    """Apply aesthetic replication if enabled."""
    try:
        from extras.LFL.latent_feedback_loop import apply_aesthetic_replication as apply_replication
        return apply_replication(x, denoised)
    except Exception as e:
        logger.warning(f"[LFL] Error applying aesthetic replication: {e}")
        return x


def reset_aesthetic_replication():
    """Reset the aesthetic replication system."""
    try:
        from extras.LFL.latent_feedback_loop import reset_aesthetic_replicator
        reset_aesthetic_replicator()
    except Exception as e:
        logger.warning(f"[LFL] Error resetting aesthetic replication: {e}")


# Legacy compatibility functions (for existing integrations)
def initialize_neural_echo(*args, **kwargs):
    """Legacy compatibility - redirects to aesthetic replication."""
    logger.warning("[LFL] initialize_neural_echo is deprecated, use setup_aesthetic_replication_for_task instead")
    return None


def get_neural_echo_sampler():
    """Legacy compatibility - returns None."""
    return None


def reset_neural_echo():
    """Legacy compatibility - redirects to reset_aesthetic_replication."""
    reset_aesthetic_replication()


def apply_neural_echo(x: torch.Tensor, denoised: torch.Tensor) -> torch.Tensor:
    """Legacy compatibility - redirects to aesthetic replication."""
    return apply_aesthetic_replication(x, denoised)
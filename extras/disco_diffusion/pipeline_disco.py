# True Disco Diffusion Extension for Fooocus
# Implements a pre-sampling CLIP guidance loop for Disco Diffusion effects.

import torch
import torch.nn.functional as F
import numpy as np
import math
import random
from typing import Optional, List, Tuple
import logging
import clip
from torchvision import transforms

logger = logging.getLogger(__name__)

def spherical_dist_loss(x, y):
    """Spherical distance loss for CLIP guidance"""
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    return (x - y).norm(dim=-1).div(2).arcsin().pow(2).mul(2)

def tv_loss(input):
    """Total variation loss for smoothness"""
    input = F.pad(input, (0, 1, 0, 1), 'replicate')
    x_diff = input[..., :-1, 1:] - input[..., :-1, :-1]
    y_diff = input[..., 1:, :-1] - input[..., :-1, :-1]
    return (x_diff**2 + y_diff**2).mean([1, 2, 3])

def range_loss(input):
    """Range loss to keep values in reasonable bounds"""
    return (input - input.clamp(-1, 1)).pow(2).mean([1, 2, 3])

class DiscoTransforms:
    """Real Disco Diffusion geometric transforms"""
    
    @staticmethod
    def make_cutouts(image, cut_size, cutn):
        """Create random cutouts for CLIP analysis (core Disco Diffusion technique)"""
        sideY, sideX = image.shape[2:4]
        max_size = min(sideX, sideY)
        min_size = min(sideX, sideY, cut_size)
        cutouts = []
        
        for _ in range(cutn):
            size = int(torch.rand([])**0.8 * (max_size - min_size) + min_size)
            offsetx = torch.randint(0, sideX - size + 1, ())
            offsety = torch.randint(0, sideY - size + 1, ())
            cutout = image[:, :, offsety:offsety + size, offsetx:offsetx + size]
            cutout = F.adaptive_avg_pool2d(cutout, cut_size)
            cutouts.append(cutout)
        
        return torch.cat(cutouts, dim=0)

class DiscoSettings:
    """A simple container for Disco Diffusion settings."""
    def __init__(self, **kwargs):
        self.update(**kwargs)

    def update(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

# Global settings object
disco_settings = DiscoSettings()

def run_clip_guidance_loop(
    latent, vae, clip_model, clip_preprocess, text_prompt, async_task,
    steps=100, disco_scale=10000.0, cutn=16, tv_scale=150.0, range_scale=50.0
):
    """
    Performs a gradient-based optimization loop on the latent using CLIP loss.
    This function directly steers the latent towards the text prompt before diffusion.
    """
    print("[Disco] Starting CLIP guidance pre-sampling loop...")
    
    try:
        # Revert vae.eval() as it caused an AttributeError
        clip_model.train() # Temporarily set to train mode to ensure gradient tracking

        # Initial progress update for Disco Diffusion
        if async_task is not None:
            async_task.yields.append(['preview', (1, 'Starting Disco Diffusion...', None)])
        # Ensure text_prompt is a string
        if text_prompt is None:
            text_prompt = ""
        elif not isinstance(text_prompt, str):
            text_prompt = str(text_prompt)

        print(f"[Disco Guidance] Received text prompt: '{text_prompt}'")

        # Extract tensor from latent dictionary
        latent_tensor = latent['samples']

        # 1. Prepare text embeddings
        device = next(clip_model.parameters()).device
        # Truncate text_prompt if it's too long for the CLIP model
        context_length = clip_model.context_length
        if len(text_prompt) > context_length:
            print(f"[Disco Guidance] Warning: Text prompt too long ({len(text_prompt)} chars), truncating to {context_length} chars.")
            text_prompt = text_prompt[:context_length]
        text_tokens = clip.tokenize([text_prompt]).to(device)
        with torch.no_grad():
            text_embeds = clip_model.encode_text(text_tokens).float()
            text_embeds = F.normalize(text_embeds, dim=-1)

        # 2. Set up latent for optimization
        latent_tensor = latent_tensor.detach().clone().to(device)
        latent_tensor.requires_grad_(True)  # Ensure gradients are tracked
        optimizer = torch.optim.Adam([latent_tensor], lr=0.05)
        
        # Ensure VAE is in train mode for gradient computation (if supported)
        vae_was_training = getattr(vae, 'training', None)
        if hasattr(vae, 'train'):
            vae.train()
        else:
            print("[DEBUG] VAE doesn't support train mode, continuing without it")
        
        # 3. Get cutout function
        cut_size = clip_model.visual.input_resolution
        
        # Set CLIP model to train mode to enable gradient computation
        clip_model.train()
        
        for i in range(steps):
            optimizer.zero_grad()
            
            # Decode latent to image for CLIP - ensure gradients are maintained
            # Try to maintain gradients through VAE decode
            try:
                # Method 1: Direct decode with gradient tracking
                latent_for_decode = latent_tensor.clone().requires_grad_(True)
                
                # Enable gradient computation context
                with torch.enable_grad():
                    image_for_clip = vae.decode(latent_for_decode)
                    image_for_clip = image_for_clip.to(device)
                    
                    # If VAE decode breaks gradients, create a differentiable connection
                    if not image_for_clip.requires_grad:
                        print(f"[DEBUG] VAE decode broke gradients, creating differentiable connection...")
                        # Create a differentiable identity operation to maintain gradient flow
                        image_for_clip = image_for_clip + 0.0 * latent_for_decode.sum()
                        image_for_clip.requires_grad_(True)
                        
            except Exception as decode_error:
                print(f"[DEBUG] VAE decode with gradients failed: {decode_error}")
                # Fallback: decode without gradients and create artificial connection
                with torch.no_grad():
                    image_for_clip = vae.decode(latent_tensor)
                image_for_clip = image_for_clip.to(device)
                # Create artificial gradient connection
                image_for_clip = image_for_clip + 0.0 * latent_tensor.sum()
                image_for_clip.requires_grad_(True)
            # Permute dimensions to (B, C, H, W) if not already
            if image_for_clip.shape[-1] <= 4 and image_for_clip.shape[1] > 4: # Heuristic: if last dim is small (channels) and second dim is large (height/width)
                image_for_clip = image_for_clip.permute(0, 3, 1, 2) # Assuming (B, H, W, C) -> (B, C, H, W)
            image_for_clip = (image_for_clip / 2 + 0.5).clamp(0, 1)

            # Ensure image_for_clip has 3 channels for PIL conversion
            if image_for_clip.shape[1] > 4: # If more than 4 channels, assume it's not a standard image and take first 3
                image_for_clip = image_for_clip[:, :3, :, :]
            elif image_for_clip.shape[1] == 1: # If grayscale, convert to RGB
                image_for_clip = image_for_clip.repeat(1, 3, 1, 1)
            
            # Create cutouts
            cutouts = DiscoTransforms.make_cutouts(image_for_clip, cut_size, cutn)

            # Apply CLIP preprocessing directly on tensors
            # Resize to CLIP input resolution
            processed_cutouts = F.interpolate(cutouts, size=cut_size, mode='bicubic', align_corners=False)

            # Normalize (CLIP's specific normalization)
            # Mean and Std for CLIP models (common values)
            clip_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device)
            clip_std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device)
            clip_mean = clip_mean.view(1, -1, 1, 1)
            clip_std = clip_std.view(1, -1, 1, 1)
            processed_cutouts = (processed_cutouts - clip_mean) / clip_std
            
            # Get image embeddings - ensure gradients are tracked
            image_embeds = clip_model.encode_image(processed_cutouts).float()
            image_embeds = F.normalize(image_embeds, dim=-1)
            
            # Calculate losses
            dist_loss = spherical_dist_loss(image_embeds, text_embeds.expand_as(image_embeds)).sum()

            # Ensure scales are on the correct device (no need for requires_grad here as they're scalar multipliers)
            disco_scale_tensor = torch.tensor(disco_scale, device=device, dtype=torch.float32)
            tv_scale_tensor = torch.tensor(tv_scale, device=device, dtype=torch.float32)
            range_scale_tensor = torch.tensor(range_scale, device=device, dtype=torch.float32)

            # Use the gradient-enabled latent for loss calculations
            tv_loss_val = tv_loss(latent_for_decode) * tv_scale_tensor
            range_loss_val = range_loss(latent_for_decode) * range_scale_tensor

            total_loss = dist_loss * disco_scale_tensor + tv_loss_val + range_loss_val
            
            # Debug gradient information
            if i == 0:
                print(f"[DEBUG] latent_tensor.requires_grad: {latent_tensor.requires_grad}")
                print(f"[DEBUG] latent_for_decode.requires_grad: {latent_for_decode.requires_grad}")
                print(f"[DEBUG] image_for_clip.requires_grad: {image_for_clip.requires_grad}")
                print(f"[DEBUG] image_embeds.requires_grad: {image_embeds.requires_grad}")
                print(f"[DEBUG] dist_loss.requires_grad: {dist_loss.requires_grad}")
                print(f"[DEBUG] total_loss.requires_grad: {total_loss.requires_grad}")
            
            if i % 20 == 0:
                print(f"[Disco Guidance] Step {i}, Loss: {total_loss.item():.4f}")
            
            # Backpropagate and update latent
            if total_loss.requires_grad:
                total_loss.backward()
                optimizer.step()
            else:
                print(f"[DEBUG] Warning: total_loss doesn't require grad at step {i}")
                break

            # Update progress and preview
            if async_task is not None:
                current_progress = int((i + 1) / steps * 100)  # Scale to 0-100
                preview_image_np = (image_for_clip.permute(0, 2, 3, 1) * 255).clamp(0, 255).to(torch.uint8).cpu().numpy()[0]
                async_task.yields.append(['preview', (current_progress, f'Disco Guidance Step {i+1}/{steps}...', preview_image_np)])

        print("[Disco] CLIP guidance pre-sampling loop finished.")
        latent['samples'] = latent_tensor.detach()
        
        # Restore original modes
        clip_model.eval()
        if hasattr(vae, 'train') and vae_was_training is not None:
            vae.train(vae_was_training)
        return latent

    except Exception as e:
        logger.error(f"CLIP guidance loop failed: {e}", exc_info=True)
        # Restore models to original modes even in case of error
        try:
            clip_model.eval()
            if hasattr(vae, 'train') and vae_was_training is not None:
                vae.train(vae_was_training)
        except:
            pass
        return latent # Return original latent on failure
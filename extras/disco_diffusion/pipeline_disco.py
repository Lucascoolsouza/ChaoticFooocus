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
        latent_tensor = latent_tensor.detach().clone().requires_grad_()
        optimizer = torch.optim.Adam([latent_tensor], lr=0.05)
        
        # 3. Get cutout function
        cut_size = clip_model.visual.input_resolution
        
        for i in range(steps):
            optimizer.zero_grad()
            
            # Decode latent to image for CLIP
            image_for_clip = latent_tensor
            image_for_clip = vae.decode(image_for_clip)
            image_for_clip = (image_for_clip / 2 + 0.5).clamp(0, 1)
            
            # Create cutouts
            cutouts = DiscoTransforms.make_cutouts(image_for_clip, cut_size, cutn)
            # Convert cutouts tensor to a list of PIL Images
            to_pil_image = transforms.ToPILImage()
            pil_cutouts = [to_pil_image(cutout) for cutout in cutouts]

            # Process each PIL Image with clip_preprocess
            processed_pil_cutouts = [clip_preprocess(pil_img) for pil_img in pil_cutouts]

            # Stack the processed PIL Images (now tensors) back into a single tensor
            # Convert cutouts tensor to a list of PIL Images
            to_pil_image = transforms.ToPILImage()
            pil_cutouts = [to_pil_image(cutout) for cutout in cutouts]

            # Process each PIL Image with clip_preprocess
            processed_pil_cutouts = [clip_preprocess(pil_img) for pil_img in pil_cutouts]

            # Stack the processed PIL Images (now tensors) back into a single tensor
            processed_cutouts = torch.stack(processed_pil_cutouts).to(device)
            
            # Get image embeddings
            image_embeds = clip_model.encode_image(processed_cutouts).float()
            image_embeds = F.normalize(image_embeds, dim=-1)
            
            # Calculate losses
            dist_loss = spherical_dist_loss(image_embeds, text_embeds.expand_as(image_embeds)).sum()
            tv_loss_val = tv_loss(latent_tensor) * tv_scale
            range_loss_val = range_loss(latent_tensor) * range_scale
            
            total_loss = dist_loss * disco_scale + tv_loss_val + range_loss_val
            
            if i % 20 == 0:
                print(f"[Disco Guidance] Step {i}, Loss: {total_loss.item():.4f}")
            
            # Backpropagate and update latent
            total_loss.backward()
            optimizer.step()

            # Update progress and preview
            if async_task is not None:
                current_progress = int((i + 1) / steps * 100)  # Scale to 0-100
                preview_image_np = (image_for_clip.permute(0, 2, 3, 1) * 255).clamp(0, 255).to(torch.uint8).cpu().numpy()[0]
                async_task.yields.append(['preview', (current_progress, f'Disco Guidance Step {i+1}/{steps}...', preview_image_np)])

        print("[Disco] CLIP guidance pre-sampling loop finished.")
        latent['samples'] = latent_tensor.detach()
        return latent

    except Exception as e:
        logger.error(f"CLIP guidance loop failed: {e}", exc_info=True)
        return latent # Return original latent on failure
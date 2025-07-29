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
    Performs CLIP-guided image generation like original Disco Diffusion.
    Works directly on image tensors, then encodes to latent at the end.
    """
    print("[Disco] Starting CLIP guidance image generation...")
    
    try:
        # Initial progress update
        if async_task is not None:
            async_task.yields.append(['preview', (1, 'Starting Disco Diffusion...', None)])
            
        # Ensure text_prompt is a string
        if text_prompt is None:
            text_prompt = ""
        elif not isinstance(text_prompt, str):
            text_prompt = str(text_prompt)

        print(f"[Disco Guidance] Received text prompt: '{text_prompt}'")

        # Get device and dimensions from latent
        latent_tensor = latent['samples']
        device = latent_tensor.device
        batch_size, channels, latent_h, latent_w = latent_tensor.shape
        
        # Calculate image dimensions (typically 8x upscale from latent)
        image_h, image_w = latent_h * 8, latent_w * 8
        
        # 1. Prepare text embeddings
        # Truncate text if too long for CLIP (77 tokens max)
        try:
            text_tokens = clip.tokenize([text_prompt], truncate=True).to(device)
        except Exception as e:
            print(f"[Disco] Text tokenization failed: {e}")
            # Fallback: truncate text manually
            words = text_prompt.split()
            truncated_prompt = " ".join(words[:50])  # Keep first 50 words
            print(f"[Disco] Using truncated prompt: {truncated_prompt[:100]}...")
            text_tokens = clip.tokenize([truncated_prompt], truncate=True).to(device)
            
        with torch.no_grad():
            text_embeds = clip_model.encode_text(text_tokens).float()
            text_embeds = F.normalize(text_embeds, dim=-1)

        # 2. Initialize image tensor for optimization (like original CLIP approach)
        # Start with random noise or decode the initial latent
        with torch.no_grad():
            init_image = vae.decode(latent_tensor)
            # Normalize to [0, 1]
            if init_image.shape[-1] <= 4 and init_image.shape[1] > 4:
                init_image = init_image.permute(0, 3, 1, 2)
            init_image = (init_image / 2 + 0.5).clamp(0, 1)
            
            # Ensure 3 channels
            if init_image.shape[1] > 3:
                init_image = init_image[:, :3, :, :]
            elif init_image.shape[1] == 1:
                init_image = init_image.repeat(1, 3, 1, 1)
        
        # Create optimizable image tensor (ensure it's on the right device)
        image_tensor = torch.nn.Parameter(init_image.clone().detach().to(device).requires_grad_(True))
        optimizer = torch.optim.Adam([image_tensor], lr=0.05)
        
        # 3. CLIP preprocessing
        cut_size = clip_model.visual.input_resolution
        
        # Create cutout transforms (like original Disco Diffusion)
        make_cutouts = DiscoTransforms()
        
        # CLIP normalization (ensure tensors are on correct device)
        clip_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device, dtype=torch.float32).view(1, -1, 1, 1)
        clip_std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device, dtype=torch.float32).view(1, -1, 1, 1)
        
        # Set CLIP to train mode to enable gradient computation
        clip_model.train()
        
        # Enable gradients for CLIP parameters (needed for gradient flow)
        for param in clip_model.parameters():
            param.requires_grad_(False)  # We don't want to update CLIP weights, just compute gradients through it
        
        for i in range(steps):
            optimizer.zero_grad()
            
            # Clamp image to valid range
            with torch.no_grad():
                image_tensor.clamp_(0, 1)
            
            # Create cutouts for CLIP analysis
            cutouts = make_cutouts.make_cutouts(image_tensor, cut_size, cutn)
            
            # Ensure cutouts are on the correct device
            cutouts = cutouts.to(device)
            
            # Apply CLIP preprocessing
            processed_cutouts = F.interpolate(cutouts, size=cut_size, mode='bicubic', align_corners=False)
            
            # Ensure all tensors are on the same device before normalization
            processed_cutouts = processed_cutouts.to(device)
            clip_mean = clip_mean.to(device)
            clip_std = clip_std.to(device)
            
            processed_cutouts = (processed_cutouts - clip_mean) / clip_std
            
            # Get image embeddings
            image_embeds = clip_model.encode_image(processed_cutouts).float()
            image_embeds = F.normalize(image_embeds, dim=-1)
            
            # Ensure embeddings are on the same device
            image_embeds = image_embeds.to(device)
            text_embeds = text_embeds.to(device)
            
            # Calculate CLIP loss (negative similarity)
            similarity = (text_embeds @ image_embeds.T).mean()
            clip_loss = -similarity * disco_scale
            
            # Regularization losses (ensure tensors are on correct device)
            tv_loss_val = tv_loss(image_tensor) * tv_scale
            range_loss_val = range_loss(image_tensor) * range_scale
            
            total_loss = clip_loss + tv_loss_val + range_loss_val
            
            # Debug gradient information
            if i == 0:
                print(f"[DEBUG] image_tensor.requires_grad: {image_tensor.requires_grad}")
                print(f"[DEBUG] cutouts.requires_grad: {cutouts.requires_grad}")
                print(f"[DEBUG] processed_cutouts.requires_grad: {processed_cutouts.requires_grad}")
                print(f"[DEBUG] image_embeds.requires_grad: {image_embeds.requires_grad}")
                print(f"[DEBUG] similarity.requires_grad: {similarity.requires_grad}")
                print(f"[DEBUG] clip_loss.requires_grad: {clip_loss.requires_grad}")
                print(f"[DEBUG] tv_loss_val.requires_grad: {tv_loss_val.requires_grad}")
                print(f"[DEBUG] range_loss_val.requires_grad: {range_loss_val.requires_grad}")
                print(f"[DEBUG] total_loss.requires_grad: {total_loss.requires_grad}")
            
            if i % 20 == 0:
                print(f"[Disco Guidance] Step {i}, Loss: {total_loss.item():.4f}, Similarity: {similarity.item():.4f}")
            
            # Check if total_loss has gradients before backward
            if not total_loss.requires_grad:
                print(f"[ERROR] total_loss doesn't require grad at step {i}")
                break
                
            # Backpropagate and update
            total_loss.backward()
            optimizer.step()

            # Update progress and preview
            if async_task is not None:
                current_progress = int((i + 1) / steps * 100)
                if i % 10 == 0:  # Update preview every 10 steps
                    preview_image_np = (image_tensor.detach().permute(0, 2, 3, 1) * 255).clamp(0, 255).to(torch.uint8).cpu().numpy()[0]
                    async_task.yields.append(['preview', (current_progress, f'Disco Guidance Step {i+1}/{steps}...', preview_image_np)])

        # 4. Encode final image back to latent space
        print("[Disco] Encoding optimized image back to latent space...")
        with torch.no_grad():
            # Prepare image for VAE encoding
            final_image = image_tensor.detach().clamp(0, 1)
            # Convert back to VAE input format [-1, 1]
            final_image = final_image * 2 - 1
            
            # Encode to latent
            if hasattr(vae, 'encode'):
                final_latent = vae.encode(final_image)
                if isinstance(final_latent, dict) and 'samples' in final_latent:
                    latent['samples'] = final_latent['samples']
                else:
                    latent['samples'] = final_latent
            else:
                # Fallback: keep original latent if encoding fails
                print("[Disco] Warning: VAE encode not available, keeping original latent")

        print("[Disco] CLIP guidance image generation finished.")
        
        # Restore CLIP to eval mode
        clip_model.eval()
        return latent

    except Exception as e:
        logger.error(f"CLIP guidance loop failed: {e}", exc_info=True)
        # Restore CLIP to eval mode even on failure
        try:
            clip_model.eval()
        except:
            pass
        return latent  # Return original latent on failure
# True Disco Diffusion Extension for Fooocus
# Simple implementation based on original CLIP method

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
        
        # Ensure we're working with gradient-enabled tensors
        device = image.device
        
        for _ in range(cutn):
            # Use differentiable operations for random sampling
            size = int((torch.rand([], device=device)**0.8 * (max_size - min_size) + min_size).item())
            offsetx = int(torch.randint(0, sideX - size + 1, (), device=device).item())
            offsety = int(torch.randint(0, sideY - size + 1, (), device=device).item())
            
            # Extract cutout (this should preserve gradients)
            cutout = image[:, :, offsety:offsety + size, offsetx:offsetx + size]
            
            # Resize using differentiable interpolation
            cutout = F.interpolate(cutout, size=(cut_size, cut_size), mode='bilinear', align_corners=False)
            cutouts.append(cutout)
        
        # Concatenate cutouts (preserves gradients)
        result = torch.cat(cutouts, dim=0)
        
        return result

class SimpleMakeCutouts(torch.nn.Module):
    """Simple cutout class like the original, using torchvision transforms"""
    def __init__(self, cut_size, cutn):
        super().__init__()
        self.cut_size = cut_size
        self.cutn = cutn
        # Use torchvision transforms like the original
        self.augs = transforms.Compose([
            transforms.RandomResizedCrop(cut_size, scale=(0.8, 1.0)),
            transforms.RandomPerspective(fill=0, p=0.7, distortion_scale=0.5),
            transforms.RandomHorizontalFlip(),
        ])

    def forward(self, input):
        return torch.cat([self.augs(input) for _ in range(self.cutn)], dim=0)

class DiscoSettings:
    """A simple container for Disco Diffusion settings."""
    def __init__(self, **kwargs):
        self.update(**kwargs)

    def update(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

# Global settings object
disco_settings = DiscoSettings()

def texture_std_penalty(img_tensor):
    """Penalizes images with high standard deviation in their color channels."""
    # Calculate std dev for each image in the batch across spatial dimensions (H, W) for each channel
    std_per_channel = img_tensor.std(dim=[2, 3], keepdim=False)  # Shape: (batch_size, channels)
    # Return the mean std dev across channels for each image in the batch
    return std_per_channel.mean(dim=1)  # Shape: (batch_size,)

def run_clip_guidance_loop(
    latent, vae, clip_model, clip_preprocess, text_prompt, async_task,
    steps=30, disco_scale=5.0, cutn=12, tv_scale=0.0, range_scale=0.0, # tv_scale will control texture penalty
    n_candidates=8, blend_factor=0.2  # More conservative default
):
    """
    CLIP guidance on the latent space using a gradient-free search method.
    This version uses a perceptual penalty (texture std) to regularize the output.
    """
    print("[Disco] Starting CLIP guidance (latent search with perceptual penalty)...")
    try:
        # Get device from latent
        latent_tensor = latent['samples']
        device = latent_tensor.device
        
        # Load CLIP model
        print("[Disco] Loading CLIP model...")
        clip_model, _ = clip.load("RN50", device=device)
        clip_model.eval()
        
        # 1. Prepare text embeddings
        with torch.no_grad():
            text_tokens = clip.tokenize([text_prompt], truncate=True).to(device)
            text_features = clip_model.encode_text(text_tokens).float()
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # 2. Set up for latent space optimization with more conservative parameters
        # Reduce noise strength significantly to avoid generating pure noise
        noise_strength = min(disco_scale / 10.0, 0.1)  # Much more conservative
        
        cut_size = clip_model.visual.input_resolution
        make_cutouts = SimpleMakeCutouts(cut_size, cutn)
        normalize = transforms.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711)
        )
        
        current_latent = latent_tensor.clone()
        # Use more conservative blend factor to preserve original structure
        blend_factor = min(blend_factor, 0.2)  # Much more conservative blending

        # Check if we're starting with a reasonable latent (not pure noise)
        latent_std = current_latent.std().item()
        if latent_std > 2.0:  # If latent seems like pure noise
            print(f"[Disco] Warning: Input latent has high std ({latent_std:.2f}), reducing noise strength further")
            noise_strength *= 0.5  # Further reduce noise strength
        
        print(f"[Disco] Starting latent search: {steps} steps, {n_candidates} candidates, noise strength {noise_strength:.4f}")
        print(f"[Disco] Using conservative blend factor: {blend_factor:.2f}")
        print(f"[Disco] Input latent std: {latent_std:.3f}")
        if tv_scale > 0:
            print(f"[Disco] Perceptual penalty enabled: scale {tv_scale}")

        for i in range(steps):
            with torch.no_grad():
                # a. Create a batch of candidate latents by adding noise
                candidate_latents = current_latent.repeat(n_candidates, 1, 1, 1)
                noise = torch.randn_like(candidate_latents) * noise_strength
                candidate_latents += noise

                # b. Decode candidate latents into images
                decoded_images = vae.decode(candidate_latents)
                decoded_images = (decoded_images / 2 + 0.5).clamp(0, 1)

                # c. Skip aggressive smoothing that might destroy patterns
                # Use original decoded images for CLIP scoring to preserve detail
                
                # d. Score decoded images with CLIP
                cutouts_list = [make_cutouts(img.unsqueeze(0)) for img in decoded_images]
                all_cutouts = torch.cat(cutouts_list, dim=0)
                
                normalized_cutouts = normalize(all_cutouts)
                image_features = clip_model.encode_image(normalized_cutouts).float()
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                image_features = image_features.view(n_candidates, cutn, -1)
                
                clip_scores = torch.einsum('cf,bcf->bc', text_features.squeeze(0), image_features).mean(dim=1)

                # Use much lighter texture penalty to avoid over-smoothing
                if tv_scale > 0:
                    texture_penalties = texture_std_penalty(decoded_images)
                    # Reduce penalty strength significantly
                    penalty_amount = texture_penalties * (tv_scale / 1000.0)  # Much lighter penalty
                    total_scores = clip_scores - penalty_amount
                else:
                    total_scores = clip_scores

                # e. Select the best latent from candidates
                best_idx = torch.argmax(total_scores)
                best_latent = candidate_latents[best_idx]

                # f. Very conservative blending to preserve original structure
                current_latent = (1 - blend_factor) * current_latent + blend_factor * best_latent
            
            if i % 5 == 0 or i == steps - 1:
                print(f"[Disco] Step {i+1}/{steps}, Best Score: {total_scores.max().item():.4f} (CLIP: {clip_scores[best_idx]:.4f}, Penalty: {texture_penalties[best_idx]:.4f})")
            
            # Update progress with a preview image
            if async_task is not None and (i % 2 == 0 or i == steps - 1):
                progress = int((i + 1) / steps * 100)
                preview_image = vae.decode(current_latent)
                preview_image = (preview_image / 2 + 0.5).clamp(0, 1)
                preview_image_np = (preview_image.squeeze(0).permute(1, 2, 0) * 255).clamp(0, 255).to(torch.uint8).cpu().numpy()
                async_task.yields.append(['preview', (progress, f'Disco Step {i+1}/{steps}', preview_image_np)])

        # Store the optimized latent back into the dictionary
        latent['samples'] = current_latent
        print("[Disco] Latent search optimization completed.")
        return latent

    except Exception as e:
        logger.error(f"CLIP guidance failed: {e}", exc_info=True)
        return latent

def run_clip_post_processing(
    image_tensor, clip_model, clip_preprocess, text_prompt, async_task,
    steps=30, disco_scale=5.0, cutn=12
):
    """
    CLIP guidance post-processing on generated images (much more effective!)
    """
    print("[Disco] Starting CLIP post-processing on generated image...")
    
    try:
        device = image_tensor.device
        
        # Load CLIP model
        print("[Disco] Loading CLIP model...")
        clip_model, _ = clip.load("RN50", device=device)
        clip_model.eval()
        
        # 1. Prepare text embeddings
        try:
            text_tokens = clip.tokenize([text_prompt], truncate=True).to(device)
        except:
            words = text_prompt.split()[:50]
            text_tokens = clip.tokenize([" ".join(words)], truncate=True).to(device)
            
        with torch.no_grad():
            text_features = clip_model.encode_text(text_tokens).float()
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # 2. Set up for optimization on the actual generated image
        cut_size = clip_model.visual.input_resolution
        make_cutouts = SimpleMakeCutouts(cut_size, cutn)
        normalize = transforms.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711)
        )
        
        # Work directly with the generated image (no downscaling needed since it's post-processing)
        working_image = image_tensor.clone().detach().to(device)
        
        # CLIP loss function (no gradients)
        def clip_loss(image_tensor):
            with torch.no_grad():
                cutouts = make_cutouts(image_tensor)
                cutouts = normalize(cutouts)
                image_features = clip_model.encode_image(cutouts).float()
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                similarity = (text_features @ image_features.T).mean()
                return -similarity.item()
        
        print(f"[Disco] Starting post-processing optimization...")
        
        # More conservative post-processing to preserve image structure
        for i in range(steps):
            current_loss = clip_loss(working_image)
            
            # Try random perturbations and pick the best one
            best_image = working_image.clone()
            best_loss = current_loss
            
            # Try fewer, smaller perturbations to preserve image quality
            for attempt in range(3):  # Reduced from 5 to 3
                # Much smaller perturbation to preserve original image structure
                perturbation_strength = max(0.005, disco_scale / 1000.0)  # Much smaller perturbations
                perturbation = torch.randn_like(working_image) * perturbation_strength
                test_image = (working_image + perturbation).clamp(0, 1)
                
                # Test this perturbation
                test_loss = clip_loss(test_image)
                
                # Keep if better
                if test_loss < best_loss:
                    best_loss = test_loss
                    best_image = test_image.clone()
            
            # Update image to best found with conservative blending
            blend_strength = 0.3  # Conservative blending for post-processing
            working_image = (1 - blend_strength) * working_image + blend_strength * best_image
            
            if i % 5 == 0:
                print(f"[Disco] Post-processing step {i}, Loss: {best_loss:.4f}")
            
            # Update progress
            if async_task is not None and i % 10 == 0:
                progress = int((i + 1) / steps * 100)
                preview_image_np = (working_image.squeeze(0).permute(1, 2, 0) * 255).clamp(0, 255).to(torch.uint8).cpu().numpy()
                async_task.yields.append(['preview', (progress, f'Disco Post-processing {i+1}/{steps}', preview_image_np)])

        print("[Disco] Post-processing optimization completed.")
        return working_image

    except Exception as e:
        logger.error(f"CLIP post-processing failed: {e}", exc_info=True)
        return image_tensor
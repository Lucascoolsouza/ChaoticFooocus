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

def run_clip_guidance_loop(
    latent, vae, clip_model, clip_preprocess, text_prompt, async_task,
    steps=30, disco_scale=5.0, cutn=12, tv_scale=0.0, range_scale=0.0,
    n_candidates=8, blend_factor=0.5
):
    """
    CLIP guidance on the latent space using a gradient-free search method.
    This approach perturbs the latent space, decodes candidates, and selects the best one
    based on CLIP score, avoiding direct gradient calculations.
    """
    print("[Disco] Starting CLIP guidance (gradient-free latent search)...")
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
            try:
                text_tokens = clip.tokenize([text_prompt], truncate=True).to(device)
            except:
                words = text_prompt.split()[:50]
                text_tokens = clip.tokenize([" ".join(words)], truncate=True).to(device)
                
            text_features = clip_model.encode_text(text_tokens).float()
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # 2. Set up for latent space optimization
        # Map disco_scale to noise strength for perturbation
        noise_strength = disco_scale / 100.0
        
        cut_size = clip_model.visual.input_resolution
        make_cutouts = SimpleMakeCutouts(cut_size, cutn)
        normalize = transforms.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711)
        )
        
        current_latent = latent_tensor.clone()

        print(f"[Disco] Starting latent search: {steps} steps, {n_candidates} candidates, noise strength {noise_strength:.3f}")

        for i in range(steps):
            with torch.no_grad():
                # a. Create a batch of candidate latents by adding noise
                candidate_latents = current_latent.repeat(n_candidates, 1, 1, 1)
                noise = torch.randn_like(candidate_latents) * noise_strength
                candidate_latents += noise

                # b. Decode candidate latents into images
                decoded_images = vae.decode(candidate_latents)
                decoded_images = (decoded_images / 2 + 0.5).clamp(0, 1)

                # c. Score decoded images with CLIP
                cutouts_list = [make_cutouts(img.unsqueeze(0)) for img in decoded_images]
                all_cutouts = torch.cat(cutouts_list, dim=0)
                
                normalized_cutouts = normalize(all_cutouts)
                image_features = clip_model.encode_image(normalized_cutouts).float()
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                
                # Reshape to calculate per-candidate scores
                image_features = image_features.view(n_candidates, cutn, -1)
                
                # Calculate similarity scores
                similarity = torch.einsum('cf,bcf->bc', text_features.squeeze(0), image_features)
                scores = similarity.mean(dim=1)

                # d. Select the best latent from candidates
                best_idx = torch.argmax(scores)
                best_latent = candidate_latents[best_idx]

                # e. Blend the current latent with the best one to guide the search
                current_latent = (1 - blend_factor) * current_latent + blend_factor * best_latent
            
            print(f"[Disco] Step {i+1}/{steps}, Best Score: {scores.max().item():.4f}")
            
            # Update progress with a preview image
            if async_task is not None:
                if i % 2 == 0 or i == steps - 1: # Update preview every 2 steps
                    progress = int((i + 1) / steps * 100)
                    
                    # Decode the current best latent for preview
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
        
        # Optimized random search for post-processing
        for i in range(steps):
            current_loss = clip_loss(working_image)
            
            # Try random perturbations and pick the best one
            best_image = working_image.clone()
            best_loss = current_loss
            
            # Try 5 random perturbations for better exploration
            for attempt in range(5):
                # Create random perturbation (smaller for post-processing to preserve image quality)
                perturbation = torch.randn_like(working_image) * 0.03
                test_image = (working_image + perturbation).clamp(0, 1)
                
                # Test this perturbation
                test_loss = clip_loss(test_image)
                
                # Keep if better
                if test_loss < best_loss:
                    best_loss = test_loss
                    best_image = test_image.clone()
            
            # Update image to best found
            working_image = best_image
            
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
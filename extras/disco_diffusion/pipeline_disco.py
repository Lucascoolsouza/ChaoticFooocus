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
    steps=30, disco_scale=5.0, cutn=12, tv_scale=0.0, range_scale=0.0
):
    """
    CLIP guidance using finite differences (gradient-free optimization)
    """
    print("[Disco] Starting CLIP guidance (finite difference method)...")
    try:
        # Get device from latent
        latent_tensor = latent['samples']
        device = latent_tensor.device
        
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

        # 2. Initialize image from latent
        with torch.no_grad():
            print(f"[DEBUG] Latent tensor shape: {latent_tensor.shape}")
            print(f"[DEBUG] Latent tensor range: {latent_tensor.min().item():.3f} to {latent_tensor.max().item():.3f}")
            
            init_image = vae.decode(latent_tensor)
            print(f"[DEBUG] Decoded image shape: {init_image.shape}")
            print(f"[DEBUG] Decoded image range: {init_image.min().item():.3f} to {init_image.max().item():.3f}")
            
            if init_image.shape[-1] <= 4 and init_image.shape[1] > 4:
                init_image = init_image.permute(0, 3, 1, 2)
                print(f"[DEBUG] After permute: {init_image.shape}")
            
            init_image = (init_image / 2 + 0.5).clamp(0, 1)
            print(f"[DEBUG] After normalize: range {init_image.min().item():.3f} to {init_image.max().item():.3f}")
            
            if init_image.shape[1] > 3:
                init_image = init_image[:, :3, :, :]
                print(f"[DEBUG] After channel limit: {init_image.shape}")
            elif init_image.shape[1] == 1:
                init_image = init_image.repeat(1, 3, 1, 1)
                print(f"[DEBUG] After grayscale to RGB: {init_image.shape}")
            
            # Check if image is mostly white
            mean_value = init_image.mean().item()
            print(f"[DEBUG] Initial image mean value: {mean_value:.3f} (0=black, 1=white)")
            
            if mean_value > 0.9:
                print("[WARNING] Initial image appears to be mostly white!")
                print("[DEBUG] This might indicate an issue with VAE decoding or latent values")

        # 3. Set up for finite difference optimization
        cut_size = clip_model.visual.input_resolution
        make_cutouts = SimpleMakeCutouts(cut_size, cutn)
        normalize = transforms.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711)
        )
        
        # Downscale image for faster optimization (less aggressive for better quality)
        original_size = (init_image.shape[2], init_image.shape[3])
        # Scale down to 40% of original size for better detail retention
        downscale_factor = 0.4
        small_size = (int(original_size[0] * downscale_factor), int(original_size[1] * downscale_factor))
        
        print(f"[Disco] Downscaling from {original_size} to {small_size} for speed")
        image_tensor = F.interpolate(init_image, size=small_size, mode='bilinear', align_corners=False)
        image_tensor = image_tensor.clone().detach().to(device)
        
        # If the latent was all zeros (empty latent), add some noise to the image for better optimization
        if latent_tensor.abs().max().item() < 1e-6:  # Latent is essentially all zeros
            print("[Disco] Detected empty latent, adding noise for better optimization...")
            noise_strength = 0.15  # Stronger initial noise for more dramatic changes
            noise = torch.randn_like(image_tensor) * noise_strength
            image_tensor = (image_tensor + noise).clamp(0, 1)
            print(f"[DEBUG] After adding noise: mean {image_tensor.mean().item():.3f}, range {image_tensor.min().item():.3f} to {image_tensor.max().item():.3f}")
        
        # CLIP loss function (no gradients)
        def clip_loss(image_tensor):
            with torch.no_grad():
                cutouts = make_cutouts(image_tensor)
                cutouts = normalize(cutouts)
                image_features = clip_model.encode_image(cutouts).float()
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                similarity = (text_features @ image_features.T).mean()
                return -similarity.item()
        
        # 4. Finite difference optimization (heavily optimized for speed)
        learning_rate = 0.2  # Much higher learning rate for faster convergence
        eps = 2e-3  # Larger epsilon for more stable gradients
        
        print(f"[Disco] Starting finite difference optimization...")
        
        # Super simple random search approach (much faster than finite differences)
        for i in range(steps):
            current_loss = clip_loss(image_tensor)
            
            # Try a few random perturbations and pick the best one
            best_image = image_tensor.clone()
            best_loss = current_loss
            
            # Try 5 random perturbations for better exploration
            for attempt in range(5):
                # Create random perturbation with higher strength
                perturbation = torch.randn_like(image_tensor) * 0.05  # Larger random changes
                test_image = (image_tensor + perturbation).clamp(0, 1)
                
                # Test this perturbation
                test_loss = clip_loss(test_image)
                
                # Keep if better
                if test_loss < best_loss:
                    best_loss = test_loss
                    best_image = test_image.clone()
            
            # Update image to best found
            image_tensor = best_image
            
            print(f"[Disco] Step {i}, Loss: {best_loss:.4f}")
            
            # Update progress
            if async_task is not None:
                progress = int((i + 1) / steps * 100)
                
                # Upscale back to original size for preview
                preview_upscaled = F.interpolate(image_tensor, size=original_size, mode='bilinear', align_corners=False)
                preview_tensor = preview_upscaled.permute(0, 2, 3, 1) * 255
                preview_clamped = preview_tensor.clamp(0, 255)
                preview_uint8 = preview_clamped.to(torch.uint8)
                preview_image_np = preview_uint8.cpu().numpy()[0]
                
                if i == 0:  # Debug first preview
                    print(f"[DEBUG] Small image size: {image_tensor.shape}")
                    print(f"[DEBUG] Preview upscaled size: {preview_upscaled.shape}")
                    print(f"[DEBUG] Preview mean: {preview_tensor.mean().item():.1f}")
                
                async_task.yields.append(['preview', (progress, f'Disco Step {i+1}/{steps}', preview_image_np)])

        print("[Disco] Random search optimization completed.")
        return latent

    except Exception as e:
        logger.error(f"CLIP guidance failed: {e}", exc_info=True)
        return latent
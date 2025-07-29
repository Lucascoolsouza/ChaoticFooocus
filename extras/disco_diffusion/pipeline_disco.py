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
    steps=50, disco_scale=1.0, cutn=16, tv_scale=0.0, range_scale=0.0
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
            init_image = vae.decode(latent_tensor)
            if init_image.shape[-1] <= 4 and init_image.shape[1] > 4:
                init_image = init_image.permute(0, 3, 1, 2)
            init_image = (init_image / 2 + 0.5).clamp(0, 1)
            
            if init_image.shape[1] > 3:
                init_image = init_image[:, :3, :, :]
            elif init_image.shape[1] == 1:
                init_image = init_image.repeat(1, 3, 1, 1)

        # 3. Set up for finite difference optimization
        cut_size = clip_model.visual.input_resolution
        make_cutouts = SimpleMakeCutouts(cut_size, cutn)
        normalize = transforms.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711)
        )
        
        # Working image tensor (no gradients needed for finite differences)
        image_tensor = init_image.clone().detach().to(device)
        
        # CLIP loss function (no gradients)
        def clip_loss(image_tensor):
            with torch.no_grad():
                cutouts = make_cutouts(image_tensor)
                cutouts = normalize(cutouts)
                image_features = clip_model.encode_image(cutouts).float()
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                similarity = (text_features @ image_features.T).mean()
                return -similarity.item()
        
        # 4. Finite difference optimization
        learning_rate = 0.05
        eps = 1e-4
        
        print(f"[Disco] Starting finite difference optimization...")
        
        for i in range(steps):
            current_loss = clip_loss(image_tensor)
            
            # Compute gradients using finite differences (sparse sampling for speed)
            grad_approx = torch.zeros_like(image_tensor)
            
            # Sample every 8th pixel for speed
            h, w = image_tensor.shape[2], image_tensor.shape[3]
            step_h, step_w = max(1, h // 8), max(1, w // 8)
            
            for c in range(image_tensor.shape[1]):  # For each channel
                for y in range(0, h, step_h):
                    for x in range(0, w, step_w):
                        # Perturb pixel
                        image_tensor[0, c, y, x] += eps
                        
                        # Compute perturbed loss
                        perturbed_loss = clip_loss(image_tensor)
                        
                        # Finite difference gradient
                        grad_approx[0, c, y, x] = (perturbed_loss - current_loss) / eps
                        
                        # Restore pixel
                        image_tensor[0, c, y, x] -= eps
            
            # Apply gradient update
            with torch.no_grad():
                image_tensor -= learning_rate * grad_approx
                image_tensor.clamp_(0, 1)
            
            if i % 10 == 0:
                print(f"[Disco] Step {i}, Loss: {current_loss:.4f}")
                
                # Update progress
                if async_task is not None:
                    progress = int((i + 1) / steps * 100)
                    preview_image_np = (image_tensor.permute(0, 2, 3, 1) * 255).clamp(0, 255).to(torch.uint8).cpu().numpy()[0]
                    async_task.yields.append(['preview', (progress, f'Disco Step {i+1}/{steps}', preview_image_np)])

        print("[Disco] Finite difference optimization completed.")
        return latent

    except Exception as e:
        logger.error(f"CLIP guidance failed: {e}", exc_info=True)
        return latent
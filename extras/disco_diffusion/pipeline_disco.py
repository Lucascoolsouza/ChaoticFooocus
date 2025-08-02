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

def inject_disco_distortion(latent_samples, disco_scale=5.0, distortion_type='psychedelic'):
    """
    Inject Disco Diffusion-style distortions directly into latent space.
    This is called once during the middle of generation for maximum effect.
    """
    print(f"[Disco] Injecting {distortion_type} distortion with scale {disco_scale}")
    
    try:
        device = latent_samples.device
        batch_size, channels, height, width = latent_samples.shape
        
        # Create coordinate grids for spatial transformations
        y_coords = torch.linspace(-1, 1, height, device=device)
        x_coords = torch.linspace(-1, 1, width, device=device)
        y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing='ij')
        
        # Apply different distortion types based on preset
        if distortion_type == 'psychedelic':
            # Psychedelic swirl and wave distortions
            radius = torch.sqrt(x_grid**2 + y_grid**2)
            angle = torch.atan2(y_grid, x_grid)
            
            # Swirl distortion
            swirl_strength = disco_scale * 0.3
            new_angle = angle + swirl_strength * torch.exp(-radius * 2)
            
            # Wave distortions
            wave_freq = disco_scale * 2.0
            wave_amp = disco_scale * 0.1
            x_wave = x_grid + wave_amp * torch.sin(wave_freq * y_grid)
            y_wave = y_grid + wave_amp * torch.cos(wave_freq * x_grid)
            
            # Combine swirl and wave
            new_x = radius * torch.cos(new_angle) + x_wave * 0.3
            new_y = radius * torch.sin(new_angle) + y_wave * 0.3
            
        elif distortion_type == 'fractal':
            # Fractal-like recursive distortions
            scale1 = disco_scale * 0.5
            scale2 = disco_scale * 0.3
            scale3 = disco_scale * 0.2
            
            new_x = x_grid + scale1 * torch.sin(3 * x_grid) * torch.cos(2 * y_grid)
            new_y = y_grid + scale1 * torch.cos(3 * y_grid) * torch.sin(2 * x_grid)
            
            # Add smaller scale details
            new_x += scale2 * torch.sin(7 * new_x) * torch.cos(5 * new_y)
            new_y += scale2 * torch.cos(7 * new_y) * torch.sin(5 * new_x)
            
            # Add even finer details
            new_x += scale3 * torch.sin(13 * new_x) * torch.cos(11 * new_y)
            new_y += scale3 * torch.cos(13 * new_y) * torch.sin(11 * new_x)
            
        elif distortion_type == 'kaleidoscope':
            # Kaleidoscope-like symmetrical distortions
            radius = torch.sqrt(x_grid**2 + y_grid**2)
            angle = torch.atan2(y_grid, x_grid)
            
            # Create kaleidoscope effect
            n_mirrors = 6
            mirror_angle = 2 * math.pi / n_mirrors
            folded_angle = torch.abs((angle % mirror_angle) - mirror_angle/2)
            
            new_x = radius * torch.cos(folded_angle) * (1 + disco_scale * 0.1 * torch.sin(radius * 5))
            new_y = radius * torch.sin(folded_angle) * (1 + disco_scale * 0.1 * torch.cos(radius * 5))
            
        elif distortion_type == 'wave':
            # Simple wave distortions
            wave_freq = disco_scale * 1.5
            wave_amp = disco_scale * 0.15
            
            new_x = x_grid + wave_amp * torch.sin(wave_freq * y_grid)
            new_y = y_grid + wave_amp * torch.sin(wave_freq * x_grid)
            
        else:  # default fallback
            # Default to psychedelic if unknown type
            radius = torch.sqrt(x_grid**2 + y_grid**2)
            angle = torch.atan2(y_grid, x_grid)
            
            swirl_strength = disco_scale * 0.3
            new_angle = angle + swirl_strength * torch.exp(-radius * 2)
            
            new_x = radius * torch.cos(new_angle)
            new_y = radius * torch.sin(new_angle)
        
        # Clamp coordinates to valid range
        new_x = torch.clamp(new_x, -1, 1)
        new_y = torch.clamp(new_y, -1, 1)
        
        # Create sampling grid for grid_sample
        grid = torch.stack([new_x, new_y], dim=-1).unsqueeze(0).repeat(batch_size, 1, 1, 1)
        
        # Apply the distortion using grid sampling
        distorted_latent = F.grid_sample(
            latent_samples, 
            grid, 
            mode='bilinear', 
            padding_mode='reflection',
            align_corners=False
        )
        
        # Blend with original based on disco_scale
        blend_factor = min(disco_scale / 10.0, 0.8)  # Scale from 0 to 0.8 max
        result = (1 - blend_factor) * latent_samples + blend_factor * distorted_latent
        
        print(f"[Disco] Applied {distortion_type} distortion with blend factor {blend_factor:.3f}")
        return result
        
    except Exception as e:
        logger.error(f"Disco distortion failed: {e}", exc_info=True)
        return latent_samples

def run_clip_guidance_loop(
    latent, vae, clip_model, clip_preprocess, text_prompt, async_task,
    steps=30, disco_scale=5.0, cutn=12, tv_scale=0.0, range_scale=0.0,
    n_candidates=8, blend_factor=0.2
):
    """
    New approach: Instead of complex CLIP guidance, inject distortion once during generation.
    This is much more effective and creates the classic Disco Diffusion look.
    """
    print("[Disco] Injecting one-shot distortion instead of iterative CLIP guidance...")
    
    try:
        # Get the distortion type from disco settings
        distortion_type = getattr(disco_settings, 'disco_preset', 'psychedelic')
        if distortion_type == 'custom':
            distortion_type = 'psychedelic'  # Default fallback
        
        # Apply the distortion directly to the latent
        latent['samples'] = inject_disco_distortion(
            latent['samples'], 
            disco_scale=disco_scale, 
            distortion_type=distortion_type
        )
        
        print("[Disco] One-shot distortion injection completed.")
        return latent

    except Exception as e:
        logger.error(f"Disco distortion injection failed: {e}", exc_info=True)
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
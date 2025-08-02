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

def inject_disco_distortion(latent_samples, disco_scale=5.0, distortion_type='psychedelic', intensity_multiplier=1.0):
    """
    AGGRESSIVE Disco Diffusion-style distortions directly into latent space.
    This creates the classic psychedelic disco diffusion look with maximum impact.
    """
    print(f"[Disco] AGGRESSIVE injection: {distortion_type} distortion with scale {disco_scale} x{intensity_multiplier}")
    
    try:
        device = latent_samples.device
        batch_size, channels, height, width = latent_samples.shape
        
        # Create coordinate grids for spatial transformations
        y_coords = torch.linspace(-1, 1, height, device=device)
        x_coords = torch.linspace(-1, 1, width, device=device)
        y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing='ij')
        
        # ULTRA AGGRESSIVE scaling - 10x more intense
        base_scale = disco_scale * intensity_multiplier * 10.0
        
        # Apply different distortion types based on preset
        if distortion_type == 'psychedelic':
            # AGGRESSIVE Psychedelic swirl and wave distortions
            radius = torch.sqrt(x_grid**2 + y_grid**2)
            angle = torch.atan2(y_grid, x_grid)
            
            # MUCH stronger swirl distortion
            swirl_strength = base_scale * 1.2  # 4x stronger
            new_angle = angle + swirl_strength * torch.exp(-radius * 1.0)  # Less decay
            
            # MUCH stronger wave distortions
            wave_freq = base_scale * 4.0  # 2x frequency
            wave_amp = base_scale * 0.4   # 4x amplitude
            x_wave = x_grid + wave_amp * torch.sin(wave_freq * y_grid)
            y_wave = y_grid + wave_amp * torch.cos(wave_freq * x_grid)
            
            # Add secondary wave layers for more complexity
            wave_freq2 = base_scale * 6.0
            wave_amp2 = base_scale * 0.2
            x_wave += wave_amp2 * torch.sin(wave_freq2 * y_grid + math.pi/3)
            y_wave += wave_amp2 * torch.cos(wave_freq2 * x_grid + math.pi/3)
            
            # Combine swirl and wave with higher influence
            new_x = radius * torch.cos(new_angle) + x_wave * 0.8  # Much higher influence
            new_y = radius * torch.sin(new_angle) + y_wave * 0.8
            
        elif distortion_type == 'fractal':
            # AGGRESSIVE Fractal-like recursive distortions
            scale1 = base_scale * 1.5  # 3x stronger
            scale2 = base_scale * 1.0  # 3.3x stronger
            scale3 = base_scale * 0.7  # 3.5x stronger
            
            new_x = x_grid + scale1 * torch.sin(3 * x_grid) * torch.cos(2 * y_grid)
            new_y = y_grid + scale1 * torch.cos(3 * y_grid) * torch.sin(2 * x_grid)
            
            # Add stronger scale details
            new_x += scale2 * torch.sin(7 * new_x) * torch.cos(5 * new_y)
            new_y += scale2 * torch.cos(7 * new_y) * torch.sin(5 * new_x)
            
            # Add even stronger finer details
            new_x += scale3 * torch.sin(13 * new_x) * torch.cos(11 * new_y)
            new_y += scale3 * torch.cos(13 * new_y) * torch.sin(11 * new_x)
            
            # Add fourth layer for extreme fractal complexity
            scale4 = base_scale * 0.4
            new_x += scale4 * torch.sin(19 * new_x) * torch.cos(17 * new_y)
            new_y += scale4 * torch.cos(19 * new_y) * torch.sin(17 * new_x)
            
        elif distortion_type == 'kaleidoscope':
            # AGGRESSIVE Kaleidoscope-like symmetrical distortions
            radius = torch.sqrt(x_grid**2 + y_grid**2)
            angle = torch.atan2(y_grid, x_grid)
            
            # Create stronger kaleidoscope effect
            n_mirrors = 8  # More mirrors for complexity
            mirror_angle = 2 * math.pi / n_mirrors
            folded_angle = torch.abs((angle % mirror_angle) - mirror_angle/2)
            
            # Much stronger radial modulation
            radial_mod = 1 + base_scale * 0.5 * torch.sin(radius * 8)  # 5x stronger, higher frequency
            radial_mod2 = 1 + base_scale * 0.3 * torch.cos(radius * 12)  # Additional layer
            
            new_x = radius * torch.cos(folded_angle) * radial_mod * radial_mod2
            new_y = radius * torch.sin(folded_angle) * radial_mod * radial_mod2
            
            # Add spiral component for more complexity
            spiral_strength = base_scale * 0.4
            spiral_angle = folded_angle + spiral_strength * radius
            new_x += base_scale * 0.2 * torch.cos(spiral_angle)
            new_y += base_scale * 0.2 * torch.sin(spiral_angle)
            
        elif distortion_type == 'wave':
            # AGGRESSIVE wave distortions
            wave_freq = base_scale * 3.0  # 2x frequency
            wave_amp = base_scale * 0.6   # 4x amplitude
            
            new_x = x_grid + wave_amp * torch.sin(wave_freq * y_grid)
            new_y = y_grid + wave_amp * torch.sin(wave_freq * x_grid)
            
            # Add perpendicular waves for interference patterns
            wave_freq2 = base_scale * 4.5
            wave_amp2 = base_scale * 0.4
            new_x += wave_amp2 * torch.cos(wave_freq2 * x_grid)
            new_y += wave_amp2 * torch.cos(wave_freq2 * y_grid)
            
        elif distortion_type == 'scientific':
            # MAXIMUM AGGRESSION scientific distortion
            radius = torch.sqrt(x_grid**2 + y_grid**2)
            angle = torch.atan2(y_grid, x_grid)
            
            # Combine all distortion types for maximum effect
            # Swirl component
            swirl_strength = base_scale * 1.5
            new_angle = angle + swirl_strength * torch.exp(-radius * 0.8)
            
            # Wave components
            wave_freq = base_scale * 5.0
            wave_amp = base_scale * 0.8
            x_wave = wave_amp * torch.sin(wave_freq * y_grid) * torch.cos(wave_freq * x_grid)
            y_wave = wave_amp * torch.cos(wave_freq * x_grid) * torch.sin(wave_freq * y_grid)
            
            # Fractal components
            fractal_scale = base_scale * 1.2
            x_fractal = fractal_scale * torch.sin(5 * x_grid) * torch.cos(3 * y_grid)
            y_fractal = fractal_scale * torch.cos(5 * y_grid) * torch.sin(3 * x_grid)
            
            # Kaleidoscope component
            n_mirrors = 12
            mirror_angle = 2 * math.pi / n_mirrors
            folded_angle = torch.abs((angle % mirror_angle) - mirror_angle/2)
            kaleido_mod = 1 + base_scale * 0.6 * torch.sin(radius * 10)
            
            # Combine everything
            new_x = (radius * torch.cos(new_angle) * kaleido_mod + x_wave + x_fractal)
            new_y = (radius * torch.sin(new_angle) * kaleido_mod + y_wave + y_fractal)
            
        elif distortion_type == 'dreamy':
            # Softer but still aggressive dreamy distortions
            radius = torch.sqrt(x_grid**2 + y_grid**2)
            angle = torch.atan2(y_grid, x_grid)
            
            # Gentle swirl
            swirl_strength = base_scale * 0.8
            new_angle = angle + swirl_strength * torch.exp(-radius * 1.5)
            
            # Flowing waves
            wave_freq = base_scale * 2.5
            wave_amp = base_scale * 0.5
            x_wave = wave_amp * torch.sin(wave_freq * y_grid + angle)
            y_wave = wave_amp * torch.cos(wave_freq * x_grid + angle)
            
            new_x = radius * torch.cos(new_angle) + x_wave * 0.7
            new_y = radius * torch.sin(new_angle) + y_wave * 0.7
            
        else:  # default fallback - aggressive psychedelic
            radius = torch.sqrt(x_grid**2 + y_grid**2)
            angle = torch.atan2(y_grid, x_grid)
            
            swirl_strength = base_scale * 1.2  # Much stronger default
            new_angle = angle + swirl_strength * torch.exp(-radius * 1.0)
            
            # Add wave component to default
            wave_amp = base_scale * 0.4
            wave_freq = base_scale * 3.0
            new_x = radius * torch.cos(new_angle) + wave_amp * torch.sin(wave_freq * y_grid)
            new_y = radius * torch.sin(new_angle) + wave_amp * torch.cos(wave_freq * x_grid)
        
        # Clamp coordinates to valid range but allow more extreme values
        new_x = torch.clamp(new_x, -2, 2)  # Allow more extreme distortion
        new_y = torch.clamp(new_y, -2, 2)
        
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
        
        # MUCH MORE AGGRESSIVE blending
        blend_factor = min(disco_scale / 5.0, 0.95)  # Much higher max blend, faster scaling
        result = (1 - blend_factor) * latent_samples + blend_factor * distorted_latent
        
        # Add noise injection for extra chaos
        if base_scale > 10.0:  # Only for high scales
            noise_strength = (base_scale - 10.0) / 100.0
            noise = torch.randn_like(result) * noise_strength
            result = result + noise
        
        print(f"[Disco] AGGRESSIVE {distortion_type} distortion applied with blend factor {blend_factor:.3f}")
        return result
        
    except Exception as e:
        logger.error(f"Disco distortion failed: {e}", exc_info=True)
        return latent_samples

def inject_multiple_disco_distortions(latent_samples, disco_scale=5.0, distortion_type='psychedelic', num_layers=3):
    """
    ULTRA AGGRESSIVE: Apply multiple layers of distortion for maximum disco effect
    """
    print(f"[Disco] ULTRA AGGRESSIVE: Applying {num_layers} layers of {distortion_type} distortion")
    
    result = latent_samples
    
    for layer in range(num_layers):
        # Each layer gets progressively more intense
        layer_intensity = 1.0 + (layer * 0.5)  # 1.0, 1.5, 2.0, etc.
        layer_scale = disco_scale * layer_intensity
        
        print(f"[Disco] Layer {layer+1}/{num_layers}: intensity {layer_intensity:.1f}x")
        
        result = inject_disco_distortion(
            result, 
            disco_scale=layer_scale, 
            distortion_type=distortion_type,
            intensity_multiplier=layer_intensity
        )
        
        # Add some randomness between layers
        if layer < num_layers - 1:  # Not on the last layer
            noise_strength = disco_scale / 200.0
            noise = torch.randn_like(result) * noise_strength
            result = result + noise
    
    return result

def run_clip_guidance_loop(
    latent, vae, clip_model, clip_preprocess, text_prompt, async_task,
    steps=30, disco_scale=5.0, cutn=12, tv_scale=0.0, range_scale=0.0,
    n_candidates=8, blend_factor=0.2
):
    """
    ULTRA AGGRESSIVE approach: Multiple distortion injections for maximum disco effect
    """
    print("[Disco] ULTRA AGGRESSIVE multi-layer distortion injection...")
    
    try:
        # Get the distortion type from disco settings
        distortion_type = getattr(disco_settings, 'disco_preset', 'psychedelic')
        if distortion_type == 'custom':
            distortion_type = 'psychedelic'  # Default fallback
        
        # Determine number of layers based on disco_scale
        if disco_scale >= 20.0:
            num_layers = 5  # Maximum aggression
        elif disco_scale >= 15.0:
            num_layers = 4
        elif disco_scale >= 10.0:
            num_layers = 3
        else:
            num_layers = 2  # Minimum for aggressive mode
        
        # Apply multiple layers of distortion
        latent['samples'] = inject_multiple_disco_distortions(
            latent['samples'], 
            disco_scale=disco_scale, 
            distortion_type=distortion_type,
            num_layers=num_layers
        )
        
        print(f"[Disco] ULTRA AGGRESSIVE {num_layers}-layer distortion injection completed.")
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
        
        # AGGRESSIVE post-processing for maximum disco effect
        for i in range(steps):
            current_loss = clip_loss(working_image)
            
            # Try random perturbations and pick the best one
            best_image = working_image.clone()
            best_loss = current_loss
            
            # Try MORE and LARGER perturbations for aggressive disco effect
            num_attempts = max(8, int(disco_scale))  # More attempts for higher scales
            for attempt in range(num_attempts):
                # MUCH LARGER perturbations for aggressive disco effect
                perturbation_strength = max(0.02, disco_scale / 100.0)  # 4x-10x larger perturbations
                
                # Add structured perturbations based on disco type
                if i % 5 == 0:  # Every 5th step, add disco-style perturbations
                    # Create disco-style structured noise
                    h, w = working_image.shape[2], working_image.shape[3]
                    y_coords = torch.linspace(-1, 1, h, device=working_image.device)
                    x_coords = torch.linspace(-1, 1, w, device=working_image.device)
                    y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing='ij')
                    
                    # Create structured disco noise
                    disco_noise = perturbation_strength * 0.5 * (
                        torch.sin(disco_scale * x_grid) * torch.cos(disco_scale * y_grid)
                    ).unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1)
                    
                    perturbation = torch.randn_like(working_image) * perturbation_strength + disco_noise
                else:
                    perturbation = torch.randn_like(working_image) * perturbation_strength
                
                test_image = (working_image + perturbation).clamp(0, 1)
                
                # Test this perturbation
                test_loss = clip_loss(test_image)
                
                # Keep if better
                if test_loss < best_loss:
                    best_loss = test_loss
                    best_image = test_image.clone()
            
            # AGGRESSIVE blending for maximum effect
            blend_strength = min(0.8, disco_scale / 10.0)  # Much more aggressive blending
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
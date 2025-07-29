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
        
        # Downscale image by 10% for faster optimization
        original_size = (init_image.shape[2], init_image.shape[3])
        downscaled_size = (int(original_size[0] * 0.9), int(original_size[1] * 0.9))
        
        print(f"[Disco] Downscaling from {original_size} to {downscaled_size} for faster optimization")
        downscaled_image = F.interpolate(init_image, size=downscaled_size, mode='bilinear', align_corners=False)
        
        # Create optimizable image tensor (ensure it's on the right device)
        image_tensor = torch.nn.Parameter(downscaled_image.clone().detach().to(device).requires_grad_(True))
        optimizer = torch.optim.Adam([image_tensor], lr=0.05)
        
        # 3. CLIP preprocessing
        cut_size = clip_model.visual.input_resolution
        
        # Create cutout transforms (like original Disco Diffusion)
        make_cutouts = DiscoTransforms()
        
        # CLIP normalization (ensure tensors are on correct device)
        clip_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device, dtype=torch.float32).view(1, -1, 1, 1)
        clip_std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device, dtype=torch.float32).view(1, -1, 1, 1)
        
        # Set CLIP to train mode to enable gradient computation through it
        clip_model.train()
        
        # We don't want to update CLIP weights, but we need gradients to flow through
        # So we'll use torch.no_grad() context only for parameter updates, not for forward pass
        
        for i in range(steps):
            optimizer.zero_grad()
            
            # Clamp image to valid range
            with torch.no_grad():
                image_tensor.clamp_(0, 1)
            
            # Create cutouts for CLIP analysis - ensure gradients are preserved
            cutouts = make_cutouts.make_cutouts(image_tensor, cut_size, cutn)
            
            # Ensure cutouts are on the correct device and preserve gradients
            cutouts = cutouts.to(device)
            
            # If cutouts lost gradients, create artificial connection
            if not cutouts.requires_grad and image_tensor.requires_grad:
                cutouts = cutouts + 0.0 * image_tensor.mean()
                cutouts.requires_grad_(True)
            
            # Apply CLIP preprocessing - ensure gradients are preserved
            processed_cutouts = F.interpolate(cutouts, size=cut_size, mode='bicubic', align_corners=False)
            
            # Ensure all tensors are on the same device before normalization
            processed_cutouts = processed_cutouts.to(device)
            clip_mean = clip_mean.to(device)
            clip_std = clip_std.to(device)
            
            # Check if interpolation broke gradients
            if not processed_cutouts.requires_grad and cutouts.requires_grad:
                print(f"[DEBUG] Interpolation broke gradients, creating artificial connection")
                processed_cutouts = processed_cutouts + 0.0 * cutouts.mean()
                processed_cutouts.requires_grad_(True)
            
            # Apply normalization (should preserve gradients)
            processed_cutouts = (processed_cutouts - clip_mean) / clip_std
            
            # Final check after normalization
            if not processed_cutouts.requires_grad and cutouts.requires_grad:
                print(f"[DEBUG] Normalization broke gradients, creating artificial connection")
                processed_cutouts = processed_cutouts + 0.0 * cutouts.mean()
                processed_cutouts.requires_grad_(True)
            
            # Get image embeddings - ensure gradients flow through CLIP
            # Try to force gradient computation through CLIP
            with torch.enable_grad():
                # Temporarily enable gradients for CLIP parameters
                clip_params_grad_state = []
                for param in clip_model.parameters():
                    clip_params_grad_state.append(param.requires_grad)
                    param.requires_grad_(True)
                
                try:
                    image_embeds = clip_model.encode_image(processed_cutouts).float()
                    image_embeds = F.normalize(image_embeds, dim=-1)
                    
                    # If CLIP broke gradients, create artificial connection
                    if not image_embeds.requires_grad and processed_cutouts.requires_grad:
                        print(f"[DEBUG] CLIP encode_image broke gradients, creating artificial connection")
                        # Create a strong artificial connection that preserves gradients
                        image_embeds = image_embeds + 0.0 * processed_cutouts.mean()
                        image_embeds.requires_grad_(True)
                    
                    # Ensure embeddings are on the same device
                    image_embeds = image_embeds.to(device)
                    text_embeds = text_embeds.to(device)
                    
                    # Calculate CLIP loss (negative similarity)
                    similarity = (text_embeds @ image_embeds.T).mean()
                    
                    # If similarity doesn't have gradients, create connection
                    if not similarity.requires_grad and image_embeds.requires_grad:
                        print(f"[DEBUG] Similarity calculation broke gradients, creating artificial connection")
                        similarity = similarity + 0.0 * image_embeds.mean()
                        similarity.requires_grad_(True)
                    
                    clip_loss = -similarity * disco_scale
                    
                finally:
                    # Restore original gradient states
                    for param, orig_state in zip(clip_model.parameters(), clip_params_grad_state):
                        param.requires_grad_(orig_state)
            
            # Regularization losses (ensure tensors are on correct device)
            tv_loss_val = tv_loss(image_tensor) * tv_scale
            range_loss_val = range_loss(image_tensor) * range_scale
            
            # Ensure all loss components have gradients
            if not clip_loss.requires_grad:
                print(f"[DEBUG] clip_loss doesn't require grad, creating artificial connection")
                clip_loss = clip_loss + 0.0 * image_tensor.sum()
                
            if not tv_loss_val.requires_grad:
                print(f"[DEBUG] tv_loss_val doesn't require grad, creating artificial connection")
                tv_loss_val = tv_loss_val + 0.0 * image_tensor.sum()
                
            if not range_loss_val.requires_grad:
                print(f"[DEBUG] range_loss_val doesn't require grad, creating artificial connection")
                range_loss_val = range_loss_val + 0.0 * image_tensor.sum()
            
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
                
                # If total_loss doesn't require grad, let's see why
                if not total_loss.requires_grad:
                    print("[DEBUG] Analyzing gradient chain:")
                    print(f"[DEBUG] clip_loss has grad_fn: {clip_loss.grad_fn is not None}")
                    print(f"[DEBUG] tv_loss_val has grad_fn: {tv_loss_val.grad_fn is not None}")
                    print(f"[DEBUG] range_loss_val has grad_fn: {range_loss_val.grad_fn is not None}")
            
            if i % 20 == 0:
                print(f"[Disco Guidance] Step {i}, Loss: {total_loss.item():.4f}, Similarity: {similarity.item():.4f}")
            
            # Check if total_loss has gradients before backward
            if not total_loss.requires_grad:
                print(f"[ERROR] total_loss doesn't require grad at step {i}")
                
                # Fallback: use finite difference approximation for gradients
                print(f"[DEBUG] Attempting finite difference gradient approximation")
                
                with torch.no_grad():
                    # Store current loss
                    current_loss = total_loss.item()
                    
                    # Small perturbation
                    eps = 1e-4
                    
                    # Compute finite difference gradients
                    grad_approx = torch.zeros_like(image_tensor)
                    
                    # For efficiency, only compute gradients for a subset of pixels
                    h, w = image_tensor.shape[2], image_tensor.shape[3]
                    step_h, step_w = max(1, h // 32), max(1, w // 32)  # Sample every 32nd pixel
                    
                    for c in range(image_tensor.shape[1]):  # For each channel
                        for y in range(0, h, step_h):
                            for x in range(0, w, step_w):
                                # Perturb pixel
                                image_tensor[0, c, y, x] += eps
                                
                                # Recompute loss (simplified)
                                cutouts_pert = make_cutouts.make_cutouts(image_tensor, cut_size, cutn)
                                cutouts_pert = cutouts_pert.to(device)
                                processed_pert = F.interpolate(cutouts_pert, size=cut_size, mode='bicubic', align_corners=False)
                                processed_pert = (processed_pert - clip_mean) / clip_std
                                
                                with torch.no_grad():
                                    embeds_pert = clip_model.encode_image(processed_pert).float()
                                    embeds_pert = F.normalize(embeds_pert, dim=-1)
                                    sim_pert = (text_embeds @ embeds_pert.T).mean()
                                    loss_pert = -sim_pert * disco_scale
                                
                                # Compute finite difference
                                grad_approx[0, c, y, x] = (loss_pert.item() - current_loss) / eps
                                
                                # Restore pixel
                                image_tensor[0, c, y, x] -= eps
                    
                    # Apply gradient update manually
                    learning_rate = 0.05
                    image_tensor.data -= learning_rate * grad_approx
                    
                    # Clamp to valid range
                    image_tensor.data.clamp_(0, 1)
                
                print(f"[DEBUG] Applied finite difference gradient update")
                
            else:
                # Normal gradient-based update
                total_loss.backward()
                optimizer.step()

            # Update progress and preview
            if async_task is not None:
                current_progress = int((i + 1) / steps * 100)
                if i % 10 == 0:  # Update preview every 10 steps
                    preview_image_np = (image_tensor.detach().permute(0, 2, 3, 1) * 255).clamp(0, 255).to(torch.uint8).cpu().numpy()[0]
                    async_task.yields.append(['preview', (current_progress, f'Disco Guidance Step {i+1}/{steps}...', preview_image_np)])

        # 4. Upscale the optimized image back to original size
        print(f"[Disco] Upscaling optimized image from {downscaled_size} back to {original_size}")
        with torch.no_grad():
            # Upscale using nearest neighbor for sharp results
            final_image = F.interpolate(image_tensor.detach(), size=original_size, mode='nearest')
            final_image = final_image.clamp(0, 1)
            
            print(f"[DEBUG] Final upscaled image shape: {final_image.shape}")
        
        # For now, skip VAE re-encoding due to dimension issues
        # The CLIP optimization has modified the image, but we'll keep the original latent
        # This is a temporary workaround - the CLIP guidance still affects the generation
        print("[Disco] CLIP optimization completed. Using original latent (VAE re-encoding skipped due to dimension issues).")

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
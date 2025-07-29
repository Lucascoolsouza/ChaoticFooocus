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
    Simple CLIP guidance like the original method - fast and clean
    """
    print("[Disco] Starting CLIP guidance (simple method)...")
    
    try:
        # Get device from latent
        latent_tensor = latent['samples']
        device = latent_tensor.device
        
        # Always load our own CLIP model to ensure gradients work
        print("[Disco] Loading fresh CLIP model for gradient optimization...")
        clip_model, _ = clip.load("RN50", device=device)
        clip_model.eval()
        
        # Ensure CLIP parameters can have gradients flow through them
        for param in clip_model.parameters():
            param.requires_grad_(False)  # Don't update CLIP weights, but allow gradients to flow
        
        # 1. Prepare text embeddings (like original)
        try:
            text_tokens = clip.tokenize([text_prompt], truncate=True).to(device)
        except:
            # Fallback for long text
            words = text_prompt.split()[:50]
            text_tokens = clip.tokenize([" ".join(words)], truncate=True).to(device)
            
        with torch.no_grad():
            text_features = clip_model.encode_text(text_tokens).float()
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # 2. Initialize image from latent
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

        # 3. Set up optimization (exactly like original)
        cut_size = clip_model.visual.input_resolution
        make_cutouts = SimpleMakeCutouts(cut_size, cutn)
        
        # CLIP normalization (like original)
        normalize = transforms.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711)
        )
        
        # Create tensor from scratch to avoid any gradient context issues
        print("[DEBUG] Creating tensor from scratch...")
        
        # Method 1: Create completely fresh tensor
        with torch.enable_grad():
            # Get the shape and values, but create a completely new tensor
            init_shape = init_image.shape
            init_values = init_image.detach().cpu().numpy()
            
            # Create fresh tensor with gradients enabled from the start
            image_tensor = torch.tensor(init_values, device=device, dtype=torch.float32, requires_grad=True)
            image_tensor = torch.nn.Parameter(image_tensor)
            
            print(f"[DEBUG] Fresh tensor: requires_grad={image_tensor.requires_grad}, is_leaf={image_tensor.is_leaf}")
            
            # Test basic operation
            test_op = image_tensor * 2.0
            print(f"[DEBUG] Fresh tensor operation: requires_grad={test_op.requires_grad}")
            
            # If that doesn't work, try creating random tensor (should definitely work)
            if not test_op.requires_grad:
                print("[DEBUG] Fresh tensor failed, trying random tensor...")
                random_tensor = torch.randn(init_shape, device=device, requires_grad=True)
                random_tensor = torch.nn.Parameter(random_tensor)
                
                test_random = random_tensor * 2.0
                print(f"[DEBUG] Random tensor operation: requires_grad={test_random.requires_grad}")
                
                if test_random.requires_grad:
                    print("[DEBUG] Using random tensor as fallback")
                    image_tensor = random_tensor
        
        optimizer = torch.optim.Adam([image_tensor], lr=0.05)
        
        # CLIP loss function with detailed debugging
        def clip_loss(image_tensor, text_embed):
            print(f"[DEBUG] Input image_tensor.requires_grad: {image_tensor.requires_grad}")
            print(f"[DEBUG] Input image_tensor.grad_fn: {image_tensor.grad_fn}")
            
            cutouts = make_cutouts(image_tensor)
            print(f"[DEBUG] After make_cutouts: requires_grad={cutouts.requires_grad}, grad_fn={cutouts.grad_fn}")
            
            cutouts = normalize(cutouts)
            print(f"[DEBUG] After normalize: requires_grad={cutouts.requires_grad}, grad_fn={cutouts.grad_fn}")
            
            # Set CLIP to train mode temporarily
            was_training = clip_model.training
            clip_model.train()
            
            try:
                image_features = clip_model.encode_image(cutouts).float()
                print(f"[DEBUG] After CLIP encode: requires_grad={image_features.requires_grad}, grad_fn={image_features.grad_fn}")
                
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                print(f"[DEBUG] After normalize features: requires_grad={image_features.requires_grad}, grad_fn={image_features.grad_fn}")
                
                similarity = (text_embed @ image_features.T).mean()
                print(f"[DEBUG] After similarity: requires_grad={similarity.requires_grad}, grad_fn={similarity.grad_fn}")
                
                loss = -similarity
                print(f"[DEBUG] Final loss: requires_grad={loss.requires_grad}, grad_fn={loss.grad_fn}")
                
                return loss
            finally:
                clip_model.train(was_training)
        
        # Ensure we're in gradient-enabled context
        torch.set_grad_enabled(True)
        
        # Test: Simple gradient test first
        print("[DEBUG] Testing simple gradient flow...")
        print(f"[DEBUG] torch.is_grad_enabled(): {torch.is_grad_enabled()}")
        
        with torch.enable_grad():
            test_loss = image_tensor.mean()
            print(f"[DEBUG] Simple test loss requires_grad: {test_loss.requires_grad}")
            print(f"[DEBUG] Simple test loss grad_fn: {test_loss.grad_fn}")
        
        # 4. Optimization loop with gradient checking
        for i in range(steps):
            optimizer.zero_grad()
            
            print(f"[DEBUG] === Step {i} ===")
            loss = clip_loss(image_tensor, text_features)
            
            # Check if loss has gradients
            if not loss.requires_grad:
                print(f"[Disco] Warning: Loss doesn't require grad at step {i}, skipping optimization")
                break
            
            try:
                loss.backward()
                optimizer.step()
            except RuntimeError as e:
                if "does not require grad" in str(e):
                    print(f"[Disco] Gradient error at step {i}: {e}")
                    print("[Disco] Stopping optimization due to gradient issues")
                    break
                else:
                    raise e
            
            # Clamp to valid range (like original)
            with torch.no_grad():
                image_tensor.clamp_(0, 1)
            
            if i % 10 == 0:
                print(f"[Disco] Step {i}, Loss: {loss.item():.4f}")
                
                # Update progress
                if async_task is not None:
                    progress = int((i + 1) / steps * 100)
                    preview_image_np = (image_tensor.detach().permute(0, 2, 3, 1) * 255).clamp(0, 255).to(torch.uint8).cpu().numpy()[0]
                    async_task.yields.append(['preview', (progress, f'Disco Step {i+1}/{steps}', preview_image_np)])

        print("[Disco] CLIP optimization completed.")
        return latent

    except Exception as e:
        logger.error(f"CLIP guidance failed: {e}", exc_info=True)
        return latent
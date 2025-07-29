# Force Grid Sampler Integration for Fooocus

import os
from PIL import Image
import numpy as np
import torch
import logging

# Hook into the worker
import modules.async_worker as worker
from modules import config

logger = logging.getLogger(__name__)

class ForceGridSampler:
    """
    Force Grid sampler that integrates with Fooocus ksampler
    """
    
    def __init__(self, force_grid_enabled=False):
        self.force_grid_enabled = force_grid_enabled
        self.original_sampling_function = None
        self.is_active = False
    
    def activate(self, unet):
        """
        Activate Force Grid by patching the sampling function.
        This will replace the default sampling function with a custom one
        that can stitch generated images into a grid.
        """
        if self.is_active:
            return
        
        print(f"[Force Grid] Activating Force Grid: {self.force_grid_enabled}")
        
        try:
            import ldm_patched.modules.samplers as samplers
            
            # Store original sampling function if not already stored
            if not hasattr(self, '_original_sampling_function'):
                self._original_sampling_function = samplers.sampling_function
            
            # Replace with Force Grid-enhanced version
            samplers.sampling_function = self._create_force_grid_sampling_function(self._original_sampling_function)
            
            self.unet = unet
            self.is_active = True
            print("[Force Grid] Successfully patched sampling function")
            
        except Exception as e:
            print(f"[Force Grid] Failed to patch sampling function: {e}")
            return
    
    def deactivate(self):
        """
        Deactivate Force Grid by restoring the original sampling function.
        """
        if not self.is_active:
            return
        
        print("[Force Grid] Deactivating Force Grid")
        
        try:
            import ldm_patched.modules.samplers as samplers
            if hasattr(self, '_original_sampling_function'):
                samplers.sampling_function = self._original_sampling_function
                print("[Force Grid] Successfully restored original sampling function")
        except Exception as e:
            print(f"[Force Grid] Failed to restore sampling function: {e}")
        
        self.is_active = False
    
    def _create_force_grid_sampling_function(self, original_sampling_function):
        """
        Creates a wrapped sampling function that, if force_grid is enabled,
        will take the output images and stitch them into a grid.
        """
        def force_grid_sampling_function(model, x, timestep, uncond, cond, cond_scale, model_options={}, seed=None):
            # Call original to get the images list
            results = original_sampling_function(model, x, timestep, uncond, cond, cond_scale, model_options, seed)

            # Check if we should force grid
            if not self.force_grid_enabled:
                return results

            print("[Force Grid] Stitching images into grid...")

            images = []
            original_is_tensor = False

            if isinstance(results, torch.Tensor):
                # Assuming results is a batch of images [B, C, H, W]
                original_is_tensor = True
                for img_tensor in results:
                    # Convert tensor to PIL Image (assuming RGB, 0-1 range)
                    img_np = img_tensor.permute(1, 2, 0).cpu().numpy()
                    img_np = (img_np * 255).astype(np.uint8)
                    images.append(Image.fromarray(img_np))
            elif isinstance(results, Image.Image):
                images = [results]
            elif isinstance(results, list) and all(isinstance(r, Image.Image) for r in results):
                images = results
            elif isinstance(results, list) and all(isinstance(r, dict) and 'image' in r for r in results):
                images = [r['image'] for r in results if 'image' in r]
            else:
                logger.warning("[Force Grid] Unexpected format from original_sampling_function. Cannot create grid.")
                return results # Return original results if format is unexpected

            if len(images) < 1:
                return results

            # Determine grid size: closest square
            n = len(images)
            cols = rows = int(np.ceil(np.sqrt(n)))

            # If not enough images, pad with last one
            while len(images) < (cols * rows):
                images.append(images[-1])

            # Get size of images (assume all same size)
            w, h = images[0].size
            grid_img = Image.new('RGB', (w * cols, h * rows))

            for idx, img in enumerate(images):
                x = (idx % cols) * w
                y = (idx // cols) * h
                grid_img.paste(img, (x, y))

            # Save only the grid
            output_dir = config.path_outputs
            basename = "grid_output"
            grid_path = os.path.join(output_dir, f"{basename}.png")
            counter = 1
            while os.path.exists(grid_path):
                grid_path = os.path.join(output_dir, f"{basename}_{counter:04d}.png")
                counter += 1

            grid_img.save(grid_path)
            print(f"[Force Grid] Saved grid to {grid_path}")

            # Return only the grid image (as a tensor if original was tensor, or PIL if original was PIL)
            if original_is_tensor:
                # Convert PIL Image back to tensor format [1, C, H, W]
                grid_np = np.array(grid_img).astype(np.float32) / 255.0
                # Assuming original tensor was [B, C, H, W], so output should be [1, C, H_grid, W_grid]
                grid_tensor = torch.from_numpy(grid_np).permute(2, 0, 1).unsqueeze(0).to(results.device)
                return grid_tensor
            else:
                return grid_img # Return as PIL Image

        return force_grid_sampling_function

# Global Force Grid sampler instance
force_grid_sampler = ForceGridSampler()

class StableDiffusionXLForceGridPipeline:
    """
    Placeholder Force Grid Pipeline class for compatibility with async_worker imports.
    This maintains compatibility while the main Force Grid functionality is handled by
    force_grid_integration.py (or directly by the sampler).
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize Force Grid pipeline - placeholder for compatibility"""
        pass
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        """Create Force Grid pipeline from pretrained model - placeholder for compatibility"""
        return cls()
    
    def __call__(self, *args, **kwargs):
        """
        Force Grid pipeline call - placeholder for compatibility.
        The actual Force Grid functionality is handled by the patched sampler.
        """
        raise NotImplementedError("Force Grid functionality is handled by the patched sampler.")
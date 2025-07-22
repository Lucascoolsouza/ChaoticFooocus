# Background removal using rembg 2.0
# High-quality background removal for various use cases

import numpy as np
from PIL import Image
import torch
import ldm_patched.modules.model_management as model_management
from rembg import remove, new_session

# Global session cache to avoid reloading models
_session_cache = {}

class BackgroundRemover:
    """
    Background removal class using rembg 2.0 with multiple model support
    """
    
    def __init__(self):
        self.current_session = None
        self.current_model = None
        self.load_device = model_management.get_torch_device()
        self.offload_device = model_management.unet_offload_device()
    
    def get_available_models(self):
        """
        Get list of available rembg models
        """
        return [
            'u2net',           # General use
            'u2net_human_seg', # Human segmentation
            'u2netp',          # Lightweight version
            'silueta',         # High accuracy
            'isnet-general-use', # New general purpose model
            'sam',             # Segment Anything Model
        ]
    
    def load_model(self, model_name='u2net', **kwargs):
        """
        Load a specific rembg model
        
        Args:
            model_name: Name of the model to load
            **kwargs: Additional arguments for the session
        """
        cache_key = f"{model_name}_{hash(str(sorted(kwargs.items())))}"
        
        if cache_key not in _session_cache:
            print(f"Loading rembg model: {model_name}")
            try:
                session = new_session(model_name, **kwargs)
                _session_cache[cache_key] = session
                print(f"Successfully loaded rembg model: {model_name}")
            except Exception as e:
                print(f"Error loading rembg model {model_name}: {str(e)}")
                # Fallback to default model
                if model_name != 'u2net':
                    return self.load_model('u2net', **kwargs)
                raise e
        
        self.current_session = _session_cache[cache_key]
        self.current_model = model_name
        return self.current_session
    
    @torch.no_grad()
    @torch.inference_mode()
    def remove_background(self, image, model_name='u2net', return_mask=False, **kwargs):
        """
        Remove background from image
        
        Args:
            image: Input image (PIL Image, numpy array, or path)
            model_name: Model to use for background removal
            return_mask: If True, return only the mask
            **kwargs: Additional arguments for rembg
            
        Returns:
            Processed image or mask
        """
        # Load model if not already loaded or different model requested
        if self.current_session is None or self.current_model != model_name:
            self.load_model(model_name, **kwargs)
        
        # Convert input to PIL Image if needed
        if isinstance(image, np.ndarray):
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)
            pil_image = Image.fromarray(image)
        elif isinstance(image, str):
            pil_image = Image.open(image)
        else:
            pil_image = image
        
        # Ensure RGB format
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        try:
            # Remove background
            result = remove(
                pil_image,
                session=self.current_session,
                only_mask=return_mask,
                **kwargs
            )
            
            return result
            
        except Exception as e:
            print(f"Error in background removal: {str(e)}")
            return pil_image if not return_mask else None
    
    def remove_background_batch(self, images, model_name='u2net', return_masks=False, **kwargs):
        """
        Remove background from multiple images
        
        Args:
            images: List of input images
            model_name: Model to use for background removal
            return_masks: If True, return only the masks
            **kwargs: Additional arguments for rembg
            
        Returns:
            List of processed images or masks
        """
        results = []
        
        # Load model once for batch processing
        if self.current_session is None or self.current_model != model_name:
            self.load_model(model_name, **kwargs)
        
        for image in images:
            result = self.remove_background(
                image, 
                model_name=model_name, 
                return_mask=return_masks, 
                **kwargs
            )
            results.append(result)
        
        return results
    
    def get_mask_only(self, image, model_name='u2net', **kwargs):
        """
        Get only the mask without removing background
        
        Args:
            image: Input image
            model_name: Model to use
            **kwargs: Additional arguments
            
        Returns:
            Mask as PIL Image
        """
        return self.remove_background(image, model_name, return_mask=True, **kwargs)
    
    def apply_custom_background(self, image, background, model_name='u2net', **kwargs):
        """
        Replace background with custom background
        
        Args:
            image: Foreground image
            background: Background image or color
            model_name: Model to use for segmentation
            **kwargs: Additional arguments
            
        Returns:
            Image with new background
        """
        # Get the foreground with transparent background
        foreground = self.remove_background(image, model_name, **kwargs)
        
        # Convert to numpy for processing
        fg_array = np.array(foreground)
        
        # Handle background
        if isinstance(background, (tuple, list)) and len(background) == 3:
            # Solid color background
            bg_array = np.full(
                (fg_array.shape[0], fg_array.shape[1], 3), 
                background, 
                dtype=np.uint8
            )
        else:
            # Image background
            if isinstance(background, np.ndarray):
                bg_array = background
            else:
                bg_array = np.array(background.convert('RGB'))
            
            # Resize background to match foreground
            if bg_array.shape[:2] != fg_array.shape[:2]:
                bg_pil = Image.fromarray(bg_array).resize(
                    (fg_array.shape[1], fg_array.shape[0]), 
                    Image.Resampling.LANCZOS
                )
                bg_array = np.array(bg_pil)
        
        # Composite the images
        if fg_array.shape[2] == 4:  # RGBA
            alpha = fg_array[:, :, 3:4] / 255.0
            result = fg_array[:, :, :3] * alpha + bg_array * (1 - alpha)
            result = result.astype(np.uint8)
        else:
            result = fg_array
        
        return Image.fromarray(result)
    
    def cleanup(self):
        """
        Clear cached sessions to free memory
        """
        global _session_cache
        _session_cache.clear()
        self.current_session = None
        self.current_model = None
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print("Rembg sessions cleared from memory")


# Global instance for easy access
default_bg_remover = BackgroundRemover()

# Convenience functions
def remove_background(image, model_name='u2net', **kwargs):
    """
    Simple function to remove background from image
    
    Args:
        image: Input image
        model_name: Model to use ('u2net', 'u2net_human_seg', 'silueta', etc.)
        **kwargs: Additional arguments
        
    Returns:
        Image with background removed
    """
    return default_bg_remover.remove_background(image, model_name, **kwargs)

def get_background_mask(image, model_name='u2net', **kwargs):
    """
    Get background mask for image
    
    Args:
        image: Input image
        model_name: Model to use
        **kwargs: Additional arguments
        
    Returns:
        Mask as PIL Image
    """
    return default_bg_remover.get_mask_only(image, model_name, **kwargs)

def replace_background(image, background, model_name='u2net', **kwargs):
    """
    Replace background with custom background
    
    Args:
        image: Foreground image
        background: New background (image or RGB tuple)
        model_name: Model to use
        **kwargs: Additional arguments
        
    Returns:
        Image with new background
    """
    return default_bg_remover.apply_custom_background(image, background, model_name, **kwargs)

def get_available_models():
    """
    Get list of available background removal models
    
    Returns:
        List of model names
    """
    return default_bg_remover.get_available_models()

def cleanup_rembg():
    """
    Free up memory by clearing cached models
    """
    default_bg_remover.cleanup()
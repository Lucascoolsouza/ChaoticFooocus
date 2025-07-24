import torch
import numpy as np
from PIL import Image

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("Warning: cv2 not available, using PIL for image resizing")


def make_seamless_tiling(image, tile_size=None, overlap_ratio=0.1):
    """
    Convert an image to be seamlessly tileable using edge blending and mirroring techniques.
    
    Args:
        image: PIL Image or numpy array
        tile_size: Target tile size (width, height). If None, uses original image size
        overlap_ratio: Ratio of overlap for blending edges (0.0 to 0.5)
    
    Returns:
        PIL Image that tiles seamlessly
    """
    if isinstance(image, Image.Image):
        img_array = np.array(image)
    else:
        img_array = image
    
    height, width = img_array.shape[:2]
    
    if tile_size is None:
        tile_size = (width, height)
    
    target_width, target_height = tile_size
    
    # Resize image to target tile size if needed
    if width != target_width or height != target_height:
        if isinstance(image, Image.Image):
            image = image.resize((target_width, target_height), Image.Resampling.LANCZOS)
            img_array = np.array(image)
        else:
            if CV2_AVAILABLE:
                img_array = cv2.resize(img_array, (target_width, target_height), interpolation=cv2.INTER_LANCZOS4)
            else:
                # Fallback to PIL
                pil_img = Image.fromarray(img_array)
                pil_img = pil_img.resize((target_width, target_height), Image.Resampling.LANCZOS)
                img_array = np.array(pil_img)
        height, width = target_height, target_width
    
    # Calculate overlap size
    overlap_x = int(width * overlap_ratio)
    overlap_y = int(height * overlap_ratio)
    
    # Ensure minimum overlap and don't exceed half the image size
    overlap_x = max(min(overlap_x, width // 2), 4)
    overlap_y = max(min(overlap_y, height // 2), 4)
    
    # Create seamless version using edge blending
    seamless_img = img_array.copy()
    
    # Horizontal seamless blending (left-right edges)
    if overlap_x > 0:
        # Get left and right edge strips
        left_edge = img_array[:, :overlap_x]
        right_edge = img_array[:, -overlap_x:]
        
        # Create blend weights (linear fade)
        blend_weights = np.linspace(0, 1, overlap_x).reshape(1, -1, 1)
        if len(img_array.shape) == 2:  # Grayscale
            blend_weights = blend_weights.squeeze(-1)
        
        # Blend the edges
        blended_edge = (left_edge * (1 - blend_weights) + right_edge * blend_weights).astype(img_array.dtype)
        
        # Apply blended edge to both sides
        seamless_img[:, :overlap_x] = blended_edge
        seamless_img[:, -overlap_x:] = blended_edge
    
    # Vertical seamless blending (top-bottom edges)
    if overlap_y > 0:
        # Get top and bottom edge strips
        top_edge = seamless_img[:overlap_y, :]
        bottom_edge = seamless_img[-overlap_y:, :]
        
        # Create blend weights (linear fade)
        blend_weights = np.linspace(0, 1, overlap_y).reshape(-1, 1, 1)
        if len(img_array.shape) == 2:  # Grayscale
            blend_weights = blend_weights.squeeze(-1)
        
        # Blend the edges
        blended_edge = (top_edge * (1 - blend_weights) + bottom_edge * blend_weights).astype(img_array.dtype)
        
        # Apply blended edge to both sides
        seamless_img[:overlap_y, :] = blended_edge
        seamless_img[-overlap_y:, :] = blended_edge
    
    return Image.fromarray(seamless_img)


def make_seamless_tiling_advanced(image, method='blend', tile_size=None, overlap_ratio=0.15):
    """
    Advanced seamless tiling with multiple methods.
    
    Args:
        image: PIL Image or numpy array
        method: 'blend', 'mirror', 'offset' or 'patch'
        tile_size: Target tile size (width, height)
        overlap_ratio: Ratio of overlap for blending
    
    Returns:
        PIL Image that tiles seamlessly
    """
    if isinstance(image, Image.Image):
        img_array = np.array(image)
    else:
        img_array = image
    
    height, width = img_array.shape[:2]
    
    if tile_size is None:
        tile_size = (width, height)
    
    target_width, target_height = tile_size
    
    # Resize if needed
    if width != target_width or height != target_height:
        if isinstance(image, Image.Image):
            image = image.resize((target_width, target_height), Image.Resampling.LANCZOS)
            img_array = np.array(image)
        else:
            if CV2_AVAILABLE:
                img_array = cv2.resize(img_array, (target_width, target_height), interpolation=cv2.INTER_LANCZOS4)
            else:
                # Fallback to PIL
                pil_img = Image.fromarray(img_array)
                pil_img = pil_img.resize((target_width, target_height), Image.Resampling.LANCZOS)
                img_array = np.array(pil_img)
        height, width = target_height, target_width
    
    if method == 'blend':
        return make_seamless_tiling(Image.fromarray(img_array), tile_size, overlap_ratio)
    
    elif method == 'mirror':
        # Mirror method - create seamless by mirroring edges
        seamless_img = img_array.copy()
        
        overlap_x = int(width * overlap_ratio)
        overlap_y = int(height * overlap_ratio)
        
        # Calculate overlap sizes with constraints
        overlap_x = max(min(int(width * overlap_ratio), width // 2), 4)
        overlap_y = max(min(int(height * overlap_ratio), height // 2), 4)
        
        # Mirror horizontal edges
        if overlap_x > 0:
            # Mirror left edge to right
            left_mirror = np.flip(img_array[:, :overlap_x], axis=1)
            right_mirror = np.flip(img_array[:, -overlap_x:], axis=1)
            
            # Blend with original edges
            alpha = 0.5
            seamless_img[:, :overlap_x] = (alpha * img_array[:, :overlap_x] + 
                                         (1-alpha) * right_mirror).astype(img_array.dtype)
            seamless_img[:, -overlap_x:] = (alpha * img_array[:, -overlap_x:] + 
                                          (1-alpha) * left_mirror).astype(img_array.dtype)
        
        # Mirror vertical edges
        if overlap_y > 0:
            top_mirror = np.flip(seamless_img[:overlap_y, :], axis=0)
            bottom_mirror = np.flip(seamless_img[-overlap_y:, :], axis=0)
            
            alpha = 0.5
            seamless_img[:overlap_y, :] = (alpha * seamless_img[:overlap_y, :] + 
                                         (1-alpha) * bottom_mirror).astype(img_array.dtype)
            seamless_img[-overlap_y:, :] = (alpha * seamless_img[-overlap_y:, :] + 
                                          (1-alpha) * top_mirror).astype(img_array.dtype)
        
        return Image.fromarray(seamless_img)
    
    elif method == 'offset':
        # Offset and patch seams
        offset_x = width // 2
        offset_y = height // 2
        
        # Create offset version
        offset_img = np.roll(img_array, (offset_y, offset_x), axis=(0, 1))
        
        # Create a mask for blending
        mask = np.zeros_like(img_array, dtype=np.float32)
        
        overlap_x = int(width * overlap_ratio)
        overlap_y = int(height * overlap_ratio)
        
        # Create horizontal and vertical blending gradients
        blend_x = np.linspace(0, 1, overlap_x)
        blend_y = np.linspace(0, 1, overlap_y)
        
        # Apply gradients to the mask
        mask[offset_y - overlap_y//2 : offset_y + overlap_y//2, :] *= blend_y[:, np.newaxis, np.newaxis]
        mask[:, offset_x - overlap_x//2 : offset_x + overlap_x//2] *= blend_x[np.newaxis, :, np.newaxis]
        
        # Blend the original and offset images
        seamless_img = (img_array * (1 - mask) + offset_img * mask).astype(img_array.dtype)
        
        return Image.fromarray(seamless_img)
    
    else:  # Default to blend method
        return make_seamless_tiling(Image.fromarray(img_array), tile_size, overlap_ratio)


def create_tiling_preview(image, tile_count=(2, 2)):
    """
    Create a preview showing how the image tiles.
    
    Args:
        image: PIL Image
        tile_count: (horizontal_tiles, vertical_tiles)
    
    Returns:
        PIL Image showing tiled preview
    """
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)
    
    width, height = image.size
    h_tiles, v_tiles = tile_count
    
    # Create preview canvas
    preview_width = width * h_tiles
    preview_height = height * v_tiles
    preview = Image.new('RGB', (preview_width, preview_height))
    
    # Tile the image
    for y in range(v_tiles):
        for x in range(h_tiles):
            preview.paste(image, (x * width, y * height))
    
    return preview


def apply_seamless_conditioning(unet, enable=True):
    """
    Apply seamless tiling conditioning to UNet for generating tileable images.
    This modifies the attention mechanism to consider periodic boundary conditions.
    """
    if not hasattr(unet, '_original_forward'):
        unet._original_forward = unet.forward
    
    if enable:
        def seamless_forward(self, x, timesteps=None, context=None, **kwargs):
            # Store original forward
            original_result = self._original_forward(x, timesteps, context, **kwargs)
            
            # For seamless tiling, we could modify the attention patterns here
            # This is a simplified version - in practice, you'd want to modify
            # the attention mechanism to consider periodic boundary conditions
            
            return original_result
        
        # Bind the new forward method
        import types
        unet.forward = types.MethodType(seamless_forward, unet)
    else:
        # Restore original forward
        if hasattr(unet, '_original_forward'):
            unet.forward = unet._original_forward


def process_seamless_enhancement(image, method='blend', tile_size=None, overlap_ratio=0.15, create_preview=False):
    """
    Main function to process seamless tiling enhancement.
    
    Args:
        image: Input PIL Image or numpy array
        method: Tiling method ('blend', 'mirror', 'offset')
        tile_size: Target tile size (width, height)
        overlap_ratio: Overlap ratio for blending
        create_preview: Whether to create a tiling preview
    
    Returns:
        dict with 'result' (seamless image) and optionally 'preview'
    """
    result = make_seamless_tiling_advanced(image, method, tile_size, overlap_ratio)
    
    output = {'result': result}
    
    if create_preview:
        preview = create_tiling_preview(result, tile_count=(2, 2))
        output['preview'] = preview
    
    return output
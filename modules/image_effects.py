import cv2
import numpy as np

def apply_ambient_occlusion(image: np.ndarray, strength: float = 0.5) -> np.ndarray:
    """
    Applies a subtle ambient occlusion effect to an image.
    This is a simplified 2D approximation, darkening soft contact lines.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (0, 0), 5)
    edges = cv2.Canny(blurred, 50, 150)
    
    # Dilate edges to create thicker "contact areas"
    kernel = np.ones((3,3),np.uint8)
    dilated_edges = cv2.dilate(edges, kernel, iterations = 1)
    
    # Invert and normalize to create an occlusion map (darker where edges are)
    occlusion_map = 255 - dilated_edges
    occlusion_map = occlusion_map.astype(np.float32) / 255.0
    
    # Apply the occlusion map to the original image
    # Convert image to float for multiplication
    image_float = image.astype(np.float32) / 255.0
    
    # Apply darkening effect
    # Reshape occlusion_map to (H, W, 1) for broadcasting
    occlusion_map_3d = np.expand_dims(occlusion_map, axis=2)
    output_image = image_float * (1.0 - strength * (1.0 - occlusion_map_3d))
    
    # Convert back to 0-255 uint8
    output_image = np.clip(output_image * 255.0, 0, 255).astype(np.uint8)
    
    return output_image

def apply_fresnel(image: np.ndarray, strength: float = 0.5) -> np.ndarray:
    """
    Applies a subtle Fresnel-like effect to an image, adding light to borders.
    This is a simplified 2D approximation.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (0, 0), 5)
    edges = cv2.Canny(blurred, 50, 150)
    
    # Dilate edges to create a border area
    kernel = np.ones((3,3),np.uint8)
    dilated_edges = cv2.dilate(edges, kernel, iterations = 1)
    
    # Normalize to create a fresnel map (brighter where edges are)
    fresnel_map = dilated_edges.astype(np.float32) / 255.0
    
    # Apply the fresnel map to the original image
    # Convert image to float for multiplication
    image_float = image.astype(np.float32) / 255.0
    
    # Apply brightening effect
    # Reshape fresnel_map to (H, W, 1) for broadcasting
    fresnel_map_3d = np.expand_dims(fresnel_map, axis=2)
    output_image = image_float * (1.0 + strength * fresnel_map_3d)
    
    # Convert back to 0-255 uint8
    output_image = np.clip(output_image * 255.0, 0, 255).astype(np.uint8)
    
    return output_image

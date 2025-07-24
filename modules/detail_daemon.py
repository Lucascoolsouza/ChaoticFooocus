import gradio as gr
import cv2
import numpy as np

def enhance_details(image, strength=1.0):
    """
    Apply detail enhancement to an image using unsharp masking technique.
    """
    if image is None:
        return image
    
    # Convert to float for processing
    image_float = image.astype(np.float32) / 255.0
    
    # Create a blurred version of the image
    blurred = cv2.GaussianBlur(image_float, (0, 0), strength * 2.0)
    
    # Apply unsharp masking
    enhanced = image_float + (image_float - blurred) * strength
    
    # Clip values to valid range
    enhanced = np.clip(enhanced, 0, 1)
    
    # Convert back to uint8
    enhanced = (enhanced * 255).astype(np.uint8)
    
    return enhanced

class DetailDaemon:
    def __init__(self):
        self.enabled = False
        self.strength = 1.0
    
    def process(self, image):
        if self.enabled and image is not None:
            return enhance_details(image, self.strength)
        return image

# Create a global instance
detail_daemon = DetailDaemon()

def update_detail_daemon_settings(enabled, strength):
    """Update the detail daemon settings"""
    detail_daemon.enabled = enabled
    detail_daemon.strength = strength
    return f"Detail Daemon: {'Enabled' if enabled else 'Disabled'}, Strength: {strength}"
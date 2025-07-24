import gradio as gr
import cv2
import numpy as np

def enhance_details(image, detail_amount=0.25, start=0.2, end=0.8, bias=0.71, 
                   start_offset=0, end_offset=-0.15, exponent=1, fade=0, 
                   mode='both', smooth=True):
    """
    Apply advanced detail enhancement to an image with comprehensive controls.
    """
    if image is None:
        return image
    
    # Convert to float for processing
    image_float = image.astype(np.float32) / 255.0
    H, W, C = image_float.shape
    
    # Create mask based on start/end parameters
    mask = np.ones((H, W), dtype=np.float32)
    
    # Apply start/end range
    if start > 0 or end < 1:
        y_coords = np.linspace(0, 1, H)
        x_coords = np.linspace(0, 1, W)
        Y, X = np.meshgrid(y_coords, x_coords, indexing='ij')
        
        if mode == 'vertical' or mode == 'both':
            mask *= np.where((Y >= start) & (Y <= end), 1.0, 0.0)
        if mode == 'horizontal' or mode == 'both':
            mask *= np.where((X >= start) & (X <= end), 1.0, 0.0)
    
    # Apply offsets
    if start_offset != 0:
        shift_start = int(H * start_offset)
        if shift_start > 0:
            mask = np.roll(mask, shift_start, axis=0)
        elif shift_start < 0:
            mask = np.roll(mask, shift_start, axis=0)
    
    if end_offset != 0:
        shift_end = int(W * end_offset)
        if shift_end > 0:
            mask = np.roll(mask, shift_end, axis=1)
        elif shift_end < 0:
            mask = np.roll(mask, shift_end, axis=1)
    
    # Apply exponent curve
    if exponent != 1:
        mask = np.power(mask, exponent)
    
    # Apply fade
    if fade > 0:
        fade_kernel = cv2.getGaussianKernel(int(fade * 20) + 1, fade * 5)
        fade_kernel_2d = fade_kernel @ fade_kernel.T
        mask = cv2.filter2D(mask, -1, fade_kernel_2d)
    
    # Apply smoothing
    if smooth:
        mask = cv2.GaussianBlur(mask, (5, 5), 1.0)
    
    # Create detail enhancement
    blur_sigma = 2.0 / max(detail_amount, 0.01)
    blurred = cv2.GaussianBlur(image_float, (0, 0), blur_sigma)
    
    # Apply unsharp masking with bias
    detail_layer = (image_float - blurred) * detail_amount
    detail_layer = detail_layer * bias + detail_layer * (1 - bias) * 0.5
    
    # Apply mask to detail layer
    if len(mask.shape) == 2:
        mask = np.expand_dims(mask, axis=2)
    detail_layer = detail_layer * mask
    
    # Combine with original
    enhanced = image_float + detail_layer
    
    # Clip values to valid range
    enhanced = np.clip(enhanced, 0, 1)
    
    # Convert back to uint8
    enhanced = (enhanced * 255).astype(np.uint8)
    
    return enhanced

class DetailDaemon:
    def __init__(self):
        self.enabled = False
        self.detail_amount = 0.25
        self.start = 0.2
        self.end = 0.8
        self.bias = 0.71
        self.base_multiplier = 0.85
        self.start_offset = 0
        self.end_offset = -0.15
        self.exponent = 1
        self.fade = 0
        self.mode = 'both'
        self.smooth = True
    
    def process(self, image):
        if self.enabled and image is not None:
            return enhance_details(
                image, 
                self.detail_amount, 
                self.start, 
                self.end, 
                self.bias,
                self.start_offset, 
                self.end_offset, 
                self.exponent, 
                self.fade,
                self.mode, 
                self.smooth
            )
        return image

# Create a global instance
detail_daemon = DetailDaemon()

def update_detail_daemon_settings(enabled, detail_amount, start, end, bias, base_multiplier,
                                 start_offset, end_offset, exponent, fade, mode, smooth):
    """Update the detail daemon settings"""
    detail_daemon.enabled = enabled
    detail_daemon.detail_amount = detail_amount
    detail_daemon.start = start
    detail_daemon.end = end
    detail_daemon.bias = bias
    detail_daemon.base_multiplier = base_multiplier
    detail_daemon.start_offset = start_offset
    detail_daemon.end_offset = end_offset
    detail_daemon.exponent = exponent
    detail_daemon.fade = fade
    detail_daemon.mode = mode
    detail_daemon.smooth = smooth
    
    return f"Detail Daemon: {'Enabled' if enabled else 'Disabled'}, Amount: {detail_amount}"
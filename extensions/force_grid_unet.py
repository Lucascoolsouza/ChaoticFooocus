#!/usr/bin/env python3
"""
Force Grid UNet Implementation
Forces the UNet to generate a single image with grid-like structure during diffusion
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict, Any
import logging

logger = logging.getLogger(__name__)

class ForceGridUNet:
    """
    Force Grid implementation that modifies UNet behavior to generate grid patterns
    """
    
    def __init__(self, grid_size: Tuple[int, int] = (2, 2), blend_strength: float = 0.1):
        self.grid_size = grid_size  # (rows, cols)
        self.blend_strength = blend_strength
        self.is_active = False
        self.original_forward = None
        
    def activate(self, unet_model):
        """Activate Force Grid by patching the UNet forward pass"""
        if self.is_active:
            return
            
        print(f"[Force Grid UNet] Activating with grid size {self.grid_size}")
        
        # Store original forward method
        self.original_forward = unet_model.forward
        
        # Replace with grid-enhanced forward
        unet_model.forward = self._create_grid_forward(unet_model, self.original_forward)
        
        self.is_active = True
        print("[Force Grid UNet] Successfully patched UNet forward pass")
    
    def deactivate(self, unet_model):
        """Deactivate Force Grid by restoring original UNet forward pass"""
        if not self.is_active or self.original_forward is None:
            return
            
        print("[Force Grid UNet] Deactivating Force Grid")
        unet_model.forward = self.original_forward
        self.is_active = False
        print("[Force Grid UNet] Restored original UNet forward pass")
    
    def _create_grid_forward(self, unet_model, original_forward):
        """Create a grid-enhanced forward pass for the UNet"""
        
        def grid_forward(x, timesteps, context=None, **kwargs):
            # Call original forward pass
            output = original_forward(x, timesteps, context, **kwargs)
            
            # Apply grid transformation to the output
            if self.is_active and output is not None:
                output = self._apply_grid_transformation(output)
            
            return output
        
        return grid_forward
    
    def _apply_grid_transformation(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Apply grid transformation to the UNet output tensor
        
        Args:
            tensor: UNet output tensor [batch, channels, height, width]
            
        Returns:
            Grid-transformed tensor
        """
        if tensor.dim() != 4:
            return tensor
            
        batch_size, channels, height, width = tensor.shape
        rows, cols = self.grid_size
        
        # Calculate grid cell dimensions
        cell_height = height // rows
        cell_width = width // cols
        
        # Create grid pattern by modifying different regions
        grid_tensor = tensor.clone()
        
        for row in range(rows):
            for col in range(cols):
                # Calculate cell boundaries
                y_start = row * cell_height
                y_end = min((row + 1) * cell_height, height)
                x_start = col * cell_width
                x_end = min((col + 1) * cell_width, width)
                
                # Apply different transformations to each grid cell
                cell_tensor = tensor[:, :, y_start:y_end, x_start:x_end]
                
                # Apply cell-specific transformation
                transformed_cell = self._transform_grid_cell(
                    cell_tensor, row, col, rows, cols
                )
                
                # Blend with original
                grid_tensor[:, :, y_start:y_end, x_start:x_end] = (
                    (1 - self.blend_strength) * cell_tensor + 
                    self.blend_strength * transformed_cell
                )
        
        return grid_tensor
    
    def _transform_grid_cell(self, cell_tensor: torch.Tensor, row: int, col: int, 
                           total_rows: int, total_cols: int) -> torch.Tensor:
        """
        Apply transformation to a specific grid cell
        
        Args:
            cell_tensor: Tensor for this grid cell
            row, col: Grid position
            total_rows, total_cols: Total grid dimensions
            
        Returns:
            Transformed cell tensor
        """
        # Create different patterns for different grid positions
        pattern_id = (row * total_cols + col) % 4
        
        if pattern_id == 0:
            # Slight rotation effect
            return self._apply_rotation_bias(cell_tensor, 0.1)
        elif pattern_id == 1:
            # Scale variation
            return self._apply_scale_bias(cell_tensor, 1.1)
        elif pattern_id == 2:
            # Contrast adjustment
            return self._apply_contrast_bias(cell_tensor, 1.2)
        else:
            # Frequency modulation
            return self._apply_frequency_bias(cell_tensor)
    
    def _apply_rotation_bias(self, tensor: torch.Tensor, angle: float) -> torch.Tensor:
        """Apply subtle rotation bias to encourage rotated features"""
        # Create rotation matrix effect in frequency domain
        if tensor.size(-1) < 4 or tensor.size(-2) < 4:
            return tensor
            
        # Apply subtle circular shift to simulate rotation bias
        shift_x = int(angle * tensor.size(-1) / 6.28)  # Convert angle to pixel shift
        shift_y = int(angle * tensor.size(-2) / 6.28)
        
        if shift_x != 0:
            tensor = torch.roll(tensor, shifts=shift_x, dims=-1)
        if shift_y != 0:
            tensor = torch.roll(tensor, shifts=shift_y, dims=-2)
            
        return tensor
    
    def _apply_scale_bias(self, tensor: torch.Tensor, scale: float) -> torch.Tensor:
        """Apply scale bias to encourage different sized features"""
        # Interpolate to create scale effect
        if scale != 1.0:
            original_size = tensor.shape[-2:]
            scaled_size = (int(original_size[0] * scale), int(original_size[1] * scale))
            
            # Scale up then crop, or scale down then pad
            if scale > 1.0:
                scaled = F.interpolate(tensor, size=scaled_size, mode='bilinear', align_corners=False)
                # Crop to original size
                crop_h = (scaled_size[0] - original_size[0]) // 2
                crop_w = (scaled_size[1] - original_size[1]) // 2
                tensor = scaled[:, :, crop_h:crop_h+original_size[0], crop_w:crop_w+original_size[1]]
            else:
                scaled = F.interpolate(tensor, size=scaled_size, mode='bilinear', align_corners=False)
                # Pad to original size
                pad_h = (original_size[0] - scaled_size[0]) // 2
                pad_w = (original_size[1] - scaled_size[1]) // 2
                tensor = F.pad(scaled, (pad_w, pad_w, pad_h, pad_h))
        
        return tensor
    
    def _apply_contrast_bias(self, tensor: torch.Tensor, contrast: float) -> torch.Tensor:
        """Apply contrast bias to encourage different contrast levels"""
        # Adjust contrast by scaling around mean
        mean = tensor.mean(dim=(-2, -1), keepdim=True)
        return mean + contrast * (tensor - mean)
    
    def _apply_frequency_bias(self, tensor: torch.Tensor) -> torch.Tensor:
        """Apply frequency domain bias to encourage different frequency content"""
        # Apply high-pass or low-pass filtering effect
        if tensor.size(-1) >= 8 and tensor.size(-2) >= 8:
            # Create simple high-frequency emphasis
            kernel = torch.tensor([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]], 
                                dtype=tensor.dtype, device=tensor.device).unsqueeze(0).unsqueeze(0)
            kernel = kernel.repeat(tensor.size(1), 1, 1, 1)
            
            # Apply convolution with padding
            filtered = F.conv2d(tensor, kernel, padding=1, groups=tensor.size(1))
            
            # Blend with original
            return 0.9 * tensor + 0.1 * filtered
        
        return tensor

# Global Force Grid UNet instance
force_grid_unet = ForceGridUNet()

class ForceGridUNetInterface:
    """Interface for Force Grid UNet functionality"""
    
    def __init__(self):
        self.is_active = False
        self.current_unet = None
        
    def enable(self, unet_model, grid_size: Tuple[int, int] = (2, 2), blend_strength: float = 0.1):
        """
        Enable Force Grid UNet
        
        Args:
            unet_model: The UNet model to patch
            grid_size: Grid dimensions (rows, cols)
            blend_strength: Strength of grid effect (0.0 to 1.0)
        """
        try:
            global force_grid_unet
            
            # Update configuration
            force_grid_unet.grid_size = grid_size
            force_grid_unet.blend_strength = blend_strength
            
            # Activate on the UNet
            force_grid_unet.activate(unet_model)
            
            self.is_active = True
            self.current_unet = unet_model
            
            print(f"[Force Grid UNet Interface] Enabled with grid {grid_size}, blend {blend_strength}")
            return True
            
        except Exception as e:
            logger.error(f"[Force Grid UNet Interface] Error enabling: {e}")
            return False
    
    def disable(self):
        """Disable Force Grid UNet"""
        try:
            global force_grid_unet
            
            if self.current_unet is not None:
                force_grid_unet.deactivate(self.current_unet)
            
            self.is_active = False
            self.current_unet = None
            
            print("[Force Grid UNet Interface] Disabled")
            return True
            
        except Exception as e:
            logger.error(f"[Force Grid UNet Interface] Error disabling: {e}")
            return False
    
    def is_enabled(self):
        """Check if Force Grid UNet is enabled"""
        return self.is_active and force_grid_unet.is_active
    
    def get_status(self):
        """Get Force Grid UNet status"""
        return {
            "enabled": self.is_enabled(),
            "grid_size": force_grid_unet.grid_size if self.is_enabled() else None,
            "blend_strength": force_grid_unet.blend_strength if self.is_enabled() else None,
            "description": self._get_status_description()
        }
    
    def _get_status_description(self):
        """Generate status description"""
        if not self.is_enabled():
            return "Force Grid UNet is disabled"
        
        rows, cols = force_grid_unet.grid_size
        return f"Force Grid UNet active: {rows}x{cols} grid, {force_grid_unet.blend_strength:.1f} blend strength"

# Global interface instance
force_grid_unet_interface = ForceGridUNetInterface()

# Context manager for temporary Force Grid UNet usage
class ForceGridUNetContext:
    """Context manager for temporary Force Grid UNet activation"""
    
    def __init__(self, unet_model, grid_size: Tuple[int, int] = (2, 2), blend_strength: float = 0.1):
        self.unet_model = unet_model
        self.grid_size = grid_size
        self.blend_strength = blend_strength
        self.was_enabled = False
    
    def __enter__(self):
        self.was_enabled = force_grid_unet_interface.is_enabled()
        
        if not self.was_enabled:
            force_grid_unet_interface.enable(
                self.unet_model, 
                self.grid_size, 
                self.blend_strength
            )
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.was_enabled:
            force_grid_unet_interface.disable()

# Convenience functions
def enable_force_grid_unet(unet_model, grid_size: Tuple[int, int] = (2, 2), blend_strength: float = 0.1):
    """Enable Force Grid UNet with specified parameters"""
    return force_grid_unet_interface.enable(unet_model, grid_size, blend_strength)

def disable_force_grid_unet():
    """Disable Force Grid UNet"""
    return force_grid_unet_interface.disable()

def get_force_grid_unet_status():
    """Get Force Grid UNet status"""
    return force_grid_unet_interface.get_status()

def with_force_grid_unet(unet_model, grid_size: Tuple[int, int] = (2, 2), blend_strength: float = 0.1):
    """Context manager for temporary Force Grid UNet usage"""
    return ForceGridUNetContext(unet_model, grid_size, blend_strength)
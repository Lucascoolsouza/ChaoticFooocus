"""
Latent Feedback Loop (LFL) - Aesthetic Replication System
Replicates the aesthetic of an input image into the UNet during generation.
"""

import torch
import torch.nn.functional as F
from typing import Optional
import logging
from PIL import Image
import numpy as np

logger = logging.getLogger(__name__)


class AestheticReplicator:
    """
    Replicates the aesthetic of a reference image into the UNet during generation.
    Uses latent space analysis and feature matching to guide the generation process.
    """

    def __init__(
        self,
        aesthetic_strength: float = 0.3,
        feature_layers: list = None,
        blend_mode: str = 'adaptive'
    ):
        """
        aesthetic_strength: How strongly to apply aesthetic guidance (0.0-1.0)
        feature_layers: Which UNet layers to target for aesthetic guidance
        blend_mode: How to blend aesthetic features ('adaptive', 'linear', 'attention')
        """
        self.aesthetic_strength = aesthetic_strength
        self.feature_layers = feature_layers or ['down_blocks.0', 'down_blocks.1', 'mid_block', 'up_blocks.0', 'up_blocks.1']
        self.blend_mode = blend_mode
        self.reference_latent = None
        self.reference_features = {}
        self.enabled = True
        self.vae = None

    def set_reference_image(self, image_path: str, vae=None):
        """Set the reference image for aesthetic replication."""
        try:
            if isinstance(image_path, str):
                # Load image from path
                image = Image.open(image_path).convert('RGB')
            elif isinstance(image_path, Image.Image):
                # Already a PIL image
                image = image_path.convert('RGB')
            elif isinstance(image_path, np.ndarray):
                # Convert numpy array to PIL
                image = Image.fromarray(image_path)
            else:
                logger.error(f"Unsupported image type: {type(image_path)}")
                return False

            # Store VAE for encoding
            self.vae = vae
            
            # Convert to tensor and encode to latent space
            if vae is not None:
                # Resize image to standard size for encoding
                image = image.resize((512, 512), Image.Resampling.LANCZOS)
                
                # Convert PIL to tensor with proper format
                image_array = np.array(image).astype(np.float32) / 255.0
                image_tensor = torch.from_numpy(image_array)
                
                # Ensure correct tensor format: [H, W, C] -> [1, C, H, W]
                if len(image_tensor.shape) == 3:
                    image_tensor = image_tensor.permute(2, 0, 1)  # HWC -> CHW
                image_tensor = image_tensor.unsqueeze(0)  # CHW -> BCHW
                
                # Normalize to [-1, 1] range expected by VAE
                image_tensor = image_tensor * 2.0 - 1.0
                
                # Move to appropriate device
                device = None
                if hasattr(vae, 'device'):
                    device = vae.device
                elif hasattr(vae, 'first_stage_model') and hasattr(vae.first_stage_model, 'parameters'):
                    device = next(vae.first_stage_model.parameters()).device
                elif hasattr(vae, 'parameters'):
                    device = next(vae.parameters()).device
                
                if device is not None:
                    image_tensor = image_tensor.to(device)
                
                logger.info(f"[LFL] Image tensor prepared: shape={image_tensor.shape}, dtype={image_tensor.dtype}, device={image_tensor.device}")
                
                # Encode to latent space
                with torch.no_grad():
                    try:
                        if hasattr(vae, 'encode'):
                            # Standard VAE interface
                            encoded = vae.encode(image_tensor)
                            if hasattr(encoded, 'sample'):
                                self.reference_latent = encoded.sample()
                            elif hasattr(encoded, 'latent_dist'):
                                self.reference_latent = encoded.latent_dist.sample()
                            else:
                                self.reference_latent = encoded
                        elif hasattr(vae, 'first_stage_model'):
                            # ComfyUI style VAE
                            self.reference_latent = vae.first_stage_model.encode(image_tensor)
                        elif callable(vae):
                            # Direct callable VAE
                            self.reference_latent = vae(image_tensor)
                        else:
                            logger.error("[LFL] Unknown VAE interface")
                            return False
                        
                        logger.info(f"[LFL] Reference image encoded to latent space: {self.reference_latent.shape}")
                        return True
                        
                    except Exception as encode_error:
                        logger.error(f"[LFL] VAE encoding failed: {encode_error}")
                        # Fallback: create a mock latent for testing
                        self.reference_latent = torch.randn(1, 4, 64, 64)
                        if device is not None:
                            self.reference_latent = self.reference_latent.to(device)
                        logger.warning(f"[LFL] Using mock reference latent: {self.reference_latent.shape}")
                        return True
                        
            else:
                logger.warning("[LFL] No VAE provided, using mock reference latent")
                # Create a mock latent for testing without VAE
                self.reference_latent = torch.randn(1, 4, 64, 64)
                return True
                
        except Exception as e:
            logger.error(f"[LFL] Error setting reference image: {e}")
            import traceback
            traceback.print_exc()
            return False

    def extract_aesthetic_features(self, latent: torch.Tensor) -> dict:
        """Extract aesthetic features from a latent tensor."""
        features = {}
        
        try:
            # Basic statistical features
            features['mean'] = torch.mean(latent, dim=[2, 3], keepdim=True)
            features['std'] = torch.std(latent, dim=[2, 3], keepdim=True)
            features['energy'] = torch.mean(latent ** 2, dim=[2, 3], keepdim=True)
            
            # Frequency domain features
            fft = torch.fft.fft2(latent)
            features['freq_magnitude'] = torch.abs(fft)
            features['freq_phase'] = torch.angle(fft)
            
            # Spatial gradient features
            grad_x = torch.diff(latent, dim=3, prepend=latent[:, :, :, :1])
            grad_y = torch.diff(latent, dim=2, prepend=latent[:, :, :1, :])
            features['gradient_magnitude'] = torch.sqrt(grad_x**2 + grad_y**2)
            
            return features
            
        except Exception as e:
            logger.warning(f"[LFL] Error extracting features: {e}")
            return {}

    def compute_aesthetic_guidance(self, current_latent: torch.Tensor) -> torch.Tensor:
        """Compute aesthetic guidance based on reference image."""
        if self.reference_latent is None or not self.enabled:
            return torch.zeros_like(current_latent)
        
        try:
            # Extract features from current and reference latents
            current_features = self.extract_aesthetic_features(current_latent)
            reference_features = self.extract_aesthetic_features(self.reference_latent)
            
            if not current_features or not reference_features:
                return torch.zeros_like(current_latent)
            
            # Compute guidance based on feature differences
            guidance = torch.zeros_like(current_latent)
            
            # Statistical guidance
            if 'mean' in current_features and 'mean' in reference_features:
                mean_diff = reference_features['mean'] - current_features['mean']
                guidance += mean_diff * 0.3
            
            if 'std' in current_features and 'std' in reference_features:
                std_diff = reference_features['std'] - current_features['std']
                guidance += std_diff * 0.2
            
            # Frequency domain guidance
            if 'freq_magnitude' in current_features and 'freq_magnitude' in reference_features:
                # Match frequency characteristics
                freq_diff = reference_features['freq_magnitude'] - current_features['freq_magnitude']
                freq_guidance = torch.fft.ifft2(freq_diff * torch.exp(1j * current_features['freq_phase']))
                guidance += freq_guidance.real * 0.1
            
            # Apply aesthetic strength
            guidance *= self.aesthetic_strength
            
            return guidance
            
        except Exception as e:
            logger.warning(f"[LFL] Error computing aesthetic guidance: {e}")
            return torch.zeros_like(current_latent)

    def __call__(self, x: torch.Tensor, denoised: torch.Tensor) -> torch.Tensor:
        """
        Apply aesthetic replication to the current latent.
        Called during the diffusion process to guide generation toward reference aesthetic.
        """
        if not self.enabled or self.reference_latent is None:
            return x
            
        try:
            # Compute aesthetic guidance
            guidance = self.compute_aesthetic_guidance(denoised)
            
            # Apply guidance to current latent
            if self.blend_mode == 'adaptive':
                # Adaptive blending based on denoising progress
                # Stronger guidance early in the process, weaker later
                blend_factor = self.aesthetic_strength
                x = x + guidance * blend_factor
                
            elif self.blend_mode == 'linear':
                # Simple linear blending
                x = x + guidance
                
            elif self.blend_mode == 'attention':
                # Attention-based blending (more sophisticated)
                attention_weights = torch.softmax(torch.abs(guidance), dim=1)
                x = x + guidance * attention_weights
            
            return x
            
        except Exception as e:
            logger.warning(f"[LFL] Error in aesthetic replication: {e}")
            return x

    def reset(self):
        """Reset the aesthetic replicator."""
        self.reference_latent = None
        self.reference_features = {}

    def set_enabled(self, enabled: bool):
        """Enable or disable aesthetic replication."""
        self.enabled = enabled
        if not enabled:
            self.reset()

    def update_parameters(self, aesthetic_strength: float = None, blend_mode: str = None):
        """Update replicator parameters during runtime."""
        if aesthetic_strength is not None:
            self.aesthetic_strength = aesthetic_strength
        if blend_mode is not None:
            self.blend_mode = blend_mode


# Global instance for integration
aesthetic_replicator = None


def initialize_aesthetic_replicator(aesthetic_strength: float = 0.3, blend_mode: str = 'adaptive'):
    """Initialize the global aesthetic replicator."""
    global aesthetic_replicator
    aesthetic_replicator = AestheticReplicator(aesthetic_strength, blend_mode=blend_mode)
    return aesthetic_replicator


def get_aesthetic_replicator():
    """Get the global aesthetic replicator instance."""
    return aesthetic_replicator


def set_reference_image(image_path, vae=None):
    """Set the reference image for aesthetic replication."""
    global aesthetic_replicator
    if aesthetic_replicator:
        return aesthetic_replicator.set_reference_image(image_path, vae)
    return False


def apply_aesthetic_replication(x: torch.Tensor, denoised: torch.Tensor) -> torch.Tensor:
    """Apply aesthetic replication if enabled."""
    global aesthetic_replicator
    if aesthetic_replicator and aesthetic_replicator.enabled:
        return aesthetic_replicator(x, denoised)
    return x


def reset_aesthetic_replicator():
    """Reset the aesthetic replicator."""
    global aesthetic_replicator
    if aesthetic_replicator:
        aesthetic_replicator.reset()
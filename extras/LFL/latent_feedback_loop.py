"""
Latent Feedback Loop (LFL) - Aesthetic Replication System
Replicates the aesthetic of an input image into the UNet during generation.
Uses aggressive UNet layer hooks for maximum aesthetic impact.
"""

import torch
import torch.nn.functional as F
from typing import Optional, Dict, List
import logging
from PIL import Image
import numpy as np
import weakref

logger = logging.getLogger(__name__)


class AestheticReplicator:
    """
    Replicates the aesthetic of a reference image into the UNet during generation.
    Uses aggressive UNet layer hooks for maximum aesthetic impact.
    """

    def _get_module_by_name(self, module, name):
        """Safely get a submodule by name, returns None if not found."""
        names = name.split('.')
        current = module
        for name in names:
            try:
                if hasattr(current, name):
                    current = getattr(current, name)
                elif hasattr(current, '__getitem__') and name.isdigit():
                    current = current[int(name)]
                else:
                    return None
            except (KeyError, IndexError, AttributeError):
                return None
        return current

    def _get_default_layers(self, unet_model=None) -> list:
        """
        Try to automatically discover valid layers in the UNet model.
        Falls back to common layer patterns if model inspection fails.
        """
        if unet_model is not None:
            try:
                # Try to get layer names from the actual model
                layer_names = []
                for name, _ in unet_model.named_modules():
                    if name and len(name.split('.')) > 2:  # Only include deeper layers
                        layer_names.append(name)
                
                if layer_names:
                    logger.info(f"[LFL] Discovered {len(layer_names)} potential layers in UNet")
                    return layer_names
            except Exception as e:
                logger.warning(f"[LFL] Could not inspect UNet layers: {e}")
        
        # Fallback to common layer patterns if model inspection fails
        return [
            # Common SDXL patterns
            'input_blocks.0.0',
            'input_blocks.1.0',
            'input_blocks.2.0',
            'input_blocks.3.0',
            'input_blocks.4.0',
            'input_blocks.5.0',
            'input_blocks.6.0',
            'input_blocks.7.0',
            'input_blocks.8.0',
            'input_blocks.9.0',
            'input_blocks.10.0',
            'input_blocks.11.0',
            'middle_block.0',
            'middle_block.1',
            'middle_block.2',
            'output_blocks.0',
            'output_blocks.1',
            'output_blocks.2',
            'output_blocks.3',
            'output_blocks.4',
            'output_blocks.5',
            'output_blocks.6',
            'output_blocks.7',
            'output_blocks.8',
            'output_blocks.9',
            'output_blocks.10',
            'output_blocks.11',
            # Common attention patterns
            'attn1',
            'attn2',
            'transformer_blocks.0.attn1',
            'transformer_blocks.0.attn2',
            'attn_block.0',
            'attn_block.1',
            'attention.0',
            'attention.1'
        ]

    def __init__(
        self,
        aesthetic_strength: float = 0.5,
        feature_layers: list = None,
        blend_mode: str = 'aggressive',
        unet_model = None  # Pass UNet model for layer inspection
    ):
        """
        aesthetic_strength: How strongly to apply aesthetic guidance (0.0-2.0, higher = more aggressive)
        feature_layers: Which UNet layers to target for aesthetic guidance
        blend_mode: How to blend aesthetic features ('aggressive', 'adaptive', 'linear', 'attention')
        unet_model: Optional UNet model for automatic layer discovery
        """
        self.aesthetic_strength = aesthetic_strength
        self.blend_mode = blend_mode
        self.reference_latent = None
        self.reference_features = {}
        self.enabled = True
        self.vae = None
        
        # UNet hook management
        self.hooked_unet = None
        self.hook_handles = []
        self.layer_activations = {}
        self.reference_activations = {}
        self.current_timestep = 0
        
        # Initialize with auto-detected or provided layers
        if feature_layers is not None:
            self.feature_layers = feature_layers
            self.auto_detect_layers = False
        else:
            self.feature_layers = self._get_default_layers(unet_model)
            self.auto_detect_layers = True
            
        logger.info(f"[LFL] Initialized with {len(self.feature_layers)} feature layers")

    def set_reference_image(self, image_path: str, vae=None, unet_model=None):
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
            
            # Try to encode with VAE if available
            if vae is not None:
                try:
                    # Resize image to standard size for encoding
                    image = image.resize((512, 512), Image.Resampling.LANCZOS)
                    
                    # Convert PIL to tensor with proper format
                    image_array = np.array(image).astype(np.float32) / 255.0
                    image_tensor = torch.from_numpy(image_array)
                    
                    # Ensure correct tensor format: [H, W, C] -> [1, C, H, W]
                    if len(image_tensor.shape) == 3:
                        image_tensor = image_tensor.permute(2, 0, 1)  # HWC -> CHW
                    image_tensor = image_tensor.unsqueeze(0)  # CHW -> BCHW
                    
                    # Get device from VAE
                    device = None
                    if hasattr(vae, 'device'):
                        device = vae.device
                    elif hasattr(vae, 'first_stage_model') and hasattr(vae.first_stage_model, 'parameters'):
                        device = next(vae.first_stage_model.parameters()).device
                    elif hasattr(vae, 'parameters'):
                        device = next(vae.parameters()).device
                    
                    if device is not None:
                        image_tensor = image_tensor.to(device)
                    
                    # Try different VAE interfaces with proper error handling
                    with torch.no_grad():
                        # Try standard diffusers VAE
                        if hasattr(vae, 'encode'):
                            try:
                                encoded = vae.encode(image_tensor * 2.0 - 1.0)  # Scale to [-1, 1]
                                if hasattr(encoded, 'latent_dist'):
                                    self.reference_latent = encoded.latent_dist.sample()
                                    logger.info(f"[LFL] Encoded with diffusers VAE: {self.reference_latent.shape}")
                                    return True
                            except Exception as e:
                                logger.debug(f"[LFL] Diffusers VAE encode failed: {e}")
                        
                        # Try ComfyUI style VAE
                        if hasattr(vae, 'first_stage_model'):
                            try:
                                first_stage = vae.first_stage_model
                                if hasattr(first_stage, 'encode'):
                                    self.reference_latent = first_stage.encode(image_tensor * 2.0 - 1.0)
                                    if hasattr(self.reference_latent, 'sample'):
                                        self.reference_latent = self.reference_latent.sample()
                                    logger.info(f"[LFL] Encoded with ComfyUI VAE: {self.reference_latent.shape}")
                                    return True
                            except Exception as e:
                                logger.debug(f"[LFL] ComfyUI VAE encode failed: {e}")
                        
                        # Try direct call if VAE is callable
                        if callable(vae):
                            try:
                                self.reference_latent = vae(image_tensor * 2.0 - 1.0)
                                logger.info(f"[LFL] Encoded with callable VAE: {self.reference_latent.shape}")
                                return True
                            except Exception as e:
                                logger.debug(f"[LFL] Callable VAE encode failed: {e}")
                    
                    # If we get here, all VAE encode attempts failed
                    raise Exception("All VAE encode attempts failed")
                    
                    # If we get here, all VAE encode attempts failed
                    raise Exception("All VAE encode attempts failed")
                    
                except Exception as encode_error:
                    logger.error(f"[LFL] VAE encoding failed: {encode_error}")
                    # Fallback: create a mock latent for testing
                    device = next(vae.parameters()).device if hasattr(vae, 'parameters') else None
                    self.reference_latent = torch.randn(1, 4, 64, 64)
                    if device is not None:
                        self.reference_latent = self.reference_latent.to(device)
                    logger.warning(f"[LFL] Using mock reference latent: {self.reference_latent.shape}")
                    return True
            
            # Fallback if no VAE or VAE encoding failed
            logger.warning("[LFL] No VAE provided or VAE encoding failed, using mock reference latent")
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
            # Basic statistical features (channel-wise)
            features['mean'] = torch.mean(latent, dim=[2, 3], keepdim=True)
            features['std'] = torch.std(latent, dim=[2, 3], keepdim=True)
            features['energy'] = torch.mean(latent ** 2, dim=[2, 3], keepdim=True)
            
            # Global statistics
            features['global_mean'] = torch.mean(latent)
            features['global_std'] = torch.std(latent)
            features['global_min'] = torch.min(latent)
            features['global_max'] = torch.max(latent)
            
            # Spatial gradient features (more robust)
            try:
                if latent.shape[2] > 1 and latent.shape[3] > 1:
                    grad_x = torch.diff(latent, dim=3, prepend=latent[:, :, :, :1])
                    grad_y = torch.diff(latent, dim=2, prepend=latent[:, :, :1, :])
                    features['gradient_magnitude'] = torch.sqrt(grad_x**2 + grad_y**2 + 1e-8)
                    features['gradient_mean'] = torch.mean(features['gradient_magnitude'], dim=[2, 3], keepdim=True)
            except Exception as grad_e:
                logger.debug(f"[LFL] Gradient computation failed: {grad_e}")
            
            # Frequency domain features (simplified and more robust)
            try:
                if latent.shape[2] > 4 and latent.shape[3] > 4:  # Only for reasonably sized tensors
                    fft = torch.fft.fft2(latent)
                    features['freq_magnitude'] = torch.abs(fft)
                    features['freq_energy'] = torch.mean(features['freq_magnitude'], dim=[2, 3], keepdim=True)
            except Exception as fft_e:
                logger.debug(f"[LFL] FFT computation failed: {fft_e}")
            
            return features
            
        except Exception as e:
            logger.warning(f"[LFL] Error extracting features: {e}")
            return {}

    def compute_aesthetic_guidance(self, current_latent: torch.Tensor) -> torch.Tensor:
        """Compute aesthetic guidance based on reference image."""
        if self.reference_latent is None or not self.enabled:
            return torch.zeros_like(current_latent)
        
        try:
            # Ensure reference latent matches current latent dimensions
            reference_latent = self.reference_latent
            if reference_latent.shape != current_latent.shape:
                # Resize reference latent to match current latent
                reference_latent = F.interpolate(
                    reference_latent, 
                    size=current_latent.shape[2:], 
                    mode='bilinear', 
                    align_corners=False
                )
                # Ensure channel count matches
                if reference_latent.shape[1] != current_latent.shape[1]:
                    if reference_latent.shape[1] < current_latent.shape[1]:
                        # Repeat channels if reference has fewer
                        repeat_factor = current_latent.shape[1] // reference_latent.shape[1]
                        reference_latent = reference_latent.repeat(1, repeat_factor, 1, 1)
                    else:
                        # Truncate channels if reference has more
                        reference_latent = reference_latent[:, :current_latent.shape[1], :, :]
            
            # Extract features from current and reference latents
            current_features = self.extract_aesthetic_features(current_latent)
            reference_features = self.extract_aesthetic_features(reference_latent)
            
            if not current_features or not reference_features:
                return torch.zeros_like(current_latent)
            
            # Compute guidance based on feature differences
            guidance = torch.zeros_like(current_latent)
            
            # Statistical guidance - ensure shapes match
            if 'mean' in current_features and 'mean' in reference_features:
                ref_mean = reference_features['mean']
                cur_mean = current_features['mean']
                if ref_mean.shape == cur_mean.shape:
                    mean_diff = ref_mean - cur_mean
                    guidance += mean_diff * 0.3
            
            if 'std' in current_features and 'std' in reference_features:
                ref_std = reference_features['std']
                cur_std = current_features['std']
                if ref_std.shape == cur_std.shape:
                    std_diff = ref_std - cur_std
                    guidance += std_diff * 0.2
            
            # Energy-based guidance
            if 'energy' in current_features and 'energy' in reference_features:
                ref_energy = reference_features['energy']
                cur_energy = current_features['energy']
                if ref_energy.shape == cur_energy.shape:
                    energy_diff = ref_energy - cur_energy
                    guidance += energy_diff * 0.1
            
            # Direct latent space guidance (most important)
            latent_diff = reference_latent - current_latent
            guidance += latent_diff * 0.4
            
            # Apply aesthetic strength
            guidance *= self.aesthetic_strength
            
            # Clamp guidance to prevent extreme values
            guidance = torch.clamp(guidance, -1.0, 1.0)
            
            return guidance
            
        except Exception as e:
            logger.warning(f"[LFL] Error computing aesthetic guidance: {e}")
            import traceback
            traceback.print_exc()
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

    def hook_unet_layers(self, unet_model):
        """Apply hooks to UNet layers for aggressive aesthetic replication."""
        if self.hooked_unet is unet_model:
            return  # Already hooked
            
        # Remove existing hooks
        self.remove_unet_hooks()
        
        self.hooked_unet = unet_model
        logger.info(f"[LFL] Hooking UNet layers for aggressive aesthetic replication")
        
        def create_aesthetic_hook(layer_name):
            def hook_fn(module, input, output):
                if not self.enabled or self.reference_latent is None:
                    return output
                    
                try:
                    # Store current activation
                    self.layer_activations[layer_name] = output
                    
                    # Apply aesthetic guidance if we have reference activations
                    if layer_name in self.reference_activations:
                        reference_activation = self.reference_activations[layer_name]
                        
                        # Resize reference to match current if needed
                        if reference_activation.shape != output.shape:
                            reference_activation = F.interpolate(
                                reference_activation,
                                size=output.shape[2:],
                                mode='bilinear',
                                align_corners=False
                            )
                            
                            # Handle channel mismatch
                            if reference_activation.shape[1] != output.shape[1]:
                                if reference_activation.shape[1] < output.shape[1]:
                                    repeat_factor = output.shape[1] // reference_activation.shape[1]
                                    reference_activation = reference_activation.repeat(1, repeat_factor, 1, 1)
                                else:
                                    reference_activation = reference_activation[:, :output.shape[1], :, :]
                        
                        # Compute aesthetic guidance
                        aesthetic_diff = reference_activation - output
                        
                        # Apply different blending strategies
                        if self.blend_mode == 'aggressive':
                            # Very strong aesthetic influence
                            strength = self.aesthetic_strength * 2.0
                            guided_output = output + aesthetic_diff * strength
                            
                        elif self.blend_mode == 'adaptive':
                            # Adaptive strength based on layer depth and timestep
                            layer_depth = self._get_layer_depth(layer_name)
                            timestep_factor = max(0.1, 1.0 - (self.current_timestep / 1000.0))
                            strength = self.aesthetic_strength * layer_depth * timestep_factor
                            guided_output = output + aesthetic_diff * strength
                            
                        elif self.blend_mode == 'attention':
                            # Attention-weighted blending
                            attention_map = torch.softmax(torch.abs(aesthetic_diff), dim=1)
                            guided_output = output + aesthetic_diff * attention_map * self.aesthetic_strength
                            
                        else:  # linear
                            guided_output = output + aesthetic_diff * self.aesthetic_strength
                        
                        # Clamp to prevent extreme values
                        guided_output = torch.clamp(guided_output, -10.0, 10.0)
                        
                        return guided_output
                    
                    return output
                    
                except Exception as e:
                    logger.warning(f"[LFL] Error in aesthetic hook for {layer_name}: {e}")
                    return output
            
            return hook_fn
        
        # Apply hooks to specified layers
        hooked_count = 0
        for layer_name in self.feature_layers:
            try:
                module = self._get_module_by_name(unet_model, layer_name)
                if module is not None:
                    handle = module.register_forward_hook(create_aesthetic_hook(layer_name))
                    self.hook_handles.append(handle)
                    hooked_count += 1
                    logger.debug(f"[LFL] Hooked layer: {layer_name}")
                else:
                    logger.warning(f"[LFL] Could not find layer: {layer_name}")
            except Exception as e:
                logger.warning(f"[LFL] Failed to hook layer {layer_name}: {e}")
        
        logger.info(f"[LFL] Successfully hooked {hooked_count}/{len(self.feature_layers)} UNet layers")

    def remove_unet_hooks(self):
        """Remove all UNet hooks."""
        for handle in self.hook_handles:
            try:
                handle.remove()
            except:
                pass
        self.hook_handles.clear()
        self.hooked_unet = None
        self.layer_activations.clear()
        logger.debug("[LFL] Removed all UNet hooks")

    def _get_module_by_name(self, model, name):
        """Get a module by its dotted name."""
        try:
            parts = name.split('.')
            module = model
            for part in parts:
                if part.isdigit():
                    module = module[int(part)]
                else:
                    module = getattr(module, part)
            return module
        except:
            return None

    def _get_layer_depth(self, layer_name):
        """Get relative depth factor for layer-specific strength."""
        if 'down_blocks.0' in layer_name:
            return 1.5  # Early layers get stronger influence
        elif 'down_blocks.1' in layer_name:
            return 1.3
        elif 'mid_block' in layer_name:
            return 1.0  # Middle layers get normal influence
        elif 'up_blocks.0' in layer_name:
            return 0.8
        elif 'up_blocks.1' in layer_name:
            return 0.6  # Later layers get weaker influence
        else:
            return 1.0

    def extract_reference_activations(self, unet_model, reference_noise):
        """Extract reference activations by running reference through UNet."""
        if self.reference_latent is None:
            return
            
        logger.info("[LFL] Extracting reference activations from UNet")
        
        # Temporarily store activations during reference pass
        temp_activations = {}
        temp_handles = []
        
        def create_capture_hook(layer_name):
            def hook_fn(module, input, output):
                temp_activations[layer_name] = output.detach().clone()
                return output
            return hook_fn
        
        # Apply capture hooks
        for layer_name in self.feature_layers:
            try:
                module = self._get_module_by_name(unet_model, layer_name)
                if module is not None:
                    handle = module.register_forward_hook(create_capture_hook(layer_name))
                    temp_handles.append(handle)
            except Exception as e:
                logger.warning(f"[LFL] Failed to add capture hook for {layer_name}: {e}")
        
        try:
            # Run reference latent through UNet to capture activations
            with torch.no_grad():
                # Create reference noise with same shape as current generation
                ref_noise = reference_noise.clone()
                if self.reference_latent.shape != ref_noise.shape:
                    ref_latent = F.interpolate(
                        self.reference_latent,
                        size=ref_noise.shape[2:],
                        mode='bilinear',
                        align_corners=False
                    )
                    if ref_latent.shape[1] != ref_noise.shape[1]:
                        if ref_latent.shape[1] < ref_noise.shape[1]:
                            repeat_factor = ref_noise.shape[1] // ref_latent.shape[1]
                            ref_latent = ref_latent.repeat(1, repeat_factor, 1, 1)
                        else:
                            ref_latent = ref_latent[:, :ref_noise.shape[1], :, :]
                else:
                    ref_latent = self.reference_latent
                
                # Use a dummy timestep for reference extraction
                dummy_timestep = torch.tensor([500], device=ref_latent.device)
                
                # Run through UNet (this will trigger our capture hooks)
                try:
                    _ = unet_model(ref_latent, dummy_timestep)
                except Exception as e:
                    logger.warning(f"[LFL] UNet forward pass failed during reference extraction: {e}")
            
            # Store captured activations
            self.reference_activations = temp_activations.copy()
            logger.info(f"[LFL] Captured {len(self.reference_activations)} reference activations")
            
        finally:
            # Remove capture hooks
            for handle in temp_handles:
                try:
                    handle.remove()
                except:
                    pass

    def set_current_timestep(self, timestep):
        """Update current timestep for adaptive blending."""
        self.current_timestep = timestep

    def reset(self):
        """Reset the aesthetic replicator."""
        self.remove_unet_hooks()
        self.reference_latent = None
        self.reference_features = {}
        self.reference_activations.clear()
        self.layer_activations.clear()

    def set_enabled(self, enabled: bool):
        """Enable or disable aesthetic replication."""
        self.enabled = enabled
        if not enabled:
            self.remove_unet_hooks()

    def update_parameters(self, aesthetic_strength: float = None, blend_mode: str = None):
        """Update replicator parameters during runtime."""
        if aesthetic_strength is not None:
            self.aesthetic_strength = aesthetic_strength
        if blend_mode is not None:
            self.blend_mode = blend_mode


# Global instance for integration
aesthetic_replicator = None


def initialize_aesthetic_replicator(aesthetic_strength: float = 0.5, blend_mode: str = 'aggressive'):
    """Initialize the global aesthetic replicator with aggressive settings."""
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


def hook_unet_for_aesthetic_replication(unet_model, reference_noise=None):
    """Hook UNet layers for aggressive aesthetic replication."""
    global aesthetic_replicator
    if aesthetic_replicator and aesthetic_replicator.enabled:
        aesthetic_replicator.hook_unet_layers(unet_model)
        
        # Extract reference activations if we have reference noise
        if reference_noise is not None:
            aesthetic_replicator.extract_reference_activations(unet_model, reference_noise)
        
        logger.info("[LFL] UNet hooked for aggressive aesthetic replication")
        return True
    return False


def unhook_unet_aesthetic_replication():
    """Remove UNet hooks for aesthetic replication."""
    global aesthetic_replicator
    if aesthetic_replicator:
        aesthetic_replicator.remove_unet_hooks()
        logger.info("[LFL] UNet hooks removed")


def set_aesthetic_timestep(timestep):
    """Update current timestep for adaptive blending."""
    global aesthetic_replicator
    if aesthetic_replicator:
        aesthetic_replicator.set_current_timestep(timestep)


def apply_aesthetic_replication(x: torch.Tensor, denoised: torch.Tensor) -> torch.Tensor:
    """Apply aesthetic replication if enabled (legacy callback method)."""
    global aesthetic_replicator
    if aesthetic_replicator and aesthetic_replicator.enabled:
        return aesthetic_replicator(x, denoised)
    return x


def reset_aesthetic_replicator():
    """Reset the aesthetic replicator."""
    global aesthetic_replicator
    if aesthetic_replicator:
        aesthetic_replicator.reset()
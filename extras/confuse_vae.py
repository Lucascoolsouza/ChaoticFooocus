import torch
import torch.nn.functional as F
import math
import random
from ldm_patched.modules.sd import VAE
import ldm_patched.modules.utils
import ldm_patched.modules.model_management

class ConfuseVAE(VAE):
    def __init__(self, sd=None, device=None, config=None, dtype=None, artistic_strength=0.0):
        super().__init__(sd=sd, device=device, config=config, dtype=dtype)
        self.artistic_strength = artistic_strength

    def _apply_artistic_confusion(self, samples):
        """Apply various artistic confusion effects to the latent samples"""
        if self.artistic_strength <= 0:
            return samples
        
        # Create a copy to avoid modifying the original
        confused_samples = samples.clone()
        
        # Scale the strength (0-10 range mapped to 0-1 for internal use)
        strength = min(self.artistic_strength / 10.0, 1.0)
        
        # 1. Latent Space Noise (subtle background texture)
        if strength > 0.1:
            noise_strength = strength * 0.3
            noise = torch.randn_like(confused_samples) * noise_strength
            confused_samples = confused_samples + noise
        
        # 2. Channel Mixing (color confusion)
        if strength > 0.2:
            mix_strength = strength * 0.5
            # Randomly mix channels
            if random.random() < mix_strength:
                # Swap some channels
                perm = torch.randperm(confused_samples.shape[1])
                confused_samples = confused_samples[:, perm]
        
        # 3. Spatial Distortion (geometric confusion)
        if strength > 0.3:
            distort_strength = strength * 0.4
            if random.random() < distort_strength:
                # Apply slight rotation or shearing in latent space
                angle = (random.random() - 0.5) * distort_strength * 0.2
                cos_a, sin_a = math.cos(angle), math.sin(angle)
                
                # Create rotation matrix
                rotation_matrix = torch.tensor([
                    [cos_a, -sin_a, 0],
                    [sin_a, cos_a, 0]
                ], dtype=confused_samples.dtype, device=confused_samples.device)
                
                # Apply rotation to spatial dimensions
                grid = F.affine_grid(
                    rotation_matrix.unsqueeze(0), 
                    confused_samples.shape, 
                    align_corners=False
                )
                confused_samples = F.grid_sample(
                    confused_samples, 
                    grid, 
                    mode='bilinear', 
                    padding_mode='reflection',
                    align_corners=False
                )
        
        # 4. Frequency Domain Confusion (texture distortion)
        if strength > 0.4:
            freq_strength = strength * 0.3
            if random.random() < freq_strength:
                # Apply FFT-based distortion
                fft_samples = torch.fft.fft2(confused_samples)
                
                # Add phase noise
                phase_noise = torch.randn_like(fft_samples.real) * freq_strength * 0.1
                fft_samples = fft_samples * torch.exp(1j * phase_noise)
                
                confused_samples = torch.fft.ifft2(fft_samples).real
        
        # 5. Quantization Effects (posterization-like)
        if strength > 0.5:
            quant_strength = strength * 0.6
            if random.random() < quant_strength:
                # Quantize latent values
                quantization_levels = max(2, int(16 * (1 - quant_strength)))
                confused_samples = torch.round(confused_samples * quantization_levels) / quantization_levels
        
        # 6. Contrast and Saturation Confusion
        if strength > 0.6:
            contrast_strength = strength * 0.4
            # Random contrast adjustment per channel
            for c in range(confused_samples.shape[1]):
                if random.random() < contrast_strength:
                    contrast_factor = 1.0 + (random.random() - 0.5) * contrast_strength
                    mean_val = confused_samples[:, c:c+1].mean()
                    confused_samples[:, c:c+1] = (confused_samples[:, c:c+1] - mean_val) * contrast_factor + mean_val
        
        # 7. Latent Space Warping (advanced geometric distortion)
        if strength > 0.7:
            warp_strength = strength * 0.3
            if random.random() < warp_strength:
                # Create a warping field
                h, w = confused_samples.shape[-2:]
                warp_field = torch.randn(1, h, w, 2, device=confused_samples.device) * warp_strength * 0.05
                
                # Create base grid
                base_grid = torch.stack(torch.meshgrid(
                    torch.linspace(-1, 1, h, device=confused_samples.device),
                    torch.linspace(-1, 1, w, device=confused_samples.device),
                    indexing='ij'
                ), dim=-1).unsqueeze(0)
                
                # Apply warping
                warped_grid = base_grid + warp_field
                confused_samples = F.grid_sample(
                    confused_samples, 
                    warped_grid, 
                    mode='bilinear', 
                    padding_mode='reflection',
                    align_corners=False
                )
        
        # 8. Extreme Confusion (chaos mode)
        if strength > 0.8:
            chaos_strength = (strength - 0.8) * 5  # 0.8-1.0 maps to 0-1
            if random.random() < chaos_strength:
                # Mix with random latent patterns
                chaos_pattern = torch.randn_like(confused_samples) * chaos_strength * 0.2
                confused_samples = confused_samples * (1 - chaos_strength * 0.3) + chaos_pattern
                
                # Random channel shuffling
                if random.random() < chaos_strength:
                    for b in range(confused_samples.shape[0]):
                        perm = torch.randperm(confused_samples.shape[1])
                        confused_samples[b] = confused_samples[b, perm]
        
        return confused_samples

    def decode(self, samples_in):
        """Decode latent samples with artistic confusion effects"""
        if self.artistic_strength > 0:
            samples_in = self._apply_artistic_confusion(samples_in)
        
        return super().decode(samples_in)

    def decode_tiled(self, samples, tile_x=64, tile_y=64, overlap=16):
        """Decode tiled samples with artistic confusion effects"""
        if self.artistic_strength > 0:
            samples = self._apply_artistic_confusion(samples)
        
        return super().decode_tiled(samples, tile_x, tile_y, overlap)

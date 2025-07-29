import torch
from ldm_patched.modules.sd import VAE
import ldm_patched.modules.utils
import ldm_patched.modules.model_management

class ConfuseVAE(VAE):
    def __init__(self, sd=None, device=None, config=None, dtype=None, artistic_strength=0.0):
        super().__init__(sd=sd, device=device, config=config, dtype=dtype)
        self.artistic_strength = artistic_strength

    def decode(self, samples_in):
        if self.artistic_strength > 0:
            # Apply artistic effect: add noise to latent samples
            noise = torch.randn_like(samples_in) * self.artistic_strength
            samples_in = samples_in + noise

            # Optional: Add more complex artistic effects here, e.g.,
            # - Quantization: samples_in = torch.round(samples_in / quantization_factor) * quantization_factor
            # - Color manipulation (requires decoding first, then re-encoding or direct pixel manipulation)

        return super().decode(samples_in)

    def decode_tiled(self, samples, tile_x=64, tile_y=64, overlap = 16):
        if self.artistic_strength > 0:
            # Apply artistic effect: add noise to latent samples
            noise = torch.randn_like(samples) * self.artistic_strength
            samples = samples + noise
        return super().decode_tiled(samples, tile_x, tile_y, overlap)

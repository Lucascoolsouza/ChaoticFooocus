import os
import random
import hashlib
import json
import torch
import numpy as np
from PIL import Image, ImageOps, ImageSequence, PngImagePlugin
import ldm_patched.utils.path_utils
import ldm_patched.modules.model_management

# --- NAG logic integration (add this import, or implement normalize_attention below) ---
# from ChenDarYen.Normalized-Attention-Guidance.nag import normalize_attention

def normalize_attention(tensor, eps=1e-6):
    """
    Simple normalization logic for attention maps.
    Replace or extend with the actual implementation from the NAG repo as needed.
    """
    norm = tensor.norm(p=2, dim=-1, keepdim=True)
    return tensor / (norm + eps)

# --- Classes for image/latent processing (existing code preserved) ---

MAX_RESOLUTION = 2048  # adjust as needed

class EmptyLatentImage:
    def __init__(self):
        self.device = ldm_patched.modules.model_management.intermediate_device()

    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "width": ("INT", {"default": 512, "min": 16, "max": MAX_RESOLUTION, "step": 8}),
                              "height": ("INT", {"default": 512, "min": 16, "max": MAX_RESOLUTION, "step": 8}),
                              "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096})}}
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "generate"

    CATEGORY = "latent"

    def generate(self, width, height, batch_size=1):
        latent = torch.zeros([batch_size, 4, height // 8, width // 8], device=self.device)
        return ({"samples":latent}, )

class EmptyImage:
    def __init__(self, device="cpu"):
        self.device = device

    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "width": ("INT", {"default": 512, "min": 1, "max": MAX_RESOLUTION, "step": 1}),
                              "height": ("INT", {"default": 512, "min": 1, "max": MAX_RESOLUTION, "step": 1}),
                              "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
                              "color": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFF, "step": 1, "display": "color"}),
                              }}
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate"

    CATEGORY = "image"

    def generate(self, width, height, batch_size=1, color=0):
        r = torch.full([batch_size, height, width, 1], ((color >> 16) & 0xFF) / 0xFF)
        g = torch.full([batch_size, height, width, 1], ((color >> 8) & 0xFF) / 0xFF)
        b = torch.full([batch_size, height, width, 1], ((color) & 0xFF) / 0xFF)
        return (torch.cat((r, g, b), dim=-1), )

# --- Main SDXL pipeline entry point with NAG integration ---

class StableDiffusionXLPipelinePatched:
    def __init__(self, unet, tokenizer, vae, device="cuda"):
        self.unet = unet
        self.tokenizer = tokenizer
        self.vae = vae
        self.device = device

    def encode_prompt(self, prompt):
        if prompt is None:
            return None
        # Placeholder for actual encoding logic (tokenizer, text encoder, etc.)
        # Should return a tensor or suitable conditioning vector
        # Example (pseudo):
        # return self.tokenizer.encode(prompt).to(self.device)
        return torch.randn(1, 77, 768, device=self.device)  # Dummy: replace with actual

    def apply_nag_to_conditioning(self, cond):
        # Use NAG normalization (replace with real function from NAG repo if available)
        return normalize_attention(cond)

    def diffusion_process(self, pos_cond, neg_cond, width=1024, height=1024, steps=30, **kwargs):
        # Placeholder for actual diffusion process
        # Pass both positive and negative conditioning to UNet/denoising loop
        # Example (pseudo):
        # latents = sample_latents(...)
        # for t in range(steps):
        #     latents = self.unet(latents, pos_cond, neg_cond, t)
        # return self.vae.decode(latents)
        # Here, just return a dummy image for structure

        dummy_img = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
        return Image.fromarray(dummy_img)

    def generate(self, prompt, negative_prompt=None, width=1024, height=1024, steps=30, **kwargs):
        # Step 1: Encode prompts
        pos_cond = self.encode_prompt(prompt)
        neg_cond = self.encode_prompt(negative_prompt) if negative_prompt else None

        # Step 2: Apply NAG to the negative prompt
        if neg_cond is not None:
            neg_cond = self.apply_nag_to_conditioning(neg_cond)

        # Step 3: Pass both to diffusion process
        output = self.diffusion_process(
            pos_cond, neg_cond, width=width, height=height, steps=steps, **kwargs
        )
        return output

# --- Image save & preview classes ---

class SaveImage:
    def __init__(self):
        self.output_dir = ldm_patched.utils.path_utils.get_output_directory()
        self.type = "output"
        self.prefix_append = ""
        self.compress_level = 4

    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"images": ("IMAGE", ), },
                "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
                }

    def __call__(self, images, prompt=None, extra_pnginfo=None):
        filename_prefix = "output"
        filename_prefix += self.prefix_append
        full_output_folder, filename, counter, subfolder, filename_prefix = ldm_patched.utils.path_utils.get_save_image_path(
            filename_prefix, self.output_dir, images[0].shape[1], images[0].shape[0]
        )
        results = []
        for image in images:
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            metadata = None
            if not hasattr(self, "disable_server_info") or not self.disable_server_info:
                metadata = PngImagePlugin.PngInfo()
                if prompt is not None:
                    metadata.add_text("prompt", json.dumps(prompt))
                if extra_pnginfo is not None:
                    for x in extra_pnginfo:
                        metadata.add_text(x, json.dumps(extra_pnginfo[x]))

            file = f"{filename}_{counter:05}_.png"
            img.save(os.path.join(full_output_folder, file), pnginfo=metadata, compress_level=self.compress_level)
            results.append({
                "filename": file,
                "subfolder": subfolder,
                "type": self.type
            })
            counter += 1

        return { "ui": { "images": results } }

class PreviewImage(SaveImage):
    def __init__(self):
        self.output_dir = ldm_patched.utils.path_utils.get_temp_directory()
        self.type = "temp"
        self.prefix_append = "_temp_" + ''.join(random.choice("abcdefghijklmnopqrstupvxyz") for x in range(5))
        self.compress_level = 1

    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"images": ("IMAGE", ), },
                "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
                }

# --- Class registry for node-based pipeline ---

PIPELINE_NODES = {
    "EmptyLatentImage": EmptyLatentImage,
    "EmptyImage": EmptyImage,
    "SaveImage": SaveImage,
    "PreviewImage": PreviewImage,
    "StableDiffusionXLPipelinePatched": StableDiffusionXLPipelinePatched,
    # ... other nodes as needed ...
}

# --- Utility functions and additional classes are preserved as in original file ---
# (If there are other classes, methods, or logic, copy as needed below.)

# --- End of file ---
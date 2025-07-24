from collections import OrderedDict
import numpy as np

import math
import modules.core as core
import torch
from ldm_patched.contrib.external_upscale_model import ImageUpscaleWithModel
from ldm_patched.pfn.architecture.RRDB import RRDBNet as ESRGAN
from modules.config import downloading_upscale_model, downloading_ultrasharp_model, downloading_realistic_rescaler_model
import modules.flags
from modules.seamless_tiling import process_seamless_enhancement

import sys
import os
latent_upscale_path = os.path.join(os.path.dirname(__file__), '../extras/latent-upscale-main/scripts')
if latent_upscale_path not in sys.path:
    sys.path.append(latent_upscale_path)
try:
    from latent_upscale import Script as LatentUpscaleScript
except ImportError:
    LatentUpscaleScript = None

opImageUpscaleWithModel = ImageUpscaleWithModel()
model_default = None
model_ultrasharp = None
model_realistic_rescaler = None

def perform_upscale_without_tiling(img, model_name, model_var, download_func, async_task=None, vae=None):
    global model_realistic_rescaler, model_ultrasharp

    print(f'Processing {model_name} on image shape {img.shape}...')

    if model_name == "UltraSharp" and async_task is not None:
        # Enhance prompt for sharpening
        sharpening_keywords = "sharp, detailed, crisp, high quality, ultra detailed"
        if async_task.prompt and sharpening_keywords not in async_task.prompt:
            async_task.prompt = f"{async_task.prompt}, {sharpening_keywords}"
        # Add negative prompt to avoid blur
        blur_negative = "blurry, soft, out of focus, low quality"
        if async_task.negative_prompt and blur_negative not in async_task.negative_prompt:
            async_task.negative_prompt = f"{async_task.negative_prompt}, {blur_negative}"
        elif not async_task.negative_prompt:
            async_task.negative_prompt = blur_negative

    # Load model if not already loaded
    if model_var[0] is None:
        model_filename = download_func()
        print(f"Loading {model_name} model from {model_filename}")
        sd = torch.load(model_filename, weights_only=True)
        sdo = OrderedDict()
        for k, v in sd.items():
            sdo[k.replace('residual_block_', 'RDB')] = v
        del sd
        model_var[0] = ESRGAN(sdo)
        model_var[0].cpu()
        model_var[0].eval()

    # Convert to PyTorch tensor
    img_tensor = core.numpy_to_pytorch(img)

    try:
        if torch.cuda.is_available():
            model_var[0] = model_var[0].cuda()

        result = opImageUpscaleWithModel.upscale(model_var[0], img_tensor.permute(0, 3, 1, 2))[0]

        if result.shape[1] == 3:
            pass
        elif result.shape[1] == 4:          # SD latent – decode only this
            result = core.decode_vae(vae, {'samples': result})[0]
        else:                               # 4096 or 64 → already pixel-space
            try:
                final_layer = model_var[0].model[-1]
                expected_in_channels = final_layer.in_channels
                if result.shape[1] == expected_in_channels:
                    result = final_layer(result)
                else:
                    print(f"[Upscaler] ⚠️ Final layer expected {expected_in_channels} channels, got {result.shape[1]}. Skipping final conv layer.")
                result = torch.clamp(result, 0, 1)
            except Exception as conv_e:
                print(f"[Upscaler] ⚠️ Failed applying final conv layer: {str(conv_e)}. Skipping...")
                result = torch.clamp(result, 0, 1)

        model_var[0].cpu()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return core.pytorch_to_numpy(result)[0]
    except Exception as e:
        model_var[0].cpu()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise e

def perform_latent_upscale(img, async_task=None, vae_model=None):
    # Integrate latent upscale logic
    if LatentUpscaleScript is None:
        print('LatentUpscaleScript not found. Returning original image.')
        return img
    # Create a dummy processing object with required attributes
    class DummyProcessing:
        def __init__(self, init_images):
            self.init_images = [img]
            self.sampler_name = 'euler'
            self.sd_model = None
            self.image_mask = None
            self.inpainting_mask_invert = False
            self.mask_blur_x = 0
            self.mask_blur_y = 0
            self.inpaint_full_res = False
            self.inpaint_full_res_padding = 0
            self.resize_mode = 0
            self.width = img.shape[1]
            self.height = img.shape[0]
            self.latent_mask = None
            self.color_corrections = None
            self.overlay_images = []
            self.mask_for_overlay = None
            self.paste_to = None
    # Determine upscaling factor
    scale_map = {"1x": 1.0, "1.5x": 1.5, "2x": 2.0, "4x": 4.0}
    upscale_size = getattr(async_task, 'latent_upscale_size', '2x')
    scale = scale_map.get(upscale_size, 2.0)
    # Resize image if needed
    if scale != 1.0:
        from PIL import Image
        if isinstance(img, np.ndarray):
            pil_img = Image.fromarray(img.astype(np.uint8))
        else:
            pil_img = img
        new_w = int(pil_img.width * scale)
        new_h = int(pil_img.height * scale)
        pil_img = pil_img.resize((new_w, new_h), Image.LANCZOS)
        img = np.array(pil_img)
    p = DummyProcessing(img)
    script = LatentUpscaleScript()
    # Use options from async_task if available
    upscale_method = getattr(async_task, 'latent_upscale_method', 'bilinear')
    scheduler = getattr(async_task, 'latent_upscale_scheduler', 'normal')
    # Run the latent upscale script
    try:
        result = script.run(p, upscale_method, scheduler)
        # The script's run method may return a Processed object or image, handle accordingly
        if hasattr(result, 'images') and isinstance(result.images, list) and len(result.images) > 0:
            return result.images[0]
        elif isinstance(result, (list, tuple)) and len(result) > 0:
            return result[0]
        else:
            return img
    except Exception as e:
        print(f'Latent Upscale failed: {e}')
        return img

def perform_seamless_tiling(img, async_task=None):
    """
    Apply seamless tiling to make the image tileable.
    """
    from PIL import Image
    
    print(f'Applying seamless tiling to image with shape {str(img.shape)} ...')
    
    # Convert numpy array to PIL Image
    if isinstance(img, np.ndarray):
        pil_img = Image.fromarray(img.astype(np.uint8))
    else:
        pil_img = img
    
    # Get tiling method from async_task if available, otherwise use default
    method = 'blend'  # Default method
    overlap_ratio = 0.15  # Default overlap ratio
    
    if async_task and hasattr(async_task, 'seamless_tiling_method'):
        method = async_task.seamless_tiling_method
    if async_task and hasattr(async_task, 'seamless_tiling_overlap'):
        overlap_ratio = async_task.seamless_tiling_overlap
    
    # Process seamless tiling
    result = process_seamless_enhancement(
        pil_img, 
        method=method, 
        overlap_ratio=overlap_ratio,
        create_preview=False
    )
    
    # Convert back to numpy array
    seamless_img = np.array(result['result'])
    
    print(f'Seamless tiling completed. Output shape: {str(seamless_img.shape)}')
    
    return seamless_img


def perform_upscale(img, method, async_task=None, vae_model=None):
    global model_default, model_ultrasharp, model_realistic_rescaler, final_vae

    print(f'Upscaling image with shape {str(img.shape)} using method {method} ...')

    method = method.casefold()

    if method == modules.flags.ultrasharp.casefold():
        return perform_upscale_without_tiling(img, "UltraSharp", [model_ultrasharp], downloading_ultrasharp_model, async_task=async_task, vae=vae_model)
    elif method == modules.flags.realistic_rescaler.casefold():
        return perform_upscale_without_tiling(img, "Realistic Rescaler", [model_realistic_rescaler], downloading_realistic_rescaler_model, vae=vae_model)
    elif method == modules.flags.latent_upscale.casefold():
        return perform_latent_upscale(img, async_task=async_task, vae_model=vae_model)
    elif method == modules.flags.seamless_tiling.casefold():
        return perform_seamless_tiling(img, async_task=async_task)
    else: # Default upscaling
        if model_default is None:
            model_filename = downloading_upscale_model()
            sd = torch.load(model_filename, weights_only=True)
            sdo = OrderedDict()
            for k, v in sd.items():
                sdo[k.replace('residual_block_', 'RDB')] = v
            del sd
            model_default = ESRGAN(sdo)
            model_default.cpu()
            model_default.eval()
        model_to_use = model_default

    try:
        img = core.numpy_to_pytorch(img)
        
        # Move model to GPU temporarily for processing
        if torch.cuda.is_available():
            model_to_use = model_to_use.cuda()
        
        img_tensor = img.permute(0, 3, 1, 2)  # NHWC → NCHW
        result = opImageUpscaleWithModel.upscale(model_to_use, img_tensor)[0]
        
        # Move model back to CPU to free VRAM
        model_to_use.cpu()
        
        # Clear VRAM cache after processing
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        img = core.pytorch_to_numpy(img)[0]
        
        return img
        
    except Exception as e:
        print(f"Upscaling failed: {str(e)}")
        # Ensure model is moved back to CPU on error
        if 'model_to_use' in locals():
            model_to_use.cpu()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise e
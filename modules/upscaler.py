from collections import OrderedDict
import numpy as np

import modules.core as core
import torch
from ldm_patched.contrib.external_upscale_model import ImageUpscaleWithModel
from ldm_patched.pfn.architecture.RRDB import RRDBNet as ESRGAN
from modules.config import downloading_upscale_model, downloading_ultrasharp_model, downloading_web_photo_model, downloading_realistic_rescaler_model, downloading_skin_contrast_model, downloading_four_x_nomos_model, downloading_faces_model
import modules.flags

opImageUpscaleWithModel = ImageUpscaleWithModel()
model_default = None
model_ultrasharp = None
model_web_photo = None
model_realistic_rescaler = None
model_skin_contrast = None
model_four_x_nomos = None
model_faces = None


def perform_ultrasharp_tiled(img, tile_size=512, overlap=64):
    """
    Perform UltraSharp processing with tiling for memory efficiency
    """
    global model_ultrasharp
    
    print(f'Processing UltraSharp with tiling (tile_size={tile_size}, overlap={overlap}) on image shape {img.shape}...')
    
    # Load UltraSharp model if not already loaded
    if model_ultrasharp is None:
        model_filename = downloading_ultrasharp_model()
        print(f"Loading UltraSharp model from {model_filename}")
        sd = torch.load(model_filename, weights_only=True)
        sdo = OrderedDict()
        for k, v in sd.items():
            sdo[k.replace('residual_block_', 'RDB')] = v
        del sd
        model_ultrasharp = ESRGAN(sdo)
        model_ultrasharp.cpu()
        model_ultrasharp.eval()
    
    # Convert to PyTorch tensor
    img_tensor = core.numpy_to_pytorch(img)
    
    # Get image dimensions
    _, _, h, w = img_tensor.shape
    
    # If image is small enough, process without tiling
    if h <= tile_size and w <= tile_size:
        print("Image small enough, processing without tiling")
        try:
            if torch.cuda.is_available():
                model_ultrasharp = model_ultrasharp.cuda()
            
            result = opImageUpscaleWithModel.upscale(model_ultrasharp, img_tensor)[0]
            
            model_ultrasharp.cpu()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return core.pytorch_to_numpy(result)[0]
        except Exception as e:
            model_ultrasharp.cpu()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise e
    
    # Process with tiling
    print(f"Processing with tiling: {h}x{w} -> tiles of {tile_size}x{tile_size}")
    
    # Calculate output dimensions (4x upscale)
    scale_factor = 4
    out_h, out_w = h * scale_factor, w * scale_factor
    
    # Initialize output tensor
    output = torch.zeros((1, 3, out_h, out_w), dtype=img_tensor.dtype)
    
    try:
        if torch.cuda.is_available():
            model_ultrasharp = model_ultrasharp.cuda()
        
        # Process tiles
        for y in range(0, h, tile_size - overlap):
            for x in range(0, w, tile_size - overlap):
                # Calculate tile boundaries
                y_end = min(y + tile_size, h)
                x_end = min(x + tile_size, w)
                
                # Extract tile
                tile = img_tensor[:, :, y:y_end, x:x_end]
                
                print(f"Processing tile [{y}:{y_end}, {x}:{x_end}] -> shape {tile.shape}")
                
                # Process tile
                tile_result = opImageUpscaleWithModel.upscale(model_ultrasharp, tile)[0]
                
                # Calculate output position
                out_y = y * scale_factor
                out_x = x * scale_factor
                out_y_end = min(out_y + tile_result.shape[2], out_h)
                out_x_end = min(out_x + tile_result.shape[3], out_w)
                
                # Handle overlap blending
                if overlap > 0 and (y > 0 or x > 0):
                    # Simple averaging for overlapping regions
                    overlap_scaled = overlap * scale_factor
                    
                    # Blend with existing content
                    existing = output[:, :, out_y:out_y_end, out_x:out_x_end]
                    new_content = tile_result[:, :, :out_y_end-out_y, :out_x_end-out_x]
                    
                    # Create blend mask for smooth transitions
                    blend_mask = torch.ones_like(new_content)
                    
                    if y > 0:  # Top overlap
                        for i in range(min(overlap_scaled, new_content.shape[2])):
                            alpha = i / overlap_scaled
                            blend_mask[:, :, i, :] = alpha
                    
                    if x > 0:  # Left overlap
                        for i in range(min(overlap_scaled, new_content.shape[3])):
                            alpha = i / overlap_scaled
                            blend_mask[:, :, :, i] = alpha
                    
                    # Apply blending
                    output[:, :, out_y:out_y_end, out_x:out_x_end] = (
                        existing * (1 - blend_mask) + new_content * blend_mask
                    )
                else:
                    # No overlap, direct copy
                    output[:, :, out_y:out_y_end, out_x:out_x_end] = tile_result[:, :, :out_y_end-out_y, :out_x_end-out_x]
                
                # Clear tile from memory
                del tile, tile_result
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        # Move model back to CPU
        model_ultrasharp.cpu()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return core.pytorch_to_numpy(output)[0]
        
    except Exception as e:
        print(f"Tiled UltraSharp processing failed: {str(e)}")
        model_ultrasharp.cpu()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise e


def apply_ultrasharp_vary(img, tile_size=512, overlap=64):
    """
    Apply UltraSharp as a vary filter with tiling support
    This function can be called from the pipeline for UltraSharp vary processing
    """
    try:
        return perform_ultrasharp_tiled(img, tile_size=tile_size, overlap=overlap)
    except Exception as e:
        print(f"UltraSharp vary processing failed: {e}")
        print("Falling back to original image")
        return img


def perform_tiled_upscale(img, model_name, model_var, download_func, tile_size=512, overlap=64, async_task=None, vae=None):
    global model_web_photo, model_realistic_rescaler, model_skin_contrast, model_four_x_nomos, model_faces, model_ultrasharp

    print(f'Processing {model_name} with tiling (tile_size={tile_size}, overlap={overlap}) on image shape {img.shape}...')

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

    # Get image dimensions
    _, _, h, w = img_tensor.shape

    # If image is small enough, process without tiling
    if h <= tile_size and w <= tile_size:
        print("Image small enough, processing without tiling")
        try:
            if torch.cuda.is_available():
                model_var[0] = model_var[0].cuda()

            result = opImageUpscaleWithModel.upscale(model_var[0], img_tensor.permute(0, 3, 1, 2))[0]

            if result.shape[1] == 4:  # If the output is a latent (4 channels)
                if vae is None:
                    raise ValueError(f"Upscaler '{model_name}' produced a latent (4 channels) but no VAE was provided for decoding.")
                result = core.decode_vae(vae, {'samples': result})[0]
            elif result.shape[1] != 3:  # If it's not 3 channels (RGB) or 4 channels (latent)
                raise ValueError(f"Upscaler '{model_name}' produced an unexpected number of channels: {result.shape[1]}. Expected 3 (RGB) or 4 (latent).")

            model_var[0].cpu()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            return core.pytorch_to_numpy(result)[0]
        except Exception as e:
            model_var[0].cpu()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise e

    # Process with tiling
    print(f"Processing with tiling: {h}x{w} -> tiles of {tile_size}x{tile_size}")

    # Calculate output dimensions (4x upscale)
    scale_factor = 4
    out_h, out_w = h * scale_factor, w * scale_factor

    # Initialize output tensor
    output = torch.zeros((1, 3, out_h, out_w), dtype=img_tensor.dtype)

    try:
        if torch.cuda.is_available():
            model_var[0] = model_var[0].cuda()

        # Process tiles
        for y in range(0, h, tile_size - overlap):
            for x in range(0, w, tile_size - overlap):
                # Calculate tile boundaries
                y_end = min(y + tile_size, h)
                x_end = min(x + tile_size, w)

                # Extract tile
                tile = img_tensor[:, :, y:y_end, x:x_end]

                print(f"Processing tile [{y}:{y_end}, {x}:{x_end}] -> shape {tile.shape}")

                # Process tile
                tile_result = opImageUpscaleWithModel.upscale(model_var[0], tile.permute(0, 3, 1, 2))[0]

                if tile_result.shape[1] == 4:  # If the output is a latent (4 channels)
                    if vae is None:
                        raise ValueError(f"Upscaler '{model_name}' produced a latent (4 channels) but no VAE was provided for decoding.")
                    tile_result = core.decode_vae(vae, {'samples': tile_result})[0]
                elif tile_result.shape[1] != 3:  # If it's not 3 channels (RGB) or 4 channels (latent)
                    raise ValueError(f"Upscaler '{model_name}' produced an unexpected number of channels: {tile_result.shape[1]}. Expected 3 (RGB) or 4 (latent).")

                # Calculate output position
                out_y = y * scale_factor
                out_x = x * scale_factor
                out_y_end = min(out_y + tile_result.shape[2], out_h)
                out_x_end = min(out_x + tile_result.shape[3], out_w)

                # Handle overlap blending
                if overlap > 0 and (y > 0 or x > 0):
                    # Simple averaging for overlapping regions
                    overlap_scaled = overlap * scale_factor

                    # Blend with existing content
                    existing = output[:, :, out_y:out_y_end, out_x:out_x_end]
                    new_content = tile_result[:, :, :out_y_end-out_y, :out_x_end-out_x]

                    # Create blend mask for smooth transitions
                    blend_mask = torch.ones_like(new_content)

                    if y > 0:  # Top overlap
                        for i in range(min(overlap_scaled, new_content.shape[2])):
                            alpha = i / overlap_scaled
                            blend_mask[:, :, i, :] = alpha

                    if x > 0:  # Left overlap
                        for i in range(min(overlap_scaled, new_content.shape[3])):
                            alpha = i / overlap_scaled
                            blend_mask[:, :, :, i] = alpha

                    # Apply blending
                    output[:, :, out_y:out_y_end, out_x:out_x_end] = (
                        existing * (1 - blend_mask) + new_content * blend_mask
                    )
                else:
                    # No overlap, direct copy
                    output[:, :, out_y:out_y_end, out_x:out_x_end] = tile_result[:, :, :out_y_end-out_y, :out_x_end-out_x]

                # Clear tile from memory
                del tile, tile_result
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # Move model back to CPU
        model_var[0].cpu()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return core.pytorch_to_numpy(output)[0]

    except Exception as e:
        print(f"Tiled {model_name} processing failed: {str(e)}")
        model_var[0].cpu()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise e


def perform_upscale(img, method, async_task=None, vae_model=None):
    global model_default, model_ultrasharp, model_web_photo, model_realistic_rescaler, model_skin_contrast, model_four_x_nomos, model_faces, final_vae

    print(f'Upscaling image with shape {str(img.shape)} using method {method} ...')

    method = method.casefold()

    if method == modules.flags.ultrasharp.casefold():
        return perform_tiled_upscale(img, "UltraSharp", [model_ultrasharp], downloading_ultrasharp_model, async_task=async_task, vae=vae_model)
    elif method == modules.flags.web_photo.casefold():
        return perform_tiled_upscale(img, "Web Photo", [model_web_photo], downloading_web_photo_model, vae=vae_model)
    elif method == modules.flags.realistic_rescaler.casefold():
        return perform_tiled_upscale(img, "Realistic Rescaler", [model_realistic_rescaler], downloading_realistic_rescaler_model, vae=vae_model)
    elif method == modules.flags.skin_contrast.casefold():
        return perform_tiled_upscale(img, "Skin Contrast", [model_skin_contrast], downloading_skin_contrast_model, vae=vae_model)
    elif method == modules.flags.four_x_nomos.casefold():
        return perform_tiled_upscale(img, "4xNomos", [model_four_x_nomos], downloading_four_x_nomos_model, vae=vae_model)
    elif method == modules.flags.faces.casefold():
        return perform_tiled_upscale(img, "Faces", [model_faces], downloading_faces_model, vae=vae_model)
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
        
        img = opImageUpscaleWithModel.upscale(model_to_use, img)[0]
        
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

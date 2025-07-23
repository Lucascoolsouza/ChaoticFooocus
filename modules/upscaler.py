from collections import OrderedDict

import modules.core as core
import torch
from ldm_patched.contrib.external_upscale_model import ImageUpscaleWithModel
from ldm_patched.pfn.architecture.RRDB import RRDBNet as ESRGAN
from modules.config import downloading_upscale_model, downloading_ultrasharp_model
import modules.flags

opImageUpscaleWithModel = ImageUpscaleWithModel()
model_default = None
model_ultrasharp = None


def perform_upscale(img, method):
    global model_default, model_ultrasharp

    print(f'Upscaling image with shape {str(img.shape)} using method {method} ...')

    if method == modules.flags.ultrasharp:
        # Ultrasharp now works as a vary filter, not an upscaler
        # This should not be called for ultrasharp anymore
        print("Warning: Ultrasharp is now a vary filter, not an upscaler!")
        return img  # Return original image unchanged
        
    # Handle other upscale methods (default upscaling)
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

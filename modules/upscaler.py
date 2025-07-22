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
        if model_ultrasharp is None:
            model_filename = downloading_ultrasharp_model()
            sd = torch.load(model_filename, weights_only=True)
            sdo = OrderedDict()
            for k, v in sd.items():
                sdo[k.replace('residual_block_', 'RDB')] = v
            del sd
            model_ultrasharp = ESRGAN(sdo)
            model_ultrasharp.cpu()
            model_ultrasharp.eval()
        model_to_use = model_ultrasharp
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

    img = core.numpy_to_pytorch(img)
    img = opImageUpscaleWithModel.upscale(model_to_use, img)[0]
    img = core.pytorch_to_numpy(img)[0]

    return img

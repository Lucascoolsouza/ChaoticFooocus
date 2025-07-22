import os
import torch
import ldm_patched.modules.model_management as model_management

from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from modules.model_loader import load_file_from_url
from modules.config import path_clip_vision
from ldm_patched.modules.model_patcher import ModelPatcher

# Import from transformers
from transformers import AutoProcessor, Blip2ForConditionalGeneration
from PIL import Image

blip_image_eval_size = 384
# blip_repo_root = os.path.join(os.path.dirname(__file__), 'BLIP') # No longer needed

class Interrogator:
    def __init__(self):
        self.blip_model = None
        self.processor = None # Add processor
        self.load_device = torch.device('cpu')
        self.offload_device = torch.device('cpu')
        self.dtype = torch.float32

    @torch.no_grad()
    @torch.inference_mode()
    def interrogate(self, img_rgb):
        if self.blip_model is None:
            # Load BLIP-2 model and processor from Hugging Face
            model_name = "Salesforce/blip2-flan-t5-xl" # Using a specific BLIP-2 model
            self.processor = AutoProcessor.from_pretrained(model_name, local_files_only=False)
            model = Blip2ForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.float16, local_files_only=False)

            model.eval()

            self.load_device = model_management.text_encoder_device()
            self.offload_device = model_management.text_encoder_offload_device()
            self.dtype = torch.float16 if model_management.should_use_fp16(device=self.load_device) else torch.float32

            model.to(self.offload_device)

            if self.dtype == torch.float16:
                model.half()

            self.blip_model = ModelPatcher(model, load_device=self.load_device, offload_device=self.offload_device)

        model_management.load_model_gpu(self.blip_model)

        # Convert img_rgb (numpy array) to PIL Image
        pil_image = Image.fromarray(img_rgb)

        # Prepare inputs using the processor
        inputs = self.processor(images=pil_image, return_tensors="pt").to(device=self.load_device, dtype=self.dtype)

        # Generate caption
        generated_ids = self.blip_model.model.generate(**inputs, max_new_tokens=120)
        caption = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

        return caption


default_interrogator = Interrogator().interrogate

# Simple CLIP Guidance - Based on Original Method
import torch
import torch.nn.functional as F
import clip
import logging
from torchvision import transforms

logger = logging.getLogger(__name__)

class SimpleMakeCutouts(torch.nn.Module):
    """Simple cutout class like the original, using torchvision transforms"""
    def __init__(self, cut_size, cutn):
        super().__init__()
        self.cut_size = cut_size
        self.cutn = cutn
        # Use torchvision transforms like the original
        self.augs = transforms.Compose([
            transforms.RandomResizedCrop(cut_size, scale=(0.8, 1.0)),
            transforms.RandomPerspective(fill=0, p=0.7, distortion_scale=0.5),
            transforms.RandomHorizontalFlip(),
        ])

    def forward(self, input):
        return torch.cat([self.augs(input) for _ in range(self.cutn)], dim=0)

def run_clip_guidance_loop(
    latent, vae, clip_model, clip_preprocess, text_prompt, async_task,
    steps=50, disco_scale=1.0, cutn=16, tv_scale=0.0, range_scale=0.0
):
    """
    Simple CLIP guidance like the original method - fast and clean
    """
    print("[Disco] Starting CLIP guidance (simple method)...")
    
    try:
        # Get device from latent
        latent_tensor = latent['samples']
        device = latent_tensor.device
        
        # 1. Prepare text embeddings (like original)
        try:
            text_tokens = clip.tokenize([text_prompt], truncate=True).to(device)
        except:
            # Fallback for long text
            words = text_prompt.split()[:50]
            text_tokens = clip.tokenize([" ".join(words)], truncate=True).to(device)
            
        with torch.no_grad():
            text_features = clip_model.encode_text(text_tokens).float()
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # 2. Initialize image from latent
        with torch.no_grad():
            init_image = vae.decode(latent_tensor)
            # Normalize to [0, 1]
            if init_image.shape[-1] <= 4 and init_image.shape[1] > 4:
                init_image = init_image.permute(0, 3, 1, 2)
            init_image = (init_image / 2 + 0.5).clamp(0, 1)
            
            # Ensure 3 channels
            if init_image.shape[1] > 3:
                init_image = init_image[:, :3, :, :]
            elif init_image.shape[1] == 1:
                init_image = init_image.repeat(1, 3, 1, 1)

        # 3. Set up optimization (exactly like original)
        cut_size = clip_model.visual.input_resolution
        make_cutouts = SimpleMakeCutouts(cut_size, cutn)
        
        # CLIP normalization (like original)
        normalize = transforms.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711)
        )
        
        # Create optimizable parameter (like original)
        image_tensor = torch.nn.Parameter(init_image.clone().detach().to(device))
        optimizer = torch.optim.Adam([image_tensor], lr=0.05)
        
        # CLIP loss function (exactly like original)
        def clip_loss(image_tensor, text_embed):
            cutouts = make_cutouts(image_tensor)
            cutouts = normalize(cutouts)
            image_features = clip_model.encode_image(cutouts).float()
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            similarity = (text_embed @ image_features.T).mean()
            return -similarity
        
        # 4. Optimization loop (exactly like original)
        for i in range(steps):
            optimizer.zero_grad()
            loss = clip_loss(image_tensor, text_features)
            loss.backward()
            optimizer.step()
            
            # Clamp to valid range (like original)
            with torch.no_grad():
                image_tensor.clamp_(0, 1)
            
            if i % 10 == 0:
                print(f"[Disco] Step {i}, Loss: {loss.item():.4f}")
                
                # Update progress
                if async_task is not None:
                    progress = int((i + 1) / steps * 100)
                    preview_image_np = (image_tensor.detach().permute(0, 2, 3, 1) * 255).clamp(0, 255).to(torch.uint8).cpu().numpy()[0]
                    async_task.yields.append(['preview', (progress, f'Disco Step {i+1}/{steps}', preview_image_np)])

        print("[Disco] CLIP optimization completed.")
        return latent

    except Exception as e:
        logger.error(f"CLIP guidance failed: {e}", exc_info=True)
        return latent
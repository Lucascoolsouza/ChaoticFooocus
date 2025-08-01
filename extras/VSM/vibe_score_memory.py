# vibe_memory.py
# Fooocus WebUI extension – Aesthetic Memory / VibeScore
# Uses a JSON file to remember liked/disliked CLIP embeddings and
# automatically steers new generations toward or away from those vibes.

import json
import os
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, List, Tuple
import logging
import torchvision.transforms as transforms
from PIL import Image

# Try to import CLIP, fallback gracefully if not available
try:
    import clip
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    print("[VibeMemory] Warning: CLIP not available. Install with: pip install git+https://github.com/openai/CLIP.git")

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Simple small helper: cosine similarity
# ------------------------------------------------------------------
cos = F.cosine_similarity


# ------------------------------------------------------------------
# VibeMemory class
# ------------------------------------------------------------------
class VibeMemory:
    """
    Manages a memory.json with liked/disliked CLIP vectors.
    Provides:
      - load / save
      - add like/dislike from PIL image or tensor
      - score (higher = more aligned with likes, less with dislikes)
    """

    def __init__(self,
                 memory_path: str = "memory.json",
                 clip_model_name: str = "ViT-B/32"):
        self.memory_path = Path(memory_path)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model = None
        self.preprocess = None

        # Load CLIP if available
        if CLIP_AVAILABLE:
            try:
                self.clip_model, self.preprocess = clip.load(clip_model_name,
                                                           device=self.device)
                self.clip_model.eval()
                logger.info(f"[VibeMemory] Loaded CLIP model: {clip_model_name}")
            except Exception as e:
                logger.warning(f"[VibeMemory] Failed to load CLIP: {e}")
                self.clip_model = None

        # In-memory store
        self.data = {"liked": [], "disliked": []}
        self._load()

    # ----------------------------------------------------------
    # I/O
    # ----------------------------------------------------------
    def _load(self):
        if self.memory_path.exists():
            with open(self.memory_path, "r", encoding="utf-8") as f:
                self.data = json.load(f)
            logger.info(f"[VibeMemory] Loaded {len(self.data['liked'])} liked, "
                        f"{len(self.data['disliked'])} disliked vectors.")

    def _save(self):
        self.memory_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.memory_path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2)
        logger.debug("[VibeMemory] Saved.")

    # ----------------------------------------------------------
    # Embedding helpers
    # ----------------------------------------------------------
    def image_to_embedding(self, image: Image.Image) -> List[float]:
        """Return CLIP embedding for a PIL Image."""
        if not self.clip_model:
            logger.warning("[VibeMemory] CLIP model not available")
            return [0.0] * 512  # Return dummy embedding
        
        tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            emb = self.clip_model.encode_image(tensor)
        return emb.squeeze().cpu().tolist()

    def tensor_to_embedding(self, tensor: torch.Tensor) -> List[float]:
        """Accepts a (3, H, W) tensor in [0,1] or [-1,1]."""
        if tensor.dim() == 4:
            tensor = tensor.squeeze(0)
        # Convert to PIL
        tensor = (tensor + 1) / 2 if tensor.min() < 0 else tensor
        tensor = tensor.clamp(0, 1)
        pil = transforms.ToPILImage()(tensor.cpu())
        return self.image_to_embedding(pil)

    # ----------------------------------------------------------
    # Public API
    # ----------------------------------------------------------
    def add_like(self, image_or_tensor):
        """Accept PIL.Image or torch.Tensor."""
        emb = self._dispatch(image_or_tensor)
        self.data["liked"].append(emb)
        self._save()

    def add_dislike(self, image_or_tensor):
        emb = self._dispatch(image_or_tensor)
        self.data["disliked"].append(emb)
        self._save()

    def _dispatch(self, obj):
        if isinstance(obj, Image.Image):
            return self.image_to_embedding(obj)
        elif isinstance(obj, torch.Tensor):
            return self.tensor_to_embedding(obj)
        else:
            raise TypeError("Need PIL.Image or torch.Tensor.")

    # ----------------------------------------------------------
    # Scoring
    # ----------------------------------------------------------
    def score(self, embedding: torch.Tensor) -> float:
        """
        Return scalar score:
          +large  -> very aligned with likes
          0       -> neutral
          -large  -> very aligned with dislikes
        """
        if not self.clip_model or not (self.data["liked"] or self.data["disliked"]):
            return 0.0

        emb = embedding.to(self.device)
        if emb.dim() == 1:
            emb = emb.unsqueeze(0)

        liked_tensor = torch.tensor(self.data["liked"],
                                    device=self.device) if self.data["liked"] else None
        disliked_tensor = torch.tensor(self.data["disliked"],
                                       device=self.device) if self.data["disliked"] else None

        score = 0.0
        if liked_tensor is not None:
            score += cos(emb, liked_tensor, dim=1).mean().item()
        if disliked_tensor is not None:
            score -= cos(emb, disliked_tensor, dim=1).mean().item()
        return score


# ------------------------------------------------------------------
# Fooocus integration helpers
# ------------------------------------------------------------------

def apply_vibe_filter(
    latent_dict: Dict[str, torch.Tensor],
    vae,
    clip_model,
    vibe: VibeMemory,
    threshold: float = -0.05,
    max_retry: int = 5,
    async_task=None
) -> Dict[str, torch.Tensor]:
    """
    Re-generate if VibeScore < threshold, up to max_retry times.
    """
    z = latent_dict['samples']
    retry = 0
    while retry < max_retry:
        # Decode to image
        with torch.no_grad():
            img = vae.decode(z)
            img = (img / 2 + 0.5).clamp(0, 1)

        # Get CLIP embedding
        emb = vibe.tensor_to_embedding(img)
        emb_t = torch.tensor(emb, device=z.device)

        score = vibe.score(emb_t)
        logger.info(f"[VibeFilter] attempt {retry+1} score={score:.3f}")
        if score >= threshold:
            break  # good vibe

        # Re-roll
        z = torch.randn_like(z)
        retry += 1

    latent_dict['samples'] = z
    return latent_dict


# ------------------------------------------------------------------
# Quick like/dislike buttons for UI
# ------------------------------------------------------------------

def like_current(latent_dict, vae, vibe):
    with torch.no_grad():
        img = vae.decode(latent_dict['samples'])
        img = (img / 2 + 0.5).clamp(0, 1)
    vibe.add_like(img)
    logger.info("[VibeMemory] 👍 Liked current image.")


def dislike_current(latent_dict, vae, vibe):
    with torch.no_grad():
        img = vae.decode(latent_dict['samples'])
        img = (img / 2 + 0.5).clamp(0, 1)
    vibe.add_dislike(img)
    logger.info("[VibeMemory] 👎 Disliked current image.")
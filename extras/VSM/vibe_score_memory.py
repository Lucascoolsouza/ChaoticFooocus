# vibe_score_memory.py
# Enhanced Fooocus WebUI extension – Aesthetic Memory / VibeScore
# Uses a JSON file to remember liked/disliked CLIP embeddings and
# automatically steers new generations toward or away from those vibes.

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
import numpy as np
import time
from datetime import datetime

# Try to import torch and related modules gracefully
try:
    import torch
    import torch.nn.functional as F
    import torchvision.transforms as transforms
    TORCH_AVAILABLE = True
    # Simple helper: cosine similarity
    cos = F.cosine_similarity
    TensorType = torch.Tensor
except ImportError:
    TORCH_AVAILABLE = False
    print("[VibeMemory] Warning: PyTorch not available. Some features will be disabled.")
    # Create dummy functions for when torch is not available
    class DummyF:
        @staticmethod
        def cosine_similarity(*args, **kwargs):
            return 0.0
        @staticmethod
        def normalize(*args, **kwargs):
            return args[0] if args else None
    F = DummyF()
    cos = F.cosine_similarity
    TensorType = Any  # Use Any when torch is not available

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("[VibeMemory] Warning: PIL not available. Image processing will be disabled.")

# Try to import CLIP, fallback gracefully if not available
try:
    import clip
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    print("[VibeMemory] Warning: CLIP not available. Install with: pip install git+https://github.com/openai/CLIP.git")

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# VibeMemory class
# ------------------------------------------------------------------
class VibeMemory:
    """
    Enhanced Vibe Score Memory system for aesthetic preference learning.
    
    Features:
    - CLIP-based embedding storage for liked/disliked images
    - Weighted scoring with decay over time
    - Category-based organization
    - Similarity clustering to avoid redundancy
    - Export/import functionality
    - Statistics and analytics
    """

    def __init__(self,
                 memory_path: str = "memory.json",
                 clip_model_name: str = "ViT-B/32",
                 max_memories: int = 1000,
                 similarity_threshold: float = 0.85,
                 decay_factor: float = 0.95):
        self.memory_path = Path(memory_path)
        self.device = "cuda" if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu"
        self.clip_model = None
        self.preprocess = None
        self.max_memories = max_memories
        self.similarity_threshold = similarity_threshold
        self.decay_factor = decay_factor

        # Load CLIP if available
        if CLIP_AVAILABLE and TORCH_AVAILABLE:
            try:
                self.clip_model, self.preprocess = clip.load(clip_model_name,
                                                           device=self.device)
                self.clip_model.eval()
                logger.info(f"[VibeMemory] Loaded CLIP model: {clip_model_name}")
            except Exception as e:
                logger.warning(f"[VibeMemory] Failed to load CLIP: {e}")
                self.clip_model = None
        elif not TORCH_AVAILABLE:
            logger.warning("[VibeMemory] PyTorch not available, CLIP functionality disabled")
        elif not CLIP_AVAILABLE:
            logger.warning("[VibeMemory] CLIP not available, embedding functionality disabled")

        # Enhanced data structure with metadata
        self.data = {
            "liked": [],
            "disliked": [],
            "metadata": {
                "version": "2.0",
                "created": datetime.now().isoformat(),
                "last_updated": datetime.now().isoformat(),
                "total_likes": 0,
                "total_dislikes": 0,
                "clip_model": clip_model_name
            },
            "categories": {},
            "statistics": {
                "generation_count": 0,
                "filter_applications": 0,
                "average_score": 0.0
            }
        }
        self._load()

    # ----------------------------------------------------------
    # I/O and Data Management
    # ----------------------------------------------------------
    def _load(self):
        """Load memory data from file with backward compatibility."""
        if self.memory_path.exists():
            try:
                with open(self.memory_path, "r", encoding="utf-8") as f:
                    loaded_data = json.load(f)
                
                # Handle legacy format
                if "metadata" not in loaded_data:
                    logger.info("[VibeMemory] Converting legacy format to v2.0")
                    self.data["liked"] = loaded_data.get("liked", [])
                    self.data["disliked"] = loaded_data.get("disliked", [])
                    self.data["metadata"]["total_likes"] = len(self.data["liked"])
                    self.data["metadata"]["total_dislikes"] = len(self.data["disliked"])
                else:
                    self.data = loaded_data
                
                # Ensure all required fields exist
                self._ensure_data_structure()
                
                logger.info(f"[VibeMemory] Loaded {len(self.data['liked'])} liked, "
                           f"{len(self.data['disliked'])} disliked vectors.")
            except Exception as e:
                logger.error(f"[VibeMemory] Error loading data: {e}")
                self._ensure_data_structure()

    def _ensure_data_structure(self):
        """Ensure all required data structure fields exist."""
        if "metadata" not in self.data:
            self.data["metadata"] = {
                "version": "2.0",
                "created": datetime.now().isoformat(),
                "last_updated": datetime.now().isoformat(),
                "total_likes": len(self.data.get("liked", [])),
                "total_dislikes": len(self.data.get("disliked", [])),
                "clip_model": "ViT-B/32"
            }
        
        if "categories" not in self.data:
            self.data["categories"] = {}
        
        if "statistics" not in self.data:
            self.data["statistics"] = {
                "generation_count": 0,
                "filter_applications": 0,
                "average_score": 0.0
            }

    def _save(self):
        """Save memory data with metadata updates."""
        try:
            self.memory_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Update metadata
            self.data["metadata"]["last_updated"] = datetime.now().isoformat()
            self.data["metadata"]["total_likes"] = len(self.data["liked"])
            self.data["metadata"]["total_dislikes"] = len(self.data["disliked"])
            
            # Create backup if file exists
            if self.memory_path.exists():
                backup_path = self.memory_path.with_suffix('.json.bak')
                self.memory_path.rename(backup_path)
            
            with open(self.memory_path, "w", encoding="utf-8") as f:
                json.dump(self.data, f, indent=2)
            
            logger.debug("[VibeMemory] Saved successfully.")
        except Exception as e:
            logger.error(f"[VibeMemory] Error saving data: {e}")
            # Restore backup if save failed
            backup_path = self.memory_path.with_suffix('.json.bak')
            if backup_path.exists():
                backup_path.rename(self.memory_path)

    def export_data(self, export_path: str) -> bool:
        """Export memory data to a file."""
        try:
            export_path = Path(export_path)
            export_path.parent.mkdir(parents=True, exist_ok=True)
            
            export_data = {
                **self.data,
                "export_timestamp": datetime.now().isoformat(),
                "export_version": "2.0"
            }
            
            with open(export_path, "w", encoding="utf-8") as f:
                json.dump(export_data, f, indent=2)
            
            logger.info(f"[VibeMemory] Exported data to {export_path}")
            return True
        except Exception as e:
            logger.error(f"[VibeMemory] Export failed: {e}")
            return False

    def import_data(self, import_path: str, merge: bool = True) -> bool:
        """Import memory data from a file."""
        try:
            import_path = Path(import_path)
            if not import_path.exists():
                logger.error(f"[VibeMemory] Import file not found: {import_path}")
                return False
            
            with open(import_path, "r", encoding="utf-8") as f:
                imported_data = json.load(f)
            
            if merge:
                # Merge with existing data
                self.data["liked"].extend(imported_data.get("liked", []))
                self.data["disliked"].extend(imported_data.get("disliked", []))
                
                # Merge categories
                for category, items in imported_data.get("categories", {}).items():
                    if category in self.data["categories"]:
                        self.data["categories"][category].extend(items)
                    else:
                        self.data["categories"][category] = items
            else:
                # Replace existing data
                self.data["liked"] = imported_data.get("liked", [])
                self.data["disliked"] = imported_data.get("disliked", [])
                self.data["categories"] = imported_data.get("categories", {})
            
            # Remove duplicates
            self._remove_duplicates()
            self._save()
            
            logger.info(f"[VibeMemory] Imported data from {import_path}")
            return True
        except Exception as e:
            logger.error(f"[VibeMemory] Import failed: {e}")
            return False

    # ----------------------------------------------------------
    # Enhanced Embedding Processing
    # ----------------------------------------------------------
    def image_to_embedding(self, image: Image.Image) -> List[float]:
        """Return CLIP embedding for a PIL Image with error handling."""
        if not self.clip_model:
            logger.warning("[VibeMemory] CLIP model not available")
            return [0.0] * 512  # Return dummy embedding
        
        try:
            # Ensure image is in RGB format
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Resize if too large to avoid memory issues
            if max(image.size) > 1024:
                image.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
            
            tensor = self.preprocess(image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                emb = self.clip_model.encode_image(tensor)
                # Normalize embedding
                emb = F.normalize(emb, p=2, dim=-1)
            return emb.squeeze().cpu().tolist()
        except Exception as e:
            logger.error(f"[VibeMemory] Error creating embedding: {e}")
            return [0.0] * 512

    def tensor_to_embedding(self, tensor) -> List[float]:
        """Convert tensor/array to embedding with enhanced preprocessing."""
        if not TORCH_AVAILABLE:
            logger.warning("[VibeMemory] PyTorch not available, returning dummy embedding")
            return [0.0] * 512
            
        try:
            # Handle numpy arrays
            if isinstance(tensor, np.ndarray):
                tensor = torch.from_numpy(tensor).float()
            
            # Ensure it's a torch tensor
            if not isinstance(tensor, torch.Tensor):
                logger.error(f"[VibeMemory] Unsupported tensor type: {type(tensor)}")
                return [0.0] * 512
            
            # Handle different tensor dimensions
            if tensor.dim() == 4:
                tensor = tensor.squeeze(0)
            elif tensor.dim() == 2:
                # Handle grayscale by repeating channels
                tensor = tensor.unsqueeze(0).repeat(3, 1, 1)
            elif tensor.dim() == 1:
                # Handle 1D tensor - assume it's already an embedding
                if len(tensor) >= 512:
                    return tensor[:512].cpu().tolist()
                else:
                    logger.error(f"[VibeMemory] 1D tensor too short: {len(tensor)}")
                    return [0.0] * 512
            
            # Ensure we have 3 channels
            if tensor.shape[0] == 1:
                tensor = tensor.repeat(3, 1, 1)
            elif tensor.shape[0] > 3:
                tensor = tensor[:3]
            
            # Normalize tensor to [0,1] range
            if tensor.min() < 0:
                tensor = (tensor + 1) / 2
            tensor = tensor.clamp(0, 1)
            
            # Convert to PIL
            if not PIL_AVAILABLE:
                logger.warning("[VibeMemory] PIL not available, returning dummy embedding")
                return [0.0] * 512
                
            pil = transforms.ToPILImage()(tensor.cpu())
            return self.image_to_embedding(pil)
        except Exception as e:
            logger.error(f"[VibeMemory] Error converting tensor: {e}")
            import traceback
            traceback.print_exc()
            return [0.0] * 512

    def _is_similar_embedding(self, new_embedding: List[float], existing_embeddings: List[List[float]]) -> bool:
        """Check if embedding is too similar to existing ones."""
        if not TORCH_AVAILABLE or not existing_embeddings:
            return False
        
        try:
            new_tensor = torch.tensor(new_embedding)
            existing_tensor = torch.tensor(existing_embeddings)
            
            similarities = F.cosine_similarity(new_tensor.unsqueeze(0), existing_tensor, dim=1)
            max_similarity = similarities.max().item()
            
            return max_similarity > self.similarity_threshold
        except Exception as e:
            logger.error(f"[VibeMemory] Error checking similarity: {e}")
            return False

    def _remove_duplicates(self):
        """Remove duplicate embeddings based on similarity threshold."""
        try:
            for category in ["liked", "disliked"]:
                if not self.data[category]:
                    continue
                
                unique_embeddings = []
                for embedding in self.data[category]:
                    if isinstance(embedding, dict):
                        emb_data = embedding.get("embedding", [])
                    else:
                        emb_data = embedding
                    
                    if not self._is_similar_embedding(emb_data, [
                        e.get("embedding", e) if isinstance(e, dict) else e 
                        for e in unique_embeddings
                    ]):
                        unique_embeddings.append(embedding)
                
                removed_count = len(self.data[category]) - len(unique_embeddings)
                if removed_count > 0:
                    logger.info(f"[VibeMemory] Removed {removed_count} duplicate {category} embeddings")
                    self.data[category] = unique_embeddings
        except Exception as e:
            logger.error(f"[VibeMemory] Error removing duplicates: {e}")

    # ----------------------------------------------------------
    # Enhanced Public API
    # ----------------------------------------------------------
    def add_like(self, image_or_tensor, category: str = "general", weight: float = 1.0, metadata: dict = None):
        """Add a liked image with enhanced metadata support."""
        try:
            emb = self._dispatch(image_or_tensor)
            
            # Check for duplicates
            if self._is_similar_embedding(emb, [
                e.get("embedding", e) if isinstance(e, dict) else e 
                for e in self.data["liked"]
            ]):
                logger.info("[VibeMemory] Similar embedding already exists, skipping")
                return False
            
            # Create enhanced embedding entry
            embedding_entry = {
                "embedding": emb,
                "timestamp": datetime.now().isoformat(),
                "category": category,
                "weight": weight,
                "metadata": metadata or {}
            }
            
            self.data["liked"].append(embedding_entry)
            
            # Add to category
            if category not in self.data["categories"]:
                self.data["categories"][category] = {"liked": [], "disliked": []}
            self.data["categories"][category]["liked"].append(len(self.data["liked"]) - 1)
            
            # Maintain max memories limit
            self._maintain_memory_limit()
            
            self._save()
            logger.info(f"[VibeMemory] 👍 Added like (category: {category}, weight: {weight})")
            return True
        except Exception as e:
            logger.error(f"[VibeMemory] Error adding like: {e}")
            return False

    def add_dislike(self, image_or_tensor, category: str = "general", weight: float = 1.0, metadata: dict = None):
        """Add a disliked image with enhanced metadata support."""
        try:
            emb = self._dispatch(image_or_tensor)
            
            # Check for duplicates
            if self._is_similar_embedding(emb, [
                e.get("embedding", e) if isinstance(e, dict) else e 
                for e in self.data["disliked"]
            ]):
                logger.info("[VibeMemory] Similar embedding already exists, skipping")
                return False
            
            # Create enhanced embedding entry
            embedding_entry = {
                "embedding": emb,
                "timestamp": datetime.now().isoformat(),
                "category": category,
                "weight": weight,
                "metadata": metadata or {}
            }
            
            self.data["disliked"].append(embedding_entry)
            
            # Add to category
            if category not in self.data["categories"]:
                self.data["categories"][category] = {"liked": [], "disliked": []}
            self.data["categories"][category]["disliked"].append(len(self.data["disliked"]) - 1)
            
            # Maintain max memories limit
            self._maintain_memory_limit()
            
            self._save()
            logger.info(f"[VibeMemory] 👎 Added dislike (category: {category}, weight: {weight})")
            return True
        except Exception as e:
            logger.error(f"[VibeMemory] Error adding dislike: {e}")
            return False

    def _dispatch(self, obj):
        """Enhanced dispatch with better error handling."""
        if isinstance(obj, Image.Image):
            return self.image_to_embedding(obj)
        elif isinstance(obj, torch.Tensor):
            return self.tensor_to_embedding(obj)
        elif isinstance(obj, str):
            # Handle file path
            try:
                image = Image.open(obj)
                return self.image_to_embedding(image)
            except Exception as e:
                logger.error(f"[VibeMemory] Error loading image from path: {e}")
                raise TypeError(f"Could not load image from path: {obj}")
        else:
            raise TypeError("Need PIL.Image, torch.Tensor, or file path string.")

    def _maintain_memory_limit(self):
        """Maintain memory limit by removing oldest entries."""
        try:
            for category in ["liked", "disliked"]:
                if len(self.data[category]) > self.max_memories // 2:
                    # Remove oldest entries (those without timestamp first)
                    entries_with_time = []
                    entries_without_time = []
                    
                    for entry in self.data[category]:
                        if isinstance(entry, dict) and "timestamp" in entry:
                            entries_with_time.append(entry)
                        else:
                            entries_without_time.append(entry)
                    
                    # Sort by timestamp and keep most recent
                    entries_with_time.sort(key=lambda x: x["timestamp"], reverse=True)
                    keep_count = (self.max_memories // 2) - len(entries_without_time)
                    
                    if keep_count > 0:
                        self.data[category] = entries_without_time + entries_with_time[:keep_count]
                    else:
                        self.data[category] = entries_with_time[:self.max_memories // 2]
                    
                    logger.info(f"[VibeMemory] Trimmed {category} memories to {len(self.data[category])}")
        except Exception as e:
            logger.error(f"[VibeMemory] Error maintaining memory limit: {e}")

    # ----------------------------------------------------------
    # Enhanced Scoring System
    # ----------------------------------------------------------
    def score(self, embedding, category: str = None, use_weights: bool = True) -> float:
        """
        Enhanced scoring with category filtering and weighted scoring.
        
        Returns:
          +large  -> very aligned with likes
          0       -> neutral
          -large  -> very aligned with dislikes
        """
        if not TORCH_AVAILABLE:
            logger.warning("[VibeMemory] PyTorch not available, returning neutral score")
            return 0.0
            
        if not self.clip_model or not (self.data["liked"] or self.data["disliked"]):
            return 0.0

        try:
            # Handle different input types
            if isinstance(embedding, np.ndarray):
                emb = torch.from_numpy(embedding).float().to(self.device)
            elif isinstance(embedding, list):
                emb = torch.tensor(embedding, dtype=torch.float32).to(self.device)
            elif isinstance(embedding, torch.Tensor):
                emb = embedding.to(self.device)
            else:
                logger.error(f"[VibeMemory] Unsupported embedding type: {type(embedding)}")
                return 0.0
            
            if emb.dim() == 1:
                emb = emb.unsqueeze(0)
            
            # Normalize input embedding
            emb = F.normalize(emb, p=2, dim=-1)

            score = 0.0
            
            # Process liked embeddings
            if self.data["liked"]:
                liked_embeddings, liked_weights = self._extract_embeddings_and_weights(
                    self.data["liked"], category, use_weights
                )
                if liked_embeddings:
                    liked_tensor = torch.tensor(liked_embeddings, device=self.device)
                    liked_tensor = F.normalize(liked_tensor, p=2, dim=-1)
                    similarities = cos(emb, liked_tensor, dim=1)
                    
                    if use_weights and liked_weights:
                        weight_tensor = torch.tensor(liked_weights, device=self.device)
                        weighted_similarities = similarities * weight_tensor
                        score += weighted_similarities.mean().item()
                    else:
                        score += similarities.mean().item()

            # Process disliked embeddings
            if self.data["disliked"]:
                disliked_embeddings, disliked_weights = self._extract_embeddings_and_weights(
                    self.data["disliked"], category, use_weights
                )
                if disliked_embeddings:
                    disliked_tensor = torch.tensor(disliked_embeddings, device=self.device)
                    disliked_tensor = F.normalize(disliked_tensor, p=2, dim=-1)
                    similarities = cos(emb, disliked_tensor, dim=1)
                    
                    if use_weights and disliked_weights:
                        weight_tensor = torch.tensor(disliked_weights, device=self.device)
                        weighted_similarities = similarities * weight_tensor
                        score -= weighted_similarities.mean().item()
                    else:
                        score -= similarities.mean().item()

            # Update statistics
            self.data["statistics"]["generation_count"] += 1
            self.data["statistics"]["average_score"] = (
                (self.data["statistics"]["average_score"] * (self.data["statistics"]["generation_count"] - 1) + score) /
                self.data["statistics"]["generation_count"]
            )

            return score
        except Exception as e:
            logger.error(f"[VibeMemory] Error calculating score: {e}")
            return 0.0

    def _extract_embeddings_and_weights(self, entries: List, category: str = None, use_weights: bool = True):
        """Extract embeddings and weights from entries, optionally filtered by category."""
        embeddings = []
        weights = []
        
        for entry in entries:
            # Handle both old format (list) and new format (dict)
            if isinstance(entry, dict):
                if category and entry.get("category") != category:
                    continue
                embeddings.append(entry["embedding"])
                if use_weights:
                    weights.append(entry.get("weight", 1.0))
            else:
                # Legacy format
                embeddings.append(entry)
                if use_weights:
                    weights.append(1.0)
        
        return embeddings, weights if use_weights else None

    def get_detailed_score(self, embedding: Any) -> Dict:
        """Get detailed scoring breakdown."""
        try:
            base_score = self.score(embedding, use_weights=False)
            weighted_score = self.score(embedding, use_weights=True)
            
            # Category-specific scores
            category_scores = {}
            for category in self.data["categories"]:
                category_scores[category] = self.score(embedding, category=category)
            
            return {
                "base_score": base_score,
                "weighted_score": weighted_score,
                "category_scores": category_scores,
                "total_likes": len(self.data["liked"]),
                "total_dislikes": len(self.data["disliked"]),
                "categories": list(self.data["categories"].keys())
            }
        except Exception as e:
            logger.error(f"[VibeMemory] Error getting detailed score: {e}")
            return {"error": str(e)}

    def get_statistics(self) -> Dict:
        """Get comprehensive statistics about the memory system."""
        try:
            stats = {
                "memory_stats": {
                    "total_likes": len(self.data["liked"]),
                    "total_dislikes": len(self.data["disliked"]),
                    "total_memories": len(self.data["liked"]) + len(self.data["disliked"]),
                    "max_memories": self.max_memories,
                    "similarity_threshold": self.similarity_threshold
                },
                "category_stats": {},
                "usage_stats": self.data["statistics"],
                "metadata": self.data["metadata"]
            }
            
            # Category statistics
            for category, indices in self.data["categories"].items():
                stats["category_stats"][category] = {
                    "likes": len(indices.get("liked", [])),
                    "dislikes": len(indices.get("disliked", []))
                }
            
            return stats
        except Exception as e:
            logger.error(f"[VibeMemory] Error getting statistics: {e}")
            return {"error": str(e)}

    def clear_category(self, category: str) -> bool:
        """Clear all memories from a specific category."""
        try:
            if category not in self.data["categories"]:
                return False
            
            # Remove entries from main lists
            liked_indices = self.data["categories"][category].get("liked", [])
            disliked_indices = self.data["categories"][category].get("disliked", [])
            
            # Remove in reverse order to maintain indices
            for idx in sorted(liked_indices, reverse=True):
                if idx < len(self.data["liked"]):
                    del self.data["liked"][idx]
            
            for idx in sorted(disliked_indices, reverse=True):
                if idx < len(self.data["disliked"]):
                    del self.data["disliked"][idx]
            
            # Remove category
            del self.data["categories"][category]
            
            self._save()
            logger.info(f"[VibeMemory] Cleared category: {category}")
            return True
        except Exception as e:
            logger.error(f"[VibeMemory] Error clearing category: {e}")
            return False

    def clear_all(self) -> bool:
        """Clear all memory data."""
        try:
            self.data["liked"] = []
            self.data["disliked"] = []
            self.data["categories"] = {}
            self.data["statistics"] = {
                "generation_count": 0,
                "filter_applications": 0,
                "average_score": 0.0
            }
            self._save()
            logger.info("[VibeMemory] Cleared all memory data")
            return True
        except Exception as e:
            logger.error(f"[VibeMemory] Error clearing all data: {e}")
            return False

    def batch_score(self, embeddings: List, category: str = None, use_weights: bool = True) -> List[float]:
        """Score multiple embeddings efficiently."""
        if not self.clip_model or not (self.data["liked"] or self.data["disliked"]):
            return [0.0] * len(embeddings)
        
        try:
            scores = []
            for embedding in embeddings:
                score = self.score(embedding, category, use_weights)
                scores.append(score)
            return scores
        except Exception as e:
            logger.error(f"[VibeMemory] Error in batch scoring: {e}")
            return [0.0] * len(embeddings)

    def find_similar_memories(self, embedding, threshold: float = 0.8, limit: int = 10) -> Dict:
        """Find similar memories to the given embedding."""
        if not TORCH_AVAILABLE or not self.clip_model:
            return {"liked": [], "disliked": []}
        
        try:
            # Convert embedding to tensor
            if isinstance(embedding, list):
                emb_tensor = torch.tensor(embedding, dtype=torch.float32).to(self.device)
            elif isinstance(embedding, np.ndarray):
                emb_tensor = torch.from_numpy(embedding).float().to(self.device)
            else:
                emb_tensor = embedding.to(self.device)
            
            if emb_tensor.dim() == 1:
                emb_tensor = emb_tensor.unsqueeze(0)
            emb_tensor = F.normalize(emb_tensor, p=2, dim=-1)
            
            similar_memories = {"liked": [], "disliked": []}
            
            # Check liked memories
            for i, entry in enumerate(self.data["liked"]):
                entry_emb = entry.get("embedding", entry) if isinstance(entry, dict) else entry
                entry_tensor = torch.tensor(entry_emb, dtype=torch.float32).to(self.device)
                entry_tensor = F.normalize(entry_tensor.unsqueeze(0), p=2, dim=-1)
                
                similarity = F.cosine_similarity(emb_tensor, entry_tensor, dim=1).item()
                if similarity >= threshold:
                    similar_memories["liked"].append({
                        "index": i,
                        "similarity": similarity,
                        "metadata": entry.get("metadata", {}) if isinstance(entry, dict) else {}
                    })
            
            # Check disliked memories
            for i, entry in enumerate(self.data["disliked"]):
                entry_emb = entry.get("embedding", entry) if isinstance(entry, dict) else entry
                entry_tensor = torch.tensor(entry_emb, dtype=torch.float32).to(self.device)
                entry_tensor = F.normalize(entry_tensor.unsqueeze(0), p=2, dim=-1)
                
                similarity = F.cosine_similarity(emb_tensor, entry_tensor, dim=1).item()
                if similarity >= threshold:
                    similar_memories["disliked"].append({
                        "index": i,
                        "similarity": similarity,
                        "metadata": entry.get("metadata", {}) if isinstance(entry, dict) else {}
                    })
            
            # Sort by similarity and limit results
            similar_memories["liked"] = sorted(similar_memories["liked"], 
                                             key=lambda x: x["similarity"], reverse=True)[:limit]
            similar_memories["disliked"] = sorted(similar_memories["disliked"], 
                                                key=lambda x: x["similarity"], reverse=True)[:limit]
            
            return similar_memories
        except Exception as e:
            logger.error(f"[VibeMemory] Error finding similar memories: {e}")
            return {"liked": [], "disliked": []}

    def optimize_memory(self) -> Dict:
        """Optimize memory by removing redundant entries and updating weights."""
        try:
            original_liked = len(self.data["liked"])
            original_disliked = len(self.data["disliked"])
            
            # Remove duplicates
            self._remove_duplicates()
            
            # Update weights based on age (newer memories get slightly higher weight)
            current_time = datetime.now()
            for category in ["liked", "disliked"]:
                for entry in self.data[category]:
                    if isinstance(entry, dict) and "timestamp" in entry:
                        try:
                            entry_time = datetime.fromisoformat(entry["timestamp"])
                            age_days = (current_time - entry_time).days
                            # Decay weight slightly over time (but not below 0.5)
                            age_factor = max(0.5, self.decay_factor ** (age_days / 30))
                            entry["weight"] = entry.get("weight", 1.0) * age_factor
                        except Exception:
                            pass  # Skip entries with invalid timestamps
            
            self._save()
            
            result = {
                "original_liked": original_liked,
                "original_disliked": original_disliked,
                "final_liked": len(self.data["liked"]),
                "final_disliked": len(self.data["disliked"]),
                "removed_liked": original_liked - len(self.data["liked"]),
                "removed_disliked": original_disliked - len(self.data["disliked"])
            }
            
            logger.info(f"[VibeMemory] Optimization complete: {result}")
            return result
        except Exception as e:
            logger.error(f"[VibeMemory] Error optimizing memory: {e}")
            return {"error": str(e)}

    def get_memory_health(self) -> Dict:
        """Get health metrics for the memory system."""
        try:
            total_memories = len(self.data["liked"]) + len(self.data["disliked"])
            
            # Calculate diversity (average pairwise distance)
            diversity_score = 0.0
            if TORCH_AVAILABLE and total_memories > 1:
                all_embeddings = []
                for entry in self.data["liked"] + self.data["disliked"]:
                    emb = entry.get("embedding", entry) if isinstance(entry, dict) else entry
                    all_embeddings.append(emb)
                
                if len(all_embeddings) > 1:
                    embeddings_tensor = torch.tensor(all_embeddings, dtype=torch.float32)
                    embeddings_tensor = F.normalize(embeddings_tensor, p=2, dim=-1)
                    
                    # Calculate pairwise similarities
                    similarities = torch.mm(embeddings_tensor, embeddings_tensor.t())
                    # Remove diagonal (self-similarities)
                    mask = torch.eye(similarities.size(0), dtype=torch.bool)
                    similarities = similarities[~mask]
                    
                    # Diversity is 1 - average similarity
                    diversity_score = 1.0 - similarities.mean().item()
            elif not TORCH_AVAILABLE:
                # Fallback diversity calculation without torch
                diversity_score = 0.5  # Assume moderate diversity
            
            # Calculate balance
            liked_count = len(self.data["liked"])
            disliked_count = len(self.data["disliked"])
            balance_score = 1.0 - abs(liked_count - disliked_count) / max(1, liked_count + disliked_count)
            
            # Calculate memory usage
            memory_usage = total_memories / self.max_memories
            
            return {
                "total_memories": total_memories,
                "liked_count": liked_count,
                "disliked_count": disliked_count,
                "diversity_score": diversity_score,
                "balance_score": balance_score,
                "memory_usage": memory_usage,
                "health_score": (diversity_score + balance_score + (1.0 - min(1.0, memory_usage))) / 3.0,
                "recommendations": self._get_health_recommendations(diversity_score, balance_score, memory_usage)
            }
        except Exception as e:
            logger.error(f"[VibeMemory] Error calculating health metrics: {e}")
            return {"error": str(e)}

    def _get_health_recommendations(self, diversity: float, balance: float, usage: float) -> List[str]:
        """Get recommendations based on health metrics."""
        recommendations = []
        
        if diversity < 0.3:
            recommendations.append("Consider adding more diverse images to improve variety")
        
        if balance < 0.5:
            recommendations.append("Try to balance likes and dislikes for better scoring")
        
        if usage > 0.9:
            recommendations.append("Memory is nearly full, consider optimizing or increasing limit")
        elif usage < 0.1:
            recommendations.append("Add more memories to improve aesthetic learning")
        
        if not recommendations:
            recommendations.append("Memory system is healthy!")
        
        return recommendations


# ------------------------------------------------------------------
# Fooocus integration helpers
# ------------------------------------------------------------------

def apply_vibe_filter(
    latent_dict: Dict[str, Any],
    vae,
    clip_model,
    vibe: VibeMemory,
    threshold: float = -0.05,
    max_retry: int = 5,
    async_task=None
) -> Dict[str, Any]:
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
        if TORCH_AVAILABLE:
            emb_t = torch.tensor(emb, device=z.device)
        else:
            emb_t = emb

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
# Vibe Memory Integration for Fooocus
# Integrates the VSM (Vibe Score Memory) system into the main generation pipeline

import os
import logging
from pathlib import Path

# Try to import torch gracefully
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("[VibeMemory] Warning: PyTorch not available")

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("[VibeMemory] Warning: PIL not available")

try:
    import modules.config as config
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False
    print("[VibeMemory] Warning: modules.config not available")

logger = logging.getLogger(__name__)

# Global vibe memory instance
vibe_memory_instance = None

def get_vibe_memory():
    """Get or create the global vibe memory instance."""
    global vibe_memory_instance
    if vibe_memory_instance is None:
        if not TORCH_AVAILABLE:
            logger.warning("[VibeMemory] PyTorch not available, cannot initialize")
            return None
        
        try:
            from extras.VSM.vibe_score_memory import VibeMemory
            if CONFIG_AVAILABLE:
                memory_path = os.path.join(config.path_outputs, "vibe_memory.json")
            else:
                memory_path = "vibe_memory.json"
            vibe_memory_instance = VibeMemory(memory_path=memory_path)
            logger.info("[VibeMemory] Initialized vibe memory system")
        except Exception as e:
            logger.warning(f"[VibeMemory] Failed to initialize: {e}")
            vibe_memory_instance = None
    return vibe_memory_instance

def is_vibe_memory_enabled(async_task):
    """Check if vibe memory is enabled for this task."""
    return hasattr(async_task, 'vibe_memory_enabled') and async_task.vibe_memory_enabled

def apply_vibe_filtering(latent_samples, vae_model, async_task):
    """Apply vibe filtering to generated samples with enhanced error handling."""
    if not TORCH_AVAILABLE:
        logger.info("[VibeMemory] PyTorch not available, skipping vibe filtering")
        return latent_samples
        
    vibe = get_vibe_memory()
    if not vibe:
        logger.info("[VibeMemory] Vibe memory not available, skipping filtering")
        return latent_samples
        
    if not is_vibe_memory_enabled(async_task):
        logger.debug("[VibeMemory] Vibe memory not enabled for this task")
        return latent_samples
    
    if not vibe.clip_model:
        logger.warning("[VibeMemory] CLIP model not available, cannot perform filtering")
        logger.info("[VibeMemory] Install CLIP with: pip install git+https://github.com/openai/CLIP.git")
        return latent_samples
    
    # Check if we have any memories to work with
    total_memories = len(vibe.data.get("liked", [])) + len(vibe.data.get("disliked", []))
    if total_memories == 0:
        logger.info("[VibeMemory] No memories stored, skipping filtering")
        logger.info("[VibeMemory] Use 👍/👎 buttons to add image preferences")
        return latent_samples
    
    # Get parameters from async_task
    threshold = getattr(async_task, 'vibe_memory_threshold', -0.1)
    max_retries = getattr(async_task, 'vibe_memory_max_retries', 3)
    category = getattr(async_task, 'vibe_memory_category', None)
    
    logger.info(f"[VibeMemory] Applying vibe filtering (threshold: {threshold:.3f}, max_retries: {max_retries})")
    
    try:
        original_latent = latent_samples.clone()
        best_score = float('-inf')
        best_latent = latent_samples.clone()
        retry_count = 0
        
        while retry_count < max_retries:
            # Decode latent to image for scoring
            with torch.no_grad():
                decoded = vae_model.decode(latent_samples)
                # Normalize to [0,1] range
                if decoded.min() < 0:
                    decoded = (decoded + 1.0) / 2.0
                decoded = decoded.clamp(0, 1)
                
                # Convert to PIL for CLIP embedding
                if decoded.dim() == 4:
                    decoded = decoded.squeeze(0)
                
                # Get embedding and score
                embedding = vibe.tensor_to_embedding(decoded)
                if not embedding or all(x == 0 for x in embedding):
                    logger.warning("[VibeMemory] Got zero embedding, skipping this attempt")
                    retry_count += 1
                    continue
                
                score = vibe.score(embedding, category=category)
                
                logger.info(f"[VibeMemory] Attempt {retry_count + 1}/{max_retries}, Score: {score:.6f}")
                
                # Keep track of best result
                if score > best_score:
                    best_score = score
                    best_latent = latent_samples.clone()
                
                if score >= threshold:
                    logger.info(f"[VibeMemory] ✅ Accepted image with score {score:.6f}")
                    break
                
                # Generate new latent if score is too low
                if retry_count < max_retries - 1:
                    # Add controlled noise to original latent instead of completely random
                    noise_strength = 0.1 + (retry_count * 0.05)
                    latent_samples = original_latent + torch.randn_like(original_latent) * noise_strength
                    retry_count += 1
                else:
                    # Use best result if no sample met threshold
                    latent_samples = best_latent
                    logger.info(f"[VibeMemory] ⚠️ Max retries reached, using best score: {best_score:.6f}")
                    break
        
        # Update statistics
        if hasattr(vibe, 'data') and 'statistics' in vibe.data:
            vibe.data["statistics"]["filter_applications"] += 1
            vibe._save()
                    
    except Exception as e:
        logger.error(f"[VibeMemory] Error during filtering: {e}")
        import traceback
        traceback.print_exc()
        # Return original latent on error
        return original_latent if 'original_latent' in locals() else latent_samples
    
    return latent_samples

def add_like_from_image_path(image_path):
    """Add a like from an image file path."""
    if not PIL_AVAILABLE:
        return False
        
    vibe = get_vibe_memory()
    if not vibe:
        return False
    
    try:
        image = Image.open(image_path)
        vibe.add_like(image)
        logger.info(f"[VibeMemory] 👍 Added like from {image_path}")
        return True
    except Exception as e:
        logger.error(f"[VibeMemory] Failed to add like: {e}")
        return False

def add_dislike_from_image_path(image_path):
    """Add a dislike from an image file path."""
    if not PIL_AVAILABLE:
        return False
        
    vibe = get_vibe_memory()
    if not vibe:
        return False
    
    try:
        image = Image.open(image_path)
        vibe.add_dislike(image)
        logger.info(f"[VibeMemory] 👎 Added dislike from {image_path}")
        return True
    except Exception as e:
        logger.error(f"[VibeMemory] Failed to add dislike: {e}")
        return False

def get_memory_stats():
    """Get statistics about the current memory."""
    vibe = get_vibe_memory()
    if not vibe:
        return {"liked": 0, "disliked": 0, "available": False}
    
    return {
        "liked": len(vibe.data["liked"]),
        "disliked": len(vibe.data["disliked"]),
        "available": True
    }

def clear_memory():
    """Clear all vibe memory data."""
    vibe = get_vibe_memory()
    if not vibe:
        return False
    
    try:
        return vibe.clear_all()
    except Exception as e:
        logger.error(f"[VibeMemory] Failed to clear memory: {e}")
        return False

def optimize_memory():
    """Optimize the vibe memory by removing duplicates and updating weights."""
    vibe = get_vibe_memory()
    if not vibe:
        return {"error": "Vibe memory not available"}
    
    try:
        return vibe.optimize_memory()
    except Exception as e:
        logger.error(f"[VibeMemory] Failed to optimize memory: {e}")
        return {"error": str(e)}

def get_memory_health():
    """Get health metrics for the memory system."""
    vibe = get_vibe_memory()
    if not vibe:
        return {"error": "Vibe memory not available"}
    
    try:
        return vibe.get_memory_health()
    except Exception as e:
        logger.error(f"[VibeMemory] Failed to get health metrics: {e}")
        return {"error": str(e)}

def export_memory(export_path):
    """Export memory data to a file."""
    vibe = get_vibe_memory()
    if not vibe:
        return False
    
    try:
        return vibe.export_data(export_path)
    except Exception as e:
        logger.error(f"[VibeMemory] Failed to export memory: {e}")
        return False

def import_memory(import_path, merge=True):
    """Import memory data from a file."""
    vibe = get_vibe_memory()
    if not vibe:
        return False
    
    try:
        return vibe.import_data(import_path, merge)
    except Exception as e:
        logger.error(f"[VibeMemory] Failed to import memory: {e}")
        return False

def score_image_path(image_path):
    """Score an image from file path."""
    if not PIL_AVAILABLE:
        return 0.0
        
    vibe = get_vibe_memory()
    if not vibe:
        return 0.0
    
    try:
        image = Image.open(image_path)
        embedding = vibe.image_to_embedding(image)
        return vibe.score(embedding)
    except Exception as e:
        logger.error(f"[VibeMemory] Failed to score image: {e}")
        return 0.0

def find_similar_images(image_path, threshold=0.8, limit=10):
    """Find similar images in memory."""
    if not PIL_AVAILABLE:
        return {"liked": [], "disliked": []}
        
    vibe = get_vibe_memory()
    if not vibe:
        return {"liked": [], "disliked": []}
    
    try:
        image = Image.open(image_path)
        embedding = vibe.image_to_embedding(image)
        return vibe.find_similar_memories(embedding, threshold, limit)
    except Exception as e:
        logger.error(f"[VibeMemory] Failed to find similar images: {e}")
        return {"liked": [], "disliked": []}
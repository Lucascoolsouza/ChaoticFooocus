# CLIP ONNX Runtime for Disco Diffusion
# Lightweight CLIP implementation using ONNX models

import numpy as np
import torch
import torch.nn.functional as F
from typing import Optional, Union, List
import logging

logger = logging.getLogger(__name__)

class CLIPONNXModel:
    """CLIP model using ONNX runtime for lightweight inference"""
    
    def __init__(self, visual_model_path: str, textual_model_path: str):
        self.visual_model_path = visual_model_path
        self.textual_model_path = textual_model_path
        self.visual_session = None
        self.textual_session = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self._load_models()
    
    def _load_models(self):
        """Load ONNX models"""
        try:
            import onnxruntime as ort
            
            # Configure providers based on device
            if self.device == "cuda" and ort.get_available_providers().__contains__('CUDAExecutionProvider'):
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
                print(f"[Disco CLIP] Using CUDA acceleration")
            else:
                providers = ['CPUExecutionProvider']
                print(f"[Disco CLIP] Using CPU")
            
            # Load visual model
            self.visual_session = ort.InferenceSession(
                self.visual_model_path, 
                providers=providers
            )
            
            # Load textual model
            self.textual_session = ort.InferenceSession(
                self.textual_model_path,
                providers=providers
            )
            
            print(f"[Disco CLIP] ONNX models loaded successfully")
            
        except ImportError:
            print("[Disco CLIP] ONNX Runtime not available. Install with: pip install onnxruntime")
            print("[Disco CLIP] Or for GPU: pip install onnxruntime-gpu")
            raise
        except Exception as e:
            print(f"[Disco CLIP] Failed to load ONNX models: {e}")
            raise
    
    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images to CLIP embeddings"""
        try:
            # Convert torch tensor to numpy
            if isinstance(images, torch.Tensor):
                images_np = images.detach().cpu().numpy()
            else:
                images_np = images
            
            # Ensure correct shape and dtype
            if images_np.dtype != np.float32:
                images_np = images_np.astype(np.float32)
            
            # Run inference
            input_name = self.visual_session.get_inputs()[0].name
            outputs = self.visual_session.run(None, {input_name: images_np})
            
            # Convert back to torch tensor
            embeddings = torch.from_numpy(outputs[0])
            
            # Move to original device if needed
            if hasattr(images, 'device'):
                embeddings = embeddings.to(images.device)
            
            return embeddings
            
        except Exception as e:
            logger.error(f"CLIP image encoding failed: {e}")
            # Return dummy embeddings as fallback
            batch_size = images.shape[0] if hasattr(images, 'shape') else 1
            return torch.randn(batch_size, 512)  # ResNet-50 embedding size
    
    def encode_text(self, text_tokens: torch.Tensor) -> torch.Tensor:
        """Encode text tokens to CLIP embeddings"""
        try:
            # Convert torch tensor to numpy
            if isinstance(text_tokens, torch.Tensor):
                tokens_np = text_tokens.detach().cpu().numpy()
            else:
                tokens_np = text_tokens
            
            # Ensure correct dtype
            if tokens_np.dtype != np.int64:
                tokens_np = tokens_np.astype(np.int64)
            
            # Run inference
            input_name = self.textual_session.get_inputs()[0].name
            outputs = self.textual_session.run(None, {input_name: tokens_np})
            
            # Convert back to torch tensor
            embeddings = torch.from_numpy(outputs[0])
            
            # Move to original device if needed
            if hasattr(text_tokens, 'device'):
                embeddings = embeddings.to(text_tokens.device)
            
            return embeddings
            
        except Exception as e:
            logger.error(f"CLIP text encoding failed: {e}")
            # Return dummy embeddings as fallback
            batch_size = text_tokens.shape[0] if hasattr(text_tokens, 'shape') else 1
            return torch.randn(batch_size, 512)  # ResNet-50 embedding size
    
    def eval(self):
        """Set model to eval mode (compatibility with PyTorch CLIP)"""
        # ONNX models are always in eval mode
        return self

class CLIPTokenizer:
    """Simple CLIP tokenizer for ONNX models"""
    
    def __init__(self):
        # Basic tokenizer - in practice you'd want a proper one
        self.vocab_size = 49408
        self.context_length = 77
    
    def encode(self, texts: Union[str, List[str]]) -> torch.Tensor:
        """Encode text to tokens"""
        if isinstance(texts, str):
            texts = [texts]
        
        # Simple tokenization (in practice, use proper CLIP tokenizer)
        tokens = []
        for text in texts:
            # Convert text to token IDs (simplified)
            text_tokens = [hash(word) % self.vocab_size for word in text.lower().split()]
            
            # Pad or truncate to context length
            if len(text_tokens) > self.context_length - 2:
                text_tokens = text_tokens[:self.context_length - 2]
            
            # Add start and end tokens
            text_tokens = [49406] + text_tokens + [49407]  # SOT and EOT tokens
            
            # Pad to context length
            while len(text_tokens) < self.context_length:
                text_tokens.append(0)
            
            tokens.append(text_tokens)
        
        return torch.tensor(tokens, dtype=torch.long)

def load_clip_onnx(visual_path: str, textual_path: str) -> tuple:
    """Load CLIP ONNX models and return model + tokenizer"""
    try:
        model = CLIPONNXModel(visual_path, textual_path)
        tokenizer = CLIPTokenizer()
        
        return model, tokenizer
        
    except Exception as e:
        print(f"[Disco CLIP] Failed to load ONNX CLIP: {e}")
        return None, None

def preprocess_image(image: torch.Tensor) -> torch.Tensor:
    """Preprocess image for CLIP (resize, normalize)"""
    try:
        # Resize to 224x224 if needed
        if image.shape[-2:] != (224, 224):
            image = F.interpolate(image, size=(224, 224), mode='bilinear', align_corners=False)
        
        # Normalize (ImageNet stats)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        
        if image.device != mean.device:
            mean = mean.to(image.device)
            std = std.to(image.device)
        
        image = (image - mean) / std
        
        return image
        
    except Exception as e:
        logger.error(f"Image preprocessing failed: {e}")
        return image
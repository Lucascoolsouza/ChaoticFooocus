# Simple CLIP initialization for Disco Diffusion

def init_clip_simple(self):
    """Initialize CLIP model for guidance - Simple OpenAI CLIP approach"""
    try:
        import clip
        import torch
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Try different CLIP models in order of preference
        clip_models = ["RN50", "ViT-B/32", "ViT-L/14", "RN50x4"]
        
        for model_name in clip_models:
            try:
                print(f"[Disco] Loading CLIP model: {model_name}")
                self.clip_model, self.clip_preprocess = clip.load(model_name, device=device)
                self.clip_model.eval()
                print(f"[Disco] CLIP {model_name} loaded successfully on {device}")
                self.clip_model_name = model_name
                return
            except Exception as e:
                print(f"[Disco] Failed to load {model_name}: {e}")
                continue
        
        # If all models failed
        print("[Disco] Failed to load any CLIP model")
        self.clip_model = None
        
    except ImportError:
        print("[Disco] CLIP not available.")
        print("[Disco] To enable full Disco Diffusion functionality:")
        print("[Disco]   1. Install CLIP: pip install git+https://github.com/openai/CLIP.git")
        print("[Disco]   2. Install torchvision: pip install torchvision")
        print("[Disco] Using geometric transforms only (still creates psychedelic effects)")
        self.clip_model = None
    except Exception as e:
        print(f"[Disco] Failed to initialize CLIP: {e}")
        print("[Disco] Using geometric transforms only")
        self.clip_model = None
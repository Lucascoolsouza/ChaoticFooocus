# NAG Compatibility Patch
# Handles version conflicts between transformers/peft/diffusers

import sys
import warnings
from typing import Optional

def patch_transformers_cache():
    """
    Patch for transformers version compatibility.
    Creates a dummy EncoderDecoderCache if it doesn't exist.
    """
    try:
        import transformers
        
        # Check if EncoderDecoderCache exists
        if not hasattr(transformers, 'EncoderDecoderCache'):
            # Create a dummy class for compatibility
            class EncoderDecoderCache:
                def __init__(self, *args, **kwargs):
                    pass
                    
            # Add it to transformers module
            transformers.EncoderDecoderCache = EncoderDecoderCache
            
            # Also add to __all__ if it exists
            if hasattr(transformers, '__all__'):
                if 'EncoderDecoderCache' not in transformers.__all__:
                    transformers.__all__.append('EncoderDecoderCache')
                    
        return True
        
    except ImportError:
        return False

def safe_import_nag():
    """
    Safely import NAG components with compatibility patches.
    """
    try:
        # Apply compatibility patch
        patch_transformers_cache()
        
        # Try importing NAG components
        from .pipeline_sdxl_nag import NAGStableDiffusionXLPipeline, NAGSampler, nag_sampler
        
        return {
            'NAGStableDiffusionXLPipeline': NAGStableDiffusionXLPipeline,
            'NAGSampler': NAGSampler,
            'nag_sampler': nag_sampler,
            'available': True
        }
        
    except Exception as e:
        warnings.warn(f"NAG components unavailable due to dependency conflict: {e}")
        
        # Return dummy implementations
        class DummyNAGPipeline:
            def __init__(self, *args, **kwargs):
                raise RuntimeError("NAG unavailable due to dependency conflicts")
                
        class DummyNAGSampler:
            def __init__(self, *args, **kwargs):
                raise RuntimeError("NAG unavailable due to dependency conflicts")
                
        def dummy_nag_sampler(*args, **kwargs):
            raise RuntimeError("NAG unavailable due to dependency conflicts")
        
        return {
            'NAGStableDiffusionXLPipeline': DummyNAGPipeline,
            'NAGSampler': DummyNAGSampler,
            'nag_sampler': dummy_nag_sampler,
            'available': False
        }
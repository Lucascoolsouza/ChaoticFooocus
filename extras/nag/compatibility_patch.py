# NAG Compatibility Patch
# Handles version conflicts between transformers/peft/diffusers

import sys
import warnings
from typing import Optional

def patch_transformers_cache():
    """
    Patch for transformers version compatibility.
    Creates dummy cache classes if they don't exist.
    """
    try:
        import transformers
        
        # List of cache classes that might be missing
        missing_classes = ['EncoderDecoderCache', 'HybridCache']
        
        for class_name in missing_classes:
            if not hasattr(transformers, class_name):
                # Create a dummy class for compatibility
                class DummyCache:
                    def __init__(self, *args, **kwargs):
                        pass
                        
                # Add it to transformers module
                setattr(transformers, class_name, DummyCache)
                
                # Also add to __all__ if it exists
                if hasattr(transformers, '__all__'):
                    if class_name not in transformers.__all__:
                        transformers.__all__.append(class_name)
                        
        return True
        
    except ImportError:
        return False

def safe_import_nag():
    """
    Safely import NAG components with compatibility patches.
    Falls back to standalone implementation if dependencies are incompatible.
    """
    try:
        # First try to apply compatibility patch and import original NAG
        patch_transformers_cache()
        
        # Try importing NAG components
        from .pipeline_sdxl_nag import NAGStableDiffusionXLPipeline, NAGSampler, nag_sampler
        
        return {
            'NAGStableDiffusionXLPipeline': NAGStableDiffusionXLPipeline,
            'NAGSampler': NAGSampler,
            'nag_sampler': nag_sampler,
            'available': True,
            'standalone': False
        }
        
    except Exception as e:
        warnings.warn(f"Original NAG unavailable, using standalone implementation: {e}")
        
        # Fall back to standalone implementation
        try:
            from .standalone_nag import StandaloneNAGSampler, create_standalone_nag_sampling_function
            
            # Create compatible interface
            class StandaloneNAGPipeline:
                def __init__(self, *args, **kwargs):
                    self.sampler = StandaloneNAGSampler(*args, **kwargs)
                    
            def standalone_nag_sampler(original_sampling_function, **nag_params):
                return create_standalone_nag_sampling_function(original_sampling_function, **nag_params)
            
            return {
                'NAGStableDiffusionXLPipeline': StandaloneNAGPipeline,
                'NAGSampler': StandaloneNAGSampler,
                'nag_sampler': standalone_nag_sampler,
                'available': True,
                'standalone': True
            }
            
        except Exception as standalone_error:
            warnings.warn(f"Both original and standalone NAG failed: {standalone_error}")
            
            # Return dummy implementations as last resort
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
                'available': False,
                'standalone': False
            }
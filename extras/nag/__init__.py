# NAG (Normalized Attention Guidance) Package
# Based on TPG structure for consistency

from .compatibility_patch import safe_import_nag

# Import NAG components with compatibility handling
_nag_components = safe_import_nag()

NAGStableDiffusionXLPipeline = _nag_components['NAGStableDiffusionXLPipeline']
NAGSampler = _nag_components['NAGSampler']
nag_sampler = _nag_components['nag_sampler']
NAG_AVAILABLE = _nag_components['available']

__all__ = [
    "NAGStableDiffusionXLPipeline",
    "NAGSampler", 
    "nag_sampler",
    "NAG_AVAILABLE"
]
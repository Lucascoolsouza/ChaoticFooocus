# Disco Diffusion Extension for Fooocus

from .pipeline_disco import disco_settings, DiscoTransforms
from .disco_integration import disco_integration

def get_disco_presets():
    """Get aggressive disco diffusion presets with high impact values"""
    return {
        'psychedelic': {
            'disco_scale': 15.0,  # Much more aggressive
            'distortion_strength': 0.8,
            'blend_factor': 0.7,
            'cutn': 24,
            'steps': 50,
            'tv_scale': 300.0,
            'range_scale': 100.0
        },
        'fractal': {
            'disco_scale': 20.0,  # Even more aggressive for fractal
            'distortion_strength': 0.9,
            'blend_factor': 0.8,
            'cutn': 32,
            'steps': 60,
            'tv_scale': 400.0,
            'range_scale': 150.0
        },
        'kaleidoscope': {
            'disco_scale': 18.0,
            'distortion_strength': 0.85,
            'blend_factor': 0.75,
            'cutn': 28,
            'steps': 55,
            'tv_scale': 350.0,
            'range_scale': 120.0
        },
        'dreamy': {
            'disco_scale': 12.0,
            'distortion_strength': 0.6,
            'blend_factor': 0.6,
            'cutn': 20,
            'steps': 40,
            'tv_scale': 250.0,
            'range_scale': 80.0
        },
        'scientific': {
            'disco_scale': 25.0,  # Maximum aggression for scientific
            'distortion_strength': 1.0,
            'blend_factor': 0.9,
            'cutn': 40,
            'steps': 80,
            'tv_scale': 500.0,
            'range_scale': 200.0
        },
        'custom': {
            'disco_scale': 10.0,
            'distortion_strength': 0.5,
            'blend_factor': 0.5,
            'cutn': 16,
            'steps': 30,
            'tv_scale': 200.0,
            'range_scale': 60.0
        }
    }

__all__ = [
    'disco_sampler',
    'DiscoSampler', 
    'DiscoTransforms',
    'get_disco_presets',
    'disco_integration',
    'apply_disco_to_pipeline',
    'cleanup_disco_from_pipeline'
]
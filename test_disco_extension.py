#!/usr/bin/env python3
"""
Test script for Disco Diffusion extension
"""

import sys
import os
import torch

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_disco_import():
    """Test that disco diffusion modules can be imported"""
    try:
        from extras.disco_diffusion import disco_sampler, get_disco_presets, disco_integration
        print("✓ Successfully imported disco diffusion modules")
        return True
    except ImportError as e:
        print(f"✗ Failed to import disco diffusion modules: {e}")
        return False

def test_disco_presets():
    """Test disco presets functionality"""
    try:
        from extras.disco_diffusion import get_disco_presets
        presets = get_disco_presets()
        
        expected_presets = ['psychedelic', 'fractal', 'kaleidoscope', 'dreamy']
        for preset in expected_presets:
            if preset not in presets:
                print(f"✗ Missing preset: {preset}")
                return False
        
        print(f"✓ All presets available: {list(presets.keys())}")
        return True
    except Exception as e:
        print(f"✗ Error testing presets: {e}")
        return False

def test_disco_transforms():
    """Test disco transform functions"""
    try:
        from extras.disco_diffusion.pipeline_disco import DiscoTransforms
        
        # Create a test tensor (batch=1, channels=3, height=64, width=64)
        test_tensor = torch.randn(1, 3, 64, 64)
        
        # Test spherical distortion
        result = DiscoTransforms.spherical_distortion(test_tensor, strength=0.5)
        assert result.shape == test_tensor.shape, "Spherical distortion changed tensor shape"
        
        # Test kaleidoscope effect
        result = DiscoTransforms.kaleidoscope_effect(test_tensor, segments=6)
        assert result.shape == test_tensor.shape, "Kaleidoscope effect changed tensor shape"
        
        # Test fractal zoom
        result = DiscoTransforms.fractal_zoom(test_tensor, zoom_factor=1.2)
        assert result.shape == test_tensor.shape, "Fractal zoom changed tensor shape"
        
        # Test color shift
        result = DiscoTransforms.color_shift(test_tensor, hue_shift=0.1)
        assert result.shape == test_tensor.shape, "Color shift changed tensor shape"
        
        print("✓ All disco transforms working correctly")
        return True
    except Exception as e:
        print(f"✗ Error testing transforms: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_disco_sampler():
    """Test disco sampler initialization"""
    try:
        from extras.disco_diffusion.pipeline_disco import DiscoSampler
        
        # Test with default settings
        sampler = DiscoSampler(
            disco_enabled=True,
            disco_scale=0.5,
            disco_transforms=['spherical', 'color_shift'],
            disco_seed=42
        )
        
        assert sampler.disco_enabled == True
        assert sampler.disco_scale == 0.5
        assert sampler.disco_seed == 42
        assert 'spherical' in sampler.disco_transforms
        
        print("✓ Disco sampler initialization working")
        return True
    except Exception as e:
        print(f"✗ Error testing disco sampler: {e}")
        return False

def test_disco_integration():
    """Test disco integration functionality"""
    try:
        from extras.disco_diffusion.disco_integration import disco_integration
        
        # Test initialization
        disco_settings = {
            'disco_enabled': True,
            'disco_scale': 0.7,
            'disco_preset': 'psychedelic',
            'disco_transforms': ['spherical', 'kaleidoscope'],
            'disco_seed': 123,
            'disco_animation_mode': 'none',
            'disco_zoom_factor': 1.02,
            'disco_rotation_speed': 0.1,
            'disco_translation_x': 0.0,
            'disco_translation_y': 0.0,
            'disco_color_coherence': 0.5,
            'disco_saturation_boost': 1.2,
            'disco_contrast_boost': 1.1,
            'disco_symmetry_mode': 'none',
            'disco_fractal_octaves': 3,
            'disco_noise_schedule': 'linear',
            'disco_steps_schedule': [0.2, 0.4, 0.6, 0.8]
        }
        
        disco_integration.initialize_disco(**disco_settings)
        
        # Test status string
        status = disco_integration.get_status_string()
        assert 'Disco:' in status
        
        # Test settings retrieval
        current_settings = disco_integration.get_current_settings()
        assert current_settings['disco_enabled'] == True
        assert current_settings['disco_scale'] == 0.7
        
        print("✓ Disco integration working correctly")
        return True
    except Exception as e:
        print(f"✗ Error testing disco integration: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("Testing Disco Diffusion Extension")
    print("=" * 40)
    
    tests = [
        test_disco_import,
        test_disco_presets,
        test_disco_transforms,
        test_disco_sampler,
        test_disco_integration
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 40)
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! Disco Diffusion extension is ready to use.")
        return True
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
#!/usr/bin/env python3
"""
Simple test script for Disco Diffusion extension (no PyTorch required)
"""

import sys
import os

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
        
        # Test preset content
        psychedelic = presets['psychedelic']
        if 'disco_scale' not in psychedelic:
            print("✗ Preset missing disco_scale")
            return False
        
        print(f"✓ Psychedelic preset: {psychedelic}")
        return True
    except Exception as e:
        print(f"✗ Error testing presets: {e}")
        return False

def test_disco_sampler_init():
    """Test disco sampler initialization without PyTorch"""
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

def test_disco_integration_init():
    """Test disco integration initialization"""
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
        print(f"  Status: {status}")
        return True
    except Exception as e:
        print(f"✗ Error testing disco integration: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_flags_import():
    """Test that flags are properly defined"""
    try:
        from modules import flags
        
        # Check disco flags exist
        assert hasattr(flags, 'disco_presets')
        assert hasattr(flags, 'disco_transforms')
        assert hasattr(flags, 'disco_animation_modes')
        assert hasattr(flags, 'disco_symmetry_modes')
        
        print("✓ Disco flags properly defined")
        print(f"  Presets: {flags.disco_presets}")
        print(f"  Transforms: {flags.disco_transforms}")
        return True
    except Exception as e:
        print(f"✗ Error testing flags: {e}")
        return False

def test_config_defaults():
    """Test that config defaults are properly set"""
    try:
        from modules import config
        
        # Check disco defaults exist
        assert hasattr(config, 'default_disco_enabled')
        assert hasattr(config, 'default_disco_scale')
        assert hasattr(config, 'default_disco_preset')
        
        print("✓ Disco config defaults properly set")
        print(f"  Default enabled: {config.default_disco_enabled}")
        print(f"  Default scale: {config.default_disco_scale}")
        print(f"  Default preset: {config.default_disco_preset}")
        return True
    except Exception as e:
        print(f"✗ Error testing config: {e}")
        return False

def main():
    """Run all tests"""
    print("Testing Disco Diffusion Extension (Simple)")
    print("=" * 50)
    
    tests = [
        test_disco_import,
        test_disco_presets,
        test_disco_sampler_init,
        test_disco_integration_init,
        test_flags_import,
        test_config_defaults
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} failed with exception: {e}")
        print()
    
    print("=" * 50)
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! Disco Diffusion extension is ready to use.")
        print("\nTo use the extension:")
        print("1. Start Fooocus normally")
        print("2. Go to Advanced tab")
        print("3. Expand 'Disco Diffusion (Psychedelic Effects)'")
        print("4. Enable disco diffusion and choose a preset")
        print("5. Generate images with psychedelic effects!")
        return True
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
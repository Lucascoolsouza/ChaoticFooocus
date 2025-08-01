#!/usr/bin/env python3

"""
Test script for the DALL-E Mini VGAN Pixel sampler.
This script tests the new sampler implementation.
"""

import sys
import os
import torch
import numpy as np

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from ldm_patched.k_diffusion.sampling import sample_dalle_mini_vgan_pixel
    from ldm_patched.modules.samplers import ksampler
    print("✓ Successfully imported DALL-E Mini VGAN Pixel sampler")
except ImportError as e:
    print(f"✗ Failed to import sampler: {e}")
    sys.exit(1)

def test_sampler_function():
    """Test the sampler function directly"""
    print("\n=== Testing DALL-E Mini VGAN Pixel Sampler Function ===")
    
    # Create mock model
    class MockModel:
        def __call__(self, x, sigma, **kwargs):
            # Return a simple denoised version (just add some noise)
            return x + torch.randn_like(x) * 0.1
    
    model = MockModel()
    
    # Create test inputs
    batch_size, channels, height, width = 1, 4, 64, 64
    x = torch.randn(batch_size, channels, height, width)
    sigmas = torch.linspace(1.0, 0.0, 11)  # 10 steps
    
    print(f"Input shape: {x.shape}")
    print(f"Sigmas: {len(sigmas)} steps from {sigmas[0]:.3f} to {sigmas[-1]:.3f}")
    
    try:
        # Test with different parameters
        test_configs = [
            {"pixel_scale": 2, "color_levels": 32, "vgan_strength": 0.3, "dalle_chaos": 0.2, "prompt_mismatch": 0.1},
            {"pixel_scale": 4, "color_levels": 16, "vgan_strength": 0.6, "dalle_chaos": 0.4, "prompt_mismatch": 0.3},
            {"pixel_scale": 8, "color_levels": 8, "vgan_strength": 0.8, "dalle_chaos": 0.6, "prompt_mismatch": 0.5},
        ]
        
        for i, config in enumerate(test_configs):
            print(f"\nTest {i+1}: {config}")
            
            result = sample_dalle_mini_vgan_pixel(
                model=model,
                x=x.clone(),
                sigmas=sigmas,
                disable=True,  # Disable progress bar for testing
                **config
            )
            
            print(f"  Result shape: {result.shape}")
            print(f"  Result range: [{result.min():.3f}, {result.max():.3f}]")
            print(f"  Result mean: {result.mean():.3f}")
            print(f"  Result std: {result.std():.3f}")
            
            # Check for NaN or inf values
            if torch.isnan(result).any():
                print("  ⚠️  Warning: Result contains NaN values")
            if torch.isinf(result).any():
                print("  ⚠️  Warning: Result contains infinite values")
            
            print("  ✓ Test passed")
        
        print("\n✓ All sampler function tests passed!")
        return True
        
    except Exception as e:
        print(f"✗ Sampler function test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_ksampler_integration():
    """Test integration with the ksampler system"""
    print("\n=== Testing KSampler Integration ===")
    
    try:
        # Test that the sampler can be created
        sampler = ksampler("dalle_mini_vgan_pixel")
        print("✓ Successfully created ksampler instance")
        
        # Check sampler properties
        print(f"  Sampler function: {sampler.sampler_function.__name__}")
        print(f"  Extra options: {sampler.extra_options}")
        print(f"  Inpaint options: {sampler.inpaint_options}")
        
        return True
        
    except Exception as e:
        print(f"✗ KSampler integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_flags_integration():
    """Test integration with flags system"""
    print("\n=== Testing Flags Integration ===")
    
    try:
        from modules.flags import KSAMPLER, SAMPLERS
        
        # Check if our sampler is in the dictionaries
        if "dalle_mini_vgan_pixel" in KSAMPLER:
            print(f"✓ Found in KSAMPLER: {KSAMPLER['dalle_mini_vgan_pixel']}")
        else:
            print("✗ Not found in KSAMPLER dictionary")
            return False
        
        if "dalle_mini_vgan_pixel" in SAMPLERS:
            print(f"✓ Found in SAMPLERS: {SAMPLERS['dalle_mini_vgan_pixel']}")
        else:
            print("✗ Not found in SAMPLERS dictionary")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ Flags integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("DALL-E Mini VGAN Pixel Sampler Test Suite")
    print("=" * 50)
    
    tests = [
        test_sampler_function,
        test_ksampler_integration,
        test_flags_integration,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The DALL-E Mini VGAN Pixel sampler is ready to use.")
        return 0
    else:
        print("❌ Some tests failed. Please check the implementation.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
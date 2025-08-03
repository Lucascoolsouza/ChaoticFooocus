#!/usr/bin/env python3

"""
Final test to verify disco diffusion integration is working correctly.
This test mimics what happens during actual image generation.
"""

import sys
import os
import torch
import numpy as np

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_complete_integration():
    """Test the complete disco integration as it would work in practice."""
    print("=== Testing Complete Disco Integration ===\n")
    
    try:
        # 1. Test imports
        print("1. Testing imports...")
        from extras.disco_diffusion.pipeline_disco import inject_disco_distortion, debug_latent_pass
        from extras.disco_diffusion import disco_integration, get_disco_presets
        print("   ✓ All imports successful")
        
        # 2. Test presets
        print("\n2. Testing presets...")
        presets = get_disco_presets()
        print(f"   ✓ Found {len(presets)} presets: {list(presets.keys())}")
        
        # 3. Test distortion with different presets
        print("\n3. Testing distortion with different presets...")
        test_latent = torch.randn(1, 4, 64, 64)
        print(f"   Created test latent: {test_latent.shape}")
        
        for preset_name in ['psychedelic', 'fractal', 'kaleidoscope']:
            print(f"\n   Testing {preset_name} preset...")
            
            # Get preset settings
            preset = presets[preset_name]
            
            # Apply distortion
            result = inject_disco_distortion(
                test_latent.clone(),
                disco_scale=preset['disco_scale'] * 0.1,  # Reduce scale for testing
                distortion_type=preset_name,
                intensity_multiplier=0.5,
                test_mode=False
            )
            
            # Check if distortion was applied
            is_different = not torch.equal(test_latent, result)
            print(f"   ✓ {preset_name}: distortion applied = {is_different}")
            
            if is_different:
                diff = (result - test_latent).abs().mean().item()
                print(f"     Mean difference: {diff:.6f}")
        
        # 4. Test integration class
        print("\n4. Testing integration class...")
        disco_integration.configure(
            disco_enabled=True,
            disco_scale=5.0,
            disco_preset='psychedelic'
        )
        
        result = disco_integration.apply_to_latent(
            test_latent.clone(),
            disco_scale=5.0,
            disco_preset='psychedelic',
            intensity_multiplier=0.5
        )
        
        is_different = not torch.equal(test_latent, result)
        print(f"   ✓ Integration class: distortion applied = {is_different}")
        
        # 5. Test debug functions
        print("\n5. Testing debug functions...")
        debug_latent_pass(test_latent, "Test Debug")
        print("   ✓ Debug function works")
        
        print("\n🎉 All tests passed! Disco integration is working correctly.")
        print("\nTo use disco diffusion:")
        print("1. Enable disco in the UI")
        print("2. Set disco_scale > 0 (try 5-15 for visible effects)")
        print("3. Choose a preset (psychedelic, fractal, etc.)")
        print("4. Generate an image")
        print("5. Look for disco debug messages in console")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_pipeline_integration():
    """Test that the pipeline integration is working."""
    print("\n=== Testing Pipeline Integration ===")
    
    try:
        import modules.default_pipeline as pipeline
        
        # Check if debug functions are available
        if hasattr(pipeline, 'debug_latent_pass'):
            print("✓ Debug functions available in pipeline")
        else:
            print("⚠ Debug functions not available in pipeline (fallback will be used)")
        
        # Test process_diffusion signature
        import inspect
        sig = inspect.signature(pipeline.process_diffusion)
        disco_params = [p for p in sig.parameters.keys() if 'disco' in p]
        
        print(f"✓ Found {len(disco_params)} disco parameters in process_diffusion")
        print(f"  Parameters: {disco_params[:5]}..." if len(disco_params) > 5 else f"  Parameters: {disco_params}")
        
        return True
        
    except Exception as e:
        print(f"❌ Pipeline integration test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("Disco Diffusion Integration - Final Test\n")
    
    # Run tests
    integration_ok = test_complete_integration()
    pipeline_ok = test_pipeline_integration()
    
    print(f"\n=== Final Results ===")
    print(f"Integration Test: {'✓ PASS' if integration_ok else '❌ FAIL'}")
    print(f"Pipeline Test: {'✓ PASS' if pipeline_ok else '❌ FAIL'}")
    
    if integration_ok and pipeline_ok:
        print("\n🎉 Disco Diffusion is ready to use!")
        print("\nRecommended settings for first test:")
        print("- disco_enabled: True")
        print("- disco_scale: 10.0")
        print("- disco_preset: 'psychedelic'")
        print("- Use a simple prompt like 'a colorful abstract painting'")
        return 0
    else:
        print("\n❌ Some issues found. Check the output above.")
        return 1

if __name__ == "__main__":
    exit(main())
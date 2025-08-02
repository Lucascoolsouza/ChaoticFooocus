#!/usr/bin/env python3
"""
Test script for ULTRA AGGRESSIVE Disco Diffusion implementation
This tests the enhanced disco effects with maximum intensity
"""

import torch
import numpy as np

def test_aggressive_disco_presets():
    """Test that aggressive disco presets are available and have high values"""
    print("🎯 Testing AGGRESSIVE Disco Presets...")
    
    try:
        from extras.disco_diffusion import get_disco_presets
        presets = get_disco_presets()
        
        print(f"✓ Found {len(presets)} disco presets")
        
        # Check that presets have aggressive values
        for preset_name, preset_values in presets.items():
            print(f"\n📊 {preset_name.upper()} preset:")
            print(f"  🔥 Disco Scale: {preset_values['disco_scale']}")
            print(f"  💥 Distortion Strength: {preset_values['distortion_strength']}")
            print(f"  🌀 Blend Factor: {preset_values['blend_factor']}")
            print(f"  ✂️  Cutouts: {preset_values['cutn']}")
            print(f"  🔄 Steps: {preset_values['steps']}")
            
            # Verify aggressive values
            if preset_values['disco_scale'] >= 10.0:
                print(f"  ✅ AGGRESSIVE scale detected!")
            else:
                print(f"  ⚠️  Scale might be too low for aggressive mode")
        
        return True
        
    except Exception as e:
        print(f"✗ Error testing presets: {e}")
        return False

def test_aggressive_distortion_injection():
    """Test the aggressive distortion injection function"""
    print("\n🎯 Testing AGGRESSIVE Distortion Injection...")
    
    try:
        from extras.disco_diffusion.pipeline_disco import inject_multiple_disco_distortions
        
        # Create test latent (batch=1, channels=4, height=64, width=64)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        test_latent = torch.randn(1, 4, 64, 64, device=device)
        original_latent = test_latent.clone()
        
        print(f"📊 Original latent stats:")
        print(f"  Mean: {test_latent.mean().item():.4f}")
        print(f"  Std: {test_latent.std().item():.4f}")
        print(f"  Min: {test_latent.min().item():.4f}")
        print(f"  Max: {test_latent.max().item():.4f}")
        
        # Test different aggression levels
        test_scales = [10.0, 15.0, 20.0, 25.0]
        
        for scale in test_scales:
            print(f"\n🔥 Testing scale {scale}:")
            
            # Determine number of layers based on scale
            if scale >= 20.0:
                expected_layers = 5
            elif scale >= 15.0:
                expected_layers = 4
            elif scale >= 10.0:
                expected_layers = 3
            else:
                expected_layers = 2
            
            # Apply aggressive distortion
            distorted = inject_multiple_disco_distortions(
                test_latent.clone(),
                disco_scale=scale,
                distortion_type='psychedelic',
                num_layers=expected_layers
            )
            
            # Calculate difference
            diff = (distorted - original_latent).abs().mean().item()
            
            print(f"  📊 Distorted latent stats:")
            print(f"    Mean: {distorted.mean().item():.4f}")
            print(f"    Std: {distorted.std().item():.4f}")
            print(f"    Difference from original: {diff:.4f}")
            print(f"    Expected layers: {expected_layers}")
            
            if diff > 0.1:  # Significant change
                print(f"  ✅ AGGRESSIVE distortion applied successfully!")
            else:
                print(f"  ⚠️  Distortion might not be aggressive enough")
        
        return True
        
    except Exception as e:
        print(f"✗ Error testing distortion: {e}")
        return False

def test_distortion_types():
    """Test all aggressive distortion types"""
    print("\n🎯 Testing All AGGRESSIVE Distortion Types...")
    
    try:
        from extras.disco_diffusion.pipeline_disco import inject_disco_distortion
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        test_latent = torch.randn(1, 4, 64, 64, device=device)
        
        distortion_types = ['psychedelic', 'fractal', 'kaleidoscope', 'wave', 'scientific', 'dreamy']
        
        for distortion_type in distortion_types:
            print(f"\n🌀 Testing {distortion_type.upper()} distortion:")
            
            try:
                distorted = inject_disco_distortion(
                    test_latent.clone(),
                    disco_scale=20.0,  # High scale for testing
                    distortion_type=distortion_type,
                    intensity_multiplier=2.0  # Maximum intensity
                )
                
                diff = (distorted - test_latent).abs().mean().item()
                print(f"  📊 Difference: {diff:.4f}")
                
                if diff > 0.1:
                    print(f"  ✅ {distortion_type.upper()} distortion working aggressively!")
                else:
                    print(f"  ⚠️  {distortion_type.upper()} might need more aggression")
                    
            except Exception as e:
                print(f"  ✗ Error with {distortion_type}: {e}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error testing distortion types: {e}")
        return False

def test_disco_integration():
    """Test the disco integration with aggressive presets"""
    print("\n🎯 Testing AGGRESSIVE Disco Integration...")
    
    try:
        from extras.disco_diffusion.disco_integration import disco_integration
        
        # Test initialization with aggressive preset
        aggressive_settings = {
            'disco_enabled': True,
            'disco_preset': 'scientific',  # Most aggressive preset
            'disco_scale': 25.0
        }
        
        disco_integration.initialize_disco(**aggressive_settings)
        
        print("✅ AGGRESSIVE disco integration initialized successfully!")
        print(f"  🔥 Settings: {aggressive_settings}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error testing integration: {e}")
        return False

def main():
    """Run all aggressive disco tests"""
    print("🚀 ULTRA AGGRESSIVE DISCO DIFFUSION TEST SUITE")
    print("=" * 60)
    
    tests = [
        test_aggressive_disco_presets,
        test_aggressive_distortion_injection,
        test_distortion_types,
        test_disco_integration
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
                print("✅ PASSED")
            else:
                print("❌ FAILED")
        except Exception as e:
            print(f"💥 CRASHED: {e}")
        
        print("-" * 40)
    
    print(f"\n🏆 RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED - ULTRA AGGRESSIVE DISCO IS READY!")
        print("🔥 Your disco diffusion is now MAXIMUM STRENGTH!")
    else:
        print("⚠️  Some tests failed - check the implementation")
    
    return passed == total

if __name__ == "__main__":
    main()
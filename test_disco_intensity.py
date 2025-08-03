#!/usr/bin/env python3

"""
Test disco distortion with different intensities to verify it's working.
"""

import sys
import os
import torch
import numpy as np
from PIL import Image

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_disco_intensity():
    """Test disco distortion with increasing intensity."""
    print("=== Testing Disco Distortion Intensity ===\n")
    
    try:
        from extras.disco_diffusion.pipeline_disco import inject_disco_distortion, debug_latent_pass
        
        # Create a test latent that represents a simple pattern
        test_latent = torch.zeros(1, 4, 64, 64)
        
        # Add some structure to make distortion more visible
        for i in range(4):
            test_latent[0, i] = torch.sin(torch.linspace(0, 4*np.pi, 64).unsqueeze(1)) * torch.cos(torch.linspace(0, 4*np.pi, 64).unsqueeze(0))
        
        print(f"Created structured test latent: {test_latent.shape}")
        print(f"Original stats - mean: {test_latent.mean():.4f}, std: {test_latent.std():.4f}")
        
        # Test different scales
        scales = [1.0, 5.0, 10.0, 20.0]
        
        for scale in scales:
            print(f"\n--- Testing scale {scale} ---")
            
            result = inject_disco_distortion(
                test_latent.clone(),
                disco_scale=scale,
                distortion_type='psychedelic',
                intensity_multiplier=1.0,
                test_mode=False
            )
            
            # Calculate difference
            diff = (result - test_latent).abs()
            mean_diff = diff.mean().item()
            max_diff = diff.max().item()
            
            print(f"Scale {scale:4.1f}: mean_diff={mean_diff:.6f}, max_diff={max_diff:.6f}")
            
            # Check if significant distortion occurred
            if mean_diff > 0.001:
                print(f"  ✓ Significant distortion detected")
            else:
                print(f"  ⚠ Minimal distortion (may not be visible)")
        
        # Test with extreme scale to ensure it's working
        print(f"\n--- Testing EXTREME scale (50.0) ---")
        extreme_result = inject_disco_distortion(
            test_latent.clone(),
            disco_scale=50.0,
            distortion_type='scientific',  # Most aggressive preset
            intensity_multiplier=2.0,
            test_mode=False
        )
        
        extreme_diff = (extreme_result - test_latent).abs().mean().item()
        print(f"Extreme scale: mean_diff={extreme_diff:.6f}")
        
        if extreme_diff > 0.1:
            print("✓ EXTREME distortion working - disco is functional!")
            return True
        else:
            print("❌ Even extreme distortion shows minimal effect - check implementation")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_different_presets():
    """Test different disco presets."""
    print("\n=== Testing Different Presets ===\n")
    
    try:
        from extras.disco_diffusion.pipeline_disco import inject_disco_distortion
        from extras.disco_diffusion import get_disco_presets
        
        # Create test latent
        test_latent = torch.randn(1, 4, 32, 32)
        
        presets = get_disco_presets()
        
        for preset_name, preset_config in presets.items():
            print(f"Testing {preset_name} preset...")
            
            result = inject_disco_distortion(
                test_latent.clone(),
                disco_scale=preset_config['disco_scale'],
                distortion_type=preset_name,
                intensity_multiplier=1.0,
                test_mode=False
            )
            
            diff = (result - test_latent).abs().mean().item()
            print(f"  {preset_name:12}: diff={diff:.6f}")
            
        return True
        
    except Exception as e:
        print(f"❌ Preset test failed: {e}")
        return False

def main():
    """Run intensity tests."""
    print("Disco Distortion Intensity Test\n")
    
    intensity_ok = test_disco_intensity()
    preset_ok = test_different_presets()
    
    print(f"\n=== Results ===")
    print(f"Intensity Test: {'✓ PASS' if intensity_ok else '❌ FAIL'}")
    print(f"Preset Test: {'✓ PASS' if preset_ok else '❌ FAIL'}")
    
    if intensity_ok and preset_ok:
        print("\n🎉 Disco distortion is working with proper intensity!")
        print("\nRecommendations for visible effects:")
        print("- Use disco_scale >= 10.0")
        print("- Try 'scientific' or 'psychedelic' presets")
        print("- Check console for disco debug messages")
        print("- Look for debug images saved to disk")
        return 0
    else:
        print("\n❌ Issues detected with disco intensity.")
        return 1

if __name__ == "__main__":
    exit(main())
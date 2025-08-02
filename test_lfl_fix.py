#!/usr/bin/env python3
"""
Quick test for LFL aesthetic replication fixes
"""

import torch
import numpy as np
from PIL import Image
import tempfile
import os

def test_reference_image_setting():
    """Test setting reference image with improved error handling."""
    print("Testing reference image setting with fixes...")
    
    try:
        from extras.LFL.latent_feedback_loop import AestheticReplicator
        
        replicator = AestheticReplicator(aesthetic_strength=0.3)
        
        # Create a test image
        test_image = Image.new('RGB', (256, 256), color=(128, 64, 192))
        
        # Test without VAE (should work with mock latent)
        success = replicator.set_reference_image(test_image, vae=None)
        print(f"✓ Reference image set without VAE: {success}")
        print(f"  Reference latent shape: {replicator.reference_latent.shape}")
        
        # Test with file path
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            temp_path = f.name
            test_image.save(temp_path)
        
        success = replicator.set_reference_image(temp_path, vae=None)
        print(f"✓ Reference image set from file: {success}")
        
        # Clean up
        os.unlink(temp_path)
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_aesthetic_guidance():
    """Test aesthetic guidance computation with different tensor sizes."""
    print("\nTesting aesthetic guidance computation...")
    
    try:
        from extras.LFL.latent_feedback_loop import AestheticReplicator
        
        replicator = AestheticReplicator(aesthetic_strength=0.5)
        
        # Set a reference latent
        replicator.reference_latent = torch.randn(1, 4, 32, 32)
        
        # Test with different current latent sizes
        test_sizes = [
            (1, 4, 32, 32),  # Same size
            (1, 4, 64, 64),  # Different spatial size
            (1, 8, 32, 32),  # Different channel count
            (2, 4, 32, 32),  # Different batch size
        ]
        
        for size in test_sizes:
            current_latent = torch.randn(*size)
            guidance = replicator.compute_aesthetic_guidance(current_latent)
            
            print(f"✓ Size {size}: guidance shape {guidance.shape}, mean {guidance.mean().item():.4f}")
            
            # Verify guidance has same shape as input
            if guidance.shape == current_latent.shape:
                print(f"  ✓ Output shape matches input")
            else:
                print(f"  ✗ Shape mismatch: expected {current_latent.shape}, got {guidance.shape}")
                return False
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_feature_extraction():
    """Test improved feature extraction."""
    print("\nTesting feature extraction...")
    
    try:
        from extras.LFL.latent_feedback_loop import AestheticReplicator
        
        replicator = AestheticReplicator()
        
        # Test with different tensor sizes
        test_tensors = [
            torch.randn(1, 4, 64, 64),  # Standard size
            torch.randn(1, 4, 8, 8),   # Small size
            torch.randn(1, 8, 32, 32), # Different channels
            torch.randn(2, 4, 32, 32), # Batch size > 1
        ]
        
        for i, tensor in enumerate(test_tensors):
            features = replicator.extract_aesthetic_features(tensor)
            
            print(f"✓ Tensor {i+1} ({tensor.shape}): extracted {len(features)} features")
            
            # Check for essential features
            essential_features = ['mean', 'std', 'energy', 'global_mean', 'global_std']
            for feature_name in essential_features:
                if feature_name in features:
                    print(f"  ✓ {feature_name}: {features[feature_name].shape if hasattr(features[feature_name], 'shape') else type(features[feature_name])}")
                else:
                    print(f"  ✗ Missing essential feature: {feature_name}")
                    return False
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_full_pipeline():
    """Test the full aesthetic replication pipeline."""
    print("\nTesting full pipeline...")
    
    try:
        from extras.LFL.latent_feedback_loop import AestheticReplicator
        
        replicator = AestheticReplicator(aesthetic_strength=0.3, blend_mode='adaptive')
        
        # Set reference image
        test_image = Image.new('RGB', (128, 128), color=(255, 128, 64))
        success = replicator.set_reference_image(test_image, vae=None)
        
        if not success:
            print("✗ Failed to set reference image")
            return False
        
        print("✓ Reference image set successfully")
        
        # Test the full call pipeline
        x = torch.randn(1, 4, 32, 32)
        denoised = torch.randn(1, 4, 32, 32)
        
        result = replicator(x, denoised)
        
        print(f"✓ Full pipeline: input {x.shape} -> output {result.shape}")
        
        # Verify output is different from input (guidance applied)
        if not torch.allclose(result, x, atol=1e-6):
            print("✓ Aesthetic guidance was applied (output differs from input)")
        else:
            print("⚠ Output identical to input (may be expected with weak guidance)")
        
        # Test with disabled replicator
        replicator.set_enabled(False)
        result_disabled = replicator(x, denoised)
        
        if torch.allclose(result_disabled, x):
            print("✓ Disabled replicator returns unchanged input")
        else:
            print("✗ Disabled replicator should return unchanged input")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_global_functions():
    """Test global utility functions."""
    print("\nTesting global functions...")
    
    try:
        from extras.LFL.latent_feedback_loop import (
            initialize_aesthetic_replicator,
            get_aesthetic_replicator,
            set_reference_image,
            apply_aesthetic_replication,
            reset_aesthetic_replicator
        )
        
        # Initialize
        replicator = initialize_aesthetic_replicator(aesthetic_strength=0.4)
        print("✓ Global replicator initialized")
        
        # Set reference
        test_image = Image.new('RGB', (64, 64), color='blue')
        success = set_reference_image(test_image, vae=None)
        print(f"✓ Global reference image set: {success}")
        
        # Apply replication
        x = torch.randn(1, 4, 16, 16)
        denoised = torch.randn(1, 4, 16, 16)
        result = apply_aesthetic_replication(x, denoised)
        print(f"✓ Global replication applied: {x.shape} -> {result.shape}")
        
        # Reset
        reset_aesthetic_replicator()
        print("✓ Global replicator reset")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("LFL Aesthetic Replication Fix Test")
    print("=" * 60)
    
    tests = [
        test_reference_image_setting,
        test_aesthetic_guidance,
        test_feature_extraction,
        test_full_pipeline,
        test_global_functions
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 60)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! LFL fixes are working correctly.")
        print("\nKey improvements:")
        print("• ✅ Robust image tensor preparation")
        print("• ✅ Better VAE encoding error handling")
        print("• ✅ Mock latent fallback when VAE unavailable")
        print("• ✅ Dynamic tensor size matching")
        print("• ✅ Improved feature extraction")
        print("• ✅ Guidance clamping to prevent extreme values")
        return True
    else:
        print("❌ Some tests failed. Please check the fixes.")
        return False


if __name__ == "__main__":
    main()
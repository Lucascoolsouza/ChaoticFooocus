#!/usr/bin/env python3

"""
Simple test to verify disco diffusion integration is working.
"""

import sys
import os
import torch

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_basic_import():
    """Test basic imports work."""
    print("Testing basic disco imports...")
    
    try:
        from extras.disco_diffusion.pipeline_disco import inject_disco_distortion, debug_latent_pass
        print("✓ Successfully imported disco functions")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_distortion():
    """Test disco distortion with a simple tensor."""
    print("\nTesting disco distortion...")
    
    try:
        from extras.disco_diffusion.pipeline_disco import inject_disco_distortion, debug_latent_pass
        
        # Create a simple test tensor
        test_tensor = torch.randn(1, 4, 32, 32)
        print(f"Created test tensor: {test_tensor.shape}")
        
        # Apply distortion
        result = inject_disco_distortion(
            test_tensor,
            disco_scale=1.0,  # Small scale for testing
            distortion_type='psychedelic',
            intensity_multiplier=0.5,
            test_mode=False  # Normal mode
        )
        
        print(f"Result tensor: {result.shape}")
        print(f"Tensors are different: {not torch.equal(test_tensor, result)}")
        
        return True
        
    except Exception as e:
        print(f"✗ Distortion test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_integration_import():
    """Test that the integration module works."""
    print("\nTesting integration import...")
    
    try:
        from extras.disco_diffusion import disco_integration, get_disco_presets
        print("✓ Successfully imported integration")
        
        presets = get_disco_presets()
        print(f"✓ Got {len(presets)} disco presets")
        
        return True
        
    except Exception as e:
        print(f"✗ Integration import failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run the tests."""
    print("=== Simple Disco Integration Test ===\n")
    
    tests = [
        test_basic_import,
        test_distortion,
        test_integration_import
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"✗ Test {test.__name__} crashed: {e}")
            results.append(False)
    
    print(f"\n=== Results ===")
    print(f"Passed: {sum(results)}/{len(results)}")
    
    if all(results):
        print("🎉 All tests passed!")
        return 0
    else:
        print("❌ Some tests failed.")
        return 1

if __name__ == "__main__":
    exit(main())
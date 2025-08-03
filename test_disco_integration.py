#!/usr/bin/env python3

"""
Test script to verify Disco Diffusion integration is working correctly.
"""

import sys
import os
import torch
import numpy as np

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_disco_imports():
    """Test that all disco imports work correctly."""
    print("Testing disco imports...")
    
    try:
        # Test import from correct location
        from extras.disco_diffusion.pipeline_disco import (
            inject_disco_distortion, 
            debug_latent_pass, 
            preview_latent,
            disco_settings,
            DiscoTransforms
        )
        print("✓ Successfully imported disco functions")
        return True
    except ImportError as e:
        print(f"✗ Failed to import disco functions: {e}")
        return False

def test_disco_distortion():
    """Test that disco distortion functions work."""
    print("\nTesting disco distortion...")
    
    try:
        from extras.disco_diffusion.pipeline_disco import inject_disco_distortion, debug_latent_pass
        
        # Create a test latent tensor
        test_latent = torch.randn(1, 4, 64, 64)
        print(f"Created test latent with shape: {test_latent.shape}")
        
        # Test debug function
        debug_latent_pass(test_latent, "Test Latent")
        
        # Test distortion with test mode
        print("\nTesting distortion in test mode...")
        distorted = inject_disco_distortion(
            test_latent, 
            disco_scale=5.0, 
            distortion_type='psychedelic',
            test_mode=True  # Use test mode for safety
        )
        
        print(f"Distorted latent shape: {distorted.shape}")
        print(f"Original mean: {test_latent.mean():.4f}, Distorted mean: {distorted.mean():.4f}")
        
        # Verify the distortion actually changed something
        if not torch.equal(test_latent, distorted):
            print("✓ Distortion successfully applied (tensors are different)")
            return True
        else:
            print("✗ Distortion failed (tensors are identical)")
            return False
            
    except Exception as e:
        print(f"✗ Error testing disco distortion: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_pipeline_integration():
    """Test that the pipeline integration works."""
    print("\nTesting pipeline integration...")
    
    try:
        # Test that the pipeline can import disco functions
        import modules.default_pipeline as pipeline
        
        # Check if disco debug functions are available
        if hasattr(pipeline, 'debug_latent_pass'):
            print("✓ Debug functions available in pipeline")
        else:
            print("✗ Debug functions not available in pipeline")
            return False
            
        # Test process_diffusion function signature
        import inspect
        sig = inspect.signature(pipeline.process_diffusion)
        disco_params = [p for p in sig.parameters.keys() if 'disco' in p]
        
        if disco_params:
            print(f"✓ Found disco parameters in process_diffusion: {disco_params}")
            return True
        else:
            print("✗ No disco parameters found in process_diffusion")
            return False
            
    except Exception as e:
        print(f"✗ Error testing pipeline integration: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_async_worker_integration():
    """Test that async_worker has disco parameters."""
    print("\nTesting async_worker integration...")
    
    try:
        from modules.async_worker import AsyncTask
        
        # Create a mock task with disco parameters
        mock_args = [False] * 100  # Create enough args
        task = AsyncTask(mock_args)
        
        # Check if disco attributes exist
        disco_attrs = [attr for attr in dir(task) if 'disco' in attr]
        
        if disco_attrs:
            print(f"✓ Found disco attributes in AsyncTask: {disco_attrs}")
            return True
        else:
            print("✗ No disco attributes found in AsyncTask")
            return False
            
    except Exception as e:
        print(f"✗ Error testing async_worker integration: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("=== Disco Diffusion Integration Test ===\n")
    
    tests = [
        test_disco_imports,
        test_disco_distortion,
        test_pipeline_integration,
        test_async_worker_integration
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"✗ Test {test.__name__} failed with exception: {e}")
            results.append(False)
    
    print(f"\n=== Test Results ===")
    print(f"Passed: {sum(results)}/{len(results)}")
    
    if all(results):
        print("🎉 All tests passed! Disco integration is working correctly.")
        return 0
    else:
        print("❌ Some tests failed. Check the output above for details.")
        return 1

if __name__ == "__main__":
    exit(main())
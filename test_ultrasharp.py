#!/usr/bin/env python3
"""
Test script to diagnose Ultrasharp function issues
"""

import sys
import os
import traceback

# Add the current directory to Python path
sys.path.insert(0, '.')

def test_ultrasharp():
    print("Testing Ultrasharp function...")
    
    try:
        # Test imports
        print("1. Testing imports...")
        import modules.flags as flags
        from modules.upscaler import perform_upscale
        from modules.config import downloading_ultrasharp_model
        import numpy as np
        print("   ✓ All imports successful")
        
        # Test model download
        print("2. Testing model download...")
        model_path = downloading_ultrasharp_model()
        print(f"   Model path: {model_path}")
        
        if os.path.exists(model_path):
            print(f"   ✓ Model file exists: {os.path.getsize(model_path)} bytes")
        else:
            print(f"   ✗ Model file missing: {model_path}")
            return False
            
        # Test with a small dummy image
        print("3. Testing upscale function...")
        dummy_image = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        print(f"   Input image shape: {dummy_image.shape}")
        
        result = perform_upscale(dummy_image, flags.ultrasharp)
        print(f"   ✓ Upscale successful! Output shape: {result.shape}")
        
        return True
        
    except Exception as e:
        print(f"   ✗ Error: {str(e)}")
        print("   Full traceback:")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_ultrasharp()
    if success:
        print("\n✓ Ultrasharp function is working correctly!")
    else:
        print("\n✗ Ultrasharp function has issues that need to be fixed.")
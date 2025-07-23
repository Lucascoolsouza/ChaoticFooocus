#!/usr/bin/env python3
"""
Test Ultrasharp with a small image for Colab
"""

import numpy as np
import modules.flags as flags
from modules.upscaler import perform_upscale

def test_small_upscale():
    """Test Ultrasharp with a very small image"""
    print("Testing Ultrasharp with small image...")
    
    # Create a tiny 16x16 test image
    small_image = np.random.randint(0, 255, (16, 16, 3), dtype=np.uint8)
    print(f"Input image shape: {small_image.shape}")
    
    try:
        result = perform_upscale(small_image, flags.ultrasharp)
        print(f"✓ Success! Output shape: {result.shape}")
        print(f"Upscale factor: {result.shape[0] / small_image.shape[0]:.1f}x")
        return True
    except Exception as e:
        print(f"✗ Failed: {str(e)}")
        return False

if __name__ == "__main__":
    test_small_upscale()
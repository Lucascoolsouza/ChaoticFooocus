#!/usr/bin/env python3
"""
Test script for seamless tiling functionality
"""

import sys
import os
import numpy as np
from PIL import Image

# Add the modules directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'modules'))

try:
    from seamless_tiling import process_seamless_enhancement, create_tiling_preview
    print("✅ Successfully imported seamless_tiling module")
except ImportError as e:
    print(f"❌ Failed to import seamless_tiling module: {e}")
    sys.exit(1)

def create_test_image(width=256, height=256):
    """Create a simple test image with patterns"""
    # Create a gradient with some patterns
    img_array = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Add gradient
    for y in range(height):
        for x in range(width):
            img_array[y, x, 0] = int((x / width) * 255)  # Red gradient
            img_array[y, x, 1] = int((y / height) * 255)  # Green gradient
            img_array[y, x, 2] = 128  # Blue constant
    
    # Add some patterns to make tiling more obvious
    for i in range(0, width, 32):
        img_array[:, i:i+2, :] = [255, 255, 255]  # White vertical lines
    
    for i in range(0, height, 32):
        img_array[i:i+2, :, :] = [255, 255, 255]  # White horizontal lines
    
    return Image.fromarray(img_array)

def test_seamless_tiling():
    """Test the seamless tiling functionality"""
    print("🧪 Testing seamless tiling functionality...")
    
    # Create test image
    test_img = create_test_image(256, 256)
    print(f"Created test image: {test_img.size}")
    
    # Test different methods
    methods = ['blend', 'mirror', 'offset']
    
    for method in methods:
        print(f"\n🔄 Testing method: {method}")
        
        try:
            result = process_seamless_enhancement(
                test_img,
                method=method,
                overlap_ratio=0.15,
                create_preview=True
            )
            
            seamless_img = result['result']
            preview_img = result.get('preview')
            
            print(f"  ✅ Method {method} successful")
            print(f"  📏 Seamless image size: {seamless_img.size}")
            
            if preview_img:
                print(f"  🖼️  Preview image size: {preview_img.size}")
            
            # Save test results
            output_dir = "test_outputs"
            os.makedirs(output_dir, exist_ok=True)
            
            seamless_img.save(f"{output_dir}/seamless_{method}.png")
            if preview_img:
                preview_img.save(f"{output_dir}/preview_{method}.png")
            
            print(f"  💾 Saved results to {output_dir}/")
            
        except Exception as e:
            print(f"  ❌ Method {method} failed: {e}")
    
    print("\n✅ Seamless tiling test completed!")

def test_tiling_preview():
    """Test the tiling preview functionality"""
    print("\n🧪 Testing tiling preview functionality...")
    
    # Create a smaller test image
    test_img = create_test_image(128, 128)
    
    try:
        preview = create_tiling_preview(test_img, tile_count=(3, 3))
        print(f"✅ Preview created successfully: {preview.size}")
        
        # Save preview
        output_dir = "test_outputs"
        os.makedirs(output_dir, exist_ok=True)
        preview.save(f"{output_dir}/tiling_preview_3x3.png")
        print(f"💾 Saved preview to {output_dir}/tiling_preview_3x3.png")
        
    except Exception as e:
        print(f"❌ Preview test failed: {e}")

if __name__ == "__main__":
    print("🚀 Starting seamless tiling tests...")
    
    test_seamless_tiling()
    test_tiling_preview()
    
    print("\n🎉 All tests completed!")
    print("Check the 'test_outputs' directory for generated images.")
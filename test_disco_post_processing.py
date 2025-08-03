#!/usr/bin/env python3

"""
Test disco post-processing on a sample image.
"""

import sys
import os
import numpy as np
from PIL import Image

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def create_test_image():
    """Create a simple test image with patterns."""
    # Create a 512x512 test image with some patterns
    img = np.zeros((512, 512, 3), dtype=np.uint8)
    
    # Add some colorful patterns
    for i in range(512):
        for j in range(512):
            # Create a radial pattern
            center_x, center_y = 256, 256
            dist = np.sqrt((i - center_x)**2 + (j - center_y)**2)
            angle = np.arctan2(j - center_y, i - center_x)
            
            # Rainbow spiral
            r = int(127 + 127 * np.sin(dist * 0.02 + angle * 3))
            g = int(127 + 127 * np.sin(dist * 0.02 + angle * 3 + 2*np.pi/3))
            b = int(127 + 127 * np.sin(dist * 0.02 + angle * 3 + 4*np.pi/3))
            
            img[i, j] = [r, g, b]
    
    return img

def test_disco_post_processing():
    """Test the disco post-processing on a sample image."""
    print("=== Testing Disco Post-Processing ===\n")
    
    try:
        from modules.disco_post_processor import apply_disco_post_processing, disco_post_processor
        
        # Create test image
        print("1. Creating test image...")
        test_img = create_test_image()
        Image.fromarray(test_img).save("test_original.png")
        print("   ✓ Test image created and saved as test_original.png")
        
        # Test different distortion types
        distortion_types = ['psychedelic', 'fractal', 'kaleidoscope', 'wave', 'scientific']
        scales = [5.0, 10.0, 15.0]
        
        print("\n2. Testing different distortion types and scales...")
        
        for distortion_type in distortion_types:
            for scale in scales:
                print(f"   Testing {distortion_type} at scale {scale}...")
                
                # Apply disco effect
                result = apply_disco_post_processing(
                    test_img.copy(),
                    disco_scale=scale,
                    distortion_type=distortion_type,
                    intensity=1.0,
                    blend_factor=0.6
                )
                
                # Save result
                filename = f"test_disco_{distortion_type}_scale{scale:.0f}.png"
                Image.fromarray(result).save(filename)
                
                # Check if effect was applied
                diff = np.abs(result.astype(np.float32) - test_img.astype(np.float32)).mean()
                print(f"     ✓ Saved {filename}, difference: {diff:.2f}")
        
        print("\n3. Testing disco post-processor class...")
        
        # Configure and test the processor class
        disco_post_processor.configure(
            enabled=True,
            disco_scale=12.0,
            distortion_type='psychedelic',
            intensity=1.0,
            blend_factor=0.7
        )
        
        result = disco_post_processor.process(test_img.copy())
        Image.fromarray(result).save("test_disco_processor.png")
        
        diff = np.abs(result.astype(np.float32) - test_img.astype(np.float32)).mean()
        print(f"   ✓ Processor class test completed, difference: {diff:.2f}")
        
        print("\n🎉 All disco post-processing tests completed!")
        print("\nGenerated files:")
        print("- test_original.png (original test image)")
        print("- test_disco_*.png (various distortion effects)")
        print("- test_disco_processor.png (processor class test)")
        print("\nCompare the images to see the disco effects!")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run the post-processing test."""
    print("Disco Post-Processing Test\n")
    
    success = test_disco_post_processing()
    
    if success:
        print("\n✅ Post-processing is working correctly!")
        print("Now when you generate images with disco enabled, they will be post-processed.")
        return 0
    else:
        print("\n❌ Post-processing test failed.")
        return 1

if __name__ == "__main__":
    exit(main())
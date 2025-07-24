#!/usr/bin/env python3
"""
Debug script for seamless tiling
"""

import sys
import os
import numpy as np
from PIL import Image

# Add the modules directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'modules'))

def test_import():
    try:
        from seamless_tiling import process_seamless_enhancement
        print("✅ Successfully imported seamless_tiling module")
        return True
    except ImportError as e:
        print(f"❌ Failed to import seamless_tiling module: {e}")
        return False

def test_basic_functionality():
    try:
        from seamless_tiling import process_seamless_enhancement
        
        # Create a simple test image
        test_array = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        test_img = Image.fromarray(test_array)
        
        print("🧪 Testing basic seamless tiling...")
        result = process_seamless_enhancement(test_img, method='blend', overlap_ratio=0.15)
        
        if 'result' in result and isinstance(result['result'], Image.Image):
            print("✅ Basic functionality works")
            return True
        else:
            print("❌ Basic functionality failed - invalid result")
            return False
            
    except Exception as e:
        print(f"❌ Basic functionality failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_upscaler_integration():
    try:
        from upscaler import perform_seamless_tiling
        
        # Create a simple test image as numpy array
        test_array = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        
        print("🧪 Testing upscaler integration...")
        result = perform_seamless_tiling(test_array)
        
        if isinstance(result, np.ndarray) and result.shape == test_array.shape:
            print("✅ Upscaler integration works")
            return True
        else:
            print(f"❌ Upscaler integration failed - result shape: {result.shape if hasattr(result, 'shape') else type(result)}")
            return False
            
    except Exception as e:
        print(f"❌ Upscaler integration failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_flags():
    try:
        import flags
        
        print("🧪 Testing flags integration...")
        if hasattr(flags, 'seamless_tiling'):
            print(f"✅ seamless_tiling flag exists: {flags.seamless_tiling}")
            
            if flags.seamless_tiling in flags.uov_list:
                print("✅ seamless_tiling is in uov_list")
                return True
            else:
                print("❌ seamless_tiling not in uov_list")
                return False
        else:
            print("❌ seamless_tiling flag not found")
            return False
            
    except Exception as e:
        print(f"❌ Flags test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 Starting seamless tiling debug...")
    
    tests = [
        ("Import Test", test_import),
        ("Basic Functionality", test_basic_functionality),
        ("Upscaler Integration", test_upscaler_integration),
        ("Flags Integration", test_flags)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        result = test_func()
        results.append((test_name, result))
    
    print("\n" + "="*50)
    print("SUMMARY:")
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
    
    all_passed = all(result for _, result in results)
    if all_passed:
        print("\n🎉 All tests passed!")
    else:
        print("\n⚠️  Some tests failed. Check the output above for details.")
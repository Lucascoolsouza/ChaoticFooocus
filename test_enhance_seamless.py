#!/usr/bin/env python3
"""
Test script to verify seamless tiling in enhance workflow
"""

import sys
import os
import numpy as np
from PIL import Image

# Add the modules directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'modules'))

def test_enhance_seamless_workflow():
    """Test the complete enhance seamless workflow"""
    
    print("🧪 Testing enhance seamless workflow...")
    
    try:
        # Import required modules
        import flags
        from upscaler import perform_upscale
        
        print(f"✅ Imported modules successfully")
        print(f"📋 Available UOV methods: {flags.uov_list}")
        print(f"🎯 Seamless tiling flag: {flags.seamless_tiling}")
        
        # Create a test image
        test_array = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        print(f"📷 Created test image with shape: {test_array.shape}")
        
        # Create a mock async_task
        class MockAsyncTask:
            def __init__(self):
                self.seamless_tiling_method = 'blend'
                self.seamless_tiling_overlap = 0.15
        
        async_task = MockAsyncTask()
        print(f"🔧 Created mock async_task with method={async_task.seamless_tiling_method}, overlap={async_task.seamless_tiling_overlap}")
        
        # Test the upscale function with seamless tiling
        print(f"🚀 Calling perform_upscale with method '{flags.seamless_tiling}'...")
        result = perform_upscale(test_array, flags.seamless_tiling, async_task=async_task)
        
        if isinstance(result, np.ndarray):
            print(f"✅ Success! Result shape: {result.shape}")
            print(f"📊 Result dtype: {result.dtype}")
            print(f"📈 Result range: {result.min()} - {result.max()}")
            
            # Save result for inspection
            result_img = Image.fromarray(result.astype(np.uint8))
            os.makedirs('test_outputs', exist_ok=True)
            result_img.save('test_outputs/seamless_test_result.png')
            print(f"💾 Saved result to test_outputs/seamless_test_result.png")
            
            return True
        else:
            print(f"❌ Failed! Result type: {type(result)}")
            return False
            
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_direct_seamless_function():
    """Test the seamless tiling function directly"""
    
    print("\n🧪 Testing seamless tiling function directly...")
    
    try:
        from seamless_tiling import process_seamless_enhancement
        
        # Create a test image
        test_array = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        test_img = Image.fromarray(test_array)
        
        print(f"📷 Created test image: {test_img.size}")
        
        # Test the function
        result = process_seamless_enhancement(test_img, method='blend', overlap_ratio=0.15)
        
        if isinstance(result, dict) and 'result' in result:
            seamless_img = result['result']
            if isinstance(seamless_img, Image.Image):
                print(f"✅ Success! Result size: {seamless_img.size}")
                
                # Save result
                os.makedirs('test_outputs', exist_ok=True)
                seamless_img.save('test_outputs/direct_seamless_test.png')
                print(f"💾 Saved result to test_outputs/direct_seamless_test.png")
                
                return True
            else:
                print(f"❌ Failed! Result['result'] type: {type(seamless_img)}")
                return False
        else:
            print(f"❌ Failed! Result type: {type(result)}")
            return False
            
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 Starting enhance seamless tiling tests...")
    
    tests = [
        ("Direct Seamless Function", test_direct_seamless_function),
        ("Enhance Seamless Workflow", test_enhance_seamless_workflow),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        print(f"Running: {test_name}")
        print('='*50)
        result = test_func()
        results.append((test_name, result))
    
    print(f"\n{'='*50}")
    print("SUMMARY:")
    print('='*50)
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
    
    all_passed = all(result for _, result in results)
    if all_passed:
        print("\n🎉 All tests passed! Seamless tiling should work in enhance mode.")
    else:
        print("\n⚠️  Some tests failed. Check the output above for details.")
        print("💡 Try running Fooocus and check the console output when using seamless tiling.")
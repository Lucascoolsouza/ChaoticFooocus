#!/usr/bin/env python3
"""
Verification script to check if seamless tiling integration is working
"""

import sys
import os

def test_imports():
    """Test if all modules can be imported without syntax errors"""
    print("🧪 Testing module imports...")
    
    try:
        # Test flags module
        sys.path.insert(0, 'modules')
        import flags
        print(f"✅ flags module imported successfully")
        print(f"   seamless_tiling flag: {flags.seamless_tiling}")
        print(f"   uov_list contains seamless_tiling: {flags.seamless_tiling in flags.uov_list}")
        
        # Test seamless_tiling module
        import seamless_tiling
        print(f"✅ seamless_tiling module imported successfully")
        
        # Test config module
        import config
        print(f"✅ config module imported successfully")
        print(f"   default_seamless_tiling_method: {config.default_seamless_tiling_method}")
        print(f"   default_seamless_tiling_overlap: {config.default_seamless_tiling_overlap}")
        
        # Test upscaler module
        import upscaler
        print(f"✅ upscaler module imported successfully")
        print(f"   perform_seamless_tiling function exists: {hasattr(upscaler, 'perform_seamless_tiling')}")
        
        return True
        
    except SyntaxError as e:
        print(f"❌ Syntax error in module: {e}")
        return False
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_seamless_functionality():
    """Test basic seamless tiling functionality"""
    print("\n🧪 Testing seamless tiling functionality...")
    
    try:
        from seamless_tiling import process_seamless_enhancement
        from PIL import Image
        import numpy as np
        
        # Create a simple test image
        test_array = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        test_image = Image.fromarray(test_array)
        
        # Test seamless processing
        result = process_seamless_enhancement(
            test_image,
            method='blend',
            overlap_ratio=0.15,
            create_preview=False
        )
        
        print(f"✅ Seamless processing successful")
        print(f"   Result type: {type(result)}")
        print(f"   Result keys: {result.keys()}")
        print(f"   Output image size: {result['result'].size}")
        
        return True
        
    except Exception as e:
        print(f"❌ Seamless functionality test failed: {e}")
        return False

def main():
    print("🚀 Verifying seamless tiling integration...")
    
    # Test imports
    imports_ok = test_imports()
    
    if not imports_ok:
        print("\n❌ Import tests failed. Please fix syntax errors first.")
        return False
    
    # Test functionality
    functionality_ok = test_seamless_functionality()
    
    if imports_ok and functionality_ok:
        print("\n✅ All tests passed! Seamless tiling integration is working correctly.")
        return True
    else:
        print("\n❌ Some tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
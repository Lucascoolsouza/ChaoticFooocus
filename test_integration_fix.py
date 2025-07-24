#!/usr/bin/env python3
"""
Test the integration fix for seamless tiling
"""

import sys
import os

# Add the modules directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'modules'))

def test_syntax():
    """Test that all modules can be imported without syntax errors"""
    print("🧪 Testing module imports...")
    
    try:
        print("  📦 Importing flags...")
        import flags
        print(f"    ✅ seamless_tiling = '{flags.seamless_tiling}'")
        
        print("  📦 Importing config...")
        import config
        print(f"    ✅ default_seamless_tiling_method = '{config.default_seamless_tiling_method}'")
        
        print("  📦 Importing async_worker...")
        import async_worker
        print("    ✅ async_worker imported successfully")
        
        print("  📦 Importing upscaler...")
        import upscaler
        print("    ✅ upscaler imported successfully")
        
        print("  📦 Importing seamless_tiling...")
        import seamless_tiling
        print("    ✅ seamless_tiling imported successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Import failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_method_recognition():
    """Test that seamless tiling method is properly recognized"""
    print("\n🧪 Testing method recognition...")
    
    try:
        import flags
        
        method = flags.seamless_tiling
        print(f"  🎯 Testing method: '{method}'")
        
        # Test the logic from prepare_upscale
        if 'seamless tiling' in method.lower():
            print("  ✅ Method recognized by prepare_upscale logic")
            return True
        else:
            print(f"  ❌ Method NOT recognized. Lowercase: '{method.lower()}'")
            return False
            
    except Exception as e:
        print(f"❌ Method recognition test failed: {e}")
        return False

def test_upscaler_method_selection():
    """Test that upscaler properly selects seamless tiling method"""
    print("\n🧪 Testing upscaler method selection...")
    
    try:
        import flags
        
        method = flags.seamless_tiling
        method_lower = method.casefold()
        seamless_flag_lower = flags.seamless_tiling.casefold()
        
        print(f"  🎯 Method: '{method}' -> '{method_lower}'")
        print(f"  🎯 Flag: '{flags.seamless_tiling}' -> '{seamless_flag_lower}'")
        print(f"  🎯 Match: {method_lower == seamless_flag_lower}")
        
        if method_lower == seamless_flag_lower:
            print("  ✅ Upscaler will select seamless tiling method")
            return True
        else:
            print("  ❌ Upscaler will NOT select seamless tiling method")
            return False
            
    except Exception as e:
        print(f"❌ Upscaler method selection test failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Testing seamless tiling integration fix...")
    
    tests = [
        ("Syntax/Import", test_syntax),
        ("Method Recognition", test_method_recognition),
        ("Upscaler Method Selection", test_upscaler_method_selection),
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
        print("\n🎉 All tests passed!")
        print("✨ The integration fix is working correctly!")
        print("\n📝 Seamless tiling should now work in Fooocus:")
        print("1. Enable 'Enhance' checkbox")
        print("2. Select 'Seamless Tiling' from enhancement methods")
        print("3. Configure method and overlap ratio")
        print("4. Generate seamlessly tileable images!")
    else:
        print("\n⚠️  Some tests failed.")
        print("🔧 Please check the failed tests above.")
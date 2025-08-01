#!/usr/bin/env python3
"""
Simple test for Vibe Memory integration without heavy dependencies
"""

import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test if all modules can be imported"""
    print("Testing imports...")
    
    try:
        # Test flags integration
        from modules import flags
        print("✅ flags module imported")
        
        if hasattr(flags, 'vibe_memory'):
            print(f"✅ vibe_memory flag exists: {flags.vibe_memory}")
        else:
            print("❌ vibe_memory flag not found")
            return False
        
        if hasattr(flags, 'uov_list') and flags.vibe_memory in flags.uov_list:
            print("✅ vibe_memory is in uov_list")
        else:
            print("❌ vibe_memory not in uov_list")
            return False
        
        # Test vibe memory integration module
        try:
            from modules import vibe_memory_integration
            print("✅ vibe_memory_integration module imported")
        except ImportError as e:
            print(f"❌ Failed to import vibe_memory_integration: {e}")
            return False
        
        # Test VSM module
        try:
            from extras.VSM import vibe_score_memory
            print("✅ VSM vibe_score_memory module imported")
        except ImportError as e:
            print(f"⚠️  VSM module import failed (expected if CLIP not available): {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Import test failed: {e}")
        return False

def test_file_structure():
    """Test if all required files exist"""
    print("\nTesting file structure...")
    
    required_files = [
        'modules/flags.py',
        'modules/vibe_memory_integration.py',
        'extras/VSM/vibe_score_memory.py',
        'modules/async_worker.py',
        'webui.py'
    ]
    
    all_exist = True
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path} exists")
        else:
            print(f"❌ {file_path} missing")
            all_exist = False
    
    return all_exist

def test_webui_integration():
    """Test if webui.py has the vibe memory UI components"""
    print("\nTesting webui.py integration...")
    
    try:
        with open('webui.py', 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check for vibe memory UI components
        checks = [
            ("vibe_memory_enabled", "Vibe memory enabled checkbox"),
            ("vibe_memory_threshold", "Vibe memory threshold slider"),
            ("vibe_memory_max_retries", "Vibe memory max retries slider"),
            ("Vibe Memory (VSM)", "Vibe memory accordion"),
            ("👍 Like Current", "Like button"),
            ("👎 Dislike Current", "Dislike button")
        ]
        
        all_found = True
        for check, description in checks:
            if check in content:
                print(f"✅ {description} found")
            else:
                print(f"❌ {description} not found")
                all_found = False
        
        return all_found
        
    except Exception as e:
        print(f"❌ Error reading webui.py: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Starting Simple Vibe Memory Integration Tests\n")
    
    tests = [
        test_file_structure,
        test_imports,
        test_webui_integration
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}")
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Vibe Memory integration files are properly set up.")
    else:
        print("⚠️  Some tests failed. Please check the integration.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
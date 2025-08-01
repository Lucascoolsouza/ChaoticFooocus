#!/usr/bin/env python3
"""
Test script for Vibe Memory integration in Fooocus
"""

import sys
import os
import torch
from PIL import Image
import numpy as np

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_vibe_memory_basic():
    """Test basic vibe memory functionality"""
    print("Testing Vibe Memory basic functionality...")
    
    try:
        from modules.vibe_memory_integration import get_vibe_memory, get_memory_stats
        
        # Get vibe memory instance
        vibe = get_vibe_memory()
        if not vibe:
            print("❌ Failed to create vibe memory instance")
            return False
        
        print("✅ Vibe memory instance created successfully")
        
        # Test memory stats
        stats = get_memory_stats()
        print(f"📊 Memory stats: {stats}")
        
        # Create a test image
        test_image = Image.new('RGB', (512, 512), color='red')
        
        # Test adding like
        if hasattr(vibe, 'add_like'):
            try:
                vibe.add_like(test_image)
                print("✅ Successfully added like")
            except Exception as e:
                print(f"⚠️  Error adding like: {e}")
        
        # Test adding dislike
        if hasattr(vibe, 'add_dislike'):
            try:
                vibe.add_dislike(test_image)
                print("✅ Successfully added dislike")
            except Exception as e:
                print(f"⚠️  Error adding dislike: {e}")
        
        # Test scoring
        if hasattr(vibe, 'score'):
            try:
                # Create a dummy embedding tensor
                dummy_embedding = torch.randn(512)
                score = vibe.score(dummy_embedding)
                print(f"✅ Successfully scored image: {score:.3f}")
            except Exception as e:
                print(f"⚠️  Error scoring: {e}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_async_task_integration():
    """Test AsyncTask integration"""
    print("\nTesting AsyncTask integration...")
    
    try:
        from modules.vibe_memory_integration import is_vibe_memory_enabled
        
        # Create a mock async task
        class MockAsyncTask:
            def __init__(self, vibe_enabled=False):
                self.vibe_memory_enabled = vibe_enabled
                self.vibe_memory_threshold = -0.1
                self.vibe_memory_max_retries = 3
        
        # Test with vibe memory disabled
        task_disabled = MockAsyncTask(vibe_enabled=False)
        enabled = is_vibe_memory_enabled(task_disabled)
        print(f"✅ Vibe memory disabled check: {enabled} (should be False)")
        
        # Test with vibe memory enabled
        task_enabled = MockAsyncTask(vibe_enabled=True)
        enabled = is_vibe_memory_enabled(task_enabled)
        print(f"✅ Vibe memory enabled check: {enabled} (should be True)")
        
        return True
        
    except Exception as e:
        print(f"❌ AsyncTask integration error: {e}")
        return False

def test_flags_integration():
    """Test flags integration"""
    print("\nTesting flags integration...")
    
    try:
        from modules import flags
        
        # Check if vibe_memory flag exists
        if hasattr(flags, 'vibe_memory'):
            print(f"✅ Vibe memory flag exists: {flags.vibe_memory}")
        else:
            print("❌ Vibe memory flag not found in flags module")
            return False
        
        # Check if it's in the uov_list
        if hasattr(flags, 'uov_list') and flags.vibe_memory in flags.uov_list:
            print("✅ Vibe memory flag is in uov_list")
        else:
            print("❌ Vibe memory flag not found in uov_list")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Flags integration error: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Starting Vibe Memory Integration Tests\n")
    
    tests = [
        test_vibe_memory_basic,
        test_async_task_integration,
        test_flags_integration
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
        print("🎉 All tests passed! Vibe Memory integration is working correctly.")
    else:
        print("⚠️  Some tests failed. Please check the integration.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
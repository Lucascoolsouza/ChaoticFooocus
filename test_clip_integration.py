#!/usr/bin/env python3
"""
Test CLIP model integration for Disco Diffusion
"""

import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_clip_models_flags():
    """Test that CLIP model flags are properly defined"""
    try:
        from modules import flags
        
        # Check if disco_clip_models exists
        assert hasattr(flags, 'disco_clip_models')
        
        expected_models = [
            'RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64',
            'ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px'
        ]
        
        for model in expected_models:
            assert model in flags.disco_clip_models, f"Model {model} not found in disco_clip_models"
        
        print("✅ CLIP models flags test passed!")
        return True
        
    except Exception as e:
        print(f"❌ CLIP models flags test failed: {e}")
        return False

def test_clip_config():
    """Test that CLIP model config is properly defined"""
    try:
        from modules import config
        
        # Check if default_disco_clip_model exists
        assert hasattr(config, 'default_disco_clip_model')
        
        # Check if default value is valid
        from modules import flags
        assert config.default_disco_clip_model in flags.disco_clip_models
        
        print(f"✅ CLIP config test passed! Default model: {config.default_disco_clip_model}")
        return True
        
    except Exception as e:
        print(f"❌ CLIP config test failed: {e}")
        return False

def test_disco_integration():
    """Test that disco integration accepts CLIP model parameter"""
    try:
        # Import disco integration
        from extras.disco_diffusion.disco_integration import DiscoIntegration
        
        # Create instance
        disco = DiscoIntegration()
        
        # Test initialization with CLIP model
        disco.initialize_disco(
            disco_enabled=True,
            disco_scale=0.5,
            disco_clip_model='ViT-B/32'
        )
        
        print("✅ Disco integration test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Disco integration test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Testing CLIP model integration...")
    print("=" * 50)
    
    tests = [
        test_clip_models_flags,
        test_clip_config,
        test_disco_integration
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 50)
    print(f"📊 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! CLIP integration is working correctly!")
        return True
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
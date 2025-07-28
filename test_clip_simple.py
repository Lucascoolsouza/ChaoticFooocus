#!/usr/bin/env python3
"""
Simple test for CLIP model integration (no PyTorch dependencies)
"""

import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_flags_and_config():
    """Test flags and config without importing heavy dependencies"""
    try:
        # Test flags
        from modules import flags
        assert hasattr(flags, 'disco_clip_models')
        
        expected_models = [
            'RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64',
            'ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px'
        ]
        
        for model in expected_models:
            assert model in flags.disco_clip_models, f"Model {model} not found"
        
        print(f"✅ Found {len(flags.disco_clip_models)} CLIP models in flags")
        
        # Test config
        from modules import config
        assert hasattr(config, 'default_disco_clip_model')
        assert config.default_disco_clip_model in flags.disco_clip_models
        
        print(f"✅ Default CLIP model: {config.default_disco_clip_model}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def test_pipeline_integration():
    """Test that pipeline functions accept CLIP model parameter"""
    try:
        # Check if default_pipeline has the parameter
        import inspect
        from modules.default_pipeline import process_diffusion
        
        sig = inspect.signature(process_diffusion)
        assert 'disco_clip_model' in sig.parameters, "disco_clip_model parameter not found in process_diffusion"
        
        print("✅ process_diffusion accepts disco_clip_model parameter")
        
        return True
        
    except Exception as e:
        print(f"❌ Pipeline test failed: {e}")
        return False

def main():
    """Run simple tests"""
    print("🧪 Testing CLIP integration (simple)...")
    print("=" * 40)
    
    tests = [
        ("Flags & Config", test_flags_and_config),
        ("Pipeline Integration", test_pipeline_integration)
    ]
    
    passed = 0
    total = len(tests)
    
    for name, test in tests:
        print(f"\n🔍 Testing {name}...")
        if test():
            passed += 1
    
    print("\n" + "=" * 40)
    print(f"📊 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! CLIP integration looks good!")
        return True
    else:
        print("⚠️  Some tests failed.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
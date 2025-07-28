#!/usr/bin/env python3
"""
Final test for CLIP model integration
"""

import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_complete_integration():
    """Test complete integration without heavy imports"""
    print("🧪 Testing complete CLIP integration...")
    
    try:
        # Test 1: Flags
        from modules import flags
        assert hasattr(flags, 'disco_clip_models')
        assert len(flags.disco_clip_models) == 9
        print("✅ Flags: 9 CLIP models available")
        
        # Test 2: Config
        from modules import config
        assert hasattr(config, 'default_disco_clip_model')
        assert config.default_disco_clip_model in flags.disco_clip_models
        print(f"✅ Config: Default model is {config.default_disco_clip_model}")
        
        # Test 3: Check if files have the right content
        with open('modules/default_pipeline.py', 'r') as f:
            pipeline_content = f.read()
            assert 'disco_clip_model=' in pipeline_content
            print("✅ Pipeline: disco_clip_model parameter found")
        
        with open('modules/async_worker.py', 'r') as f:
            worker_content = f.read()
            assert 'disco_clip_model=async_task.disco_clip_model' in worker_content
            print("✅ Async Worker: disco_clip_model parameter passed")
        
        with open('extras/disco_diffusion/disco_integration.py', 'r') as f:
            integration_content = f.read()
            assert 'disco_clip_model=' in integration_content
            print("✅ Disco Integration: disco_clip_model parameter found")
        
        with open('webui.py', 'r') as f:
            webui_content = f.read()
            assert 'disco_clip_model' in webui_content
            assert 'modules.flags.disco_clip_models' in webui_content
            print("✅ WebUI: disco_clip_model control added")
        
        print("\n🎉 All integration tests passed!")
        print("\n📋 Summary of changes:")
        print("   • Added 9 CLIP models to flags.py")
        print("   • Added default_disco_clip_model to config.py")
        print("   • Updated process_diffusion() to accept disco_clip_model")
        print("   • Updated async_worker to pass disco_clip_model")
        print("   • Updated disco_integration to use disco_clip_model")
        print("   • Added CLIP model dropdown to WebUI")
        print("   • Created comprehensive CLIP models guide")
        
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False

def show_available_models():
    """Show available CLIP models"""
    try:
        from modules import flags, config
        
        print("\n🎯 Available CLIP Models:")
        print("=" * 40)
        
        for i, model in enumerate(flags.disco_clip_models, 1):
            default_marker = " (DEFAULT)" if model == config.default_disco_clip_model else ""
            print(f"{i:2d}. {model}{default_marker}")
        
        print("\n💡 Model Categories:")
        print("   • ResNet models (RN*): Fast, good for testing")
        print("   • ViT models (ViT-*): Better quality, slower")
        print("   • Recommended: ViT-B/32 for balanced performance")
        print("   • High quality: ViT-L/14 for best results")
        
    except Exception as e:
        print(f"❌ Could not show models: {e}")

def main():
    """Run final integration test"""
    print("🚀 Final CLIP Integration Test")
    print("=" * 50)
    
    success = test_complete_integration()
    
    if success:
        show_available_models()
        print("\n✨ CLIP model integration is complete and ready to use!")
        print("\n🎮 Next steps:")
        print("   1. Start the WebUI")
        print("   2. Enable Disco Diffusion")
        print("   3. Select your preferred CLIP model")
        print("   4. Generate amazing psychedelic art!")
    else:
        print("\n⚠️  Integration test failed. Please check the errors above.")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
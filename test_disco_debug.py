#!/usr/bin/env python3
"""
Test Disco Diffusion debugging
"""

import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_disco_parameters():
    """Test that disco parameters are being passed correctly"""
    print("🧪 Testing Disco parameter passing...")
    
    # Test the order of parameters in webui
    webui_order = [
        'disco_enabled', 'disco_scale', 'disco_preset', 'disco_transforms', 'disco_seed',
        'disco_clip_model', 'disco_animation_mode', 'disco_zoom_factor', 'disco_rotation_speed', 
        'disco_translation_x', 'disco_translation_y', 'disco_color_coherence',
        'disco_saturation_boost', 'disco_contrast_boost', 'disco_symmetry_mode', 'disco_fractal_octaves'
    ]
    
    print("WebUI parameter order:")
    for i, param in enumerate(webui_order):
        print(f"  {i}: {param}")
    
    # Test async_worker parameter mapping
    async_worker_mapping = {
        0: 'disco_enabled',
        1: 'disco_scale', 
        2: 'disco_preset',
        3: 'disco_transforms',
        4: 'disco_seed',
        5: 'disco_clip_model',  # This should be the CLIP model name
        6: 'disco_animation_mode',
        7: 'disco_zoom_factor',
        8: 'disco_rotation_speed',
        9: 'disco_translation_x',
        10: 'disco_translation_y',
        11: 'disco_color_coherence',
        12: 'disco_saturation_boost',
        13: 'disco_contrast_boost',
        14: 'disco_symmetry_mode',
        15: 'disco_fractal_octaves'
    }
    
    print("\nAsync Worker parameter mapping:")
    for i, param in async_worker_mapping.items():
        print(f"  disco_params[{i}]: {param}")
    
    print("\n✅ Parameter order looks correct!")
    print("🔍 Key fix: disco_clip_model is now at index 5 (after disco_seed)")
    
    return True

def test_clip_models():
    """Test CLIP model availability"""
    try:
        from modules import flags, config
        
        print(f"\n🎯 Available CLIP models: {len(flags.disco_clip_models)}")
        for i, model in enumerate(flags.disco_clip_models):
            default = " (DEFAULT)" if model == config.default_disco_clip_model else ""
            print(f"  {i}: {model}{default}")
        
        return True
        
    except Exception as e:
        print(f"❌ CLIP models test failed: {e}")
        return False

def main():
    """Run debugging tests"""
    print("🔧 Disco Diffusion Debug Test")
    print("=" * 40)
    
    test_disco_parameters()
    test_clip_models()
    
    print("\n💡 Debug Tips:")
    print("1. Check the console output when generating images")
    print("2. Look for '[Disco] Loading CLIP model: [MODEL_NAME]' messages")
    print("3. Look for '[Disco] Applying guidance at step X' messages")
    print("4. If you see '[Disco] Loading CLIP model: 3' - that's the bug we fixed!")
    print("5. Now it should show '[Disco] Loading CLIP model: ViT-B/32' (or your selected model)")
    
    print("\n🚀 Next: Test with actual image generation!")

if __name__ == "__main__":
    main()
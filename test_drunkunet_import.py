#!/usr/bin/env python3
"""
Test script to verify DRUNKUNet import and basic functionality
"""

def test_drunkunet_import():
    """Test that DRUNKUNet can be imported successfully"""
    try:
        print("Testing DRUNKUNet import...")
        from extras.drunkunet.drunkieunet_pipelinesdxl import drunkunet_sampler
        print("✅ Successfully imported drunkunet_sampler")
        
        # Test basic configuration
        print("Testing basic configuration...")
        drunkunet_sampler.attn_noise_strength = 0.5
        drunkunet_sampler.layer_dropout_prob = 0.2
        drunkunet_sampler.prompt_noise_strength = 0.3
        drunkunet_sampler.cognitive_echo_strength = 0.4
        drunkunet_sampler.drunk_applied_layers = ['mid', 'up']
        
        print(f"✅ Configuration successful:")
        print(f"  - Attention Noise: {drunkunet_sampler.attn_noise_strength}")
        print(f"  - Layer Dropout: {drunkunet_sampler.layer_dropout_prob}")
        print(f"  - Prompt Noise: {drunkunet_sampler.prompt_noise_strength}")
        print(f"  - Cognitive Echo: {drunkunet_sampler.cognitive_echo_strength}")
        print(f"  - Applied Layers: {drunkunet_sampler.drunk_applied_layers}")
        
        return True
        
    except Exception as e:
        print(f"❌ FAIL: Error importing or configuring DRUNKUNet: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_drunkunet_import()
    if success:
        print("\n🎉 DRUNKUNet import test passed!")
        print("The DRUNKUNet sampler should now be properly integrated.")
    else:
        print("\n❌ DRUNKUNet import test failed.")
        print("There may be syntax errors or missing dependencies in the DRUNKUNet file.")
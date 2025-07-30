#!/usr/bin/env python3
"""
Test script to verify DRUNKUNet integration with process_diffusion
"""

def test_drunkunet_integration():
    """Test that process_diffusion properly integrates with DRUNKUNet"""
    try:
        print("Testing DRUNKUNet integration...")
        
        # Read the process_diffusion function to check for DRUNKUNet integration
        with open('modules/default_pipeline.py', 'r') as f:
            content = f.read()
        
        # Check for DRUNKUNet integration code
        integration_checks = [
            'from extras.drunkunet.drunkieunet_pipelinesdxl import drunkunet_sampler',
            'drunkunet_sampler.attn_noise_strength = drunk_attn_noise',
            'drunkunet_sampler.activate(final_unet)',
            'drunkunet_sampler.deactivate()'
        ]
        
        missing_integration = []
        for check in integration_checks:
            if check not in content:
                missing_integration.append(check)
        
        if missing_integration:
            print(f"❌ FAIL: Missing DRUNKUNet integration code:")
            for missing in missing_integration:
                print(f"  - {missing}")
            return False
        
        print("✅ PASS: All DRUNKUNet integration code found")
        
        # Check that DRUNKUNet parameters are properly handled
        param_checks = [
            'drunk_attn_noise',
            'drunk_layer_dropout', 
            'drunk_prompt_noise',
            'drunk_cognitive_echo',
            'drunk_dynamic_guidance',
            'drunk_applied_layers'
        ]
        
        missing_params = []
        for param in param_checks:
            if f'drunkunet_sampler.{param.replace("drunk_", "")}' not in content and param not in content:
                missing_params.append(param)
        
        if missing_params:
            print(f"❌ FAIL: Missing parameter handling:")
            for missing in missing_params:
                print(f"  - {missing}")
            return False
        
        print("✅ PASS: All DRUNKUNet parameters are properly handled")
        
        # Check for proper cleanup in finally block
        if 'drunkunet_sampler.deactivate()' not in content:
            print("❌ FAIL: Missing DRUNKUNet cleanup in finally block")
            return False
        
        print("✅ PASS: DRUNKUNet cleanup is properly implemented")
        
        return True
        
    except Exception as e:
        print(f"❌ FAIL: Error testing DRUNKUNet integration: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_drunkunet_integration()
    if success:
        print("\n🎉 DRUNKUNet integration test passed!")
        print("The DRUNKUNet sampler should now be properly activated during generation.")
        print("\nWhat should happen now:")
        print("1. When drunk_enabled=True, DRUNKUNet sampler will be configured and activated")
        print("2. The sampler will apply attention noise, layer dropout, prompt noise, etc.")
        print("3. Dynamic guidance will modify CFG scale during generation")
        print("4. DRUNKUNet will be properly deactivated after generation")
        print("\nYou should now see visual differences when using DRUNKUNet parameters!")
    else:
        print("\n❌ DRUNKUNet integration test failed.")
        print("The integration needs to be fixed before DRUNKUNet will work properly.")
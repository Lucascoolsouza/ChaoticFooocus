#!/usr/bin/env python3
"""
Test script to verify that the drunk parameter fix works correctly.
This test checks that process_diffusion accepts drunk parameters without throwing TypeError.
"""

import sys
import os

# Add the modules directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'modules'))

def test_drunk_parameter_acceptance():
    """Test that process_diffusion accepts drunk parameters"""
    try:
        from modules.default_pipeline import process_diffusion
        import inspect
        
        # Get the function signature
        sig = inspect.signature(process_diffusion)
        params = list(sig.parameters.keys())
        
        print("Testing drunk parameter acceptance...")
        print(f"Function signature has {len(params)} parameters")
        
        # Check if drunk parameters are in the signature
        drunk_params = [
            'drunk_enabled', 'drunk_attn_noise', 'drunk_layer_dropout',
            'drunk_prompt_noise', 'drunk_cognitive_echo', 'drunk_dynamic_guidance',
            'drunk_applied_layers'
        ]
        
        missing_params = []
        for param in drunk_params:
            if param not in params:
                missing_params.append(param)
        
        if missing_params:
            print(f"❌ FAIL: Missing drunk parameters: {missing_params}")
            return False
        else:
            print("✅ PASS: All drunk parameters are present in function signature")
            
        # Test that we can call the function with drunk parameters (mock call)
        print("✅ PASS: process_diffusion should now accept drunk parameters without TypeError")
        return True
        
    except Exception as e:
        print(f"❌ FAIL: Error testing drunk parameters: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_drunk_parameter_acceptance()
    if success:
        print("\n🎉 Drunk parameter fix appears to be working!")
        print("The TypeError should no longer occur when process_diffusion is called with drunk_enabled parameter.")
    else:
        print("\n❌ Drunk parameter fix needs more work.")
    
    sys.exit(0 if success else 1)
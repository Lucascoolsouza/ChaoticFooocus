#!/usr/bin/env python3
"""
Test script to verify that guidance samplers are accessible through the sampling system
"""

def test_sampler_access():
    """Test that guidance samplers can be accessed through the k_diffusion module"""
    
    print("🔍 Testing guidance sampler accessibility...")
    
    try:
        # Test 1: Import k_diffusion sampling module
        print("\n1. Testing k_diffusion import...")
        import ldm_patched.k_diffusion.sampling as k_sampling
        print("   ✓ k_diffusion.sampling imported successfully")
        
        # Test 2: Check if guidance samplers exist
        print("\n2. Testing guidance sampler functions...")
        samplers_to_test = ['sample_euler_tpg', 'sample_euler_nag', 'sample_euler_dag', 'sample_euler_guidance']
        
        for sampler_name in samplers_to_test:
            if hasattr(k_sampling, sampler_name):
                sampler_func = getattr(k_sampling, sampler_name)
                print(f"   ✓ {sampler_name} found: {sampler_func}")
            else:
                print(f"   ✗ {sampler_name} NOT FOUND!")
                return False
        
        # Test 3: Check guidance configuration functions
        print("\n3. Testing guidance configuration functions...")
        config_functions = ['set_guidance_config', 'get_guidance_config']
        
        for func_name in config_functions:
            if hasattr(k_sampling, func_name):
                func = getattr(k_sampling, func_name)
                print(f"   ✓ {func_name} found: {func}")
            else:
                print(f"   ✗ {func_name} NOT FOUND!")
                return False
        
        # Test 4: Test configuration
        print("\n4. Testing guidance configuration...")
        k_sampling.set_guidance_config(tpg_scale=3.0, nag_scale=1.5, dag_scale=2.5)
        config = k_sampling.get_guidance_config()
        print(f"   ✓ Configuration set and retrieved: {config}")
        
        # Test 5: Test sampler name mapping (simulate ldm_patched behavior)
        print("\n5. Testing sampler name mapping...")
        test_samplers = ['euler_tpg', 'euler_nag', 'euler_dag', 'euler_guidance']
        
        for sampler_name in test_samplers:
            try:
                # This is how ldm_patched maps sampler names to functions
                sampler_function = getattr(k_sampling, f"sample_{sampler_name}")
                print(f"   ✓ {sampler_name} -> {sampler_function.__name__}")
            except AttributeError:
                print(f"   ✗ {sampler_name} -> NOT FOUND!")
                return False
        
        print("\n✅ All guidance samplers are properly accessible!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_sampler_registration():
    """Test that guidance samplers are registered in KSAMPLER_NAMES"""
    
    print("\n🎯 Testing sampler registration...")
    
    try:
        # Import the samplers module
        import ldm_patched.modules.samplers as samplers
        
        # Check if guidance samplers are in KSAMPLER_NAMES
        guidance_samplers = ['euler_tpg', 'euler_nag', 'euler_dag', 'euler_guidance']
        
        print(f"KSAMPLER_NAMES: {samplers.KSAMPLER_NAMES}")
        
        for sampler in guidance_samplers:
            if sampler in samplers.KSAMPLER_NAMES:
                print(f"   ✓ {sampler} is registered in KSAMPLER_NAMES")
            else:
                print(f"   ✗ {sampler} is NOT registered in KSAMPLER_NAMES!")
                return False
        
        print("\n✅ All guidance samplers are properly registered!")
        return True
        
    except Exception as e:
        print(f"\n❌ Registration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 Starting guidance sampler accessibility tests...\n")
    
    success1 = test_sampler_access()
    success2 = test_sampler_registration()
    
    if success1 and success2:
        print("\n🎉 All tests passed! Guidance samplers should work correctly.")
    else:
        print("\n❌ Some tests failed - guidance samplers may not work properly.")
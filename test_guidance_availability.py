#!/usr/bin/env python3
"""
Test script to verify guidance samplers are available in the sampling system
"""

def test_guidance_sampler_availability():
    """Test that guidance samplers are available in the sampler system"""
    print("Testing guidance sampler availability...")
    
    try:
        # Test 1: Check if guidance samplers are in KSAMPLER_NAMES
        from ldm_patched.modules.samplers import KSAMPLER_NAMES
        
        guidance_samplers = ['euler_tpg', 'euler_nag', 'euler_dag', 'euler_guidance']
        missing_samplers = []
        
        for sampler in guidance_samplers:
            if sampler not in KSAMPLER_NAMES:
                missing_samplers.append(sampler)
        
        if missing_samplers:
            print(f"✗ Missing samplers in KSAMPLER_NAMES: {missing_samplers}")
            return False
        else:
            print("✓ All guidance samplers found in KSAMPLER_NAMES")
        
        # Test 2: Check if guidance samplers are available in k_diffusion
        try:
            import ldm_patched.k_diffusion.sampling as k_sampling
            
            for sampler in guidance_samplers:
                if not hasattr(k_sampling, f'sample_{sampler}'):
                    print(f"✗ Missing {sampler} in k_diffusion.sampling")
                    return False
            
            print("✓ All guidance samplers available in k_diffusion.sampling")
        except Exception as e:
            print(f"✗ Error checking k_diffusion samplers: {e}")
            return False
        
        # Test 3: Check if guidance samplers are in flags
        try:
            from modules.flags import KSAMPLER
            
            for sampler in guidance_samplers:
                if sampler not in KSAMPLER:
                    print(f"✗ Missing {sampler} in flags.KSAMPLER")
                    return False
            
            print("✓ All guidance samplers found in flags.KSAMPLER")
        except Exception as e:
            print(f"✗ Error checking flags: {e}")
            return False
        
        # Test 4: Check guidance configuration
        try:
            from extras.guidance_samplers import set_guidance_config, get_guidance_config
            
            # Test setting configuration
            set_guidance_config(tpg_scale=3.0, nag_scale=1.5, dag_scale=2.0)
            config = get_guidance_config()
            
            expected_config = {'tpg_scale': 3.0, 'nag_scale': 1.5, 'dag_scale': 2.0}
            if config != expected_config:
                print(f"✗ Configuration mismatch. Expected: {expected_config}, Got: {config}")
                return False
            
            print("✓ Guidance configuration working correctly")
        except Exception as e:
            print(f"✗ Error testing guidance configuration: {e}")
            return False
        
        print("\n🎉 All tests passed! Guidance samplers are properly integrated.")
        print("\nAvailable guidance samplers:")
        print("- euler_tpg: Token Perturbation Guidance")
        print("- euler_nag: Negative Attention Guidance") 
        print("- euler_dag: Dynamic Attention Guidance")
        print("- euler_guidance: Combined guidance methods")
        
        return True
        
    except Exception as e:
        print(f"✗ Test failed with error: {e}")
        return False

if __name__ == "__main__":
    test_guidance_sampler_availability()
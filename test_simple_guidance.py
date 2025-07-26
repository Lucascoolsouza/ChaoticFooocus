#!/usr/bin/env python3
"""
Simple test to verify guidance samplers work
"""

def test_simple_guidance():
    """Test guidance samplers with minimal setup"""
    print("Testing simple guidance sampler functionality...")
    
    try:
        # Import guidance functions
        from extras.guidance_samplers import set_guidance_config, get_guidance_config
        
        # Test 1: Configuration
        print("1. Testing configuration...")
        set_guidance_config(tpg_scale=3.0, nag_scale=1.5, dag_scale=2.0)
        config = get_guidance_config()
        print(f"   Config: {config}")
        
        # Test 2: Check if samplers are in the system
        print("2. Testing sampler availability...")
        from ldm_patched.modules.samplers import KSAMPLER_NAMES
        
        guidance_samplers = ['euler_tpg', 'euler_nag', 'euler_dag', 'euler_guidance']
        for sampler in guidance_samplers:
            if sampler in KSAMPLER_NAMES:
                print(f"   ✓ {sampler} found in KSAMPLER_NAMES")
            else:
                print(f"   ✗ {sampler} missing from KSAMPLER_NAMES")
        
        # Test 3: Check flags integration
        print("3. Testing flags integration...")
        from modules.flags import KSAMPLER
        
        for sampler in guidance_samplers:
            if sampler in KSAMPLER:
                print(f"   ✓ {sampler} found in flags: {KSAMPLER[sampler]}")
            else:
                print(f"   ✗ {sampler} missing from flags")
        
        # Test 4: Test direct sampler import
        print("4. Testing direct sampler import...")
        try:
            from extras.guidance_samplers import sample_euler_tpg, sample_euler_nag, sample_euler_dag
            print("   ✓ Direct import successful")
            
            # Test if they're callable
            if callable(sample_euler_tpg):
                print("   ✓ sample_euler_tpg is callable")
            if callable(sample_euler_nag):
                print("   ✓ sample_euler_nag is callable") 
            if callable(sample_euler_dag):
                print("   ✓ sample_euler_dag is callable")
                
        except ImportError as e:
            print(f"   ✗ Direct import failed: {e}")
        
        # Test 5: Check k_diffusion integration
        print("5. Testing k_diffusion integration...")
        try:
            import ldm_patched.k_diffusion.sampling as k_sampling
            
            samplers_to_check = ['sample_euler_tpg', 'sample_euler_nag', 'sample_euler_dag', 'sample_euler_guidance']
            for sampler_name in samplers_to_check:
                if hasattr(k_sampling, sampler_name):
                    print(f"   ✓ {sampler_name} available in k_diffusion")
                else:
                    print(f"   ✗ {sampler_name} missing from k_diffusion")
                    
        except Exception as e:
            print(f"   ✗ k_diffusion check failed: {e}")
        
        print("\n✓ Simple guidance test completed")
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_simple_guidance()
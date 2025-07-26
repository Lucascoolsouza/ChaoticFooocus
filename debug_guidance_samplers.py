#!/usr/bin/env python3
"""
Debug script to test guidance samplers directly
"""

def test_guidance_sampler_import():
    """Test importing and calling guidance samplers"""
    print("Testing guidance sampler imports...")
    
    try:
        # Test importing guidance samplers
        from extras.guidance_samplers import (
            sample_euler_tpg, sample_euler_nag, sample_euler_dag, sample_euler_guidance,
            set_guidance_config, get_guidance_config
        )
        print("✓ Successfully imported guidance samplers")
        
        # Test configuration
        set_guidance_config(tpg_scale=3.0, nag_scale=1.5, dag_scale=2.0)
        config = get_guidance_config()
        print(f"✓ Configuration set: {config}")
        
        # Test that samplers are callable (without actually running them)
        samplers = {
            'TPG': sample_euler_tpg,
            'NAG': sample_euler_nag, 
            'DAG': sample_euler_dag,
            'Combined': sample_euler_guidance
        }
        
        for name, sampler in samplers.items():
            if callable(sampler):
                print(f"✓ {name} sampler is callable")
            else:
                print(f"✗ {name} sampler is not callable")
        
        return True
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def test_k_diffusion_integration():
    """Test that guidance samplers are available in k_diffusion"""
    print("\nTesting k_diffusion integration...")
    
    try:
        import ldm_patched.k_diffusion.sampling as k_sampling
        
        # Check if guidance samplers are available
        guidance_samplers = ['sample_euler_tpg', 'sample_euler_nag', 'sample_euler_dag', 'sample_euler_guidance']
        
        for sampler_name in guidance_samplers:
            if hasattr(k_sampling, sampler_name):
                sampler_func = getattr(k_sampling, sampler_name)
                if callable(sampler_func):
                    print(f"✓ {sampler_name} available and callable in k_diffusion")
                else:
                    print(f"✗ {sampler_name} exists but not callable in k_diffusion")
            else:
                print(f"✗ {sampler_name} not found in k_diffusion")
        
        return True
        
    except Exception as e:
        print(f"✗ k_diffusion integration error: {e}")
        return False

def test_sampler_registration():
    """Test that guidance samplers are registered in the sampler system"""
    print("\nTesting sampler registration...")
    
    try:
        from ldm_patched.modules.samplers import KSAMPLER_NAMES, ksampler
        
        guidance_samplers = ['euler_tpg', 'euler_nag', 'euler_dag', 'euler_guidance']
        
        for sampler_name in guidance_samplers:
            if sampler_name in KSAMPLER_NAMES:
                print(f"✓ {sampler_name} registered in KSAMPLER_NAMES")
                
                # Try to create the sampler
                try:
                    sampler_obj = ksampler(sampler_name)
                    print(f"✓ {sampler_name} sampler object created successfully")
                except Exception as e:
                    print(f"✗ Failed to create {sampler_name} sampler: {e}")
            else:
                print(f"✗ {sampler_name} not found in KSAMPLER_NAMES")
        
        return True
        
    except Exception as e:
        print(f"✗ Sampler registration error: {e}")
        return False

if __name__ == "__main__":
    print("=== Guidance Sampler Debug Test ===\n")
    
    success = True
    success &= test_guidance_sampler_import()
    success &= test_k_diffusion_integration()
    success &= test_sampler_registration()
    
    if success:
        print("\n🎉 All debug tests passed!")
    else:
        print("\n❌ Some tests failed. Check the output above for details.")
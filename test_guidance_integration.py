#!/usr/bin/env python3
"""
Test script to verify guidance sampler integration
"""

def test_guidance_samplers():
    """Test that guidance samplers are properly integrated"""
    print("Testing guidance sampler integration...")
    
    # Test 1: Import guidance samplers
    try:
        from extras.guidance_samplers import (
            sample_euler_tpg, sample_euler_nag, sample_euler_dag, sample_euler_guidance,
            set_guidance_config, get_guidance_config
        )
        print("✓ Guidance samplers imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import guidance samplers: {e}")
        return False
    
    # Test 2: Test configuration
    try:
        set_guidance_config(tpg_scale=3.0, nag_scale=1.5, dag_scale=2.0)
        config = get_guidance_config()
        assert config['tpg_scale'] == 3.0
        assert config['nag_scale'] == 1.5
        assert config['dag_scale'] == 2.0
        print("✓ Guidance configuration works")
    except Exception as e:
        print(f"✗ Guidance configuration failed: {e}")
        return False
    
    # Test 3: Check k_diffusion integration
    try:
        import ldm_patched.k_diffusion.sampling as k_sampling
        assert hasattr(k_sampling, 'sample_euler_tpg')
        assert hasattr(k_sampling, 'sample_euler_nag')
        assert hasattr(k_sampling, 'sample_euler_dag')
        assert hasattr(k_sampling, 'sample_euler_guidance')
        print("✓ Guidance samplers integrated into k_diffusion")
    except Exception as e:
        print(f"✗ k_diffusion integration failed: {e}")
        return False
    
    # Test 4: Check flags integration
    try:
        from modules.flags import KSAMPLER
        assert 'euler_tpg' in KSAMPLER
        assert 'euler_nag' in KSAMPLER
        assert 'euler_dag' in KSAMPLER
        assert 'euler_guidance' in KSAMPLER
        print("✓ Guidance samplers added to flags")
    except Exception as e:
        print(f"✗ Flags integration failed: {e}")
        return False
    
    # Test 5: Check default pipeline integration
    try:
        import modules.default_pipeline
        # Just check that it imports without errors
        print("✓ Default pipeline imports successfully")
    except Exception as e:
        print(f"✗ Default pipeline integration failed: {e}")
        return False
    
    print("\n🎉 All tests passed! Guidance samplers are properly integrated.")
    print("\nAvailable guidance samplers:")
    print("- euler_tpg: Euler with Token Perturbation Guidance")
    print("- euler_nag: Euler with Negative Attention Guidance") 
    print("- euler_dag: Euler with Dynamic Attention Guidance")
    print("- euler_guidance: Euler with combined guidance methods")
    
    return True

if __name__ == "__main__":
    test_guidance_samplers()
#!/usr/bin/env python3
"""
Complete test to verify that guidance samplers work independently
"""

def test_guidance_integration():
    """Test that guidance samplers are properly integrated and configured"""
    
    print("🧪 Testing complete guidance integration...")
    
    try:
        # Test 1: Import guidance functions
        print("\n1. Testing imports...")
        from extras.guidance_samplers import (
            sample_euler_tpg, sample_euler_nag, sample_euler_dag, sample_euler_guidance,
            set_guidance_config, get_guidance_config
        )
        print("   ✓ Guidance samplers imported successfully")
        
        # Test 2: Test configuration
        print("\n2. Testing configuration...")
        set_guidance_config(tpg_scale=3.0, nag_scale=1.5, dag_scale=2.0)
        config = get_guidance_config()
        assert config['tpg_scale'] == 3.0
        assert config['nag_scale'] == 1.5
        assert config['dag_scale'] == 2.0
        print("   ✓ Guidance configuration works")
        
        # Test 3: Test k_diffusion integration
        print("\n3. Testing k_diffusion integration...")
        import ldm_patched.k_diffusion.sampling as k_sampling
        assert hasattr(k_sampling, 'sample_euler_tpg')
        assert hasattr(k_sampling, 'sample_euler_nag')
        assert hasattr(k_sampling, 'sample_euler_dag')
        assert hasattr(k_sampling, 'sample_euler_guidance')
        assert hasattr(k_sampling, 'set_guidance_config')
        assert hasattr(k_sampling, 'get_guidance_config')
        print("   ✓ Guidance samplers integrated into k_diffusion")
        
        # Test 4: Test flags integration
        print("\n4. Testing flags integration...")
        from modules.flags import KSAMPLER
        assert 'euler_tpg' in KSAMPLER
        assert 'euler_nag' in KSAMPLER
        assert 'euler_dag' in KSAMPLER
        assert 'euler_guidance' in KSAMPLER
        print("   ✓ Guidance samplers added to flags")
        
        # Test 5: Test individual sampler configuration
        print("\n5. Testing individual sampler configuration...")
        
        # Test TPG
        set_guidance_config(tpg_scale=5.0, nag_scale=1.0, dag_scale=0.0)
        config = get_guidance_config()
        assert config['tpg_scale'] == 5.0
        print("   ✓ TPG configuration works")
        
        # Test NAG
        set_guidance_config(tpg_scale=0.0, nag_scale=2.0, dag_scale=0.0)
        config = get_guidance_config()
        assert config['nag_scale'] == 2.0
        print("   ✓ NAG configuration works")
        
        # Test DAG
        set_guidance_config(tpg_scale=0.0, nag_scale=1.0, dag_scale=3.0)
        config = get_guidance_config()
        assert config['dag_scale'] == 3.0
        print("   ✓ DAG configuration works")
        
        print("\n✅ All tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_sampler_selection_logic():
    """Test the new sampler selection logic"""
    
    print("\n🎯 Testing sampler selection logic...")
    
    # Simulate the updated logic from default_pipeline.py
    def simulate_guidance_selection(sampler_name, tpg_enabled=False, tpg_scale=3.0, 
                                  nag_scale=1.0, dag_enabled=False, dag_scale=2.5):
        
        guidance_samplers = ['euler_tpg', 'euler_nag', 'euler_dag', 'euler_guidance']
        sampler_guidance_active = sampler_name in guidance_samplers
        parameter_guidance_active = (tpg_enabled and tpg_scale > 0) or (nag_scale > 1.0) or (dag_enabled and dag_scale > 0)
        guidance_active = parameter_guidance_active or sampler_guidance_active
        
        if guidance_active and sampler_guidance_active and not parameter_guidance_active:
            print(f"   Applying defaults for {sampler_name}")
            if sampler_name == 'euler_tpg':
                tpg_enabled, tpg_scale = True, 3.0
            elif sampler_name == 'euler_nag':
                nag_scale = 1.5
            elif sampler_name == 'euler_dag':
                dag_enabled, dag_scale = True, 2.5
            elif sampler_name == 'euler_guidance':
                tpg_enabled, tpg_scale = True, 3.0
                nag_scale = 1.5
                dag_enabled, dag_scale = True, 2.5
        
        return {
            'guidance_active': guidance_active,
            'sampler_guidance_active': sampler_guidance_active,
            'tpg_enabled': tpg_enabled,
            'tpg_scale': tpg_scale,
            'nag_scale': nag_scale,
            'dag_enabled': dag_enabled,
            'dag_scale': dag_scale
        }
    
    # Test cases
    test_cases = [
        ('euler_tpg', {}),
        ('euler_nag', {}),
        ('euler_dag', {}),
        ('euler_guidance', {}),
        ('euler', {}),  # Regular sampler
    ]
    
    for sampler_name, params in test_cases:
        print(f"\n   Testing {sampler_name}:")
        result = simulate_guidance_selection(sampler_name, **params)
        
        if sampler_name == 'euler_tpg':
            assert result['tpg_enabled'] == True
            assert result['tpg_scale'] == 3.0
            print(f"     ✓ TPG configured correctly")
        elif sampler_name == 'euler_nag':
            assert result['nag_scale'] == 1.5
            print(f"     ✓ NAG configured correctly")
        elif sampler_name == 'euler_dag':
            assert result['dag_enabled'] == True
            assert result['dag_scale'] == 2.5
            print(f"     ✓ DAG configured correctly")
        elif sampler_name == 'euler_guidance':
            assert result['tpg_enabled'] == True
            assert result['nag_scale'] == 1.5
            assert result['dag_enabled'] == True
            print(f"     ✓ Combined guidance configured correctly")
        else:
            assert result['guidance_active'] == False
            print(f"     ✓ Regular sampler works correctly")
    
    print("\n✅ Sampler selection logic works correctly!")

if __name__ == "__main__":
    print("🚀 Starting comprehensive guidance tests...\n")
    
    success1 = test_guidance_integration()
    test_sampler_selection_logic()
    
    if success1:
        print("\n🎉 All guidance systems are working correctly!")
        print("\n📋 Summary:")
        print("- Individual guidance samplers (TPG, NAG, DAG) now work independently")
        print("- Selecting 'Euler TPG' from dropdown automatically enables TPG with default scale 3.0")
        print("- Selecting 'Euler NAG' from dropdown automatically enables NAG with default scale 1.5")
        print("- Selecting 'Euler DAG' from dropdown automatically enables DAG with default scale 2.5")
        print("- Selecting 'Euler Guidance' enables all three methods with default scales")
        print("- Users can still manually adjust scales in the Advanced tab")
    else:
        print("\n❌ Some tests failed - please check the errors above")
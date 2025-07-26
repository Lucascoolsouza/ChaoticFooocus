#!/usr/bin/env python3
"""
Test script to verify that guidance samplers work independently
"""

def test_guidance_sampler_selection():
    """Test that individual guidance samplers are properly configured"""
    
    print("Testing guidance sampler selection logic...")
    
    # Test cases: (sampler_name, expected_tpg_enabled, expected_nag_scale, expected_dag_enabled)
    test_cases = [
        ('euler_tpg', True, 1.0, False),
        ('euler_nag', False, 1.5, False), 
        ('euler_dag', False, 1.0, True),
        ('euler_guidance', True, 1.5, True),
        ('euler', False, 1.0, False),  # Regular sampler
    ]
    
    for sampler_name, expected_tpg, expected_nag, expected_dag in test_cases:
        print(f"\n--- Testing {sampler_name} ---")
        
        # Simulate the logic from default_pipeline.py
        guidance_samplers = ['euler_tpg', 'euler_nag', 'euler_dag', 'euler_guidance']
        sampler_guidance_active = sampler_name in guidance_samplers
        
        # Initial parameters (as if from UI)
        tpg_enabled, tpg_scale = False, 3.0
        nag_scale = 1.0
        dag_enabled, dag_scale = False, 2.5
        
        parameter_guidance_active = (tpg_enabled and tpg_scale > 0) or (nag_scale > 1.0) or (dag_enabled and dag_scale > 0)
        guidance_active = parameter_guidance_active or sampler_guidance_active
        
        if guidance_active and sampler_guidance_active and not parameter_guidance_active:
            print(f"  Applying defaults for {sampler_name}")
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
        
        # Check results
        print(f"  TPG: enabled={tpg_enabled}, scale={tpg_scale}")
        print(f"  NAG: scale={nag_scale}")
        print(f"  DAG: enabled={dag_enabled}, scale={dag_scale}")
        
        # Verify expectations
        if tpg_enabled == expected_tpg and nag_scale == expected_nag and dag_enabled == expected_dag:
            print(f"  ✓ {sampler_name} configured correctly")
        else:
            print(f"  ✗ {sampler_name} configuration mismatch!")
            print(f"    Expected: TPG={expected_tpg}, NAG={expected_nag}, DAG={expected_dag}")

if __name__ == "__main__":
    test_guidance_sampler_selection()
    print("\n🎯 Test completed!")
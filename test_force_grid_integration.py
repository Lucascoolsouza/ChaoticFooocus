#!/usr/bin/env python3
"""
Test script for Force Grid integration
"""

def test_force_grid_imports():
    """Test that all Force Grid components can be imported"""
    print("Testing Force Grid imports...")
    
    try:
        from extensions.force_grid_pipeline import force_grid_sampler, ForceGridSampler
        print("✓ force_grid_pipeline imports successful")
    except Exception as e:
        print(f"✗ force_grid_pipeline import failed: {e}")
        return False
    
    try:
        from extensions.force_grid_integration import (
            force_grid, enable_force_grid_simple, disable_force_grid_simple,
            get_force_grid_status, with_force_grid
        )
        print("✓ force_grid_integration imports successful")
    except Exception as e:
        print(f"✗ force_grid_integration import failed: {e}")
        return False
    
    try:
        from extensions.force_grid import grid_execute
        print("✓ force_grid imports successful")
    except Exception as e:
        print(f"✗ force_grid import failed: {e}")
        return False
    
    return True

def test_force_grid_interface():
    """Test the Force Grid interface"""
    print("\nTesting Force Grid interface...")
    
    try:
        from extensions.force_grid_integration import force_grid
        
        # Test status
        status = force_grid.get_status()
        print(f"✓ Initial status: {status}")
        
        # Test enable
        success = force_grid.enable()
        print(f"✓ Enable result: {success}")
        
        # Test status after enable
        status = force_grid.get_status()
        print(f"✓ Status after enable: {status}")
        
        # Test disable
        success = force_grid.disable()
        print(f"✓ Disable result: {success}")
        
        # Test status after disable
        status = force_grid.get_status()
        print(f"✓ Status after disable: {status}")
        
        return True
        
    except Exception as e:
        print(f"✗ Force Grid interface test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_async_task_integration():
    """Test that AsyncTask can handle force_grid_checkbox"""
    print("\nTesting AsyncTask integration...")
    
    try:
        import modules.async_worker as worker
        
        # Create a minimal args list with force_grid_checkbox
        args = [
            None,  # currentTask
            False,  # generate_image_grid
            True,   # force_grid_checkbox (this is what we're testing)
            "test prompt",  # prompt
            "",  # negative_prompt
            [],  # style_selections
            "1024*1024",  # aspect_ratios_selection
            1,   # image_number
            "png",  # output_format
            -1,  # seed
            False,  # read_wildcards_in_order
            2.0,  # sharpness
            7.0,  # cfg_scale
            "default",  # base_model_name
            "None",  # refiner_model_name
            0.8,  # refiner_switch
        ]
        
        # Add enough dummy values for loras (3 values per lora * default_max_lora_number)
        from modules.config import default_max_lora_number
        for _ in range(default_max_lora_number):
            args.extend([False, "None", 1.0])
        
        # Add remaining required parameters with dummy values
        remaining_params = [
            False,  # input_image_checkbox
            "uov",  # current_tab
            "disabled",  # uov_method
            None,  # uov_input_image
            "2x",  # latent_upscale_method
            "normal",  # latent_upscale_scheduler
            1024,  # latent_upscale_size
            [],  # outpaint_selections
            None,  # inpaint_input_image
            "",  # inpaint_additional_prompt
            None,  # inpaint_mask_image_upload
            False,  # disable_preview
            False,  # disable_intermediate_results
            False,  # disable_seed_increment
            False,  # black_out_nsfw
            1.5,  # adm_scaler_positive
            0.8,  # adm_scaler_negative
            0.3,  # adm_scaler_end
            7.0,  # adaptive_cfg
            2,  # clip_skip
            "dpmpp_2m_sde_gpu",  # sampler_name
            "karras",  # scheduler_name
            "Default",  # vae_name
            -1,  # overwrite_step
            -1,  # overwrite_switch
            -1,  # overwrite_width
            -1,  # overwrite_height
            -1,  # overwrite_vary_strength
            -1,  # overwrite_upscale_strength
            1,  # upscale_loops
            False,  # mixing_image_prompt_and_vary_upscale
            False,  # mixing_image_prompt_and_inpaint
            False,  # debugging_cn_preprocessor
            False,  # skipping_cn_preprocessor
            64,  # canny_low_threshold
            128,  # canny_high_threshold
            "joint",  # refiner_swap_method
            0.5,  # controlnet_softness
            False,  # freeu_enabled
            1.01,  # freeu_b1
            1.02,  # freeu_b2
            0.99,  # freeu_s1
            0.95,  # freeu_s2
            False,  # debugging_inpaint_preprocessor
            False,  # inpaint_disable_initial_latent
            "v1",  # inpaint_engine
            1.0,  # inpaint_strength
            0.618,  # inpaint_respective_field
            False,  # inpaint_advanced_masking_checkbox
            False,  # invert_mask_checkbox
            0,  # inpaint_erode_or_dilate
            False,  # save_final_enhanced_image_only
            False,  # save_metadata_to_images
            "fooocus",  # metadata_scheme
        ]
        
        # Add detail daemon parameters (12 params)
        detail_params = [
            False,  # detail_daemon_enabled
            0.25,   # detail_daemon_amount
            0.2,    # detail_daemon_start
            0.8,    # detail_daemon_end
            0.71,   # detail_daemon_bias
            0.85,   # detail_daemon_base_multiplier
            0,      # detail_daemon_start_offset
            -0.15,  # detail_daemon_end_offset
            1,      # detail_daemon_exponent
            0,      # detail_daemon_fade
            'both', # detail_daemon_mode
            True,   # detail_daemon_smooth
        ]
        
        # Add TPG parameters (5 params)
        tpg_params = [
            False,  # tpg_enabled
            3.0,    # tpg_scale
            ['mid', 'up'],  # tpg_applied_layers
            1.0,    # tpg_shuffle_strength
            True,   # tpg_adaptive_strength
        ]
        
        # Add NAG parameters (6 params)
        nag_params = [
            False,  # nag_enabled
            1.5,    # nag_scale
            5.0,    # nag_tau
            0.5,    # nag_alpha
            "",     # nag_negative_prompt
            1.0,    # nag_end
        ]
        
        # Add Disco parameters (16 params)
        disco_params = [
            False,  # disco_enabled
            0.5,    # disco_scale
            'custom',  # disco_preset
            None,   # disco_transforms
            None,   # disco_seed
            'none', # disco_animation_mode
            1.02,   # disco_zoom_factor
            0.1,    # disco_rotation_speed
            0.0,    # disco_translation_x
            0.0,    # disco_translation_y
            0.5,    # disco_color_coherence
            1.2,    # disco_saturation_boost
            1.1,    # disco_contrast_boost
            'none', # disco_symmetry_mode
            3,      # disco_fractal_octaves
            'RN50', # disco_clip_model
        ]
        
        args.extend(remaining_params)
        args.extend(detail_params)
        args.extend(tpg_params)
        args.extend(nag_params)
        args.extend(disco_params)
        
        # Create AsyncTask
        task = worker.AsyncTask(args=args)
        
        # Check if force_grid_checkbox was properly set
        if hasattr(task, 'force_grid_checkbox'):
            print(f"✓ AsyncTask.force_grid_checkbox = {task.force_grid_checkbox}")
            return True
        else:
            print("✗ AsyncTask missing force_grid_checkbox attribute")
            return False
            
    except Exception as e:
        print(f"✗ AsyncTask integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all Force Grid integration tests"""
    print("Force Grid Integration Test")
    print("=" * 50)
    
    tests = [
        test_force_grid_imports,
        test_force_grid_interface,
        test_async_task_integration,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 50)
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("✓ All Force Grid integration tests passed!")
        return True
    else:
        print("✗ Some Force Grid integration tests failed!")
        return False

if __name__ == "__main__":
    main()
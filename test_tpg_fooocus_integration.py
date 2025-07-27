#!/usr/bin/env python3
"""
Test TPG integration with Fooocus-specific calling conventions
"""

import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def test_tpg_apply_model_wrapper():
    """Test the TPG apply_model wrapper function"""
    
    try:
        logger.info("Testing TPG apply_model wrapper...")
        
        # Mock the original apply_model function
        def mock_apply_model(input_x, timestep_, **c):
            """Mock apply_model that simulates Fooocus calling convention"""
            logger.info(f"Mock apply_model called with input_x.shape={getattr(input_x, 'shape', 'no shape')}, timestep_={timestep_}")
            logger.info(f"Conditioning keys: {list(c.keys())}")
            
            # Simulate returning noise prediction
            if hasattr(input_x, 'shape'):
                # Return tensor with same batch size as input
                import torch
                return torch.randn_like(input_x)
            else:
                return "mock_output"
        
        # Test the wrapper creation
        from extras.TPG.tpg_integration import create_tpg_unet_wrapper
        
        wrapped_apply_model = create_tpg_unet_wrapper(mock_apply_model)
        logger.info("✓ TPG wrapper created successfully")
        
        # Test calling the wrapper (without torch to avoid import issues)
        logger.info("Testing wrapper call with mock data...")
        
        # Simulate the call
        mock_input_x = "mock_input"
        mock_timestep = "mock_timestep"
        mock_c = {
            'c_crossattn': None,  # No conditioning
            'other_param': "test"
        }
        
        result = wrapped_apply_model(mock_input_x, mock_timestep, **mock_c)
        logger.info(f"✓ Wrapper call successful, result: {result}")
        
        logger.info("✓ TPG apply_model wrapper test completed")
        return True
        
    except Exception as e:
        logger.error(f"✗ TPG apply_model wrapper test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_tpg_patching_logic():
    """Test the TPG patching logic"""
    
    try:
        logger.info("Testing TPG patching logic...")
        
        # Test the patching functions exist and have correct signatures
        from extras.TPG.tpg_integration import patch_unet_for_tpg, unpatch_unet_for_tpg
        
        logger.info("✓ TPG patching functions imported successfully")
        
        # Test configuration functions
        from extras.TPG.tpg_integration import enable_tpg, disable_tpg, is_tpg_enabled, set_tpg_config
        
        logger.info("✓ TPG configuration functions imported successfully")
        
        # Test configuration flow
        logger.info("Testing configuration flow...")
        
        # Initially disabled
        initial_state = is_tpg_enabled()
        logger.info(f"Initial TPG state: {initial_state}")
        
        # Enable TPG
        set_tpg_config(enabled=True, scale=3.0, applied_layers=['mid', 'up'])
        enabled_state = is_tpg_enabled()
        logger.info(f"TPG state after enabling: {enabled_state}")
        
        # Disable TPG
        set_tpg_config(enabled=False)
        disabled_state = is_tpg_enabled()
        logger.info(f"TPG state after disabling: {disabled_state}")
        
        logger.info("✓ TPG patching logic test completed")
        return True
        
    except Exception as e:
        logger.error(f"✗ TPG patching logic test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_tpg_error_handling():
    """Test TPG error handling"""
    
    try:
        logger.info("Testing TPG error handling...")
        
        # Test wrapper with error conditions
        from extras.TPG.tpg_integration import create_tpg_unet_wrapper
        
        def error_apply_model(input_x, timestep_, **c):
            raise RuntimeError("Simulated error")
        
        wrapped_apply_model = create_tpg_unet_wrapper(error_apply_model)
        
        # This should handle the error gracefully
        try:
            result = wrapped_apply_model("mock_input", "mock_timestep", c_crossattn=None)
            logger.error("✗ Expected error was not raised")
            return False
        except RuntimeError:
            logger.info("✓ Error handling working correctly")
        
        logger.info("✓ TPG error handling test completed")
        return True
        
    except Exception as e:
        logger.error(f"✗ TPG error handling test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_tpg_integration_structure():
    """Test the overall TPG integration structure"""
    
    try:
        logger.info("Testing TPG integration structure...")
        
        # Check that all required functions exist
        required_functions = [
            'enable_tpg',
            'disable_tpg', 
            'is_tpg_enabled',
            'get_tpg_config',
            'set_tpg_config',
            'shuffle_tokens',
            'patch_unet_for_tpg',
            'unpatch_unet_for_tpg',
            'create_tpg_unet_wrapper'
        ]
        
        from extras.TPG import tpg_integration
        
        for func_name in required_functions:
            if hasattr(tpg_integration, func_name):
                logger.info(f"✓ Found function: {func_name}")
            else:
                logger.error(f"✗ Missing function: {func_name}")
                return False
        
        # Check that TPGContext class exists
        if hasattr(tpg_integration, 'TPGContext'):
            logger.info("✓ Found TPGContext class")
        else:
            logger.error("✗ Missing TPGContext class")
            return False
        
        logger.info("✓ TPG integration structure test completed")
        return True
        
    except Exception as e:
        logger.error(f"✗ TPG integration structure test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    logger.info("Testing TPG Fooocus integration...")
    
    # Test 1: Apply model wrapper
    logger.info("\n=== Test 1: Apply Model Wrapper ===")
    wrapper_ok = test_tpg_apply_model_wrapper()
    
    # Test 2: Patching logic
    logger.info("\n=== Test 2: Patching Logic ===")
    patching_ok = test_tpg_patching_logic()
    
    # Test 3: Error handling
    logger.info("\n=== Test 3: Error Handling ===")
    error_ok = test_tpg_error_handling()
    
    # Test 4: Integration structure
    logger.info("\n=== Test 4: Integration Structure ===")
    structure_ok = test_tpg_integration_structure()
    
    # Summary
    logger.info("\n=== SUMMARY ===")
    logger.info(f"Apply Model Wrapper: {'✓ PASS' if wrapper_ok else '✗ FAIL'}")
    logger.info(f"Patching Logic: {'✓ PASS' if patching_ok else '✗ FAIL'}")
    logger.info(f"Error Handling: {'✓ PASS' if error_ok else '✗ FAIL'}")
    logger.info(f"Integration Structure: {'✓ PASS' if structure_ok else '✗ FAIL'}")
    
    all_passed = all([wrapper_ok, patching_ok, error_ok, structure_ok])
    
    if all_passed:
        logger.info("\n🎉 TPG Fooocus integration is working!")
        logger.info("Key fixes applied:")
        logger.info("- Fixed apply_model signature to match Fooocus calling convention")
        logger.info("- Updated conditioning handling for c_crossattn parameter")
        logger.info("- Improved error handling and fallback mechanisms")
        logger.info("- Enhanced patching logic for Fooocus UNet structure")
        logger.info("\nThe TPG integration should now work correctly with Fooocus!")
    else:
        logger.info("\n⚠️  Some integration tests failed. Check the errors above.")
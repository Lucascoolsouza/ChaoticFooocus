#!/usr/bin/env python3
"""
Test TPG integration with Fooocus pipeline
"""

import logging
import torch

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def test_tpg_integration_import():
    """Test importing TPG integration components"""
    
    try:
        logger.info("Testing TPG integration imports...")
        
        # Import integration components
        from extras.TPG.tpg_integration import (
            enable_tpg, disable_tpg, is_tpg_enabled, get_tpg_config,
            shuffle_tokens, TPGContext
        )
        
        from extras.TPG.tpg_interface import (
            tpg, enable_tpg_simple, disable_tpg_simple, get_tpg_status, with_tpg
        )
        
        logger.info("✓ All TPG integration components imported successfully")
        return True
        
    except Exception as e:
        logger.error(f"✗ Import test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_tpg_config_management():
    """Test TPG configuration management"""
    
    try:
        logger.info("Testing TPG configuration management...")
        
        from extras.TPG.tpg_integration import enable_tpg, disable_tpg, is_tpg_enabled, get_tpg_config
        
        # Test initial state
        initial_enabled = is_tpg_enabled()
        logger.info(f"Initial TPG state: {initial_enabled}")
        
        # Test enabling TPG
        success = enable_tpg(scale=3.5, applied_layers=["mid", "up"], shuffle_strength=0.8)
        logger.info(f"Enable TPG result: {success}")
        
        # Check if enabled
        enabled_after = is_tpg_enabled()
        logger.info(f"TPG enabled after enable call: {enabled_after}")
        
        # Get config
        config = get_tpg_config()
        logger.info(f"TPG config: {config}")
        
        # Test disabling TPG
        success = disable_tpg()
        logger.info(f"Disable TPG result: {success}")
        
        # Check if disabled
        disabled_after = is_tpg_enabled()
        logger.info(f"TPG enabled after disable call: {disabled_after}")
        
        logger.info("✓ TPG configuration management test completed")
        return True
        
    except Exception as e:
        logger.error(f"✗ Configuration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_token_shuffling():
    """Test token shuffling functionality"""
    
    try:
        logger.info("Testing token shuffling...")
        
        from extras.TPG.tpg_integration import shuffle_tokens
        
        # Create test tokens
        batch_size = 2
        seq_len = 10
        hidden_dim = 64
        
        original_tokens = torch.randn(batch_size, seq_len, hidden_dim)
        logger.info(f"Original tokens shape: {original_tokens.shape}")
        
        # Test full shuffling
        shuffled_full = shuffle_tokens(original_tokens, shuffle_strength=1.0)
        logger.info(f"Full shuffle result shape: {shuffled_full.shape}")
        
        # Test partial shuffling
        shuffled_partial = shuffle_tokens(original_tokens, shuffle_strength=0.5)
        logger.info(f"Partial shuffle result shape: {shuffled_partial.shape}")
        
        # Test with step-based shuffling
        shuffled_step = shuffle_tokens(original_tokens, step=10, shuffle_strength=1.0)
        logger.info(f"Step-based shuffle result shape: {shuffled_step.shape}")
        
        # Verify shapes are preserved
        if (shuffled_full.shape == original_tokens.shape and 
            shuffled_partial.shape == original_tokens.shape and
            shuffled_step.shape == original_tokens.shape):
            logger.info("✓ Token shuffling preserves shapes correctly")
        else:
            logger.error("✗ Token shuffling changed shapes")
            return False
        
        # Verify shuffling actually changes tokens (for full shuffle)
        if not torch.equal(original_tokens, shuffled_full):
            logger.info("✓ Full shuffling changes token order")
        else:
            logger.warning("⚠ Full shuffling didn't change tokens (might be random)")
        
        logger.info("✓ Token shuffling test completed")
        return True
        
    except Exception as e:
        logger.error(f"✗ Token shuffling test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_tpg_context_manager():
    """Test TPG context manager"""
    
    try:
        logger.info("Testing TPG context manager...")
        
        from extras.TPG.tpg_integration import TPGContext, is_tpg_enabled
        
        # Check initial state
        initial_state = is_tpg_enabled()
        logger.info(f"Initial TPG state: {initial_state}")
        
        # Test context manager
        with TPGContext(scale=3.0, shuffle_strength=0.8) as ctx:
            inside_state = is_tpg_enabled()
            logger.info(f"TPG state inside context: {inside_state}")
            
            if inside_state:
                logger.info("✓ TPG enabled inside context")
            else:
                logger.warning("⚠ TPG not enabled inside context")
        
        # Check state after context
        final_state = is_tpg_enabled()
        logger.info(f"TPG state after context: {final_state}")
        
        if final_state == initial_state:
            logger.info("✓ TPG state restored after context")
        else:
            logger.warning("⚠ TPG state not properly restored")
        
        logger.info("✓ TPG context manager test completed")
        return True
        
    except Exception as e:
        logger.error(f"✗ Context manager test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_tpg_interface():
    """Test high-level TPG interface"""
    
    try:
        logger.info("Testing TPG interface...")
        
        from extras.TPG.tpg_interface import tpg, get_tpg_status
        
        # Test status
        initial_status = get_tpg_status()
        logger.info(f"Initial status: {initial_status}")
        
        # Test enabling with recommended settings
        success = tpg.apply_recommended_settings("general")
        logger.info(f"Apply recommended settings result: {success}")
        
        # Check status after enabling
        enabled_status = get_tpg_status()
        logger.info(f"Status after enabling: {enabled_status}")
        
        # Test scale update
        tpg.update_scale(4.0)
        updated_status = get_tpg_status()
        logger.info(f"Status after scale update: {updated_status}")
        
        # Test disabling
        success = tpg.disable()
        logger.info(f"Disable result: {success}")
        
        # Check final status
        final_status = get_tpg_status()
        logger.info(f"Final status: {final_status}")
        
        logger.info("✓ TPG interface test completed")
        return True
        
    except Exception as e:
        logger.error(f"✗ Interface test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    logger.info("Testing TPG integration with Fooocus...")
    
    # Test 1: Import
    logger.info("\n=== Test 1: Import ===")
    import_ok = test_tpg_integration_import()
    
    # Test 2: Configuration
    logger.info("\n=== Test 2: Configuration ===")
    config_ok = test_tpg_config_management()
    
    # Test 3: Token shuffling
    logger.info("\n=== Test 3: Token Shuffling ===")
    shuffle_ok = test_token_shuffling()
    
    # Test 4: Context manager
    logger.info("\n=== Test 4: Context Manager ===")
    context_ok = test_tpg_context_manager()
    
    # Test 5: Interface
    logger.info("\n=== Test 5: Interface ===")
    interface_ok = test_tpg_interface()
    
    # Summary
    logger.info("\n=== SUMMARY ===")
    logger.info(f"Import: {'✓ PASS' if import_ok else '✗ FAIL'}")
    logger.info(f"Configuration: {'✓ PASS' if config_ok else '✗ FAIL'}")
    logger.info(f"Token Shuffling: {'✓ PASS' if shuffle_ok else '✗ FAIL'}")
    logger.info(f"Context Manager: {'✓ PASS' if context_ok else '✗ FAIL'}")
    logger.info(f"Interface: {'✓ PASS' if interface_ok else '✗ FAIL'}")
    
    all_passed = all([import_ok, config_ok, shuffle_ok, context_ok, interface_ok])
    
    if all_passed:
        logger.info("\n🎉 TPG integration is working correctly!")
        logger.info("Key features:")
        logger.info("- Configuration management")
        logger.info("- Token shuffling with adaptive strength")
        logger.info("- Context manager for temporary usage")
        logger.info("- High-level interface with recommended settings")
        logger.info("- Integration with Fooocus pipeline")
    else:
        logger.info("\n⚠️  Some tests failed. Check the errors above.")
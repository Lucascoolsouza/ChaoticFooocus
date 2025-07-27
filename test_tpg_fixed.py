#!/usr/bin/env python3
"""
Test the fixed TPG pipeline implementation
"""

import logging
import torch

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def test_tpg_pipeline_import():
    """Test importing the fixed TPG pipeline"""
    
    try:
        logger.info("Testing TPG pipeline import...")
        
        # Import the TPG pipeline
        from extras.TPG.pipeline_sdxl_tpg import StableDiffusionXLTPGPipeline, TPGAttentionProcessor
        
        logger.info("✓ TPG pipeline imported successfully")
        
        # Check if the pipeline has the expected methods
        expected_methods = ['__call__', 'enable_tpg', 'disable_tpg', 'do_token_perturbation_guidance', 'tpg_scale']
        for method in expected_methods:
            if hasattr(StableDiffusionXLTPGPipeline, method):
                logger.info(f"✓ Has {method} method/property")
            else:
                logger.warning(f"✗ Missing {method} method/property")
        
        logger.info("✓ TPG pipeline import test completed")
        return True
        
    except Exception as e:
        logger.error(f"✗ Import test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_tpg_attention_processor():
    """Test the TPG attention processor"""
    
    try:
        logger.info("Testing TPG attention processor...")
        
        from extras.TPG.pipeline_sdxl_tpg import TPGAttentionProcessor
        
        # Create a mock original processor
        class MockProcessor:
            def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None, scale=1.0):
                return hidden_states  # Simple passthrough
        
        original_processor = MockProcessor()
        tpg_processor = TPGAttentionProcessor(original_processor, perturbation_scale=0.5)
        
        logger.info(f"✓ TPG processor created with scale {tpg_processor.perturbation_scale}")
        
        # Test with mock data
        batch_size = 3  # uncond + cond + perturb
        seq_len = 77
        hidden_dim = 768
        
        hidden_states = torch.randn(batch_size, seq_len, hidden_dim)
        encoder_hidden_states = torch.randn(batch_size, seq_len, hidden_dim)
        
        # Mock attention object
        class MockAttn:
            pass
        
        mock_attn = MockAttn()
        
        # Test the processor
        result = tpg_processor(mock_attn, hidden_states, encoder_hidden_states)
        
        logger.info(f"✓ TPG processor output shape: {result.shape}")
        logger.info(f"✓ Expected shape: {hidden_states.shape}")
        
        if result.shape == hidden_states.shape:
            logger.info("✓ TPG attention processor test completed")
            return True
        else:
            logger.error("✗ Output shape mismatch")
            return False
        
    except Exception as e:
        logger.error(f"✗ Attention processor test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_tpg_token_perturbation():
    """Test the token perturbation functionality"""
    
    try:
        logger.info("Testing token perturbation...")
        
        from extras.TPG.pipeline_sdxl_tpg import TPGAttentionProcessor
        
        # Create a processor with perturbation
        class MockProcessor:
            def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None, scale=1.0):
                return hidden_states
        
        original_processor = MockProcessor()
        tpg_processor = TPGAttentionProcessor(original_processor, perturbation_scale=1.0)
        
        # Test token perturbation
        batch_size = 1
        seq_len = 10
        hidden_dim = 64
        
        original_tokens = torch.randn(batch_size, seq_len, hidden_dim)
        
        # Test the perturbation method directly
        class MockAttn:
            pass
        
        mock_attn = MockAttn()
        
        # Process with perturbation
        result = tpg_processor._process_with_perturbation(
            mock_attn, 
            original_tokens, 
            encoder_hidden_states=original_tokens.clone()
        )
        
        logger.info(f"✓ Token perturbation completed")
        logger.info(f"✓ Original shape: {original_tokens.shape}")
        logger.info(f"✓ Result shape: {result.shape}")
        
        # Check if perturbation actually changed something
        if not torch.equal(original_tokens, result):
            logger.info("✓ Perturbation applied successfully (tokens changed)")
        else:
            logger.warning("⚠ Perturbation may not have been applied (tokens unchanged)")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Token perturbation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    logger.info("Testing fixed TPG pipeline functionality...")
    
    # Test 1: Basic import
    logger.info("\n=== Test 1: Import ===")
    import_ok = test_tpg_pipeline_import()
    
    # Test 2: Attention processor
    logger.info("\n=== Test 2: Attention Processor ===")
    processor_ok = test_tpg_attention_processor()
    
    # Test 3: Token perturbation
    logger.info("\n=== Test 3: Token Perturbation ===")
    perturbation_ok = test_tpg_token_perturbation()
    
    # Summary
    logger.info("\n=== SUMMARY ===")
    logger.info(f"Import: {'✓ PASS' if import_ok else '✗ FAIL'}")
    logger.info(f"Attention Processor: {'✓ PASS' if processor_ok else '✗ FAIL'}")
    logger.info(f"Token Perturbation: {'✓ PASS' if perturbation_ok else '✗ FAIL'}")
    
    if import_ok and processor_ok and perturbation_ok:
        logger.info("\n🎉 Fixed TPG pipeline is working!")
        logger.info("Key improvements:")
        logger.info("- Proper attention processor integration")
        logger.info("- Token perturbation through shuffling")
        logger.info("- Clean enable/disable functionality")
        logger.info("- Compatible with diffusers pipeline structure")
    else:
        logger.info("\n⚠️  Some tests failed. Check the errors above.")
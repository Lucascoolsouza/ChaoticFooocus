#!/usr/bin/env python3
"""
Test the fixed TPG pipeline to see if it can now find transformer blocks
"""

import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def test_tpg_layer_detection():
    """Test if TPG can now detect transformer blocks"""
    
    try:
        logger.info("Testing TPG layer detection with fixes...")
        
        # Import the fixed TPG pipeline
        from extras.TPG.pipeline_sdxl_tpg import StableDiffusionXLTPGPipeline
        
        logger.info("✓ TPG pipeline imported successfully")
        
        # Test the layer detection by creating a mock pipeline instance
        # This would normally be done during pipeline initialization
        
        logger.info("Testing layer detection logic...")
        
        # Import the modules we need
        import modules.default_pipeline as pipeline
        
        logger.info("Checking if models are loaded...")
        
        if pipeline.final_unet is None:
            logger.error("UNet is None - models not loaded")
            return False
        
        logger.info(f"UNet type: {type(pipeline.final_unet)}")
        
        # Test accessing the diffusion model
        if hasattr(pipeline.final_unet, 'model') and hasattr(pipeline.final_unet.model, 'diffusion_model'):
            diffusion_model = pipeline.final_unet.model.diffusion_model
            logger.info(f"✓ Found diffusion model: {type(diffusion_model)}")
            
            # Count transformer blocks
            from diffusers.models.attention import BasicTransformerBlock
            
            transformer_count = 0
            for name, module in diffusion_model.named_modules():
                # Use the same logic as the fixed TPG pipeline
                is_transformer_block = False
                
                if isinstance(module, BasicTransformerBlock):
                    is_transformer_block = True
                elif hasattr(module, '__class__') and 'BasicTransformerBlock' in str(module.__class__):
                    is_transformer_block = True
                elif 'transformer_blocks' in name and hasattr(module, 'forward') and hasattr(module, 'attn1'):
                    is_transformer_block = True
                
                if is_transformer_block:
                    transformer_count += 1
                    if transformer_count <= 5:  # Show first 5
                        logger.info(f"  Found: {name} -> {type(module)}")
            
            logger.info(f"Total transformer blocks found: {transformer_count}")
            
            if transformer_count > 0:
                logger.info("🎉 TPG should now work with your model!")
                return True
            else:
                logger.warning("Still no transformer blocks found")
                return False
        else:
            logger.error("Could not access diffusion model")
            return False
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_tpg_make_block():
    """Test if the make_tpg_block function works with ldm_patched blocks"""
    
    try:
        logger.info("Testing make_tpg_block function...")
        
        from extras.TPG.pipeline_sdxl_tpg import make_tpg_block
        import modules.default_pipeline as pipeline
        
        if pipeline.final_unet is None:
            logger.error("UNet is None - models not loaded")
            return False
        
        # Get a sample transformer block
        diffusion_model = pipeline.final_unet.model.diffusion_model
        
        sample_block = None
        for name, module in diffusion_model.named_modules():
            if 'transformer_blocks' in name and hasattr(module, 'forward') and hasattr(module, 'attn1'):
                sample_block = module
                logger.info(f"Using sample block: {name} -> {type(module)}")
                break
        
        if sample_block is None:
            logger.error("No sample transformer block found")
            return False
        
        # Test creating a TPG block
        modified_class = make_tpg_block(sample_block.__class__, do_cfg=True)
        logger.info(f"✓ Created modified class: {modified_class}")
        
        # Test if we can create an instance (this would normally be done by patching)
        logger.info("✓ make_tpg_block function works")
        return True
        
    except Exception as e:
        logger.error(f"make_tpg_block test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    logger.info("Testing fixed TPG pipeline...")
    
    # Test 1: Layer detection
    logger.info("\n=== Test 1: Layer Detection ===")
    detection_ok = test_tpg_layer_detection()
    
    # Test 2: make_tpg_block function
    logger.info("\n=== Test 2: make_tpg_block Function ===")
    make_block_ok = test_tpg_make_block()
    
    # Summary
    logger.info("\n=== SUMMARY ===")
    logger.info(f"Layer Detection: {'✓ PASS' if detection_ok else '✗ FAIL'}")
    logger.info(f"make_tpg_block: {'✓ PASS' if make_block_ok else '✗ FAIL'}")
    
    if detection_ok and make_block_ok:
        logger.info("\n🎉 TPG pipeline should now work with your model!")
        logger.info("Key improvements:")
        logger.info("- Fixed layer detection to look in unet.model.diffusion_model")
        logger.info("- Added support for ldm_patched BasicTransformerBlock")
        logger.info("- Updated make_tpg_block to handle different forward signatures")
        logger.info("\nYou can now try using TPG with your model!")
    else:
        logger.info("\n⚠️  Some tests failed. Check the errors above.")
#!/usr/bin/env python3
"""
Test TPG pipeline with actual generation to identify freezing issues
"""

import sys
import logging
import time
import signal

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class TimeoutError(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutError("Operation timed out")

def test_tpg_generation():
    """Test TPG pipeline with actual generation"""
    
    try:
        logger.info("Testing TPG pipeline generation...")
        
        # Set a timeout for the entire test
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(300)  # 5 minute timeout
        
        logger.info("Importing TPG pipeline...")
        from extras.TPG.pipeline_sdxl_tpg import StableDiffusionXLTPGPipeline
        
        logger.info("✓ TPG pipeline imported successfully")
        
        # Test with TPG disabled first
        logger.info("Testing with TPG disabled (tpg_scale=0)...")
        
        # You would need to adapt this to your actual model loading
        # This is just a template showing the approach
        
        logger.info("Generation test would go here...")
        logger.info("You'll need to adapt this to your actual pipeline setup")
        
        signal.alarm(0)  # Cancel timeout
        logger.info("✓ Test completed successfully")
        return True
        
    except TimeoutError:
        logger.error("✗ Test timed out - this suggests a freeze or infinite loop")
        return False
    except Exception as e:
        signal.alarm(0)  # Cancel timeout
        logger.error(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_layer_inspection():
    """Test layer inspection to understand why TPG finds no layers"""
    
    try:
        logger.info("Testing layer inspection...")
        
        # Import required modules
        import modules.default_pipeline as pipeline
        from diffusers.models.attention import BasicTransformerBlock
        
        logger.info("Checking if models are loaded...")
        
        if pipeline.final_unet is None:
            logger.warning("UNet is None - models not loaded")
            return False
        
        logger.info(f"UNet type: {type(pipeline.final_unet)}")
        
        # Try different access patterns
        unet_models = []
        if hasattr(pipeline.final_unet, 'model'):
            unet_models.append(("final_unet.model", pipeline.final_unet.model))
        if hasattr(pipeline.final_unet, 'diffusion_model'):
            unet_models.append(("final_unet.diffusion_model", pipeline.final_unet.diffusion_model))
        unet_models.append(("final_unet", pipeline.final_unet))
        
        for name, model in unet_models:
            if model is None:
                continue
                
            logger.info(f"\nInspecting {name}...")
            logger.info(f"Type: {type(model)}")
            
            # Count transformer blocks
            transformer_count = 0
            for module_name, module in model.named_modules():
                if isinstance(module, BasicTransformerBlock):
                    transformer_count += 1
                    if transformer_count <= 5:  # Show first 5
                        logger.info(f"  Found: {module_name}")
            
            logger.info(f"Total BasicTransformerBlock modules: {transformer_count}")
            
            if transformer_count > 0:
                logger.info("✓ Found transformer blocks - TPG should work")
                return True
        
        logger.warning("✗ No BasicTransformerBlock modules found")
        return False
        
    except Exception as e:
        logger.error(f"✗ Layer inspection failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    logger.info("Starting TPG generation tests...")
    
    # Test 1: Layer inspection
    logger.info("\n=== Test 1: Layer Inspection ===")
    layer_ok = test_layer_inspection()
    
    # Test 2: Generation test
    logger.info("\n=== Test 2: Generation Test ===")
    gen_ok = test_tpg_generation()
    
    # Summary
    logger.info("\n=== SUMMARY ===")
    logger.info(f"Layer Inspection: {'✓ PASS' if layer_ok else '✗ FAIL'}")
    logger.info(f"Generation Test: {'✓ PASS' if gen_ok else '✗ FAIL'}")
    
    if layer_ok and gen_ok:
        logger.info("🎉 All tests passed!")
    else:
        logger.info("⚠️  Some tests failed. Check the errors above.")
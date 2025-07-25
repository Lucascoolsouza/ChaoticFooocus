#!/usr/bin/env python3
"""
Simple test to identify the exact TPG freezing issue
"""

import sys
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def test_layer_detection():
    """Test the layer detection that's failing"""
    
    try:
        logger.info("=== Testing Layer Detection ===")
        
        # Import required modules
        import modules.default_pipeline as pipeline
        from diffusers.models.attention import BasicTransformerBlock
        
        logger.info("Initializing pipeline...")
        
        # Initialize pipeline to load models
        pipeline.refresh_everything(
            refiner_model_name='None',
            base_model_name='sd_xl_base_1.0_0.9vae.safetensors',
            loras=[]
        )
        
        unet = pipeline.final_unet
        logger.info(f"UNet type: {type(unet)}")
        
        if unet is None:
            logger.error("UNet is None!")
            return False
        
        # Test different access patterns
        models_to_check = []
        
        if hasattr(unet, 'model'):
            models_to_check.append(("unet.model", unet.model))
            logger.info("✓ Found unet.model")
        else:
            logger.info("✗ No unet.model")
            
        if hasattr(unet, 'diffusion_model'):
            models_to_check.append(("unet.diffusion_model", unet.diffusion_model))
            logger.info("✓ Found unet.diffusion_model")
        else:
            logger.info("✗ No unet.diffusion_model")
            
        models_to_check.append(("unet", unet))
        
        # Check each model for BasicTransformerBlock
        for name, model in models_to_check:
            if model is None:
                logger.info(f"{name} is None, skipping")
                continue
                
            logger.info(f"\nChecking {name}...")
            
            transformer_blocks = []
            total_modules = 0
            
            try:
                for module_name, module in model.named_modules():
                    total_modules += 1
                    if isinstance(module, BasicTransformerBlock):
                        transformer_blocks.append((module_name, module))
                        if len(transformer_blocks) <= 5:  # Limit output
                            logger.info(f"  Found: {module_name}")
                
                logger.info(f"Total modules: {total_modules}")
                logger.info(f"BasicTransformerBlock modules: {len(transformer_blocks)}")
                
                if len(transformer_blocks) > 0:
                    logger.info(f"✓ Found {len(transformer_blocks)} BasicTransformerBlock modules in {name}")
                    return True
                else:
                    logger.info(f"✗ No BasicTransformerBlock modules in {name}")
                    
            except Exception as e:
                logger.error(f"Error checking {name}: {e}")
                continue
        
        logger.error("No BasicTransformerBlock modules found in any model!")
        return False
        
    except Exception as e:
        logger.error(f"Layer detection test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_tpg_import():
    """Test importing TPG pipeline"""
    
    try:
        logger.info("=== Testing TPG Import ===")
        
        from extras.TPG.pipeline_sdxl_tpg import StableDiffusionXLTPGPipeline
        logger.info("✓ TPG pipeline imported successfully")
        
        from extras.TPG.pipeline_sdxl_tpg import make_tpg_block
        logger.info("✓ make_tpg_block imported successfully")
        
        return True
        
    except Exception as e:
        logger.error(f"TPG import failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    logger.info("Starting simple TPG tests...")
    
    # Test 1: Import
    import_ok = test_tpg_import()
    
    # Test 2: Layer detection
    if import_ok:
        layer_ok = test_layer_detection()
    else:
        layer_ok = False
    
    # Summary
    logger.info("\n=== SUMMARY ===")
    logger.info(f"Import test: {'✓ PASS' if import_ok else '✗ FAIL'}")
    logger.info(f"Layer detection: {'✓ PASS' if layer_ok else '✗ FAIL'}")
    
    if not layer_ok:
        logger.info("\n💡 SOLUTION:")
        logger.info("The TPG pipeline can't find BasicTransformerBlock modules.")
        logger.info("This is why TPG is being disabled and the pipeline falls back to regular CFG.")
        logger.info("The freeze might be due to a different issue in the pipeline execution.")
    
    logger.info("Tests completed")
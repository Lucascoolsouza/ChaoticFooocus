#!/usr/bin/env python3
"""
Minimal test to isolate TPG freezing issue
"""

import torch
import logging
import sys

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def test_tpg_bypass():
    """Test TPG pipeline with TPG completely disabled"""
    
    try:
        logger.info("Testing TPG pipeline with TPG disabled...")
        
        # Import the pipeline
        from extras.TPG.pipeline_sdxl_tpg import StableDiffusionXLTPGPipeline
        
        logger.info("Pipeline imported successfully")
        
        # Test basic properties
        logger.info("Testing basic pipeline properties...")
        
        # You would load your actual pipeline here
        # pipe = StableDiffusionXLTPGPipeline.from_pretrained(...)
        
        logger.info("Basic test completed")
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()

def test_layer_detection():
    """Test just the layer detection part"""
    
    try:
        logger.info("Testing layer detection...")
        
        # This would test the layer detection logic in isolation
        # You'd need to pass your actual UNet model here
        
        logger.info("Layer detection test completed")
        
    except Exception as e:
        logger.error(f"Layer detection test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    logger.info("Starting minimal TPG tests...")
    test_tpg_bypass()
    test_layer_detection()
    logger.info("Tests completed")
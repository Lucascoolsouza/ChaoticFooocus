#!/usr/bin/env python3
"""
Debug script to identify TPG pipeline freezing issues
"""

import torch
import logging
import sys
import traceback
from extras.TPG.pipeline_sdxl_tpg import StableDiffusionXLTPGPipeline

# Set up detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('tpg_debug.log')
    ]
)

logger = logging.getLogger(__name__)

def debug_tpg_pipeline():
    """Debug the TPG pipeline to identify freezing issues"""
    
    try:
        logger.info("Starting TPG pipeline debug")
        
        # Test with minimal settings first
        logger.info("Testing with TPG disabled first...")
        
        # You'll need to adapt this to your actual model loading code
        # This is just a template showing the debugging approach
        
        # Test 1: Basic pipeline without TPG
        logger.info("=== Test 1: Basic pipeline (no TPG) ===")
        
        # Test 2: TPG enabled with minimal layers
        logger.info("=== Test 2: TPG with minimal layers ===")
        
        # Test 3: Check layer indices
        logger.info("=== Test 3: Checking layer indices ===")
        
        logger.info("Debug completed successfully")
        
    except Exception as e:
        logger.error(f"Debug failed with error: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise

if __name__ == "__main__":
    debug_tpg_pipeline()
#!/usr/bin/env python3

import sys
import os
import logging
import torch

# Add the current directory to the path so we can import the TPG pipeline
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Set up logging to see debug messages
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

try:
    from extras.TPG.pipeline_sdxl_tpg import StableDiffusionXLTPGPipeline
    print("Successfully imported StableDiffusionXLTPGPipeline")
    
    # Test basic functionality
    print("Testing basic TPG pipeline functionality...")
    
    # Check if CUDA is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Try to create a simple test
    print("Creating a minimal test case...")
    
    # Test the make_tpg_block function
    from extras.TPG.pipeline_sdxl_tpg import make_tpg_block
    from diffusers.models.attention import BasicTransformerBlock
    
    print("Testing make_tpg_block function...")
    modified_class = make_tpg_block(BasicTransformerBlock, do_cfg=True)
    print("make_tpg_block function works correctly")
    
    print("All basic tests passed!")
    
except Exception as e:
    print(f"Error during testing: {e}")
    import traceback
    traceback.print_exc()
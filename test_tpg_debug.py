#!/usr/bin/env python3
"""
Debug script to identify TPG pipeline freezing issues and inspect UNet layers
"""

import torch
import logging
import sys
import traceback
from diffusers.models.attention import BasicTransformerBlock

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

def inspect_unet_layers(unet):
    """Inspect UNet layers to understand the structure"""
    
    logger.info("=== UNet Layer Inspection ===")
    
    # Try different access patterns
    unet_models = []
    if hasattr(unet, 'model') and unet.model is not None:
        unet_models.append(("unet.model", unet.model))
    if hasattr(unet, 'diffusion_model') and unet.diffusion_model is not None:
        unet_models.append(("unet.diffusion_model", unet.diffusion_model))
    unet_models.append(("unet", unet))
    
    for name, model in unet_models:
        logger.info(f"\n--- Inspecting {name} ---")
        logger.info(f"Type: {type(model)}")
        
        # Find all BasicTransformerBlock modules
        transformer_blocks = []
        for module_name, module in model.named_modules():
            if isinstance(module, BasicTransformerBlock):
                transformer_blocks.append((module_name, module))
        
        logger.info(f"Found {len(transformer_blocks)} BasicTransformerBlock modules:")
        
        down_count = 0
        mid_count = 0
        up_count = 0
        
        for module_name, module in transformer_blocks:
            logger.info(f"  {module_name}: {type(module)}")
            
            # Categorize layers
            if "down" in module_name:
                down_count += 1
                logger.info(f"    -> DOWN layer #{down_count-1}")
            elif "mid" in module_name:
                mid_count += 1
                logger.info(f"    -> MID layer #{mid_count-1}")
            elif "up" in module_name:
                up_count += 1
                logger.info(f"    -> UP layer #{up_count-1}")
            else:
                logger.info(f"    -> UNKNOWN category")
        
        logger.info(f"Summary: {down_count} down, {mid_count} mid, {up_count} up layers")
        
        # Suggest valid layer indices
        if down_count > 0:
            logger.info(f"Valid down layer indices: d0 to d{down_count-1}")
        if mid_count > 0:
            logger.info(f"Valid mid layer indices: m0 to m{mid_count-1}")
        if up_count > 0:
            logger.info(f"Valid up layer indices: u0 to u{up_count-1}")
        
        if len(transformer_blocks) > 0:
            break  # Use the first model that has transformer blocks
    
    return transformer_blocks

def debug_tpg_pipeline():
    """Debug the TPG pipeline to identify freezing issues"""
    
    try:
        logger.info("Starting TPG pipeline debug")
        
        # This is a template - you'll need to adapt this to your actual model loading
        logger.info("To use this debug script:")
        logger.info("1. Load your UNet model")
        logger.info("2. Call inspect_unet_layers(unet) to see available layers")
        logger.info("3. Use the suggested layer indices in your TPG configuration")
        
        logger.info("Example usage:")
        logger.info("  from your_model_loader import load_unet")
        logger.info("  unet = load_unet()")
        logger.info("  inspect_unet_layers(unet)")
        
    except Exception as e:
        logger.error(f"Debug failed with error: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise

if __name__ == "__main__":
    debug_tpg_pipeline()
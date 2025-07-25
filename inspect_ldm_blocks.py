#!/usr/bin/env python3
"""
Inspect the actual transformer block types in ldm_patched models
"""

import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def inspect_transformer_blocks():
    """Inspect what transformer blocks are actually available"""
    
    try:
        import modules.default_pipeline as pipeline
        
        logger.info("Inspecting transformer block types...")
        
        unet = pipeline.final_unet
        if unet is None:
            logger.error("UNet is None")
            return
        
        # Access the actual diffusion model
        if hasattr(unet, 'model') and hasattr(unet.model, 'diffusion_model'):
            diffusion_model = unet.model.diffusion_model
        elif hasattr(unet, 'model'):
            diffusion_model = unet.model
        else:
            diffusion_model = unet
        
        logger.info(f"Diffusion model type: {type(diffusion_model)}")
        
        # Look for transformer blocks
        transformer_blocks = []
        for name, module in diffusion_model.named_modules():
            if 'transformer_blocks' in name and not name.endswith('transformer_blocks'):
                # This is an individual transformer block
                transformer_blocks.append((name, type(module).__name__, module))
                if len(transformer_blocks) <= 5:  # Show details for first 5
                    logger.info(f"Found transformer block: {name} -> {type(module)}")
                    logger.info(f"  Module attributes: {[attr for attr in dir(module) if not attr.startswith('_')][:10]}")
        
        logger.info(f"Total transformer blocks found: {len(transformer_blocks)}")
        
        if transformer_blocks:
            # Analyze the first transformer block
            first_name, first_type, first_module = transformer_blocks[0]
            logger.info(f"\nAnalyzing first transformer block: {first_name}")
            logger.info(f"Type: {first_type}")
            logger.info(f"Module: {type(first_module)}")
            
            # Check if it has a forward method we can modify
            if hasattr(first_module, 'forward'):
                logger.info("✓ Has forward method - can potentially be modified for TPG")
                
                # Check the forward method signature
                import inspect
                sig = inspect.signature(first_module.forward)
                logger.info(f"Forward signature: {sig}")
            else:
                logger.info("✗ No forward method found")
        
        return transformer_blocks
        
    except Exception as e:
        logger.error(f"Error inspecting transformer blocks: {e}")
        import traceback
        traceback.print_exc()
        return []

if __name__ == "__main__":
    blocks = inspect_transformer_blocks()
    
    if blocks:
        logger.info(f"\n🎉 Found {len(blocks)} transformer blocks that could potentially work with TPG!")
        logger.info("These blocks could be modified to support token perturbation guidance.")
    else:
        logger.info("\n❌ No suitable transformer blocks found for TPG.")
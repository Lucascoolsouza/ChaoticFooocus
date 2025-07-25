#!/usr/bin/env python3
"""
Debug script to identify TPG pipeline freezing issues in ChaoticFooocus
"""

import sys
import os
import torch
import logging
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Set up detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def debug_unet_structure():
    """Debug the UNet structure to understand why TPG can't find layers"""
    
    try:
        # Import your modules
        import modules.default_pipeline as pipeline
        from diffusers.models.attention import BasicTransformerBlock
        
        # Also try to import ComfyUI attention modules
        try:
            from ldm_patched.ldm.modules.attention import CrossAttention
            COMFYUI_AVAILABLE = True
        except ImportError:
            CrossAttention = None
            COMFYUI_AVAILABLE = False
        
        logger.info("Accessing already loaded UNet model...")
        
        # The models should already be loaded by default_pipeline
        unet = pipeline.final_unet
        
        if unet is None:
            logger.error("UNet is None - models not loaded properly")
            return
        
        logger.info(f"UNet type: {type(unet)}")
        logger.info(f"UNet attributes: {dir(unet)}")
        
        # Try different access patterns
        models_to_check = []
        
        if hasattr(unet, 'model'):
            models_to_check.append(("unet.model", unet.model))
        if hasattr(unet, 'diffusion_model'):
            models_to_check.append(("unet.diffusion_model", unet.diffusion_model))
        models_to_check.append(("unet", unet))
        
        for name, model in models_to_check:
            logger.info(f"\n=== Checking {name} ===")
            logger.info(f"Type: {type(model)}")
            
            if model is None:
                logger.info("Model is None, skipping")
                continue
            
            # Count all modules
            total_modules = 0
            transformer_blocks = []
            
            try:
                for module_name, module in model.named_modules():
                    total_modules += 1
                    if isinstance(module, BasicTransformerBlock):
                        transformer_blocks.append((module_name, module))
                        
                logger.info(f"Total modules: {total_modules}")
                logger.info(f"BasicTransformerBlock modules found: {len(transformer_blocks)}")
                
                if transformer_blocks:
                    logger.info("BasicTransformerBlock locations:")
                    down_count = mid_count = up_count = 0
                    
                    for module_name, module in transformer_blocks:
                        logger.info(f"  {module_name}")
                        
                        if "down" in module_name.lower():
                            logger.info(f"    -> DOWN layer d{down_count}")
                            down_count += 1
                        elif "mid" in module_name.lower():
                            logger.info(f"    -> MID layer m{mid_count}")
                            mid_count += 1
                        elif "up" in module_name.lower():
                            logger.info(f"    -> UP layer u{up_count}")
                            up_count += 1
                    
                    logger.info(f"\nSummary: {down_count} down, {mid_count} mid, {up_count} up")
                    
                    if down_count > 0:
                        logger.info(f"Valid down indices: d0 to d{down_count-1}")
                    if mid_count > 0:
                        logger.info(f"Valid mid indices: m0 to m{mid_count-1}")
                    if up_count > 0:
                        logger.info(f"Valid up indices: u0 to u{up_count-1}")
                    
                    break  # Found transformer blocks, no need to check other models
                else:
                    logger.info("No BasicTransformerBlock modules found in this model")
                    
            except Exception as e:
                logger.error(f"Error inspecting {name}: {e}")
                continue
        
        if not any(len(transformer_blocks) > 0 for _, model in models_to_check if model is not None):
            logger.warning("No BasicTransformerBlock modules found in any model variant!")
            logger.info("This explains why TPG is being disabled.")
            
            # Let's see what attention modules we do have
            logger.info("\nLooking for other attention-related modules...")
            for name, model in models_to_check:
                if model is None:
                    continue
                    
                attention_modules = []
                for module_name, module in model.named_modules():
                    if "attention" in module_name.lower() or "attn" in module_name.lower():
                        attention_modules.append((module_name, type(module).__name__))
                
                if attention_modules:
                    logger.info(f"\nAttention modules in {name}:")
                    for module_name, module_type in attention_modules[:10]:  # Limit output
                        logger.info(f"  {module_name}: {module_type}")
                    if len(attention_modules) > 10:
                        logger.info(f"  ... and {len(attention_modules) - 10} more")
        
    except ImportError as e:
        logger.error(f"Import error: {e}")
        logger.info("Make sure you're running this from the ChaoticFooocus directory")
    except Exception as e:
        logger.error(f"Error during debugging: {e}")
        import traceback
        logger.error(traceback.format_exc())

def test_basic_pipeline():
    """Test if the basic pipeline works without TPG"""
    
    try:
        logger.info("\n=== Testing Basic Pipeline (No TPG) ===")
        
        # Import your pipeline
        import modules.default_pipeline as pipeline
        
        # Test basic pipeline access
        logger.info("Testing pipeline access...")
        
        logger.info(f"Pipeline final_unet: {type(pipeline.final_unet)}")
        logger.info(f"Pipeline final_vae: {type(pipeline.final_vae)}")
        logger.info(f"Pipeline final_clip: {type(pipeline.final_clip)}")
        
        if pipeline.final_unet is not None:
            logger.info("✓ UNet is loaded and accessible")
        else:
            logger.warning("⚠️  UNet is None")
        
        logger.info("Basic pipeline test completed successfully")
        
    except Exception as e:
        logger.error(f"Error testing basic pipeline: {e}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    logger.info("Starting ChaoticFooocus TPG Debug")
    
    debug_unet_structure()
    test_basic_pipeline()
    
    logger.info("Debug completed")
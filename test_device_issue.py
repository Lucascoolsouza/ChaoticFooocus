#!/usr/bin/env python3
"""
Test to understand the device issue with TPG layer modification
"""

import sys
import os
import torch
import logging

# Add the project root to the path
sys.path.insert(0, '/content/ChaoticFooocus')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_device_preservation():
    """Test if device is preserved during layer modification"""
    logger.info("Testing device preservation during layer modification...")
    
    try:
        from extras.TPG.pipeline_sdxl_tpg import make_tpg_block
        import torch.nn as nn
        
        # Create a test transformer block on GPU
        class TestTransformerBlock(nn.Module):
            def __init__(self):
                super().__init__()
                self.norm1 = nn.LayerNorm(768)
                self.attn = nn.MultiheadAttention(768, 8)
                self.norm2 = nn.LayerNorm(768)
                self.mlp = nn.Sequential(
                    nn.Linear(768, 3072),
                    nn.GELU(),
                    nn.Linear(3072, 768)
                )
            
            def forward(self, x, context=None, transformer_options={}):
                return x
        
        # Make it look like ldm_patched block
        TestTransformerBlock.__module__ = 'ldm_patched.modules.attention'
        TestTransformerBlock.__name__ = 'BasicTransformerBlock'
        
        # Create original block and move to GPU
        original_block = TestTransformerBlock()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        original_block.to(device)
        
        logger.info(f"Original block device: {next(original_block.parameters()).device}")
        
        # Test the current TPG modification approach
        logger.info("Testing current TPG approach...")
        modified_class = make_tpg_block(TestTransformerBlock, do_cfg=True)
        
        # Current approach
        modified_instance = modified_class.__new__(modified_class)
        modified_instance.__dict__.update(original_block.__dict__)
        
        logger.info(f"Modified instance device before .to(): {next(modified_instance.parameters()).device}")
        
        # Try to move to device
        modified_instance.to(device)
        
        logger.info(f"Modified instance device after .to(): {next(modified_instance.parameters()).device}")
        
        # Check all submodules
        for name, module in modified_instance.named_modules():
            if hasattr(module, 'weight') and module.weight is not None:
                logger.info(f"  {name}: {module.weight.device}")
        
        # Test alternative approach: proper initialization
        logger.info("\nTesting alternative approach...")
        
        # Alternative: Initialize properly then copy state
        alt_modified = modified_class()  # Proper initialization
        alt_modified.load_state_dict(original_block.state_dict())  # Copy weights
        alt_modified.to(device)  # Move to device
        
        logger.info(f"Alternative approach device: {next(alt_modified.parameters()).device}")
        
        # Check all submodules
        for name, module in alt_modified.named_modules():
            if hasattr(module, 'weight') and module.weight is not None:
                logger.info(f"  {name}: {module.weight.device}")
        
        return True
        
    except Exception as e:
        logger.error(f"Device test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    logger.info("Starting Device Issue Test")
    logger.info("=" * 50)
    
    if test_device_preservation():
        logger.info("✓ Device test completed")
    else:
        logger.error("✗ Device test failed")

if __name__ == "__main__":
    main()
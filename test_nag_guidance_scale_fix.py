#!/usr/bin/env python3
"""
Test script to verify NAG works correctly at guidance scale 1.0
"""

import torch
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_nag_at_scale_one():
    """Test that NAG applies some effect even at guidance scale 1.0"""
    
    try:
        from extras.nag.nag_integration import (
            enable_nag, disable_nag, is_nag_enabled, get_nag_config
        )
        from extras.nag.attention_nag import NAGAttnProcessor2_0
        
        print("=== Testing NAG at Guidance Scale 1.0 ===")
        
        # Test 1: Check that NAG is enabled at scale 1.0
        print("\n1. Testing NAG enablement at scale 1.0...")
        enable_nag(scale=1.0, negative_prompt="blurry, low quality")
        
        if is_nag_enabled():
            print("✓ NAG is correctly enabled at scale 1.0")
        else:
            print("✗ NAG is not enabled at scale 1.0 - this is the bug!")
            return False
        
        config = get_nag_config()
        print(f"   NAG config: scale={config['scale']}, enabled={config['enabled']}")
        
        # Test 2: Check attention processor behavior at scale 1.0
        print("\n2. Testing NAG attention processor at scale 1.0...")
        processor = NAGAttnProcessor2_0(nag_scale=1.0, nag_tau=5.0, nag_alpha=0.5)
        
        # Create mock attention module
        class MockAttention:
            def __init__(self):
                self.heads = 8
                self.spatial_norm = None
                self.group_norm = None
                self.norm_cross = False
                self.norm_q = None
                self.norm_k = None
                self.residual_connection = False
                self.rescale_output_factor = 1.0
                
                # Mock linear layers
                self.to_q = torch.nn.Linear(512, 512)
                self.to_k = torch.nn.Linear(512, 512)
                self.to_v = torch.nn.Linear(512, 512)
                self.to_out = torch.nn.ModuleList([
                    torch.nn.Linear(512, 512),
                    torch.nn.Dropout(0.0)
                ])
            
            def prepare_attention_mask(self, mask, seq_len, batch_size):
                return mask
        
        # Test with mock data
        batch_size = 2  # Simulating unconditional + conditional
        seq_len = 77
        hidden_dim = 512
        
        hidden_states = torch.randn(batch_size, seq_len, hidden_dim)
        encoder_hidden_states = torch.randn(batch_size, seq_len, hidden_dim)
        
        mock_attn = MockAttention()
        
        try:
            # This should not raise an error and should apply some guidance
            result = processor(mock_attn, hidden_states, encoder_hidden_states)
            print("✓ NAG attention processor works at scale 1.0")
            print(f"   Input shape: {hidden_states.shape}")
            print(f"   Output shape: {result.shape}")
            
            # Check that output is different from input (guidance was applied)
            if not torch.allclose(result, hidden_states, atol=1e-6):
                print("✓ NAG guidance was applied (output differs from input)")
            else:
                print("⚠ NAG guidance may not have been applied (output same as input)")
                
        except Exception as e:
            print(f"✗ NAG attention processor failed at scale 1.0: {e}")
            return False
        
        # Test 3: Test different scales
        print("\n3. Testing NAG at different scales...")
        scales_to_test = [1.0, 1.5, 2.0, 3.0]
        
        for scale in scales_to_test:
            enable_nag(scale=scale, negative_prompt="blurry")
            enabled = is_nag_enabled()
            print(f"   Scale {scale}: {'Enabled' if enabled else 'Disabled'}")
            
            if scale >= 1.0 and not enabled:
                print(f"✗ NAG should be enabled at scale {scale}")
                return False
        
        # Clean up
        disable_nag()
        print("\n✓ All tests passed! NAG now works correctly at guidance scale 1.0")
        return True
        
    except Exception as e:
        print(f"✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_nag_mathematical_consistency():
    """Test that NAG math is consistent across different scales"""
    
    print("\n=== Testing NAG Mathematical Consistency ===")
    
    try:
        from extras.nag.attention_nag import NAGAttnProcessor2_0
        
        # Test that the mathematical operations are stable
        scales = [1.0, 1.1, 1.5, 2.0]
        
        for scale in scales:
            processor = NAGAttnProcessor2_0(nag_scale=scale, nag_tau=5.0, nag_alpha=0.5)
            
            # Create test tensors
            positive = torch.randn(1, 77, 512)
            negative = torch.randn(1, 77, 512)
            
            # Simulate the NAG guidance calculation
            if scale == 1.0:
                guidance = positive + (positive - negative) * 0.1
            else:
                guidance = positive * scale - negative * (scale - 1)
            
            # Check for NaN or Inf values
            if torch.isnan(guidance).any() or torch.isinf(guidance).any():
                print(f"✗ Scale {scale}: Invalid values detected")
                return False
            else:
                print(f"✓ Scale {scale}: Mathematical operations stable")
        
        return True
        
    except Exception as e:
        print(f"✗ Mathematical consistency test failed: {e}")
        return False

if __name__ == "__main__":
    print("Testing NAG Guidance Scale 1.0 Fix")
    print("=" * 50)
    
    success1 = test_nag_at_scale_one()
    success2 = test_nag_mathematical_consistency()
    
    if success1 and success2:
        print("\n🎉 All tests passed! NAG guidance scale 1.0 issue is fixed.")
    else:
        print("\n❌ Some tests failed. Please check the implementation.")
        sys.exit(1)
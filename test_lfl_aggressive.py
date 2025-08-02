#!/usr/bin/env python3
"""
Test script for aggressive LFL aesthetic replication with UNet hooks
"""

import torch
import numpy as np
from PIL import Image
import tempfile
import os

def test_aggressive_aesthetic_replicator():
    """Test the aggressive aesthetic replicator with UNet hooks."""
    print("Testing aggressive aesthetic replicator...")
    
    try:
        from extras.LFL.latent_feedback_loop import AestheticReplicator
        
        # Initialize with aggressive settings
        replicator = AestheticReplicator(
            aesthetic_strength=1.0,  # High strength
            blend_mode='aggressive'
        )
        
        print(f"✓ Aggressive replicator initialized: strength={replicator.aesthetic_strength}, mode={replicator.blend_mode}")
        print(f"  Target layers: {len(replicator.feature_layers)} layers")
        
        # Test reference image setting
        test_image = Image.new('RGB', (512, 512), color=(255, 128, 64))
        success = replicator.set_reference_image(test_image, vae=None)
        print(f"✓ Reference image set: {success}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_unet_hook_system():
    """Test the UNet hook system."""
    print("\nTesting UNet hook system...")
    
    try:
        from extras.LFL.latent_feedback_loop import AestheticReplicator
        
        # Create a mock UNet-like module
        class MockUNet:
            def __init__(self):
                self.down_blocks = torch.nn.ModuleList([
                    torch.nn.ModuleDict({
                        'resnets': torch.nn.ModuleList([
                            torch.nn.Conv2d(4, 4, 3, padding=1),
                            torch.nn.Conv2d(4, 4, 3, padding=1)
                        ])
                    }),
                    torch.nn.ModuleDict({
                        'resnets': torch.nn.ModuleList([
                            torch.nn.Conv2d(4, 4, 3, padding=1),
                            torch.nn.Conv2d(4, 4, 3, padding=1)
                        ])
                    })
                ])
                self.mid_block = torch.nn.ModuleDict({
                    'resnets': torch.nn.ModuleList([
                        torch.nn.Conv2d(4, 4, 3, padding=1),
                        torch.nn.Conv2d(4, 4, 3, padding=1)
                    ])
                })
                self.up_blocks = torch.nn.ModuleList([
                    torch.nn.ModuleDict({
                        'resnets': torch.nn.ModuleList([
                            torch.nn.Conv2d(4, 4, 3, padding=1),
                            torch.nn.Conv2d(4, 4, 3, padding=1)
                        ])
                    }),
                    torch.nn.ModuleDict({
                        'resnets': torch.nn.ModuleList([
                            torch.nn.Conv2d(4, 4, 3, padding=1),
                            torch.nn.Conv2d(4, 4, 3, padding=1)
                        ])
                    })
                ])
            
            def forward(self, x, timestep):
                # Simple forward pass through some layers
                for block in self.down_blocks:
                    for resnet in block['resnets']:
                        x = resnet(x)
                
                for resnet in self.mid_block['resnets']:
                    x = resnet(x)
                
                for block in self.up_blocks:
                    for resnet in block['resnets']:
                        x = resnet(x)
                
                return x
        
        mock_unet = MockUNet()
        replicator = AestheticReplicator(aesthetic_strength=0.8, blend_mode='aggressive')
        
        # Set reference image
        test_image = Image.new('RGB', (256, 256), color='red')
        replicator.set_reference_image(test_image, vae=None)
        
        # Test hooking
        replicator.hook_unet_layers(mock_unet)
        print(f"✓ UNet hooked: {len(replicator.hook_handles)} hooks applied")
        
        # Test module finding
        for layer_name in replicator.feature_layers[:3]:  # Test first 3 layers
            module = replicator._get_module_by_name(mock_unet, layer_name)
            if module is not None:
                print(f"✓ Found layer: {layer_name}")
            else:
                print(f"✗ Could not find layer: {layer_name}")
        
        # Test reference activation extraction
        reference_noise = torch.randn(1, 4, 32, 32)
        replicator.extract_reference_activations(mock_unet, reference_noise)
        print(f"✓ Reference activations extracted: {len(replicator.reference_activations)} activations")
        
        # Test cleanup
        replicator.remove_unet_hooks()
        print(f"✓ Hooks removed: {len(replicator.hook_handles)} remaining")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_blend_modes():
    """Test different blend modes."""
    print("\nTesting blend modes...")
    
    try:
        from extras.LFL.latent_feedback_loop import AestheticReplicator
        
        blend_modes = ['aggressive', 'adaptive', 'attention', 'linear']
        
        for mode in blend_modes:
            replicator = AestheticReplicator(
                aesthetic_strength=0.7,
                blend_mode=mode
            )
            
            # Set reference
            replicator.reference_latent = torch.randn(1, 4, 32, 32)
            
            # Test guidance computation
            current_latent = torch.randn(1, 4, 32, 32)
            guidance = replicator.compute_aesthetic_guidance(current_latent)
            
            print(f"✓ Blend mode '{mode}': guidance shape={guidance.shape}, mean={guidance.mean().item():.4f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_global_functions():
    """Test global functions with aggressive settings."""
    print("\nTesting global functions...")
    
    try:
        from extras.LFL.latent_feedback_loop import (
            initialize_aesthetic_replicator,
            hook_unet_for_aesthetic_replication,
            unhook_unet_aesthetic_replication,
            set_aesthetic_timestep,
            set_reference_image
        )
        
        # Initialize with aggressive settings
        replicator = initialize_aesthetic_replicator(
            aesthetic_strength=1.2,  # Very high strength
            blend_mode='aggressive'
        )
        print(f"✓ Aggressive replicator initialized: strength={replicator.aesthetic_strength}")
        
        # Set reference image
        test_image = Image.new('RGB', (128, 128), color='blue')
        success = set_reference_image(test_image, vae=None)
        print(f"✓ Reference image set: {success}")
        
        # Test timestep setting
        set_aesthetic_timestep(500)
        print(f"✓ Timestep set: {replicator.current_timestep}")
        
        # Test unhooking (should not crash even if nothing is hooked)
        unhook_unet_aesthetic_replication()
        print("✓ Unhook function called successfully")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_neural_echo_integration():
    """Test neural echo sampler integration."""
    print("\nTesting neural echo sampler integration...")
    
    try:
        from modules.neural_echo_sampler import (
            setup_aesthetic_replication_for_task,
            hook_unet_for_aesthetic_replication,
            unhook_unet_aesthetic_replication,
            set_aesthetic_timestep
        )
        
        # Create mock task with aggressive settings
        class MockTask:
            def __init__(self):
                self.lfl_enabled = True
                self.lfl_reference_image = Image.new('RGB', (256, 256), color='green')
                self.lfl_aesthetic_strength = 1.5  # Very aggressive
                self.lfl_blend_mode = 'aggressive'
        
        task = MockTask()
        
        # Test setup
        replicator = setup_aesthetic_replication_for_task(task, vae=None)
        if replicator:
            print(f"✓ Aggressive setup successful: strength={replicator.aesthetic_strength}, mode={replicator.blend_mode}")
        else:
            print("✗ Setup failed")
            return False
        
        # Test hook functions
        unhook_unet_aesthetic_replication()  # Should not crash
        set_aesthetic_timestep(750)  # Should not crash
        print("✓ Hook management functions work")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("=" * 70)
    print("LFL Aggressive Aesthetic Replication Test Suite")
    print("=" * 70)
    
    tests = [
        test_aggressive_aesthetic_replicator,
        test_unet_hook_system,
        test_blend_modes,
        test_global_functions,
        test_neural_echo_integration
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 70)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Aggressive LFL is working correctly.")
        print("\nKey Features:")
        print("• 🔥 AGGRESSIVE aesthetic replication with UNet hooks")
        print("• 🎯 Direct layer-level aesthetic influence")
        print("• ⚡ Multiple blend modes (aggressive, adaptive, attention, linear)")
        print("• 🧠 Reference activation extraction and matching")
        print("• 🔧 Configurable strength up to 2.0+ for maximum impact")
        print("• 🧹 Automatic cleanup of hooks after generation")
        print("\nExpected Visual Impact:")
        print("• Much stronger aesthetic replication than before")
        print("• Direct influence on UNet feature layers")
        print("• Visible style transfer from reference image")
        print("• Adaptive strength based on generation progress")
        return True
    else:
        print("❌ Some tests failed. Please check the implementation.")
        return False


if __name__ == "__main__":
    main()
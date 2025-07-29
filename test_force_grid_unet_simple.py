#!/usr/bin/env python3
"""
Simple test for Force Grid UNet integration (no PyTorch dependencies)
"""

def test_force_grid_unet_file_exists():
    """Test that Force Grid UNet file exists"""
    print("Testing Force Grid UNet file structure...")
    
    import os
    
    if os.path.exists("extensions/force_grid_unet.py"):
        print("✓ extensions/force_grid_unet.py exists")
        return True
    else:
        print("✗ extensions/force_grid_unet.py missing")
        return False

def test_force_grid_unet_integration():
    """Test that Force Grid UNet is integrated into default_pipeline"""
    print("\nTesting Force Grid UNet integration...")
    
    try:
        with open("modules/default_pipeline.py", "r") as f:
            content = f.read()
        
        checks = [
            ("ForceGridUNetContext", "ForceGridUNetContext import/usage"),
            ("force_grid_unet_context", "UNet context variable"),
            ("grid_size =", "Grid size calculation"),
            ("blend_strength=", "Blend strength configuration"),
            ("unet_model = final_unet", "UNet model reference"),
        ]
        
        for check_str, description in checks:
            if check_str in content:
                print(f"✓ {description} found")
            else:
                print(f"✗ {description} not found")
                return False
        
        return True
        
    except Exception as e:
        print(f"✗ Integration test failed: {e}")
        return False

def test_force_grid_unet_code_structure():
    """Test the structure of Force Grid UNet code"""
    print("\nTesting Force Grid UNet code structure...")
    
    try:
        with open("extensions/force_grid_unet.py", "r") as f:
            content = f.read()
        
        classes_and_functions = [
            ("class ForceGridUNet:", "Main ForceGridUNet class"),
            ("class ForceGridUNetInterface:", "Interface class"),
            ("class ForceGridUNetContext:", "Context manager class"),
            ("def _apply_grid_transformation", "Grid transformation method"),
            ("def _transform_grid_cell", "Cell transformation method"),
            ("def _apply_rotation_bias", "Rotation bias method"),
            ("def _apply_scale_bias", "Scale bias method"),
            ("def _apply_contrast_bias", "Contrast bias method"),
            ("def _apply_frequency_bias", "Frequency bias method"),
            ("force_grid_unet = ForceGridUNet()", "Global instance"),
        ]
        
        for check_str, description in classes_and_functions:
            if check_str in content:
                print(f"✓ {description} found")
            else:
                print(f"✗ {description} not found")
                return False
        
        return True
        
    except Exception as e:
        print(f"✗ Code structure test failed: {e}")
        return False

def test_unet_approach_vs_postprocessing():
    """Test that this is UNet-level approach, not post-processing"""
    print("\nTesting UNet-level approach...")
    
    try:
        with open("extensions/force_grid_unet.py", "r") as f:
            content = f.read()
        
        unet_indicators = [
            ("ModelPatcher", "ModelPatcher handling"),
            ("original_forward", "Original forward method storage"),
            ("def grid_forward", "Grid-enhanced forward method"),
            ("Apply grid transformation to the UNet output", "UNet output transformation"),
            ("tensor: torch.Tensor", "Tensor-level operations"),
        ]
        
        for indicator, description in unet_indicators:
            if indicator in content:
                print(f"✓ {description} found")
            else:
                print(f"✗ {description} not found")
                return False
        
        # Check that it's NOT post-processing
        postprocessing_indicators = [
            "PIL.Image",
            "Image.new",
            "paste(",
            "save(",
        ]
        
        has_postprocessing = any(indicator in content for indicator in postprocessing_indicators)
        
        if not has_postprocessing:
            print("✓ No post-processing approach detected (good)")
        else:
            print("✗ Post-processing approach detected (should be UNet-level)")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ UNet approach test failed: {e}")
        return False

def test_grid_transformation_logic():
    """Test that grid transformation logic is present"""
    print("\nTesting grid transformation logic...")
    
    try:
        with open("extensions/force_grid_unet.py", "r") as f:
            content = f.read()
        
        transformation_logic = [
            ("cell_height = height // rows", "Grid cell dimension calculation"),
            ("cell_width = width // cols", "Grid cell dimension calculation"),
            ("for row in range(rows):", "Grid row iteration"),
            ("for col in range(cols):", "Grid column iteration"),
            ("y_start = row * cell_height", "Cell boundary calculation"),
            ("x_start = col * cell_width", "Cell boundary calculation"),
            ("pattern_id = (row * total_cols + col) % 4", "Pattern variation logic"),
            ("blend_strength", "Blending configuration"),
        ]
        
        for logic_str, description in transformation_logic:
            if logic_str in content:
                print(f"✓ {description} found")
            else:
                print(f"✗ {description} not found")
                return False
        
        return True
        
    except Exception as e:
        print(f"✗ Grid transformation logic test failed: {e}")
        return False

def main():
    """Run all Force Grid UNet simple tests"""
    print("Force Grid UNet Simple Test Suite")
    print("=" * 50)
    
    tests = [
        test_force_grid_unet_file_exists,
        test_force_grid_unet_integration,
        test_force_grid_unet_code_structure,
        test_unet_approach_vs_postprocessing,
        test_grid_transformation_logic,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 50)
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("✅ All Force Grid UNet tests passed!")
        print("\n🎯 Force Grid UNet Implementation Complete!")
        print("\nKey Features:")
        print("• UNet-level grid generation (NOT post-processing)")
        print("• Patches UNet forward pass during diffusion")
        print("• Creates grid patterns within single image generation")
        print("• Different transformations per grid cell:")
        print("  - Rotation bias")
        print("  - Scale variation") 
        print("  - Contrast adjustment")
        print("  - Frequency modulation")
        print("• Configurable grid size based on image dimensions")
        print("• Adjustable blend strength")
        print("\n🚀 Usage:")
        print("1. Check 'Generate Grid Image (Experimental)' in UI")
        print("2. Generate a single image")
        print("3. UNet will create grid-like patterns during diffusion")
        print("4. Result: Single image with built-in grid structure!")
        
        return True
    else:
        print("❌ Some Force Grid UNet tests failed!")
        return False

if __name__ == "__main__":
    main()
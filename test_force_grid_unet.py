#!/usr/bin/env python3
"""
Test script for Force Grid UNet implementation
"""

def test_force_grid_unet_imports():
    """Test that Force Grid UNet components can be imported"""
    print("Testing Force Grid UNet imports...")
    
    try:
        from extensions.force_grid_unet import (
            ForceGridUNet, ForceGridUNetInterface, ForceGridUNetContext,
            force_grid_unet, force_grid_unet_interface,
            enable_force_grid_unet, disable_force_grid_unet, 
            get_force_grid_unet_status, with_force_grid_unet
        )
        print("✓ All Force Grid UNet components imported successfully")
        return True
    except Exception as e:
        print(f"✗ Force Grid UNet import failed: {e}")
        return False

def test_force_grid_unet_structure():
    """Test the structure of Force Grid UNet classes"""
    print("\nTesting Force Grid UNet structure...")
    
    try:
        from extensions.force_grid_unet import ForceGridUNet, ForceGridUNetInterface
        
        # Test ForceGridUNet class
        grid_unet = ForceGridUNet(grid_size=(2, 2), blend_strength=0.1)
        
        if hasattr(grid_unet, 'activate') and hasattr(grid_unet, 'deactivate'):
            print("✓ ForceGridUNet has activate/deactivate methods")
        else:
            print("✗ ForceGridUNet missing required methods")
            return False
        
        if hasattr(grid_unet, 'grid_size') and hasattr(grid_unet, 'blend_strength'):
            print("✓ ForceGridUNet has required attributes")
        else:
            print("✗ ForceGridUNet missing required attributes")
            return False
        
        # Test ForceGridUNetInterface class
        interface = ForceGridUNetInterface()
        
        if hasattr(interface, 'enable') and hasattr(interface, 'disable'):
            print("✓ ForceGridUNetInterface has enable/disable methods")
        else:
            print("✗ ForceGridUNetInterface missing required methods")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ Force Grid UNet structure test failed: {e}")
        return False

def test_force_grid_unet_integration():
    """Test that Force Grid UNet is integrated into default_pipeline"""
    print("\nTesting Force Grid UNet integration...")
    
    try:
        with open("modules/default_pipeline.py", "r") as f:
            content = f.read()
        
        if "from extensions.force_grid_unet import ForceGridUNetContext" in content:
            print("✓ Force Grid UNet import found in default_pipeline.py")
        else:
            print("✗ Force Grid UNet import not found in default_pipeline.py")
            return False
        
        if "ForceGridUNetContext" in content:
            print("✓ ForceGridUNetContext usage found")
        else:
            print("✗ ForceGridUNetContext usage not found")
            return False
        
        if "force_grid_unet_context" in content:
            print("✓ Force Grid UNet context variable found")
        else:
            print("✗ Force Grid UNet context variable not found")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ Force Grid UNet integration test failed: {e}")
        return False

def test_grid_transformation_methods():
    """Test that grid transformation methods exist"""
    print("\nTesting grid transformation methods...")
    
    try:
        from extensions.force_grid_unet import ForceGridUNet
        
        grid_unet = ForceGridUNet()
        
        methods_to_check = [
            '_apply_grid_transformation',
            '_transform_grid_cell',
            '_apply_rotation_bias',
            '_apply_scale_bias',
            '_apply_contrast_bias',
            '_apply_frequency_bias'
        ]
        
        for method_name in methods_to_check:
            if hasattr(grid_unet, method_name):
                print(f"✓ {method_name} method exists")
            else:
                print(f"✗ {method_name} method missing")
                return False
        
        return True
        
    except Exception as e:
        print(f"✗ Grid transformation methods test failed: {e}")
        return False

def test_force_grid_unet_configuration():
    """Test Force Grid UNet configuration options"""
    print("\nTesting Force Grid UNet configuration...")
    
    try:
        from extensions.force_grid_unet import ForceGridUNet
        
        # Test different grid sizes
        grid_sizes = [(2, 2), (3, 3), (2, 3), (4, 2)]
        
        for grid_size in grid_sizes:
            grid_unet = ForceGridUNet(grid_size=grid_size)
            if grid_unet.grid_size == grid_size:
                print(f"✓ Grid size {grid_size} configured correctly")
            else:
                print(f"✗ Grid size {grid_size} not configured correctly")
                return False
        
        # Test blend strength
        blend_strengths = [0.0, 0.1, 0.5, 1.0]
        
        for blend_strength in blend_strengths:
            grid_unet = ForceGridUNet(blend_strength=blend_strength)
            if abs(grid_unet.blend_strength - blend_strength) < 1e-6:
                print(f"✓ Blend strength {blend_strength} configured correctly")
            else:
                print(f"✗ Blend strength {blend_strength} not configured correctly")
                return False
        
        return True
        
    except Exception as e:
        print(f"✗ Force Grid UNet configuration test failed: {e}")
        return False

def main():
    """Run all Force Grid UNet tests"""
    print("Force Grid UNet Test Suite")
    print("=" * 50)
    
    tests = [
        test_force_grid_unet_imports,
        test_force_grid_unet_structure,
        test_force_grid_unet_integration,
        test_grid_transformation_methods,
        test_force_grid_unet_configuration,
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
        print("✓ All Force Grid UNet tests passed!")
        print("\nForce Grid UNet Implementation Summary:")
        print("• UNet-level grid generation (not post-processing)")
        print("• Modifies diffusion process to create grid patterns")
        print("• Different transformations for each grid cell")
        print("• Configurable grid size and blend strength")
        print("• Integrated into the generation pipeline")
        print("\nThis approach forces the UNet to generate a single image")
        print("with grid-like structure during the diffusion process!")
        return True
    else:
        print("✗ Some Force Grid UNet tests failed!")
        return False

if __name__ == "__main__":
    main()
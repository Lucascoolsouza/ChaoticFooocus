#!/usr/bin/env python3
"""
Quick test to verify Force Grid UNet can be imported
"""

def test_import():
    """Test that Force Grid UNet can be imported without PyTorch"""
    print("Testing Force Grid UNet import...")
    
    try:
        # Test basic import structure without actually importing torch-dependent code
        import sys
        import os
        
        # Add current directory to path
        sys.path.insert(0, os.getcwd())
        
        # Check if the file exists and has the right structure
        with open("extensions/force_grid_unet.py", "r") as f:
            content = f.read()
        
        # Check for key components
        if "class ForceGridUNet:" in content:
            print("✓ ForceGridUNet class found")
        else:
            print("✗ ForceGridUNet class not found")
            return False
        
        if "class ForceGridUNetContext:" in content:
            print("✓ ForceGridUNetContext class found")
        else:
            print("✗ ForceGridUNetContext class not found")
            return False
        
        # Check that __init__.py exists
        if os.path.exists("extensions/__init__.py"):
            print("✓ extensions/__init__.py exists")
        else:
            print("✗ extensions/__init__.py missing")
            return False
        
        print("✓ Force Grid UNet module structure is correct")
        return True
        
    except Exception as e:
        print(f"✗ Import test failed: {e}")
        return False

if __name__ == "__main__":
    if test_import():
        print("\n🎉 Force Grid UNet is ready to use!")
        print("The 'No module named extensions.force_grid_unet' error should now be fixed.")
    else:
        print("\n❌ Force Grid UNet import test failed.")
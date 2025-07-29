#!/usr/bin/env python3
"""
Simple test to verify ConfuseVAE is working
"""

import torch
import sys
import os

def test_confuse_vae_import():
    """Test that ConfuseVAE can be imported and instantiated"""
    print("Testing ConfuseVAE import and instantiation...")
    
    try:
        from extras.confuse_vae import ConfuseVAE
        print("✓ ConfuseVAE imported successfully")
        
        # Create a dummy VAE state dict (minimal)
        dummy_vae_sd = {
            'decoder.conv_in.weight': torch.randn(512, 4, 3, 3),
            'decoder.conv_in.bias': torch.randn(512),
            'decoder.conv_out.weight': torch.randn(3, 512, 3, 3),
            'decoder.conv_out.bias': torch.randn(3),
        }
        
        # Test with different artistic strengths
        for strength in [0.0, 2.5, 5.0, 7.5, 10.0]:
            try:
                confuse_vae = ConfuseVAE(sd=dummy_vae_sd, artistic_strength=strength)
                print(f"✓ ConfuseVAE created with strength {strength}")
                print(f"  - Artistic strength: {confuse_vae.artistic_strength}")
                
                # Test the confusion method with dummy data
                dummy_samples = torch.randn(1, 4, 64, 64)
                confused_samples = confuse_vae._apply_artistic_confusion(dummy_samples)
                
                if strength > 0:
                    # Check if samples were modified
                    if not torch.equal(dummy_samples, confused_samples):
                        print(f"  - ✓ Samples modified (confusion applied)")
                    else:
                        print(f"  - ⚠ Samples not modified (might be random)")
                else:
                    # For strength 0, samples should be unchanged
                    if torch.equal(dummy_samples, confused_samples):
                        print(f"  - ✓ Samples unchanged (strength 0)")
                    else:
                        print(f"  - ✗ Samples modified when strength is 0")
                
            except Exception as e:
                print(f"✗ Error creating ConfuseVAE with strength {strength}: {e}")
                return False
        
        return True
        
    except ImportError as e:
        print(f"✗ Failed to import ConfuseVAE: {e}")
        return False
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False

def test_integration_in_sd_module():
    """Test that the integration in sd.py is working"""
    print("\nTesting integration in sd.py...")
    
    try:
        # Check if the integration code exists
        with open("ldm_patched/modules/sd.py", "r") as f:
            content = f.read()
        
        if "from extras.confuse_vae import ConfuseVAE" in content:
            print("✓ ConfuseVAE import found in sd.py")
        else:
            print("✗ ConfuseVAE import not found in sd.py")
            return False
        
        if "ConfuseVAE(sd=vae_sd, artistic_strength=artistic_strength)" in content:
            print("✓ ConfuseVAE instantiation found in sd.py")
        else:
            print("✗ ConfuseVAE instantiation not found in sd.py")
            return False
        
        if "if artistic_strength > 0:" in content:
            print("✓ Artistic strength condition found in sd.py")
        else:
            print("✗ Artistic strength condition not found in sd.py")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ Error checking sd.py: {e}")
        return False

def test_parameter_flow():
    """Test that artistic_strength parameter flows through the system"""
    print("\nTesting parameter flow...")
    
    files_to_check = [
        ("webui.py", "artistic_strength = gr.Slider"),
        ("modules/async_worker.py", "self.artistic_strength"),
        ("modules/default_pipeline.py", "artistic_strength="),
        ("modules/core.py", "artistic_strength="),
        ("ldm_patched/modules/sd.py", "artistic_strength")
    ]
    
    all_good = True
    for filename, pattern in files_to_check:
        try:
            with open(filename, "r") as f:
                content = f.read()
            
            if pattern in content:
                print(f"✓ {filename}: {pattern} found")
            else:
                print(f"✗ {filename}: {pattern} not found")
                all_good = False
                
        except Exception as e:
            print(f"✗ Error checking {filename}: {e}")
            all_good = False
    
    return all_good

def debug_why_not_working():
    """Debug why ConfuseVAE might not be working"""
    print("\nDebugging why ConfuseVAE might not be working...")
    
    # Check if the file exists
    if os.path.exists("extras/confuse_vae.py"):
        print("✓ extras/confuse_vae.py exists")
    else:
        print("✗ extras/confuse_vae.py does not exist")
        return
    
    # Check if extras is a package
    if os.path.exists("extras/__init__.py"):
        print("✓ extras/__init__.py exists (package)")
    else:
        print("⚠ extras/__init__.py does not exist (not a package)")
        print("  Creating __init__.py...")
        try:
            with open("extras/__init__.py", "w") as f:
                f.write("# Extras package\n")
            print("✓ Created extras/__init__.py")
        except Exception as e:
            print(f"✗ Failed to create extras/__init__.py: {e}")
    
    # Test import path
    try:
        sys.path.insert(0, '.')
        from extras.confuse_vae import ConfuseVAE
        print("✓ Import works from current directory")
    except Exception as e:
        print(f"✗ Import failed: {e}")
        
        # Try alternative import
        try:
            sys.path.insert(0, 'extras')
            import confuse_vae
            print("✓ Alternative import works")
        except Exception as e2:
            print(f"✗ Alternative import also failed: {e2}")

def main():
    """Run all tests"""
    print("ConfuseVAE Simple Test Suite")
    print("=" * 50)
    
    success = True
    success &= test_confuse_vae_import()
    success &= test_integration_in_sd_module()
    success &= test_parameter_flow()
    debug_why_not_working()
    
    print("\n" + "=" * 50)
    if success:
        print("✓ All tests passed! ConfuseVAE should be working.")
        print("If it's still not working, try setting artistic_strength > 0 in the UI.")
    else:
        print("✗ Some tests failed. Check the output above for issues.")

if __name__ == "__main__":
    main()
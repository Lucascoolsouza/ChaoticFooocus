#!/usr/bin/env python3
"""
Debug test for ConfuseVAE without torch dependencies
"""

def test_file_structure():
    """Test that all required files exist and have the right content"""
    print("Testing file structure...")
    
    import os
    
    # Check if files exist
    files_to_check = [
        "extras/__init__.py",
        "extras/confuse_vae.py",
        "ldm_patched/modules/sd.py",
        "modules/async_worker.py",
        "webui.py"
    ]
    
    for file_path in files_to_check:
        if os.path.exists(file_path):
            print(f"✓ {file_path} exists")
        else:
            print(f"✗ {file_path} missing")
            return False
    
    return True

def test_integration_code():
    """Test that the integration code is present"""
    print("\nTesting integration code...")
    
    # Check sd.py integration
    try:
        with open("ldm_patched/modules/sd.py", "r") as f:
            sd_content = f.read()
        
        required_patterns = [
            "from extras.confuse_vae import ConfuseVAE",
            "if artistic_strength > 0:",
            "vae = ConfuseVAE(sd=vae_sd, artistic_strength=artistic_strength)"
        ]
        
        for pattern in required_patterns:
            if pattern in sd_content:
                print(f"✓ sd.py: {pattern}")
            else:
                print(f"✗ sd.py: {pattern} missing")
                return False
        
    except Exception as e:
        print(f"✗ Error checking sd.py: {e}")
        return False
    
    # Check webui.py for UI control
    try:
        with open("webui.py", "r") as f:
            webui_content = f.read()
        
        if "artistic_strength" in webui_content and "gr.Slider" in webui_content:
            print("✓ webui.py: artistic_strength slider found")
        else:
            print("✗ webui.py: artistic_strength slider missing")
            return False
        
        if "ctrls += [artistic_strength]" in webui_content:
            print("✓ webui.py: artistic_strength in controls")
        else:
            print("✗ webui.py: artistic_strength not in controls")
            return False
            
    except Exception as e:
        print(f"✗ Error checking webui.py: {e}")
        return False
    
    # Check async_worker.py for parameter passing
    try:
        with open("modules/async_worker.py", "r") as f:
            worker_content = f.read()
        
        if "self.artistic_strength" in worker_content:
            print("✓ async_worker.py: artistic_strength parameter found")
        else:
            print("✗ async_worker.py: artistic_strength parameter missing")
            return False
            
    except Exception as e:
        print(f"✗ Error checking async_worker.py: {e}")
        return False
    
    return True

def test_confuse_vae_syntax():
    """Test that ConfuseVAE has correct syntax"""
    print("\nTesting ConfuseVAE syntax...")
    
    try:
        with open("extras/confuse_vae.py", "r") as f:
            content = f.read()
        
        # Check for required methods
        required_methods = [
            "def __init__",
            "def _apply_artistic_confusion",
            "def decode",
            "def decode_tiled"
        ]
        
        for method in required_methods:
            if method in content:
                print(f"✓ {method} found")
            else:
                print(f"✗ {method} missing")
                return False
        
        # Check for artistic_strength usage
        if "self.artistic_strength" in content:
            print("✓ artistic_strength attribute found")
        else:
            print("✗ artistic_strength attribute missing")
            return False
        
        # Check for strength conditions
        if "if self.artistic_strength" in content:
            print("✓ artistic_strength conditions found")
        else:
            print("✗ artistic_strength conditions missing")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ Error checking confuse_vae.py: {e}")
        return False

def check_why_not_working():
    """Provide debugging information"""
    print("\nDebugging why ConfuseVAE might not be working...")
    
    print("\nPossible reasons:")
    print("1. artistic_strength is set to 0 in the UI")
    print("   → Check that you're setting the 'Artistic Strength' slider > 0")
    
    print("2. The UI control is not connected properly")
    print("   → Check that artistic_strength is in the ctrls list")
    
    print("3. The parameter is not being passed through the pipeline")
    print("   → Check async_worker.py parameter handling")
    
    print("4. Import error at runtime")
    print("   → Check that extras/__init__.py exists")
    
    print("5. VAE is not being reloaded after changes")
    print("   → Try changing the model or restarting the application")
    
    print("\nTo test if it's working:")
    print("1. Set 'Artistic Strength' to 5.0 in the UI")
    print("2. Generate an image")
    print("3. Compare with strength 0.0")
    print("4. You should see visible distortion/artistic effects")

def create_debug_patch():
    """Create a debug patch to add logging"""
    print("\nCreating debug patch...")
    
    debug_patch = '''
# Add this to the beginning of _apply_artistic_confusion method in confuse_vae.py:
def _apply_artistic_confusion(self, samples):
    """Apply various artistic confusion effects to the latent samples"""
    print(f"[DEBUG] ConfuseVAE: artistic_strength = {self.artistic_strength}")
    print(f"[DEBUG] ConfuseVAE: samples shape = {samples.shape}")
    
    if self.artistic_strength <= 0:
        print("[DEBUG] ConfuseVAE: No effects applied (strength <= 0)")
        return samples
    
    print(f"[DEBUG] ConfuseVAE: Applying artistic effects with strength {self.artistic_strength}")
    # ... rest of the method
'''
    
    print("Debug patch:")
    print(debug_patch)
    
    print("\nThis will help you see if:")
    print("- ConfuseVAE is being called")
    print("- What artistic_strength value is being used")
    print("- If the effects are being applied")

def main():
    """Run all debug tests"""
    print("ConfuseVAE Debug Test Suite")
    print("=" * 50)
    
    success = True
    success &= test_file_structure()
    success &= test_integration_code()
    success &= test_confuse_vae_syntax()
    
    check_why_not_working()
    create_debug_patch()
    
    print("\n" + "=" * 50)
    if success:
        print("✓ All integration code is in place!")
        print("If ConfuseVAE is still not working, make sure:")
        print("1. Set artistic_strength > 0 in the UI")
        print("2. Restart the application after changes")
        print("3. Add debug logging to see what's happening")
    else:
        print("✗ Some integration code is missing.")
        print("Check the output above and fix the missing parts.")

if __name__ == "__main__":
    main()
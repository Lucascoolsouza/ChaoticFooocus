#!/usr/bin/env python3
"""
Test to verify that artistic_strength parameter is being passed correctly
"""

def test_parameter_order():
    """Test that the parameter order matches between webui.py and async_worker.py"""
    print("Testing parameter order fix...")
    
    # Check webui.py ctrls order
    try:
        with open("webui.py", "r") as f:
            webui_content = f.read()
        
        # Find the artistic_strength position relative to disco parameters
        disco_guidance_pos = webui_content.find("disco_guidance_steps")
        artistic_strength_pos = webui_content.find("ctrls += [artistic_strength]")
        
        if disco_guidance_pos < artistic_strength_pos:
            print("✓ webui.py: disco_guidance_steps comes before artistic_strength")
        else:
            print("✗ webui.py: Parameter order is wrong")
            return False
            
    except Exception as e:
        print(f"✗ Error checking webui.py: {e}")
        return False
    
    # Check async_worker.py processing order
    try:
        with open("modules/async_worker.py", "r") as f:
            worker_content = f.read()
        
        # Find the processing order
        disco_guidance_pos = worker_content.find("self.disco_guidance_steps = args.pop()")
        artistic_strength_pos = worker_content.find("self.artistic_strength = args.pop()")
        
        if disco_guidance_pos < artistic_strength_pos:
            print("✓ async_worker.py: disco_guidance_steps processed before artistic_strength")
        else:
            print("✗ async_worker.py: Parameter processing order is wrong")
            return False
            
    except Exception as e:
        print(f"✗ Error checking async_worker.py: {e}")
        return False
    
    return True

def test_debug_logging():
    """Test that debug logging is in place"""
    print("\nTesting debug logging...")
    
    files_to_check = [
        ("extras/confuse_vae.py", "[DEBUG] ConfuseVAE"),
        ("ldm_patched/modules/sd.py", "[DEBUG] Creating ConfuseVAE"),
        ("modules/async_worker.py", "[DEBUG] Confuse VAE artistic_strength")
    ]
    
    all_good = True
    for filename, pattern in files_to_check:
        try:
            with open(filename, "r") as f:
                content = f.read()
            
            if pattern in content:
                print(f"✓ {filename}: Debug logging found")
            else:
                print(f"✗ {filename}: Debug logging missing")
                all_good = False
                
        except Exception as e:
            print(f"✗ Error checking {filename}: {e}")
            all_good = False
    
    return all_good

def main():
    """Run all tests"""
    print("Artistic Strength Parameter Fix Test")
    print("=" * 50)
    
    success = True
    success &= test_parameter_order()
    success &= test_debug_logging()
    
    print("\n" + "=" * 50)
    if success:
        print("✓ All tests passed!")
        print("\nTo test the fix:")
        print("1. Restart the application")
        print("2. Set 'Artistic Strength' to 5.0 in the UI")
        print("3. Generate an image")
        print("4. Check console for debug messages:")
        print("   - [DEBUG] Confuse VAE artistic_strength: 5.0")
        print("   - [DEBUG] Creating ConfuseVAE with artistic_strength = 5.0")
        print("   - [DEBUG] ConfuseVAE.decode called with artistic_strength = 5.0")
    else:
        print("✗ Some tests failed. Check the output above.")

if __name__ == "__main__":
    main()
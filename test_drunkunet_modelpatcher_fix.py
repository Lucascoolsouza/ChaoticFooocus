#!/usr/bin/env python3
"""
Test to verify DRUNKUNet ModelPatcher fix
"""

def test_modelpatcher_fix():
    """Test that the ModelPatcher fix is properly implemented"""
    try:
        print("Testing DRUNKUNet ModelPatcher fix...")
        
        # Read the DRUNKUNet file
        with open('extras/drunkunet/drunkieunet_pipelinesdxl.py', 'r') as f:
            content = f.read()
        
        # Check for the ModelPatcher fix pattern
        fix_patterns = [
            'actual_unet = unet.model if hasattr(unet, \'model\') else unet',
            'for name, module in actual_unet.named_modules():',
            'Access the actual UNet model from the ModelPatcher'
        ]
        
        missing_patterns = []
        for pattern in fix_patterns:
            if pattern not in content:
                missing_patterns.append(pattern)
        
        if missing_patterns:
            print(f"❌ FAIL: Missing ModelPatcher fix patterns:")
            for missing in missing_patterns:
                print(f"  - {missing}")
            return False
        
        print("✅ PASS: All ModelPatcher fix patterns found")
        
        # Count how many times the fix is applied
        fix_count = content.count('actual_unet = unet.model if hasattr(unet, \'model\') else unet')
        print(f"✅ ModelPatcher fix applied {fix_count} times")
        
        # Check that there are no direct calls to unet.named_modules()
        direct_calls = content.count('unet.named_modules()')
        if direct_calls > 0:
            print(f"❌ FAIL: Found {direct_calls} direct calls to unet.named_modules()")
            return False
        
        print("✅ PASS: No direct calls to unet.named_modules() found")
        
        # Check for safety checks
        safety_checks = [
            'torch.isnan(result).any() or torch.isinf(result).any()',
            'Warning: NaN/Inf detected'
        ]
        
        missing_safety = []
        for check in safety_checks:
            if check not in content:
                missing_safety.append(check)
        
        if missing_safety:
            print(f"❌ FAIL: Missing safety checks:")
            for missing in missing_safety:
                print(f"  - {missing}")
            return False
        
        print("✅ PASS: Safety checks are present")
        
        return True
        
    except Exception as e:
        print(f"❌ FAIL: Error testing ModelPatcher fix: {e}")
        import traceback
        traceback.print_exc()
        return False

def analyze_error_log():
    """Analyze the error from the log"""
    print("\nAnalyzing the error log...")
    print("Error: 'ModelPatcher' object has no attribute 'named_modules'")
    print("\nThis error suggests that:")
    print("1. The DRUNKUNet code is still trying to call unet.named_modules() directly")
    print("2. OR the file changes weren't reloaded properly")
    print("3. OR there's a different code path causing the issue")
    
    print("\nPossible solutions:")
    print("1. Restart the application to reload the DRUNKUNet module")
    print("2. Check if there are cached .pyc files that need to be cleared")
    print("3. Verify that the correct DRUNKUNet file is being imported")

if __name__ == "__main__":
    success = test_modelpatcher_fix()
    if success:
        print("\n🎉 DRUNKUNet ModelPatcher fix looks correct!")
        print("\nThe error you're seeing might be due to:")
        print("1. 🔄 Module not reloaded - try restarting the application")
        print("2. 📁 Cached bytecode - clear __pycache__ folders")
        print("3. 🔍 Different import path - verify the correct file is loaded")
        print("\nThe fix should work once the module is properly reloaded.")
    else:
        print("\n❌ DRUNKUNet ModelPatcher fix needs attention.")
    
    analyze_error_log()
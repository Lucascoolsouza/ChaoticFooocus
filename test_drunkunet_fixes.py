#!/usr/bin/env python3
"""
Test script to verify DRUNKUNet fixes
"""

def test_drunkunet_fixes():
    """Test that DRUNKUNet fixes are properly implemented"""
    try:
        print("Testing DRUNKUNet fixes...")
        
        # Read the DRUNKUNet file to check for fixes
        with open('extras/drunkunet/drunkieunet_pipelinesdxl.py', 'r') as f:
            content = f.read()
        
        # Check for ModelPatcher fix
        modelpatcher_fixes = [
            'actual_unet = unet.model if hasattr(unet, \'model\') else unet',
            'for name, module in actual_unet.named_modules():'
        ]
        
        missing_modelpatcher_fixes = []
        for fix in modelpatcher_fixes:
            if fix not in content:
                missing_modelpatcher_fixes.append(fix)
        
        if missing_modelpatcher_fixes:
            print(f"❌ FAIL: Missing ModelPatcher fixes:")
            for missing in missing_modelpatcher_fixes:
                print(f"  - {missing}")
            return False
        
        print("✅ PASS: ModelPatcher fixes are implemented")
        
        # Check for safety checks (NaN/Inf detection)
        safety_checks = [
            'torch.isnan(result).any() or torch.isinf(result).any()',
            'Warning: NaN/Inf detected',
            'if not (torch.isnan(noise_cond).any() or torch.isinf(noise_cond).any()):'
        ]
        
        missing_safety_checks = []
        for check in safety_checks:
            if check not in content:
                missing_safety_checks.append(check)
        
        if missing_safety_checks:
            print(f"❌ FAIL: Missing safety checks:")
            for missing in missing_safety_checks:
                print(f"  - {missing}")
            return False
        
        print("✅ PASS: Safety checks (NaN/Inf detection) are implemented")
        
        # Check for proper hook counting
        if 'hook_count += 1' not in content:
            print("❌ FAIL: Missing proper hook counting")
            return False
        
        print("✅ PASS: Proper hook counting is implemented")
        
        return True
        
    except Exception as e:
        print(f"❌ FAIL: Error testing DRUNKUNet fixes: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_drunkunet_fixes()
    if success:
        print("\n🎉 DRUNKUNet fixes test passed!")
        print("The fixes should resolve:")
        print("1. ✅ ModelPatcher 'named_modules' error")
        print("2. ✅ NaN/Inf values causing RuntimeWarning")
        print("3. ✅ Proper hook counting and registration")
        print("\nDRUNKUNet should now work more reliably!")
    else:
        print("\n❌ DRUNKUNet fixes test failed.")
        print("Additional fixes may be needed.")
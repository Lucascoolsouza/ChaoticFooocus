#!/usr/bin/env python3

"""
Quick syntax check for the updated modules.
"""

import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that the modules can be imported without syntax errors."""
    print("Testing module imports...")
    
    try:
        print("1. Testing disco_daemon...")
        import modules.disco_daemon
        print("   ✓ disco_daemon imported successfully")
        
        print("2. Testing default_pipeline...")
        import modules.default_pipeline
        print("   ✓ default_pipeline imported successfully")
        
        print("3. Testing disco_daemon functionality...")
        from modules.disco_daemon import disco_daemon, apply_disco_distortion
        import torch
        
        # Test basic functionality
        test_tensor = torch.randn(1, 4, 32, 32)
        result = apply_disco_distortion(test_tensor, disco_scale=5.0, distortion_type='psychedelic')
        
        print(f"   ✓ Disco distortion applied: {not torch.equal(test_tensor, result)}")
        
        print("4. Testing daemon configuration...")
        status = disco_daemon.update_settings(enabled=True, disco_scale=10.0, distortion_type='fractal')
        print(f"   ✓ Daemon configured: {status}")
        
        return True
        
    except Exception as e:
        print(f"❌ Import test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run the syntax check."""
    print("=== Syntax and Import Check ===\n")
    
    success = test_imports()
    
    if success:
        print("\n🎉 All modules imported successfully!")
        print("The disco integration should now work without syntax errors.")
        return 0
    else:
        print("\n❌ There are still issues to fix.")
        return 1

if __name__ == "__main__":
    exit(main())
#!/usr/bin/env python3
"""
Debug script to test Force Grid import and identify specific errors
"""

def test_force_grid_import():
    """Test importing Force Grid and identify any issues"""
    try:
        print("Testing Force Grid import...")
        
        # Test basic import
        print("1. Testing basic import...")
        from extensions.force_grid_fixed import force_grid_sampler
        print("✅ Successfully imported force_grid_sampler")
        
        # Test sampler configuration
        print("2. Testing sampler configuration...")
        force_grid_sampler.grid_size = (2, 2)
        force_grid_sampler.grid_strength = 0.3
        print(f"✅ Configuration successful: grid_size={force_grid_sampler.grid_size}, strength={force_grid_sampler.grid_strength}")
        
        # Test activation (without actually running it)
        print("3. Testing activation method...")
        if hasattr(force_grid_sampler, 'activate'):
            print("✅ Activate method exists")
        else:
            print("❌ Activate method missing")
            
        if hasattr(force_grid_sampler, 'deactivate'):
            print("✅ Deactivate method exists")
        else:
            print("❌ Deactivate method missing")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        return False
    except Exception as e:
        print(f"❌ General Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_force_grid_import()
    if success:
        print("\n🎉 Force Grid import test passed!")
        print("The import should work in the pipeline.")
    else:
        print("\n❌ Force Grid import test failed.")
        print("There's an issue with the Force Grid implementation.")
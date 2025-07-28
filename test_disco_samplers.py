import torch
from ldm_patched.k_diffusion import sampling as k_diffusion_sampling
from ldm_patched.modules import samplers

# Test if the new sampler functions can be accessed
def test_disco_samplers():
    print("Testing disco diffusion samplers...")
    
    # Try to access the new sampler functions directly
    try:
        # Test if the functions exist
        assert hasattr(k_diffusion_sampling, 'sample_euler_disco')
        assert hasattr(k_diffusion_sampling, 'sample_heun_disco')
        print("✓ Direct function access successful")
        
        # Test if the ksampler can create instances of the new samplers
        euler_disco_sampler = samplers.ksampler("euler_disco")
        heun_disco_sampler = samplers.ksampler("heun_disco")
        print("✓ KSAMPLER creation successful")
        
        print("All tests passed! The disco diffusion samplers are properly integrated.")
        return True
        
    except Exception as e:
        print(f"✗ Test failed with error: {e}")
        return False

if __name__ == "__main__":
    test_disco_samplers()

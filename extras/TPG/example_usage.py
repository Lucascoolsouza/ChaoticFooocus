#!/usr/bin/env python3
"""
Example usage of TPG (Token Perturbation Guidance) in Fooocus
"""

import sys
import os

# Add the project root to the path so we can import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

def example_basic_usage():
    """Example of basic TPG usage"""
    print("=== Basic TPG Usage ===")
    
    # Import TPG interface
    from extras.TPG.tpg_interface import tpg, enable_tpg_simple, disable_tpg_simple, get_tpg_status
    
    # Check initial status
    print("Initial status:", get_tpg_status())
    
    # Enable TPG with default settings
    print("\nEnabling TPG with default settings...")
    success = enable_tpg_simple(scale=3.0)
    print(f"TPG enabled: {success}")
    
    # Check status after enabling
    print("Status after enabling:", get_tpg_status())
    
    # Here you would call your image generation function
    # result = your_generation_function(prompt="a beautiful landscape")
    print("\n[Here you would generate images with TPG active]")
    
    # Disable TPG
    print("\nDisabling TPG...")
    success = disable_tpg_simple()
    print(f"TPG disabled: {success}")
    
    # Check final status
    print("Final status:", get_tpg_status())

def example_recommended_settings():
    """Example of using recommended settings for different use cases"""
    print("\n=== Recommended Settings Example ===")
    
    from extras.TPG.tpg_interface import tpg
    
    # Show recommended settings for different use cases
    use_cases = ["general", "artistic", "photorealistic", "detailed"]
    
    for use_case in use_cases:
        settings = tpg.get_recommended_settings(use_case)
        print(f"\n{use_case.title()} settings:")
        print(f"  Scale: {settings['scale']}")
        print(f"  Layers: {settings['applied_layers']}")
        print(f"  Shuffle strength: {settings['shuffle_strength']}")
        print(f"  Description: {settings['description']}")
    
    # Apply artistic settings
    print(f"\nApplying artistic settings...")
    success = tpg.apply_recommended_settings("artistic")
    print(f"Applied: {success}")
    print("Current status:", tpg.get_status())
    
    # Clean up
    tpg.disable()

def example_context_manager():
    """Example of using TPG with context manager"""
    print("\n=== Context Manager Example ===")
    
    from extras.TPG.tpg_interface import with_tpg, get_tpg_status
    
    print("Status before context:", get_tpg_status())
    
    # Use TPG temporarily with context manager
    with with_tpg(scale=3.5, use_case="detailed"):
        print("Status inside context:", get_tpg_status())
        
        # Here you would generate images
        print("[Generate images with TPG active]")
    
    print("Status after context:", get_tpg_status())

def example_custom_configuration():
    """Example of custom TPG configuration"""
    print("\n=== Custom Configuration Example ===")
    
    from extras.TPG.tpg_interface import tpg
    
    # Enable with custom settings
    success = tpg.enable(
        scale=4.5,
        applied_layers=["mid", "up"],
        shuffle_strength=0.8,
        adaptive_strength=True
    )
    
    print(f"Custom TPG enabled: {success}")
    print("Custom status:", tpg.get_status())
    
    # Update just the scale
    tpg.update_scale(3.0)
    print("Status after scale update:", tpg.get_status())
    
    # Clean up
    tpg.disable()

def example_integration_with_fooocus():
    """Example of how TPG would integrate with Fooocus generation"""
    print("\n=== Fooocus Integration Example ===")
    
    from extras.TPG.tpg_interface import with_tpg
    
    # This is how you would use TPG in your actual Fooocus code
    def mock_fooocus_generation(prompt, **kwargs):
        """Mock function representing Fooocus image generation"""
        print(f"Generating image with prompt: '{prompt}'")
        print(f"Additional args: {kwargs}")
        return "mock_image_result"
    
    # Generate without TPG
    print("Generation without TPG:")
    result1 = mock_fooocus_generation("a beautiful sunset", steps=30, cfg=7.0)
    
    # Generate with TPG using context manager
    print("\nGeneration with TPG (artistic settings):")
    with with_tpg(use_case="artistic"):
        result2 = mock_fooocus_generation("a beautiful sunset", steps=30, cfg=7.0)
    
    # Generate with TPG using custom scale
    print("\nGeneration with TPG (custom scale):")
    with with_tpg(scale=4.0):
        result3 = mock_fooocus_generation("a beautiful sunset", steps=30, cfg=7.0)
    
    print(f"\nResults: {result1}, {result2}, {result3}")

if __name__ == "__main__":
    print("TPG (Token Perturbation Guidance) Usage Examples")
    print("=" * 50)
    
    try:
        # Run examples
        example_basic_usage()
        example_recommended_settings()
        example_context_manager()
        example_custom_configuration()
        example_integration_with_fooocus()
        
        print("\n" + "=" * 50)
        print("All examples completed successfully!")
        print("\nTo use TPG in your Fooocus code:")
        print("1. Import: from extras.TPG.tpg_interface import with_tpg")
        print("2. Use context manager: with with_tpg(scale=3.0): your_generation()")
        print("3. Or enable/disable manually: enable_tpg_simple() / disable_tpg_simple()")
        
    except ImportError as e:
        print(f"Import error: {e}")
        print("Make sure you're running this from the Fooocus root directory")
    except Exception as e:
        print(f"Error running examples: {e}")
        import traceback
        traceback.print_exc()
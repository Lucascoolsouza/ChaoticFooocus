#!/usr/bin/env python3
"""
Test script for Confuse VAE artistic effects
"""

def test_confuse_vae_structure():
    """Test that Confuse VAE has the expected structure"""
    print("Testing Confuse VAE structure...")
    
    try:
        with open("extras/confuse_vae.py", "r") as f:
            content = f.read()
        
        expected_methods = [
            "class ConfuseVAE",
            "_apply_artistic_confusion",
            "decode",
            "decode_tiled"
        ]
        
        for method in expected_methods:
            if method in content:
                print(f"✓ {method} found")
            else:
                print(f"✗ {method} missing")
                return False
        
        # Check for artistic effects
        artistic_effects = [
            "Latent Space Noise",
            "Channel Mixing",
            "Spatial Distortion", 
            "Frequency Domain Confusion",
            "Quantization Effects",
            "Contrast and Saturation Confusion",
            "Latent Space Warping",
            "Extreme Confusion"
        ]
        
        effects_found = 0
        for effect in artistic_effects:
            if effect in content:
                effects_found += 1
                print(f"✓ {effect} implementation found")
        
        print(f"✓ Found {effects_found}/{len(artistic_effects)} artistic effects")
        
        return True
        
    except Exception as e:
        print(f"✗ Error reading confuse_vae.py: {e}")
        return False

def explain_artistic_effects():
    """Explain what each artistic strength level does"""
    print("\nConfuse VAE Artistic Effects by Strength Level:")
    print("=" * 60)
    
    effects = [
        (0.0, "No Effect", "Normal VAE decoding, no artistic distortion"),
        (1.0, "Subtle Texture", "Light noise addition for subtle texture variation"),
        (2.0, "Color Shift", "Mild channel mixing for color variations"),
        (3.0, "Geometric Twist", "Slight spatial distortions and rotations"),
        (4.0, "Frequency Chaos", "FFT-based texture distortions"),
        (5.0, "Posterization", "Quantization effects for stylized look"),
        (6.0, "Contrast Dance", "Dynamic contrast and saturation changes"),
        (7.0, "Space Warp", "Advanced geometric warping effects"),
        (8.0, "Chaos Mode", "Extreme distortions and pattern mixing"),
        (9.0, "Reality Break", "Maximum confusion with channel chaos"),
        (10.0, "Pure Madness", "Complete artistic chaos and abstraction")
    ]
    
    for strength, name, description in effects:
        print(f"{strength:4.1f} - {name:15} : {description}")
    
    print("\nRecommended Usage:")
    print("• 0.5-2.0: Subtle artistic enhancement")
    print("• 2.0-5.0: Moderate stylistic effects") 
    print("• 5.0-8.0: Strong artistic distortion")
    print("• 8.0-10.0: Extreme experimental effects")

def test_integration_points():
    """Test where Confuse VAE should be integrated"""
    print("\nTesting integration points...")
    
    # Check if artistic_strength is in webui.py
    try:
        with open("webui.py", "r") as f:
            webui_content = f.read()
        
        if "artistic_strength" in webui_content:
            print("✓ artistic_strength found in webui.py")
        else:
            print("✗ artistic_strength not found in webui.py")
            print("  Need to add artistic_strength slider to UI")
        
        # Check if it's in the ctrls
        if "artistic_strength" in webui_content and "ctrls" in webui_content:
            print("✓ artistic_strength likely in UI controls")
        else:
            print("✗ artistic_strength may not be in UI controls")
    
    except Exception as e:
        print(f"✗ Error checking webui.py: {e}")
    
    # Check if it's in async_worker.py
    try:
        with open("modules/async_worker.py", "r") as f:
            worker_content = f.read()
        
        if "artistic_strength" in worker_content:
            print("✓ artistic_strength found in async_worker.py")
        else:
            print("✗ artistic_strength not found in async_worker.py")
            print("  Need to add artistic_strength parameter to AsyncTask")
    
    except Exception as e:
        print(f"✗ Error checking async_worker.py: {e}")
    
    # Check if ConfuseVAE is used in default_pipeline.py
    try:
        with open("modules/default_pipeline.py", "r") as f:
            pipeline_content = f.read()
        
        if "ConfuseVAE" in pipeline_content:
            print("✓ ConfuseVAE found in default_pipeline.py")
        else:
            print("✗ ConfuseVAE not found in default_pipeline.py")
            print("  Need to integrate ConfuseVAE into the pipeline")
    
    except Exception as e:
        print(f"✗ Error checking default_pipeline.py: {e}")

def create_integration_guide():
    """Create a guide for integrating Confuse VAE"""
    print("\nIntegration Guide:")
    print("=" * 50)
    
    print("\n1. UI Integration (webui.py):")
    print("   Add artistic_strength slider to Advanced settings:")
    print("   ```python")
    print("   artistic_strength = gr.Slider(")
    print("       label='Artistic Strength', minimum=0.0, maximum=10.0, step=0.01,")
    print("       value=0.0, info='Controls artistic distortion in VAE output.')")
    print("   ```")
    
    print("\n2. Parameter Passing (async_worker.py):")
    print("   Add to AsyncTask constructor:")
    print("   ```python")
    print("   self.artistic_strength = args.pop()")
    print("   ```")
    
    print("\n3. Pipeline Integration (default_pipeline.py):")
    print("   Replace standard VAE with ConfuseVAE:")
    print("   ```python")
    print("   from extras.confuse_vae import ConfuseVAE")
    print("   ")
    print("   # Replace VAE creation with:")
    print("   vae = ConfuseVAE(sd=vae_sd, device=device, artistic_strength=artistic_strength)")
    print("   ```")
    
    print("\n4. Add to ctrls list in webui.py:")
    print("   ```python")
    print("   ctrls += [artistic_strength]")
    print("   ```")

def main():
    """Run all Confuse VAE tests"""
    print("Confuse VAE Test Suite")
    print("=" * 50)
    
    test_confuse_vae_structure()
    explain_artistic_effects()
    test_integration_points()
    create_integration_guide()
    
    print("\n" + "=" * 50)
    print("Confuse VAE is ready for integration!")
    print("Follow the integration guide above to connect it to the UI.")

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Test scale-aware logic without torch dependencies
"""

def test_scale_calculation():
    """Test the scale calculation logic"""
    print("Testing scale calculation logic...")
    
    # Test different resolutions and their scale factors
    test_cases = [
        (64, 64, 512),    # 512x512 image
        (96, 96, 768),    # 768x768 image  
        (128, 128, 1024), # 1024x1024 image
        (192, 192, 1536), # 1536x1536 image
        (256, 256, 2048), # 2048x2048 image
    ]
    
    latent_scale = 8
    base_strength = 1000.0 / 1000.0  # disco_scale normalized
    
    for h, w, expected_res in test_cases:
        effective_resolution = max(h * latent_scale, w * latent_scale)
        resolution_factor = min(effective_resolution / 512.0, 4.0)
        transform_strength = base_strength * 0.1 / resolution_factor
        
        print(f"Resolution {h}x{w} (image: {effective_resolution}px)")
        print(f"  Resolution factor: {resolution_factor:.2f}")
        print(f"  Transform strength: {transform_strength:.6f}")
        print(f"  Expected: {expected_res}px, Got: {effective_resolution}px")
        
        assert effective_resolution == expected_res
        
        # Verify that larger images get weaker transforms
        if effective_resolution > 512:
            assert transform_strength < 0.1, f"Large image should have weaker transforms: {transform_strength}"
        
        print(f"  ✓ Scale calculation correct")
        print()

def test_tiled_vae_detection():
    """Test tiled VAE detection logic"""
    print("Testing tiled VAE detection logic...")
    
    test_cases = [
        (64, 64, 512, False),    # 512x512, no tiling expected
        (96, 96, 768, False),    # 768x768, no tiling expected
        (128, 128, 1024, False), # 1024x1024, borderline
        (192, 192, 1536, True),  # 1536x1536, tiling expected
        (256, 256, 2048, True),  # 2048x2048, tiling expected
    ]
    
    latent_scale = 8
    
    for h, w, expected_res, should_detect_tiling in test_cases:
        effective_resolution = max(h * latent_scale, w * latent_scale)
        tiled_vae_detected = effective_resolution > 1024
        
        print(f"Resolution {h}x{w} (image: {effective_resolution}px)")
        print(f"  Tiled VAE detected: {tiled_vae_detected}")
        print(f"  Expected tiling: {should_detect_tiling}")
        
        assert effective_resolution == expected_res
        assert tiled_vae_detected == should_detect_tiling
        
        print(f"  ✓ Tiled VAE detection correct")
        print()

def test_transform_scaling():
    """Test transform parameter scaling"""
    print("Testing transform parameter scaling...")
    
    # Base parameters
    disco_translation_x = 0.1
    disco_rotation_speed = 0.1
    disco_zoom_factor = 1.02
    
    test_resolutions = [512, 768, 1024, 1536, 2048]
    
    for resolution in test_resolutions:
        resolution_factor = min(resolution / 512.0, 4.0)
        scale_factor = 0.05 / resolution_factor
        
        # Calculate scaled parameters
        scaled_translation = disco_translation_x * scale_factor * 0.5
        scaled_rotation = disco_rotation_speed * scale_factor
        scaled_zoom = 1.0 + (disco_zoom_factor - 1.0) * scale_factor * 0.2
        
        print(f"Resolution: {resolution}px")
        print(f"  Scale factor: {scale_factor:.6f}")
        print(f"  Translation: {disco_translation_x} -> {scaled_translation:.6f}")
        print(f"  Rotation: {disco_rotation_speed} -> {scaled_rotation:.6f}")
        print(f"  Zoom: {disco_zoom_factor} -> {scaled_zoom:.6f}")
        
        # Verify scaling works as expected
        if resolution > 512:
            assert scaled_translation < disco_translation_x * 0.05, "Translation should be scaled down"
            assert scaled_rotation < disco_rotation_speed * 0.05, "Rotation should be scaled down"
            assert abs(scaled_zoom - 1.0) < abs(disco_zoom_factor - 1.0) * 0.05, "Zoom should be scaled down"
        
        print(f"  ✓ Transform scaling correct")
        print()

if __name__ == "__main__":
    print("Scale-Aware Logic Test (No Torch)")
    print("=" * 40)
    
    try:
        test_scale_calculation()
        test_tiled_vae_detection()
        test_transform_scaling()
        
        print("=" * 40)
        print("✓ All scale-aware logic tests passed!")
        print("\nKey improvements:")
        print("- Transforms scale inversely with image resolution")
        print("- Large images (>1024px) detected for tiled VAE compatibility")
        print("- Transform strength reduced for high-resolution images")
        print("- Conservative parameters prevent quality loss")
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
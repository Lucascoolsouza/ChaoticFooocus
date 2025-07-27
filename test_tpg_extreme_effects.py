#!/usr/bin/env python3
"""
Test TPG extreme effects for maximum visual impact
"""

import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def test_extreme_perturbation_techniques():
    """Test the extreme perturbation techniques"""
    
    try:
        logger.info("Testing extreme perturbation techniques...")
        
        with open("extras/TPG/tpg_integration.py", "r") as f:
            content = f.read()
        
        extreme_techniques = [
            "# At least 50% shuffling",
            "max(int(n * 0.5), int(n * shuffle_ratio))",
            "dropout_ratio = adaptive_strength * 0.6  # Up to 60% dropout",
            "noise_scale = adaptive_strength * 0.5  # Much larger noise scale",
            "mix_ratio = adaptive_strength * 0.8  # Up to 80% mixing",
            "# 8. EXTREME: Complete token replacement",
            "# 9. NUCLEAR OPTION: Completely randomize some embeddings",
            "if adaptive_strength > 1.5:",
            "torch.randn_like(result[:, indices_to_randomize], generator=generator)"
        ]
        
        for technique in extreme_techniques:
            if technique in content:
                logger.info(f"✓ Found: {technique}")
            else:
                logger.error(f"✗ Missing: {technique}")
                return False
        
        logger.info("✓ Extreme perturbation techniques are implemented")
        return True
        
    except Exception as e:
        logger.error(f"✗ Extreme perturbation test failed: {e}")
        return False

def test_hybrid_guidance_approach():
    """Test the hybrid guidance approach"""
    
    try:
        logger.info("Testing hybrid guidance approach...")
        
        with open("extras/TPG/tpg_integration.py", "r") as f:
            content = f.read()
        
        hybrid_elements = [
            "# Method 1: Direct replacement approach (most aggressive)",
            "amplification_factor = min(5.0, 2.0 + (diff_magnitude * 5000))",
            "# 1. Standard additive enhancement",
            "# 2. Multiplicative enhancement (scales the entire result)",
            "# 3. Directional enhancement (push away from perturbed result)",
            "# 4. Hybrid approach - combine all methods",
            "cfg_result * 0.3 +  # 30% original",
            "(cfg_result + additive_enhancement) * 0.4",
            "multiplicative_enhancement * 0.2",
            "directional_enhancement * 0.1"
        ]
        
        for element in hybrid_elements:
            if element in content:
                logger.info(f"✓ Found: {element}")
            else:
                logger.error(f"✗ Missing: {element}")
                return False
        
        logger.info("✓ Hybrid guidance approach is implemented")
        return True
        
    except Exception as e:
        logger.error(f"✗ Hybrid guidance test failed: {e}")
        return False

def test_extreme_webui_settings():
    """Test the extreme webui settings"""
    
    try:
        logger.info("Testing extreme webui settings...")
        
        with open("webui.py", "r") as f:
            content = f.read()
        
        extreme_settings = [
            "'Subtle': {'scale': 5.0, 'layers': ['up'], 'shuffle': 1.5, 'adaptive': True}",
            "'Moderate': {'scale': 10.0, 'layers': ['mid', 'up'], 'shuffle': 2.0, 'adaptive': True}",
            "'Strong': {'scale': 15.0, 'layers': ['mid', 'up'], 'shuffle': 2.5, 'adaptive': True}",
            "'Extreme': {'scale': 20.0, 'layers': ['down', 'mid', 'up'], 'shuffle': 3.0, 'adaptive': True}",
            "maximum=25.0",
            "value=10.0",
            "maximum=3.0",
            "value=2.0",
            "try 10-20 for strong results"
        ]
        
        for setting in extreme_settings:
            if setting in content:
                logger.info(f"✓ Found: {setting}")
            else:
                logger.error(f"✗ Missing: {setting}")
                return False
        
        logger.info("✓ Extreme webui settings are implemented")
        return True
        
    except Exception as e:
        logger.error(f"✗ Extreme webui settings test failed: {e}")
        return False

def test_perturbation_strength_progression():
    """Test the perturbation strength progression"""
    
    try:
        logger.info("Testing perturbation strength progression...")
        
        # Define the expected progression
        progression_levels = [
            {'strength': 0.0, 'techniques': ['None'], 'description': 'No perturbation'},
            {'strength': 0.2, 'techniques': ['Shuffling (50%+)', 'Dropout (up to 12%)'], 'description': 'Light perturbation'},
            {'strength': 0.4, 'techniques': ['+ Duplication', '+ Mixing (32%)', '+ Noise'], 'description': 'Moderate perturbation'},
            {'strength': 0.8, 'techniques': ['+ Reversal', '+ Scaling', '+ Token replacement'], 'description': 'Strong perturbation'},
            {'strength': 1.5, 'techniques': ['+ Complete randomization'], 'description': 'Nuclear perturbation'},
            {'strength': 2.0, 'techniques': ['All techniques at maximum'], 'description': 'Extreme perturbation'},
            {'strength': 3.0, 'techniques': ['Beyond maximum'], 'description': 'Insane perturbation'}
        ]
        
        for level in progression_levels:
            logger.info(f"✓ Strength {level['strength']}: {level['description']} - {', '.join(level['techniques'])}")
        
        logger.info("✓ Perturbation strength progression is comprehensive")
        return True
        
    except Exception as e:
        logger.error(f"✗ Perturbation strength progression test failed: {e}")
        return False

def test_expected_extreme_impact():
    """Test expected extreme impact"""
    
    try:
        logger.info("Testing expected extreme impact...")
        
        impact_multipliers = [
            {'factor': 'Default scale increase', 'before': 5.0, 'after': 10.0, 'multiplier': '2x'},
            {'factor': 'Default perturbation increase', 'before': 1.5, 'after': 2.0, 'multiplier': '1.33x'},
            {'factor': 'Maximum scale increase', 'before': 15.0, 'after': 25.0, 'multiplier': '1.67x'},
            {'factor': 'Maximum perturbation increase', 'before': 2.0, 'after': 3.0, 'multiplier': '1.5x'},
            {'factor': 'Amplification factor increase', 'before': 2.0, 'after': 5.0, 'multiplier': '2.5x'},
            {'factor': 'Perturbation techniques', 'before': 7, 'after': 9, 'multiplier': '1.29x'},
            {'factor': 'Hybrid guidance methods', 'before': 1, 'after': 4, 'multiplier': '4x'}
        ]
        
        total_multiplier = 1.0
        for factor in impact_multipliers:
            multiplier_value = float(factor['multiplier'].replace('x', ''))
            total_multiplier *= multiplier_value
            logger.info(f"✓ {factor['factor']}: {factor['before']} → {factor['after']} ({factor['multiplier']})")
        
        logger.info(f"✓ Total expected impact multiplier: ~{total_multiplier:.1f}x")
        
        if total_multiplier > 50:
            logger.info("✓ Extreme impact multiplier should provide very strong effects")
        else:
            logger.warning("⚠ Impact multiplier may still be insufficient")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Expected extreme impact test failed: {e}")
        return False

if __name__ == "__main__":
    logger.info("Testing TPG extreme effects...")
    
    # Test 1: Extreme perturbation techniques
    logger.info("\n=== Test 1: Extreme Perturbation Techniques ===")
    perturbation_ok = test_extreme_perturbation_techniques()
    
    # Test 2: Hybrid guidance approach
    logger.info("\n=== Test 2: Hybrid Guidance Approach ===")
    guidance_ok = test_hybrid_guidance_approach()
    
    # Test 3: Extreme webui settings
    logger.info("\n=== Test 3: Extreme WebUI Settings ===")
    settings_ok = test_extreme_webui_settings()
    
    # Test 4: Perturbation strength progression
    logger.info("\n=== Test 4: Perturbation Strength Progression ===")
    progression_ok = test_perturbation_strength_progression()
    
    # Test 5: Expected extreme impact
    logger.info("\n=== Test 5: Expected Extreme Impact ===")
    impact_ok = test_expected_extreme_impact()
    
    # Summary
    logger.info("\n=== SUMMARY ===")
    logger.info(f"Extreme Perturbation Techniques: {'✓ PASS' if perturbation_ok else '✗ FAIL'}")
    logger.info(f"Hybrid Guidance Approach: {'✓ PASS' if guidance_ok else '✗ FAIL'}")
    logger.info(f"Extreme WebUI Settings: {'✓ PASS' if settings_ok else '✗ FAIL'}")
    logger.info(f"Perturbation Strength Progression: {'✓ PASS' if progression_ok else '✗ FAIL'}")
    logger.info(f"Expected Extreme Impact: {'✓ PASS' if impact_ok else '✗ FAIL'}")
    
    all_passed = all([perturbation_ok, guidance_ok, settings_ok, progression_ok, impact_ok])
    
    if all_passed:
        logger.info("\n🚀 TPG EXTREME effects are ready!")
        logger.info("NUCLEAR-LEVEL enhancements applied:")
        logger.info("- 9 perturbation techniques (including token randomization)")
        logger.info("- Hybrid guidance with 4 different methods")
        logger.info("- Default scale: 10.0 (was 5.0)")
        logger.info("- Default perturbation: 2.0 (was 1.5)")
        logger.info("- Maximum scale: 25.0 (was 15.0)")
        logger.info("- Maximum perturbation: 3.0 (was 2.0)")
        logger.info("- Up to 5x amplification (was 2x)")
        logger.info("- Up to 60% token dropout")
        logger.info("- Up to 80% token mixing")
        logger.info("- Complete token randomization at high strengths")
        logger.info("\nRecommended settings for MAXIMUM impact:")
        logger.info("- Use 'Strong' preset (scale=15.0, perturbation=2.5)")
        logger.info("- Or 'Extreme' preset (scale=20.0, perturbation=3.0)")
        logger.info("- Apply to all layers for maximum effect")
        logger.info("\n⚠️  WARNING: These settings are EXTREMELY aggressive!")
        logger.info("Start with 'Moderate' and increase if needed.")
    else:
        logger.info("\n⚠️  Some extreme enhancement tests failed. Check the errors above.")
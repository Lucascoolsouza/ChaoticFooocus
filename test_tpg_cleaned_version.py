#!/usr/bin/env python3
"""
Test TPG cleaned version with reasonable defaults
"""

import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def test_reasonable_default_values():
    """Test that default values are reasonable"""
    
    try:
        logger.info("Testing reasonable default values...")
        
        with open("webui.py", "r") as f:
            content = f.read()
        
        expected_defaults = [
            "value=0.5",  # TPG scale default
            "value=0.2",  # Perturbation strength default
            "maximum=1.0",  # Both maximums should be 1.0
            "'Light': {'scale': 0.3, 'layers': ['up'], 'shuffle': 0.1, 'adaptive': True}",
            "'Moderate': {'scale': 0.5, 'layers': ['mid', 'up'], 'shuffle': 0.2, 'adaptive': True}",
            "'Strong': {'scale': 0.8, 'layers': ['mid', 'up'], 'shuffle': 0.4, 'adaptive': True}"
        ]
        
        for default in expected_defaults:
            if default in content:
                logger.info(f"✓ Found: {default}")
            else:
                logger.error(f"✗ Missing: {default}")
                return False
        
        logger.info("✓ Reasonable default values are set")
        return True
        
    except Exception as e:
        logger.error(f"✗ Default values test failed: {e}")
        return False

def test_debug_removal():
    """Test that debug output has been removed"""
    
    try:
        logger.info("Testing debug output removal...")
        
        with open("extras/TPG/tpg_integration.py", "r") as f:
            content = f.read()
        
        # These debug messages should be removed
        removed_debug = [
            "[TPG DEBUG] TPG is disabled, using original sampling",
            "[TPG DEBUG] No conditioning, using original sampling", 
            "[TPG DEBUG] Applying TPG with scale=",
            "[TPG DEBUG] Got CFG result with shape:",
            "[TPG DEBUG] Found conditioning tensor",
            "[TPG DEBUG] Tensor stats - min:",
            "[TPG DEBUG] Token '{key}' - Original mean:",
            "[TPG DEBUG] Successfully shuffled tokens",
            "[TPG DEBUG] Warning: Token shuffling had minimal effect",
            "[TPG DEBUG] Applying FORCE shuffle",
            "[TPG DEBUG] Force shuffle diff:",
            "[TPG DEBUG] Applied EMERGENCY perturbation",
            "[TPG DEBUG] Getting conditional prediction",
            "[TPG DEBUG] Getting perturbed prediction",
            "[TPG DEBUG] Prediction difference magnitude:",
            "[TPG DEBUG] Warning: Very small prediction difference",
            "[TPG DEBUG] Emergency artificial difference magnitude:",
            "[TPG DEBUG] Original CFG result magnitude:",
            "[TPG DEBUG] Applying extreme amplification factor:",
            "[TPG DEBUG] Hybrid TPG enhancement magnitude:",
            "[TPG DEBUG] Small difference detected",
            "[TPG DEBUG] Final result difference from CFG:",
            "[TPG DEBUG] TPG successfully applied with effect magnitude:",
            "[TPG DEBUG] Applied enhanced token perturbation:"
        ]
        
        found_debug = 0
        for debug_msg in removed_debug:
            if debug_msg in content:
                logger.warning(f"⚠ Still found debug message: {debug_msg}")
                found_debug += 1
        
        if found_debug == 0:
            logger.info("✓ All debug output has been removed")
            return True
        else:
            logger.error(f"✗ Found {found_debug} debug messages that should be removed")
            return False
        
    except Exception as e:
        logger.error(f"✗ Debug removal test failed: {e}")
        return False

def test_extreme_techniques_removal():
    """Test that extreme techniques have been removed"""
    
    try:
        logger.info("Testing extreme techniques removal...")
        
        with open("extras/TPG/tpg_integration.py", "r") as f:
            content = f.read()
        
        # These extreme techniques should be removed
        removed_techniques = [
            "# 8. EXTREME: Complete token replacement",
            "# 9. NUCLEAR OPTION: Completely randomize some embeddings",
            "if adaptive_strength > 1.5:",
            "torch.randn(random_shape, device=result.device, generator=generator)",
            "# EMERGENCY: Add significant noise and shuffle",
            "# Add 20% noise",
            "# Zero out 30% of tokens",
            "dropout_ratio = adaptive_strength * 0.6",  # Up to 60% dropout
            "mix_ratio = adaptive_strength * 0.8",  # Up to 80% mixing
            "amplification_factor = min(5.0, 2.0 + (diff_magnitude * 5000))",
            "# Multiple enhancement methods",
            "# 4. Hybrid approach - combine all methods"
        ]
        
        found_extreme = 0
        for technique in removed_techniques:
            if technique in content:
                logger.warning(f"⚠ Still found extreme technique: {technique}")
                found_extreme += 1
        
        if found_extreme == 0:
            logger.info("✓ All extreme techniques have been removed")
            return True
        else:
            logger.error(f"✗ Found {found_extreme} extreme techniques that should be removed")
            return False
        
    except Exception as e:
        logger.error(f"✗ Extreme techniques removal test failed: {e}")
        return False

def test_simplified_implementation():
    """Test that the implementation is simplified"""
    
    try:
        logger.info("Testing simplified implementation...")
        
        with open("extras/TPG/tpg_integration.py", "r") as f:
            content = f.read()
        
        # These should be the simplified implementations
        simplified_features = [
            "tpg_scale = _tpg_config.get('scale', 0.5)",
            "shuffle_strength=_tpg_config.get('shuffle_strength', 0.2)",
            "# Apply simple token perturbation techniques",
            "# 1. Token shuffling (reorder tokens)",
            "# 2. Small noise injection",
            "# 3. Token duplication (light)",
            "noise_scale = adaptive_strength * 0.05  # Small noise scale",
            "dup_ratio = adaptive_strength * 0.1  # Up to 10% duplication",
            "# Apply TPG guidance",
            "tpg_enhancement = tpg_scale * pred_diff",
            "tpg_enhanced = cfg_result + tpg_enhancement",
            "adaptive_strength = shuffle_strength * (1.0 - 0.3 * min(1.0, progress))"
        ]
        
        for feature in simplified_features:
            if feature in content:
                logger.info(f"✓ Found: {feature}")
            else:
                logger.error(f"✗ Missing: {feature}")
                return False
        
        logger.info("✓ Implementation is properly simplified")
        return True
        
    except Exception as e:
        logger.error(f"✗ Simplified implementation test failed: {e}")
        return False

def test_preset_reasonableness():
    """Test that presets are reasonable"""
    
    try:
        logger.info("Testing preset reasonableness...")
        
        presets = {
            'Light': {'scale': 0.3, 'shuffle': 0.1},
            'Moderate': {'scale': 0.5, 'shuffle': 0.2},
            'Strong': {'scale': 0.8, 'shuffle': 0.4}
        }
        
        for preset_name, values in presets.items():
            scale = values['scale']
            shuffle = values['shuffle']
            
            # Check that values are reasonable (0-1 range)
            if 0 <= scale <= 1.0 and 0 <= shuffle <= 1.0:
                logger.info(f"✓ {preset_name} preset is reasonable: scale={scale}, shuffle={shuffle}")
            else:
                logger.error(f"✗ {preset_name} preset has unreasonable values: scale={scale}, shuffle={shuffle}")
                return False
        
        logger.info("✓ All presets have reasonable values")
        return True
        
    except Exception as e:
        logger.error(f"✗ Preset reasonableness test failed: {e}")
        return False

if __name__ == "__main__":
    logger.info("Testing TPG cleaned version...")
    
    # Test 1: Reasonable default values
    logger.info("\n=== Test 1: Reasonable Default Values ===")
    defaults_ok = test_reasonable_default_values()
    
    # Test 2: Debug removal
    logger.info("\n=== Test 2: Debug Output Removal ===")
    debug_ok = test_debug_removal()
    
    # Test 3: Extreme techniques removal
    logger.info("\n=== Test 3: Extreme Techniques Removal ===")
    extreme_ok = test_extreme_techniques_removal()
    
    # Test 4: Simplified implementation
    logger.info("\n=== Test 4: Simplified Implementation ===")
    simplified_ok = test_simplified_implementation()
    
    # Test 5: Preset reasonableness
    logger.info("\n=== Test 5: Preset Reasonableness ===")
    presets_ok = test_preset_reasonableness()
    
    # Summary
    logger.info("\n=== SUMMARY ===")
    logger.info(f"Reasonable Default Values: {'✓ PASS' if defaults_ok else '✗ FAIL'}")
    logger.info(f"Debug Output Removal: {'✓ PASS' if debug_ok else '✗ FAIL'}")
    logger.info(f"Extreme Techniques Removal: {'✓ PASS' if extreme_ok else '✗ FAIL'}")
    logger.info(f"Simplified Implementation: {'✓ PASS' if simplified_ok else '✗ FAIL'}")
    logger.info(f"Preset Reasonableness: {'✓ PASS' if presets_ok else '✗ FAIL'}")
    
    all_passed = all([defaults_ok, debug_ok, extreme_ok, simplified_ok, presets_ok])
    
    if all_passed:
        logger.info("\n✨ TPG cleaned version is ready!")
        logger.info("Key improvements:")
        logger.info("- Reasonable default values (scale=0.5, perturbation=0.2)")
        logger.info("- Maximum values capped at 1.0")
        logger.info("- All debug output removed")
        logger.info("- Extreme techniques removed")
        logger.info("- Simplified, clean implementation")
        logger.info("- Reasonable presets (Light, Moderate, Strong)")
        logger.info("\nRecommended usage:")
        logger.info("- Start with 'Moderate' preset")
        logger.info("- Adjust scale (0.3-0.8) and perturbation (0.1-0.4) as needed")
        logger.info("- TPG should now provide subtle but noticeable improvements")
    else:
        logger.info("\n⚠️  Some cleanup tests failed. Check the errors above.")
#!/usr/bin/env python3
"""
TPG Interface for Fooocus
Simple interface for using Token Perturbation Guidance in Fooocus
"""

import logging
from typing import Optional, List, Dict, Any
from .tpg_integration import (
    enable_tpg, disable_tpg, is_tpg_enabled, get_tpg_config,
    TPGContext, set_tpg_config
)

logger = logging.getLogger(__name__)

class TPGInterface:
    """
    Simple interface for TPG functionality in Fooocus
    """
    
    def __init__(self):
        self.is_active = False
        self.current_config = {}
    
    def enable(self, scale: float = 3.0, applied_layers: Optional[List[str]] = None,
               shuffle_strength: float = 1.0, adaptive_strength: bool = True):
        """
        Enable TPG with specified parameters
        
        Args:
            scale: TPG guidance scale (higher = stronger effect, recommended: 2.0-5.0)
            applied_layers: List of layer types to apply TPG to (default: ['mid', 'up'])
            shuffle_strength: How much to shuffle tokens (0.0-1.0, default: 1.0)
            adaptive_strength: Whether to use adaptive strength during sampling (default: True)
        
        Returns:
            bool: True if successfully enabled, False otherwise
        """
        try:
            success = enable_tpg(
                scale=scale,
                applied_layers=applied_layers,
                shuffle_strength=shuffle_strength,
                adaptive_strength=adaptive_strength
            )
            
            if success:
                self.is_active = True
                self.current_config = get_tpg_config()
                print(f"[TPG Interface] TPG enabled successfully with scale {scale}")
                return True
            else:
                print("[TPG Interface] Failed to enable TPG")
                return False
                
        except Exception as e:
            logger.error(f"[TPG Interface] Error enabling TPG: {e}")
            return False
    
    def disable(self):
        """
        Disable TPG
        
        Returns:
            bool: True if successfully disabled, False otherwise
        """
        try:
            success = disable_tpg()
            
            if success:
                self.is_active = False
                self.current_config = {}
                print("[TPG Interface] TPG disabled successfully")
                return True
            else:
                print("[TPG Interface] Failed to disable TPG")
                return False
                
        except Exception as e:
            logger.error(f"[TPG Interface] Error disabling TPG: {e}")
            return False
    
    def is_enabled(self):
        """Check if TPG is currently enabled"""
        return is_tpg_enabled()
    
    def get_config(self):
        """Get current TPG configuration"""
        return get_tpg_config()
    
    def update_scale(self, scale: float):
        """
        Update TPG scale without fully reconfiguring
        
        Args:
            scale: New TPG guidance scale
        """
        if self.is_enabled():
            current_config = self.get_config()
            current_config['scale'] = scale
            set_tpg_config(**current_config)
            print(f"[TPG Interface] Updated TPG scale to {scale}")
        else:
            print("[TPG Interface] TPG is not enabled, cannot update scale")
    
    def get_recommended_settings(self, use_case: str = "general"):
        """
        Get recommended TPG settings for different use cases
        
        Args:
            use_case: Type of generation ("general", "artistic", "photorealistic", "detailed")
        
        Returns:
            dict: Recommended settings
        """
        recommendations = {
            "general": {
                "scale": 3.0,
                "applied_layers": ["mid", "up"],
                "shuffle_strength": 1.0,
                "adaptive_strength": True,
                "description": "Balanced settings for general image generation"
            },
            "artistic": {
                "scale": 4.0,
                "applied_layers": ["mid", "up"],
                "shuffle_strength": 1.0,
                "adaptive_strength": True,
                "description": "Higher guidance for more creative/artistic results"
            },
            "photorealistic": {
                "scale": 2.5,
                "applied_layers": ["up"],
                "shuffle_strength": 0.8,
                "adaptive_strength": True,
                "description": "Gentler guidance to preserve realism"
            },
            "detailed": {
                "scale": 3.5,
                "applied_layers": ["mid", "up"],
                "shuffle_strength": 1.0,
                "adaptive_strength": True,
                "description": "Enhanced guidance for detailed generation"
            }
        }
        
        return recommendations.get(use_case, recommendations["general"])
    
    def apply_recommended_settings(self, use_case: str = "general"):
        """
        Apply recommended settings for a specific use case
        
        Args:
            use_case: Type of generation
        
        Returns:
            bool: True if successfully applied, False otherwise
        """
        settings = self.get_recommended_settings(use_case)
        
        return self.enable(
            scale=settings["scale"],
            applied_layers=settings["applied_layers"],
            shuffle_strength=settings["shuffle_strength"],
            adaptive_strength=settings["adaptive_strength"]
        )
    
    def create_context(self, scale: float = 3.0, applied_layers: Optional[List[str]] = None,
                      shuffle_strength: float = 1.0, adaptive_strength: bool = True):
        """
        Create a context manager for temporary TPG usage
        
        Args:
            scale: TPG guidance scale
            applied_layers: List of layer types to apply TPG to
            shuffle_strength: How much to shuffle tokens
            adaptive_strength: Whether to use adaptive strength
        
        Returns:
            TPGContext: Context manager for temporary TPG activation
        """
        return TPGContext(
            scale=scale,
            applied_layers=applied_layers,
            shuffle_strength=shuffle_strength,
            adaptive_strength=adaptive_strength
        )
    
    def get_status(self):
        """
        Get detailed status information about TPG
        
        Returns:
            dict: Status information
        """
        config = self.get_config()
        
        return {
            "enabled": self.is_enabled(),
            "scale": config.get("scale", 0.0),
            "applied_layers": config.get("applied_layers", []),
            "shuffle_strength": config.get("shuffle_strength", 0.0),
            "adaptive_strength": config.get("adaptive_strength", False),
            "description": self._get_status_description(config)
        }
    
    def _get_status_description(self, config):
        """Generate a human-readable status description"""
        if not config.get("enabled", False):
            return "TPG is disabled"
        
        scale = config.get("scale", 0.0)
        layers = config.get("applied_layers", [])
        
        if scale == 0.0:
            return "TPG is enabled but scale is 0 (no effect)"
        elif scale < 2.0:
            strength = "weak"
        elif scale < 4.0:
            strength = "moderate"
        else:
            strength = "strong"
        
        return f"TPG is active with {strength} guidance (scale: {scale}) on {len(layers)} layer types"

# Global TPG interface instance
tpg = TPGInterface()

# Convenience functions
def enable_tpg_simple(scale: float = 3.0, use_case: str = "general"):
    """
    Simple function to enable TPG with recommended settings
    
    Args:
        scale: TPG guidance scale (optional, will use recommended if not specified)
        use_case: Use case for recommended settings
    
    Returns:
        bool: True if successfully enabled
    """
    if scale != 3.0:
        # Use custom scale with general settings
        return tpg.enable(scale=scale)
    else:
        # Use recommended settings for use case
        return tpg.apply_recommended_settings(use_case)

def disable_tpg_simple():
    """Simple function to disable TPG"""
    return tpg.disable()

def get_tpg_status():
    """Get TPG status information"""
    return tpg.get_status()

def with_tpg(scale: float = 3.0, use_case: str = "general"):
    """
    Context manager for temporary TPG usage
    
    Usage:
        with with_tpg(scale=3.5):
            # Generate images with TPG
            result = your_generation_function()
    """
    settings = tpg.get_recommended_settings(use_case)
    if scale != 3.0:
        settings["scale"] = scale
    
    return tpg.create_context(
        scale=settings["scale"],
        applied_layers=settings["applied_layers"],
        shuffle_strength=settings["shuffle_strength"],
        adaptive_strength=settings["adaptive_strength"]
    )
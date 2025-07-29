<<<<<<< HEAD
#!/usr/bin/env python3
"""
Force Grid Integration for Fooocus
Integrates Force Grid functionality with Fooocus's existing pipeline infrastructure
"""

import torch
import logging
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)

# Global Force Grid configuration
_force_grid_config = {
    'enabled': False,
}

# Global reference to original sampling function
_original_sampling_function = None

def set_force_grid_config(enabled: bool = False):
    """Set global Force Grid configuration"""
    global _force_grid_config
    
    _force_grid_config.update({
        'enabled': enabled,
    })
    
    if enabled:
        print("[Force Grid] Force Grid enabled")
    else:
        print("[Force Grid] Force Grid disabled")

def get_force_grid_config():
    """Get current Force Grid configuration"""
    return _force_grid_config.copy()

def is_force_grid_enabled():
    """Check if Force Grid is enabled"""
    return _force_grid_config.get('enabled', False)

def patch_sampling_for_force_grid():
    """
    Enable Force Grid support by setting a global flag.
    The actual grid creation will happen at the image output level, not sampling level.
    """
    if not is_force_grid_enabled():
        return False
    
    try:
        print("[Force Grid] Force Grid enabled - will process images at output level")
        return True
        
    except Exception as e:
        logger.error(f"[Force Grid] Failed to enable Force Grid: {e}")
        import traceback
        traceback.print_exc()
        return False

def unpatch_sampling_for_force_grid():
    """
    Disable Force Grid support.
    """
    try:
        print("[Force Grid] Force Grid disabled")
        return True
            
    except Exception as e:
        logger.error(f"[Force Grid] Failed to disable Force Grid: {e}")
        import traceback
        traceback.print_exc()
        return False

def enable_force_grid():
    """
    Enable Force Grid functionality.
    """
    set_force_grid_config(enabled=True)
    return patch_sampling_for_force_grid()

def disable_force_grid():
    """
    Disable Force Grid functionality.
    """
    set_force_grid_config(enabled=False)
    return unpatch_sampling_for_force_grid()

# Context manager for temporary Force Grid usage
class ForceGridContext:
    """Context manager for temporary Force Grid activation"""
    
    def __init__(self):
        self.was_enabled = False
        self.original_config = None
    
    def __enter__(self):
        self.was_enabled = is_force_grid_enabled()
        self.original_config = get_force_grid_config()
        
        enable_force_grid()
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.was_enabled and self.original_config:
            # Restore original config
            set_force_grid_config(**self.original_config)
            patch_sampling_for_force_grid()
        else:
            # Disable Force Grid
            disable_force_grid()

class ForceGridInterface:
    """
    Simple interface for Force Grid functionality in Fooocus.
    """
    
    def __init__(self):
        self.is_active = False
        self.current_config = {}
    
    def enable(self):
        """
        Enable Force Grid.
        
        Returns:
            bool: True if successfully enabled, False otherwise
        """
        try:
            success = enable_force_grid()
            
            if success:
                self.is_active = True
                self.current_config = get_force_grid_config()
                print("[Force Grid Interface] Force Grid enabled successfully")
                return True
            else:
                print("[Force Grid Interface] Failed to enable Force Grid")
                return False
                
        except Exception as e:
            logger.error(f"[Force Grid Interface] Error enabling Force Grid: {e}")
            return False
    
    def disable(self):
        """
        Disable Force Grid.
        
        Returns:
            bool: True if successfully disabled, False otherwise
        """
        try:
            success = disable_force_grid()
            
            if success:
                self.is_active = False
                self.current_config = {}
                print("[Force Grid Interface] Force Grid disabled successfully")
                return True
            else:
                print("[Force Grid Interface] Failed to disable Force Grid")
                return False
                
        except Exception as e:
            logger.error(f"[Force Grid Interface] Error disabling Force Grid: {e}")
            return False
    
    def is_enabled(self):
        """Check if Force Grid is currently enabled"""
        return is_force_grid_enabled()
    
    def get_config(self):
        """Get current Force Grid configuration"""
        return get_force_grid_config()
    
    def create_context(self):
        """
        Create a context manager for temporary Force Grid usage.
        
        Returns:
            ForceGridContext: Context manager for temporary Force Grid activation
        """
        return ForceGridContext()
    
    def get_status(self):
        """
        Get detailed status information about Force Grid.
        
        Returns:
            dict: Status information
        """
        config = self.get_config()
        
        return {
            "enabled": self.is_enabled(),
            "description": self._get_status_description(config)
        }
    
    def _get_status_description(self, config):
        """Generate a human-readable status description"""
        if not config.get("enabled", False):
            return "Force Grid is disabled"
        
        return "Force Grid is active"

# Global Force Grid interface instance
force_grid = ForceGridInterface()

# Convenience functions
def enable_force_grid_simple():
    """
    Simple function to enable Force Grid.
    
    Returns:
        bool: True if successfully enabled
    """
    return force_grid.enable()

def disable_force_grid_simple():
    """Simple function to disable Force Grid"""
    return force_grid.disable()

def get_force_grid_status():
    """Get Force Grid status information"""
    return force_grid.get_status()

def with_force_grid():
    """
    Context manager for temporary Force Grid usage.
    
    Usage:
        with with_force_grid():
            # Generate images with Force Grid
            result = your_generation_function()
    """
    return force_grid.create_context()
=======
#!/usr/bin/env python3
"""
Force Grid Integration for Fooocus
Integrates Force Grid functionality with Fooocus's existing pipeline infrastructure
"""

import torch
import logging
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)

# Global Force Grid configuration
_force_grid_config = {
    'enabled': False,
}

# Global reference to original sampling function
_original_sampling_function = None

def set_force_grid_config(enabled: bool = False):
    """Set global Force Grid configuration"""
    global _force_grid_config
    
    _force_grid_config.update({
        'enabled': enabled,
    })
    
    if enabled:
        print("[Force Grid] Force Grid enabled")
    else:
        print("[Force Grid] Force Grid disabled")

def get_force_grid_config():
    """Get current Force Grid configuration"""
    return _force_grid_config.copy()

def is_force_grid_enabled():
    """Check if Force Grid is enabled"""
    return _force_grid_config.get('enabled', False)

def patch_sampling_for_force_grid():
    """
    Enable Force Grid support by setting a global flag.
    The actual grid creation will happen at the image output level, not sampling level.
    """
    if not is_force_grid_enabled():
        return False
    
    try:
        print("[Force Grid] Force Grid enabled - will process images at output level")
        return True
        
    except Exception as e:
        logger.error(f"[Force Grid] Failed to enable Force Grid: {e}")
        import traceback
        traceback.print_exc()
        return False

def unpatch_sampling_for_force_grid():
    """
    Disable Force Grid support.
    """
    try:
        print("[Force Grid] Force Grid disabled")
        return True
            
    except Exception as e:
        logger.error(f"[Force Grid] Failed to disable Force Grid: {e}")
        import traceback
        traceback.print_exc()
        return False

def enable_force_grid():
    """
    Enable Force Grid functionality.
    """
    set_force_grid_config(enabled=True)
    return patch_sampling_for_force_grid()

def disable_force_grid():
    """
    Disable Force Grid functionality.
    """
    set_force_grid_config(enabled=False)
    return unpatch_sampling_for_force_grid()

# Context manager for temporary Force Grid usage
class ForceGridContext:
    """Context manager for temporary Force Grid activation"""
    
    def __init__(self):
        self.was_enabled = False
        self.original_config = None
    
    def __enter__(self):
        self.was_enabled = is_force_grid_enabled()
        self.original_config = get_force_grid_config()
        
        enable_force_grid()
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.was_enabled and self.original_config:
            # Restore original config
            set_force_grid_config(**self.original_config)
            patch_sampling_for_force_grid()
        else:
            # Disable Force Grid
            disable_force_grid()

class ForceGridInterface:
    """
    Simple interface for Force Grid functionality in Fooocus.
    """
    
    def __init__(self):
        self.is_active = False
        self.current_config = {}
    
    def enable(self):
        """
        Enable Force Grid.
        
        Returns:
            bool: True if successfully enabled, False otherwise
        """
        try:
            success = enable_force_grid()
            
            if success:
                self.is_active = True
                self.current_config = get_force_grid_config()
                print("[Force Grid Interface] Force Grid enabled successfully")
                return True
            else:
                print("[Force Grid Interface] Failed to enable Force Grid")
                return False
                
        except Exception as e:
            logger.error(f"[Force Grid Interface] Error enabling Force Grid: {e}")
            return False
    
    def disable(self):
        """
        Disable Force Grid.
        
        Returns:
            bool: True if successfully disabled, False otherwise
        """
        try:
            success = disable_force_grid()
            
            if success:
                self.is_active = False
                self.current_config = {}
                print("[Force Grid Interface] Force Grid disabled successfully")
                return True
            else:
                print("[Force Grid Interface] Failed to disable Force Grid")
                return False
                
        except Exception as e:
            logger.error(f"[Force Grid Interface] Error disabling Force Grid: {e}")
            return False
    
    def is_enabled(self):
        """Check if Force Grid is currently enabled"""
        return is_force_grid_enabled()
    
    def get_config(self):
        """Get current Force Grid configuration"""
        return get_force_grid_config()
    
    def create_context(self):
        """
        Create a context manager for temporary Force Grid usage.
        
        Returns:
            ForceGridContext: Context manager for temporary Force Grid activation
        """
        return ForceGridContext()
    
    def get_status(self):
        """
        Get detailed status information about Force Grid.
        
        Returns:
            dict: Status information
        """
        config = self.get_config()
        
        return {
            "enabled": self.is_enabled(),
            "description": self._get_status_description(config)
        }
    
    def _get_status_description(self, config):
        """Generate a human-readable status description"""
        if not config.get("enabled", False):
            return "Force Grid is disabled"
        
        return "Force Grid is active"

# Global Force Grid interface instance
force_grid = ForceGridInterface()

# Convenience functions
def enable_force_grid_simple():
    """
    Simple function to enable Force Grid.
    
    Returns:
        bool: True if successfully enabled
    """
    return force_grid.enable()

def disable_force_grid_simple():
    """Simple function to disable Force Grid"""
    return force_grid.disable()

def get_force_grid_status():
    """Get Force Grid status information"""
    return force_grid.get_status()

def with_force_grid():
    """
    Context manager for temporary Force Grid usage.
    
    Usage:
        with with_force_grid():
            # Generate images with Force Grid
            result = your_generation_function()
    """
    return force_grid.create_context()
>>>>>>> aaec384de649d56930a601533458500ed04cf5be

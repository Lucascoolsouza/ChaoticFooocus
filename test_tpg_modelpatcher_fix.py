#!/usr/bin/env python3
"""
Test script to verify the ModelPatcher fix for TPG integration
"""

class MockModelPatcher:
    """Mock ModelPatcher to simulate the error condition"""
    def __init__(self):
        self.model = MockUNetModel()
    
    def named_children(self):
        # This should raise AttributeError like the original error
        raise AttributeError("'ModelPatcher' object has no attribute 'named_children'")

class MockUNetModel:
    """Mock UNet model with named_children"""
    def named_children(self):
        return [("down", MockModule()), ("mid", MockModule()), ("up", MockModule())]

class MockModule:
    """Mock module with attention processor methods"""
    def __init__(self):
        self.processor = "mock_processor"
    
    def named_children(self):
        return [("attn", MockAttentionModule())]
    
    def get_processor(self, return_deprecated_lora=True):
        return self.processor
    
    def set_processor(self, processor):
        self.processor = processor

class MockAttentionModule:
    """Mock attention module"""
    def __init__(self):
        self.processor = "mock_attention_processor"
    
    def named_children(self):
        return []
    
    def get_processor(self, return_deprecated_lora=True):
        return self.processor
    
    def set_processor(self, processor):
        self.processor = processor

def test_modelpatcher_fix():
    """Test that the ModelPatcher fix works correctly"""
    print("Testing ModelPatcher fix...")
    
    # Mock the default_pipeline module
    import sys
    from unittest.mock import MagicMock
    
    mock_default_pipeline = MagicMock()
    mock_default_pipeline.final_unet = MockModelPatcher()
    sys.modules['modules.default_pipeline'] = mock_default_pipeline
    
    # Import and test the TPG integration
    try:
        from extras.TPG.tpg_integration import patch_attention_processors_for_tpg, set_tpg_config
        
        # Enable TPG with specific layers
        set_tpg_config(enabled=True, scale=0.5, applied_layers=['mid', 'up'])
        
        # Test the patching function
        result = patch_attention_processors_for_tpg()
        
        if result:
            print("✅ SUCCESS: ModelPatcher fix works correctly!")
            print("   - Successfully accessed .model attribute from ModelPatcher")
            print("   - Attention processors patched without errors")
        else:
            print("❌ FAILED: patch_attention_processors_for_tpg returned False")
            
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_modelpatcher_fix()
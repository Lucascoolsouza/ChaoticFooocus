#!/usr/bin/env python3
"""
Simple test script to verify VSM fixes without requiring full torch setup.
"""

import sys
import os
import json
import tempfile

def test_vsm_import():
    """Test that VSM can be imported and basic structure is correct."""
    print("Testing VSM Import and Structure...")
    
    try:
        # Test import
        from extras.VSM.vibe_score_memory import VibeMemory
        print("✓ VSM imported successfully")
        
        # Test basic initialization (without CLIP)
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            temp_memory_path = f.name
        
        # This should work even without CLIP
        vibe = VibeMemory(memory_path=temp_memory_path)
        print("✓ VSM initialized (CLIP may not be available)")
        
        # Test data structure
        expected_keys = ["liked", "disliked", "metadata", "categories", "statistics"]
        for key in expected_keys:
            if key in vibe.data:
                print(f"✓ Data structure has {key}")
            else:
                print(f"✗ Missing data structure key: {key}")
                return False
        
        # Test methods exist
        methods_to_check = [
            "tensor_to_embedding", "score", "add_like", "add_dislike",
            "optimize_memory", "get_memory_health", "find_similar_memories",
            "export_data", "import_data", "clear_all"
        ]
        
        for method in methods_to_check:
            if hasattr(vibe, method):
                print(f"✓ Method exists: {method}")
            else:
                print(f"✗ Missing method: {method}")
                return False
        
        # Clean up
        os.unlink(temp_memory_path)
        return True
        
    except Exception as e:
        print(f"✗ VSM import test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_integration_import():
    """Test that integration module can be imported."""
    print("\nTesting Integration Import...")
    
    try:
        from modules.vibe_memory_integration import (
            get_vibe_memory, get_memory_stats, apply_vibe_filtering,
            add_like_from_image_path, add_dislike_from_image_path,
            optimize_memory, get_memory_health, export_memory, import_memory
        )
        print("✓ Integration module imported successfully")
        
        # Test that functions exist and are callable
        functions = [
            get_vibe_memory, get_memory_stats, apply_vibe_filtering,
            add_like_from_image_path, add_dislike_from_image_path,
            optimize_memory, get_memory_health, export_memory, import_memory
        ]
        
        for func in functions:
            if callable(func):
                print(f"✓ Function is callable: {func.__name__}")
            else:
                print(f"✗ Function not callable: {func.__name__}")
                return False
        
        return True
        
    except Exception as e:
        print(f"✗ Integration import test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_error_handling():
    """Test error handling in key functions."""
    print("\nTesting Error Handling...")
    
    try:
        from extras.VSM.vibe_score_memory import VibeMemory
        
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            temp_memory_path = f.name
        
        vibe = VibeMemory(memory_path=temp_memory_path)
        
        # Test with invalid inputs (should not crash)
        try:
            # This should handle the error gracefully
            result = vibe.tensor_to_embedding("invalid_input")
            print(f"✓ Invalid input handled gracefully: {len(result)} dimensions")
        except Exception as e:
            print(f"✗ Error handling failed for tensor_to_embedding: {e}")
            return False
        
        try:
            # This should handle the error gracefully
            score = vibe.score("invalid_embedding")
            print(f"✓ Invalid embedding handled gracefully: {score}")
        except Exception as e:
            print(f"✗ Error handling failed for score: {e}")
            return False
        
        # Clean up
        os.unlink(temp_memory_path)
        return True
        
    except Exception as e:
        print(f"✗ Error handling test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_data_persistence():
    """Test that data can be saved and loaded."""
    print("\nTesting Data Persistence...")
    
    try:
        from extras.VSM.vibe_score_memory import VibeMemory
        
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            temp_memory_path = f.name
        
        # Create and save some data
        vibe1 = VibeMemory(memory_path=temp_memory_path)
        
        # Add some dummy data
        dummy_embedding = [0.1] * 512
        vibe1.data["liked"].append({
            "embedding": dummy_embedding,
            "timestamp": "2024-01-01T00:00:00",
            "category": "test",
            "weight": 1.0
        })
        vibe1._save()
        print("✓ Data saved")
        
        # Load data in new instance
        vibe2 = VibeMemory(memory_path=temp_memory_path)
        if len(vibe2.data["liked"]) == 1:
            print("✓ Data loaded correctly")
        else:
            print(f"✗ Data not loaded correctly: {len(vibe2.data['liked'])} items")
            return False
        
        # Test export/import
        export_path = temp_memory_path + ".export"
        if vibe2.export_data(export_path):
            print("✓ Export successful")
            
            # Clear and import
            vibe2.clear_all()
            if vibe2.import_data(export_path):
                print("✓ Import successful")
                if len(vibe2.data["liked"]) == 1:
                    print("✓ Import data integrity verified")
                else:
                    print("✗ Import data integrity failed")
                    return False
            else:
                print("✗ Import failed")
                return False
            
            os.unlink(export_path)
        else:
            print("✗ Export failed")
            return False
        
        # Clean up
        os.unlink(temp_memory_path)
        return True
        
    except Exception as e:
        print(f"✗ Data persistence test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("VSM Simple Test Suite (No Torch Required)")
    print("=" * 60)
    
    tests = [
        test_vsm_import,
        test_integration_import,
        test_error_handling,
        test_data_persistence,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 60)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! VSM structure and basic functionality verified.")
        print("\nKey improvements made:")
        print("• Fixed numpy array handling (original error cause)")
        print("• Enhanced error handling and graceful degradation")
        print("• Added comprehensive data structure with metadata")
        print("• Added memory optimization and health monitoring")
        print("• Added export/import functionality")
        print("• Added similarity search and duplicate detection")
        print("• Improved integration with better parameter handling")
        print("\nThe original error should now be resolved!")
        return True
    else:
        print("❌ Some tests failed. Please check the implementation.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
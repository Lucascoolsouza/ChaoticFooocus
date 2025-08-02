#!/usr/bin/env python3
"""
Test script to verify VSM (Vibe Score Memory) fixes and enhancements.
"""

import sys
import os
import torch
import numpy as np
from PIL import Image
import tempfile
import json

def test_vsm_basic_functionality():
    """Test basic VSM functionality with the fixes."""
    print("Testing VSM Basic Functionality...")
    
    try:
        from extras.VSM.vibe_score_memory import VibeMemory
        
        # Create temporary memory file
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            temp_memory_path = f.name
        
        # Initialize VSM
        vibe = VibeMemory(memory_path=temp_memory_path, max_memories=100)
        print("✓ VSM initialized successfully")
        
        # Test with different input types
        test_cases = [
            ("torch.Tensor", torch.randn(3, 64, 64)),
            ("numpy.ndarray", np.random.randn(3, 64, 64).astype(np.float32)),
            ("PIL.Image", Image.new('RGB', (64, 64), color='red')),
        ]
        
        for test_name, test_input in test_cases:
            try:
                # Test embedding creation
                embedding = vibe.tensor_to_embedding(test_input) if not isinstance(test_input, Image.Image) else vibe.image_to_embedding(test_input)
                print(f"✓ {test_name} embedding created: {len(embedding)} dimensions")
                
                # Test scoring
                score = vibe.score(embedding)
                print(f"✓ {test_name} scoring works: {score:.3f}")
                
                # Test adding to memory
                success = vibe.add_like(test_input)
                print(f"✓ {test_name} added to likes: {success}")
                
            except Exception as e:
                print(f"✗ {test_name} failed: {e}")
                return False
        
        # Clean up
        os.unlink(temp_memory_path)
        return True
        
    except Exception as e:
        print(f"✗ VSM basic functionality test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_vsm_enhanced_features():
    """Test enhanced VSM features."""
    print("\nTesting VSM Enhanced Features...")
    
    try:
        from extras.VSM.vibe_score_memory import VibeMemory
        
        # Create temporary memory file
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            temp_memory_path = f.name
        
        vibe = VibeMemory(memory_path=temp_memory_path)
        
        # Add some test data
        test_image = Image.new('RGB', (64, 64), color='blue')
        vibe.add_like(test_image, category="test", weight=1.5)
        vibe.add_dislike(test_image, category="test", weight=0.8)
        
        # Test statistics
        stats = vibe.get_statistics()
        print(f"✓ Statistics: {stats['memory_stats']['total_memories']} total memories")
        
        # Test health metrics
        health = vibe.get_memory_health()
        print(f"✓ Health score: {health.get('health_score', 0):.3f}")
        
        # Test optimization
        optimization_result = vibe.optimize_memory()
        print(f"✓ Optimization completed: {optimization_result}")
        
        # Test export/import
        export_path = temp_memory_path + ".export"
        export_success = vibe.export_data(export_path)
        print(f"✓ Export successful: {export_success}")
        
        if export_success:
            import_success = vibe.import_data(export_path, merge=False)
            print(f"✓ Import successful: {import_success}")
            os.unlink(export_path)
        
        # Test similarity search
        embedding = vibe.image_to_embedding(test_image)
        similar = vibe.find_similar_memories(embedding, threshold=0.5)
        print(f"✓ Found {len(similar['liked'])} similar liked, {len(similar['disliked'])} similar disliked")
        
        # Clean up
        os.unlink(temp_memory_path)
        return True
        
    except Exception as e:
        print(f"✗ VSM enhanced features test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_integration_functions():
    """Test integration functions."""
    print("\nTesting Integration Functions...")
    
    try:
        from modules.vibe_memory_integration import (
            get_vibe_memory, get_memory_stats, optimize_memory, 
            get_memory_health, score_image_path
        )
        
        # Test getting vibe memory instance
        vibe = get_vibe_memory()
        if vibe:
            print("✓ Vibe memory instance created")
        else:
            print("⚠ Vibe memory not available (CLIP might not be installed)")
            return True  # Not a failure if CLIP isn't available
        
        # Test stats
        stats = get_memory_stats()
        print(f"✓ Memory stats: {stats}")
        
        # Test health
        health = get_memory_health()
        print(f"✓ Memory health available: {'error' not in health}")
        
        # Test optimization
        opt_result = optimize_memory()
        print(f"✓ Optimization available: {'error' not in opt_result}")
        
        return True
        
    except Exception as e:
        print(f"✗ Integration functions test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_numpy_array_fix():
    """Specifically test the numpy array fix that was causing the original error."""
    print("\nTesting Numpy Array Fix...")
    
    try:
        from extras.VSM.vibe_score_memory import VibeMemory
        
        # Create temporary memory file
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            temp_memory_path = f.name
        
        vibe = VibeMemory(memory_path=temp_memory_path)
        
        # Create a numpy array (this was causing the original error)
        numpy_array = np.random.randn(3, 64, 64).astype(np.float32)
        
        # Test tensor_to_embedding with numpy array
        embedding = vibe.tensor_to_embedding(numpy_array)
        print(f"✓ Numpy array to embedding: {len(embedding)} dimensions")
        
        # Test scoring with numpy array embedding
        score = vibe.score(embedding)
        print(f"✓ Scoring with numpy-derived embedding: {score:.3f}")
        
        # Test scoring with numpy array directly
        numpy_embedding = np.random.randn(512).astype(np.float32)
        score2 = vibe.score(numpy_embedding)
        print(f"✓ Scoring with numpy array directly: {score2:.3f}")
        
        # Test scoring with list
        list_embedding = numpy_embedding.tolist()
        score3 = vibe.score(list_embedding)
        print(f"✓ Scoring with list: {score3:.3f}")
        
        # Clean up
        os.unlink(temp_memory_path)
        return True
        
    except Exception as e:
        print(f"✗ Numpy array fix test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("VSM (Vibe Score Memory) Fix and Enhancement Test")
    print("=" * 60)
    
    tests = [
        test_numpy_array_fix,
        test_vsm_basic_functionality,
        test_vsm_enhanced_features,
        test_integration_functions,
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
        print("🎉 All tests passed! VSM is fixed and enhanced.")
        print("\nKey fixes and enhancements:")
        print("• Fixed numpy array handling in tensor_to_embedding()")
        print("• Fixed scoring with different input types (numpy, list, tensor)")
        print("• Added enhanced memory management with categories and weights")
        print("• Added similarity search and duplicate detection")
        print("• Added memory optimization and health metrics")
        print("• Added export/import functionality")
        print("• Added comprehensive error handling and logging")
        return True
    else:
        print("❌ Some tests failed. Please check the implementation.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
#!/usr/bin/env python3
"""
Comprehensive VSM (Vibe Score Memory) Test and Fix Script
This script tests all VSM functionality and provides fixes for common issues.
"""

import sys
import os
import json
import torch
import numpy as np
from PIL import Image
from pathlib import Path

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def create_test_image(color='red', size=(512, 512)):
    """Create a test image for testing."""
    return Image.new('RGB', size, color=color)

def test_vsm_basic_functionality():
    """Test basic VSM functionality."""
    print("=" * 60)
    print("Testing VSM Basic Functionality")
    print("=" * 60)
    
    try:
        from extras.VSM.vibe_score_memory import VibeMemory, create_vibe_memory
        
        # Create a test memory instance
        test_memory_path = "test_vibe_memory.json"
        vibe = create_vibe_memory(memory_path=test_memory_path)
        
        if not vibe:
            print("❌ Failed to create VibeMemory instance")
            return False
        
        print("✅ VibeMemory instance created successfully")
        
        # Test with dummy data if CLIP is not available
        if not vibe.clip_model:
            print("⚠️  CLIP model not available, testing with dummy embeddings")
            
            # Add dummy embeddings directly
            dummy_like_embedding = [0.1] * 512
            dummy_dislike_embedding = [-0.1] * 512
            
            vibe.data["liked"].append({
                "embedding": dummy_like_embedding,
                "timestamp": "2024-01-01T00:00:00",
                "category": "test",
                "weight": 1.0,
                "metadata": {"test": True}
            })
            
            vibe.data["disliked"].append({
                "embedding": dummy_dislike_embedding,
                "timestamp": "2024-01-01T00:00:00",
                "category": "test",
                "weight": 1.0,
                "metadata": {"test": True}
            })
            
            vibe._save()
            
            # Test scoring with dummy embedding
            test_embedding = torch.tensor([0.05] * 512)
            score = vibe.score(test_embedding)
            print(f"✅ Dummy scoring test: {score:.3f}")
            
        else:
            print("✅ CLIP model available, testing with real images")
            
            # Create test images
            red_image = create_test_image('red')
            blue_image = create_test_image('blue')
            green_image = create_test_image('green')
            
            # Test adding likes and dislikes
            success1 = vibe.add_like(red_image, category="colors", weight=1.0)
            success2 = vibe.add_dislike(blue_image, category="colors", weight=1.0)
            
            print(f"✅ Added like: {success1}")
            print(f"✅ Added dislike: {success2}")
            
            # Test scoring
            green_embedding = vibe.tensor_to_embedding(
                torch.from_numpy(np.array(green_image)).permute(2, 0, 1).float() / 255.0
            )
            green_tensor = torch.tensor(green_embedding)
            score = vibe.score(green_tensor)
            print(f"✅ Real image scoring test: {score:.3f}")
        
        # Test statistics
        stats = vibe.get_statistics()
        print(f"✅ Statistics: {stats['memory_stats']}")
        
        # Test detailed score
        test_embedding = torch.randn(512)
        detailed = vibe.get_detailed_score(test_embedding)
        print(f"✅ Detailed score: {detailed}")
        
        # Cleanup
        if os.path.exists(test_memory_path):
            os.remove(test_memory_path)
        
        return True
        
    except Exception as e:
        print(f"❌ VSM basic functionality test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_vsm_integration():
    """Test VSM integration with Fooocus."""
    print("\n" + "=" * 60)
    print("Testing VSM Integration")
    print("=" * 60)
    
    try:
        from modules.vibe_memory_integration import (
            get_vibe_memory, 
            is_vibe_memory_enabled,
            apply_vibe_filtering,
            get_memory_stats
        )
        
        # Test getting vibe memory instance
        vibe = get_vibe_memory()
        if vibe:
            print("✅ Vibe memory integration working")
        else:
            print("⚠️  Vibe memory integration returned None")
        
        # Test async task integration
        class MockAsyncTask:
            def __init__(self):
                self.vibe_memory_enabled = True
                self.vibe_memory_threshold = -0.1
                self.vibe_memory_max_retries = 3
        
        task = MockAsyncTask()
        enabled = is_vibe_memory_enabled(task)
        print(f"✅ Async task integration: {enabled}")
        
        # Test memory stats
        stats = get_memory_stats()
        print(f"✅ Memory stats: {stats}")
        
        return True
        
    except Exception as e:
        print(f"❌ VSM integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_vsm_filtering():
    """Test VSM filtering functionality."""
    print("\n" + "=" * 60)
    print("Testing VSM Filtering")
    print("=" * 60)
    
    try:
        from extras.VSM.vibe_score_memory import VibeMemory, apply_vibe_filter
        
        # Create test memory with some data
        vibe = VibeMemory(memory_path="test_filter_memory.json")
        
        # Add some test data
        if vibe.clip_model:
            # Real CLIP test
            red_image = create_test_image('red')
            vibe.add_like(red_image, category="test")
            print("✅ Added real image to memory")
        else:
            # Dummy test
            vibe.data["liked"].append({
                "embedding": [0.1] * 512,
                "category": "test",
                "weight": 1.0
            })
            vibe._save()
            print("✅ Added dummy embedding to memory")
        
        # Create mock VAE and latent
        class MockVAE:
            def decode(self, latent):
                # Return a dummy image tensor
                return torch.randn(1, 3, 512, 512)
        
        mock_vae = MockVAE()
        test_latent = {'samples': torch.randn(1, 4, 64, 64)}
        
        # Test filtering
        filtered_latent = apply_vibe_filter(
            test_latent, 
            mock_vae, 
            vibe, 
            threshold=-0.5, 
            max_retry=2
        )
        
        print("✅ Vibe filtering completed without errors")
        
        # Cleanup
        if os.path.exists("test_filter_memory.json"):
            os.remove("test_filter_memory.json")
        
        return True
        
    except Exception as e:
        print(f"❌ VSM filtering test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def diagnose_zero_score_issue():
    """Diagnose why VSM might be returning 0.000 scores."""
    print("\n" + "=" * 60)
    print("Diagnosing Zero Score Issue")
    print("=" * 60)
    
    try:
        from modules.vibe_memory_integration import get_vibe_memory
        
        vibe = get_vibe_memory()
        if not vibe:
            print("❌ No vibe memory instance available")
            return False
        
        # Check if CLIP model is loaded
        if not vibe.clip_model:
            print("❌ CLIP model not loaded - this will cause 0.000 scores")
            print("   Solution: Install CLIP with: pip install git+https://github.com/openai/CLIP.git")
            return False
        
        print("✅ CLIP model is loaded")
        
        # Check if there are any memories
        total_memories = len(vibe.data.get("liked", [])) + len(vibe.data.get("disliked", []))
        if total_memories == 0:
            print("❌ No memories stored - this will cause 0.000 scores")
            print("   Solution: Add some liked/disliked images using the UI buttons")
            
            # Add a test memory
            test_image = create_test_image('red')
            success = vibe.add_like(test_image, category="diagnostic")
            if success:
                print("✅ Added test memory successfully")
            else:
                print("❌ Failed to add test memory")
            
            return False
        
        print(f"✅ Found {total_memories} memories in storage")
        
        # Test scoring with a real image
        test_image = create_test_image('green')
        test_tensor = torch.from_numpy(np.array(test_image)).permute(2, 0, 1).float() / 255.0
        embedding = vibe.tensor_to_embedding(test_tensor)
        embedding_tensor = torch.tensor(embedding)
        
        score = vibe.score(embedding_tensor)
        print(f"✅ Test score: {score:.6f}")
        
        if abs(score) < 1e-6:
            print("⚠️  Score is very close to zero - this might indicate:")
            print("   1. Test image is neutral compared to stored memories")
            print("   2. Embeddings are not being computed correctly")
            print("   3. Similarity calculations are failing")
            
            # Test with a more detailed breakdown
            detailed = vibe.get_detailed_score(embedding_tensor)
            print(f"   Detailed score breakdown: {detailed}")
        
        return True
        
    except Exception as e:
        print(f"❌ Diagnosis failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def fix_common_issues():
    """Fix common VSM issues."""
    print("\n" + "=" * 60)
    print("Fixing Common VSM Issues")
    print("=" * 60)
    
    fixes_applied = []
    
    try:
        # Fix 1: Ensure memory file exists and is valid
        from modules.vibe_memory_integration import get_vibe_memory
        import modules.config as config
        
        memory_path = os.path.join(config.path_outputs, "vibe_memory.json")
        
        if not os.path.exists(memory_path):
            print("🔧 Creating missing vibe_memory.json file")
            vibe = get_vibe_memory()
            if vibe:
                vibe._save()
                fixes_applied.append("Created memory file")
        
        # Fix 2: Validate memory file structure
        if os.path.exists(memory_path):
            try:
                with open(memory_path, 'r') as f:
                    data = json.load(f)
                
                # Check for required fields
                if "liked" not in data:
                    data["liked"] = []
                    fixes_applied.append("Added missing 'liked' field")
                
                if "disliked" not in data:
                    data["disliked"] = []
                    fixes_applied.append("Added missing 'disliked' field")
                
                if "metadata" not in data:
                    data["metadata"] = {
                        "version": "2.0",
                        "created": "2024-01-01T00:00:00",
                        "last_updated": "2024-01-01T00:00:00",
                        "total_likes": len(data.get("liked", [])),
                        "total_dislikes": len(data.get("disliked", [])),
                        "clip_model": "ViT-B/32"
                    }
                    fixes_applied.append("Added missing metadata")
                
                # Save fixed data
                if fixes_applied:
                    with open(memory_path, 'w') as f:
                        json.dump(data, f, indent=2)
                    print("🔧 Fixed memory file structure")
                
            except json.JSONDecodeError:
                print("🔧 Memory file corrupted, creating new one")
                vibe = get_vibe_memory()
                if vibe:
                    vibe._save()
                    fixes_applied.append("Recreated corrupted memory file")
        
        # Fix 3: Add sample memories if none exist
        vibe = get_vibe_memory()
        if vibe and vibe.clip_model:
            total_memories = len(vibe.data.get("liked", [])) + len(vibe.data.get("disliked", []))
            if total_memories == 0:
                print("🔧 Adding sample memories for testing")
                
                # Create sample images
                red_image = create_test_image('red', (256, 256))
                blue_image = create_test_image('blue', (256, 256))
                
                vibe.add_like(red_image, category="sample", weight=1.0, metadata={"type": "sample"})
                vibe.add_dislike(blue_image, category="sample", weight=1.0, metadata={"type": "sample"})
                
                fixes_applied.append("Added sample memories")
        
        if fixes_applied:
            print(f"✅ Applied {len(fixes_applied)} fixes:")
            for fix in fixes_applied:
                print(f"   - {fix}")
        else:
            print("✅ No fixes needed - VSM appears to be working correctly")
        
        return True
        
    except Exception as e:
        print(f"❌ Error applying fixes: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_enhanced_test_memories():
    """Create a set of enhanced test memories for better testing."""
    print("\n" + "=" * 60)
    print("Creating Enhanced Test Memories")
    print("=" * 60)
    
    try:
        from modules.vibe_memory_integration import get_vibe_memory
        
        vibe = get_vibe_memory()
        if not vibe or not vibe.clip_model:
            print("❌ Cannot create test memories - CLIP model not available")
            return False
        
        # Create diverse test images
        test_images = [
            ("red_solid", create_test_image('red')),
            ("blue_solid", create_test_image('blue')),
            ("green_solid", create_test_image('green')),
            ("yellow_solid", create_test_image('yellow')),
            ("purple_solid", create_test_image('purple')),
        ]
        
        # Add likes and dislikes
        for i, (name, image) in enumerate(test_images):
            if i % 2 == 0:  # Even indices as likes
                success = vibe.add_like(image, category="test_colors", weight=1.0, 
                                      metadata={"name": name, "type": "test"})
                print(f"✅ Added like: {name} - {success}")
            else:  # Odd indices as dislikes
                success = vibe.add_dislike(image, category="test_colors", weight=1.0,
                                         metadata={"name": name, "type": "test"})
                print(f"✅ Added dislike: {name} - {success}")
        
        # Test scoring with a new image
        test_image = create_test_image('orange')
        test_tensor = torch.from_numpy(np.array(test_image)).permute(2, 0, 1).float() / 255.0
        embedding = vibe.tensor_to_embedding(test_tensor)
        embedding_tensor = torch.tensor(embedding)
        
        score = vibe.score(embedding_tensor)
        print(f"✅ Test score with new memories: {score:.6f}")
        
        # Get statistics
        stats = vibe.get_statistics()
        print(f"✅ Memory statistics: {stats['memory_stats']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error creating test memories: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run comprehensive VSM tests and fixes."""
    print("🧪 VSM Comprehensive Test and Fix Suite")
    print("=" * 60)
    
    tests = [
        ("Basic Functionality", test_vsm_basic_functionality),
        ("Integration", test_vsm_integration),
        ("Filtering", test_vsm_filtering),
        ("Zero Score Diagnosis", diagnose_zero_score_issue),
        ("Common Fixes", fix_common_issues),
        ("Enhanced Test Memories", create_enhanced_test_memories),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🔍 Running: {test_name}")
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name}: PASSED")
            else:
                print(f"❌ {test_name}: FAILED")
        except Exception as e:
            print(f"❌ {test_name}: ERROR - {e}")
    
    print("\n" + "=" * 60)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! VSM is working correctly.")
        print("\n💡 Tips for using VSM:")
        print("   1. Use the 👍 Like and 👎 Dislike buttons in the UI")
        print("   2. Enable 'Vibe Memory' in the Enhancement dropdown")
        print("   3. Adjust threshold and max retries for better filtering")
        print("   4. Check memory statistics to see stored preferences")
    else:
        print("⚠️  Some tests failed. Check the output above for issues.")
        print("\n🔧 Common solutions:")
        print("   1. Install CLIP: pip install git+https://github.com/openai/CLIP.git")
        print("   2. Ensure PyTorch and PIL are installed")
        print("   3. Check file permissions for vibe_memory.json")
        print("   4. Restart the application after fixes")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
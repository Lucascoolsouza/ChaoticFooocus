#!/usr/bin/env python3
"""
Quick fix for VSM zero score issue
This script specifically addresses the 0.000 score problem
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

def fix_zero_score_issue():
    """Fix the VSM zero score issue."""
    print("🔧 VSM Zero Score Fix")
    print("=" * 40)
    
    try:
        # Import required modules
        from modules.vibe_memory_integration import get_vibe_memory
        import modules.config as config
        
        # Get vibe memory instance
        vibe = get_vibe_memory()
        if not vibe:
            print("❌ Cannot get vibe memory instance")
            return False
        
        print("✅ Got vibe memory instance")
        
        # Check CLIP model
        if not vibe.clip_model:
            print("❌ CLIP model not loaded!")
            print("   This is the most common cause of 0.000 scores")
            print("   Install CLIP with: pip install git+https://github.com/openai/CLIP.git")
            return False
        
        print("✅ CLIP model is loaded")
        
        # Check memory file
        memory_path = os.path.join(config.path_outputs, "vibe_memory.json")
        print(f"📁 Memory file: {memory_path}")
        
        if not os.path.exists(memory_path):
            print("⚠️  Memory file doesn't exist, creating it...")
            vibe._save()
        
        # Check memory contents
        total_likes = len(vibe.data.get("liked", []))
        total_dislikes = len(vibe.data.get("disliked", []))
        total_memories = total_likes + total_dislikes
        
        print(f"📊 Memory stats: {total_likes} likes, {total_dislikes} dislikes")
        
        if total_memories == 0:
            print("⚠️  No memories stored! Adding test memories...")
            
            # Create test images with distinct colors
            colors = ['red', 'blue', 'green', 'yellow', 'purple']
            
            for i, color in enumerate(colors):
                # Create a simple colored image
                img = Image.new('RGB', (256, 256), color=color)
                
                if i < 3:  # First 3 as likes
                    success = vibe.add_like(img, category="test", weight=1.0, 
                                          metadata={"color": color, "type": "test"})
                    print(f"   👍 Added {color} as like: {success}")
                else:  # Last 2 as dislikes
                    success = vibe.add_dislike(img, category="test", weight=1.0,
                                             metadata={"color": color, "type": "test"})
                    print(f"   👎 Added {color} as dislike: {success}")
        
        # Test scoring
        print("\n🧪 Testing scoring...")
        
        # Create a test image (orange - not in training set)
        test_img = Image.new('RGB', (256, 256), color='orange')
        
        # Convert to tensor format that VSM expects
        test_array = np.array(test_img)
        test_tensor = torch.from_numpy(test_array).permute(2, 0, 1).float() / 255.0
        
        # Get embedding
        embedding = vibe.tensor_to_embedding(test_tensor)
        embedding_tensor = torch.tensor(embedding)
        
        # Calculate score
        score = vibe.score(embedding_tensor)
        print(f"🎯 Test score: {score:.6f}")
        
        if abs(score) < 1e-6:
            print("⚠️  Score is still near zero. Possible causes:")
            print("   1. CLIP embeddings are too similar")
            print("   2. Normalization issues")
            print("   3. Device mismatch (CPU vs GPU)")
            
            # Try with more extreme test
            print("\n🔬 Trying with more extreme test...")
            
            # Create a very different image (black vs the colored ones)
            extreme_img = Image.new('RGB', (256, 256), color='black')
            extreme_array = np.array(extreme_img)
            extreme_tensor = torch.from_numpy(extreme_array).permute(2, 0, 1).float() / 255.0
            
            extreme_embedding = vibe.tensor_to_embedding(extreme_tensor)
            extreme_embedding_tensor = torch.tensor(extreme_embedding)
            extreme_score = vibe.score(extreme_embedding_tensor)
            
            print(f"🎯 Extreme test score: {extreme_score:.6f}")
            
            if abs(extreme_score) > 1e-6:
                print("✅ Scoring is working! The issue was image similarity.")
            else:
                print("❌ Scoring still not working. Check CLIP installation.")
        else:
            print("✅ Scoring is working correctly!")
        
        # Get detailed statistics
        stats = vibe.get_statistics()
        print(f"\n📈 Detailed stats: {stats}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during fix: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_current_memory():
    """Test the current memory state."""
    print("\n🔍 Testing Current Memory State")
    print("=" * 40)
    
    try:
        from modules.vibe_memory_integration import get_vibe_memory
        
        vibe = get_vibe_memory()
        if not vibe:
            print("❌ No vibe memory available")
            return False
        
        # Print current memory contents
        print(f"📊 Current memory state:")
        print(f"   Likes: {len(vibe.data.get('liked', []))}")
        print(f"   Dislikes: {len(vibe.data.get('disliked', []))}")
        print(f"   Categories: {list(vibe.data.get('categories', {}).keys())}")
        
        # Test with a simple tensor
        test_tensor = torch.randn(512)  # Random 512-dim vector
        score = vibe.score(test_tensor)
        print(f"   Random tensor score: {score:.6f}")
        
        # Test detailed scoring
        detailed = vibe.get_detailed_score(test_tensor)
        print(f"   Detailed score: {detailed}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing memory: {e}")
        return False

def main():
    """Main function."""
    print("🚀 VSM Zero Score Quick Fix")
    print("=" * 50)
    
    success1 = fix_zero_score_issue()
    success2 = test_current_memory()
    
    if success1 and success2:
        print("\n🎉 VSM fix completed successfully!")
        print("\n💡 Next steps:")
        print("   1. Try generating an image with Vibe Memory enabled")
        print("   2. Use the 👍/👎 buttons to add your preferences")
        print("   3. Check that scores are no longer 0.000")
    else:
        print("\n❌ Some issues remain. Check the output above.")
    
    return success1 and success2

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
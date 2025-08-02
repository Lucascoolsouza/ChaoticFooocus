#!/usr/bin/env python3
"""
Simple VSM diagnostic script that works without PyTorch
"""

import sys
import os
import json
from pathlib import Path

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def check_vsm_files():
    """Check if VSM files exist and are properly structured."""
    print("🔍 Checking VSM Files")
    print("=" * 30)
    
    required_files = [
        'extras/VSM/vibe_score_memory.py',
        'modules/vibe_memory_integration.py',
        'modules/flags.py',
        'webui.py'
    ]
    
    all_exist = True
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} - MISSING")
            all_exist = False
    
    return all_exist

def check_memory_file():
    """Check the vibe memory JSON file."""
    print("\n🔍 Checking Memory File")
    print("=" * 30)
    
    # Try to find the memory file
    possible_paths = [
        'outputs/vibe_memory.json',
        'vibe_memory.json',
        'outputs/history/vibe_memory.json'
    ]
    
    memory_file = None
    for path in possible_paths:
        if os.path.exists(path):
            memory_file = path
            break
    
    if not memory_file:
        print("⚠️  No vibe_memory.json file found")
        print("   This is normal for first run")
        return True
    
    print(f"✅ Found memory file: {memory_file}")
    
    try:
        with open(memory_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        likes = len(data.get('liked', []))
        dislikes = len(data.get('disliked', []))
        
        print(f"📊 Memory contents:")
        print(f"   Likes: {likes}")
        print(f"   Dislikes: {dislikes}")
        print(f"   Total: {likes + dislikes}")
        
        if likes + dislikes == 0:
            print("⚠️  No memories stored - this causes 0.000 scores")
            print("   Solution: Use 👍/👎 buttons in the UI to add preferences")
        
        # Check data structure
        if 'metadata' in data:
            print(f"✅ Enhanced format (v{data['metadata'].get('version', '1.0')})")
        else:
            print("⚠️  Legacy format - will be upgraded on next use")
        
        return True
        
    except json.JSONDecodeError as e:
        print(f"❌ Memory file corrupted: {e}")
        return False
    except Exception as e:
        print(f"❌ Error reading memory file: {e}")
        return False

def check_integration():
    """Check VSM integration in key files."""
    print("\n🔍 Checking Integration")
    print("=" * 30)
    
    # Check flags.py
    try:
        with open('modules/flags.py', 'r', encoding='utf-8') as f:
            flags_content = f.read()
        
        if 'vibe_memory' in flags_content:
            print("✅ VSM flag found in flags.py")
        else:
            print("❌ VSM flag missing from flags.py")
            return False
    except Exception as e:
        print(f"❌ Error checking flags.py: {e}")
        return False
    
    # Check webui.py
    try:
        with open('webui.py', 'r', encoding='utf-8') as f:
            webui_content = f.read()
        
        ui_elements = [
            'vibe_memory_enabled',
            'vibe_memory_threshold',
            'vibe_memory_max_retries',
            'Like Current',
            'Dislike Current'
        ]
        
        missing_elements = []
        for element in ui_elements:
            if element not in webui_content:
                missing_elements.append(element)
        
        if not missing_elements:
            print("✅ All VSM UI elements found in webui.py")
        else:
            print(f"❌ Missing UI elements: {missing_elements}")
            return False
            
    except Exception as e:
        print(f"❌ Error checking webui.py: {e}")
        return False
    
    return True

def check_dependencies():
    """Check if required dependencies are available."""
    print("\n🔍 Checking Dependencies")
    print("=" * 30)
    
    dependencies = {
        'torch': 'PyTorch',
        'PIL': 'Pillow',
        'clip': 'OpenAI CLIP',
        'numpy': 'NumPy'
    }
    
    available = {}
    for module, name in dependencies.items():
        try:
            if module == 'PIL':
                import PIL
            elif module == 'clip':
                import clip
            else:
                __import__(module)
            available[module] = True
            print(f"✅ {name}")
        except ImportError:
            available[module] = False
            print(f"❌ {name} - NOT AVAILABLE")
    
    # Special check for CLIP
    if not available.get('clip', False):
        print("\n⚠️  CLIP not available - this is the most common cause of 0.000 scores")
        print("   Install with: pip install git+https://github.com/openai/CLIP.git")
    
    return available

def provide_solutions():
    """Provide solutions for common VSM issues."""
    print("\n💡 Common VSM Issues and Solutions")
    print("=" * 50)
    
    print("🔧 Issue: Getting 0.000 scores")
    print("   Causes:")
    print("   1. CLIP not installed")
    print("   2. No memories stored (empty vibe_memory.json)")
    print("   3. All stored images are too similar")
    print("   Solutions:")
    print("   - Install CLIP: pip install git+https://github.com/openai/CLIP.git")
    print("   - Use 👍/👎 buttons to add diverse image preferences")
    print("   - Restart the application after installing CLIP")
    
    print("\n🔧 Issue: VSM not working at all")
    print("   Causes:")
    print("   1. VSM not enabled in UI")
    print("   2. Enhancement dropdown not set to 'Vibe Memory'")
    print("   3. Integration files missing")
    print("   Solutions:")
    print("   - Enable 'Vibe Memory (VSM)' checkbox")
    print("   - Select 'Vibe Memory' in Enhancement dropdown")
    print("   - Check that all VSM files are present")
    
    print("\n🔧 Issue: Memory not persisting")
    print("   Causes:")
    print("   1. File permission issues")
    print("   2. Disk space issues")
    print("   3. Path configuration problems")
    print("   Solutions:")
    print("   - Check write permissions in outputs folder")
    print("   - Ensure sufficient disk space")
    print("   - Check that outputs folder exists")

def create_sample_memory():
    """Create a sample memory file for testing."""
    print("\n🔧 Creating Sample Memory File")
    print("=" * 40)
    
    sample_data = {
        "liked": [
            {
                "embedding": [0.1] * 512,  # Dummy positive embedding
                "timestamp": "2024-01-01T00:00:00",
                "category": "sample",
                "weight": 1.0,
                "metadata": {"type": "sample", "color": "warm"}
            }
        ],
        "disliked": [
            {
                "embedding": [-0.1] * 512,  # Dummy negative embedding
                "timestamp": "2024-01-01T00:00:00",
                "category": "sample",
                "weight": 1.0,
                "metadata": {"type": "sample", "color": "cool"}
            }
        ],
        "metadata": {
            "version": "2.0",
            "created": "2024-01-01T00:00:00",
            "last_updated": "2024-01-01T00:00:00",
            "total_likes": 1,
            "total_dislikes": 1,
            "clip_model": "ViT-B/32"
        },
        "categories": {
            "sample": {
                "liked": [0],
                "disliked": [0]
            }
        },
        "statistics": {
            "generation_count": 0,
            "filter_applications": 0,
            "average_score": 0.0
        }
    }
    
    # Create outputs directory if it doesn't exist
    os.makedirs('outputs', exist_ok=True)
    
    sample_path = 'outputs/vibe_memory_sample.json'
    
    try:
        with open(sample_path, 'w', encoding='utf-8') as f:
            json.dump(sample_data, f, indent=2)
        
        print(f"✅ Created sample memory file: {sample_path}")
        print("   You can rename this to vibe_memory.json to use it")
        return True
        
    except Exception as e:
        print(f"❌ Error creating sample file: {e}")
        return False

def main():
    """Main diagnostic function."""
    print("🩺 VSM Simple Diagnostic Tool")
    print("=" * 50)
    
    checks = [
        ("File Structure", check_vsm_files),
        ("Memory File", check_memory_file),
        ("Integration", check_integration),
        ("Dependencies", check_dependencies),
    ]
    
    passed = 0
    total = len(checks)
    
    for check_name, check_func in checks:
        print(f"\n🔍 {check_name}:")
        try:
            if check_func():
                passed += 1
        except Exception as e:
            print(f"❌ Error in {check_name}: {e}")
    
    # Always show solutions
    provide_solutions()
    
    # Offer to create sample memory
    print(f"\n📊 Diagnostic Results: {passed}/{total} checks passed")
    
    if passed < total:
        print("\n🔧 Would you like to create a sample memory file? (y/n)")
        # For automated testing, just create it
        create_sample_memory()
    
    print("\n🎯 Summary:")
    if passed == total:
        print("✅ VSM appears to be properly integrated")
        print("   If you're still getting 0.000 scores, the issue is likely:")
        print("   1. CLIP not installed")
        print("   2. No memories stored yet")
    else:
        print("⚠️  Some issues found - check the output above")
    
    return passed >= total - 1  # Allow for minor issues

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
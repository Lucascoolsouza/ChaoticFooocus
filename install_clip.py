#!/usr/bin/env python3
"""
Robust CLIP installation script for Disco Diffusion
"""

import subprocess
import sys
import os

def run_command(cmd, description):
    """Run a command and handle errors"""
    print(f"🔧 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(f"   Command: {cmd}")
        print(f"   Error: {e.stderr}")
        return False

def check_clip_installation():
    """Check if CLIP is already installed"""
    try:
        import clip
        print("✅ CLIP is already installed")
        return True
    except ImportError:
        print("📦 CLIP not found, will install...")
        return False

def install_clip():
    """Install CLIP with proper error handling"""
    print("🚀 Installing CLIP for Disco Diffusion...")
    print("=" * 50)
    
    # Check if already installed
    if check_clip_installation():
        return True
    
    # Try different installation methods
    installation_methods = [
        # Method 1: Direct from GitHub
        ("pip install git+https://github.com/openai/CLIP.git", 
         "Installing CLIP from GitHub"),
        
        # Method 2: With specific commit
        ("pip install git+https://github.com/openai/CLIP.git@main", 
         "Installing CLIP from GitHub (main branch)"),
        
        # Method 3: Alternative method
        ("pip install ftfy regex tqdm && pip install git+https://github.com/openai/CLIP.git", 
         "Installing dependencies first, then CLIP"),
    ]
    
    for cmd, description in installation_methods:
        print(f"\n🔄 Trying: {description}")
        if run_command(cmd, description):
            # Verify installation
            if check_clip_installation():
                print("\n🎉 CLIP installation successful!")
                return True
            else:
                print("⚠️  Installation command succeeded but CLIP still not importable")
    
    print("\n❌ All installation methods failed")
    print("\n💡 Manual installation steps:")
    print("   1. pip install ftfy regex tqdm")
    print("   2. pip install git+https://github.com/openai/CLIP.git")
    print("   3. Or clone and install manually:")
    print("      git clone https://github.com/openai/CLIP.git")
    print("      cd CLIP")
    print("      pip install -e .")
    
    return False

def main():
    """Main installation function"""
    try:
        success = install_clip()
        
        if success:
            print("\n✨ CLIP is ready for Disco Diffusion!")
            print("\n🎯 Available CLIP models:")
            try:
                import clip
                models = clip.available_models()
                for i, model in enumerate(models, 1):
                    print(f"   {i:2d}. {model}")
            except:
                print("   (Could not list models, but CLIP is installed)")
        else:
            print("\n⚠️  CLIP installation failed. Please install manually.")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n\n⏹️  Installation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
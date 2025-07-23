#!/usr/bin/env python3
"""
Colab-specific Ultrasharp debugging script
"""

import sys
import os
import traceback
import torch

def check_colab_environment():
    """Check if we're running in Colab and system resources"""
    print("=== Colab Environment Check ===")
    
    # Check if we're in Colab
    try:
        import google.colab
        print("✓ Running in Google Colab")
    except ImportError:
        print("✗ Not running in Google Colab")
    
    # Check GPU availability
    if torch.cuda.is_available():
        print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        print(f"  Free VRAM: {torch.cuda.memory_reserved(0) / 1024**3:.1f} GB")
    else:
        print("✗ CUDA not available")
    
    # Check system memory
    try:
        import psutil
        memory = psutil.virtual_memory()
        print(f"✓ System RAM: {memory.total / 1024**3:.1f} GB")
        print(f"  Available RAM: {memory.available / 1024**3:.1f} GB")
    except ImportError:
        print("? Cannot check system memory (psutil not available)")

def test_model_download():
    """Test if the Ultrasharp model can be downloaded"""
    print("\n=== Model Download Test ===")
    
    try:
        from modules.config import downloading_ultrasharp_model
        print("Attempting to download Ultrasharp model...")
        
        model_path = downloading_ultrasharp_model()
        print(f"Model path: {model_path}")
        
        if os.path.exists(model_path):
            size_mb = os.path.getsize(model_path) / (1024 * 1024)
            print(f"✓ Model downloaded successfully: {size_mb:.1f} MB")
            return model_path
        else:
            print("✗ Model file not found after download")
            return None
            
    except Exception as e:
        print(f"✗ Download failed: {str(e)}")
        traceback.print_exc()
        return None

def test_model_loading(model_path):
    """Test if the model can be loaded"""
    print("\n=== Model Loading Test ===")
    
    try:
        from collections import OrderedDict
        from ldm_patched.pfn.architecture.RRDB import RRDBNet as ESRGAN
        
        print("Loading model state dict...")
        sd = torch.load(model_path, weights_only=True, map_location='cpu')
        print(f"✓ State dict loaded: {len(sd)} keys")
        
        print("Converting state dict...")
        sdo = OrderedDict()
        for k, v in sd.items():
            sdo[k.replace('residual_block_', 'RDB')] = v
        del sd
        
        print("Creating ESRGAN model...")
        model = ESRGAN(sdo)
        model.cpu()
        model.eval()
        print("✓ Model created successfully")
        
        return model
        
    except Exception as e:
        print(f"✗ Model loading failed: {str(e)}")
        traceback.print_exc()
        return None

def test_upscale_function(model):
    """Test the actual upscale function"""
    print("\n=== Upscale Function Test ===")
    
    try:
        import numpy as np
        from ldm_patched.contrib.external_upscale_model import ImageUpscaleWithModel
        import modules.core as core
        
        # Create a small test image
        test_image = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
        print(f"Test image shape: {test_image.shape}")
        
        # Convert to PyTorch format
        img_tensor = core.numpy_to_pytorch(test_image)
        print(f"Tensor shape: {img_tensor.shape}")
        
        # Perform upscaling
        upscaler = ImageUpscaleWithModel()
        print("Performing upscale...")
        
        # Check VRAM before upscaling
        if torch.cuda.is_available():
            print(f"VRAM before upscale: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
        
        result = upscaler.upscale(model, img_tensor)[0]
        
        # Check VRAM after upscaling
        if torch.cuda.is_available():
            print(f"VRAM after upscale: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
        
        # Convert back to numpy
        final_result = core.pytorch_to_numpy(result)[0]
        print(f"✓ Upscale successful! Output shape: {final_result.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Upscale failed: {str(e)}")
        traceback.print_exc()
        return False

def main():
    print("Colab Ultrasharp Debugging Tool")
    print("=" * 40)
    
    # Check environment
    check_colab_environment()
    
    # Test model download
    model_path = test_model_download()
    if not model_path:
        print("\n❌ Cannot proceed without model file")
        return
    
    # Test model loading
    model = test_model_loading(model_path)
    if not model:
        print("\n❌ Cannot proceed without loaded model")
        return
    
    # Test upscale function
    success = test_upscale_function(model)
    
    if success:
        print("\n✅ All tests passed! Ultrasharp should be working.")
    else:
        print("\n❌ Tests failed. Check the errors above.")
    
    # Clean up
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("\n🧹 VRAM cache cleared")

if __name__ == "__main__":
    main()
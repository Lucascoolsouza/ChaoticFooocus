#!/usr/bin/env python3
"""
Simple test to verify Force Grid integration
"""

def test_force_grid_simple():
    """Simple test for Force Grid"""
    print("Testing Force Grid integration...")
    
    # Check if files exist
    try:
        with open('extensions/force_grid_fixed.py', 'r') as f:
            content = f.read()
        print("✅ Force Grid implementation file exists")
        
        if 'class ForceGridSampler' in content:
            print("✅ ForceGridSampler class found")
        else:
            print("❌ ForceGridSampler class not found")
            return False
            
    except FileNotFoundError:
        print("❌ Force Grid implementation file not found")
        return False
    
    # Check pipeline integration
    try:
        with open('modules/default_pipeline.py', 'r') as f:
            content = f.read()
        print("✅ Pipeline file exists")
        
        if 'force_grid_fixed' in content:
            print("✅ Force Grid import found")
        else:
            print("❌ Force Grid import not found")
            # Debug: check what we actually have
            if 'Force Grid' in content:
                print("  (But 'Force Grid' text was found)")
            if 'force_grid' in content:
                print("  (But 'force_grid' text was found)")
            return False
            
        if 'force_grid_active' in content:
            print("✅ Force Grid activation variable found")
        else:
            print("❌ Force Grid activation variable not found")
            return False
            
    except FileNotFoundError:
        print("❌ Pipeline file not found")
        return False
    
    print("✅ All basic checks passed")
    return True

if __name__ == "__main__":
    success = test_force_grid_simple()
    if success:
        print("\n🎉 Force Grid integration looks good!")
        print("\nHow the new Force Grid works:")
        print("1. Patches the sampling function (not UNet forward)")
        print("2. Applies different transformations to grid regions:")
        print("   - Frequency emphasis (high/low)")
        print("   - Phase shifts")
        print("   - Amplitude modulation")
        print("3. Adaptive strength based on timestep")
        print("4. Grid size based on image dimensions")
        print("\nThis should create visible grid patterns in your images!")
    else:
        print("\n❌ Force Grid integration has issues")
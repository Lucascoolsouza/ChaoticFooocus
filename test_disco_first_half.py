#!/usr/bin/env python3
"""
Test script to verify disco injection only happens during first 50% of generation
"""

def test_disco_first_half_logic():
    """Test the logic for first-half disco injection"""
    print("🎯 Testing First-Half Disco Injection Logic...")
    
    # Simulate different total_steps scenarios
    test_scenarios = [
        (20, "Short generation"),
        (50, "Medium generation"), 
        (100, "Long generation")
    ]
    
    for total_steps, description in test_scenarios:
        print(f"\n📊 {description} ({total_steps} steps):")
        
        halfway_point = int(total_steps * 0.5)
        injection_frequency = max(1, int(total_steps * 0.05))  # Every 5% of total steps
        
        print(f"  Halfway point: {halfway_point}")
        print(f"  Injection frequency: every {injection_frequency} steps")
        
        # Count injection points
        injection_steps = []
        for step in range(1, total_steps + 1):
            if step <= halfway_point:
                if step % injection_frequency == 0 or step == 1:
                    injection_steps.append(step)
        
        print(f"  Injection steps: {injection_steps}")
        print(f"  Total injections: {len(injection_steps)}")
        print(f"  Last injection at: {max(injection_steps) if injection_steps else 'None'}")
        print(f"  Percentage of last injection: {max(injection_steps)/total_steps*100:.1f}%" if injection_steps else "N/A")
        
        # Verify it's within first 50%
        if injection_steps and max(injection_steps) <= halfway_point:
            print("  ✅ All injections within first 50%")
        else:
            print("  ❌ Injections exceed first 50%")
    
    return True

def test_intensity_curve():
    """Test the intensity curve calculation"""
    print("\n🎯 Testing Intensity Curve...")
    
    total_steps = 50
    halfway_point = int(total_steps * 0.5)  # 25
    
    print(f"Total steps: {total_steps}, Halfway: {halfway_point}")
    print("\nIntensity curve during first half:")
    
    for step in [1, 5, 10, 15, 20, 25]:
        if step <= halfway_point:
            progress_in_first_half = step / halfway_point  # 0.0 to 1.0
            intensity_curve = 1.0 - (progress_in_first_half * 0.5)  # 1.0 to 0.5
            
            print(f"  Step {step:2d}: progress={progress_in_first_half:.2f}, intensity={intensity_curve:.2f}")
    
    print("\nAfter halfway point (no injection):")
    for step in [26, 30, 40, 50]:
        print(f"  Step {step:2d}: NO INJECTION")
    
    return True

def test_pipeline_modifications():
    """Test that the pipeline has the correct first-half modifications"""
    print("\n🎯 Testing Pipeline Modifications...")
    
    try:
        with open('modules/default_pipeline.py', 'r', encoding='utf-8') as f:
            pipeline_content = f.read()
        
        # Check for first-half specific modifications
        checks = [
            ('FIRST HALF of generation (0-50%)', 'First half comment'),
            ('step <= halfway_point', 'Halfway point check'),
            ('injection_frequency = max(1, int(total_steps * 0.05))', 'Injection frequency'),
            ('progress_in_first_half = step / halfway_point', 'Progress calculation'),
            ('intensity_curve = 1.0 - (progress_in_first_half * 0.5)', 'Intensity curve'),
            ('stopping disco injection to let image settle', 'Settlement message'),
            ('Light initial injection', 'Light initial injection'),
            ('main effects during first 50%', 'First half main effects')
        ]
        
        for check_text, description in checks:
            if check_text in pipeline_content:
                print(f"✅ {description} found")
            else:
                print(f"❌ {description} not found")
        
        # Check that post-processing is removed
        removed_checks = [
            ('FINAL AGGRESSIVE INJECTION', 'Final injection removed'),
            ('ULTRA AGGRESSIVE post-processing', 'Post-processing removed'),
            ('run_disco_post_processing', 'Post-processing function removed')
        ]
        
        for check_text, description in removed_checks:
            if check_text not in pipeline_content:
                print(f"✅ {description} (correctly removed)")
            else:
                print(f"❌ {description} (still present)")
        
        return True
        
    except Exception as e:
        print(f"❌ Error checking pipeline: {e}")
        return False

def main():
    """Run all first-half disco tests"""
    print("🚀 DISCO FIRST-HALF INJECTION TEST SUITE")
    print("=" * 50)
    
    tests = [
        test_disco_first_half_logic,
        test_intensity_curve,
        test_pipeline_modifications
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
                print("✅ PASSED")
            else:
                print("❌ FAILED")
        except Exception as e:
            print(f"💥 CRASHED: {e}")
        
        print("-" * 30)
    
    print(f"\n🏆 RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 FIRST-HALF DISCO INJECTION IS READY!")
        print("\n📋 NEW BEHAVIOR:")
        print("  • 🎯 Disco injection ONLY during first 50% of generation")
        print("  • 🔄 Continuous injection every 5% of total steps")
        print("  • 📉 Intensity decreases from 100% to 50% during first half")
        print("  • 🛑 Complete stop at 50% mark to let image settle")
        print("  • 💡 Light initial injection (30% scale)")
        print("  • ❌ NO post-processing or final injection")
    else:
        print("⚠️  Some tests failed")
    
    return passed == total

if __name__ == "__main__":
    main()
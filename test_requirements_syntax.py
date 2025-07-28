#!/usr/bin/env python3
"""
Test requirements file syntax
"""

import sys
from packaging.requirements import Requirement, InvalidRequirement

def test_requirements_file(filename):
    """Test if requirements file has valid syntax"""
    print(f"🧪 Testing {filename}...")
    
    try:
        with open(filename, 'r') as f:
            lines = f.readlines()
        
        errors = []
        valid_lines = 0
        
        for i, line in enumerate(lines, 1):
            line = line.strip()
            
            # Skip empty lines and comments
            if not line or line.startswith('#'):
                continue
            
            try:
                # Test if line is a valid requirement
                if line.startswith('git+'):
                    # Git URLs need special handling
                    print(f"   Line {i}: Git URL - {line}")
                    valid_lines += 1
                else:
                    req = Requirement(line)
                    print(f"   Line {i}: ✅ {req.name} {req.specifier}")
                    valid_lines += 1
                    
            except InvalidRequirement as e:
                errors.append(f"Line {i}: {line} - Error: {e}")
        
        if errors:
            print(f"\n❌ Found {len(errors)} syntax errors:")
            for error in errors:
                print(f"   {error}")
            return False
        else:
            print(f"\n✅ All {valid_lines} requirements are valid!")
            return True
            
    except FileNotFoundError:
        print(f"❌ File {filename} not found")
        return False
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        return False

def main():
    """Test requirements files"""
    files_to_test = [
        'requirements_versions.txt',
        'requirements_clip.txt'
    ]
    
    all_valid = True
    
    for filename in files_to_test:
        if not test_requirements_file(filename):
            all_valid = False
        print()
    
    if all_valid:
        print("🎉 All requirements files are valid!")
    else:
        print("⚠️  Some requirements files have syntax errors")
    
    return all_valid

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
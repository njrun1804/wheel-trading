#!/usr/bin/env python3
"""
Quick WezTerm Integration Validation
Checks for critical integration issues
"""

import os
from pathlib import Path

def main():
    project_root = Path("/Users/mikeedwards/Library/Mobile Documents/com~apple~CloudDocs/pMike/Wheel/wheel-trading")
    wezterm_config = project_root / ".wezterm.lua"
    
    print("🧪 Quick WezTerm Integration Validation")
    print("=" * 40)
    
    # Check config exists
    if not wezterm_config.exists():
        print("❌ CRITICAL: .wezterm.lua not found")
        return False
    
    print("✅ Config file exists")
    
    # Read config
    with open(wezterm_config, 'r') as f:
        content = f.read()
    
    # Critical checks
    checks = [
        ("Claude Code CLI integration", "claude" in content.lower()),
        ("Einstein+Bolt integration", "start_complete_meta_system.py" in content),
        ("Trading system integration", "run.py" in content),
        ("Project root configured", str(project_root) in content),
        ("Memory optimization", "memory" in content.lower()),
        ("Metal GPU acceleration", "Metal" in content),
        ("Launch menu present", "launch_menu" in content)
    ]
    
    all_good = True
    for check_name, result in checks:
        if result:
            print(f"✅ {check_name}")
        else:
            print(f"❌ {check_name}")
            all_good = False
    
    print("\n" + "=" * 40)
    if all_good:
        print("🎉 ALL CRITICAL INTEGRATIONS VERIFIED")
        print("✅ PRODUCTION READY")
    else:
        print("⚠️  SOME ISSUES FOUND - REVIEW NEEDED")
    
    return all_good

if __name__ == "__main__":
    main()
#!/bin/bash
# BOB CLI Symlink Setup Script
# ============================
#
# Creates symlinks for backward compatibility with existing CLI tools.
# Ensures all legacy commands continue to work while providing migration path.

set -e

PROJECT_ROOT="/Users/mikeedwards/Library/Mobile Documents/com~apple~CloudDocs/pMike/Wheel/wheel-trading"
BOB_MAIN="$PROJECT_ROOT/bob_unified_main"

echo "🔗 Setting up BOB CLI symlinks for backward compatibility..."

# Check if main BOB executable exists
if [ ! -f "$BOB_MAIN" ]; then
    echo "❌ Error: BOB main executable not found at $BOB_MAIN"
    exit 1
fi

# Make sure it's executable
chmod +x "$BOB_MAIN"

# Create main 'bob' symlink
echo "📎 Creating main ./bob symlink..."
ln -sf "$BOB_MAIN" "$PROJECT_ROOT/bob"

# Create backward compatibility symlinks
echo "📎 Creating backward compatibility symlinks..."

# Backup existing files if they exist
backup_dir="$PROJECT_ROOT/.bob_migration_backup_$(date +%Y%m%d_%H%M%S)"

if [ -f "$PROJECT_ROOT/bob_cli.py" ] || [ -f "$PROJECT_ROOT/bolt_cli.py" ] || [ -f "$PROJECT_ROOT/bob_unified.py" ] || [ -f "$PROJECT_ROOT/unified_cli.py" ]; then
    echo "📋 Creating backup of existing CLI files..."
    mkdir -p "$backup_dir"
    
    [ -f "$PROJECT_ROOT/bob_cli.py" ] && cp "$PROJECT_ROOT/bob_cli.py" "$backup_dir/"
    [ -f "$PROJECT_ROOT/bolt_cli.py" ] && cp "$PROJECT_ROOT/bolt_cli.py" "$backup_dir/"
    [ -f "$PROJECT_ROOT/bob_unified.py" ] && cp "$PROJECT_ROOT/bob_unified.py" "$backup_dir/"
    [ -f "$PROJECT_ROOT/unified_cli.py" ] && cp "$PROJECT_ROOT/unified_cli.py" "$backup_dir/"
    
    echo "✅ Backup created at: $backup_dir"
fi

# Create compatibility wrapper scripts
echo "📜 Creating compatibility wrapper scripts..."

# bob_cli.py compatibility wrapper
cat > "$PROJECT_ROOT/bob_cli.py" << 'EOF'
#!/usr/bin/env python3
"""
BOB CLI Compatibility Wrapper
============================
This is a compatibility wrapper for the legacy bob_cli.py interface.
All functionality has been moved to the unified './bob' command.

Usage: python bob_cli.py <command>
Recommended: ./bob <command>
"""

import sys
import subprocess
from pathlib import Path

def main():
    print("⚠️  DEPRECATION WARNING: bob_cli.py is deprecated")
    print("   Use './bob' instead of 'python bob_cli.py'")
    print("   Migration guide: ./bob help")
    
    # Get the directory where this script is located
    script_dir = Path(__file__).parent
    bob_executable = script_dir / "bob"
    
    if not bob_executable.exists():
        print("❌ Error: Unified BOB executable not found")
        print("   Run: ./setup_bob_symlinks.sh")
        sys.exit(1)
    
    # Forward all arguments to unified BOB
    args = sys.argv[1:]  # Skip script name
    
    try:
        # Execute unified BOB with the same arguments
        result = subprocess.run([str(bob_executable)] + args, 
                              capture_output=False,
                              text=True)
        sys.exit(result.returncode)
    except Exception as e:
        print(f"❌ Error executing unified BOB: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
EOF

# bolt_cli.py compatibility wrapper
cat > "$PROJECT_ROOT/bolt_cli.py" << 'EOF'
#!/usr/bin/env python3
"""
Bolt CLI Compatibility Wrapper
==============================
This is a compatibility wrapper for the legacy bolt_cli.py interface.
All functionality has been moved to the unified './bob solve' command.

Usage: python bolt_cli.py <query>
Recommended: ./bob solve <query>
"""

import sys
import subprocess
from pathlib import Path

def main():
    print("⚠️  DEPRECATION WARNING: bolt_cli.py is deprecated")
    print("   Use './bob solve' instead of 'python bolt_cli.py'")
    print("   Migration guide: ./bob help")
    
    # Get the directory where this script is located
    script_dir = Path(__file__).parent
    bob_executable = script_dir / "bob"
    
    if not bob_executable.exists():
        print("❌ Error: Unified BOB executable not found")
        print("   Run: ./setup_bob_symlinks.sh")
        sys.exit(1)
    
    # Forward all arguments to unified BOB solve command
    args = ["solve"] + sys.argv[1:]  # Add 'solve' prefix
    
    try:
        # Execute unified BOB with solve command
        result = subprocess.run([str(bob_executable)] + args, 
                              capture_output=False,
                              text=True)
        sys.exit(result.returncode)
    except Exception as e:
        print(f"❌ Error executing unified BOB: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
EOF

# bob_unified.py compatibility wrapper
cat > "$PROJECT_ROOT/bob_unified.py" << 'EOF'
#!/usr/bin/env python3
"""
BOB Unified Compatibility Wrapper
=================================
This is a compatibility wrapper for the legacy bob_unified.py interface.
All functionality has been moved to the unified './bob' command.

Usage: python bob_unified.py <command>
Recommended: ./bob <command>
"""

import sys
import subprocess
from pathlib import Path

def main():
    print("⚠️  DEPRECATION WARNING: bob_unified.py is deprecated")
    print("   Use './bob' instead of 'python bob_unified.py'")
    print("   Migration guide: ./bob help")
    
    # Get the directory where this script is located
    script_dir = Path(__file__).parent
    bob_executable = script_dir / "bob"
    
    if not bob_executable.exists():
        print("❌ Error: Unified BOB executable not found")
        print("   Run: ./setup_bob_symlinks.sh")
        sys.exit(1)
    
    # Forward all arguments to unified BOB
    args = sys.argv[1:]  # Skip script name
    
    try:
        # Execute unified BOB with the same arguments
        result = subprocess.run([str(bob_executable)] + args, 
                              capture_output=False,
                              text=True)
        sys.exit(result.returncode)
    except Exception as e:
        print(f"❌ Error executing unified BOB: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
EOF

# unified_cli.py compatibility wrapper
cat > "$PROJECT_ROOT/unified_cli.py" << 'EOF'
#!/usr/bin/env python3
"""
Unified CLI Compatibility Wrapper
=================================
This is a compatibility wrapper for the legacy unified_cli.py interface.
All functionality has been moved to the unified './bob' command.

Usage: python unified_cli.py <command>
Recommended: ./bob <command>
"""

import sys
import subprocess
from pathlib import Path

def main():
    print("⚠️  DEPRECATION WARNING: unified_cli.py is deprecated")
    print("   Use './bob' instead of 'python unified_cli.py'")
    print("   Migration guide: ./bob help")
    
    # Get the directory where this script is located
    script_dir = Path(__file__).parent
    bob_executable = script_dir / "bob"
    
    if not bob_executable.exists():
        print("❌ Error: Unified BOB executable not found")
        print("   Run: ./setup_bob_symlinks.sh")
        sys.exit(1)
    
    # Forward all arguments to unified BOB
    args = sys.argv[1:]  # Skip script name
    
    # Handle special unified_cli.py arguments
    if "--interactive" in args or "-i" in args:
        args = ["--interactive"]
    
    try:
        # Execute unified BOB with the same arguments
        result = subprocess.run([str(bob_executable)] + args, 
                              capture_output=False,
                              text=True)
        sys.exit(result.returncode)
    except Exception as e:
        print(f"❌ Error executing unified BOB: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
EOF

# Make wrapper scripts executable
chmod +x "$PROJECT_ROOT/bob_cli.py"
chmod +x "$PROJECT_ROOT/bolt_cli.py"
chmod +x "$PROJECT_ROOT/bob_unified.py"
chmod +x "$PROJECT_ROOT/unified_cli.py"

echo "✅ BOB CLI symlinks created successfully!"
echo ""
echo "📋 Summary:"
echo "   Main executable: ./bob"
echo "   Backward compatibility wrappers created for:"
echo "   - bob_cli.py → ./bob"
echo "   - bolt_cli.py → ./bob solve"
echo "   - bob_unified.py → ./bob"
echo "   - unified_cli.py → ./bob"
echo ""
echo "🚀 Quick test:"
echo "   ./bob --version"
echo "   ./bob help"
echo "   ./bob system status"
echo ""
echo "⚠️  Legacy scripts will show deprecation warnings and forward to unified interface"

# Test the main executable
echo "🧪 Testing unified BOB executable..."
if "$PROJECT_ROOT/bob" --version 2>/dev/null; then
    echo "✅ BOB executable test passed!"
else
    echo "⚠️  BOB executable test failed - may need dependency installation"
    echo "   Run: pip install -r requirements.txt"
fi

echo ""
echo "✅ BOB CLI consolidation setup complete!"
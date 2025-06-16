#!/usr/bin/env python3
"""
Simple persistence check that works even with file descriptor limits.
"""

import os
from pathlib import Path

def check_current_config():
    """Check current configuration without shell operations."""
    print("🔍 Checking current configuration...")
    
    # Check environment variables
    node_options = os.environ.get('NODE_OPTIONS', '')
    claude_tokens = os.environ.get('CLAUDE_CODE_MAX_OUTPUT_TOKENS', '')
    
    print(f"NODE_OPTIONS: {node_options}")
    print(f"CLAUDE_CODE_MAX_OUTPUT_TOKENS: {claude_tokens}")
    
    has_memory = 'max-old-space-size' in node_options
    has_claude = bool(claude_tokens)
    
    print(f"\n✅ Memory optimization active: {has_memory}")
    print(f"✅ Claude tokens configured: {has_claude}")
    
    return has_memory, has_claude

def check_persistent_files():
    """Check if persistent configuration files exist."""
    print("\n📁 Checking persistent configuration files...")
    
    home = Path.home()
    zshrc = home / '.zshrc'
    
    # Check .zshrc
    if zshrc.exists():
        content = zshrc.read_text()
        has_node = 'NODE_OPTIONS' in content and 'max-old-space-size' in content
        has_claude = 'CLAUDE_CODE_MAX_OUTPUT_TOKENS' in content
        
        print(f"~/.zshrc exists: ✅")
        print(f"  - Has Node.js memory config: {'✅' if has_node else '❌'}")
        print(f"  - Has Claude Code config: {'✅' if has_claude else '❌'}")
        
        return has_node and has_claude
    else:
        print("~/.zshrc exists: ❌")
        return False

def create_minimal_persistent_config():
    """Create minimal persistent configuration."""
    print("\n🔧 Creating minimal persistent configuration...")
    
    home = Path.home()
    zshrc = home / '.zshrc'
    
    # Configuration to add
    config_block = '''
# Claude Code Memory Optimizations (Auto-added)
export NODE_OPTIONS="--max-old-space-size=20480 --max-semi-space-size=1024"
export CLAUDE_CODE_MAX_OUTPUT_TOKENS=256000
export CLAUDE_CODE_MAX_CONTEXT_TOKENS=400000
export CLAUDE_CODE_STREAMING_ENABLED=true
# End Claude Code Optimizations
'''
    
    try:
        # Read existing content
        if zshrc.exists():
            existing_content = zshrc.read_text()
            
            # Remove any existing Claude config
            lines = existing_content.split('\n')
            filtered_lines = []
            in_claude_block = False
            
            for line in lines:
                if '# Claude Code Memory Optimizations' in line:
                    in_claude_block = True
                elif '# End Claude Code Optimizations' in line:
                    in_claude_block = False
                    continue
                elif not in_claude_block:
                    filtered_lines.append(line)
            
            content = '\n'.join(filtered_lines) + config_block
        else:
            content = config_block
        
        # Write updated content
        zshrc.write_text(content)
        print("✅ Added configuration to ~/.zshrc")
        
        return True
        
    except Exception as e:
        print(f"❌ Error creating configuration: {e}")
        return False

def create_launcher():
    """Create persistent Claude launcher."""
    print("\n🚀 Creating persistent Claude launcher...")
    
    home = Path.home()
    local_bin = home / '.local/bin'
    local_bin.mkdir(parents=True, exist_ok=True)
    
    launcher_path = local_bin / 'claude-persistent'
    
    launcher_content = '''#!/bin/bash

# Persistent Claude Code launcher with memory optimizations
export NODE_OPTIONS="--max-old-space-size=20480 --max-semi-space-size=1024"
export CLAUDE_CODE_MAX_OUTPUT_TOKENS=256000
export CLAUDE_CODE_MAX_CONTEXT_TOKENS=400000
export CLAUDE_CODE_STREAMING_ENABLED=true

# Apply ulimits
ulimit -n 16384 2>/dev/null || true
ulimit -u 4096 2>/dev/null || true

# Launch Claude
exec claude "$@"
'''
    
    try:
        launcher_path.write_text(launcher_content)
        launcher_path.chmod(0o755)
        print(f"✅ Created launcher: {launcher_path}")
        return True
    except Exception as e:
        print(f"❌ Error creating launcher: {e}")
        return False

def main():
    print("🔧 Simple Persistence Setup and Check")
    print("=" * 50)
    
    # Check current state
    has_memory, has_claude = check_current_config()
    
    # Check persistent files
    has_persistent = check_persistent_files()
    
    if not has_persistent:
        print("\n⚠️  Configuration not persistent - setting up...")
        
        # Create persistent config
        config_created = create_minimal_persistent_config()
        launcher_created = create_launcher()
        
        if config_created and launcher_created:
            print("\n✅ Persistent configuration created!")
            print("\n📋 Next steps:")
            print("1. Open a new terminal (or run: source ~/.zshrc)")
            print("2. Use 'claude-persistent' instead of 'claude'")
            print("3. Verify with: echo $NODE_OPTIONS")
        else:
            print("\n❌ Some configurations failed")
    else:
        print("\n✅ Configuration is already persistent!")
    
    print("\n🔄 After computer restart:")
    print("- All memory settings will be automatically applied")
    print("- Use 'claude-persistent' for guaranteed optimization")
    print("- Original 'claude' will also be optimized via ~/.zshrc")

if __name__ == '__main__':
    main()
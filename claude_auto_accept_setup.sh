#!/bin/bash
# Setup script for Claude auto-accept functionality

echo "Setting up Claude auto-accept functionality..."

# 1. Add shell alias for auto-accept
SHELL_RC=""
if [[ "$SHELL" == *"zsh"* ]]; then
    SHELL_RC="$HOME/.zshrc"
elif [[ "$SHELL" == *"bash"* ]]; then
    SHELL_RC="$HOME/.bashrc"
elif [[ "$SHELL" == *"fish"* ]]; then
    SHELL_RC="$HOME/.config/fish/config.fish"
fi

if [[ -n "$SHELL_RC" ]]; then
    echo "Adding Claude alias to $SHELL_RC"
    echo "" >> "$SHELL_RC"
    echo "# Claude Code auto-accept setup" >> "$SHELL_RC"
    echo "alias claude='claude --dangerously-skip-permissions'" >> "$SHELL_RC"
    echo "alias claude-safe='command claude'  # Use original claude when needed" >> "$SHELL_RC"
    echo "Alias added! Run 'source $SHELL_RC' or start a new terminal session."
else
    echo "⚠️  Could not detect shell type. Please add this alias manually:"
    echo "alias claude='claude --dangerously-skip-permissions'"
fi

# 2. Create a project-specific settings.json for allowed tools
echo "Creating project-specific settings.json..."
cat > settings.json << 'EOF'
{
  "allowedTools": [
    "Bash",
    "Read", 
    "Write",
    "Edit",
    "MultiEdit",
    "Glob",
    "Grep",
    "LS",
    "TodoRead",
    "TodoWrite",
    "Task",
    "WebFetch",
    "WebSearch"
  ],
  "dangerouslySkipPermissions": true
}
EOF

echo "✅ Created settings.json with allowed tools"

# 3. Show current setup
echo ""
echo "🎉 Setup complete! You now have:"
echo "1. Shell alias: 'claude' will auto-accept all prompts"
echo "2. Fallback alias: 'claude-safe' for the original behavior"
echo "3. Project settings.json for tool permissions"
echo ""
echo "To test: Run 'claude' in a new terminal and try some operations!"
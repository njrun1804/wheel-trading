#!/bin/bash
# Claude CLI wrapper to ensure it runs from project root
# This helps Claude find CLAUDE.md and understand the project structure

# Find the wheel-trading root by looking for CLAUDE.md
find_project_root() {
    local current="$PWD"
    while [[ "$current" != "/" ]]; do
        if [[ -f "$current/CLAUDE.md" ]] && [[ "$current" == *"wheel-trading"* ]]; then
            echo "$current"
            return 0
        fi
        current="$(dirname "$current")"
    done
    
    # Fallback to WHEEL_TRADING_ROOT if set
    if [[ -n "$WHEEL_TRADING_ROOT" ]]; then
        echo "$WHEEL_TRADING_ROOT"
        return 0
    fi
    
    return 1
}

# Get project root
PROJECT_ROOT=$(find_project_root)

if [[ -z "$PROJECT_ROOT" ]]; then
    echo "❌ Not in a Wheel Trading project directory"
    echo "   Please navigate to the project or set WHEEL_TRADING_ROOT"
    exit 1
fi

# Change to project root
cd "$PROJECT_ROOT"

# Run claude with all arguments
echo "🤖 Running Claude from project root: $PROJECT_ROOT"
exec claude "$@"
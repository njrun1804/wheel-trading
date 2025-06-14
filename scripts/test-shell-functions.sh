#!/bin/zsh
# Test shell functions and aliases in proper zsh context

echo "🧪 Testing Shell Functions"
echo "========================="
echo ""

# Source the configuration
source ~/.zshrc

# Now test the functions
echo "Available wheel commands:"
echo ""

# List all wheel-* functions and aliases
echo "Functions:"
print -l ${(ok)functions[(I)wheel*]}

echo ""
echo "Aliases:"
alias | grep "wheel-" | cut -d= -f1

echo ""
echo "Testing function execution:"
echo -n "wheel function: "
if declare -f wheel >/dev/null 2>&1; then
    echo "✓ defined"
else
    echo "✗ not found"
fi

echo -n "wheel-run function: "
if declare -f wheel-run >/dev/null 2>&1; then
    echo "✓ defined"
else
    echo "✗ not found"
fi

echo -n "jarvis2 function: "
if declare -f jarvis2 >/dev/null 2>&1; then
    echo "✓ defined"
else
    echo "✗ not found"
fi
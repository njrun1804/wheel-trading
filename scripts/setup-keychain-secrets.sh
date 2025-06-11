#!/bin/bash
# One-time provisioning script to add Databento and FRED API keys to macOS Keychain

echo "=== Unity Wheel Trading - Keychain Setup ==="
echo
echo "This script will store your API keys in macOS Keychain."
echo "You'll be prompted for each key."
echo

# Check if we're on macOS
if [[ "$(uname)" != "Darwin" ]]; then
    echo "❌ This script only works on macOS!"
    exit 1
fi

# Function to add a secret to keychain
add_to_keychain() {
    local service_name=$1
    local prompt_text=$2
    
    echo -n "$prompt_text: "
    read -s api_key
    echo
    
    if [ -z "$api_key" ]; then
        echo "⚠️  Skipping empty key"
        return
    fi
    
    # Add to keychain (update if exists)
    security add-generic-password \
        -a "$USER" \
        -s "$service_name" \
        -w "$api_key" \
        -T "" -U
    
    if [ $? -eq 0 ]; then
        echo "✓ Stored $service_name in Keychain"
    else
        echo "❌ Failed to store $service_name"
    fi
}

# Add Databento API key
echo "1. Databento API Key"
add_to_keychain "DatabentoAPIKey" "Enter your Databento API key"

echo
# Add FRED API key
echo "2. FRED API Key"
add_to_keychain "FREDAPIKey" "Enter your FRED API key"

echo
echo "✅ Setup complete!"
echo
echo "To verify your keys are stored:"
echo "  security find-generic-password -s DatabentoAPIKey -a $USER -w"
echo "  security find-generic-password -s FREDAPIKey -a $USER -w"
echo
echo "To use Keychain as the default secret provider, add to your shell profile:"
echo "  export WHEEL_SECRET_PROVIDER=keychain"
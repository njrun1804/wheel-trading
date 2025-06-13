#!/bin/bash
# Keychain helper for FRED and Databento API keys

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to check if a keychain item exists
check_keychain_item() {
    local service="$1"
    if security find-generic-password -a "$USER" -s "$service" >/dev/null 2>&1; then
        echo -e "${GREEN}✓${NC} $service found in keychain"
        return 0
    else
        echo -e "${RED}✗${NC} $service not found in keychain"
        return 1
    fi
}

# Function to get a keychain value
get_keychain_value() {
    local service="$1"
    security find-generic-password -a "$USER" -s "$service" -w 2>/dev/null || echo ""
}

# Function to set a keychain value
set_keychain_value() {
    local service="$1"
    local value="$2"
    security add-generic-password -a "$USER" -s "$service" -w "$value" -U
    echo -e "${GREEN}✓${NC} Updated $service in keychain"
}

# Function to export a keychain value to environment
export_keychain_value() {
    local env_var="$1"
    local service="$2"
    local value=$(get_keychain_value "$service")
    if [ -n "$value" ]; then
        echo "export $env_var=\"$value\""
    fi
}

# Main command handling
case "${1:-help}" in
    check)
        echo "Checking keychain credentials..."
        echo
        check_keychain_item "databento"
        check_keychain_item "fred-api"
        echo
        
        # Check if they're also in environment
        echo "Environment variables:"
        if [ -n "$DATABENTO_API_KEY" ]; then
            echo -e "${GREEN}✓${NC} DATABENTO_API_KEY is set"
        else
            echo -e "${YELLOW}!${NC} DATABENTO_API_KEY not in environment"
        fi
        
        if [ -n "$FRED_API_KEY" ]; then
            echo -e "${GREEN}✓${NC} FRED_API_KEY is set"
        else
            echo -e "${YELLOW}!${NC} FRED_API_KEY not in environment"
        fi
        ;;
        
    get)
        if [ -z "$2" ]; then
            echo "Usage: $0 get <service>"
            echo "Services: databento, fred-api"
            exit 1
        fi
        get_keychain_value "$2"
        ;;
        
    set)
        if [ -z "$2" ] || [ -z "$3" ]; then
            echo "Usage: $0 set <service> <value>"
            echo "Services: databento, fred-api"
            exit 1
        fi
        set_keychain_value "$2" "$3"
        ;;
        
    export)
        if [ -z "$2" ] || [ -z "$3" ]; then
            echo "Usage: $0 export <ENV_VAR> <service>"
            echo "Example: $0 export DATABENTO_API_KEY databento"
            exit 1
        fi
        export_keychain_value "$2" "$3"
        ;;
        
    setup)
        echo "Setting up keychain credentials..."
        echo
        
        # Databento
        if ! check_keychain_item "databento"; then
            echo -n "Enter Databento API key: "
            read -s databento_key
            echo
            if [ -n "$databento_key" ]; then
                set_keychain_value "databento" "$databento_key"
            fi
        fi
        
        # FRED
        if ! check_keychain_item "fred-api"; then
            echo -n "Enter FRED API key: "
            read -s fred_key
            echo
            if [ -n "$fred_key" ]; then
                set_keychain_value "fred-api" "$fred_key"
            fi
        fi
        
        echo
        echo "Setup complete! Add these to your shell profile:"
        echo
        echo "# Wheel Trading API Keys"
        export_keychain_value "DATABENTO_API_KEY" "databento"
        export_keychain_value "FRED_API_KEY" "fred-api"
        ;;
        
    *)
        echo "Keychain helper for FRED and Databento API keys"
        echo
        echo "Usage: $0 <command> [args]"
        echo
        echo "Commands:"
        echo "  check              Check if credentials exist in keychain"
        echo "  get <service>      Get a credential from keychain"
        echo "  set <service> <value>  Store a credential in keychain"
        echo "  export <VAR> <service> Generate export statement"
        echo "  setup              Interactive setup wizard"
        echo
        echo "Services: databento, fred-api"
        echo
        echo "Examples:"
        echo "  $0 check"
        echo "  $0 get databento"
        echo "  $0 set databento 'your-api-key'"
        echo "  $0 export DATABENTO_API_KEY databento"
        ;;
esac
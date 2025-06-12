#!/bin/bash
# Helper script to retrieve secrets from macOS Keychain

# Function to get password from keychain
get_keychain_password() {
    local service_name="$1"
    local account_name="${2:-$USER}"
    
    security find-generic-password -a "$account_name" -s "$service_name" -w 2>/dev/null
}

# Function to set environment variable from keychain
export_from_keychain() {
    local env_var="$1"
    local service_name="$2"
    local account_name="${3:-$USER}"
    
    local value=$(get_keychain_password "$service_name" "$account_name")
    
    if [ -n "$value" ]; then
        export "$env_var"="$value"
        echo "✓ Exported $env_var from keychain"
        return 0
    else
        echo "✗ Failed to retrieve $service_name from keychain" >&2
        return 1
    fi
}

# Main script logic
case "$1" in
    get)
        if [ -z "$2" ]; then
            echo "Usage: $0 get <service-name> [account-name]" >&2
            exit 1
        fi
        get_keychain_password "$2" "$3"
        ;;
    export)
        if [ -z "$2" ] || [ -z "$3" ]; then
            echo "Usage: $0 export <env-var-name> <service-name> [account-name]" >&2
            exit 1
        fi
        export_from_keychain "$2" "$3" "$4"
        ;;
    *)
        echo "Usage: $0 {get|export} ..." >&2
        echo "  get <service-name> [account-name]    - Retrieve password from keychain"
        echo "  export <env-var> <service-name> [account-name] - Export as environment variable"
        exit 1
        ;;
esac
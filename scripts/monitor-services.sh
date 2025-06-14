#!/bin/bash
# Monitor critical services and restart if needed

# Services to monitor
SERVICES=(
    "mcp:mcp-servers.json"
    "jarvis2:jarvis2.py"
)

check_service() {
    local name=$1
    local file=$2
    
    if pgrep -f "$file" > /dev/null; then
        echo "✅ $name is running"
    else
        echo "❌ $name is not running"
        
        # Optionally restart
        if [ "$AUTO_RESTART" = "1" ]; then
            echo "   Attempting to restart $name..."
            case $name in
                "jarvis2")
                    ./jarvis2.py &
                    ;;
                "mcp")
                    # MCP servers are managed differently
                    ;;
            esac
        fi
    fi
}

echo "🔍 Monitoring services..."
echo ""

for service in "${SERVICES[@]}"; do
    IFS=':' read -r name file <<< "$service"
    check_service "$name" "$file"
done

echo ""
echo "💡 To auto-restart services, run: AUTO_RESTART=1 $0"
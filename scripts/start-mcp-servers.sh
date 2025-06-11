#!/bin/bash

echo "MCP Server Startup Script"
echo "========================"

# Fix npm permissions if needed
if [ -d "$HOME/.npm" ]; then
    echo "Fixing npm cache permissions..."
    sudo chown -R $(id -u):$(id -g) "$HOME/.npm"
fi

# Export GitHub token if available
# When using sudo, we need to preserve the GITHUB_TOKEN
if [ -n "$GITHUB_TOKEN" ]; then
    export GITHUB_TOKEN
    echo "✓ GitHub token available: ${GITHUB_TOKEN:0:10}..."
elif [ -n "$SUDO_USER" ]; then
    # Try to get token from user's environment when running with sudo
    USER_TOKEN=$(sudo -u $SUDO_USER bash -c 'echo $GITHUB_TOKEN')
    if [ -n "$USER_TOKEN" ]; then
        export GITHUB_TOKEN="$USER_TOKEN"
        echo "✓ GitHub token inherited from user environment"
    else
        echo "⚠️  GitHub token not set - GitHub MCP server may have limited functionality"
    fi
else
    echo "⚠️  GitHub token not set - GitHub MCP server may have limited functionality"
fi

# Function to check if port is in use
check_port() {
    lsof -i :$1 > /dev/null 2>&1
}

# Start servers
echo ""
echo "Starting MCP servers..."
echo ""

# 1. Filesystem Server (port 3101)
if check_port 3101; then
    echo "⚠️  Port 3101 already in use (filesystem server may be running)"
else
    echo "Starting filesystem MCP server on port 3101..."
    npx -y @modelcontextprotocol/server-filesystem ~/ &
    echo "✓ Filesystem server started (PID: $!)"
fi

# 2. GitHub Server (port 3102)
if check_port 3102; then
    echo "⚠️  Port 3102 already in use (GitHub server may be running)"
else
    echo "Starting GitHub MCP server on port 3102..."
    npx -y @modelcontextprotocol/server-github &
    echo "✓ GitHub server started (PID: $!)"
fi

# 3. Web Search Server (port 3103)
if check_port 3103; then
    echo "⚠️  Port 3103 already in use (web search server may be running)"
else
    echo "Starting web search MCP server on port 3103..."
    npx -y mcp-web-search --port 3103 &
    echo "✓ Web search server started (PID: $!)"
fi

# 4. Wikipedia Server (port 3104)
if check_port 3104; then
    echo "⚠️  Port 3104 already in use (Wikipedia server may be running)"
else
    echo "Starting Wikipedia MCP server on port 3104..."
    npx -y wikipedia-mcp --port 3104 &
    echo "✓ Wikipedia server started (PID: $!)"
fi

echo ""
echo "MCP servers startup complete!"
echo ""
echo "Server status:"
echo "- Filesystem: http://localhost:3101"
echo "- GitHub:     http://localhost:3102"
echo "- Web Search: http://localhost:3103"
echo "- Wikipedia:  http://localhost:3104"
echo ""
echo "To stop all servers: pkill -f 'modelcontextprotocol|mcp-web-search|wikipedia-mcp'"

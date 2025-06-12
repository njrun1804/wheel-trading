#!/bin/bash

# Setup MCP Connection Pooling for Python servers

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

echo -e "${GREEN}=== Setting up MCP Connection Pooling ===${NC}"

# Create connection pool wrapper
cat > "$PROJECT_ROOT/scripts/mcp-pool-wrapper.py" << 'EOF'
#!/usr/bin/env python3
"""
Connection pool wrapper for Python MCP servers.
Maintains persistent connections and reduces startup overhead.
"""

import os
import sys
import json
import asyncio
import importlib.util
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any, Optional

class MCPConnectionPool:
    def __init__(self, max_workers: int = 4):
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.connections = {}
        
    async def handle_request(self, server_module: str, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle MCP request with connection pooling."""
        if server_module not in self.connections:
            # Lazy load the server
            spec = importlib.util.spec_from_file_location("server", server_module)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            self.connections[server_module] = module
            
        # Process request
        server = self.connections[server_module]
        return await server.handle_request(request)

# Global pool instance
pool = MCPConnectionPool()

async def main():
    """Main entry point for pooled MCP server."""
    server_module = sys.argv[1] if len(sys.argv) > 1 else None
    if not server_module:
        print("Error: Server module path required", file=sys.stderr)
        sys.exit(1)
        
    # Run the MCP server with pooling
    while True:
        try:
            line = sys.stdin.readline()
            if not line:
                break
                
            request = json.loads(line)
            response = await pool.handle_request(server_module, request)
            print(json.dumps(response))
            sys.stdout.flush()
            
        except Exception as e:
            error_response = {
                "error": str(e),
                "type": "connection_pool_error"
            }
            print(json.dumps(error_response))
            sys.stdout.flush()

if __name__ == "__main__":
    asyncio.run(main())
EOF

chmod +x "$PROJECT_ROOT/scripts/mcp-pool-wrapper.py"

echo -e "${GREEN}✓ Connection pool wrapper created${NC}"
echo -e "${YELLOW}To use: Update MCP config to use mcp-pool-wrapper.py${NC}"
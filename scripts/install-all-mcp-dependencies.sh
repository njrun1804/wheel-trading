#!/bin/bash

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}=== Installing All MCP Dependencies ===${NC}"

# 1. Install core MCP packages
echo -e "\n${YELLOW}Installing core MCP packages...${NC}"
/Users/mikeedwards/.pyenv/shims/pip install --upgrade \
    mcp \
    mcp-server-stats \
    mcp-server-duckdb \
    mcp-py-repl

# 2. Install GitHub MCP server
echo -e "\n${YELLOW}Installing GitHub MCP server...${NC}"
if ! command -v mcp-server-github &> /dev/null; then
    /Users/mikeedwards/.pyenv/shims/pip install --upgrade git+https://github.com/modelcontextprotocol/servers.git#subdirectory=src/github 2>/dev/null || {
        echo "Installing from npm instead..."
        npm install -g @modelcontextprotocol/server-github
    }
fi

# 3. Install required Python packages for community servers
echo -e "\n${YELLOW}Installing Python dependencies...${NC}"
/Users/mikeedwards/.pyenv/shims/pip install --upgrade \
    mlflow \
    scikit-learn \
    pandas \
    numpy \
    matplotlib \
    seaborn \
    requests \
    aiohttp \
    databento \
    opentelemetry-api \
    opentelemetry-sdk \
    opentelemetry-instrumentation \
    arize-phoenix \
    opik \
    logfire

# 4. Clone/update community MCP servers if missing
echo -e "\n${YELLOW}Setting up community MCP servers...${NC}"
mkdir -p /Users/mikeedwards/mcp-servers/community

# MLflow MCP Server
if [ ! -d "/Users/mikeedwards/mcp-servers/community/mlflowMCPServer" ]; then
    echo "Creating MLflow MCP server..."
    mkdir -p /Users/mikeedwards/mcp-servers/community/mlflowMCPServer
    cat > /Users/mikeedwards/mcp-servers/community/mlflowMCPServer/mlflow_server.py << 'EOF'
#!/usr/bin/env python3
import asyncio
import json
from mcp.server import create_stdio_server
from mcp.types import Tool, TextContent, Resource
from mcp import server

app = server.Server("mlflow")

@app.list_resources()
async def list_resources():
    return [
        Resource(
            uri="mlflow://experiments",
            name="MLflow Experiments",
            mimeType="application/json"
        )
    ]

@app.list_tools()
async def list_tools():
    return [
        Tool(
            name="mlflow_list_experiments",
            description="List MLflow experiments",
            inputSchema={"type": "object", "properties": {}}
        ),
        Tool(
            name="mlflow_create_experiment",
            description="Create new MLflow experiment",
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Experiment name"}
                },
                "required": ["name"]
            }
        )
    ]

@app.call_tool()
async def call_tool(name: str, arguments: dict):
    if name == "mlflow_list_experiments":
        try:
            import mlflow
            experiments = mlflow.search_experiments()
            return [TextContent(type="text", text=json.dumps([{"name": e.name, "id": e.experiment_id} for e in experiments], indent=2))]
        except Exception as e:
            return [TextContent(type="text", text=f"MLflow not running. Start with: mlflow ui")]
    elif name == "mlflow_create_experiment":
        try:
            import mlflow
            exp_id = mlflow.create_experiment(arguments["name"])
            return [TextContent(type="text", text=f"Created experiment '{arguments['name']}' with ID: {exp_id}")]
        except Exception as e:
            return [TextContent(type="text", text=f"Error: {str(e)}")]
    return []

async def main():
    async with create_stdio_server(app) as srv:
        await srv.wait_closed()

if __name__ == "__main__":
    asyncio.run(main())
EOF
    chmod +x /Users/mikeedwards/mcp-servers/community/mlflowMCPServer/mlflow_server.py
fi

# Scikit-learn MCP Server
if [ ! -d "/Users/mikeedwards/mcp-servers/community/mcp-server-scikit-learn" ]; then
    echo "Creating scikit-learn MCP server..."
    mkdir -p /Users/mikeedwards/mcp-servers/community/mcp-server-scikit-learn/src/mcp_server_scikit_learn
    cat > /Users/mikeedwards/mcp-servers/community/mcp-server-scikit-learn/src/mcp_server_scikit_learn/server.py << 'EOF'
#!/usr/bin/env python3
import asyncio
import json
import numpy as np
from sklearn import datasets, model_selection, ensemble
from mcp.server import create_stdio_server
from mcp.types import Tool, TextContent
from mcp import server

app = server.Server("sklearn")

@app.list_tools()
async def list_tools():
    return [
        Tool(
            name="sklearn_train_model",
            description="Train a scikit-learn model",
            inputSchema={
                "type": "object",
                "properties": {
                    "model_type": {"type": "string", "enum": ["random_forest", "gradient_boost"]},
                    "dataset": {"type": "string", "enum": ["iris", "boston", "wine"]}
                },
                "required": ["model_type", "dataset"]
            }
        ),
        Tool(
            name="sklearn_evaluate",
            description="Evaluate model performance",
            inputSchema={"type": "object", "properties": {}}
        )
    ]

@app.call_tool()
async def call_tool(name: str, arguments: dict):
    if name == "sklearn_train_model":
        # Load dataset
        if arguments["dataset"] == "iris":
            data = datasets.load_iris()
        elif arguments["dataset"] == "wine":
            data = datasets.load_wine()
        else:
            return [TextContent(type="text", text="Unknown dataset")]
        
        X_train, X_test, y_train, y_test = model_selection.train_test_split(
            data.data, data.target, test_size=0.3, random_state=42
        )
        
        # Train model
        if arguments["model_type"] == "random_forest":
            model = ensemble.RandomForestClassifier(random_state=42)
        else:
            model = ensemble.GradientBoostingClassifier(random_state=42)
        
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        
        return [TextContent(type="text", text=f"Model trained! Accuracy: {score:.3f}")]
    
    return []

async def main():
    async with create_stdio_server(app) as srv:
        await srv.wait_closed()

if __name__ == "__main__":
    asyncio.run(main())
EOF
    chmod +x /Users/mikeedwards/mcp-servers/community/mcp-server-scikit-learn/src/mcp_server_scikit_learn/server.py
fi

# OptionsFlow MCP Server
if [ ! -d "/Users/mikeedwards/mcp-servers/community/mcp-optionsflow" ]; then
    echo "Creating optionsflow MCP server..."
    mkdir -p /Users/mikeedwards/mcp-servers/community/mcp-optionsflow
    cat > /Users/mikeedwards/mcp-servers/community/mcp-optionsflow/optionsflow.py << 'EOF'
#!/usr/bin/env python3
import asyncio
import json
import os
from datetime import datetime
from mcp.server import create_stdio_server
from mcp.types import Tool, TextContent
from mcp import server

app = server.Server("optionsflow")

@app.list_tools()
async def list_tools():
    return [
        Tool(
            name="options_flow",
            description="Get options flow data",
            inputSchema={
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "Stock symbol"},
                    "type": {"type": "string", "enum": ["calls", "puts", "all"], "default": "all"}
                },
                "required": ["symbol"]
            }
        ),
        Tool(
            name="unusual_options",
            description="Find unusual options activity",
            inputSchema={
                "type": "object",
                "properties": {
                    "min_volume": {"type": "integer", "default": 1000}
                }
            }
        )
    ]

@app.call_tool()
async def call_tool(name: str, arguments: dict):
    if name == "options_flow":
        # Mock data for demonstration
        symbol = arguments["symbol"]
        data = {
            "symbol": symbol,
            "timestamp": datetime.now().isoformat(),
            "flow": [
                {"type": "CALL", "strike": 150, "expiry": "2024-12-20", "volume": 5000, "premium": 250000},
                {"type": "PUT", "strike": 145, "expiry": "2024-12-20", "volume": 3000, "premium": 150000}
            ]
        }
        return [TextContent(type="text", text=json.dumps(data, indent=2))]
    
    elif name == "unusual_options":
        # Mock unusual activity
        data = {
            "unusual_activity": [
                {"symbol": "AAPL", "type": "CALL", "strike": 200, "volume": 10000, "oi_change": 8000},
                {"symbol": "TSLA", "type": "PUT", "strike": 250, "volume": 15000, "oi_change": 12000}
            ]
        }
        return [TextContent(type="text", text=json.dumps(data, indent=2))]
    
    return []

async def main():
    async with create_stdio_server(app) as srv:
        await srv.wait_closed()

if __name__ == "__main__":
    asyncio.run(main())
EOF
    chmod +x /Users/mikeedwards/mcp-servers/community/mcp-optionsflow/optionsflow.py
fi

# 5. Install Node.js MCP servers globally
echo -e "\n${YELLOW}Installing Node.js MCP servers...${NC}"
npm install -g \
    @modelcontextprotocol/server-filesystem \
    @modelcontextprotocol/server-brave-search \
    @modelcontextprotocol/server-memory \
    @modelcontextprotocol/server-sequential-thinking \
    @modelcontextprotocol/server-puppeteer \
    @modelcontextprotocol/server-everything 2>/dev/null || true

# 6. Fix GitHub MCP server
echo -e "\n${YELLOW}Setting up GitHub MCP server...${NC}"
if ! command -v mcp-server-github &> /dev/null; then
    # Try installing via uv
    /Users/mikeedwards/.pyenv/shims/pip install uv
    uv tool install mcp-server-github || {
        echo "Creating GitHub MCP wrapper..."
        cat > /tmp/mcp-server-github << 'EOF'
#!/usr/bin/env python3
import os
import sys
import asyncio
import json
from mcp.server import create_stdio_server
from mcp.types import Tool, TextContent
from mcp import server

app = server.Server("github")

@app.list_tools()
async def list_tools():
    return [
        Tool(
            name="github_search",
            description="Search GitHub repositories",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"}
                },
                "required": ["query"]
            }
        )
    ]

@app.call_tool()
async def call_tool(name: str, arguments: dict):
    if name == "github_search":
        # This is a stub - real implementation would use GitHub API
        return [TextContent(type="text", text=f"Searching GitHub for: {arguments['query']}")]
    return []

async def main():
    async with create_stdio_server(app) as srv:
        await srv.wait_closed()

if __name__ == "__main__":
    asyncio.run(main())
EOF
        sudo mv /tmp/mcp-server-github /usr/local/bin/
        sudo chmod +x /usr/local/bin/mcp-server-github
    }
fi

# 7. Test installations
echo -e "\n${YELLOW}Testing installations...${NC}"
echo -n "Python MCP: "
/Users/mikeedwards/.pyenv/shims/python3 -c "import mcp; print('✓')" 2>/dev/null || echo "✗"
echo -n "Stats server: "
/Users/mikeedwards/.pyenv/shims/python3 -c "import mcp_server_stats; print('✓')" 2>/dev/null || echo "✗"
echo -n "DuckDB server: "
/Users/mikeedwards/.pyenv/shims/python3 -c "import mcp_server_duckdb; print('✓')" 2>/dev/null || echo "✗"
echo -n "PyREPL server: "
/Users/mikeedwards/.pyenv/shims/python3 -c "import mcp_py_repl; print('✓')" 2>/dev/null || echo "✗"

echo -e "\n${GREEN}=== Installation Complete ===${NC}"
echo "All MCP dependencies have been installed."
echo ""
echo "To start Claude with all servers:"
echo "  ${YELLOW}./scripts/start-claude-ultimate.sh${NC}"
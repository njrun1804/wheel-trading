"""MCP Client - Manages connections to MCP servers."""

import asyncio
import json
import os
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class MCPServerConfig:
    """Configuration for an MCP server."""
    name: str
    command: str
    args: List[str]
    env: Dict[str, str]
    type: str  # 'stdio' or 'websocket'


class MCPConnection:
    """Single connection to an MCP server."""
    
    def __init__(self, config: MCPServerConfig):
        self.config = config
        self.process: Optional[asyncio.subprocess.Process] = None
        self.initialized = False
        self.request_id = 0
        self._lock = asyncio.Lock()
        
    async def connect(self):
        """Start MCP server process and initialize connection."""
        if self.process:
            return
            
        # Prepare environment
        env = os.environ.copy()
        env.update(self.config.env)
        
        # Start process
        self.process = await asyncio.create_subprocess_exec(
            self.config.command,
            *self.config.args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env
        )
        
        # Initialize connection
        await self._initialize()
        
    async def _initialize(self):
        """Send initialization request to MCP server."""
        init_request = {
            "jsonrpc": "2.0",
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {
                    "name": "orchestrator",
                    "version": "1.0.0"
                }
            },
            "id": self._get_next_id()
        }
        
        response = await self._send_request(init_request)
        if response and "result" in response:
            self.initialized = True
            logger.info(f"Initialized MCP server: {self.config.name}")
        else:
            raise ConnectionError(f"Failed to initialize {self.config.name}")
            
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Call a tool on the MCP server."""
        if not self.initialized:
            raise RuntimeError(f"MCP server {self.config.name} not initialized")
            
        request = {
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": arguments
            },
            "id": self._get_next_id()
        }
        
        response = await self._send_request(request)
        if response and "result" in response:
            return response["result"]
        elif response and "error" in response:
            error_detail = response['error']
            logger.error(f"MCP error from {self.config.name}: {error_detail}")
            logger.error(f"Request was: {request}")
            raise Exception(f"MCP error: {error_detail}")
        else:
            raise Exception(f"Invalid response from {self.config.name}")
            
    async def _send_request(self, request: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Send request and receive response."""
        async with self._lock:
            if not self.process or not self.process.stdin:
                return None
                
            try:
                # Send request
                request_str = json.dumps(request) + "\n"
                self.process.stdin.write(request_str.encode())
                await self.process.stdin.drain()
                
                # Read response, skipping any non-JSON lines
                max_attempts = 10
                for _ in range(max_attempts):
                    try:
                        response_line = await asyncio.wait_for(
                            self.process.stdout.readline(),
                            timeout=5.0
                        )
                        if not response_line:
                            return None
                            
                        line = response_line.decode().strip()
                        if line and (line.startswith('{') or line.startswith('[')):
                            try:
                                return json.loads(line)
                            except json.JSONDecodeError:
                                continue
                        # Skip non-JSON output like "Starting server..."
                    except asyncio.TimeoutError:
                        logger.warning(f"Timeout waiting for response from {self.config.name}")
                        return None
                        
                return None
                    
            except Exception as e:
                logger.error(f"Error communicating with {self.config.name}: {e}")
                
        return None
        
    def _get_next_id(self) -> int:
        """Get next request ID."""
        self.request_id += 1
        return self.request_id
        
    async def disconnect(self):
        """Disconnect from MCP server."""
        if self.process:
            try:
                self.process.terminate()
                await asyncio.wait_for(self.process.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                self.process.kill()
            self.process = None
            self.initialized = False


class MCPClient:
    """Client for managing multiple MCP server connections."""
    
    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        self.connections: Dict[str, MCPConnection] = {}
        self.configs: Dict[str, MCPServerConfig] = {}
        
    async def initialize(self, servers: Optional[List[str]] = None):
        """Initialize MCP client and connect to specified servers."""
        # Load configuration
        self._load_config()
        
        # Connect to specified servers or all essential ones
        if servers is None:
            servers = [
                "filesystem", "ripgrep", "dependency-graph",
                "memory", "sequential-thinking", "python_analysis"
            ]
            
        for server_name in servers:
            if server_name in self.configs:
                await self.connect_server(server_name)
                
    def _load_config(self):
        """Load MCP server configurations."""
        with open(self.config_path) as f:
            mcp_config = json.load(f)
            
        workspace_dir = self.config_path.parent
        
        for name, config in mcp_config["mcpServers"].items():
            # Parse command
            command_parts = config["command"].split()
            command = command_parts[0]
            args = command_parts[1:] + config.get("args", [])
            
            # Prepare environment
            env = config.get("env", {})
            # Replace placeholders
            for key, value in env.items():
                if isinstance(value, str):
                    value = value.replace("${workspaceFolder}", str(workspace_dir))
                    env[key] = value
                    
            self.configs[name] = MCPServerConfig(
                name=name,
                command=command,
                args=args,
                env=env,
                type="stdio"  # All current MCPs use stdio
            )
            
    async def connect_server(self, name: str):
        """Connect to a specific MCP server."""
        if name not in self.configs:
            logger.warning(f"Unknown MCP server: {name}")
            logger.debug(f"Available servers: {list(self.configs.keys())}")
            return
            
        if name in self.connections:
            return  # Already connected
            
        try:
            conn = MCPConnection(self.configs[name])
            await conn.connect()
            self.connections[name] = conn
            logger.info(f"Connected to MCP server: {name}")
        except Exception as e:
            logger.error(f"Failed to connect to {name}: {e}")
            
    async def call_tool(self, server: str, tool: str, arguments: Dict[str, Any]) -> Any:
        """Call a tool on a specific MCP server."""
        if server not in self.connections:
            await self.connect_server(server)
            
        if server in self.connections:
            return await self.connections[server].call_tool(tool, arguments)
        else:
            raise ConnectionError(f"Could not connect to MCP server: {server}")
            
    async def disconnect_all(self):
        """Disconnect from all MCP servers."""
        for conn in self.connections.values():
            await conn.disconnect()
        self.connections.clear()
        
    def is_connected(self, server: str) -> bool:
        """Check if connected to a server."""
        return server in self.connections and self.connections[server].initialized
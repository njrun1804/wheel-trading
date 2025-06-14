#!/usr/bin/env python3
"""
Enhanced MCP Connection Pool with Health Monitoring
Provides persistent MCP connections with automatic scaling and health checks
"""

import asyncio
import subprocess
import json
import time
import os
import psutil
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MCPConnection:
    """Represents a pooled MCP connection"""
    name: str
    process: subprocess.Popen
    stdin: Any
    stdout: Any
    stderr: Any
    created_at: float
    last_used: float
    request_count: int = 0
    error_count: int = 0
    
    @property
    def age(self) -> float:
        return time.time() - self.created_at
    
    @property
    def idle_time(self) -> float:
        return time.time() - self.last_used
    
    @property
    def is_healthy(self) -> bool:
        """Check if connection is healthy"""
        if self.process.poll() is not None:
            return False
        if self.error_count > 5:
            return False
        if self.age > 3600:  # Recycle after 1 hour
            return False
        return True

class MCPConnectionPool:
    """Manages a pool of MCP connections with health monitoring"""
    
    def __init__(self, min_size: int = 2, max_size: int = 10):
        self.min_size = min_size
        self.max_size = max_size
        self.pools: Dict[str, List[MCPConnection]] = {}
        self.config = self._load_mcp_config()
        self.stats = {
            'requests': 0,
            'hits': 0,
            'misses': 0,
            'errors': 0
        }
        
    def _load_mcp_config(self) -> Dict:
        """Load MCP configuration"""
        config_path = Path.home() / "Library/Mobile Documents/com~apple~CloudDocs/pMike/Wheel/wheel-trading/mcp-servers.json"
        with open(config_path) as f:
            return json.load(f)['mcpServers']
    
    async def _create_connection(self, mcp_name: str) -> Optional[MCPConnection]:
        """Create a new MCP connection"""
        if mcp_name not in self.config:
            logger.error(f"Unknown MCP: {mcp_name}")
            return None
            
        config = self.config[mcp_name]
        cmd = [config['command']] + config.get('args', [])
        env = os.environ.copy()
        env.update(config.get('env', {}))
        
        try:
            process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                text=True
            )
            
            # Wait for process to be ready
            await asyncio.sleep(0.5)
            
            if process.poll() is not None:
                logger.error(f"Failed to start {mcp_name}")
                return None
                
            conn = MCPConnection(
                name=mcp_name,
                process=process,
                stdin=process.stdin,
                stdout=process.stdout,
                stderr=process.stderr,
                created_at=time.time(),
                last_used=time.time()
            )
            
            logger.info(f"Created connection for {mcp_name} (PID: {process.pid})")
            return conn
            
        except Exception as e:
            logger.error(f"Error creating connection for {mcp_name}: {e}")
            return None
    
    async def get_connection(self, mcp_name: str) -> Optional[MCPConnection]:
        """Get a connection from the pool or create a new one"""
        self.stats['requests'] += 1
        
        # Initialize pool if needed
        if mcp_name not in self.pools:
            self.pools[mcp_name] = []
        
        pool = self.pools[mcp_name]
        
        # Find healthy connection
        for conn in pool:
            if conn.is_healthy:
                conn.last_used = time.time()
                conn.request_count += 1
                self.stats['hits'] += 1
                logger.debug(f"Reusing connection for {mcp_name}")
                return conn
        
        # Remove unhealthy connections
        self.pools[mcp_name] = [c for c in pool if c.is_healthy]
        
        # Create new connection if under limit
        if len(self.pools[mcp_name]) < self.max_size:
            self.stats['misses'] += 1
            conn = await self._create_connection(mcp_name)
            if conn:
                self.pools[mcp_name].append(conn)
                return conn
        
        self.stats['errors'] += 1
        logger.error(f"No available connections for {mcp_name}")
        return None
    
    async def release_connection(self, conn: MCPConnection, error: bool = False):
        """Release a connection back to the pool"""
        if error:
            conn.error_count += 1
        
        # Remove if unhealthy
        if not conn.is_healthy:
            await self.close_connection(conn)
            self.pools[conn.name].remove(conn)
    
    async def close_connection(self, conn: MCPConnection):
        """Close a specific connection"""
        try:
            conn.process.terminate()
            await asyncio.sleep(0.1)
            if conn.process.poll() is None:
                conn.process.kill()
            logger.info(f"Closed connection for {conn.name}")
        except Exception as e:
            logger.error(f"Error closing connection: {e}")
    
    async def maintain_pools(self):
        """Maintain pool health and size"""
        while True:
            for mcp_name, pool in self.pools.items():
                # Remove unhealthy connections
                healthy = [c for c in pool if c.is_healthy]
                for conn in pool:
                    if conn not in healthy:
                        await self.close_connection(conn)
                self.pools[mcp_name] = healthy
                
                # Ensure minimum connections
                while len(self.pools[mcp_name]) < self.min_size:
                    conn = await self._create_connection(mcp_name)
                    if conn:
                        self.pools[mcp_name].append(conn)
                
                # Remove idle connections over max
                if len(self.pools[mcp_name]) > self.min_size:
                    idle_conns = sorted([c for c in self.pools[mcp_name] if c.idle_time > 60], 
                                      key=lambda x: x.idle_time, reverse=True)
                    while len(self.pools[mcp_name]) > self.min_size and idle_conns:
                        conn = idle_conns.pop()
                        await self.close_connection(conn)
                        self.pools[mcp_name].remove(conn)
            
            # Log stats
            logger.info(f"Pool stats: {self.stats}")
            logger.info(f"Active connections: {sum(len(p) for p in self.pools.values())}")
            
            await asyncio.sleep(30)  # Maintain every 30 seconds
    
    async def pre_warm(self, mcp_names: List[str]):
        """Pre-warm connections for specified MCPs"""
        tasks = []
        for name in mcp_names:
            for _ in range(self.min_size):
                tasks.append(self.get_connection(name))
        
        connections = await asyncio.gather(*tasks)
        logger.info(f"Pre-warmed {len([c for c in connections if c])} connections")
    
    def get_stats(self) -> Dict:
        """Get pool statistics"""
        stats = self.stats.copy()
        stats['pools'] = {}
        
        for name, pool in self.pools.items():
            stats['pools'][name] = {
                'size': len(pool),
                'healthy': len([c for c in pool if c.is_healthy]),
                'total_requests': sum(c.request_count for c in pool),
                'total_errors': sum(c.error_count for c in pool)
            }
        
        return stats
    
    async def shutdown(self):
        """Shutdown all connections"""
        for pool in self.pools.values():
            for conn in pool:
                await self.close_connection(conn)
        self.pools.clear()
        logger.info("Connection pool shutdown complete")

# Example usage and testing
async def main():
    """Example usage of the connection pool"""
    pool = MCPConnectionPool(min_size=2, max_size=5)
    
    # Pre-warm essential MCPs
    essential_mcps = ['filesystem', 'github', 'memory', 'sequential-thinking']
    await pool.pre_warm(essential_mcps)
    
    # Start maintenance task
    maintenance_task = asyncio.create_task(pool.maintain_pools())
    
    try:
        # Simulate requests
        for i in range(10):
            conn = await pool.get_connection('filesystem')
            if conn:
                # Simulate work
                await asyncio.sleep(0.1)
                await pool.release_connection(conn)
            
            if i % 3 == 0:
                print(f"Stats: {pool.get_stats()}")
            
            await asyncio.sleep(1)
    
    finally:
        maintenance_task.cancel()
        await pool.shutdown()

if __name__ == "__main__":
    asyncio.run(main())
#!/usr/bin/env python3
"""
Network and Inter-Process Communication Optimizer
Optimizes communication between system components for M4 Pro architecture
"""

import asyncio
import logging
import multiprocessing as mp
import socket
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import json
import pickle
import queue
import mmap

try:
    import uvloop
    HAS_UVLOOP = True
except ImportError:
    HAS_UVLOOP = False

try:
    import orjson
    HAS_ORJSON = True
except ImportError:
    HAS_ORJSON = False

logger = logging.getLogger(__name__)


@dataclass
class IPCConfig:
    """Configuration for IPC optimization"""
    
    # Connection pooling
    max_connections_per_endpoint: int = 10
    connection_timeout: float = 5.0
    keepalive_interval: float = 30.0
    
    # Message optimization
    use_compression: bool = True
    compression_threshold: int = 1024  # bytes
    serialization_format: str = "pickle"  # pickle, json, orjson
    
    # Performance tuning
    socket_buffer_size: int = 65536
    tcp_nodelay: bool = True
    socket_reuse_addr: bool = True
    
    # Async optimization
    use_uvloop: bool = True
    max_async_workers: int = 16
    
    # Shared memory
    shared_memory_size_mb: int = 256
    use_mmap_for_large_data: bool = True
    mmap_threshold_kb: int = 100


class MessageSerializer:
    """Optimized message serialization"""
    
    def __init__(self, format_type: str = "pickle"):
        self.format_type = format_type
        
    def serialize(self, data: Any) -> bytes:
        """Serialize data to bytes"""
        
        if self.format_type == "pickle":
            return pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
        elif self.format_type == "json":
            return json.dumps(data).encode('utf-8')
        elif self.format_type == "orjson" and HAS_ORJSON:
            return orjson.dumps(data)
        else:
            # Fallback to pickle
            return pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
    
    def deserialize(self, data: bytes) -> Any:
        """Deserialize bytes to data"""
        
        if self.format_type == "pickle":
            return pickle.loads(data)
        elif self.format_type == "json":
            return json.loads(data.decode('utf-8'))
        elif self.format_type == "orjson" and HAS_ORJSON:
            return orjson.loads(data)
        else:
            # Fallback to pickle
            return pickle.loads(data)


class ConnectionPool:
    """Connection pool for efficient socket reuse"""
    
    def __init__(self, endpoint: str, config: IPCConfig):
        self.endpoint = endpoint
        self.config = config
        self.available_connections = queue.Queue(maxsize=config.max_connections_per_endpoint)
        self.active_connections = set()
        self.lock = threading.Lock()
        
    async def get_connection(self) -> socket.socket:
        """Get a connection from the pool"""
        
        try:
            # Try to get existing connection
            conn = self.available_connections.get_nowait()
            if self._is_connection_healthy(conn):
                with self.lock:
                    self.active_connections.add(conn)
                return conn
            else:
                conn.close()
        except queue.Empty:
            pass
        
        # Create new connection
        conn = await self._create_connection()
        with self.lock:
            self.active_connections.add(conn)
        return conn
    
    def return_connection(self, conn: socket.socket):
        """Return connection to pool"""
        
        with self.lock:
            self.active_connections.discard(conn)
        
        if self._is_connection_healthy(conn):
            try:
                self.available_connections.put_nowait(conn)
            except queue.Full:
                conn.close()
        else:
            conn.close()
    
    async def _create_connection(self) -> socket.socket:
        """Create new optimized connection"""
        
        host, port = self.endpoint.split(':')
        port = int(port)
        
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        
        # Apply optimizations
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, self.config.socket_buffer_size)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, self.config.socket_buffer_size)
        
        if self.config.tcp_nodelay:
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        
        # Set keepalive
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
        
        # Connect with timeout
        sock.settimeout(self.config.connection_timeout)
        await asyncio.get_event_loop().sock_connect(sock, (host, port))
        sock.settimeout(None)  # Reset to blocking
        
        return sock
    
    def _is_connection_healthy(self, conn: socket.socket) -> bool:
        """Check if connection is still healthy"""
        
        try:
            # Send empty data to test connection
            conn.send(b'')
            return True
        except (socket.error, OSError):
            return False
    
    def close_all_connections(self):
        """Close all connections in pool"""
        
        # Close available connections
        while not self.available_connections.empty():
            try:
                conn = self.available_connections.get_nowait()
                conn.close()
            except queue.Empty:
                break
        
        # Close active connections
        with self.lock:
            for conn in list(self.active_connections):
                try:
                    conn.close()
                except:
                    pass
            self.active_connections.clear()


class SharedMemoryManager:
    """Shared memory management for large data transfers"""
    
    def __init__(self, config: IPCConfig):
        self.config = config
        self.shared_segments = {}
        self.lock = threading.Lock()
        
    def create_segment(self, name: str, size_bytes: int) -> mmap.mmap:
        """Create shared memory segment"""
        
        with self.lock:
            if name in self.shared_segments:
                return self.shared_segments[name]
            
            # Create memory-mapped file
            segment = mmap.mmap(-1, size_bytes)
            self.shared_segments[name] = segment
            return segment
    
    def get_segment(self, name: str) -> Optional[mmap.mmap]:
        """Get existing shared memory segment"""
        
        with self.lock:
            return self.shared_segments.get(name)
    
    def write_to_segment(self, name: str, data: bytes, offset: int = 0):
        """Write data to shared memory segment"""
        
        segment = self.get_segment(name)
        if segment:
            segment.seek(offset)
            segment.write(data)
    
    def read_from_segment(self, name: str, size: int, offset: int = 0) -> bytes:
        """Read data from shared memory segment"""
        
        segment = self.get_segment(name)
        if segment:
            segment.seek(offset)
            return segment.read(size)
        return b''
    
    def cleanup(self):
        """Clean up all shared memory segments"""
        
        with self.lock:
            for segment in self.shared_segments.values():
                try:
                    segment.close()
                except:
                    pass
            self.shared_segments.clear()


class OptimizedIPCClient:
    """High-performance IPC client"""
    
    def __init__(self, config: IPCConfig):
        self.config = config
        self.serializer = MessageSerializer(config.serialization_format)
        self.connection_pools = {}
        self.shared_memory = SharedMemoryManager(config)
        
        # Initialize async event loop optimization
        if config.use_uvloop and HAS_UVLOOP:
            try:
                uvloop.install()
                logger.info("UVLoop installed for async optimization")
            except Exception as e:
                logger.warning(f"Could not install UVLoop: {e}")
    
    async def send_message(self, endpoint: str, message: Any) -> Any:
        """Send message to endpoint and get response"""
        
        # Get connection pool for endpoint
        if endpoint not in self.connection_pools:
            self.connection_pools[endpoint] = ConnectionPool(endpoint, self.config)
        
        pool = self.connection_pools[endpoint]
        
        try:
            # Get connection from pool
            conn = await pool.get_connection()
            
            try:
                # Serialize message
                serialized_data = self.serializer.serialize(message)
                
                # Check if we should use shared memory for large data
                if (len(serialized_data) > self.config.mmap_threshold_kb * 1024 and 
                    self.config.use_mmap_for_large_data):
                    response = await self._send_large_message(conn, serialized_data)
                else:
                    response = await self._send_regular_message(conn, serialized_data)
                
                return response
                
            finally:
                pool.return_connection(conn)
                
        except Exception as e:
            logger.error(f"Failed to send message to {endpoint}: {e}")
            raise
    
    async def _send_regular_message(self, conn: socket.socket, data: bytes) -> Any:
        """Send regular message over socket"""
        
        # Send message length first
        message_length = len(data)
        length_bytes = message_length.to_bytes(4, byteorder='big')
        
        loop = asyncio.get_event_loop()
        await loop.sock_sendall(conn, length_bytes)
        await loop.sock_sendall(conn, data)
        
        # Receive response length
        response_length_bytes = await loop.sock_recv(conn, 4)
        response_length = int.from_bytes(response_length_bytes, byteorder='big')
        
        # Receive response data
        response_data = b''
        while len(response_data) < response_length:
            chunk = await loop.sock_recv(conn, min(8192, response_length - len(response_data)))
            if not chunk:
                break
            response_data += chunk
        
        return self.serializer.deserialize(response_data)
    
    async def _send_large_message(self, conn: socket.socket, data: bytes) -> Any:
        """Send large message using shared memory"""
        
        # Create shared memory segment
        segment_name = f"msg_{int(time.time() * 1000000)}"
        segment = self.shared_memory.create_segment(segment_name, len(data) + 1024)
        
        # Write data to shared memory
        self.shared_memory.write_to_segment(segment_name, data)
        
        # Send shared memory reference
        reference = {
            "type": "shared_memory",
            "segment_name": segment_name,
            "data_size": len(data)
        }
        
        return await self._send_regular_message(conn, self.serializer.serialize(reference))
    
    def close(self):
        """Close all connections and cleanup"""
        
        for pool in self.connection_pools.values():
            pool.close_all_connections()
        
        self.shared_memory.cleanup()


class OptimizedIPCServer:
    """High-performance IPC server"""
    
    def __init__(self, host: str, port: int, config: IPCConfig):
        self.host = host
        self.port = port
        self.config = config
        self.serializer = MessageSerializer(config.serialization_format)
        self.shared_memory = SharedMemoryManager(config)
        self.message_handlers = {}
        self.running = False
        self.server_socket = None
        
        # Thread pool for handling connections
        self.executor = ThreadPoolExecutor(max_workers=config.max_async_workers)
        
    def register_handler(self, message_type: str, handler_func):
        """Register message handler"""
        self.message_handlers[message_type] = handler_func
    
    async def start(self):
        """Start the IPC server"""
        
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, self.config.socket_buffer_size)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, self.config.socket_buffer_size)
        
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(100)  # High backlog for concurrent connections
        self.server_socket.setblocking(False)
        
        self.running = True
        logger.info(f"IPC Server started on {self.host}:{self.port}")
        
        loop = asyncio.get_event_loop()
        
        while self.running:
            try:
                client_sock, addr = await loop.sock_accept(self.server_socket)
                # Handle connection in background
                asyncio.create_task(self._handle_client(client_sock, addr))
            except Exception as e:
                if self.running:
                    logger.error(f"Error accepting connection: {e}")
    
    async def _handle_client(self, client_sock: socket.socket, addr: Tuple[str, int]):
        """Handle client connection"""
        
        try:
            loop = asyncio.get_event_loop()
            
            while True:
                # Receive message length
                length_bytes = await loop.sock_recv(client_sock, 4)
                if not length_bytes:
                    break
                
                message_length = int.from_bytes(length_bytes, byteorder='big')
                
                # Receive message data
                message_data = b''
                while len(message_data) < message_length:
                    chunk = await loop.sock_recv(client_sock, min(8192, message_length - len(message_data)))
                    if not chunk:
                        break
                    message_data += chunk
                
                if len(message_data) != message_length:
                    break
                
                # Process message
                try:
                    message = self.serializer.deserialize(message_data)
                    response = await self._process_message(message)
                    
                    # Send response
                    response_data = self.serializer.serialize(response)
                    response_length = len(response_data)
                    length_bytes = response_length.to_bytes(4, byteorder='big')
                    
                    await loop.sock_sendall(client_sock, length_bytes)
                    await loop.sock_sendall(client_sock, response_data)
                    
                except Exception as e:
                    logger.error(f"Error processing message from {addr}: {e}")
                    error_response = {"error": str(e)}
                    error_data = self.serializer.serialize(error_response)
                    error_length = len(error_data)
                    length_bytes = error_length.to_bytes(4, byteorder='big')
                    
                    await loop.sock_sendall(client_sock, length_bytes)
                    await loop.sock_sendall(client_sock, error_data)
                    break
                    
        except Exception as e:
            logger.debug(f"Client {addr} disconnected: {e}")
        finally:
            client_sock.close()
    
    async def _process_message(self, message: Any) -> Any:
        """Process incoming message"""
        
        if isinstance(message, dict) and "type" in message:
            message_type = message["type"]
            
            if message_type == "shared_memory":
                # Handle shared memory reference
                segment_name = message["segment_name"]
                data_size = message["data_size"]
                data = self.shared_memory.read_from_segment(segment_name, data_size)
                actual_message = self.serializer.deserialize(data)
                return await self._process_message(actual_message)
            
            elif message_type in self.message_handlers:
                handler = self.message_handlers[message_type]
                if asyncio.iscoroutinefunction(handler):
                    return await handler(message)
                else:
                    loop = asyncio.get_event_loop()
                    return await loop.run_in_executor(self.executor, handler, message)
        
        # Default handler
        return {"status": "message_received", "timestamp": time.time()}
    
    def stop(self):
        """Stop the IPC server"""
        
        self.running = False
        if self.server_socket:
            self.server_socket.close()
        
        self.executor.shutdown(wait=True)
        self.shared_memory.cleanup()


class NetworkIPCOptimizer:
    """Main network and IPC optimization manager"""
    
    def __init__(self, config: Optional[IPCConfig] = None):
        self.config = config or IPCConfig()
        self.clients = {}
        self.servers = {}
        
        # Performance metrics
        self.message_count = 0
        self.total_latency = 0.0
        self.error_count = 0
        
    def create_client(self, client_id: str) -> OptimizedIPCClient:
        """Create optimized IPC client"""
        
        client = OptimizedIPCClient(self.config)
        self.clients[client_id] = client
        return client
    
    def create_server(self, server_id: str, host: str, port: int) -> OptimizedIPCServer:
        """Create optimized IPC server"""
        
        server = OptimizedIPCServer(host, port, self.config)
        self.servers[server_id] = server
        return server
    
    async def benchmark_communication(self, endpoint: str, iterations: int = 1000) -> Dict[str, float]:
        """Benchmark communication performance"""
        
        client = self.create_client("benchmark_client")
        
        # Warm-up
        test_message = {"type": "ping", "data": "x" * 1024}  # 1KB message
        
        try:
            await client.send_message(endpoint, test_message)
        except Exception as e:
            logger.warning(f"Benchmark warm-up failed: {e}")
            return {"error": str(e)}
        
        # Run benchmark
        start_time = time.time()
        successful_messages = 0
        
        for i in range(iterations):
            try:
                message = {"type": "benchmark", "iteration": i, "data": "x" * 1024}
                response = await client.send_message(endpoint, message)
                successful_messages += 1
            except Exception as e:
                logger.debug(f"Benchmark iteration {i} failed: {e}")
        
        end_time = time.time()
        total_time = end_time - start_time
        
        client.close()
        
        return {
            "total_time_seconds": total_time,
            "successful_messages": successful_messages,
            "failed_messages": iterations - successful_messages,
            "messages_per_second": successful_messages / total_time if total_time > 0 else 0,
            "average_latency_ms": (total_time / successful_messages * 1000) if successful_messages > 0 else 0,
        }
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get communication performance statistics"""
        
        return {
            "total_messages": self.message_count,
            "average_latency_ms": (self.total_latency / self.message_count * 1000) if self.message_count > 0 else 0,
            "error_rate": (self.error_count / self.message_count) if self.message_count > 0 else 0,
            "active_clients": len(self.clients),
            "active_servers": len(self.servers),
            "configuration": {
                "max_connections_per_endpoint": self.config.max_connections_per_endpoint,
                "socket_buffer_size": self.config.socket_buffer_size,
                "serialization_format": self.config.serialization_format,
                "use_compression": self.config.use_compression,
            }
        }
    
    def optimize_for_component_communication(self, component_type: str) -> IPCConfig:
        """Get optimized configuration for specific components"""
        
        base_config = IPCConfig()
        
        if component_type == "einstein_jarvis2":
            # High-frequency ML data exchange
            base_config.max_connections_per_endpoint = 20
            base_config.socket_buffer_size = 131072  # 128KB
            base_config.serialization_format = "pickle"
            base_config.use_compression = False  # CPU intensive
            
        elif component_type == "database_unity":
            # Database queries and results
            base_config.max_connections_per_endpoint = 15
            base_config.socket_buffer_size = 65536
            base_config.serialization_format = "orjson" if HAS_ORJSON else "json"
            base_config.use_compression = True
            
        elif component_type == "api_frontend":
            # API responses to frontend
            base_config.max_connections_per_endpoint = 50
            base_config.socket_buffer_size = 32768
            base_config.serialization_format = "json"
            base_config.use_compression = True
            
        elif component_type == "monitoring":
            # Monitoring and telemetry
            base_config.max_connections_per_endpoint = 5
            base_config.socket_buffer_size = 16384
            base_config.serialization_format = "json"
            base_config.use_compression = False  # Real-time priority
            
        return base_config
    
    def shutdown_all(self):
        """Shutdown all clients and servers"""
        
        # Close all clients
        for client in self.clients.values():
            try:
                client.close()
            except Exception as e:
                logger.error(f"Error closing client: {e}")
        
        # Stop all servers
        for server in self.servers.values():
            try:
                server.stop()
            except Exception as e:
                logger.error(f"Error stopping server: {e}")
        
        self.clients.clear()
        self.servers.clear()


if __name__ == "__main__":
    
    async def test_ipc_optimization():
        """Test IPC optimization system"""
        
        print("🌐 Testing Network/IPC Optimization")
        print("=" * 50)
        
        # Create optimizer
        optimizer = NetworkIPCOptimizer()
        
        # Create test server
        server = optimizer.create_server("test_server", "localhost", 9999)
        
        # Register test handler
        async def ping_handler(message):
            return {"type": "pong", "timestamp": time.time()}
        
        server.register_handler("ping", ping_handler)
        server.register_handler("benchmark", ping_handler)
        
        # Start server in background
        server_task = asyncio.create_task(server.start())
        await asyncio.sleep(1)  # Let server start
        
        try:
            # Run benchmark
            print("🏃 Running communication benchmark...")
            results = await optimizer.benchmark_communication("localhost:9999", 100)
            
            print(f"📊 Benchmark Results:")
            print(f"  Messages per second: {results['messages_per_second']:.1f}")
            print(f"  Average latency: {results['average_latency_ms']:.2f}ms")
            print(f"  Success rate: {results['successful_messages']}/{results['successful_messages'] + results['failed_messages']}")
            
            # Test component-specific optimizations
            print(f"\n🔧 Component-specific optimizations:")
            for component in ["einstein_jarvis2", "database_unity", "api_frontend", "monitoring"]:
                config = optimizer.optimize_for_component_communication(component)
                print(f"  {component}: buffer={config.socket_buffer_size}, format={config.serialization_format}")
            
            # Performance stats
            print(f"\n📈 Performance Statistics:")
            stats = optimizer.get_performance_stats()
            print(f"  Active clients: {stats['active_clients']}")
            print(f"  Active servers: {stats['active_servers']}")
            
        finally:
            # Cleanup
            server.stop()
            optimizer.shutdown_all()
            server_task.cancel()
            
            try:
                await server_task
            except asyncio.CancelledError:
                pass
            
            print("\n✅ IPC optimization test completed")
    
    # Run test
    asyncio.run(test_ipc_optimization())
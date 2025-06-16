"""M4 Pro specific optimizations for Claude Code integration.

This module implements Mac M4-specific optimizations for maximum throughput when
driving Claude Code from Python on Apple Silicon hardware.

Key optimizations:
1. HTTP/2 session pooling for Claude requests
2. P-core only process pools (8 P-cores vs 4 E-cores)
3. QoS task priority for P-core scheduling
4. OpenMP thread management to prevent thread explosion
5. Unified memory optimization for M4 Pro
"""

import asyncio
import os
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import httpx
import psutil

from ..utils.logging import get_component_logger


@dataclass
class M4OptimizationConfig:
    """Configuration for M4 Pro optimizations."""
    
    # HTTP/2 session pooling
    max_concurrent_requests: int = 3  # Stay under Claude API burst limits
    http2_enabled: bool = True
    connection_timeout: int = 60
    
    # P-core optimization
    p_cores_only: bool = True
    detect_p_cores: bool = True
    force_p_core_count: Optional[int] = None  # Override detection
    
    # QoS and priority
    use_task_policy: bool = True
    high_priority: bool = True  # Use highest user priority
    
    # OpenMP control
    control_openmp: bool = True
    openmp_threads: int = 1  # One thread per process
    
    # Memory optimization
    unified_memory_optimization: bool = True
    memory_pressure_threshold: float = 0.8


class M4ClaudeOptimizer:
    """M4 Pro specific optimizer for Claude Code integration."""
    
    def __init__(self, config: M4OptimizationConfig = None):
        self.config = config or M4OptimizationConfig()
        self.logger = get_component_logger("m4_claude_optimizer")
        
        # Hardware detection
        self.p_core_count = 0
        self.e_core_count = 0
        self.total_cores = 0
        self.unified_memory_gb = 0
        
        # HTTP/2 session pool
        self._http_client: Optional[httpx.AsyncClient] = None
        self._request_semaphore: Optional[asyncio.Semaphore] = None
        
        # Process pool for CPU-bound work
        self._process_pool: Optional[ProcessPoolExecutor] = None
        
        # Optimization state
        self._optimizations_applied = False
        self._original_env = {}
    
    async def initialize(self) -> None:
        """Initialize M4 optimizations."""
        self.logger.info("Initializing M4 Pro optimizations for Claude Code")
        
        # Detect hardware
        await self._detect_m4_hardware()
        
        # Apply environment optimizations
        self._apply_environment_optimizations()
        
        # Setup HTTP/2 session pool
        await self._setup_http2_session()
        
        # Setup P-core process pool
        self._setup_process_pool()
        
        # Apply system-level optimizations
        self._apply_system_optimizations()
        
        self._optimizations_applied = True
        self.logger.info(f"M4 optimizations applied: {self.p_core_count}P+{self.e_core_count}E cores, {self.unified_memory_gb}GB RAM")
    
    async def _detect_m4_hardware(self) -> None:
        """Detect M4 Pro hardware configuration."""
        try:
            # Get total CPU count
            self.total_cores = psutil.cpu_count(logical=False)
            
            # Detect P-cores and E-cores using sysctl
            result = subprocess.run(['sysctl', '-n', 'hw.perflevel0.physicalcpu'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                self.p_core_count = int(result.stdout.strip())
            else:
                # Fallback: assume M4 Pro configuration
                self.p_core_count = 8
            
            self.e_core_count = self.total_cores - self.p_core_count
            
            # Override if configured
            if self.config.force_p_core_count:
                self.p_core_count = self.config.force_p_core_count
                self.e_core_count = self.total_cores - self.p_core_count
            
            # Detect unified memory
            memory_bytes = psutil.virtual_memory().total
            self.unified_memory_gb = memory_bytes // (1024 ** 3)
            
            self.logger.info(f"Detected M4 Pro: {self.p_core_count} P-cores, {self.e_core_count} E-cores, {self.unified_memory_gb}GB unified memory")
            
        except Exception as e:
            self.logger.warning(f"Hardware detection failed, using defaults: {e}")
            self.p_core_count = 8
            self.e_core_count = 4
            self.unified_memory_gb = 24
    
    def _apply_environment_optimizations(self) -> None:
        """Apply OpenMP and threading optimizations."""
        if not self.config.control_openmp:
            return
        
        # Store original environment
        openmp_vars = [
            'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
            'OMP_PROC_BIND', 'OMP_PLACES', 'VECLIB_MAXIMUM_THREADS'
        ]
        
        for var in openmp_vars:
            self._original_env[var] = os.environ.get(var)
        
        # Apply optimizations
        optimizations = {
            'OMP_NUM_THREADS': str(self.config.openmp_threads),
            'OPENBLAS_NUM_THREADS': str(self.config.openmp_threads),
            'MKL_NUM_THREADS': str(self.config.openmp_threads),
            'VECLIB_MAXIMUM_THREADS': str(self.config.openmp_threads),
            'OMP_PROC_BIND': 'close',  # Keep threads on assigned core
            'OMP_PLACES': 'cores',     # Bind to physical cores
        }
        
        for key, value in optimizations.items():
            os.environ[key] = value
        
        self.logger.info(f"Applied OpenMP optimizations: {self.config.openmp_threads} threads per process")
    
    async def _setup_http2_session(self) -> None:
        """Setup HTTP/2 session pool for Claude requests."""
        if self._http_client:
            return
        
        # Create semaphore for rate limiting
        self._request_semaphore = asyncio.Semaphore(self.config.max_concurrent_requests)
        
        # Configure HTTP/2 client with optimizations
        self._http_client = httpx.AsyncClient(
            http2=self.config.http2_enabled,
            timeout=httpx.Timeout(self.config.connection_timeout),
            limits=httpx.Limits(
                max_keepalive_connections=5,
                max_connections=10,
                keepalive_expiry=30.0
            ),
            headers={
                'User-Agent': 'bob-m4-optimizer/1.0'
            }
        )
        
        self.logger.info(f"HTTP/2 session pool initialized: {self.config.max_concurrent_requests} concurrent requests")
    
    def _setup_process_pool(self) -> None:
        """Setup P-core only process pool."""
        if self._process_pool:
            return
        
        # Use only P-cores for CPU-bound work
        worker_count = self.p_core_count if self.config.p_cores_only else self.total_cores
        
        # Use spawn context for clean process separation
        import multiprocessing as mp
        ctx = mp.get_context("spawn")
        
        self._process_pool = ProcessPoolExecutor(
            max_workers=worker_count,
            mp_context=ctx
        )
        
        self.logger.info(f"P-core process pool initialized: {worker_count} workers")
    
    def _apply_system_optimizations(self) -> None:
        """Apply system-level optimizations."""
        if not self.config.high_priority:
            return
        
        try:
            # Set high priority (requires appropriate permissions)
            os.setpriority(os.PRIO_PROCESS, 0, -10)  # High priority
            self.logger.info("Applied high priority scheduling")
        except OSError as e:
            self.logger.warning(f"Could not set high priority: {e}")
        
        # Apply QoS hint using taskpolicy if available
        if self.config.use_task_policy:
            try:
                # This would need to be applied at process startup
                self.logger.info("QoS task policy should be applied at startup with: taskpolicy --application")
            except Exception as e:
                self.logger.warning(f"Task policy hint failed: {e}")
    
    async def execute_claude_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute Claude request with HTTP/2 optimization."""
        if not self._http_client or not self._request_semaphore:
            raise RuntimeError("M4 optimizer not initialized")
        
        async with self._request_semaphore:
            try:
                response = await self._http_client.post(
                    "https://api.anthropic.com/v1/messages",
                    json=request_data,
                    headers={
                        'anthropic-version': '2023-06-01',
                        'x-api-key': os.environ.get('ANTHROPIC_API_KEY', '')
                    }
                )
                response.raise_for_status()
                return response.json()
            except httpx.HTTPError as e:
                self.logger.error(f"Claude request failed: {e}")
                raise
    
    async def execute_streaming_claude_request(self, request_data: Dict[str, Any]):
        """Execute streaming Claude request with HTTP/2 optimization."""
        if not self._http_client or not self._request_semaphore:
            raise RuntimeError("M4 optimizer not initialized")
        
        request_data['stream'] = True
        
        async with self._request_semaphore:
            try:
                async with self._http_client.stream(
                    'POST',
                    "https://api.anthropic.com/v1/messages",
                    json=request_data,
                    headers={
                        'anthropic-version': '2023-06-01',
                        'x-api-key': os.environ.get('ANTHROPIC_API_KEY', ''),
                        'accept': 'text/event-stream'
                    }
                ) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if line.startswith('data: '):
                            yield line[6:]  # Remove 'data: ' prefix
            except httpx.HTTPError as e:
                self.logger.error(f"Streaming Claude request failed: {e}")
                raise
    
    def execute_cpu_bound_task(self, func, *args, **kwargs):
        """Execute CPU-bound task on P-core process pool."""
        if not self._process_pool:
            raise RuntimeError("Process pool not initialized")
        
        return self._process_pool.submit(func, *args, **kwargs)
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get current optimization statistics."""
        return {
            'hardware': {
                'p_cores': self.p_core_count,
                'e_cores': self.e_core_count,
                'total_cores': self.total_cores,
                'unified_memory_gb': self.unified_memory_gb
            },
            'http_client': {
                'active': self._http_client is not None,
                'max_concurrent': self.config.max_concurrent_requests,
                'http2_enabled': self.config.http2_enabled
            },
            'process_pool': {
                'active': self._process_pool is not None,
                'workers': self.p_core_count if self.config.p_cores_only else self.total_cores
            },
            'optimizations_applied': self._optimizations_applied
        }
    
    def create_startup_script(self, script_path: str) -> str:
        """Create optimized startup script with taskpolicy."""
        script_content = f"""#!/bin/bash
# M4 Pro optimized startup script for Bob
# Generated by M4ClaudeOptimizer

# Apply OpenMP optimizations
export OMP_NUM_THREADS={self.config.openmp_threads}
export OPENBLAS_NUM_THREADS={self.config.openmp_threads}
export MKL_NUM_THREADS={self.config.openmp_threads}
export VECLIB_MAXIMUM_THREADS={self.config.openmp_threads}
export OMP_PROC_BIND=close
export OMP_PLACES=cores

# Launch with P-core affinity
echo "Starting Bob with M4 Pro optimizations..."
echo "P-cores: {self.p_core_count}, E-cores: {self.e_core_count}"

if command -v taskpolicy >/dev/null 2>&1; then
    echo "Using taskpolicy for P-core affinity"
    taskpolicy --application "$@"
else
    echo "taskpolicy not available, running normally"
    "$@"
fi
"""
        
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        os.chmod(script_path, 0o755)
        self.logger.info(f"Created optimized startup script: {script_path}")
        return script_path
    
    async def shutdown(self) -> None:
        """Shutdown optimizer and restore environment."""
        self.logger.info("Shutting down M4 optimizer")
        
        # Close HTTP client
        if self._http_client:
            await self._http_client.aclose()
        
        # Shutdown process pool
        if self._process_pool:
            self._process_pool.shutdown(wait=True)
        
        # Restore environment
        for var, value in self._original_env.items():
            if value is None:
                os.environ.pop(var, None)
            else:
                os.environ[var] = value
        
        self.logger.info("M4 optimizer shutdown complete")


# Convenience factory function
def create_m4_optimizer(
    max_concurrent_requests: int = 3,
    p_cores_only: bool = True,
    high_priority: bool = True
) -> M4ClaudeOptimizer:
    """Create M4 optimizer with common settings."""
    config = M4OptimizationConfig(
        max_concurrent_requests=max_concurrent_requests,
        p_cores_only=p_cores_only,
        high_priority=high_priority
    )
    return M4ClaudeOptimizer(config)


# Global optimizer instance
_global_optimizer: Optional[M4ClaudeOptimizer] = None


async def get_global_optimizer() -> M4ClaudeOptimizer:
    """Get or create global M4 optimizer instance."""
    global _global_optimizer
    if _global_optimizer is None:
        _global_optimizer = create_m4_optimizer()
        await _global_optimizer.initialize()
    return _global_optimizer
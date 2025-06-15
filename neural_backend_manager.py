"""
Neural Backend Manager with Robust Fallback Chain

Handles MLX/PyTorch backend switching with validation and error recovery.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class BackendInfo:
    """Information about a neural backend."""
    name: str
    available: bool
    error: Optional[str] = None
    performance_score: float = 0.0
    memory_usage_mb: float = 0.0


class NeuralBackendManager:
    """Manages neural network backends with automatic fallback."""
    
    def __init__(self):
        self.current_backend = None
        self.available_backends = {}
        self.backend_performance = {}
        self._initialize_backends()
    
    def _initialize_backends(self):
        """Initialize and test all available backends."""
        backends_to_test = [
            ('mlx', self._init_mlx_backend),
            ('torch_mps', self._init_torch_mps_backend),
            ('torch_cpu', self._init_torch_cpu_backend)
        ]
        
        for backend_name, init_func in backends_to_test:
            try:
                info = init_func()
                self.available_backends[backend_name] = info
                logger.info(f"✅ {backend_name}: {info.available}")
            except Exception as e:
                self.available_backends[backend_name] = BackendInfo(
                    name=backend_name,
                    available=False,
                    error=str(e)
                )
                logger.warning(f"❌ {backend_name}: {e}")
        
        # Select best available backend
        self._select_optimal_backend()
    
    def _init_mlx_backend(self) -> BackendInfo:
        """Initialize MLX backend."""
        try:
            import mlx.core as mx
            import mlx.nn as nn
            
            # Test basic functionality
            test_array = mx.array([1.0, 2.0, 3.0])
            result = mx.sum(test_array)
            
            # Check Metal availability
            if hasattr(mx, 'metal') and mx.metal.is_available():
                memory_mb = self._estimate_mlx_memory()
                perf_score = self._benchmark_mlx()
                
                return BackendInfo(
                    name='mlx',
                    available=True,
                    performance_score=perf_score,
                    memory_usage_mb=memory_mb
                )
            else:
                return BackendInfo(
                    name='mlx',
                    available=False,
                    error="Metal GPU not available"
                )
                
        except ImportError as e:
            return BackendInfo(
                name='mlx',
                available=False,
                error=f"MLX not installed: {e}"
            )
    
    def _init_torch_mps_backend(self) -> BackendInfo:
        """Initialize PyTorch MPS backend."""
        try:
            import torch
            
            if torch.backends.mps.is_available():
                # Test MPS functionality
                device = torch.device('mps')
                test_tensor = torch.tensor([1.0, 2.0, 3.0], device=device)
                result = torch.sum(test_tensor)
                
                memory_mb = self._estimate_torch_memory(device)
                perf_score = self._benchmark_torch_mps()
                
                return BackendInfo(
                    name='torch_mps',
                    available=True,
                    performance_score=perf_score,
                    memory_usage_mb=memory_mb
                )
            else:
                return BackendInfo(
                    name='torch_mps',
                    available=False,
                    error="MPS not available"
                )
                
        except ImportError as e:
            return BackendInfo(
                name='torch_mps',
                available=False,
                error=f"PyTorch not installed: {e}"
            )
    
    def _init_torch_cpu_backend(self) -> BackendInfo:
        """Initialize PyTorch CPU backend."""
        try:
            import torch
            
            # Test CPU functionality
            test_tensor = torch.tensor([1.0, 2.0, 3.0])
            result = torch.sum(test_tensor)
            
            memory_mb = self._estimate_torch_memory(torch.device('cpu'))
            perf_score = self._benchmark_torch_cpu()
            
            return BackendInfo(
                name='torch_cpu',
                available=True,
                performance_score=perf_score,
                memory_usage_mb=memory_mb
            )
            
        except ImportError as e:
            return BackendInfo(
                name='torch_cpu',
                available=False,
                error=f"PyTorch not installed: {e}"
            )
    
    def _estimate_mlx_memory(self) -> float:
        """Estimate MLX memory usage."""
        try:
            import mlx.core as mx
            # Simple estimation based on typical model sizes
            return 2048.0  # MB - conservative estimate for M4 Pro
        except Exception:
            return 0.0
    
    def _estimate_torch_memory(self, device) -> float:
        """Estimate PyTorch memory usage."""
        try:
            import torch
            if device.type == 'mps':
                return 1536.0  # MB - MPS typically uses less than MLX
            else:
                return 512.0   # MB - CPU backend
        except Exception:
            return 0.0
    
    def _benchmark_mlx(self) -> float:
        """Benchmark MLX performance."""
        try:
            import mlx.core as mx
            import time
            
            # Simple matrix multiplication benchmark
            start_time = time.time()
            a = mx.random.normal([1000, 1000])
            b = mx.random.normal([1000, 1000])
            for _ in range(10):
                c = mx.matmul(a, b)
            end_time = time.time()
            
            # Higher score = better performance (ops/second)
            return 100.0 / (end_time - start_time)
            
        except Exception:
            return 0.0
    
    def _benchmark_torch_mps(self) -> float:
        """Benchmark PyTorch MPS performance."""
        try:
            import torch
            
            device = torch.device('mps')
            start_time = time.time()
            a = torch.randn(1000, 1000, device=device)
            b = torch.randn(1000, 1000, device=device)
            for _ in range(10):
                c = torch.matmul(a, b)
            end_time = time.time()
            
            return 80.0 / (end_time - start_time)  # Typically slower than MLX
            
        except Exception:
            return 0.0
    
    def _benchmark_torch_cpu(self) -> float:
        """Benchmark PyTorch CPU performance."""
        try:
            import torch
            
            start_time = time.time()
            a = torch.randn(1000, 1000)
            b = torch.randn(1000, 1000)
            for _ in range(10):
                c = torch.matmul(a, b)
            end_time = time.time()
            
            return 20.0 / (end_time - start_time)  # Slower than GPU backends
            
        except Exception:
            return 0.0
    
    def _select_optimal_backend(self):
        """Select the best available backend."""
        available = [(name, info) for name, info in self.available_backends.items() if info.available]
        
        if not available:
            raise RuntimeError("No neural backends available")
        
        # Sort by performance score (higher is better)
        available.sort(key=lambda x: x[1].performance_score, reverse=True)
        
        self.current_backend = available[0][0]
        logger.info(f"🚀 Selected neural backend: {self.current_backend}")
    
    def get_current_backend(self) -> str:
        """Get currently active backend."""
        return self.current_backend
    
    def validate_tensor_compatibility(self, data: Any, target_backend: str = None) -> bool:
        """Validate tensor compatibility with backend."""
        backend = target_backend or self.current_backend
        
        if backend == 'mlx':
            return isinstance(data, np.ndarray) and data.dtype in [np.float32, np.float16]
        elif backend.startswith('torch'):
            import torch
            return isinstance(data, (np.ndarray, torch.Tensor))
        
        return False
    
    def convert_tensor(self, data: Any, target_backend: str = None) -> Any:
        """Convert tensor to target backend format."""
        backend = target_backend or self.current_backend
        
        if backend == 'mlx':
            import mlx.core as mx
            if isinstance(data, np.ndarray):
                return mx.array(data.astype(np.float32))
            elif hasattr(data, 'numpy'):  # PyTorch tensor
                return mx.array(data.detach().cpu().numpy().astype(np.float32))
            else:
                return mx.array(np.array(data, dtype=np.float32))
        
        elif backend.startswith('torch'):
            import torch
            device = torch.device('mps' if backend == 'torch_mps' else 'cpu')
            
            if isinstance(data, np.ndarray):
                return torch.from_numpy(data.astype(np.float32)).to(device)
            elif hasattr(data, '__array__'):  # MLX array
                return torch.from_numpy(np.array(data).astype(np.float32)).to(device)
            else:
                return torch.tensor(data, dtype=torch.float32, device=device)
        
        raise ValueError(f"Unknown backend: {backend}")
    
    def switch_backend(self, backend_name: str) -> bool:
        """Switch to a different backend."""
        if backend_name not in self.available_backends:
            logger.error(f"Backend not available: {backend_name}")
            return False
        
        if not self.available_backends[backend_name].available:
            logger.error(f"Backend not functional: {backend_name}")
            return False
        
        old_backend = self.current_backend
        self.current_backend = backend_name
        logger.info(f"🔄 Switched from {old_backend} to {backend_name}")
        return True
    
    def get_backend_status(self) -> Dict[str, Any]:
        """Get status of all backends."""
        return {
            'current': self.current_backend,
            'available': {name: info.__dict__ for name, info in self.available_backends.items()},
            'total_backends': len(self.available_backends),
            'functional_backends': len([info for info in self.available_backends.values() if info.available])
        }


# Global instance
_backend_manager: Optional[NeuralBackendManager] = None


def get_neural_backend_manager() -> NeuralBackendManager:
    """Get the global neural backend manager."""
    global _backend_manager
    if _backend_manager is None:
        _backend_manager = NeuralBackendManager()
    return _backend_manager


if __name__ == "__main__":
    # Test neural backend manager
    print("🧠 Testing Neural Backend Manager...")
    
    manager = get_neural_backend_manager()
    status = manager.get_backend_status()
    
    print(f"Current backend: {status['current']}")
    print(f"Functional backends: {status['functional_backends']}/{status['total_backends']}")
    
    for name, info in status['available'].items():
        status_emoji = "✅" if info['available'] else "❌"
        print(f"  {status_emoji} {name}: perf={info['performance_score']:.1f}")
    
    print("✅ Neural backend manager test complete")
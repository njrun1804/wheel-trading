"""Isolated neural network worker process.

Runs PyTorch/MLX models in separate process to prevent blocking
the main async event loop.
"""
import logging
import multiprocessing as mp
import queue
import signal
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

from ..core.device_router import Backend, OperationType

logger = logging.getLogger(__name__)


@dataclass
class NeuralRequest:
    """Request for neural computation."""
    id: str
    type: str
    data: np.ndarray
    metadata: Dict[str, Any]


@dataclass
class NeuralResponse:
    """Response from neural computation."""
    id: str
    result: np.ndarray
    compute_time_ms: float
    backend_used: str


class NeuralWorkerProcess:
    """Isolated process for neural network operations."""

    def __init__(self, worker_id: int = 0):
        self.worker_id = worker_id
        self.process = None
        self.request_queue = mp.Queue(maxsize = 100)
        self.response_queue = mp.Queue(maxsize = 100)
        self.shutdown_event = mp.Event()
        self.ready_event = mp.Event()

    def start(self):
        """Start the worker process."""
        self.process = mp.Process(target = self._run_worker, args=(self.
            worker_id, self.request_queue, self.response_queue, self.
            shutdown_event, self.ready_event), daemon = True)
        self.process.start()
        if not self.ready_event.wait(timeout = 60):
            raise RuntimeError(
                f"Neural worker {self.worker_id} failed to initialize")
        logger.info(f"Neural worker {self.worker_id} started")

    def stop(self):
        """Stop the worker process."""
        self.shutdown_event.set()
        if self.process:
            self.process.join(timeout = 5)
            if self.process.is_alive():
                self.process.terminate()
                self.process.join()
        logger.info(f"Neural worker {self.worker_id} stopped")

    @staticmethod
    def _run_worker(worker_id: int, request_queue: mp.Queue, response_queue:
        mp.Queue, shutdown_event: mp.Event, ready_event: mp.Event):
        """Main worker loop - runs in separate process."""
        signal.signal(signal.SIGTERM, lambda s, f: shutdown_event.set())
        logger.info(f"Worker {worker_id} initializing models...")
        models = NeuralModels()
        ready_event.set()
        logger.info(f"Worker {worker_id} ready")
        while not shutdown_event.is_set():
            try:
                request = request_queue.get(timeout = 5.0)
                start_time = time.perf_counter()
                result = models.process(request)
                compute_time = (time.perf_counter() - start_time) * 1000
                response = NeuralResponse(id = request.id, result = result,
                    compute_time_ms = compute_time, backend_used = models.
                    current_backend)
                response_queue.put(response)
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}")
                if 'request' in locals():
                    response_queue.put(NeuralResponse(id = request.id, result = np.array([]), compute_time_ms=-1, backend_used=
                        'error'))
        logger.info(f"Worker {worker_id} shutting down")


class NeuralModels:
    """Container for neural models with MLX/PyTorch routing."""

    def __init__(self):
        from ..core.device_router import DeviceRouter
        self.router = DeviceRouter()
        self.current_backend = None
        self._init_models()

    def _init_models(self):
        """Initialize models based on available backends."""
        mlx_backend = self.router.route(OperationType.NEURAL_FORWARD)
        if mlx_backend == Backend.MLX_GPU:
            try:
                self._init_mlx_models()
                self.current_backend = 'mlx'
                logger.info('Using MLX models on Metal GPU')
                return
            except Exception as e:
                logger.warning(f"MLX init failed: {e}, falling back")
        try:
            self._init_torch_models()
            self.current_backend = 'torch'
            logger.info('Using PyTorch models')
        except Exception as e:
            logger.error(f"PyTorch init failed: {e}")
            raise

    def _init_mlx_models(self):
        """Initialize MLX models."""
        import mlx.core as mx
        import mlx.nn as nn


        class ValueNetwork(nn.Module):

            def __init__(self):
                super().__init__()
                self.layers = [nn.Linear(768, 512), nn.ReLU(), nn.Linear(
                    512, 256), nn.ReLU(), nn.Linear(256, 1)]

            def __call__(self, x):
                for layer in self.layers:
                    x = layer(x)
                return mx.sigmoid(x)


        class PolicyNetwork(nn.Module):

            def __init__(self):
                super().__init__()
                self.layers = [nn.Linear(768, 512), nn.ReLU(), nn.Linear(
                    512, 256), nn.ReLU(), nn.Linear(256, 50)]

            def __call__(self, x):
                for layer in self.layers:
                    x = layer(x)
                return mx.softmax(x, axis=-1)
        self.value_net = ValueNetwork()
        self.policy_net = PolicyNetwork()
        self.mx = mx

    def _init_torch_models(self):
        """Initialize PyTorch models."""
        import torch
        import torch.nn as nn


        class ValueNetwork(nn.Module):

            def __init__(self):
                super().__init__()
                self.net = nn.Sequential(nn.Linear(768, 512), nn.ReLU(), nn
                    .Linear(512, 256), nn.ReLU(), nn.Linear(256, 1), nn.
                    Sigmoid())

            def forward(self, x):
                return self.net(x)


        class PolicyNetwork(nn.Module):

            def __init__(self):
                super().__init__()
                self.net = nn.Sequential(nn.Linear(768, 512), nn.ReLU(), nn
                    .Linear(512, 256), nn.ReLU(), nn.Linear(256, 50))

            def forward(self, x):
                return torch.softmax(self.net(x), dim=-1)
        device = self.router.get_device(self.router.route(OperationType.
            NEURAL_FORWARD))
        self.value_net = ValueNetwork().to(device)
        self.policy_net = PolicyNetwork().to(device)
        self.device = device
        self.torch = torch

    def process(self, request: NeuralRequest) ->np.ndarray:
        """Process a neural request."""
        if self.current_backend == 'mlx':
            return self._process_mlx(request)
        else:
            return self._process_torch(request)

    def _process_mlx(self, request: NeuralRequest) ->np.ndarray:
        """Process with MLX."""
        data = self.mx.array(request.data)
        if request.type == 'value':
            result = self.value_net(data)
        elif request.type == 'policy':
            result = self.policy_net(data)
        else:
            raise ValueError(f"Unknown request type: {request.type}")
        self.mx.eval(result)
        return np.array(result)

    def _process_torch(self, request: NeuralRequest) ->np.ndarray:
        """Process with PyTorch."""
        data = self.torch.tensor(request.data, device = self.device)
        with self.torch.no_grad():
            if request.type == 'value':
                result = self.value_net(data)
            elif request.type == 'policy':
                result = self.policy_net(data)
            else:
                raise ValueError(f"Unknown request type: {request.type}")
        if self.device.type == 'mps':
            self.torch.mps.synchronize()
        return result.cpu().numpy()


class NeuralWorkerPool:
    """Pool of neural workers for load balancing."""

    def __init__(self, num_workers: int = 2):
        self.workers = []
        self.current_worker = 0
        for i in range(num_workers):
            worker = NeuralWorkerProcess(worker_id = i)
            worker.start()
            self.workers.append(worker)

    def get_next_worker(self) ->NeuralWorkerProcess:
        """Round-robin worker selection."""
        worker = self.workers[self.current_worker]
        self.current_worker = (self.current_worker + 1) % len(self.workers)
        return worker

    async def compute_async(self, request_type: str, data: np.ndarray,
        metadata: Optional[Dict]=None) ->np.ndarray:
        """Async interface for neural computation."""
        import asyncio
        request = NeuralRequest(id = str(uuid.uuid4()), type = request_type,
            data = data, metadata = metadata or {})
        worker = self.get_next_worker()
        try:
            queue_size = worker.request_queue.qsize()
            if queue_size > 50:
                logger.warning(
                    f"Worker {worker.worker_id} queue depth: {queue_size}")
        except NotImplementedError as e:
            logger.debug(f"Ignored exception in {'neural_worker.py'}: {e}")
        try:
            worker.request_queue.put(request, timeout = 1.0)
        except queue.Full:
            raise RuntimeError(f"Worker {worker.worker_id} queue is full")
        start_time = time.perf_counter()
        timeout = 30.0
        while time.perf_counter() - start_time < timeout:
            try:
                response = worker.response_queue.get_nowait()
                if response.id == request.id:
                    if response.compute_time_ms < 0:
                        raise RuntimeError('Neural computation failed')
                    logger.debug(
                        f"Neural compute took {response.compute_time_ms:.1f}ms on {response.backend_used}"
                        )
                    return response.result
                else:
                    worker.response_queue.put(response)
            except queue.Empty:
                await asyncio.sleep(0.001)
            if not worker.process.is_alive():
                raise RuntimeError(
                    f"Worker {worker.worker_id} died during computation")
        raise TimeoutError(f"Neural computation timed out after {timeout}s")

    def shutdown(self):
        """Shutdown all workers."""
        for worker in self.workers:
            worker.stop()

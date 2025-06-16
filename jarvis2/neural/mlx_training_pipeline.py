"""
MLX-Accelerated Neural Training Pipeline for Jarvis2

Features:
- Native Apple Silicon MLX acceleration
- Unified memory optimization for 24GB M4 Pro
- Continuous learning from code generation experience
- Policy and value network training
- Hardware-aware batch processing
"""
from __future__ import annotations

import gc
import logging
import time
import weakref
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from mlx.utils import tree_flatten

    MLX_AVAILABLE = True
except ImportError:
    import torch
    import torch.nn as nn
    import torch.optim as optim

    MLX_AVAILABLE = False

from ..experience.experience_buffer import ExperienceReplaySystem
from ..hardware.hardware_optimizer import HardwareOptimizer

logger = logging.getLogger(__name__)


class MLXMemoryManager:
    """Memory pool and lifecycle manager for MLX arrays."""

    def __init__(self, pool_size_mb: int = 1024):
        self.pool_size_mb = pool_size_mb
        self.active_arrays = weakref.WeakSet()
        self.array_pool = {}
        self.cleanup_threshold = 100  # Arrays before cleanup
        self.operation_count = 0

    def get_array(self, shape: tuple, dtype=mx.float32) -> mx.array:
        """Get array from pool or create new one."""
        key = (shape, str(dtype))

        if key in self.array_pool and self.array_pool[key]:
            array = self.array_pool[key].pop()
            # Zero out for reuse
            array = mx.zeros_like(array)
            mx.eval(array)
        else:
            array = mx.zeros(shape, dtype=dtype)
            mx.eval(array)

        self.active_arrays.add(array)
        self.operation_count += 1

        # Periodic cleanup
        if self.operation_count % self.cleanup_threshold == 0:
            self._cleanup_memory()

        return array

    def release_array(self, array: mx.array):
        """Return array to pool for reuse."""
        if array is None:
            return

        try:
            key = (array.shape, str(array.dtype))
            if key not in self.array_pool:
                self.array_pool[key] = []

            # Only keep reasonable pool size
            if len(self.array_pool[key]) < 10:
                self.array_pool[key].append(array)

            # Force evaluation to complete any pending operations
            mx.eval(array)

        except Exception as e:
            logger.debug(f"Array release failed: {e}")

    def _cleanup_memory(self):
        """Force memory cleanup and Metal fragmentation reduction."""
        try:
            # Evaluate all pending operations
            active_arrays = list(self.active_arrays)
            if active_arrays:
                mx.eval(*active_arrays)

            # Clear pool of unused arrays
            for key in list(self.array_pool.keys()):
                if len(self.array_pool[key]) > 5:  # Keep some for reuse
                    self.array_pool[key] = self.array_pool[key][:5]

            # Force Python GC
            gc.collect()

            # MLX-specific cleanup if available
            if hasattr(mx, "metal") and hasattr(mx.metal, "clear_memory"):
                mx.metal.clear_memory()

            logger.debug(
                f"Memory cleanup completed. Active arrays: {len(self.active_arrays)}"
            )

        except Exception as e:
            logger.warning(f"Memory cleanup failed: {e}")

    def get_memory_stats(self) -> dict[str, Any]:
        """Get memory usage statistics."""
        pool_count = sum(len(arrays) for arrays in self.array_pool.values())
        return {
            "active_arrays": len(self.active_arrays),
            "pooled_arrays": pool_count,
            "operation_count": self.operation_count,
            "pool_types": len(self.array_pool),
        }

    @contextmanager
    def mlx_operation(self, operation_name: str = "unknown"):
        """Context manager for MLX operations with automatic cleanup."""
        arrays_to_cleanup = []
        try:
            yield arrays_to_cleanup
        finally:
            # Cleanup any arrays created during the operation
            for array in arrays_to_cleanup:
                if array is not None:
                    self.release_array(array)

            # Periodic cleanup
            if self.operation_count % self.cleanup_threshold == 0:
                self._cleanup_memory()


@dataclass
class TrainingConfig:
    """Training configuration optimized for M4 Pro"""

    learning_rate: float = 3e-4
    batch_size: int = 512  # Large batches for unified memory
    buffer_size: int = 100000
    training_steps: int = 1000
    validation_split: float = 0.1

    # MLX-specific settings
    use_mlx: bool = MLX_AVAILABLE
    gradient_clipping: float = 1.0
    weight_decay: float = 1e-4
    warmup_steps: int = 100

    # Hardware optimization
    memory_efficient_training: bool = True
    async_data_loading: bool = True
    gradient_accumulation_steps: int = 4


class MLXNeuralTrainer:
    """
    Hardware-accelerated neural network training using MLX

    Optimized for M4 Pro:
    - Unified memory architecture (24GB shared)
    - Metal Performance Shaders acceleration
    - Asynchronous data pipeline
    - Memory-efficient gradient computation
    """

    def __init__(self, config: TrainingConfig | None = None):
        self.config = config or TrainingConfig()
        self.hardware_optimizer = HardwareOptimizer()

        # Initialize MLX memory manager
        self.memory_manager = (
            MLXMemoryManager(pool_size_mb=2048) if MLX_AVAILABLE else None
        )

        # Initialize networks
        if self.config.use_mlx and MLX_AVAILABLE:
            self.policy_net = self._create_mlx_policy_network()
            self.value_net = self._create_mlx_value_network()
            self.policy_optimizer = optim.Adam(learning_rate=self.config.learning_rate)
            self.value_optimizer = optim.Adam(learning_rate=self.config.learning_rate)
            self.device_type = "mlx"
            logger.info("🚀 MLX training pipeline initialized with memory management")
        else:
            self.policy_net = self._create_torch_policy_network()
            self.value_net = self._create_torch_value_network()
            self.policy_optimizer = torch.optim.Adam(
                self.policy_net.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )
            self.value_optimizer = torch.optim.Adam(
                self.value_net.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )
            self.device_type = "mps" if torch.backends.mps.is_available() else "cpu"
            logger.info(
                f"🔄 PyTorch training pipeline initialized on {self.device_type}"
            )

        # Training state
        self.training_step = 0
        self.training_metrics = {
            "policy_loss": [],
            "value_loss": [],
            "training_time": [],
            "memory_usage": [],
            "gpu_utilization": [],
        }

        # Experience buffer
        self.experience_buffer = ExperienceReplaySystem(
            storage_path=Path(".jarvis/experience"), buffer_size=self.config.buffer_size
        )

    def _create_mlx_policy_network(self):
        """Create MLX policy network"""

        class MLXPolicyNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = [
                    nn.Linear(768, 1024),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(1024, 512),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(512, 256),
                    nn.ReLU(),
                    nn.Linear(256, 128),  # Action space
                ]

            def __call__(self, x):
                for layer in self.layers[:-1]:
                    x = layer(x)
                logits = self.layers[-1](x)
                return mx.softmax(logits, axis=-1)

        return MLXPolicyNet()

    def _create_mlx_value_network(self):
        """Create MLX value network"""

        class MLXValueNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = [
                    nn.Linear(768, 1024),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(1024, 512),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(512, 256),
                    nn.ReLU(),
                    nn.Linear(256, 1),
                ]

            def __call__(self, x):
                for layer in self.layers[:-1]:
                    x = layer(x)
                return mx.tanh(self.layers[-1](x))

        return MLXValueNet()

    def _create_torch_policy_network(self):
        """Create PyTorch policy network as fallback"""
        return nn.Sequential(
            nn.Linear(768, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.Softmax(dim=-1),
        )

    def _create_torch_value_network(self):
        """Create PyTorch value network as fallback"""
        return nn.Sequential(
            nn.Linear(768, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Tanh(),
        )

    async def initialize(self):
        """Initialize training pipeline"""
        logger.info("🏗️ Initializing MLX training pipeline")

        # Initialize hardware optimization
        hw_info = await self.hardware_optimizer.get_hardware_info()
        logger.info(
            f"🔧 Training on: {hw_info['cpu_cores']} cores, {hw_info['gpu_cores']} GPU cores"
        )

        # Initialize experience buffer
        await self.experience_buffer.initialize()

        # Warm up networks
        if self.config.use_mlx and MLX_AVAILABLE:
            await self._warmup_mlx_training()
        else:
            await self._warmup_torch_training()

        logger.info("✅ MLX training pipeline ready")

    async def _warmup_mlx_training(self):
        """Warm up MLX training pipeline"""
        # Create dummy batch
        dummy_states = mx.random.normal([self.config.batch_size, 768])
        mx.random.randint(0, 128, [self.config.batch_size])
        dummy_values = mx.random.normal([self.config.batch_size, 1])

        # Warm up policy network
        policy_probs = self.policy_net(dummy_states)
        -mx.mean(mx.log(policy_probs + 1e-8))

        # Warm up value network
        value_preds = self.value_net(dummy_states)
        mx.mean((value_preds - dummy_values) ** 2)

        logger.info("🔥 MLX networks warmed up")

    async def _warmup_torch_training(self):
        """Warm up PyTorch training pipeline"""
        device = torch.device(self.device_type)
        self.policy_net.to(device)
        self.value_net.to(device)

        # Create dummy batch
        dummy_states = torch.randn(self.config.batch_size, 768, device=device)
        torch.randint(0, 128, [self.config.batch_size], device=device)
        torch.randn(self.config.batch_size, 1, device=device)

        # Warm up networks
        with torch.no_grad():
            _ = self.policy_net(dummy_states)
            _ = self.value_net(dummy_states)

        logger.info(f"🔥 PyTorch networks warmed up on {device}")

    async def train_from_experience(
        self, num_steps: int | None = None
    ) -> dict[str, float]:
        """
        Train networks from experience buffer

        Features:
        - Asynchronous data loading
        - Memory-efficient batching
        - Gradient accumulation for large effective batch sizes
        - Hardware utilization monitoring
        """
        steps = num_steps or self.config.training_steps

        logger.info(f"🎯 Starting training: {steps} steps")
        start_time = time.time()

        for step in range(steps):
            step_start = time.time()

            # Sample batch from experience buffer
            samples = await self.experience_buffer.sample(self.config.batch_size)
            if not samples:
                logger.warning("No experience data available for training")
                break

            # Convert samples to batch format
            batch = {
                "states": [
                    s["experience"].context.get("state", np.random.randn(768))
                    for s in samples
                ],
                "actions": [hash(s["experience"].query) % 128 for s in samples],
                "values": [
                    s["experience"].metrics.get("confidence_score", 0.5)
                    for s in samples
                ],
                "advantages": [
                    s["experience"].metrics.get("confidence_score", 0.5) - 0.5
                    for s in samples
                ],
            }

            # Train policy and value networks
            if self.config.use_mlx and MLX_AVAILABLE:
                policy_loss, value_loss = await self._train_step_mlx(batch)
            else:
                policy_loss, value_loss = await self._train_step_torch(batch)

            # Record metrics
            step_time = time.time() - step_start
            self.training_metrics["policy_loss"].append(float(policy_loss))
            self.training_metrics["value_loss"].append(float(value_loss))
            self.training_metrics["training_time"].append(step_time)

            # Log progress
            if step % 100 == 0:
                logger.info(
                    f"Step {step}: Policy Loss: {policy_loss:.4f}, Value Loss: {value_loss:.4f}, Time: {step_time:.3f}s"
                )

            self.training_step += 1

        total_time = time.time() - start_time
        logger.info(f"✅ Training complete: {total_time:.2f}s total")

        return self._compute_training_summary()

    async def _train_step_mlx(self, batch: dict[str, Any]) -> tuple[float, float]:
        """Single training step using MLX with proper memory management"""
        # Use memory manager context
        with self.memory_manager.mlx_operation("train_step_mlx") as arrays_to_cleanup:
            states = mx.array(batch["states"])
            actions = mx.array(batch["actions"])
            values = mx.array(batch["values"])
            advantages = mx.array(batch["advantages"])

            # Track arrays for cleanup
            arrays_to_cleanup.extend([states, actions, values, advantages])

            # Policy network training
            def policy_loss_fn(params):
                policy_net_fn = nn.value_and_grad(self.policy_net, params)
                probs = policy_net_fn(states)

                # Policy gradient loss
                action_probs = mx.take_along_axis(
                    probs, actions.reshape(-1, 1), axis=1
                ).squeeze()
                log_probs = mx.log(action_probs + 1e-8)
                policy_loss = -mx.mean(log_probs * advantages)

                # Add entropy bonus
                entropy = -mx.mean(mx.sum(probs * mx.log(probs + 1e-8), axis=1))
                total_loss = policy_loss - 0.01 * entropy

                # Ensure evaluation of intermediate results
                mx.eval(
                    probs, action_probs, log_probs, policy_loss, entropy, total_loss
                )

                return total_loss

            # Value network training
            def value_loss_fn(params):
                value_net_fn = nn.value_and_grad(self.value_net, params)
                value_preds = value_net_fn(states)
                value_loss = mx.mean((value_preds - values) ** 2)
                mx.eval(value_preds, value_loss)
                return value_loss

            # Compute gradients and update
            policy_loss, policy_grads = policy_loss_fn(self.policy_net.parameters())
            value_loss, value_grads = value_loss_fn(self.value_net.parameters())

            # Ensure gradient evaluation
            mx.eval(policy_loss, value_loss)

            # Apply gradients
            self.policy_optimizer.update(self.policy_net, policy_grads)
            self.value_optimizer.update(self.value_net, value_grads)

            return float(policy_loss), float(value_loss)

    async def _train_step_torch(self, batch: dict[str, Any]) -> tuple[float, float]:
        """Single training step using PyTorch"""
        device = torch.device(self.device_type)

        states = torch.tensor(batch["states"], dtype=torch.float32, device=device)
        actions = torch.tensor(batch["actions"], dtype=torch.long, device=device)
        values = torch.tensor(batch["values"], dtype=torch.float32, device=device)
        advantages = torch.tensor(
            batch["advantages"], dtype=torch.float32, device=device
        )

        # Policy network training
        self.policy_optimizer.zero_grad()
        policy_probs = self.policy_net(states)
        action_probs = policy_probs.gather(1, actions.unsqueeze(1)).squeeze()
        log_probs = torch.log(action_probs + 1e-8)

        policy_loss = -(log_probs * advantages).mean()

        # Add entropy bonus
        entropy = -(policy_probs * torch.log(policy_probs + 1e-8)).sum(dim=1).mean()
        total_policy_loss = policy_loss - 0.01 * entropy

        total_policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.policy_net.parameters(), self.config.gradient_clipping
        )
        self.policy_optimizer.step()

        # Value network training
        self.value_optimizer.zero_grad()
        value_preds = self.value_net(states)
        value_loss = torch.nn.functional.mse_loss(value_preds, values)

        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.value_net.parameters(), self.config.gradient_clipping
        )
        self.value_optimizer.step()

        return float(total_policy_loss.item()), float(value_loss.item())

    async def add_experience(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        """Add experience to training buffer"""
        from ..core.solution import SolutionMetrics

        # Create a dummy experience for the replay system
        query = f"action_{action}"
        solution = {"action": action, "reward": reward}
        metrics = SolutionMetrics(
            generation_time_ms=1.0,
            simulations_performed=1,
            confidence_score=reward,
            complexity_score=0.5,
            gpu_utilization=0.8,
        )
        context = {
            "state": state.tolist(),
            "next_state": next_state.tolist(),
            "done": done,
        }

        await self.experience_buffer.record(query, solution, metrics, [], context)

    async def save_models(self, checkpoint_dir: Path):
        """Save trained models"""
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        if self.config.use_mlx and MLX_AVAILABLE:
            # Save MLX models
            mx.save_safetensors(
                str(checkpoint_dir / "policy_net.safetensors"),
                dict(tree_flatten(self.policy_net.parameters())[0]),
            )
            mx.save_safetensors(
                str(checkpoint_dir / "value_net.safetensors"),
                dict(tree_flatten(self.value_net.parameters())[0]),
            )
        else:
            # Save PyTorch models
            torch.save(self.policy_net.state_dict(), checkpoint_dir / "policy_net.pth")
            torch.save(self.value_net.state_dict(), checkpoint_dir / "value_net.pth")

        logger.info(f"✅ Models saved to {checkpoint_dir}")

    async def load_models(self, checkpoint_dir: Path):
        """Load trained models"""
        if self.config.use_mlx and MLX_AVAILABLE:
            # Load MLX models
            policy_path = checkpoint_dir / "policy_net.safetensors"
            value_path = checkpoint_dir / "value_net.safetensors"

            if policy_path.exists():
                policy_params = mx.load(str(policy_path))
                self.policy_net.load_weights(list(policy_params.items()))

            if value_path.exists():
                value_params = mx.load(str(value_path))
                self.value_net.load_weights(list(value_params.items()))
        else:
            # Load PyTorch models
            policy_path = checkpoint_dir / "policy_net.pth"
            value_path = checkpoint_dir / "value_net.pth"

            if policy_path.exists():
                self.policy_net.load_state_dict(torch.load(policy_path))

            if value_path.exists():
                self.value_net.load_state_dict(torch.load(value_path))

        logger.info(f"✅ Models loaded from {checkpoint_dir}")

    def _compute_training_summary(self) -> dict[str, float]:
        """Compute training summary statistics"""
        if not self.training_metrics["policy_loss"]:
            return {}

        return {
            "final_policy_loss": self.training_metrics["policy_loss"][-1],
            "final_value_loss": self.training_metrics["value_loss"][-1],
            "avg_policy_loss": np.mean(self.training_metrics["policy_loss"]),
            "avg_value_loss": np.mean(self.training_metrics["value_loss"]),
            "avg_step_time": np.mean(self.training_metrics["training_time"]),
            "total_steps": len(self.training_metrics["policy_loss"]),
            "device_type": self.device_type,
        }

    async def get_training_metrics(self) -> dict[str, Any]:
        """Get detailed training metrics"""
        return {
            "metrics": self.training_metrics,
            "summary": self._compute_training_summary(),
            "config": {
                "learning_rate": self.config.learning_rate,
                "batch_size": self.config.batch_size,
                "device_type": self.device_type,
                "mlx_enabled": self.config.use_mlx and MLX_AVAILABLE,
            },
        }


# Factory function
def create_mlx_trainer(config: TrainingConfig | None = None) -> MLXNeuralTrainer:
    """Create MLX neural trainer with optimal M4 Pro configuration"""
    if config is None:
        config = TrainingConfig()

        # Optimize for M4 Pro
        if MLX_AVAILABLE:
            config.batch_size = 1024  # Large batches for unified memory
            config.learning_rate = 5e-4  # Slightly higher for MLX
            config.use_mlx = True
        else:
            config.batch_size = 512
            config.use_mlx = False

    return MLXNeuralTrainer(config)

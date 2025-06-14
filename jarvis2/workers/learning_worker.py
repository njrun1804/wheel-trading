"""Background learning worker for E-cores.

Runs continuous learning and model updates on efficiency cores
without blocking the main search processes.
"""
import json
import logging
import multiprocessing as mp
import pickle
import queue
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Experience:
    """Single experience for learning."""
    query: str
    code: str
    context: Dict[str, Any]
    value: float
    policy_actions: List[str]
    timestamp: float
    metadata: Dict[str, Any]


@dataclass
class LearningBatch:
    """Batch of experiences for training."""
    experiences: List[Experience]
    batch_id: str
    priority: float = 1.0


@dataclass
class ModelUpdate:
    """Updated model weights."""
    model_type: str
    weights: Dict[str, np.ndarray]
    version: int
    metrics: Dict[str, float]


class ReplayBuffer:
    """Prioritized experience replay buffer."""

    def __init__(self, capacity: int = 100000):
        self.capacity = capacity
        self.buffer: List[Experience] = []
        self.priorities: np.ndarray = np.zeros(capacity)
        self.position = 0
        self.size = 0

    def add(self, experience: Experience, priority: float = 1.0):
        """Add experience to buffer."""
        if self.size < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience
        self.priorities[self.position] = priority
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) ->List[Experience]:
        """Sample batch with prioritized replay."""
        if self.size == 0:
            return []
        probs = self.priorities[:self.size] / self.priorities[:self.size].sum()
        indices = np.random.choice(self.size, size = min(batch_size, self.
            size), p = probs, replace = False)
        return [self.buffer[i] for i in indices]

    def update_priorities(self, indices: List[int], priorities: List[float]):
        """Update priorities after training."""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority

    def get_stats(self) ->Dict[str, Any]:
        """Get buffer statistics."""
        return {'size': self.size, 'capacity': self.capacity, 'position':
            self.position, 'mean_priority': float(np.mean(self.priorities[:
            self.size])) if self.size > 0 else 0}


class ModelTrainer:
    """Trains value and policy networks on experiences."""

    def __init__(self, backend: str='mlx'):
        self.backend = backend
        self.value_version = 0
        self.policy_version = 0
        self._init_models()

    def _init_models(self):
        """Initialize models based on backend."""
        if self.backend == 'mlx':
            self._init_mlx_models()
        else:
            self._init_torch_models()

    def _init_mlx_models(self):
        """Initialize MLX models for training."""
        try:
            import mlx.core as mx
            import mlx.nn as nn
            import mlx.optimizers as optim
            self.value_model = nn.Sequential(nn.Linear(768, 512), nn.ReLU(),
                nn.Dropout(0.1), nn.Linear(512, 256), nn.ReLU(), nn.Dropout
                (0.1), nn.Linear(256, 1))
            self.policy_model = nn.Sequential(nn.Linear(768, 512), nn.ReLU(
                ), nn.Dropout(0.1), nn.Linear(512, 256), nn.ReLU(), nn.
                Dropout(0.1), nn.Linear(256, 50))
            self.value_optimizer = optim.Adam(learning_rate = 0.0001)
            self.policy_optimizer = optim.Adam(learning_rate = 0.0001)
            self.mx = mx
            logger.info('MLX models initialized for learning')
        except ImportError:
            logger.warning('MLX not available, falling back to PyTorch')
            self.backend = 'torch'
            self._init_torch_models()

    def _init_torch_models(self):
        """Initialize PyTorch models for training."""
        import torch
        import torch.nn as nn
        import torch.optim as optim
        self.value_model = nn.Sequential(nn.Linear(768, 512), nn.ReLU(), nn
            .Dropout(0.1), nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, 1), nn.Sigmoid())
        self.policy_model = nn.Sequential(nn.Linear(768, 512), nn.ReLU(),
            nn.Dropout(0.1), nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.1
            ), nn.Linear(256, 50))
        device = torch.device('cpu')
        self.value_model = self.value_model.to(device)
        self.policy_model = self.policy_model.to(device)
        self.value_optimizer = optim.Adam(self.value_model.parameters(), lr = 0.0001)
        self.policy_optimizer = optim.Adam(self.policy_model.parameters(),
            lr = 0.0001)
        self.torch = torch
        self.device = device
        logger.info('PyTorch models initialized for learning')

    def train_batch(self, batch: LearningBatch) ->Tuple[ModelUpdate,
        ModelUpdate]:
        """Train on a batch of experiences."""
        if self.backend == 'mlx':
            return self._train_mlx(batch)
        else:
            return self._train_torch(batch)

    def _train_mlx(self, batch: LearningBatch) ->Tuple[ModelUpdate, ModelUpdate
        ]:
        """Train with MLX."""
        features = []
        value_targets = []
        policy_targets = []
        for exp in batch.experiences:
            feature = self._create_feature_vector(exp.query)
            features.append(feature)
            value_targets.append(exp.value)
            policy_target = np.zeros(50)
            for i, action in enumerate(exp.policy_actions[:50]):
                policy_target[i] = 1.0 / len(exp.policy_actions)
            policy_targets.append(policy_target)
        X = self.mx.array(np.stack(features))
        y_value = self.mx.array(np.array(value_targets).reshape(-1, 1))
        y_policy = self.mx.array(np.stack(policy_targets))
        value_loss = 0
        for _ in range(10):

            def value_loss_fn(model, X, y):
                pred = model(X)
                return self.mx.mean((pred - y) ** 2)
            loss, grads = self.mx.value_and_grad(value_loss_fn)(self.
                value_model, X, y_value)
            self.value_optimizer.update(self.value_model, grads)
            self.mx.eval(self.value_model.parameters())
            value_loss = float(loss)
        policy_loss = 0
        for _ in range(10):

            def policy_loss_fn(model, X, y):
                logits = model(X)
                probs = self.mx.softmax(logits, axis=-1)
                return -self.mx.mean(self.mx.sum(y * self.mx.log(probs + 
                    1e-08), axis=-1))
            loss, grads = self.mx.value_and_grad(policy_loss_fn)(self.
                policy_model, X, y_policy)
            self.policy_optimizer.update(self.policy_model, grads)
            self.mx.eval(self.policy_model.parameters())
            policy_loss = float(loss)
        self.value_version += 1
        self.policy_version += 1
        value_update = ModelUpdate(model_type='value', weights = self.
            _extract_mlx_weights(self.value_model), version = self.
            value_version, metrics={'loss': value_loss, 'batch_size': len(
            batch.experiences)})
        policy_update = ModelUpdate(model_type='policy', weights = self.
            _extract_mlx_weights(self.policy_model), version = self.
            policy_version, metrics={'loss': policy_loss, 'batch_size': len
            (batch.experiences)})
        return value_update, policy_update

    def _train_torch(self, batch: LearningBatch) ->Tuple[ModelUpdate,
        ModelUpdate]:
        """Train with PyTorch."""
        features = []
        value_targets = []
        policy_targets = []
        for exp in batch.experiences:
            feature = self._create_feature_vector(exp.query)
            features.append(feature)
            value_targets.append(exp.value)
            policy_target = np.zeros(50)
            for i, action in enumerate(exp.policy_actions[:50]):
                policy_target[i] = 1.0 / len(exp.policy_actions)
            policy_targets.append(policy_target)
        X = self.torch.tensor(np.stack(features), dtype = self.torch.float32)
        y_value = self.torch.tensor(value_targets, dtype = self.torch.float32
            ).unsqueeze(1)
        y_policy = self.torch.tensor(np.stack(policy_targets), dtype = self.
            torch.float32)
        self.value_model.train()
        value_losses = []
        for _ in range(10):
            self.value_optimizer.zero_grad()
            pred = self.value_model(X)
            loss = self.torch.nn.functional.mse_loss(pred, y_value)
            loss.backward()
            self.value_optimizer.step()
            value_losses.append(loss.item())
        self.policy_model.train()
        policy_losses = []
        for _ in range(10):
            self.policy_optimizer.zero_grad()
            logits = self.policy_model(X)
            loss = self.torch.nn.functional.cross_entropy(logits, y_policy)
            loss.backward()
            self.policy_optimizer.step()
            policy_losses.append(loss.item())
        self.value_version += 1
        self.policy_version += 1
        value_update = ModelUpdate(model_type='value', weights = self.
            _extract_torch_weights(self.value_model), version = self.
            value_version, metrics={'loss': np.mean(value_losses),
            'batch_size': len(batch.experiences)})
        policy_update = ModelUpdate(model_type='policy', weights = self.
            _extract_torch_weights(self.policy_model), version = self.
            policy_version, metrics={'loss': np.mean(policy_losses),
            'batch_size': len(batch.experiences)})
        return value_update, policy_update

    def _create_feature_vector(self, query: str) ->np.ndarray:
        """Create feature vector from query using real embeddings."""
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from core.code_embeddings import LightweightCodeEmbedder, MLXCodeEmbedder
        if not hasattr(self, '_embedder'):
            try:
                self._embedder = MLXCodeEmbedder()
                logger.info('Using MLX embedder for learning')
            except Exception as e:
                self._embedder = LightweightCodeEmbedder(vector_dim = 768)
                logger.info('Using lightweight embedder for learning')
        embedding = self._embedder.embed(query)
        if len(embedding) != 768:
            if len(embedding) > 768:
                embedding = embedding[:768]
            else:
                padding = np.zeros(768 - len(embedding), dtype = np.float32)
                embedding = np.concatenate([embedding, padding])
        return embedding.astype(np.float32)

    def _extract_mlx_weights(self, model) ->Dict[str, np.ndarray]:
        """Extract weights from MLX model."""
        weights = {}
        for name, param in model.named_parameters():
            weights[name] = np.array(param)
        return weights

    def _extract_torch_weights(self, model) ->Dict[str, np.ndarray]:
        """Extract weights from PyTorch model."""
        weights = {}
        for name, param in model.named_parameters():
            weights[name] = param.detach().cpu().numpy()
        return weights


class LearningWorker:
    """Background learning worker running on E-cores."""

    def __init__(self, model_dir: str='.jarvis/models'):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents = True, exist_ok = True)
        self.experience_queue = mp.Queue(maxsize = 1000)
        self.update_queue = mp.Queue(maxsize = 10)
        self.shutdown_event = mp.Event()
        self.process = None

    def start(self):
        """Start learning process."""
        self.process = mp.Process(target = self._run_worker, args=(self.
            experience_queue, self.update_queue, self.shutdown_event, self.
            model_dir), daemon = True)
        self.process.start()
        logger.info('Learning worker started')

    def stop(self):
        """Stop learning process."""
        self.shutdown_event.set()
        if self.process:
            self.process.join(timeout = 5)
            if self.process.is_alive():
                self.process.terminate()
        logger.info('Learning worker stopped')

    def add_experience(self, experience: Experience):
        """Add experience for learning (non-blocking)."""
        try:
            self.experience_queue.put_nowait(experience)
        except queue.Full:
            logger.warning('Experience queue full, dropping experience')

    def get_model_update(self) ->Optional[ModelUpdate]:
        """Get latest model update if available."""
        try:
            return self.update_queue.get_nowait()
        except queue.Empty:
            return None

    @staticmethod
    def _run_worker(experience_queue: mp.Queue, update_queue: mp.Queue,
        shutdown_event: mp.Event, model_dir: Path):
        """Main learning loop."""
        replay_buffer = ReplayBuffer()
        trainer = ModelTrainer()
        batch_size = 32
        train_interval = 100
        experience_count = 0
        logger.info('Learning worker initialized')
        while not shutdown_event.is_set():
            try:
                collected = 0
                while collected < train_interval:
                    try:
                        exp = experience_queue.get(timeout = 0.1)
                        replay_buffer.add(exp)
                        collected += 1
                        experience_count += 1
                    except queue.Empty:
                        if shutdown_event.is_set():
                            break
                        continue
                if replay_buffer.size >= batch_size:
                    experiences = replay_buffer.sample(batch_size)
                    batch = LearningBatch(experiences = experiences, batch_id = f"batch_{experience_count}")
                    start_time = time.perf_counter()
                    value_update, policy_update = trainer.train_batch(batch)
                    train_time = time.perf_counter() - start_time
                    if experience_count % 1000 == 0:
                        LearningWorker._save_models(model_dir, value_update,
                            policy_update)
                    try:
                        update_queue.put_nowait(value_update)
                        update_queue.put_nowait(policy_update)
                    except queue.Full:
                        logger.warning('Update queue full')
                    logger.info(
                        f"Training completed in {train_time * 1000:.1f}ms, buffer size: {replay_buffer.size}"
                        )
            except Exception as e:
                logger.error(f"Learning worker error: {e}")
        logger.info('Learning worker shutting down')

    @staticmethod
    def _save_models(model_dir: Path, value_update: ModelUpdate,
        policy_update: ModelUpdate):
        """Save model updates to disk."""
        try:
            value_path = model_dir / f"value_v{value_update.version}.pkl"
            with open(value_path, 'wb') as f:
                pickle.dump(value_update, f)
            policy_path = model_dir / f"policy_v{policy_update.version}.pkl"
            with open(policy_path, 'wb') as f:
                pickle.dump(policy_update, f)
            meta_path = model_dir / 'metadata.json'
            with open(meta_path, 'w') as f:
                json.dump({'value_version': value_update.version,
                    'policy_version': policy_update.version,
                    'value_metrics': value_update.metrics, 'policy_metrics':
                    policy_update.metrics}, f, indent = 2)
            logger.info(
                f"Models saved: value_v{value_update.version}, policy_v{policy_update.version}"
                )
        except Exception as e:
            logger.error(f"Failed to save models: {e}")

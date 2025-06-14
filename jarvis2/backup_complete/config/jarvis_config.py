"""Centralized configuration for Jarvis2 to replace hardcoded values."""
from dataclasses import dataclass
from typing import Optional
import json
from pathlib import Path


@dataclass
class NeuralConfig:
    """Configuration for neural networks."""
    # Dimensions
    embedding_dim: int = 768
    hidden_dim: int = 512
    value_hidden_dim: int = 256
    policy_output_dim: int = 50  # Number of action types
    
    # Training
    learning_rate: float = 1e-4
    batch_size: int = 32
    gradient_clip_norm: float = 1.0
    dropout_rate: float = 0.1
    
    # Architecture
    num_layers: int = 3
    num_heads: int = 8  # For transformer models
    
    
@dataclass
class SearchConfig:
    """Configuration for MCTS search."""
    # Search parameters
    default_simulations: int = 2000
    exploration_constant: float = 1.414
    
    # Batch processing
    batch_size: int = 256
    max_tree_depth: int = 50
    
    # Timeouts
    search_timeout_seconds: float = 30.0
    worker_timeout_seconds: float = 60.0
    

@dataclass
class EvaluationConfig:
    """Configuration for code evaluation metrics."""
    # Score weights (should sum to 1.0)
    correctness_weight: float = 0.25
    performance_weight: float = 0.20
    readability_weight: float = 0.20
    maintainability_weight: float = 0.20
    security_weight: float = 0.15
    
    # Thresholds
    min_confidence_threshold: float = 0.3
    complexity_penalty_threshold: int = 10
    
    # Score adjustments
    syntax_valid_bonus: float = 0.1
    has_tests_bonus: float = 0.15
    has_docstring_bonus: float = 0.05
    has_type_hints_bonus: float = 0.05
    

@dataclass
class HardwareConfig:
    """Configuration for hardware utilization."""
    # Memory limits
    max_memory_gb: float = 18.0  # Metal limit on M4 Pro
    memory_reserve_percent: float = 0.15  # Reserve 15% for system
    
    # CPU allocation
    p_core_utilization: float = 0.9  # Use 90% of P-cores
    e_core_utilization: float = 1.0  # Use all E-cores
    
    # GPU
    gpu_memory_fraction: float = 0.75  # Use 75% of unified memory for GPU
    
    # Monitoring
    resource_check_interval: float = 5.0  # seconds
    high_usage_threshold: float = 0.85  # 85% threshold for warnings
    

@dataclass
class IndexConfig:
    """Configuration for vector indexes."""
    # Dimensions
    embedding_dimension: int = 768
    
    # HNSW parameters
    hnsw_m: int = 16  # Number of bi-directional links
    hnsw_ef_construction: int = 200  # Size of dynamic list during construction
    hnsw_ef_search: int = 100  # Size of dynamic list for search
    
    # Limits
    max_index_size: int = 1000000
    cache_size_mb: int = 512
    

@dataclass 
class JarvisConfig:
    """Complete Jarvis2 configuration."""
    neural: NeuralConfig
    search: SearchConfig
    evaluation: EvaluationConfig
    hardware: HardwareConfig
    index: IndexConfig
    
    # System settings
    log_level: str = "INFO"
    enable_profiling: bool = False
    checkpoint_interval: int = 1000
    
    @classmethod
    def default(cls) -> 'JarvisConfig':
        """Create default configuration."""
        return cls(
            neural=NeuralConfig(),
            search=SearchConfig(),
            evaluation=EvaluationConfig(),
            hardware=HardwareConfig(),
            index=IndexConfig()
        )
    
    @classmethod
    def from_file(cls, path: Path) -> 'JarvisConfig':
        """Load configuration from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        
        return cls(
            neural=NeuralConfig(**data.get('neural', {})),
            search=SearchConfig(**data.get('search', {})),
            evaluation=EvaluationConfig(**data.get('evaluation', {})),
            hardware=HardwareConfig(**data.get('hardware', {})),
            index=IndexConfig(**data.get('index', {})),
            log_level=data.get('log_level', 'INFO'),
            enable_profiling=data.get('enable_profiling', False),
            checkpoint_interval=data.get('checkpoint_interval', 1000)
        )
    
    def to_file(self, path: Path):
        """Save configuration to JSON file."""
        data = {
            'neural': self.neural.__dict__,
            'search': self.search.__dict__,
            'evaluation': self.evaluation.__dict__,
            'hardware': self.hardware.__dict__,
            'index': self.index.__dict__,
            'log_level': self.log_level,
            'enable_profiling': self.enable_profiling,
            'checkpoint_interval': self.checkpoint_interval
        }
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def validate(self):
        """Validate configuration values."""
        # Check evaluation weights sum to 1.0
        weights_sum = (
            self.evaluation.correctness_weight +
            self.evaluation.performance_weight +
            self.evaluation.readability_weight +
            self.evaluation.maintainability_weight +
            self.evaluation.security_weight
        )
        if abs(weights_sum - 1.0) > 0.001:
            raise ValueError(f"Evaluation weights must sum to 1.0, got {weights_sum}")
        
        # Check memory limits
        if self.hardware.max_memory_gb <= 0:
            raise ValueError("Max memory must be positive")
        
        # Check dimensions
        if self.neural.embedding_dim <= 0 or self.index.embedding_dimension <= 0:
            raise ValueError("Embedding dimensions must be positive")
        
        # Ensure neural and index dimensions match
        if self.neural.embedding_dim != self.index.embedding_dimension:
            raise ValueError(
                f"Neural embedding dim ({self.neural.embedding_dim}) must match "
                f"index embedding dim ({self.index.embedding_dimension})"
            )


# Global configuration instance
_config: Optional[JarvisConfig] = None


def get_config() -> JarvisConfig:
    """Get global configuration instance."""
    global _config
    if _config is None:
        # Try to load from file
        config_path = Path.home() / ".jarvis2" / "config.json"
        if config_path.exists():
            _config = JarvisConfig.from_file(config_path)
        else:
            _config = JarvisConfig.default()
        
        # Validate
        _config.validate()
    
    return _config


def set_config(config: JarvisConfig):
    """Set global configuration."""
    global _config
    config.validate()
    _config = config


def reset_config():
    """Reset to default configuration."""
    global _config
    _config = JarvisConfig.default()


# Example usage
if __name__ == "__main__":
    # Create default config
    config = JarvisConfig.default()
    
    # Modify some values
    config.neural.embedding_dim = 1024
    config.search.default_simulations = 5000
    
    # Save to file
    config_path = Path("jarvis2_config.json")
    config.to_file(config_path)
    print(f"Saved configuration to {config_path}")
    
    # Load from file
    loaded_config = JarvisConfig.from_file(config_path)
    print(f"Loaded config with embedding_dim: {loaded_config.neural.embedding_dim}")
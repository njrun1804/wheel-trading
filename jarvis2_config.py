"""
Jarvis2 Configuration - Centralized configuration management
Addresses hardcoded values identified by meta system audit
"""

from dataclasses import dataclass
from typing import Any


@dataclass
class TokenConfig:
    """Token management configuration"""

    max_response_tokens: int = 7500
    reserved_tokens: int = 692
    api_token_limit: int = 8192

    @property
    def safe_token_limit(self) -> int:
        return self.max_response_tokens - self.reserved_tokens


@dataclass
class MCTSConfig:
    """MCTS configuration parameters"""

    max_simulations: int = 2000
    test_simulations: int = 100
    max_simulation_depth: int = 20
    exploration_constant: float = 1.414  # sqrt(2)
    learning_rate: float = 0.01
    noise_factor: float = 0.1


@dataclass
class NeuralConfig:
    """Neural network configuration"""

    input_features: int = 20
    hidden_size: int = 10
    policy_output_size: int = 10
    weight_init_scale: float = 0.1


@dataclass
class HardwareConfig:
    """M4 Pro hardware configuration"""

    p_cores: int = 8
    e_cores: int = 4
    total_cores: int = 12
    gpu_cores: int = 20
    unified_memory_gb: int = 24
    metal_acceleration: bool = True

    @property
    def max_workers(self) -> int:
        return self.total_cores


@dataclass
class PerformanceConfig:
    """Performance and optimization settings"""

    batch_size: int = 4
    max_alternatives: int = 3
    max_options: int = 6
    consultation_timeout_seconds: int = 30
    simulation_timeout_seconds: int = 10


@dataclass
class Jarvis2Config:
    """Complete Jarvis2 configuration"""

    token: TokenConfig
    mcts: MCTSConfig
    neural: NeuralConfig
    hardware: HardwareConfig
    performance: PerformanceConfig

    def __init__(self):
        self.token = TokenConfig()
        self.mcts = MCTSConfig()
        self.neural = NeuralConfig()
        self.hardware = HardwareConfig()
        self.performance = PerformanceConfig()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "token": self.token.__dict__,
            "mcts": self.mcts.__dict__,
            "neural": self.neural.__dict__,
            "hardware": self.hardware.__dict__,
            "performance": self.performance.__dict__,
        }

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "Jarvis2Config":
        """Create config from dictionary"""
        config = cls()

        if "token" in config_dict:
            config.token = TokenConfig(**config_dict["token"])
        if "mcts" in config_dict:
            config.mcts = MCTSConfig(**config_dict["mcts"])
        if "neural" in config_dict:
            config.neural = NeuralConfig(**config_dict["neural"])
        if "hardware" in config_dict:
            config.hardware = HardwareConfig(**config_dict["hardware"])
        if "performance" in config_dict:
            config.performance = PerformanceConfig(**config_dict["performance"])

        return config


# Global configuration instance
CONFIG = Jarvis2Config()


def get_config() -> Jarvis2Config:
    """Get global configuration instance"""
    return CONFIG


def update_config(**kwargs) -> None:
    """Update global configuration"""
    global CONFIG

    for key, value in kwargs.items():
        if hasattr(CONFIG, key):
            setattr(CONFIG, key, value)
        else:
            print(f"Warning: Unknown config key: {key}")


if __name__ == "__main__":
    # Test configuration
    config = get_config()

    print("🔧 Jarvis2 Configuration:")
    print(f"  Token Limit: {config.token.max_response_tokens}")
    print(f"  MCTS Simulations: {config.mcts.max_simulations}")
    print(f"  Neural Features: {config.neural.input_features}")
    print(f"  M4 Cores: {config.hardware.total_cores}")
    print(f"  Max Workers: {config.hardware.max_workers}")

    # Test serialization
    config_dict = config.to_dict()
    restored_config = Jarvis2Config.from_dict(config_dict)

    print(
        f"✅ Config serialization test: {restored_config.token.max_response_tokens == config.token.max_response_tokens}"
    )

#!/usr/bin/env python3
"""
Example usage of the intelligent configuration system.
Demonstrates loading, validation, tracking, and health reporting.
"""

import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config import get_config, get_config_service


def main():
    """Demonstrate configuration system features."""
    print("=" * 60)
    print("WHEEL TRADING CONFIGURATION SYSTEM DEMO")
    print("=" * 60)
    print()

    # Load configuration
    print("1. Loading configuration...")
    try:
        service = get_config_service()
        config = service.get_config()
        print("✓ Configuration loaded successfully")
    except Exception as e:
        print(f"✗ Failed to load configuration: {e}")
        return

    print()

    # Demonstrate environment override
    print("2. Testing environment variable override...")
    print(f"   Current delta_target: {config.strategy.delta_target}")
    print("   Setting WHEEL_STRATEGY__DELTA_TARGET=0.25")
    os.environ["WHEEL_STRATEGY__DELTA_TARGET"] = "0.25"

    # Reload to apply override
    service.reload()
    config = service.get_config()
    print(f"   New delta_target: {config.strategy.delta_target}")
    # Note: ConfigurationService doesn't expose overrides directly
    print(f"   Configuration reloaded with environment overrides")

    print()

    # Display some configuration values
    print("3. Current Configuration Values:")
    print(f"   Trading Mode: {config.trading.mode}")
    print(f"   Risk Profile:")
    print(f"     - Max Position Size: {config.risk.max_position_size:.1%}")
    print(f"     - Kelly Fraction: {config.risk.kelly_fraction:.1%}")
    print(f"     - Max Drawdown: {config.risk.max_drawdown_percent:.1%}")
    print(f"   Strategy:")
    print(f"     - Delta Target: {config.strategy.delta_target}")
    print(f"     - DTE Target: {config.strategy.days_to_expiry_target}")
    print(f"   ML Status: {'Enabled' if config.ml.enabled else 'Disabled'}")

    print()

    # Generate health report
    print("4. Generating Configuration Health Report...")
    print("-" * 60)
    health_report = service.get_health_report()
    print(health_report)

    print()

    # Test validation
    print("5. Testing Configuration Validation...")
    # Note: ConfigurationService validates on load
    print("✓ Configuration is valid (validated on load)")

    print()

    # Configuration statistics
    print("6. Configuration Statistics:")
    stats = service.get_statistics()
    print(f"   Last loaded: {stats['last_loaded']}")
    print(f"   Load count: {stats['load_count']}")
    print(f"   Config path: {service.config_path}")

    print()

    # Using the simple get_config() function
    print("7. Using simplified get_config() function...")
    simple_config = get_config()
    print(f"   Delta target from get_config(): {simple_config.strategy.delta_target}")

    print()
    print("=" * 60)
    print("Configuration system demonstration complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

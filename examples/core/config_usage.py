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

from src.config.loader import get_config, get_config_loader


def main():
    """Demonstrate configuration system features."""
    print("=" * 60)
    print("WHEEL TRADING CONFIGURATION SYSTEM DEMO")
    print("=" * 60)
    print()

    # Load configuration
    print("1. Loading configuration...")
    try:
        loader = get_config_loader(os.getenv("WHEEL_CONFIG_PATH", "config/unified.yaml"))
        config = loader.config
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
    loader = get_config_loader(os.getenv("WHEEL_CONFIG_PATH", "config/unified.yaml"))
    config = loader.config
    print(f"   New delta_target: {config.strategy.delta_target}")
    print(f"   Overrides applied: {len(loader.overrides)}")

    print()

    # Simulate parameter usage
    print("3. Simulating parameter usage...")
    # Simulate frequent use of key parameters
    for _ in range(20):
        loader.track_parameter_usage("strategy.delta_target")
        loader.track_parameter_usage("strategy.days_to_expiry_target")
        loader.track_parameter_usage("risk.max_position_size")

    for _ in range(10):
        loader.track_parameter_usage("risk.kelly_fraction")
        loader.track_parameter_usage("ml.enabled")

    # Simulate parameter impact
    loader.track_parameter_impact("strategy.delta_target", 0.75)
    loader.track_parameter_impact("strategy.days_to_expiry_target", -0.25)
    loader.track_parameter_impact("risk.max_position_size", 0.50)

    print("✓ Tracked usage for 5 parameters")

    print()

    # Display some configuration values
    print("4. Current Configuration Values:")
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
    print("5. Generating Configuration Health Report...")
    print("-" * 60)
    print(loader.generate_health_report())

    print()

    # Show parameter suggestions
    print("6. Parameter Tuning Suggestions:")
    suggestions = loader.suggest_parameter_adjustments()[:3]
    if suggestions:
        for sug in suggestions:
            print(f"   • {sug['parameter']}: {sug['current']} → {sug['suggested']}")
            print(f"     Reason: {sug['reason']}")
            print(f"     Confidence: {sug['confidence']:.1%}")
            print()
    else:
        print("   No suggestions at this time.")

    print()

    # Test validation
    print("7. Testing Configuration Validation...")
    health = loader.health_report
    if health.get("valid"):
        print("✓ Configuration is valid")
    else:
        print("✗ Configuration has errors:")
        for error in health.get("errors", []):
            print(f"   - {error}")

    if health.get("warnings"):
        print("⚠ Warnings:")
        for warning in health["warnings"][:3]:
            print(f"   - {warning}")

    print()

    # Export configuration
    print("8. Exporting configuration...")
    export_path = Path("config_export.yaml")
    loader.export_config(export_path, format="yaml")
    print(f"✓ Configuration exported to {export_path}")

    # Cleanup
    if export_path.exists():
        export_path.unlink()

    print()
    print("=" * 60)
    print("Configuration system demonstration complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

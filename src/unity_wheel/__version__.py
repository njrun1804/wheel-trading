"""Unity Wheel Trading Bot version information."""
from __future__ import annotations


__version__ = "2.0.0"
__version_info__ = (2, 0, 0)

# Component versions
COMPONENT_VERSIONS = {
    "core": "2.0.0",
    "api": "2.0.0",
    "math": "2.0.0",
    "models": "2.0.0",
    "risk": "2.0.0",
    "strategy": "2.0.0",
    "utils": "2.0.0",
    "data": "1.0.0",
    "metrics": "1.0.0",
    "monitoring": "1.0.0",
    "observability": "1.0.0",
    "diagnostics": "2.0.0",
}

# API version for external interfaces
API_VERSION = "v2"

# Minimum compatible versions
MIN_COMPATIBLE_VERSIONS = {
    "python": "3.10",
    "numpy": "1.20.0",
    "pandas": "1.3.0",
    "scipy": "1.7.0",
    "pydantic": "2.0.0",
}


def get_version_string() -> str:
    """Get formatted version string with all component info."""
    lines = [
        f"Unity Wheel Trading Bot v{__version__}",
        "",
        "Component Versions:",
    ]

    for component, version in sorted(COMPONENT_VERSIONS.items()):
        lines.append(f"  {component:<15} {version}")

    return "\n".join(lines)
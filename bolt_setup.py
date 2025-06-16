#!/usr/bin/env python3
"""
Setup configuration for Bolt - 8-Agent Hardware-Accelerated Problem Solver

A complete system for solving complex programming problems using 8 parallel
Claude Code agents with M4 Pro hardware acceleration.
"""

import sys
from pathlib import Path

from setuptools import find_packages, setup

# Ensure we're using Python 3.12+
if sys.version_info < (3, 12):
    print("Bolt requires Python 3.12 or newer")
    sys.exit(1)


# Read version from bolt/__init__.py
def get_version():
    """Extract version from bolt/__init__.py"""
    init_file = Path(__file__).parent / "bolt" / "__init__.py"
    if init_file.exists():
        with open(init_file) as f:
            for line in f:
                if line.startswith("__version__"):
                    return line.split("=")[1].strip().strip('"').strip("'")
    return "1.0.0"


# Read long description from README
def get_long_description():
    """Get long description from README file"""
    readme_file = Path(__file__).parent / "bolt" / "README.md"
    if readme_file.exists():
        with open(readme_file, encoding="utf-8") as f:
            return f.read()
    return "Bolt - 8-Agent Hardware-Accelerated Problem Solver"


# Core dependencies
INSTALL_REQUIRES = [
    "click>=8.1.7",
    "rich>=13.7.1",
    "typer>=0.9.0",
    "psutil>=5.9.0",
    "pydantic>=2.7.0",
    "pydantic-settings>=2.6.1",
    "python-dotenv>=1.0.1",
    "pyyaml>=6.0.2",
    "aiohttp>=3.10.11",
    "httpx>=0.27.0",
    "nest-asyncio>=1.6.0",
    "tenacity>=8.2.0",
    # Data processing
    "numpy>=1.26.4,<2.0",
    "pandas>=2.2.3",
    "scipy>=1.14.0",
    # Storage and performance
    "duckdb>=0.10.0",
    "pyarrow>=14.0.0",
    "bottleneck>=1.3.7",
    "numexpr>=2.8.0",
    # Vector search and optimization (FAISS is the winner on M4 Pro)
    "faiss-cpu>=1.8.0",
    # Observability
    "opentelemetry-api>=1.22.0",
    "opentelemetry-sdk>=1.22.0",
    "opentelemetry-exporter-otlp>=1.22.0",
    # Configuration
    "attrs>=25.3.0",
    "typing-extensions>=4.14.0",
]

# Platform-specific dependencies
EXTRAS_REQUIRE = {
    # M4 Pro / Apple Silicon optimizations
    "macos": [
        "torch>=2.4.0; sys_platform=='darwin' and platform_machine=='arm64'",
        "mlx>=0.0.1; sys_platform=='darwin' and platform_machine=='arm64'",
    ],
    # Development dependencies
    "dev": [
        "pytest>=7.0.0",
        "pytest-asyncio>=0.21.0",
        "pytest-cov>=4.0.0",
        "black>=23.0.0",
        "isort>=5.12.0",
        "mypy>=1.0.0",
        "flake8>=6.0.0",
        "pre-commit>=3.0.0",
    ],
    # Full trading system integration
    "trading": [
        "databento>=0.37.0",
        "databento-dbn>=0.37.0",
        "yfinance>=0.2.37",
        "fredapi>=0.5.1",
        "pandas_market_calendars>=4.0.0",
        "exchange_calendars>=4.2.0",
        "quantlib>=1.33; sys_platform=='darwin'",
        "scikit-learn>=1.5.0",
        "polars>=0.20.0",
    ],
    # Observability and monitoring
    "monitoring": [
        "logfire>=3.18.0",
        "prometheus-client>=0.17.0",
        "grafana-api>=1.0.3",
    ],
    # GPU acceleration
    "gpu": [
        "cupy-cuda12x>=12.0.0; sys_platform=='linux'",
        "pynvml>=11.4.1",
    ],
}

# All extras combined
EXTRAS_REQUIRE["all"] = [
    dep
    for extra_deps in EXTRAS_REQUIRE.values()
    for dep in extra_deps
    if isinstance(dep, str)
]

setup(
    name="bolt-solver",
    version=get_version(),
    description="8-Agent Hardware-Accelerated Problem Solver",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    author="Generated with Claude Code",
    author_email="noreply@anthropic.com",
    url="https://github.com/njrun1804/wheel-trading",
    # Package configuration
    packages=find_packages(include=["bolt", "bolt.*"]),
    python_requires=">=3.12",
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
    # Entry points for CLI
    entry_points={
        "console_scripts": [
            "bolt=bolt.cli.main:main",
            "bolt-solve=bolt.cli.solve:main",
            "bolt-monitor=bolt.cli.monitor:main",
            "bolt-bench=bolt.cli.benchmark:main",
        ],
    },
    # Package data
    package_data={
        "bolt": [
            "*.md",
            "*.yaml",
            "*.json",
            "config/*.yaml",
            "config/*.json",
            "shaders/*.metal",
        ],
    },
    include_package_data=True,
    # Classification
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: System :: Hardware :: Symmetric Multi-processing",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: POSIX :: Linux",
        "Environment :: Console",
    ],
    # Keywords for discovery
    keywords=[
        "ai",
        "agents",
        "parallel-processing",
        "hardware-acceleration",
        "m4-pro",
        "gpu",
        "mlx",
        "claude",
        "problem-solving",
        "automation",
        "trading",
        "financial-modeling",
        "optimization",
    ],
    # Project URLs
    project_urls={
        "Bug Reports": "https://github.com/njrun1804/wheel-trading/issues",
        "Source": "https://github.com/njrun1804/wheel-trading",
        "Documentation": "https://github.com/njrun1804/wheel-trading/blob/main/bolt/README.md",
    },
    # Ensure we can install in development mode
    zip_safe=False,
)

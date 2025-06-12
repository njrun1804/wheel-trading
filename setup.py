#!/usr/bin/env python3
"""Minimal setup.py for pip compatibility with Poetry-based project."""

from setuptools import find_packages, setup

# Read version from __version__.py
version_dict = {}
with open("src/unity_wheel/__version__.py") as fp:
    exec(fp.read(), version_dict)

setup(
    name="unity-wheel",
    version=version_dict["__version__"],
    description="Personal wheel trading recommendation system",
    author="Mike Edwards",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.11",
    install_requires=[
        # Core dependencies - minimal set for CI
        "aiohttp",
        "annotated-types",
        "attrs",
        "click",
        "cryptography",
        "duckdb",
        "numpy",
        "pandas",
        "pydantic>=2.7.0",
        "pydantic-settings>=2.6.1",
        "python-dateutil",
        "python-dotenv",
        "pytz",
        "pyyaml",
        "requests",
        "rich",
        "scikit-learn",
        "scipy",
        "tenacity",
        "tqdm",
        "typing-extensions",
        # Data providers
        "databento",
        "fredapi",
        "yfinance",
        "alpaca-py",
        # Analytics
        "matplotlib",
        "seaborn",
        "plotly",
        "mlflow",
        # Trading calendars
        "exchange-calendars",
        "pandas-market-calendars",
        # Performance
        "pyarrow",
        "polars",
        "numba",
        "numexpr",
        # Async
        "asyncio",
        "asyncpg",
        "aiofiles",
    ],
    extras_require={
        "dev": [
            "pytest",
            "pytest-cov",
            "pytest-asyncio",
            "pytest-mock",
            "ruff",
            "black",
            "isort",
            "hypothesis",
            "pre-commit",
        ]
    },
    entry_points={
        "console_scripts": [
            "wheel-trading=unity_wheel.cli.run:main",
        ],
    },
)

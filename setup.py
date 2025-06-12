#!/usr/bin/env python3
"""Minimal setup.py for pip compatibility with Poetry-based project."""

from setuptools import setup, find_packages

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
        # Core dependencies
        "pydantic>=2.0",
        "pandas",
        "numpy",
        "duckdb",
        "aiohttp",
        "requests",
        "pyarrow",
        "polars",
        "matplotlib",
        "seaborn",
        "plotly",
        "scipy",
        "scikit-learn",
        "mlflow",
        # Trading specific
        "exchange-calendars",
        "fredapi",
        "yfinance",
        "alpaca-py",
        "databento",
        # Utilities
        "click",
        "pyyaml",
        "tenacity",
        "httpx",
        "rich",
        "numba",
        "numexpr",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-asyncio",
            "pytest-cov",
            "pytest-xdist",
            "hypothesis",
            "black",
            "ruff",
            "isort",
            "mypy",
            "pre-commit",
        ]
    },
    entry_points={
        "console_scripts": [
            "unity-wheel=unity_wheel.cli.run:main",
        ],
    },
)
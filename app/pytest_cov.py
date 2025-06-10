"""Minimal pytest-cov stub to accept coverage options."""

from __future__ import annotations

from typing import Any


def pytest_addoption(parser: Any) -> None:
    """Register dummy coverage options."""
    parser.addoption("--cov", action="store")
    parser.addoption("--cov-report", action="append")
    parser.addoption("--cov-fail-under", action="store")


def pytest_configure(config: Any) -> None:
    """No-op configuration hook."""
    return

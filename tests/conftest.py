# pytest_plugins = ["app.pytest_cov"]  # Commented out to avoid conflict with pytest-cov

import os
import pathlib

import pytest

from src.config.loader import get_config_loader


@pytest.fixture(autouse=True, scope="session")
def _load_test_config(monkeypatch) -> None:
    """Ensure configuration is loaded for tests."""
    config_path = pathlib.Path(__file__).resolve().parents[1] / "config.yaml"
    monkeypatch.setenv("WHEEL_CONFIG_PATH", str(config_path))
    # Force reload so each test session starts fresh
    loader = get_config_loader(str(config_path))
    loader.reload()

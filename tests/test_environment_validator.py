import os

import pytest
from src.unity_wheel.utils import validate


def test_check_python_version() -> None:
    validator = validate.EnvironmentValidator()
    validator._check_python_version()
    assert "Python Version" in validator.results
    success, msg = validator.results["Python Version"]
    assert isinstance(success, bool)
    assert isinstance(msg, str)


def test_check_imports() -> None:
    validator = validate.EnvironmentValidator()
    validator._check_imports()
    assert validator.results
    assert all(isinstance(v[0], bool) for v in validator.results.values())


@pytest.mark.skipif(
    not os.environ.get("FRED_API_KEY"),
    reason="FRED_API_KEY not set",
)
def test_run_all_checks_partial(monkeypatch) -> None:
    """run_all_checks should aggregate failures."""
    validator = validate.EnvironmentValidator()

    def mock_check_models() -> None:
        validator.results["Models"] = (False, "✗ mock")
        validator.critical_failures += 1

    monkeypatch.setattr(validator, "_check_models", mock_check_models)
    success = validator.run_all_checks()
    assert not success
    assert validator.critical_failures >= 1

from __future__ import annotations

import pytest

from unity_wheel.utils.recovery import RecoveryStrategy, validate_and_recover, with_recovery


def test_with_recovery_retry() -> None:
    """Function should retry until succeeding."""
    calls = {"count": 0}

    @with_recovery(strategy=RecoveryStrategy.RETRY, max_attempts=3, backoff_factor=0)
    def flaky() -> str:
        calls["count"] += 1
        if calls["count"] < 2:
            raise ValueError("fail")
        return "ok"

    assert flaky() == "ok"
    assert calls["count"] == 2


def test_with_recovery_fallback() -> None:
    """Fallback value should be returned when all attempts fail."""

    @with_recovery(
        strategy=RecoveryStrategy.FALLBACK,
        max_attempts=2,
        fallback_value="fallback",
        backoff_factor=0,
    )
    def always_fail() -> str:
        raise RuntimeError("boom")

    assert always_fail() == "fallback"


def test_validate_and_recover() -> None:
    """Invalid values trigger recovery function."""

    def validator(val: int) -> bool:
        return val > 0

    def recover() -> int:
        return 5

    assert validate_and_recover(3, validator, recover, "op") == 3
    assert validate_and_recover(-1, validator, recover, "op") == 5


async def _async_flaky(state: dict) -> str:
    state["count"] += 1
    if state["count"] < 2:
        raise RuntimeError("nope")
    return "ok"


@pytest.mark.asyncio
async def test_with_recovery_async() -> None:
    """Async functions also retry until success."""
    state = {"count": 0}

    wrapped = with_recovery(strategy=RecoveryStrategy.RETRY, max_attempts=3, backoff_factor=0)(
        _async_flaky
    )

    result = await wrapped(state)
    assert result == "ok"
    assert state["count"] == 2

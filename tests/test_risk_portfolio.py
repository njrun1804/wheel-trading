import sys
import types

import pytest

# Stub google cloud modules if missing
if "google" not in sys.modules:
    sys.modules["google"] = types.ModuleType("google")
    sys.modules["google.cloud"] = types.ModuleType("google.cloud")
    storage_mod = types.ModuleType("google.cloud.storage")
    exceptions_mod = types.ModuleType("google.cloud.exceptions")
    exceptions_mod.NotFound = type("NotFound", (), {})
    storage_mod.Client = object  # minimal stub
    sys.modules["google.cloud.storage"] = storage_mod
    sys.modules["google.cloud.exceptions"] = exceptions_mod

from unity_wheel.models.greeks import Greeks
from unity_wheel.models.position import Position
from unity_wheel.risk.analytics import RiskAnalyzer


@pytest.fixture
def analyzer():
    return RiskAnalyzer()


def test_aggregate_portfolio_greeks_basic(analyzer):
    pos1 = Position("U240621C00050000", 1)
    pos2 = Position("U240621P00045000", -2)
    greeks1 = Greeks(delta=0.5, gamma=0.1, vega=0.2, theta=-0.01, rho=0.05)
    greeks2 = Greeks(delta=-0.4, gamma=0.08, vega=0.15, theta=-0.02, rho=-0.04)

    aggregated, conf = analyzer.aggregate_portfolio_greeks(
        [
            (pos1, greeks1, 50.0),
            (pos2, greeks2, 50.0),
        ]
    )

    assert aggregated["delta"] == pytest.approx(1 * 100 * 0.5 + (-2) * 100 * -0.4)
    assert aggregated["gamma"] == pytest.approx(1 * 100 * 0.1 + (-2) * 100 * 0.08)
    assert aggregated["vega"] == pytest.approx(1 * 100 * 0.2 + (-2) * 100 * 0.15)
    assert aggregated["theta"] == pytest.approx(1 * 100 * -0.01 + (-2) * 100 * -0.02)
    assert aggregated["rho"] == pytest.approx(1 * 100 * 0.05 + (-2) * 100 * -0.04)
    assert aggregated["delta_dollars"] == pytest.approx(aggregated["delta"] * 50.0)
    assert conf == pytest.approx(0.95)


def test_aggregate_portfolio_greeks_missing_values(analyzer):
    pos = Position("U240621C00050000", 1)
    greeks = Greeks(delta=0.5, vega=0.2, theta=-0.01)  # gamma missing

    aggregated, conf = analyzer.aggregate_portfolio_greeks(
        [
            (pos, greeks, 25.0),
        ]
    )

    assert aggregated["gamma"] == 0
    # Missing one set of greeks out of one position -> confidence reduced by 0.2
    assert conf == pytest.approx(0.95 * 0.8)


def test_estimate_margin_requirement(analyzer):
    short_put = Position("U240621P00050000", -1)
    short_call = Position("U240621C00060000", -1)

    margin, conf = analyzer.estimate_margin_requirement(
        [
            (short_put, 55.0, 2.0),
            (short_call, 60.0, 1.5),
        ]
    )

    # Put margin: max(0.2*55 - (55-50) + 2, 0.1*50 + 2) * 100 = 800
    expected_put = 800
    # Call margin: 100 * 60 * 0.5 = 3000
    expected_call = 3000
    assert margin == pytest.approx(expected_put + expected_call)
    assert conf == pytest.approx(0.90)

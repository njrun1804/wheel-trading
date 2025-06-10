"""Comprehensive tests for Greeks model including property-based testing."""

from __future__ import annotations

from typing import Optional

import pytest
from hypothesis import given
from hypothesis import strategies as st

from unity_wheel.models.greeks import Greeks


class TestGreeksBasic:
    """Basic unit tests for Greeks model."""

    def test_greeks_creation_all_values(self) -> None:
        """Test creating Greeks with all values."""
        greeks = Greeks(
            delta=0.5,
            gamma=0.02,
            theta=-0.05,
            vega=0.15,
            rho=0.08,
        )
        assert greeks.delta == 0.5
        assert greeks.gamma == 0.02
        assert greeks.theta == -0.05
        assert greeks.vega == 0.15
        assert greeks.rho == 0.08
        assert greeks.has_all_greeks is True

    def test_greeks_creation_partial(self) -> None:
        """Test creating Greeks with partial values."""
        greeks = Greeks(delta=0.3, vega=0.1)
        assert greeks.delta == 0.3
        assert greeks.gamma is None
        assert greeks.theta is None
        assert greeks.vega == 0.1
        assert greeks.rho is None
        assert greeks.has_all_greeks is False

    def test_greeks_creation_empty(self) -> None:
        """Test creating empty Greeks."""
        greeks = Greeks()
        assert greeks.delta is None
        assert greeks.gamma is None
        assert greeks.theta is None
        assert greeks.vega is None
        assert greeks.rho is None
        assert greeks.has_all_greeks is False
        assert str(greeks) == "Greeks(empty)"

    def test_greeks_immutability(self) -> None:
        """Test that Greeks is immutable."""
        greeks = Greeks(delta=0.5)
        with pytest.raises(AttributeError):
            greeks.delta = 0.6  # type: ignore

    def test_delta_validation(self) -> None:
        """Test delta validation (-1 <= delta <= 1)."""
        # Valid deltas
        Greeks(delta=-1.0)  # Put at extreme
        Greeks(delta=0.0)  # ATM
        Greeks(delta=1.0)  # Call at extreme
        Greeks(delta=0.5)  # Normal call
        Greeks(delta=-0.3)  # Normal put

        # Invalid deltas
        with pytest.raises(ValueError, match="Delta must be between -1 and 1"):
            Greeks(delta=1.1)
        with pytest.raises(ValueError, match="Delta must be between -1 and 1"):
            Greeks(delta=-1.1)

    def test_gamma_validation(self) -> None:
        """Test gamma validation (gamma >= 0)."""
        # Valid gammas
        Greeks(gamma=0.0)
        Greeks(gamma=0.05)
        Greeks(gamma=1.0)
        Greeks(gamma=100.0)  # Extreme but possible

        # Invalid gamma
        with pytest.raises(ValueError, match="Gamma must be non-negative"):
            Greeks(gamma=-0.01)

    def test_theta_validation(self) -> None:
        """Test theta validation (typically negative, warning for positive)."""
        # Normal theta (negative)
        greeks1 = Greeks(theta=-0.05)
        assert greeks1.theta == -0.05

        # Zero theta
        greeks2 = Greeks(theta=0.0)
        assert greeks2.theta == 0.0

        # Positive theta (unusual but possible, should log warning)
        greeks3 = Greeks(theta=0.01)
        assert greeks3.theta == 0.01

    def test_vega_validation(self) -> None:
        """Test vega validation (vega >= 0)."""
        # Valid vegas
        Greeks(vega=0.0)
        Greeks(vega=0.15)
        Greeks(vega=1.0)

        # Invalid vega
        with pytest.raises(ValueError, match="Vega must be non-negative"):
            Greeks(vega=-0.01)

    def test_rho_validation(self) -> None:
        """Test rho validation (no strict bounds but warning for extreme)."""
        # Normal rhos
        Greeks(rho=0.05)  # Positive for calls
        Greeks(rho=-0.03)  # Negative for puts
        Greeks(rho=0.0)
        Greeks(rho=50.0)  # Large but reasonable

        # Extreme rho (should log warning but not fail)
        greeks = Greeks(rho=1500.0)
        assert greeks.rho == 1500.0

    def test_speed_property(self) -> None:
        """Test speed property (placeholder for now)."""
        greeks = Greeks(delta=0.5, gamma=0.02)
        assert greeks.speed is None  # Not implemented yet

    def test_greeks_string_representation(self) -> None:
        """Test human-readable string representation."""
        # All Greeks
        greeks1 = Greeks(delta=0.5, gamma=0.02, theta=-0.05, vega=0.15, rho=0.08)
        assert str(greeks1) == "Greeks(Δ=0.500, Γ=0.020, Θ=-0.050, ν=0.150, ρ=0.080)"

        # Partial Greeks
        greeks2 = Greeks(delta=0.333, vega=0.123)
        assert str(greeks2) == "Greeks(Δ=0.333, ν=0.123)"

        # Empty Greeks
        greeks3 = Greeks()
        assert str(greeks3) == "Greeks(empty)"

    def test_greeks_serialization(self) -> None:
        """Test converting Greeks to/from dictionary."""
        greeks = Greeks(delta=0.5, gamma=0.02, theta=-0.05, vega=0.15, rho=0.08)

        # To dict
        data = greeks.to_dict()
        assert data == {
            "delta": 0.5,
            "gamma": 0.02,
            "theta": -0.05,
            "vega": 0.15,
            "rho": 0.08,
        }

        # From dict
        greeks2 = Greeks.from_dict(data)
        assert greeks2.delta == greeks.delta
        assert greeks2.gamma == greeks.gamma
        assert greeks2.theta == greeks.theta
        assert greeks2.vega == greeks.vega
        assert greeks2.rho == greeks.rho

    def test_greeks_from_dict_with_none(self) -> None:
        """Test creating Greeks from dict with None values."""
        data = {
            "delta": 0.5,
            "gamma": None,
            "theta": -0.03,
            "vega": None,
            "rho": 0.02,
        }
        greeks = Greeks.from_dict(data)
        assert greeks.delta == 0.5
        assert greeks.gamma is None
        assert greeks.theta == -0.03
        assert greeks.vega is None
        assert greeks.rho == 0.02


class TestGreeksPropertyBased:
    """Property-based tests using Hypothesis."""

    @given(
        delta=st.one_of(st.none(), st.floats(min_value=-1.0, max_value=1.0)),
        gamma=st.one_of(st.none(), st.floats(min_value=0.0, max_value=10.0)),
        theta=st.one_of(st.none(), st.floats(min_value=-10.0, max_value=1.0)),
        vega=st.one_of(st.none(), st.floats(min_value=0.0, max_value=10.0)),
        rho=st.one_of(st.none(), st.floats(min_value=-100.0, max_value=100.0)),
    )
    def test_greeks_valid_ranges(
        self,
        delta: Optional[float],
        gamma: Optional[float],
        theta: Optional[float],
        vega: Optional[float],
        rho: Optional[float],
    ) -> None:
        """Test Greeks with random valid inputs."""
        greeks = Greeks(
            delta=delta,
            gamma=gamma,
            theta=theta,
            vega=vega,
            rho=rho,
        )

        # Properties that must hold
        assert greeks.delta == delta
        assert greeks.gamma == gamma
        assert greeks.theta == theta
        assert greeks.vega == vega
        assert greeks.rho == rho

        # Check has_all_greeks
        all_present = all(v is not None for v in [delta, gamma, theta, vega, rho])
        assert greeks.has_all_greeks == all_present

    @given(delta=st.floats(allow_nan=False, allow_infinity=False).filter(lambda x: x < -1 or x > 1))
    def test_invalid_delta_ranges(self, delta: float) -> None:
        """Test that invalid delta values raise errors."""
        with pytest.raises(ValueError, match="Delta must be between -1 and 1"):
            Greeks(delta=delta)

    @given(gamma=st.floats(max_value=-0.001, allow_nan=False, allow_infinity=False))
    def test_invalid_gamma_ranges(self, gamma: float) -> None:
        """Test that negative gamma values raise errors."""
        with pytest.raises(ValueError, match="Gamma must be non-negative"):
            Greeks(gamma=gamma)

    @given(vega=st.floats(max_value=-0.001, allow_nan=False, allow_infinity=False))
    def test_invalid_vega_ranges(self, vega: float) -> None:
        """Test that negative vega values raise errors."""
        with pytest.raises(ValueError, match="Vega must be non-negative"):
            Greeks(vega=vega)

    @given(
        data=st.fixed_dictionaries(
            {
                "delta": st.one_of(st.none(), st.floats(min_value=-1, max_value=1)),
                "gamma": st.one_of(st.none(), st.floats(min_value=0, max_value=10)),
                "theta": st.one_of(st.none(), st.floats(min_value=-10, max_value=1)),
                "vega": st.one_of(st.none(), st.floats(min_value=0, max_value=10)),
                "rho": st.one_of(st.none(), st.floats(min_value=-100, max_value=100)),
            }
        )
    )
    def test_greeks_dict_roundtrip(self, data: dict[str, Optional[float]]) -> None:
        """Test that Greeks can roundtrip through dict."""
        greeks = Greeks.from_dict(data)
        data2 = greeks.to_dict()
        greeks2 = Greeks.from_dict(data2)

        # All values should match
        assert greeks.delta == greeks2.delta
        assert greeks.gamma == greeks2.gamma
        assert greeks.theta == greeks2.theta
        assert greeks.vega == greeks2.vega
        assert greeks.rho == greeks2.rho


class TestGreeksEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_extreme_but_valid_values(self) -> None:
        """Test extreme but valid Greek values."""
        # Near-zero values
        greeks1 = Greeks(
            delta=1e-10,
            gamma=1e-10,
            theta=-1e-10,
            vega=1e-10,
            rho=1e-10,
        )
        assert greeks1.delta == 1e-10

        # Large but valid values
        greeks2 = Greeks(
            delta=0.9999,
            gamma=5.0,
            theta=-100.0,
            vega=50.0,
            rho=999.0,  # Large but not extreme enough for warning
        )
        assert greeks2.gamma == 5.0

    def test_typical_option_greeks(self) -> None:
        """Test typical Greek values for different option scenarios."""
        # ATM call option
        atm_call = Greeks(
            delta=0.5,
            gamma=0.05,
            theta=-0.02,
            vega=0.20,
            rho=0.15,
        )
        assert atm_call.has_all_greeks

        # Deep ITM put option
        itm_put = Greeks(
            delta=-0.95,
            gamma=0.001,
            theta=-0.001,
            vega=0.01,
            rho=-0.45,
        )
        assert itm_put.delta == -0.95

        # Far OTM call option
        otm_call = Greeks(
            delta=0.05,
            gamma=0.01,
            theta=-0.005,
            vega=0.05,
            rho=0.02,
        )
        assert otm_call.delta == 0.05

    @pytest.mark.parametrize(
        "greek_name,value",
        [
            ("delta", float("nan")),
            ("gamma", float("inf")),
            ("theta", float("-inf")),
            ("vega", float("nan")),
            ("rho", float("inf")),
        ],
    )
    def test_special_float_values(self, greek_name: str, value: float) -> None:
        """Test handling of special float values (NaN, inf)."""
        # Most of these should trigger validation errors
        kwargs = {greek_name: value}

        if greek_name == "delta":
            # NaN is not between -1 and 1
            with pytest.raises(ValueError):
                Greeks(**kwargs)
        elif greek_name in ["gamma", "vega"]:
            # inf/NaN is not >= 0 in a meaningful way
            with pytest.raises(ValueError):
                Greeks(**kwargs)
        else:
            # theta and rho might accept these
            greeks = Greeks(**kwargs)
            assert getattr(greeks, greek_name) != getattr(greeks, greek_name)  # NaN != NaN

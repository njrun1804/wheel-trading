"""Greeks model with validation ranges."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Greeks:
    """
    Option Greeks with validation ranges.
    
    All values are optional as they may not be available for stocks
    or in certain market conditions.
    
    Attributes
    ----------
    delta : Optional[float]
        Rate of change of option price w.r.t. underlying price
        Range: [-1, 1] (calls: [0, 1], puts: [-1, 0])
    gamma : Optional[float]
        Rate of change of delta w.r.t. underlying price
        Range: [0, ∞) (always positive)
    theta : Optional[float]
        Rate of change of option price w.r.t. time (per day)
        Range: (-∞, 0] (usually negative, time decay)
    vega : Optional[float]
        Rate of change of option price w.r.t. volatility
        Range: [0, ∞) (always positive)
    rho : Optional[float]
        Rate of change of option price w.r.t. risk-free rate
        Range: (-∞, ∞) (calls: positive, puts: negative)
    
    Examples
    --------
    >>> greeks = Greeks(delta=0.5, gamma=0.02, theta=-0.05, vega=0.15)
    >>> greeks.delta
    0.5
    
    >>> # Invalid delta
    >>> Greeks(delta=1.5)  # doctest: +SKIP
    ValueError: Delta must be between -1 and 1, got 1.5
    """

    delta: Optional[float] = None
    gamma: Optional[float] = None
    theta: Optional[float] = None
    vega: Optional[float] = None
    rho: Optional[float] = None

    def __post_init__(self) -> None:
        """Validate Greek values are within expected ranges."""
        # Delta validation: -1 <= delta <= 1
        if self.delta is not None:
            if not -1 <= self.delta <= 1:
                raise ValueError(f"Delta must be between -1 and 1, got {self.delta}")
        
        # Gamma validation: gamma >= 0
        if self.gamma is not None:
            if self.gamma < 0:
                raise ValueError(f"Gamma must be non-negative, got {self.gamma}")
        
        # Theta validation: typically negative but can be positive in rare cases
        # No strict validation, just warning for unusual values
        if self.theta is not None:
            if self.theta > 0:
                logger.warning(
                    "Positive theta detected (unusual)",
                    extra={"theta": self.theta}
                )
        
        # Vega validation: vega >= 0
        if self.vega is not None:
            if self.vega < 0:
                raise ValueError(f"Vega must be non-negative, got {self.vega}")
        
        # Rho validation: no strict bounds but warn on extreme values
        if self.rho is not None:
            if abs(self.rho) > 1000:  # Arbitrary large value
                logger.warning(
                    "Extreme rho value detected",
                    extra={"rho": self.rho}
                )
        
        logger.debug(
            "Greeks created",
            extra={
                "delta": self.delta,
                "gamma": self.gamma,
                "theta": self.theta,
                "vega": self.vega,
                "rho": self.rho,
            },
        )

    @property
    def has_all_greeks(self) -> bool:
        """Check if all Greeks are populated."""
        return all(
            greek is not None
            for greek in [self.delta, self.gamma, self.theta, self.vega, self.rho]
        )

    @property
    def speed(self) -> Optional[float]:
        """
        Calculate speed (rate of change of gamma).
        
        Note: This is a placeholder as speed requires price changes to calculate.
        In practice, this would be computed by the pricing engine.
        """
        # Speed = dGamma/dS (third derivative of option price)
        # Cannot be calculated from static Greeks alone
        return None

    def to_dict(self) -> dict[str, Optional[float]]:
        """Convert to dictionary for serialization."""
        return {
            "delta": self.delta,
            "gamma": self.gamma,
            "theta": self.theta,
            "vega": self.vega,
            "rho": self.rho,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Optional[float]]) -> Greeks:
        """Create Greeks from dictionary."""
        return cls(
            delta=data.get("delta"),
            gamma=data.get("gamma"),
            theta=data.get("theta"),
            vega=data.get("vega"),
            rho=data.get("rho"),
        )

    def __str__(self) -> str:
        """Human-readable string representation."""
        parts = []
        if self.delta is not None:
            parts.append(f"Δ={self.delta:.3f}")
        if self.gamma is not None:
            parts.append(f"Γ={self.gamma:.3f}")
        if self.theta is not None:
            parts.append(f"Θ={self.theta:.3f}")
        if self.vega is not None:
            parts.append(f"ν={self.vega:.3f}")
        if self.rho is not None:
            parts.append(f"ρ={self.rho:.3f}")
        
        return f"Greeks({', '.join(parts)})" if parts else "Greeks(empty)"
"""API type definitions."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Literal, TypedDict
from typing_extensions import NotRequired

# Type aliases
Action = Literal["HOLD", "ADJUST", "ROLL", "CLOSE"]
OptionType = Literal["call", "put"]


class RiskMetrics(TypedDict):
    """Risk metrics for recommendation."""

    max_loss: float
    probability_assignment: float
    expected_return: float
    edge_ratio: float
    var_95: float
    margin_required: float
    borrowing_analysis: NotRequired[Dict[str, Any]]


class Recommendation(TypedDict):
    """Trading recommendation with full context."""

    action: Action
    rationale: str
    confidence: float
    risk: RiskMetrics
    details: Dict[str, Any]


class PositionData(TypedDict):
    """Position data structure."""

    symbol: str
    quantity: int
    strike: float
    expiration: str
    option_type: OptionType
    cost_basis: float
    current_price: float


class MarketSnapshot(TypedDict):
    """Market data snapshot for decision making."""

    timestamp: datetime
    ticker: str
    current_price: float
    buying_power: float
    margin_used: float
    positions: List[PositionData]
    option_chain: Dict[str, OptionData]
    implied_volatility: float
    risk_free_rate: float


class OptionData(TypedDict):
    """Option chain data."""

    strike: float
    expiration: str
    bid: float
    ask: float
    mid: float
    volume: int
    open_interest: int
    delta: float
    gamma: float
    theta: float
    vega: float
    implied_volatility: float

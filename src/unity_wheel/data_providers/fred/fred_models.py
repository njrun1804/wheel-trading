"""FRED API data models with comprehensive validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date as Date
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, field_validator

from ...utils import get_logger

logger = get_logger(__name__)


class UpdateFrequency(str, Enum):
    """FRED series update frequencies."""

    DAILY = "d"
    WEEKLY = "w"
    BIWEEKLY = "bw"
    MONTHLY = "m"
    QUARTERLY = "q"
    SEMIANNUAL = "sa"
    ANNUAL = "a"

    @property
    def days(self) -> int:
        """Approximate days between updates."""
        mapping = {
            "d": 1,
            "w": 7,
            "bw": 14,
            "m": 30,
            "q": 90,
            "sa": 180,
            "a": 365,
        }
        return mapping[self.value]


class FREDSeries(BaseModel):
    """FRED series metadata with validation."""

    series_id: str = Field(..., description="Series identifier")
    title: str = Field(..., description="Series title")
    observation_start: Date = Field(..., description="First observation date")
    observation_end: Date = Field(..., description="Last observation date")
    frequency: UpdateFrequency = Field(..., description="Update frequency")
    units: str = Field(..., description="Data units")
    seasonal_adjustment: str = Field(..., description="Seasonal adjustment type")
    last_updated: datetime = Field(..., description="Last update timestamp")
    popularity: int = Field(0, description="Series popularity score")
    notes: Optional[str] = Field(None, description="Series notes")

    @field_validator("frequency", mode="before")
    @classmethod
    def parse_frequency(cls, v: str) -> UpdateFrequency:
        """Convert frequency string to enum."""
        if isinstance(v, UpdateFrequency):
            return v
        freq_map = {"D": "d", "W": "w", "M": "m", "Q": "q", "A": "a"}
        return UpdateFrequency(freq_map.get(v.upper(), v.lower()))

    @field_validator("last_updated", mode="before")
    @classmethod
    def parse_datetime(cls, v: Union[str, datetime]) -> datetime:
        """Parse datetime string."""
        if isinstance(v, datetime):
            return v
        return datetime.fromisoformat(v.replace("Z", "+00:00"))

    @property
    def days_since_update(self) -> int:
        """Days since last update."""
        return (datetime.now(timezone.utc) - self.last_updated).days

    @property
    def is_discontinued(self) -> bool:
        """Check if series is likely discontinued."""
        expected_update_days = self.frequency.days * 2
        return self.days_since_update > expected_update_days

    model_config = {"use_enum_values": False, "arbitrary_types_allowed": True}


class FREDObservation(BaseModel):
    """Single FRED observation with validation."""

    date: Date = Field(..., description="Observation date")
    value: Optional[float] = Field(None, description="Observation value")

    @field_validator("value", mode="before")
    @classmethod
    def parse_value(cls, v: Any) -> Optional[float]:
        """Parse value string, handling missing data."""
        if v is None or v == "." or v == "nan" or v == "NaN":
            return None
        try:
            return float(v)
        except (ValueError, TypeError):
            logger.warning(f"Failed to parse FRED value: {v}")
            return None

    @field_validator("date", mode="before")
    @classmethod
    def parse_date(cls, v: Union[str, Date]) -> Date:
        """Parse date string."""
        if isinstance(v, Date):
            return v
        return datetime.strptime(v, "%Y-%m-%d").date()

    model_config = {"arbitrary_types_allowed": True}


@dataclass
class FREDDataPoint:
    """Enhanced data point with metadata and confidence."""

    series_id: str
    date: Date
    value: Optional[float]
    confidence: float = 1.0
    is_revised: bool = False
    revision_date: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_valid(self) -> bool:
        """Check if data point is valid."""
        return self.value is not None and self.confidence > 0.5

    def age_days(self) -> int:
        """Days since observation date."""
        return (Date.today() - self.date).days


@dataclass
class FREDDataset:
    """Collection of FRED series with metadata."""

    series: FREDSeries
    observations: List[FREDObservation]
    fetch_timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def latest_value(self) -> Optional[float]:
        """Get most recent non-null value."""
        for obs in reversed(self.observations):
            if obs.value is not None:
                return obs.value
        return None

    @property
    def date_range(self) -> tuple[date, date]:
        """Get date range of observations."""
        if not self.observations:
            return self.series.observation_start, self.series.observation_end
        return self.observations[0].date, self.observations[-1].date

    def get_value(self, target_date: date) -> Optional[float]:
        """Get value for specific date."""
        for obs in self.observations:
            if obs.date == target_date:
                return obs.value
        return None

    def to_dataframe(self):
        """Convert to pandas DataFrame."""
        import pandas as pd

        data = [(obs.date, obs.value) for obs in self.observations]
        df = pd.DataFrame(data, columns=["date", "value"])
        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)
        return df


class WheelStrategyFREDSeries(str, Enum):
    """FRED series relevant to wheel strategy."""

    # Risk-free rates
    DGS3 = "DGS3"  # 3-Month Treasury
    DGS1 = "DGS1"  # 1-Year Treasury
    DFF = "DFF"  # Federal Funds Rate

    # Volatility indicators
    VIXCLS = "VIXCLS"  # CBOE VIX
    VXDCLS = "VXDCLS"  # Dow Jones VIX

    # Liquidity/stress indicators
    TEDRATE = "TEDRATE"  # TED Spread
    BAMLH0A0HYM2 = "BAMLH0A0HYM2"  # High Yield Spread

    # Economic indicators
    UNRATE = "UNRATE"  # Unemployment Rate
    CPIAUCSL = "CPIAUCSL"  # CPI

    @property
    def description(self) -> str:
        """Human-readable description."""
        descriptions = {
            "DGS3": "3-Month Treasury Constant Maturity Rate",
            "DGS1": "1-Year Treasury Constant Maturity Rate",
            "DFF": "Effective Federal Funds Rate",
            "VIXCLS": "CBOE Volatility Index: VIX",
            "VXDCLS": "CBOE DJIA Volatility Index",
            "TEDRATE": "TED Spread",
            "BAMLH0A0HYM2": "High Yield OAS",
            "UNRATE": "Unemployment Rate",
            "CPIAUCSL": "Consumer Price Index",
        }
        return descriptions.get(self.value, self.value)

    @property
    def update_frequency(self) -> UpdateFrequency:
        """Expected update frequency."""
        frequencies = {
            "DGS3": UpdateFrequency.DAILY,
            "DGS1": UpdateFrequency.DAILY,
            "DFF": UpdateFrequency.DAILY,
            "VIXCLS": UpdateFrequency.DAILY,
            "VXDCLS": UpdateFrequency.DAILY,
            "TEDRATE": UpdateFrequency.DAILY,
            "BAMLH0A0HYM2": UpdateFrequency.DAILY,
            "UNRATE": UpdateFrequency.MONTHLY,
            "CPIAUCSL": UpdateFrequency.MONTHLY,
        }
        return frequencies[self.value]

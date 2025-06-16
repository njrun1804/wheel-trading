"""FRED indicator definitions and storage logic."""

from dataclasses import dataclass


@dataclass
class FREDIndicator:
    """FRED indicator configuration."""

    series_id: str
    name: str
    description: str
    ml_column: str | None = None  # Column in ml_features table
    scale_factor: float = 1.0  # Some rates need scaling


# Complete list of FRED indicators for wheel trading
FRED_INDICATORS: dict[str, FREDIndicator] = {
    "VIXCLS": FREDIndicator(
        series_id="VIXCLS",
        name="VIX",
        description="CBOE Volatility Index",
        ml_column="vix_level",
    ),
    "DGS3MO": FREDIndicator(
        series_id="DGS3MO",
        name="3-Month Treasury",
        description="3-Month Treasury Constant Maturity Rate",
        ml_column="risk_free_rate",
        scale_factor=0.01,  # Convert from percent to decimal
    ),
    "DGS10": FREDIndicator(
        series_id="DGS10",
        name="10-Year Treasury",
        description="10-Year Treasury Constant Maturity Rate",
        ml_column="ten_year_rate",
        scale_factor=0.01,
    ),
    "TEDRATE": FREDIndicator(
        series_id="TEDRATE",
        name="TED Spread",
        description="TED Spread",
        ml_column="ted_spread",
        scale_factor=0.01,
    ),
    "DFF": FREDIndicator(
        series_id="DFF",
        name="Fed Funds Rate",
        description="Effective Federal Funds Rate",
        ml_column="fed_funds_rate",
        scale_factor=0.01,
    ),
    "DEXUSEU": FREDIndicator(
        series_id="DEXUSEU",
        name="USD/EUR",
        description="US Dollar to Euro Exchange Rate",
        ml_column="usd_eur_rate",
    ),
    "BAMLH0A0HYM2": FREDIndicator(
        series_id="BAMLH0A0HYM2",
        name="High Yield Spread",
        description="BofA High Yield Option-Adjusted Spread",
        ml_column="hy_spread",
        scale_factor=0.01,
    ),
    "UMCSENT": FREDIndicator(
        series_id="UMCSENT",
        name="Consumer Sentiment",
        description="University of Michigan Consumer Sentiment",
        ml_column="consumer_sentiment",
    ),
    "UNRATE": FREDIndicator(
        series_id="UNRATE",
        name="Unemployment Rate",
        description="Civilian Unemployment Rate",
        ml_column="unemployment_rate",
        scale_factor=0.01,
    ),
    "CPILFESL": FREDIndicator(
        series_id="CPILFESL",
        name="Core CPI",
        description="Core Consumer Price Index",
        ml_column="core_cpi",
    ),
}


def get_fred_sql_create() -> str:
    """Generate SQL to create FRED data table."""
    return """
    CREATE TABLE IF NOT EXISTS analytics.fred_data (
        series_id VARCHAR PRIMARY KEY,
        observation_date DATE NOT NULL,
        value DECIMAL(10,4) NOT NULL,
        last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(series_id, observation_date)
    );
    
    -- Also extend ml_features table with new columns
    ALTER TABLE analytics.ml_features 
    ADD COLUMN IF NOT EXISTS risk_free_rate DECIMAL(6,4);
    
    ALTER TABLE analytics.ml_features 
    ADD COLUMN IF NOT EXISTS ten_year_rate DECIMAL(6,4);
    
    ALTER TABLE analytics.ml_features 
    ADD COLUMN IF NOT EXISTS ted_spread DECIMAL(6,4);
    
    ALTER TABLE analytics.ml_features 
    ADD COLUMN IF NOT EXISTS fed_funds_rate DECIMAL(6,4);
    
    ALTER TABLE analytics.ml_features 
    ADD COLUMN IF NOT EXISTS usd_eur_rate DECIMAL(8,4);
    
    ALTER TABLE analytics.ml_features 
    ADD COLUMN IF NOT EXISTS hy_spread DECIMAL(6,4);
    
    ALTER TABLE analytics.ml_features 
    ADD COLUMN IF NOT EXISTS consumer_sentiment DECIMAL(6,2);
    
    ALTER TABLE analytics.ml_features 
    ADD COLUMN IF NOT EXISTS unemployment_rate DECIMAL(5,3);
    
    ALTER TABLE analytics.ml_features 
    ADD COLUMN IF NOT EXISTS core_cpi DECIMAL(8,2);
    """


def store_fred_observation(conn, series_id: str, date: str, value: float) -> bool:
    """Store a FRED observation in the database."""
    try:
        indicator = FRED_INDICATORS.get(series_id)
        if not indicator:
            return False

        # Apply scaling
        scaled_value = value * indicator.scale_factor

        # Store in fred_data table
        conn.execute(
            """
            INSERT OR REPLACE INTO analytics.fred_data 
            (series_id, observation_date, value)
            VALUES (?, ?, ?)
        """,
            [series_id, date, scaled_value],
        )

        # Also update ml_features if this indicator has a column
        if indicator.ml_column:
            # Check if ml_features row exists for this date
            exists = conn.execute(
                """
                SELECT 1 FROM analytics.ml_features 
                WHERE symbol = 'U' AND feature_date = ?
            """,
                [date],
            ).fetchone()

            if exists:
                # Update existing row
                conn.execute(
                    f"""
                    UPDATE analytics.ml_features 
                    SET {indicator.ml_column} = ?
                    WHERE symbol = 'U' AND feature_date = ?
                """,
                    [scaled_value, date],
                )
            else:
                # Create new row with this indicator
                conn.execute(
                    f"""
                    INSERT INTO analytics.ml_features 
                    (symbol, feature_date, {indicator.ml_column})
                    VALUES ('U', ?, ?)
                """,
                    [date, scaled_value],
                )

        return True

    except Exception as e:
        raise Exception(f"Failed to store FRED data: {e}")

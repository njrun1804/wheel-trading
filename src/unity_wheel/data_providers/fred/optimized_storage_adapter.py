"""
Optimized storage adapter for FRED data
Writes economic indicators to the new optimized database
"""

import duckdb
from datetime import datetime, date
from typing import List, Dict, Any, Optional
import logging

from .fred_models import FREDSeries, FREDObservation
from unity_wheel.utils import get_logger

logger = get_logger(__name__)

class FREDOptimizedStorageAdapter:
    """Storage adapter for FRED data in optimized database"""
    
    def __init__(self, db_path: str = "data/wheel_trading_optimized.duckdb"):
        self.db_path = db_path
        
        # Mapping of FRED series to ml_features columns
        self.series_mapping = {
            'DGS10': 'risk_free_rate_10y',     # 10-Year Treasury
            'DGS3': 'risk_free_rate_3m',       # 3-Month Treasury
            'DGS1': 'risk_free_rate_1y',       # 1-Year Treasury
            'DFF': 'federal_funds_rate',       # Federal Funds Rate
            'VIXCLS': 'vix_level',             # VIX
            'VXDCLS': 'vxd_level',             # VXD (DJIA Volatility)
            'TEDRATE': 'ted_spread',           # TED Spread
            'BAMLH0A0HYM2': 'high_yield_spread', # High Yield Spread
            'UNRATE': 'unemployment_rate',     # Unemployment Rate
            'CPIAUCSL': 'cpi_index'            # CPI
        }
        
        self._metrics = {
            "series_stored": 0,
            "observations_stored": 0,
            "features_updated": 0
        }
    
    def store_series_data(self, series_id: str, observations: List[FREDObservation]) -> int:
        """Store FRED observations and update ML features"""
        if not observations:
            return 0
            
        stored_count = 0
        
        try:
            conn = duckdb.connect(self.db_path)
            
            # Get the feature column name
            feature_column = self.series_mapping.get(series_id)
            if not feature_column:
                logger.warning(f"No mapping for FRED series: {series_id}")
                return 0
            
            # Process observations
            for obs in observations:
                try:
                    # Skip if value is invalid
                    if obs.value is None or obs.value == ".":
                        continue
                    
                    value = float(obs.value)
                    obs_date = obs.date if isinstance(obs.date, date) else datetime.strptime(obs.date, "%Y-%m-%d").date()
                    
                    # Check if row exists for this date
                    existing = conn.execute("""
                        SELECT COUNT(*) FROM analytics.ml_features 
                        WHERE symbol = 'U' AND feature_date = ?
                    """, [obs_date]).fetchone()[0]
                    
                    if existing > 0:
                        # Update existing row
                        query = f"""
                            UPDATE analytics.ml_features 
                            SET {feature_column} = ?
                            WHERE symbol = 'U' AND feature_date = ?
                        """
                        conn.execute(query, [value, obs_date])
                    else:
                        # Insert new row with this feature
                        # First set all columns to NULL except the one we're updating
                        columns = ['symbol', 'feature_date', feature_column]
                        values = ['U', obs_date, value]
                        
                        # Add market regime based on VIX if this is VIX data
                        if series_id == 'VIXCLS':
                            columns.append('market_regime')
                            if value < 20:
                                values.append('low_volatility')
                            elif value < 30:
                                values.append('normal')
                            else:
                                values.append('high_volatility')
                        
                        placeholders = ', '.join(['?' for _ in values])
                        column_list = ', '.join(columns)
                        
                        conn.execute(f"""
                            INSERT INTO analytics.ml_features ({column_list})
                            VALUES ({placeholders})
                        """, values)
                    
                    stored_count += 1
                    
                except Exception as e:
                    logger.error(f"Error storing observation for {obs_date}: {e}")
                    continue
            
            # Update metrics
            self._metrics["observations_stored"] += stored_count
            self._metrics["features_updated"] += 1
            
            conn.close()
            
            logger.info(
                f"Stored {stored_count} observations for {series_id}",
                extra={
                    "series_id": series_id,
                    "feature_column": feature_column,
                    "date_range": f"{observations[0].date} to {observations[-1].date}" if observations else "N/A"
                }
            )
            
        except Exception as e:
            logger.error(f"Error storing FRED data: {e}")
            raise
            
        return stored_count
    
    def store_multiple_series(self, series_data: Dict[str, List[FREDObservation]]) -> Dict[str, int]:
        """Store multiple FRED series at once"""
        results = {}
        
        for series_id, observations in series_data.items():
            count = self.store_series_data(series_id, observations)
            results[series_id] = count
            
        return results
    
    def get_latest_features(self, symbol: str = 'U') -> Optional[Dict[str, Any]]:
        """Get latest ML features for a symbol"""
        try:
            conn = duckdb.connect(self.db_path, read_only=True)
            
            result = conn.execute("""
                SELECT * FROM analytics.ml_features
                WHERE symbol = ?
                ORDER BY feature_date DESC
                LIMIT 1
            """, [symbol]).fetchone()
            
            if result:
                # Convert to dictionary
                columns = [desc[0] for desc in conn.description]
                features = dict(zip(columns, result))
                
                conn.close()
                return features
            
            conn.close()
            return None
            
        except Exception as e:
            logger.error(f"Error retrieving features: {e}")
            return None
    
    def calculate_derived_features(self, symbol: str = 'U'):
        """Calculate derived features from base data"""
        try:
            conn = duckdb.connect(self.db_path)
            
            # Calculate term structure slope (10Y - 3M)
            conn.execute("""
                UPDATE analytics.ml_features
                SET term_structure_slope = risk_free_rate_10y - risk_free_rate_3m
                WHERE symbol = ? 
                AND risk_free_rate_10y IS NOT NULL 
                AND risk_free_rate_3m IS NOT NULL
            """, [symbol])
            
            # Calculate IV rank and percentile (would need historical IV data)
            # For now, we'll use VIX as a proxy
            conn.execute("""
                WITH vix_stats AS (
                    SELECT 
                        MIN(vix_level) as min_vix,
                        MAX(vix_level) as max_vix,
                        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY vix_level) as median_vix
                    FROM analytics.ml_features
                    WHERE symbol = ? AND vix_level IS NOT NULL
                )
                UPDATE analytics.ml_features
                SET 
                    iv_rank = (vix_level - min_vix) / NULLIF(max_vix - min_vix, 0),
                    iv_percentile = PERCENT_RANK() OVER (ORDER BY vix_level)
                FROM vix_stats
                WHERE analytics.ml_features.symbol = ? 
                AND analytics.ml_features.vix_level IS NOT NULL
            """, [symbol, symbol])
            
            conn.close()
            
            logger.info("Calculated derived features")
            
        except Exception as e:
            logger.error(f"Error calculating derived features: {e}")
    
    def get_metrics(self) -> Dict[str, int]:
        """Return storage metrics"""
        return self._metrics.copy()
    
    def reset_metrics(self):
        """Reset storage metrics"""
        self._metrics = {
            "series_stored": 0,
            "observations_stored": 0,
            "features_updated": 0
        }
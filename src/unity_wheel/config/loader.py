"""Compatibility shim for old config.loader imports.

This redirects to the new unified config system.
"""
from __future__ import annotations


from .unified_config import get_config as _get_unified_config


def get_config() -> None:
    """Get configuration - redirects to unified config."""
    unified = _get_unified_config()
    
    # Return a compatible object that mimics the old structure
    class LegacyConfig:
        def __init__(self, unified_config):
            self._unified = unified_config
            
            # Map old attributes to new ones
            self.operations = type('operations', (), {
                'api': type('api', (), {
                    'max_concurrent_puts': unified_config.trading.max_concurrent_puts,
                    'max_position_pct': unified_config.trading.max_position_size,
                    'min_confidence': unified_config.trading.min_confidence,
                    'max_decision_time': 5.0,  # Default
                    'max_bid_ask_spread': 0.10,  # Default
                    'min_volume': 10,  # Default
                    'min_open_interest': 50  # Default
                })()
            })()
            
            self.trading = type('trading', (), {
                'execution': type('execution', (), {
                    'contracts_per_trade': unified_config.trading.contracts_per_trade,
                    'commission_per_contract': unified_config.trading.commission_per_contract
                })()
            })()
            
            self.database = type('database', (), {
                'path': 'wheel_trades.db'  # Default legacy path
            })()
            
            # Direct mappings
            self.wheel_delta_target = unified_config.trading.target_delta
            self.days_to_expiry_target = unified_config.trading.target_dte
            self.max_position_size = unified_config.trading.max_position_size
    
    return LegacyConfig(_get_unified_config())
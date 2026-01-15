"""
Sector Analysis Module
Provides sector prediction signals and weights for sector rotation strategies
"""

from .enhanced_signals import (
    get_macro_acceleration_signal,
    get_credit_spread_signal,
    get_valuation_signal,
    apply_enhanced_signals_to_sector
)

from .sector_weights_config import OPTIMAL_SECTOR_WEIGHTS

__all__ = [
    'get_macro_acceleration_signal',
    'get_credit_spread_signal',
    'get_valuation_signal',
    'apply_enhanced_signals_to_sector',
    'OPTIMAL_SECTOR_WEIGHTS'
]

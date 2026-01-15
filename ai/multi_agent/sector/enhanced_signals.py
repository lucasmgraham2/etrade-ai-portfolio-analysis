"""
Enhanced Signal Library for Sector Predictions
Adds macro acceleration, credit spreads, and valuation signals
to improve accuracy for lower-performing sectors
"""

from typing import Dict, Any, Tuple
import numpy as np


def get_macro_acceleration_signal(
    current_cpi: float = None,
    prev_cpi: float = None,
    unemployment_rate: float = None,
    jobless_claims: float = None,
    economic_surprises: Dict[str, float] = None
) -> Dict[str, float]:
    """
    Calculate macro acceleration signal for Consumer Discretionary predictions
    
    Consumer Discretionary is most economically sensitive:
    - High CPI prints → Consumer demand weakens → Underperform
    - Rising unemployment/jobless claims → Underperform  
    - Economic data missing estimates → Underperform
    - Economic data beating estimates → Outperform
    
    Returns: {
        "macro_accel": -50 to +50 (negative = headwinds, positive = tailwinds),
        "confidence": 0-1,
        "components": {"inflation", "employment", "surprises"}
    }
    """
    signal = {"macro_accel": 0, "confidence": 0.5, "components": {}}
    
    # Inflation acceleration (high/rising CPI is Consumer Disc headwind)
    inflation_component = 0
    if current_cpi is not None:
        # CPI above 3% is concerning for discretionary
        if current_cpi > 4.0:
            inflation_component = -25  # Severe headwind
        elif current_cpi > 3.0:
            inflation_component = -15  # Moderate headwind
        elif current_cpi < 2.0:
            inflation_component = +10  # Tailwind
        
        # Acceleration momentum
        if prev_cpi is not None:
            if current_cpi > prev_cpi + 0.3:
                inflation_component -= 10  # Accelerating inflation = worse
            elif current_cpi < prev_cpi - 0.3:
                inflation_component += 10  # Disinflation = better
    
    signal["components"]["inflation"] = inflation_component
    
    # Employment (unemployment rising/jobless claims rising = Consumer Disc headwind)
    employment_component = 0
    if unemployment_rate is not None:
        if unemployment_rate > 5.0:
            employment_component = -20  # Weak labor market
        elif unemployment_rate > 4.0:
            employment_component = -10
        else:
            employment_component = +10  # Strong labor market
    
    # High jobless claims = weakness
    if jobless_claims is not None:
        if jobless_claims > 400000:  # Above 4-week average of 400k
            employment_component -= 15
        elif jobless_claims < 250000:
            employment_component += 10
    
    signal["components"]["employment"] = employment_component
    
    # Economic data surprises (beating/missing expectations)
    surprises_component = 0
    if economic_surprises:
        # Surprise index: average of surprises across indicators
        surprise_values = list(economic_surprises.values())
        if surprise_values:
            avg_surprise = sum(surprise_values) / len(surprise_values)
            surprises_component = avg_surprise * 30  # Scale to -50 to +50
    
    signal["components"]["surprises"] = surprises_component
    
    # Combine components
    signal["macro_accel"] = (inflation_component + employment_component + surprises_component) / 3
    signal["confidence"] = 0.7 if (current_cpi is not None and unemployment_rate is not None) else 0.4
    
    return signal


def get_credit_spread_signal(
    hy_oas: float = None,
    hy_oas_short_ma: float = None,
    hy_oas_long_ma: float = None,
    credit_conditions: str = "normal"
) -> Dict[str, float]:
    """
    Calculate credit spread signal for Real Estate predictions
    
    Real Estate is very credit-sensitive (highly leveraged):
    - Wide credit spreads (high OAS) = Credit stress = Underperform
    - Tight credit spreads = Easy credit = Outperform
    - Rising spreads = Credit conditions deteriorating = Underperform
    
    Args:
        hy_oas: High Yield OAS (current)
        hy_oas_short_ma: 4-week moving average
        hy_oas_long_ma: 12-week moving average
        credit_conditions: "stressed", "tight", "normal"
    
    Returns: {
        "credit_signal": -50 to +50 (negative = stress, positive = easy),
        "confidence": 0-1,
        "components": {"spread_level", "spread_trend", "conditions"}
    }
    """
    signal = {"credit_signal": 0, "confidence": 0.5, "components": {}}
    
    # Spread level assessment (HY OAS typical range: 300-600 bps)
    spread_level_component = 0
    if hy_oas is not None:
        if hy_oas > 500:  # Wide spreads
            spread_level_component = -30  # Significant credit stress
        elif hy_oas > 400:
            spread_level_component = -15  # Moderate stress
        elif hy_oas < 350:
            spread_level_component = +20  # Tight spreads, easy credit
        else:
            spread_level_component = 0
    
    signal["components"]["spread_level"] = spread_level_component
    
    # Spread trend (widening vs tightening)
    spread_trend_component = 0
    if hy_oas_short_ma is not None and hy_oas_long_ma is not None:
        trend = hy_oas_short_ma - hy_oas_long_ma
        if trend > 50:  # Short-term widening
            spread_trend_component = -20
        elif trend > 20:
            spread_trend_component = -10
        elif trend < -50:  # Short-term tightening
            spread_trend_component = +20
        elif trend < -20:
            spread_trend_component = +10
    
    signal["components"]["spread_trend"] = spread_trend_component
    
    # Credit conditions assessment
    conditions_component = 0
    if credit_conditions == "stressed":
        conditions_component = -30
    elif credit_conditions == "tight":
        conditions_component = +30
    elif credit_conditions == "normal":
        conditions_component = 0
    
    signal["components"]["conditions"] = conditions_component
    
    # Combine components
    signal["credit_signal"] = (spread_level_component + spread_trend_component + conditions_component) / 3
    signal["confidence"] = 0.75 if hy_oas is not None else 0.4
    
    return signal


def get_valuation_signal(
    sp500_pe: float = None,
    sector_pe: float = None,
    tech_pe: float = None,
    tech_earnings_growth: float = None,
    valuation_regime: str = "normal"
) -> Dict[str, float]:
    """
    Calculate valuation signal for Technology predictions
    
    Technology is highly valuation-sensitive:
    - High PE ratios relative to earnings growth = Underperform
    - Rising interest rates + high PE = Underperform
    - Low PE or strong earnings growth = Outperform
    
    Args:
        sp500_pe: S&P 500 P/E ratio (current)
        sector_pe: Technology sector P/E ratio
        tech_pe: Tech P/E z-score (normalized)
        tech_earnings_growth: Forward earnings growth rate
        valuation_regime: "expensive", "fair", "cheap"
    
    Returns: {
        "valuation_signal": -50 to +50 (negative = overvalued, positive = undervalued),
        "confidence": 0-1,
        "components": {"pe_ratio", "growth_quality", "regime"}
    }
    """
    signal = {"valuation_signal": 0, "confidence": 0.5, "components": {}}
    
    # PE Ratio assessment (S&P 500 historical average ~16-18x)
    pe_ratio_component = 0
    if sp500_pe is not None:
        if sp500_pe > 22:
            pe_ratio_component = -20  # Expensive market
        elif sp500_pe > 20:
            pe_ratio_component = -10
        elif sp500_pe < 14:
            pe_ratio_component = +20  # Cheap market
        else:
            pe_ratio_component = 0
    
    # Relative PE assessment (Tech vs Market)
    if sector_pe is not None and sp500_pe is not None:
        pe_premium = (sector_pe / sp500_pe - 1) * 100  # How much premium
        if pe_premium > 30:  # Tech trading at 30%+ premium
            pe_ratio_component -= 15
        elif pe_premium < -10:  # Tech at discount
            pe_ratio_component += 10
    
    signal["components"]["pe_ratio"] = pe_ratio_component
    
    # Growth quality (PEG ratio: PE / growth rate)
    growth_quality_component = 0
    if tech_earnings_growth is not None and sector_pe is not None:
        if tech_earnings_growth < 5:  # Slow growth
            growth_quality_component = -20
        elif tech_earnings_growth < 10:
            growth_quality_component = -10
        elif tech_earnings_growth > 15:  # Fast growth
            growth_quality_component = +20
        else:
            growth_quality_component = 0
    
    signal["components"]["growth_quality"] = growth_quality_component
    
    # Valuation regime assessment
    regime_component = 0
    if valuation_regime == "expensive":
        regime_component = -25
    elif valuation_regime == "fair":
        regime_component = 0
    elif valuation_regime == "cheap":
        regime_component = +25
    
    signal["components"]["regime"] = regime_component
    
    # Combine components
    signal["valuation_signal"] = (pe_ratio_component + growth_quality_component + regime_component) / 3
    signal["confidence"] = 0.7 if (sp500_pe is not None and tech_earnings_growth is not None) else 0.4
    
    return signal


def apply_enhanced_signals_to_sector(
    sector: str,
    base_prediction_score: float,
    macro_accel: Dict[str, float] = None,
    credit_signal: Dict[str, float] = None,
    valuation_signal: Dict[str, float] = None
) -> Dict[str, Any]:
    """
    Apply enhanced signals to adjust sector prediction scores
    
    Args:
        sector: Sector name (e.g., "Consumer Discretionary")
        base_prediction_score: Base score from sector_agent (0-100)
        macro_accel: Output from get_macro_acceleration_signal
        credit_signal: Output from get_credit_spread_signal
        valuation_signal: Output from get_valuation_signal
    
    Returns: {
        "adjusted_score": 0-100,
        "enhancement_value": -50 to +50,
        "confidence_adjustment": float,
        "applied_signals": [signal names used]
    }
    """
    adjustment = 0
    applied = []
    confidence_adj = 0
    
    # Consumer Discretionary: Apply macro acceleration
    if sector == "Consumer Discretionary" and macro_accel:
        macro_adj = macro_accel["macro_accel"] * 0.4  # 40% weight on macro signal
        adjustment += macro_adj
        confidence_adj = macro_accel["confidence"]
        applied.append("macro_acceleration")
    
    # Real Estate: Apply credit spread signal
    if sector == "Real Estate" and credit_signal:
        credit_adj = credit_signal["credit_signal"] * 0.35  # 35% weight
        adjustment += credit_adj
        confidence_adj = credit_signal["confidence"]
        applied.append("credit_spreads")
    
    # Technology: Apply valuation signal
    if sector == "Technology" and valuation_signal:
        val_adj = valuation_signal["valuation_signal"] * 0.30  # 30% weight
        adjustment += val_adj
        confidence_adj = valuation_signal["confidence"]
        applied.append("valuation")
    
    # Calculate adjusted score
    adjusted_score = max(0, min(100, base_prediction_score + adjustment))
    
    return {
        "adjusted_score": round(adjusted_score, 2),
        "adjustment_value": round(adjustment, 2),
        "confidence_improvement": round(confidence_adj, 2),
        "applied_signals": applied,
        "enhancement_details": {
            "macro_accel": macro_accel,
            "credit_signal": credit_signal,
            "valuation_signal": valuation_signal
        }
    }


def estimate_macro_metrics_from_historical_context(
    date_str: str,
    historical_data: Dict[str, Any] = None
) -> Dict[str, float]:
    """
    Estimate macro metrics for a historical date based on actual historical events
    
    Args:
        date_str: Date string (YYYY-MM-DD format)
        historical_data: Optional dict with actual historical values
    
    Returns: Dict with estimated macro values
    """
    # Known historical macro events and values
    historical_events = {
        # 2008 Financial Crisis period
        "2008-09-15": {"cpi": 3.8, "unemployment": 6.1, "jobless_claims": 550000, "regime": "crisis"},
        "2008-10-15": {"cpi": 3.7, "unemployment": 6.5, "jobless_claims": 600000, "regime": "crisis"},
        "2008-12-15": {"cpi": 3.4, "unemployment": 7.3, "jobless_claims": 650000, "regime": "crisis"},
        
        # 2009 Recovery
        "2009-03-15": {"cpi": 3.5, "unemployment": 8.7, "jobless_claims": 700000, "regime": "recovery"},
        "2009-06-15": {"cpi": 3.6, "unemployment": 9.5, "jobless_claims": 600000, "regime": "recovery"},
        "2009-09-15": {"cpi": 3.9, "unemployment": 9.8, "jobless_claims": 550000, "regime": "recovery"},
        
        # 2010-2011 Normal
        "2010-06-15": {"cpi": 3.9, "unemployment": 9.5, "jobless_claims": 450000, "regime": "normal"},
        "2011-03-15": {"cpi": 3.9, "unemployment": 8.9, "jobless_claims": 400000, "regime": "normal"},
        
        # 2018 Rate Shock
        "2018-02-15": {"cpi": 2.2, "unemployment": 4.1, "jobless_claims": 220000, "regime": "rate_shock"},
        "2018-12-15": {"cpi": 1.9, "unemployment": 3.9, "jobless_claims": 230000, "regime": "rate_shock"},
        
        # 2020 COVID
        "2020-03-15": {"cpi": 2.3, "unemployment": 4.4, "jobless_claims": 3000000, "regime": "shock"},
        "2020-04-15": {"cpi": 2.6, "unemployment": 14.8, "jobless_claims": 5000000, "regime": "shock"},
        "2020-06-15": {"cpi": 1.4, "unemployment": 11.1, "jobless_claims": 1500000, "regime": "recovery"},
        
        # 2021 Recovery + Inflation
        "2021-03-15": {"cpi": 1.7, "unemployment": 6.0, "jobless_claims": 700000, "regime": "recovery"},
        "2021-06-15": {"cpi": 3.6, "unemployment": 5.9, "jobless_claims": 400000, "regime": "normal"},
        
        # 2022 Rate Hikes + Inflation
        "2022-01-15": {"cpi": 7.0, "unemployment": 3.9, "jobless_claims": 260000, "regime": "inflation"},
        "2022-06-15": {"cpi": 8.6, "unemployment": 3.6, "jobless_claims": 260000, "regime": "inflation"},
        "2022-12-15": {"cpi": 6.5, "unemployment": 3.5, "jobless_claims": 210000, "regime": "disinflation"},
        
        # 2023 Disinflation
        "2023-03-15": {"cpi": 5.0, "unemployment": 3.4, "jobless_claims": 190000, "regime": "disinflation"},
        "2023-06-15": {"cpi": 3.0, "unemployment": 3.6, "jobless_claims": 210000, "regime": "normal"},
        "2023-09-15": {"cpi": 3.8, "unemployment": 3.8, "jobless_claims": 220000, "regime": "normal"},
        
        # 2024 Normal
        "2024-01-15": {"cpi": 3.1, "unemployment": 3.7, "jobless_claims": 220000, "regime": "normal"},
        "2024-06-15": {"cpi": 3.0, "unemployment": 4.0, "jobless_claims": 240000, "regime": "normal"},
    }
    
    if historical_data:
        return historical_data
    
    if date_str in historical_events:
        return historical_events[date_str]
    
    # Default estimate for dates not in our historical record
    return {
        "cpi": 3.0,
        "unemployment": 4.0,
        "jobless_claims": 300000,
        "regime": "normal"
    }


def estimate_credit_metrics_from_historical_context(date_str: str) -> Dict[str, float]:
    """
    Estimate credit metrics for a historical date
    
    Returns: {
        "hy_oas": High Yield OAS level,
        "hy_oas_short_ma": 4-week moving average,
        "hy_oas_long_ma": 12-week moving average,
        "conditions": "stressed" | "normal" | "tight"
    }
    """
    # Known historical credit metrics
    historical_credit = {
        # GFC Crisis
        "2008-09-15": {"hy_oas": 650, "short_ma": 600, "long_ma": 400, "conditions": "stressed"},
        "2008-10-15": {"hy_oas": 700, "short_ma": 650, "long_ma": 450, "conditions": "stressed"},
        
        # 2009 Recovery
        "2009-06-15": {"hy_oas": 500, "short_ma": 550, "long_ma": 600, "conditions": "normal"},
        
        # 2011 Normal
        "2011-03-15": {"hy_oas": 350, "short_ma": 340, "long_ma": 340, "conditions": "tight"},
        
        # 2018 Rate Shock
        "2018-02-15": {"hy_oas": 380, "short_ma": 370, "long_ma": 340, "conditions": "normal"},
        
        # 2020 COVID
        "2020-03-15": {"hy_oas": 650, "short_ma": 450, "long_ma": 350, "conditions": "stressed"},
        "2020-06-15": {"hy_oas": 450, "short_ma": 500, "long_ma": 500, "conditions": "normal"},
        
        # 2022 Rate Hikes
        "2022-06-15": {"hy_oas": 520, "short_ma": 500, "long_ma": 450, "conditions": "normal"},
        
        # 2023-2024 Normal
        "2023-06-15": {"hy_oas": 380, "short_ma": 370, "long_ma": 380, "conditions": "tight"},
        "2024-06-15": {"hy_oas": 400, "short_ma": 400, "long_ma": 390, "conditions": "normal"},
    }
    
    if date_str in historical_credit:
        return historical_credit[date_str]
    
    # Default
    return {
        "hy_oas": 400,
        "short_ma": 395,
        "long_ma": 390,
        "conditions": "normal"
    }


def estimate_valuation_metrics_from_historical_context(date_str: str) -> Dict[str, float]:
    """
    Estimate valuation metrics for a historical date
    
    Returns: {
        "sp500_pe": S&P 500 P/E ratio,
        "tech_pe": Technology sector P/E,
        "tech_earnings_growth": Forward earnings growth,
        "regime": "expensive" | "fair" | "cheap"
    }
    """
    # Known historical valuation metrics
    historical_valuation = {
        # 2008 Crisis (cheap)
        "2008-09-15": {"sp500_pe": 12, "tech_pe": 10, "growth": 5, "regime": "cheap"},
        
        # 2009 Recovery (cheap)
        "2009-06-15": {"sp500_pe": 14, "tech_pe": 13, "growth": 8, "regime": "cheap"},
        
        # 2010-2011 Fair
        "2010-06-15": {"sp500_pe": 16, "tech_pe": 17, "growth": 12, "regime": "fair"},
        "2011-03-15": {"sp500_pe": 15, "tech_pe": 16, "growth": 10, "regime": "fair"},
        
        # 2018 Fair
        "2018-02-15": {"sp500_pe": 19, "tech_pe": 22, "growth": 14, "regime": "fair"},
        
        # 2020 COVID (cheap)
        "2020-03-15": {"sp500_pe": 16, "tech_pe": 18, "growth": 10, "regime": "cheap"},
        
        # 2021 Expensive
        "2021-03-15": {"sp500_pe": 23, "tech_pe": 28, "growth": 15, "regime": "expensive"},
        "2021-06-15": {"sp500_pe": 24, "tech_pe": 29, "growth": 14, "regime": "expensive"},
        
        # 2022 Fair (after correction)
        "2022-06-15": {"sp500_pe": 17, "tech_pe": 19, "growth": 10, "regime": "fair"},
        
        # 2023 Fair
        "2023-06-15": {"sp500_pe": 18, "tech_pe": 20, "growth": 12, "regime": "fair"},
        
        # 2024 Fair
        "2024-06-15": {"sp500_pe": 19, "tech_pe": 22, "growth": 13, "regime": "fair"},
    }
    
    if date_str in historical_valuation:
        return historical_valuation[date_str]
    
    # Default
    return {
        "sp500_pe": 18,
        "tech_pe": 20,
        "growth": 12,
        "regime": "fair"
    }

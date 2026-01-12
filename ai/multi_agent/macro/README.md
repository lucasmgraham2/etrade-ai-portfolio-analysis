# Macro Agent - Comprehensive Macroeconomic Analysis System

## Overview

The Macro Agent has been redesigned to provide a robust, data-driven assessment of macroeconomic conditions using a comprehensive set of indicators. The system combines **popular mainstream metrics** with **alternative lesser-known indicators** to produce a confidence score for market direction.

## Architecture

The system consists of three main components:

### 1. **Popular Metrics Analyzer** (`popular_metrics.py`)
Analyzes well-established, widely-followed economic indicators:
- **GDP Growth** - Overall economic expansion
- **Unemployment Rate** - Labor market health
- **CPI Inflation** - Consumer price pressures
- **PPI Inflation** - Producer price pressures (leading indicator)
- **Federal Funds Rate** - Monetary policy stance
- **10-Year Treasury Yield** - Risk-free rate benchmark
- **S&P 500 Performance** - Market momentum
- **ISM Manufacturing PMI** - Manufacturing sector health
- **ISM Services PMI** - Services sector health
- **Retail Sales** - Consumer spending strength
- **Consumer Confidence Index** - Consumer sentiment

### 2. **Alternative Metrics Analyzer** (`alternative_metrics.py`)
Analyzes less mainstream but valuable forward-looking indicators:
- **Yield Curve (2Y-10Y Spread)** - Recession warning signal
- **M2 Money Supply Growth** - Liquidity conditions
- **High Yield Credit Spreads** - Credit market stress
- **Leading Economic Index (LEI)** - Forward-looking economic indicator
- **Sahm Rule** - Real-time recession indicator
- **Dollar Index (DXY)** - Currency strength impact
- **Architecture Billings Index (ABI)** - 9-12 month leading indicator for construction
- **Copper Prices ("Dr. Copper")** - Global growth proxy
- **Luxury Sales Indicator** - High-end consumer health
- **Gold Price Trend** - Fear/uncertainty gauge
- **Corporate Debt to GDP** - Leverage risk assessment

### 3. **Macro Agent** (`macro_agent.py`)
Orchestrates the analysis:
- Runs both analyzers in parallel
- Combines scores using configurable weights
- Checks agreement between metric categories
- Generates comprehensive insights and recommendations

## Scoring System

### Score Scale
- **0-100** where:
  - `100` = Fully Bullish
  - `50` = Neutral
  - `0` = Fully Bearish

### Score Interpretation
- **70-100**: Very Bullish - Strong positive signals
- **60-69**: Bullish - Favorable conditions
- **55-59**: Neutral with Bullish Lean
- **45-54**: Neutral - Mixed signals
- **40-44**: Neutral with Bearish Lean
- **30-39**: Bearish - Challenging conditions
- **0-29**: Very Bearish - Strong negative signals

### Timeframe
**Medium-term (3-12 months)** - Focuses on trends that impact markets over quarters, not days.

## Configuration

### Weights Configuration (`macro_weights_config.json`)

All weights are fully adjustable to customize the analysis:

#### Category Weights
```json
"popular_metrics": {
  "total_weight_in_final_score": 0.60  // 60% of final score
}
"alternative_metrics": {
  "total_weight_in_final_score": 0.40  // 40% of final score
}
```

#### Individual Metric Weights (within each category)
Each metric has an individual weight that determines its influence within its category. Higher weight = more influence.

**Popular Metrics Example:**
```json
"gdp_growth": 0.15,           // 15% weight within popular metrics
"unemployment": 0.12,         // 12% weight
"cpi_inflation": 0.12,        // etc.
```

**Alternative Metrics Example:**
```json
"yield_curve": 0.15,          // 15% weight within alternative metrics
"high_yield_spreads": 0.12,   // 12% weight
"leading_economic_index": 0.12
```

### Adjusting Weights

To customize the analysis, edit `macro_weights_config.json`:

1. **Change category weights** to emphasize popular vs alternative metrics
2. **Change individual weights** to emphasize specific indicators
3. **Adjust scoring thresholds** to change what constitutes "bullish" vs "bearish"
4. **Modify confidence adjustments** for agreement/disagreement scenarios

**Note:** Individual weights within each category should sum to 1.0 (100%)

## Data Sources

### FRED API (Federal Reserve Economic Data)
- **Required**: Yes
- **Cost**: Free
- **Setup**: Get API key from https://fred.stlouisfed.org/
- **Usage**: GDP, unemployment, inflation, rates, yields, M2, LEI, etc.

### Alpha Vantage API
- **Required**: Yes
- **Cost**: Free tier available (rate limited)
- **Setup**: Get API key from https://www.alphavantage.co/
- **Usage**: S&P 500, copper, luxury sales proxies
- **Rate Limit**: 5 calls/minute (free tier)

## Usage

```python
from multi_agent.macro_agent import MacroAgent

# Initialize with API keys
config = {
    "api_keys": {
        "fred": "your_fred_api_key",
        "alpha_vantage": "your_alpha_vantage_key"
    }
}

agent = MacroAgent(config)

# Run analysis
results = await agent.analyze({})

# Access results
confidence_score = results["confidence_score"]["score"]  # 0-100
direction = results["confidence_score"]["direction"]     # BULLISH/NEUTRAL/BEARISH
insights = results["insights"]                           # Detailed analysis
recommendations = results["recommendations"]             # Actionable guidance
```

## Output Structure

```python
{
  "summary": "One-line summary of analysis",
  "confidence_score": {
    "score": 65.3,                    # 0-100 scale
    "direction": "BULLISH",           # BULLISH/NEUTRAL/BEARISH
    "interpretation": "...",          # Detailed interpretation
    "components": {
      "popular_metrics_score": 68.2,
      "alternative_metrics_score": 60.5,
      "popular_weight": 0.60,
      "alternative_weight": 0.40
    },
    "agreement": {
      "level": "moderate",            # high/moderate/low
      "difference": 7.7,
      "note": "..."
    }
  },
  "popular_metrics": { ... },         # Full popular metrics results
  "alternative_metrics": { ... },     # Full alternative metrics results
  "insights": [ ... ],                # Detailed insights array
  "recommendations": [ ... ],         # Actionable recommendations
  "timestamp": "2026-01-12T..."
}
```

## Key Features

### 1. **Agreement Analysis**
The system checks how well popular and alternative metrics agree:
- **High Agreement** (+5 to confidence): Both metric sets pointing same direction
- **Low Agreement** (-5 to confidence): Divergence between metric sets
- Helps identify when signals are mixed and caution is warranted

### 2. **Individual Metric Scoring**
Each metric is scored 0-100 independently:
- Allows you to see which specific metrics are bullish/bearish
- Helps identify key risk factors and positive catalysts
- Transparent scoring logic for each metric

### 3. **Flexible Weighting**
- Easily adjust any weight in the JSON config
- No code changes required
- Can create custom profiles (e.g., "growth-focused", "risk-averse")

### 4. **Comprehensive Recommendations**
- Strategic positioning advice based on score
- Equity allocation guidance
- Sector focus recommendations
- Risk tolerance adjustments
- Tactical considerations

## Monitoring Recommendations

- **Re-run monthly** or when major economic data releases occur
- **Watch for threshold crossings** (e.g., score moving from 62 to 58)
- **Monitor agreement levels** - divergence may signal transition
- **Track individual metrics** - identify which metrics are changing

## Customization Examples

### Emphasize Leading Indicators
```json
{
  "alternative_metrics": {
    "total_weight_in_final_score": 0.50  // Increase from 0.40
  }
}
```

### Focus on Inflation
```json
{
  "popular_metrics": {
    "individual_weights": {
      "cpi_inflation": 0.20,  // Increase from 0.12
      "ppi_inflation": 0.12   // Increase from 0.08
    }
  }
}
```

### Conservative (Recession-Focused)
```json
{
  "alternative_metrics": {
    "individual_weights": {
      "yield_curve": 0.20,      // Increase - key recession signal
      "sahm_rule": 0.15,        // Increase - recession indicator
      "high_yield_spreads": 0.15 // Increase - stress indicator
    }
  }
}
```

## Troubleshooting

### API Rate Limits
- Alpha Vantage free tier: 5 calls/minute
- System includes automatic delays (13 seconds between calls)
- For faster analysis, consider premium Alpha Vantage API

### Missing Data
- Some FRED series may have delays (e.g., GDP quarterly)
- System handles missing data gracefully
- Check individual metric errors in results

### Configuration Errors
- Ensure JSON is valid
- Weights should sum to 1.0 within each category
- System falls back to defaults if config file missing

## Future Enhancements

Potential additions:
- VIX (volatility index)
- MOVE Index (bond volatility)
- Actual luxury brand sales data
- Real-time construction spending
- More granular sector PMIs
- International economic indicators
- Commodity indices

## File Structure

```
multi_agent/
├── macro_agent.py               # Main orchestrator
├── popular_metrics.py           # Popular metrics analyzer
├── alternative_metrics.py       # Alternative metrics analyzer
├── macro_weights_config.json    # Configuration file
└── README_MACRO.md             # This file
```

## Questions or Issues?

The system is designed to be transparent and configurable. If you want to:
- Add new metrics
- Change scoring logic
- Adjust weights
- Customize recommendations

All components are clearly structured and documented for easy modification.

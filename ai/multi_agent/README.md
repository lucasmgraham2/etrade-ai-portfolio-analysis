# Multi-Agent Portfolio Analysis System

A sophisticated multi-agent AI system for comprehensive portfolio analysis that combines market sentiment, macroeconomic indicators, sector performance predictions, and portfolio-specific recommendations.

## Overview

This system uses four specialized AI agents working together to provide actionable investment insights:

### 🤖 Agent Architecture

1) **Sentiment Agent**
   - NewsAPI + Alpha Vantage news with transformer NLP (keyword fallback if NLP unavailable)
   - Daily caching + rate-limit lock for NewsAPI free tier
   - Optional OpenAI backfill when transformer is inconclusive

2) **Macro Agent**
   - Popular + alternative macro metrics
   - Macro confidence score (0–100) with bullish/bearish label

3) **Sector Agent**
   - 11 sectors, cached daily performance; favors fast completion if cache exists
   - Identifies favorable/unfavorable sectors and momentum

4) **Integrator Agent**
   - Blends agent outputs into BUY/HOLD/SELL and top priority actions
   - Summarizes risks and recommended sector tilts

### 🔄 How It Works

```
Portfolio Data → [Sentiment] ↘
                 [Macro]      → Integrator → Recommendations
                 [Sector]    ↗
```

- **Parallel Mode** (default): Sentiment, Macro, and Sector run together, Integrator runs after.
- **Sequential Mode** (flag): Run in series for debugging.

## Installation

### Prerequisites

- Python 3.10+ recommended
- Fresh portfolio JSON from `etrade/get_all_data.py` (created by `run_complete_analysis.py`)

### Setup

1) Install required packages (from repo root):

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r ai/requirements.txt
pip install transformers torch  # NLP extras
```

2) Configure API keys in `.env` at repo root:

```env
ALPHA_VANTAGE_API_KEY=...
NEWSAPI_KEY=...
FRED_API_KEY=...
OPENAI_API_KEY=...  # optional
```

API key help: [API_KEYS_GUIDE.md](../../API_KEYS_GUIDE.md).

## Usage

### Basic Usage

Run analysis with latest portfolio data (root or ai/):

```bash
python ai/run_multi_agent.py
# or
cd ai && python run_multi_agent.py
```

Flags (see `--help`):
- `--portfolio <path>`: use a specific portfolio JSON (default: latest in etrade_reports)
- `--sequential`: run agents in sequence

### Output

Outputs saved to `ai/analysis_reports/`:

- `multi_agent_analysis_YYYYMMDD_HHMMSS.json` (machine-readable)
- `multi_agent_report_YYYYMMDD_HHMMSS.txt` (human-readable)


## Configuration

Edit the `load_config()` function in `run_multi_agent.py` to customize:

```python
config = {
    "api_keys": {
        "alpha_vantage": "YOUR_KEY",
        "newsapi": "YOUR_KEY",
        "fred": "YOUR_KEY"
    },
    "risk_tolerance": "moderate",  # conservative, moderate, aggressive
    "sentiment": {
        "lookback_days": 7  # Days of sentiment data to analyze
    },
    "sector": {
        "lookback_days": 90  # Days of sector performance to analyze
    }
}
```

## Architecture Details

### Base Agent Class

Provides logging, timing, validation, and consistent result shape.

### Orchestrator

- Registers agents
- Runs in parallel groups or sequentially
- Passes context and compiles results
- Emits consolidated report content

### Data Flow

```python
orchestrator.register_agent(SentimentAgent(config))
orchestrator.register_agent(MacroAgent(config))
orchestrator.register_agent(SectorAgent(config))
orchestrator.register_agent(IntegratorAgent(config))
results = await orchestrator.run_parallel([["Sentiment", "Macro", "Sector"], ["Integrator"]])
report = orchestrator.generate_report()
```

## Performance

Typical execution (with cache + transformers downloaded):
- Parallel: ~1–2 minutes if downloading models first time; faster on cache hits
- Sequential: slower; use for debugging

## Troubleshooting

### No portfolio data found
- Run `python etrade/get_all_data.py` (or `python run_complete_analysis.py`) to refresh portfolio JSON.

### API rate limiting
- NewsAPI free tier: agent caches per day and stops after 429; use paid tier or accept partial coverage.
- Alpha Vantage: 5 calls/min free; sector agent caches daily data to stay within limits.

### Import/model download issues
- Ensure venv is activated and deps installed.
- First transformer download can be large; reruns use local cache.

## Future Enhancements

- [ ] Real-time data streaming
- [ ] Machine learning model integration for predictions
- [ ] Options strategy recommendations
- [ ] Tax optimization suggestions
- [ ] Multi-portfolio comparison
- [ ] Web dashboard for visualization
- [ ] Automated trade execution (with approval)
- [ ] Backtesting framework


## Support

For issues or questions, please open an issue in the GitHub repository.

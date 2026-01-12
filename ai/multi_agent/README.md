# Multi-Agent Portfolio Analysis System

A sophisticated multi-agent AI system for comprehensive portfolio analysis that combines market sentiment, macroeconomic indicators, sector performance predictions, and portfolio-specific recommendations.

## Overview

This system uses four specialized AI agents working together to provide actionable investment insights:

### 🤖 Agent Architecture

1. **Sentiment Agent** - Analyzes market mood and investor psychology
   - Scans financial news (via NewsAPI, Alpha Vantage)
   - Monitors social media sentiment (Twitter/X)
   - Analyzes forum discussions (Reddit, StockTwits)
   - Uses NLP and semantic search for comprehensive sentiment analysis

2. **Macro Agent** - Evaluates economic environment
   - Tracks GDP growth and trends
   - Monitors inflation rates (CPI, PPI)
   - Analyzes interest rates (Federal Funds Rate)
   - Assesses unemployment data
   - Evaluates Treasury yields and market indices
   - Calculates overall market favorability score (0-100)

3. **Sector Agent** - Predicts sector outperformance
   - Analyzes 11 major market sectors
   - Uses historical performance data
   - Identifies sector rotation patterns
   - Integrates economic cycle analysis
   - Provides ML-based predictions for sector winners

4. **Integrator Agent** - Synthesizes insights into actions
   - Combines outputs from all agents
   - Generates position-specific recommendations (BUY/SELL/HOLD)
   - Suggests portfolio rebalancing strategies
   - Prioritizes actions by urgency and impact
   - Assesses overall portfolio risk

### 🔄 How It Works

```
Portfolio Data → [Sentiment] → 
                 [Macro]     →  [Integrator] → Recommendations
                 [Sector]    →
```

The **Orchestrator** coordinates agent execution:
- **Parallel Mode** (default): Sentiment, Macro, and Sector agents run simultaneously for speed
- **Sequential Mode**: Agents run in order, passing context between them

## Installation

### Prerequisites

- Python 3.8+
- Active E*TRADE portfolio data (from parent etrade module)

### Setup

1. Install required packages:

```bash
cd ai
pip install -r requirements.txt
```

2. (Optional) Configure API keys for real-time data:

Create a `.env` file in the project root:

```env
# Financial Data
ALPHA_VANTAGE_API_KEY=your_key_here
NEWSAPI_KEY=your_key_here
FRED_API_KEY=your_key_here

# Social Media (optional)
TWITTER_BEARER_TOKEN=your_token_here
```

**API Key Sources:**
- [Alpha Vantage](https://www.alphavantage.co/support/#api-key) - Free tier: 5 calls/min, 500/day
- [NewsAPI](https://newsapi.org/register) - Free tier: 100 requests/day
- [FRED](https://fred.stlouisfed.org/docs/api/api_key.html) - Free with registration

> **Note:** The system works without API keys using simulated data for development/testing.

## Usage

### Basic Usage

Run analysis with most recent portfolio data:

```bash
python run_multi_agent.py
```

### Advanced Usage

Specify custom portfolio file:

```bash
python run_multi_agent.py --portfolio ../etrade_reports/etrade_data_20260111_120000.json
```

Run agents sequentially (useful for debugging):

```bash
python run_multi_agent.py --sequential
```

### Output

The system generates two files:

1. **JSON Results** (`multi_agent_analysis_YYYYMMDD_HHMMSS.json`)
   - Complete raw data from all agents
   - Detailed metrics and scores
   - Machine-readable format

2. **Text Report** (`multi_agent_report_YYYYMMDD_HHMMSS.txt`)
   - Human-readable analysis
   - Executive summary
   - Top priority actions
   - Risk assessment
   - Agent-by-agent breakdown


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

All agents inherit from `BaseAgent` which provides:
- Error handling and logging
- Execution timing
- Result formatting
- Context validation

### Orchestrator

The `AgentOrchestrator` manages:
- Agent registration
- Execution coordination (parallel/sequential)
- Context passing between agents
- Result compilation
- Report generation

### Data Flow

```python
# 1. Load portfolio
portfolio_data = load_portfolio_data()

# 2. Create orchestrator
orchestrator = AgentOrchestrator(portfolio_data)

# 3. Register agents
orchestrator.register_agent(SentimentAgent(config))
orchestrator.register_agent(MacroAgent(config))
orchestrator.register_agent(SectorAgent(config))
orchestrator.register_agent(IntegratorAgent(config))

# 4. Run analysis (parallel)
results = await orchestrator.run_parallel([
    ["Sentiment", "Macro", "Sector"],  # Run these in parallel
    ["Integrator"]                      # Run this after
])

# 5. Generate report
report = orchestrator.generate_report()
```

## Performance

Typical execution times:

- **Parallel Mode**: 3-8 seconds
- **Sequential Mode**: 8-15 seconds
- **With API calls**: 15-30 seconds (rate limiting)

## Troubleshooting

### No portfolio data found

Ensure you've run the E*TRADE data collection first:

```bash
cd ../etrade
python get_all_data.py
```

### API rate limiting

If using Alpha Vantage free tier (5 calls/min), the Sector Agent will automatically rate-limit. Consider:
- Using simulated data for development
- Upgrading to paid tier
- Implementing caching for API responses

### Import errors

Make sure you're running from the `ai` directory:

```bash
cd ai
python run_multi_agent.py
```

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

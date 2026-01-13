# E*TRADE AI Portfolio Analysis

Comprehensive portfolio analysis with advanced multi-agent AI system for investment recommendations.

## 🚀 New: Multi-Agent Analysis System

The project now includes a sophisticated multi-agent AI system that provides:

- **Sentiment Analysis**: Market psychology from news, social media, and forums
- **Macro Analysis**: Economic indicators and market favorability scoring
- **Sector Predictions**: ML-based sector outperformance forecasts  
- **Integrated Recommendations**: BUY/SELL/HOLD decisions with prioritized actions
- **Risk Assessment**: Multi-factor portfolio risk evaluation

See [ai/multi_agent/README.md](ai/multi_agent/README.md) for full details.

## 📁 Project Structure

```
etrade-ai-portfolio-analysis/
├── etrade/                      # E*TRADE data fetching
│   ├── get_all_data.py         # Fetch portfolio data from E*TRADE
│   ├── generate_ai_prompt.py   # Legacy prompt generator (deprecated)
│   ├── requirements.txt        # E*TRADE dependencies
│   └── etrade_reports/         # Downloaded portfolio data (gitignored)
│
├── ai/                          # AI analysis tools
│   ├── multi_agent/            # ⭐ NEW: Multi-agent analysis system
│   │   ├── base_agent.py       # Base agent class
│   │   ├── orchestrator.py     # Agent coordinator
│   │   ├── sentiment_agent.py  # Market sentiment analysis
│   │   ├── macro_agent.py      # Macroeconomic evaluation
│   │   ├── sector_agent.py     # Sector predictions
│   │   ├── integrator_agent.py # Portfolio recommendations
│   │   └── README.md           # Multi-agent documentation
│   ├── run_multi_agent.py      # ⭐ Run multi-agent analysis
│   ├── portfolio_advisor.py    # Legacy single-agent advisor (deprecated)
│   ├── daily_analysis.py       # Automated daily analysis
│   ├── view_analyses.py        # View past analyses
│   ├── requirements.txt        # AI dependencies
│   └── README.md               # AI module documentation
│
├── .env                         # Your API keys (create from .env.example)
├── .env.example                # Template for API keys
└── README.md                   # This file
```

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone <your-repo-url>
cd etrade-ai-portfolio-analysis
```

### 2. Set Up API Keys

**For live market data, get free API keys:**

1. Copy the environment template:
   ```bash
   Copy-Item .env.example .env
   ```

2. Get your free API keys (takes ~10 minutes):
   - **Alpha Vantage**: https://www.alphavantage.co/support/#api-key
   - **NewsAPI**: https://newsapi.org/register  
   - **FRED**: https://fred.stlouisfed.org/docs/api/api_key.html

3. Add them to `.env`:
   ```
   ALPHA_VANTAGE_API_KEY=your_key_here
   NEWSAPI_KEY=your_key_here
   FRED_API_KEY=your_key_here
   ```

**See [API_KEYS_GUIDE.md](API_KEYS_GUIDE.md) for detailed setup instructions.**

**Note:** The system works without API keys using simulated data, but live data provides real market analysis.

#### E*TRADE API (Required for Portfolio Data)
1. Go to [E*TRADE Developer Portal](https://developer.etrade.com/home)
2. Create an application to get your Consumer Key and Secret
3. Add them to `.env`:
   ```
   ETRADE_CONSUMER_KEY=your_actual_key
   ETRADE_CONSUMER_SECRET=your_actual_secret
   ```

### 3. Install Dependencies

Recommended: use a local virtual environment to keep deps isolated.

```bash
# Create venv
python -m venv .venv

# Activate (PowerShell)
.\.venv\Scripts\Activate.ps1

# Base requirements
pip install -r etrade/requirements.txt
pip install -r ai/requirements.txt

# NLP upgrades for transformer-based sentiment (optional but recommended)
pip install transformers torch
```

### 4. Run Complete Analysis (Recommended)

**One command to do everything:**
```bash
python run_complete_analysis.py
```

This will:
- Fetch your portfolio data from E*TRADE
- Run multi-agent AI analysis
- Generate JSON + text reports in ai/analysis_reports/

**Or run components separately:**

**Fetch Portfolio Data:**
```bash
python etrade/get_all_data.py
```

**Run AI Analysis:**
```bash
python ai/run_multi_agent.py
```

**Daily Analysis:**
```bash
python ai/daily_analysis.py
```

## 📊 What You Can Do

### Complete Analysis (Recommended)
```bash
python run_complete_analysis.py
```
Runs the full pipeline: fetches data and runs multi-agent AI analysis.

### Daily Analysis
```bash
python ai/daily_analysis.py
```
Automatically fetches fresh data and runs multi-agent AI analysis.

### View Past Analyses
```bash
python ai/view_analyses.py
```
Browse and compare historical AI recommendations.

### Manual Steps
If you prefer to run each step manually:
1. Fetch data: `python etrade/get_all_data.py`
2. Run analysis: `python ai/run_multi_agent.py`

## 🛠️ Troubleshooting

### "No module named 'rauth'"
```bash
cd etrade
pip install -r requirements.txt
```

### E*TRADE Authentication Issues
- Ensure your Consumer Key and Secret are correct
- Check that your E*TRADE app is approved for production access
- Try generating new credentials if needed

## 📝 License

[Add your license here]

## 🤝 Contributing

[Add contribution guidelines here]

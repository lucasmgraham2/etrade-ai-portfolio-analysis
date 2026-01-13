# E*TRADE AI Portfolio Analysis

Multi-agent AI that ingests your E*TRADE portfolio, runs sentiment/macro/sector analysis, and produces prioritized actions with risk context.

## What’s inside

- **Sentiment Agent** – transformer/NLP on NewsAPI + Alpha Vantage news, with keyword fallback
- **Macro Agent** – popular + alternative metrics for a 0–100 macro confidence score
- **Sector Agent** – sector momentum/rotation with cached daily data
- **Integrator** – blends agent outputs into BUY/HOLD/SELL plus top actions and risks

See [ai/multi_agent/README.md](ai/multi_agent/README.md) for deeper architecture.

## Project structure

```
etrade-ai-portfolio-analysis/
├── etrade/                  # Fetch E*TRADE data (auth + reports, gitignored)
├── ai/                      # Multi-agent analysis system
│   ├── multi_agent/         # Agents, orchestrator, configs
│   ├── run_multi_agent.py   # Run AI analysis only
│   ├── daily_analysis.py    # Scheduled/daily runner
│   └── view_analyses.py     # Browse saved reports
├── run_complete_analysis.py # End-to-end: fetch + analyze + report
├── .env.example             # API key template
└── README.md                # This file
```

## Quick start (Windows, PowerShell)

```bash
git clone <your-repo-url>
cd etrade-ai-portfolio-analysis

# Create and activate venv
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install deps
pip install -r etrade/requirements.txt
pip install -r ai/requirements.txt

# NLP extras for transformer sentiment (recommended)
pip install transformers torch

# Copy env template and add keys
Copy-Item .env.example .env
```

### Minimal .env (live data)
```
ETRADE_CONSUMER_KEY=...
ETRADE_CONSUMER_SECRET=...
ALPHA_VANTAGE_API_KEY=...
NEWSAPI_KEY=...
FRED_API_KEY=...
OPENAI_API_KEY=...   # optional, used as secondary NLP when transformer uncertain
```
See [API_KEYS_GUIDE.md](API_KEYS_GUIDE.md) for details.

## Run it

Full pipeline (fetch + analyze + reports):
```bash
python run_complete_analysis.py
```

Just AI analysis using latest fetched data:
```bash
python ai/run_multi_agent.py
```

Daily automation helper:
```bash
python ai/daily_analysis.py
```

View saved reports:
```bash
python ai/view_analyses.py
```

Outputs land in:
- Fresh E*TRADE data: `etrade/etrade_reports/`
- AI reports: `ai/analysis_reports/` (JSON + text)

## Notes and tips

- Hugging Face may warn about symlinks on Windows; safe to ignore, or enable Developer Mode for faster caching.
- NewsAPI free tier is capped; the sentiment agent caches per day and locks when 429s are hit.
- Without API keys, the system will simulate data; results are for testing only.

## Troubleshooting

- Missing `rauth` or E*TRADE auth libs: `pip install -r etrade/requirements.txt`
- Authentication issues: regenerate E*TRADE keys, confirm production access, re-run auth when prompted.
- Slow downloads for transformers: first run pulls model weights; subsequent runs use cache.

## Contributing

PRs welcome—add tests/docs for new features. License: MIT (see [LICENSE](LICENSE)).

# AI Analysis

This folder contains the multi-agent AI system for comprehensive portfolio analysis and investment recommendations.

See the full documentation at [ai/multi_agent/README.md](multi_agent/README.md).

## Quick Start

Use a virtual environment and install AI deps plus NLP extras for transformer sentiment:
```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r ai/requirements.txt
pip install transformers torch
```

Run the multi-agent analysis from repo root or `ai/`:
```bash
python ai/run_multi_agent.py
# or
cd ai && python run_multi_agent.py
```

This will:
- Load your latest E*TRADE portfolio data from `etrade/etrade_reports/`
- Run Sentiment, Macro, Sector, and Integrator agents (parallel by default)
- Save JSON and text reports to `ai/analysis_reports/`
- Print an executive summary with top actions

## Viewing Saved Analyses

Multi-agent reports are saved in the `ai/analysis_reports/` folder.

**List saved reports:**
```bash
python view_analyses.py
```

**Open the latest report:**
```bash
notepad analysis_reports\\multi_agent_report_YYYYMMDD_HHMMSS.txt
```

---

## Legacy Advisor (Optional)

`portfolio_advisor.py` is deprecated in favor of the multi-agent system and kept only for historical reference.

## What You Get

The AI will analyze your portfolio and provide:

1. **Executive Summary** - Key findings & top 3 recommendations
2. **Performance Assessment** - What's working, what's not
3. **Risk Analysis** - Concentration risk, diversification gaps
4. **Growth Opportunities** - Where to deploy cash, position sizing
5. **Daily Action Items** - What to monitor today
6. **Position-by-Position Guidance** - Hold/Buy/Sell recommendations

---

## Output

- **Screen Output:** Executive summary and top actions
- **Saved Files:** `analysis_reports/multi_agent_analysis_*.json` and `analysis_reports/multi_agent_report_*.txt`
- **Reusable:** Run daily to stay updated

---



# AI Portfolio Advisor

This folder contains scripts to get AI-powered portfolio analysis and investment advice with **automatic fresh data fetching**.

## Quick Start (Recommended)

### Option 1: OpenAI (GPT-4)
**Easiest to get started** - Most popular AI API

1. Add your OpenAI API key to the `.env` file in the parent folder:
   ```
   OPENAI_API_KEY="your-key-here"
   ```
   Get a key from [OpenAI](https://platform.openai.com/api-keys)

2. Install the library:
   ```bash
   pip install openai
   ```

3. Run the analysis (automatically fetches fresh data):
   ```bash
   cd ai
   python quick_start_openai.py
   ```

**What happens:**
- ✅ Fetches fresh portfolio data from E*TRADE API
- ✅ Loads data and generates comprehensive AI prompt
- ✅ Analyzes with GPT-4
- ✅ Saves timestamped analysis file
- ✅ Prints recommendations to console

**Cost:** ~$0.20-0.50 per analysis (GPT-4o) or ~$0.01-0.05 (GPT-4o-mini)

---

## Viewing Saved Analyses

All AI analyses are automatically saved as timestamped files in the `ai/` folder.

**View all your saved analyses:**
```bash
python view_analyses.py
```

This shows:
- List of all analysis files
- Portfolio value at time of analysis
- Generation date and time
- File sizes

**Open a specific analysis:**
```bash
notepad ai_analysis_20260101_193727.txt
```

---

## Full Featured Advisor (Optional)

Use `portfolio_advisor.py` for interactive mode with multiple AI provider support:

```bash
cd ai
python portfolio_advisor.py
```

Supports:
- **OpenAI**


*Note: This version does not auto-fetch fresh data. Use `quick_start_openai.py` for automatic data updates.*

---

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

- **Screen Output:** Full analysis printed to console
- **Saved File:** `ai_analysis_YYYYMMDD_HHMMSS.txt` in the `ai/` folder
- **Reusable:** Run daily after fetching new portfolio data

---

## API Key Setup

### OpenAI
- Sign up: https://platform.openai.com/signup
- Get key: https://platform.openai.com/api-keys
- Pricing: https://openai.com/api/pricing/


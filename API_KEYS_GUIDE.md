# API Keys Setup Guide - Live Market Data

This guide will help you get free API keys to enable live market data in your portfolio analysis.

## Required APIs (All Free)

### 1. Alpha Vantage - Stock Data & News Sentiment
**What it provides:** Stock prices, sector ETF data, company news sentiment  
**Free tier:** 5 calls/minute, 500 calls/day  
**Best for:** Sector performance, news sentiment analysis

**How to get:**
1. Go to https://www.alphavantage.co/support/#api-key
2. Enter your email and click "GET FREE API KEY"
3. Copy the API key they send you
4. Add to `.env`: `ALPHA_VANTAGE_API_KEY=your_key_here`

---

### 2. NewsAPI - Financial News Articles
**What it provides:** Recent news articles about stocks and companies  
**Free tier:** 100 requests/day (enough for daily analysis)  
**Best for:** Sentiment analysis from financial news

**How to get:**
1. Go to https://newsapi.org/register
2. Sign up with your email
3. Verify your email
4. Copy your API key from the dashboard
5. Add to `.env`: `NEWSAPI_KEY=your_key_here`

---

### 3. FRED - Macroeconomic Data
**What it provides:** GDP, inflation (CPI), interest rates, unemployment, Treasury yields  
**Free tier:** Unlimited requests  
**Best for:** Macro economic analysis

**How to get:**
1. Go to https://fred.stlouisfed.org/
2. Click "My Account" → "Create Account"
3. Sign up with your email
4. Once logged in, go to https://fred.stlouisfed.org/docs/api/api_key.html
5. Click "Request API Key"
6. Fill out the simple form (just need name and email)
7. Copy your API key
8. Add to `.env`: `FRED_API_KEY=your_key_here`

---

## Setup Instructions

### Step 1: Copy the environment template
```powershell
# In your project root directory
Copy-Item .env.example .env
```

### Step 2: Edit the .env file
Open `.env` in your editor and add your API keys:
```
ALPHA_VANTAGE_API_KEY=ABC123XYZ789
NEWSAPI_KEY=def456uvw012
FRED_API_KEY=ghi789rst345
```

### Step 3: Test the configuration
```powershell
cd ai
python run_multi_agent.py
```

You should see:
```
🔑 Configured APIs: alpha_vantage, newsapi, fred
```

Instead of:
```
⚠️  No API keys configured - using simulated data
```


## Troubleshooting

### "API key invalid" errors
- Double-check you copied the entire key (no spaces)
- Verify your email with the API provider
- Some services require account verification before keys work

### "Rate limit exceeded"
- Alpha Vantage: Wait 12 seconds between calls (automatic in the code)
- NewsAPI: Limit reached (100/day) - wait until tomorrow
- Solution: The agents will fall back to simulated data if rate limited

### APIs not being used
- Check that `.env` is in the project root (not in `ai/` folder)
- Verify environment variables are loading: add `print(os.getenv("ALPHA_VANTAGE_API_KEY"))` to test
- Restart your terminal/editor after editing `.env`

---

## What Each Agent Uses

| Agent | Primary API | Fallback |
|-------|-------------|----------|
| **Sentiment** | NewsAPI or Alpha Vantage | Simulated patterns |
| **Macro** | FRED | Simulated economic data |
| **Sector** | Alpha Vantage | Simulated sector returns |
| **Integrator** | (Uses other agents' data) | Always works |

---

## Cost & Limits Summary

All recommended APIs are **100% free** for personal use:

- **Alpha Vantage:** 500 calls/day (plenty for 1-2 analysis runs)
- **NewsAPI:** 100 requests/day (enough for daily analysis)
- **FRED:** Unlimited (government data)

**No credit card required for any of these services.**

---

## Next Steps

1. Get your 3 free API keys (takes ~10 minutes)
2. Add them to `.env`
3. Run `python ai/run_multi_agent.py`
4. Watch it fetch real live market data!

You'll notice the analysis takes 2-5 seconds instead of milliseconds - that's the agents doing real work fetching and analyzing live data.

# Sentiment Analysis Workarounds for Free-Tier APIs

## Problem
NewsAPI free tier is extremely limited (100 requests/day), which gets exhausted quickly with multiple portfolio symbols.

## Solutions Implemented

### 1. **Smart Source Prioritization**
- **Alpha Vantage is Primary**: Always use Alpha Vantage NEWS_SENTIMENT as the main source (reliable, good quota)
- **Supplemental Sources**: NewsAPI and Finnhub are supplemental only
- **Graceful Degradation**: If supplemental sources are unavailable, use Alpha Vantage at 100% weight

### 2. **Dynamic Weight Adjustment**
The merge function now adapts based on actual data availability:
- **Both sources have data**: 60% Alpha Vantage + 40% supplemental
- **Only Alpha Vantage has data**: 100% Alpha Vantage (not diluted)
- **Only supplemental has data**: 100% supplemental
- **Neither has data**: Neutral (0 score)

This prevents the previous issue where empty NewsAPI data was diluting good Alpha Vantage data.

### 3. **Finnhub Integration (Optional)**
Added Finnhub as a free alternative news source:
- **Free tier**: 60 API calls/minute
- **Company news endpoint**: Gets recent news for each symbol
- **NLP analysis**: Uses same transformer-based sentiment analysis
- **To enable**: Add `FINNHUB_API_KEY` to your `.env` file
- **Get free key**: https://finnhub.io/register

### 4. **Daily Lock System**
- Once NewsAPI hits 429 rate limit, a daily lock file is created
- Subsequent runs skip NewsAPI entirely for the rest of the day
- Prevents noisy warnings and wasted API calls
- Lock resets automatically the next day

### 5. **Meaningful Data Detection**
- Checks if supplemental sources returned actual data vs. empty responses
- Only merges sources when they have meaningful content
- Avoids blending with 0-value data that dilutes sentiment

## How to Use

### Option 1: Alpha Vantage Only (Current State)
**Pros**: Reliable, consistent, good quota  
**Cons**: Single source

Your current setup already works well with this approach.

### Option 2: Add Finnhub (Recommended)
1. Register at https://finnhub.io/register (free)
2. Add to `.env`:
   ```
   FINNHUB_API_KEY=your_key_here
   ```
3. Sentiment agent will automatically use it as backup

**Pros**: Free, good quota, adds diversity  
**Cons**: Requires another API key

### Option 3: Upgrade NewsAPI (Production)
If you need the highest fidelity:
- Upgrade to NewsAPI paid tier ($449/mo for business)
- Or use NewsAPI business for critical symbols only

## Current Behavior

With free-tier NewsAPI exhausted:
```
[Sentiment] NewsAPI daily cap previously hit; skipping NewsAPI calls today
[Sentiment] Using Alpha Vantage only (supplemental sources unavailable)
```

This is **working as intended**. Alpha Vantage provides quality sentiment data for all symbols.

## Configuration Options

In `run_multi_agent.py`, the sentiment agent config:
```python
{
    "api_keys": {
        "alpha_vantage": os.getenv("ALPHA_VANTAGE_KEY"),
        "newsapi": os.getenv("NEWSAPI_KEY"),  # Optional
        "finnhub": os.getenv("FINNHUB_API_KEY"),  # Optional
        "openai": os.getenv("OPENAI_API_KEY")  # Optional
    },
    "lookback_days": 7,
    "newsapi_mode": "free",  # free | everything | disabled
    "newsapi_max_calls": 15,  # Cap per run
    "newsapi_page_size": 25  # Articles per query
}
```

## Recommendations

**For your use case**:
1. ✅ Keep using Alpha Vantage as primary (you already have this)
2. ✅ Let NewsAPI quota exhaust naturally (daily lock prevents spam)
3. 🟡 Consider adding Finnhub for extra coverage (free, easy)
4. ❌ Don't upgrade NewsAPI unless you absolutely need NLP on fresh articles

**Alpha Vantage sentiment is sufficient for most portfolio analysis needs.**

## Metrics

Current performance with Alpha Vantage only:
- ✓ All 10 symbols get sentiment scores
- ✓ Pre-computed sentiment (no rate limit concerns)
- ✓ Fast execution (~6 seconds)
- ✓ Reliable daily data

## Alternative Approaches (Not Implemented)

If you want even more sources, you could add:
- **Reddit API**: Wallstreetbets sentiment (free)
- **Twitter API**: Social sentiment (limited free tier)
- **StockTwits**: Trader sentiment (free)
- **Benzinga**: News aggregator (paid)

Let me know if you want any of these implemented!

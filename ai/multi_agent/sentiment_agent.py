"""
Sentiment Agent
Analyzes market sentiment from news, social media, and forums
"""

from typing import Dict, Any, List
from datetime import datetime, timedelta
import asyncio
import aiohttp
import json
from .base_agent import BaseAgent

# NLP for sentiment analysis
try:
    from transformers import pipeline
    NLP_AVAILABLE = True
    # Load financial sentiment model (DistilBERT fine-tuned on financial data)
    try:
        sentiment_analyzer = pipeline(
            "sentiment-analysis",
            model="mrm8488/distilbert-financial-sentiment-analysis",
            device=-1  # CPU; use 0 for GPU
        )
    except Exception as e:
        # Fallback to general sentiment model if financial model unavailable
        sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
except ImportError:
    NLP_AVAILABLE = False
    sentiment_analyzer = None

try:
    from openai import AsyncOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


class SentimentAgent(BaseAgent):
    """
    Analyzes market sentiment using NLP transformer models on news articles.
    
    Uses:
    - DistilBERT transformer models for NLP-based sentiment analysis
    - News articles from Alpha Vantage (pre-computed) and NewsAPI (raw articles for NLP)
    - Keyword matching as fallback only if NLP/APIs unavailable
    
    Note: NewsAPI free tier has rate limits. For production use, upgrade to paid tier
    or use alternative news APIs with higher rate limits.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("Sentiment", config)
        self.api_keys = config.get("api_keys", {}) if config else {}
        self.lookback_days = config.get("lookback_days", 7) if config else 7
        # NewsAPI options (free-plan adherence by default)
        self.newsapi_mode = (config.get("newsapi_mode") if config else None) or "free"  # free | everything | disabled
        self.newsapi_max_calls = (config.get("newsapi_max_calls") if config else None) or 15  # cap per run
        self.newsapi_page_size = (config.get("newsapi_page_size") if config else None) or 25  # reduce load

        # Track whether we've already warned about missing NLP to avoid noisy logs
        self._nlp_warning_logged = False
        if not (NLP_AVAILABLE and sentiment_analyzer):
            self.log(
                "NLP transformers not available; falling back to keyword-based sentiment. Install `transformers` for better accuracy.",
                "WARNING",
            )
            self._nlp_warning_logged = True
        
        # Initialize OpenAI client if available
        self.openai_client = None
        if OPENAI_AVAILABLE and self.api_keys.get("openai"):
            self.openai_client = AsyncOpenAI(api_key=self.api_keys["openai"])
        
    async def analyze(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform sentiment analysis
        
        Args:
            context: Contains portfolio data with symbols
            
        Returns:
            Sentiment analysis results
        """
        # Validate context
        if not self.validate_context(context, ["portfolio"]):
            return {"error": "Missing portfolio data"}
        
        portfolio = context["portfolio"]
        symbols = portfolio.get("summary", {}).get("unique_symbols", [])
        
        if not symbols:
            return {"error": "No symbols found in portfolio"}
        
        self.log(f"Analyzing sentiment for {len(symbols)} symbols: {', '.join(symbols)}")
        
        # Analyze news sentiment only
        news_sentiment = await self._analyze_news_sentiment(symbols)
        
        # Aggregate sentiment scores from news
        overall_sentiment = self._aggregate_sentiment(symbols, news_sentiment)
        
        # Generate insights
        insights = self._generate_insights(overall_sentiment)
        
        results = {
            "summary": self._create_summary(overall_sentiment),
            "overall_sentiment": overall_sentiment,
            "news_sentiment": news_sentiment,
            "insights": insights,
            "recommendations": self._generate_recommendations(overall_sentiment),
            "data_sources": ["news"],
            "analysis_period_days": self.lookback_days
        }
        
        # Generate AI reasoning
        ai_reasoning = await self._generate_sentiment_ai_reasoning(results, symbols)
        results["ai_reasoning"] = ai_reasoning
        
        return results
    
    async def _analyze_news_sentiment(self, symbols: List[str]) -> Dict[str, Any]:
        """
        Analyze news sentiment for portfolio symbols
        
        Uses APIs like:
        - Alpha Vantage News Sentiment (primary, reliable)
        - NewsAPI (supplemental when quota available)
        - Finnhub (free alternative)
        
        Workarounds for free-tier limitations:
        - Prioritize high-value positions for NewsAPI quota
        - Dynamic weight adjustment based on data availability
        - Finnhub as backup for additional coverage
        """
        self.log("Fetching news sentiment...")
        
        has_av = "alpha_vantage" in self.api_keys
        has_newsapi = "newsapi" in self.api_keys
        has_finnhub = "finnhub" in self.api_keys
        
        # Alpha Vantage is primary and most reliable
        if has_av:
            av_data = await self._fetch_alpha_vantage_news(symbols)
            
            # Try supplemental sources if available
            supplemental_data = None
            if has_newsapi:
                newsapi_data = await self._fetch_newsapi_data(symbols)
                # Only use if we got meaningful data (not just empty due to quota)
                if newsapi_data and self._has_meaningful_data(newsapi_data):
                    supplemental_data = newsapi_data
            
            # Finnhub as additional free source
            if has_finnhub and not supplemental_data:
                finnhub_data = await self._fetch_finnhub_news(symbols)
                if finnhub_data and self._has_meaningful_data(finnhub_data):
                    supplemental_data = finnhub_data
            
            # Merge if we have supplemental data, otherwise use AV alone
            if supplemental_data:
                return self._merge_news_sources(symbols, av_data, supplemental_data)
            else:
                self.log("Using Alpha Vantage only (supplemental sources unavailable)", "INFO")
                return av_data
        
        # Fallbacks if no Alpha Vantage
        if has_newsapi:
            return await self._fetch_newsapi_data(symbols)
        if has_finnhub:
            return await self._fetch_finnhub_news(symbols)
        
        # Simulated only if no keys provided
        self.log("No news API key found, using simulated data", "WARNING")
        return {"source": "none", "symbols": {s: {"score": None, "label": "INSUFFICIENT_DATA", "count": 0, "articles_analyzed": 0} for s in symbols}, "timestamp": datetime.now().isoformat()}
    
    def _has_meaningful_data(self, data: Dict[str, Any]) -> bool:
        """Check if news data has meaningful content (not just empty due to quota/errors)"""
        if not data or not isinstance(data, dict):
            return False
        symbols_data = data.get("symbols", {})
        if not symbols_data:
            return False
        # Consider meaningful if at least one symbol has articles or non-zero score
        for symbol_info in symbols_data.values():
            if isinstance(symbol_info, dict):
                if symbol_info.get("count", 0) > 0 or abs(symbol_info.get("score", 0)) > 0.05:
                    return True
        return False
    
    async def _fetch_alpha_vantage_news(self, symbols: List[str]) -> Dict[str, Any]:
        """Fetch news sentiment from Alpha Vantage API"""
        api_key = self.api_keys.get("alpha_vantage")
        
        symbol_sentiments = {}
        async with aiohttp.ClientSession() as session:
            for symbol in symbols:
                try:
                    url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={symbol}&apikey={api_key}"
                    async with session.get(url) as response:
                        if response.status == 200:
                            data = await response.json()
                            sentiment_score = self._parse_alpha_vantage_sentiment(data)
                            symbol_sentiments[symbol] = sentiment_score
                        else:
                            self.log(f"Failed to fetch news for {symbol}: {response.status}", "WARNING")
                            symbol_sentiments[symbol] = {"score": None, "label": "INSUFFICIENT_DATA"}
                except Exception as e:
                    self.log(f"Error fetching news for {symbol}: {str(e)}", "ERROR")
                    symbol_sentiments[symbol] = {"score": None, "label": "INSUFFICIENT_DATA"}
                    
                # Rate limiting
                await asyncio.sleep(0.5)
        
        return {
            "source": "alpha_vantage",
            "symbols": symbol_sentiments,
            "timestamp": datetime.now().isoformat()
        }
    
    async def _fetch_finnhub_news(self, symbols: List[str]) -> Dict[str, Any]:
        """Fetch news from Finnhub (free tier: 60 calls/min)"""
        api_key = self.api_keys.get("finnhub")
        
        if not api_key:
            return {"source": "finnhub", "symbols": {}, "timestamp": datetime.now().isoformat()}
        
        symbol_sentiments = {}
        async with aiohttp.ClientSession() as session:
            for symbol in symbols:
                try:
                    # Finnhub company news endpoint (free tier)
                    from_date = (datetime.now() - timedelta(days=self.lookback_days)).strftime('%Y-%m-%d')
                    to_date = datetime.now().strftime('%Y-%m-%d')
                    url = f"https://finnhub.io/api/v1/company-news?symbol={symbol}&from={from_date}&to={to_date}&token={api_key}"
                    
                    async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                        if response.status == 200:
                            articles = await response.json()
                            if articles and isinstance(articles, list):
                                # Convert Finnhub format to our internal format
                                formatted_articles = [
                                    {"title": a.get("headline", ""), "description": a.get("summary", "")}
                                    for a in articles[:50]
                                ]
                                sentiment = self._analyze_news_articles(formatted_articles)
                                symbol_sentiments[symbol] = sentiment
                            else:
                                symbol_sentiments[symbol] = {"score": 0, "label": "neutral", "count": 0, "method": "none"}
                        else:
                            self.log(f"Finnhub error for {symbol}: HTTP {response.status}", "WARNING")
                            symbol_sentiments[symbol] = {"score": 0, "label": "neutral", "count": 0, "method": "none"}
                except Exception as e:
                    self.log(f"Error fetching from Finnhub for {symbol}: {str(e)}", "WARNING")
                    symbol_sentiments[symbol] = {"score": 0, "label": "neutral", "count": 0, "method": "none"}
                
                # Respect rate limits (60/min = ~1/sec)
                await asyncio.sleep(1.1)
        
        return {
            "source": "finnhub",
            "symbols": symbol_sentiments,
            "timestamp": datetime.now().isoformat()
        }
    
    async def _fetch_newsapi_data(self, symbols: List[str]) -> Dict[str, Any]:
        """Fetch news from NewsAPI and analyze sentiment with NLP"""
        api_key = self.api_keys.get("newsapi")
        
        if not api_key:
            self.log("NewsAPI key not configured, skipping NewsAPI", "WARNING")
            return {"source": "newsapi", "symbols": {}, "timestamp": datetime.now().isoformat()}
        
        # Daily cache file to avoid repeated calls within the same day
        date_str = datetime.now().strftime('%Y%m%d')
        cache_path = f"analysis_cache/newsapi_cache_{date_str}.json"
        news_cache: Dict[str, Any] = {}
        # Daily lock to skip NewsAPI once cap/429 is observed
        lock_filename = f"newsapi_lock_{date_str}.txt"
        try:
            # Try loading existing cache
            import os, json
            full_cache_path = os.path.join(os.path.dirname(__file__), "..", "analysis_cache", f"newsapi_cache_{date_str}.json")
            full_cache_path = os.path.normpath(full_cache_path)
            if os.path.exists(full_cache_path):
                with open(full_cache_path, 'r', encoding='utf-8') as f:
                    news_cache = json.load(f)
            # Check lock file
            lock_path = os.path.join(os.path.dirname(__file__), "..", "analysis_cache", lock_filename)
            lock_path = os.path.normpath(lock_path)
            if os.path.exists(lock_path):
                self.log("NewsAPI daily cap previously hit; skipping NewsAPI calls today", "INFO")
                return {"source": "newsapi", "symbols": {}, "timestamp": datetime.now().isoformat(), "adherence": "daily_cap_lock"}
        except Exception as e:
            self.log(f"Failed to load NewsAPI cache: {e}", "WARNING")

        def save_cache():
            try:
                import os, json
                full_cache_path_local = os.path.join(os.path.dirname(__file__), "..", "analysis_cache", f"newsapi_cache_{date_str}.json")
                full_cache_path_local = os.path.normpath(full_cache_path_local)
                os.makedirs(os.path.dirname(full_cache_path_local), exist_ok=True)
                with open(full_cache_path_local, 'w', encoding='utf-8') as f:
                    json.dump(news_cache, f, ensure_ascii=False, indent=2)
            except Exception as e:
                self.log(f"Failed to save NewsAPI cache: {e}", "WARNING")

        symbol_sentiments = {}
        rate_limit_hit = False
        calls_made = 0

        # Determine strategy based on mode
        mode = self.newsapi_mode
        if mode == "disabled":
            self.log("NewsAPI disabled by configuration", "INFO")
            return {"source": "newsapi", "symbols": {}, "timestamp": datetime.now().isoformat()}

        # In free mode, limit per-run requests and prioritize a subset of symbols
        symbols_to_query = symbols
        if mode == "free":
            symbols_to_query = symbols[:max(1, min(len(symbols), self.newsapi_max_calls))]
            if len(symbols_to_query) < len(symbols):
                self.log(f"NewsAPI free mode: limiting symbol queries to {len(symbols_to_query)}/{len(symbols)}", "INFO")

        async with aiohttp.ClientSession() as session:
            skip_logged = False
            for symbol in symbols_to_query:
                try:
                    # Skip if we've already hit rate limit
                    if rate_limit_hit:
                        # Avoid spamming logs; one message is enough
                        if not skip_logged:
                            self.log("Skipping remaining symbols due to NewsAPI rate limit", "INFO")
                            skip_logged = True
                        symbol_sentiments[symbol] = {"score": 0, "label": "neutral", "count": 0, "method": "none", "articles_analyzed": 0}
                        continue
                    if mode == "free" and calls_made >= self.newsapi_max_calls:
                        self.log("NewsAPI free mode cap reached; switching to headlines fallback", "INFO")
                        rate_limit_hit = True
                        symbol_sentiments[symbol] = {"score": 0, "label": "neutral", "count": 0, "method": "none", "articles_analyzed": 0}
                        continue
                    
                    # Calculate date range
                    end_date = datetime.now()
                    start_date = end_date - timedelta(days=self.lookback_days)
                    
                    # Use cache first in the same day
                    cached = news_cache.get(symbol)
                    if cached and cached.get("from") == start_date.strftime('%Y-%m-%d'):
                        articles = cached.get("articles", [])
                        self.log(f"NewsAPI cache hit: {len(articles)} articles for {symbol}", "DEBUG")
                        sentiment = self._analyze_news_articles(articles) if articles else {"score": 0, "label": "neutral", "count": 0, "method": "none", "articles_analyzed": 0}
                        symbol_sentiments[symbol] = sentiment
                        continue

                    url = (
                        f"https://newsapi.org/v2/everything?"
                        f"q={symbol}&"
                        f"from={start_date.strftime('%Y-%m-%d')}&"
                        f"to={end_date.strftime('%Y-%m-%d')}&"
                        f"sortBy=relevancy&"
                        f"language=en&"
                        f"pageSize={self.newsapi_page_size}&"
                        f"apiKey={api_key}"
                    )
                    
                    async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                        if response.status == 200:
                            data = await response.json()
                            articles = data.get("articles", [])
                            self.log(f"NewsAPI returned {len(articles)} articles for {symbol}", "DEBUG")
                            # Update calls made and cache
                            calls_made += 1
                            news_cache[symbol] = {
                                "from": start_date.strftime('%Y-%m-%d'),
                                "to": end_date.strftime('%Y-%m-%d'),
                                "count": len(articles),
                                "articles": articles,
                            }
                            save_cache()
                            
                            # Prioritize NLP transformer analysis
                            if articles:
                                sentiment = self._analyze_news_articles(articles)
                                self.log(f"{symbol}: NLP analysis - method={sentiment.get('method')}, articles_analyzed={sentiment.get('articles_analyzed', 0)}, score={sentiment.get('score', 0):.3f}", "DEBUG")
                                
                                # Only use OpenAI if NLP sentiment is weak/zero (to get deeper reasoning)
                                if sentiment.get("score", 0) == 0 and self.openai_client:
                                    self.log(f"NLP inconclusive for {symbol}, trying OpenAI analysis", "INFO")
                                    ai_sentiment = await self._analyze_news_with_ai(articles)
                                    if ai_sentiment.get("score") != 0:
                                        sentiment = ai_sentiment
                            else:
                                sentiment = {"score": 0, "label": "neutral", "count": 0, "method": "none", "articles_analyzed": 0}
                                
                            symbol_sentiments[symbol] = sentiment
                        elif response.status == 429:
                            # Rate limit hit - log and skip remaining
                            self.log(f"NewsAPI rate limit (429) - free tier exhausted", "WARNING")
                            rate_limit_hit = True
                            # Create lock file to skip future runs today
                            try:
                                import os
                                lock_path_write = os.path.join(os.path.dirname(__file__), "..", "analysis_cache", lock_filename)
                                lock_path_write = os.path.normpath(lock_path_write)
                                os.makedirs(os.path.dirname(lock_path_write), exist_ok=True)
                                with open(lock_path_write, 'w', encoding='utf-8') as f:
                                    f.write(f"429 hit at {datetime.now().isoformat()}\n")
                            except Exception as e:
                                self.log(f"Failed to write NewsAPI lock file: {e}", "WARNING")
                            symbol_sentiments[symbol] = {"score": 0, "label": "neutral", "count": 0, "method": "none", "articles_analyzed": 0}
                        else:
                            self.log(f"NewsAPI error for {symbol}: HTTP {response.status}", "WARNING")
                            symbol_sentiments[symbol] = {"score": 0, "label": "neutral", "count": 0, "method": "none", "articles_analyzed": 0}
                except Exception as e:
                    self.log(f"Error fetching from NewsAPI for {symbol}: {str(e)}", "ERROR")
                    symbol_sentiments[symbol] = {"score": 0, "label": "neutral", "count": 0, "method": "none", "articles_analyzed": 0}
                    
                # Add delay to respect rate limits
                await asyncio.sleep(1.0)
        
        # If we hit rate limit or cap, try one batch of top-headlines as fallback
        if rate_limit_hit and mode in ("free", "everything"):
            try:
                self.log("Attempting top-headlines batch fallback", "INFO")
                batch_sentiment = await self._fetch_newsapi_top_headlines(api_key)
                if batch_sentiment:
                    # Apply aggregate sentiment to remaining symbols not analyzed
                    for symbol in symbols:
                        if symbol not in symbol_sentiments:
                            symbol_sentiments[symbol] = {
                                "score": batch_sentiment["score"],
                                "label": batch_sentiment["label"],
                                "count": batch_sentiment.get("count", 0),
                                "articles_analyzed": batch_sentiment.get("articles_analyzed", 0),
                                "method": "top-headlines"
                            }
            except Exception as e:
                self.log(f"Top-headlines fallback failed: {e}", "WARNING")

        self.log(f"NewsAPI analysis complete: {len([s for s in symbol_sentiments.values() if s.get('method') == 'nlp'])} symbols with NLP analysis", "INFO")
        
        return {
            "source": "newsapi",
            "symbols": symbol_sentiments,
            "timestamp": datetime.now().isoformat()
        }

    async def _fetch_newsapi_top_headlines(self, api_key: str) -> Dict[str, Any]:
        """Fetch a single batch of top-headlines (business, en) and compute aggregate sentiment.
        This adheres to free tier by minimizing requests and still providing market context.
        """
        async with aiohttp.ClientSession() as session:
            url = (
                f"https://newsapi.org/v2/top-headlines?"
                f"category=business&"
                f"language=en&"
                f"pageSize=100&"
                f"apiKey={api_key}"
            )
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                if response.status == 200:
                    data = await response.json()
                    articles = data.get("articles", [])
                    self.log(f"Top-headlines returned {len(articles)} articles", "DEBUG")
                    sentiment = self._analyze_news_articles(articles) if articles else {"score": 0, "label": "neutral", "count": 0, "method": "none", "articles_analyzed": 0}
                    return sentiment
                elif response.status == 429:
                    self.log("Top-headlines rate limit hit (429)", "WARNING")
                    return {"score": 0, "label": "neutral", "count": 0, "method": "none", "articles_analyzed": 0}
                else:
                    self.log(f"Top-headlines error: HTTP {response.status}", "WARNING")
                    return {"score": 0, "label": "neutral", "count": 0, "method": "none", "articles_analyzed": 0}
    
    def _analyze_news_articles(self, articles: List[Dict]) -> Dict[str, Any]:
        """Analyze sentiment from news articles using NLP transformers"""
        if not articles:
            return {"score": 0, "label": "neutral", "count": 0, "method": "none"}

        # Always use NLP if available (transformer model is more accurate than keywords)
        if NLP_AVAILABLE and sentiment_analyzer:
            self.log(f"Using NLP transformer for {len(articles)} articles", "DEBUG")
            nlp_result = self._analyze_with_nlp(articles)
            self.log(f"NLP analysis complete: {nlp_result.get('articles_analyzed', 0)} articles analyzed, score {nlp_result.get('score', 0):.3f}", "INFO")
            return nlp_result
        else:
            if not self._nlp_warning_logged:
                self.log("NLP not available, transformer model not loaded - using keyword fallback", "WARNING")
                self._nlp_warning_logged = True
            # Only use keywords as absolute fallback when NLP is unavailable
            return self._analyze_with_keywords(articles)
    
    def _analyze_with_nlp(self, articles: List[Dict]) -> Dict[str, Any]:
        """Use transformer-based NLP for sentiment analysis on up to 50 articles"""
        try:
            if not sentiment_analyzer:
                raise RuntimeError("Sentiment analyzer not initialized")
                
            sentiments = []
            articles_processed = 0
            
            for article in articles[:50]:  # Analyze up to 50 articles for deeper sentiment
                try:
                    # Combine title and description for better context
                    title = article.get("title") or ""
                    description = article.get("description") or ""
                    text = f"{title} {description}"
                    
                    if not text.strip():
                        continue
                    
                    # Truncate to 512 tokens (transformer limit)
                    text = text[:512]
                    
                    # Use NLP transformer for analysis
                    result = sentiment_analyzer(text, truncation=True)
                    
                    # Parse transformer output
                    label = result[0]["label"].lower()
                    score = result[0]["score"]  # Confidence score (0-1)
                    
                    # Convert to -1 to 1 scale based on label and confidence
                    if "negative" in label:
                        sentiment_value = -score  # Negative: -1 to 0
                    elif "positive" in label:
                        sentiment_value = score   # Positive: 0 to 1
                    else:
                        sentiment_value = 0      # Neutral: 0
                    
                    sentiments.append(sentiment_value)
                    articles_processed += 1
                    
                except Exception as e:
                    self.log(f"Error analyzing article with NLP: {str(e)}", "WARNING")
                    continue
            
            if not sentiments:
                self.log("NLP analysis produced no sentiment scores, returning INSUFFICIENT_DATA", "WARNING")
                return {"score": None, "label": "INSUFFICIENT_DATA", "count": len(articles), "articles_analyzed": 0, "method": "nlp"}
            
            # Average sentiment score across all analyzed articles
            avg_score = sum(sentiments) / len(sentiments)
            
            # Categorize with thresholds
            if avg_score > 0.15:
                label = "bullish"
            elif avg_score < -0.15:
                label = "bearish"
            else:
                label = "neutral"
            
            self.log(f"NLP analysis: {articles_processed}/{len(articles)} articles, avg score {avg_score:.3f}, label {label}", "INFO")
            
            return {
                "score": round(avg_score, 3),
                "label": label,
                "count": len(articles),
                "articles_analyzed": articles_processed,
                "method": "nlp"
            }
            
        except Exception as e:
            self.log(f"NLP sentiment analysis failed: {str(e)}, falling back to keywords", "ERROR")
            return self._analyze_with_keywords(articles)
    
    def _analyze_with_keywords(self, articles: List[Dict]) -> Dict[str, Any]:
        """Fallback keyword-based sentiment analysis when NLP unavailable"""
        # Keyword matching (when NLP not available)
        positive_keywords = [
            "surge", "soar", "rally", "gain", "growth", "profit", "beat", 
            "strong", "upgrade", "bullish", "record", "high", "outperform",
            "boom", "breakout", "accelerate", "momentum", "success", "win",
            "excellent", "impressive", "positive", "fantastic", "tremendous"
        ]
        negative_keywords = [
            "plunge", "drop", "fall", "loss", "miss", "weak", "downgrade", 
            "bearish", "low", "underperform", "decline", "cut", "warning",
            "crash", "risk", "collapse", "fail", "threat", "concern",
            "poor", "negative", "disappointing", "disaster", "problem"
        ]
        
        total_score = 0
        for article in articles[:50]:  # Analyze up to 50 articles for deeper sentiment
            title = ((article.get("title") or "") + " " + (article.get("description") or "")).lower()
            
            pos_count = sum(1 for kw in positive_keywords if kw in title)
            neg_count = sum(1 for kw in negative_keywords if kw in title)
            
            if pos_count > neg_count:
                total_score += 1
            elif neg_count > pos_count:
                total_score -= 1
        
        # Normalize score to -1 to 1
        article_count = min(len(articles), 20)
        normalized_score = total_score / article_count if article_count > 0 else 0
        
        # Categorize sentiment
        if normalized_score > 0.2:
            label = "bullish"
        elif normalized_score < -0.2:
            label = "bearish"
        else:
            label = "neutral"
        
        return {
            "score": normalized_score,
            "label": label,
            "count": len(articles),
            "method": "keywords",
            "articles_analyzed": min(len(articles), 50),
        }
    
    def _parse_alpha_vantage_sentiment(self, data: Dict) -> Dict[str, Any]:
        """Parse Alpha Vantage sentiment response with improved thresholds"""
        feed = data.get("feed", [])
        
        if not feed:
            return {"score": None, "label": "INSUFFICIENT_DATA", "count": 0, "articles_analyzed": 0}
        
        total_sentiment = 0
        count = 0
        
        for item in feed:
            ticker_sentiment = item.get("ticker_sentiment", [])
            for ts in ticker_sentiment:
                score = float(ts.get("ticker_sentiment_score", 0))
                total_sentiment += score
                count += 1
        
        avg_sentiment = total_sentiment / count if count > 0 else 0
        
        # Categorize with lower thresholds for better discrimination
        # Alpha Vantage scores range from -1 to +1
        if avg_sentiment > 0.12:
            label = "bullish"
        elif avg_sentiment < -0.12:
            label = "bearish"
        else:
            label = "neutral"
        
        return {
            "score": avg_sentiment,
            "label": label,
            "count": count
        }
    
    
    
    def _aggregate_sentiment(
        self,
        symbols: List[str],
        news: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Aggregate sentiment from news source and track data quality"""
        
        aggregated = {}
        
        for symbol in symbols:
            # Get sentiment from news source
            news_sentiment = news.get("symbols", {}).get(symbol, {})
            
            # Use news score directly
            score = news_sentiment.get("score")
            label = news_sentiment.get("label", "INSUFFICIENT_DATA")
            articles_analyzed = news_sentiment.get("articles_analyzed", 0)
            
            # Handle INSUFFICIENT_DATA case
            if label == "INSUFFICIENT_DATA" or score is None:
                aggregated[symbol] = {
                    "overall_score": None,
                    "sentiment": "INSUFFICIENT_DATA",
                    "confidence": 0.0,
                    "news_score": None,
                    "method": news_sentiment.get("method", "unknown"),
                    "articles_analyzed": articles_analyzed,
                    "data_quality": "no_data"
                }
            else:
                # Determine overall label based on score
                if score is None:
                    label = "INSUFFICIENT_DATA"
                    confidence = 0.0
                elif score > 0.12:
                    label = "bullish"
                    confidence = min(abs(score) / 0.5, 1.0)
                elif score < -0.12:
                    label = "bearish"
                    confidence = min(abs(score) / 0.5, 1.0)
                else:
                    label = "neutral"
                    confidence = 1.0 - min(abs(score) / 0.12, 1.0)
                
                aggregated[symbol] = {
                    "overall_score": round(score, 3),
                    "sentiment": label,
                    "confidence": round(confidence, 2),
                    "news_score": news_sentiment.get("score", 0),
                    "method": news_sentiment.get("method", "unknown"),
                    "articles_analyzed": articles_analyzed,
                    "data_quality": "sufficient" if articles_analyzed > 0 else "partial"
                }
        
        return aggregated

    def _merge_news_sources(
        self,
        symbols: List[str],
        av_data: Dict[str, Any],
        supplemental_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Merge Alpha Vantage and supplemental source sentiment with dynamic weighting.
        Weights adapt based on actual data availability to avoid diluting good data.
        """

        merged = {}
        av_symbols = av_data.get("symbols", {}) if av_data else {}
        supp_symbols = supplemental_data.get("symbols", {}) if supplemental_data else {}
        supp_source = supplemental_data.get("source", "supplemental") if supplemental_data else "unknown"

        for symbol in symbols:
            av = av_symbols.get(symbol, {"score": 0, "label": "neutral", "count": 0})
            supp = supp_symbols.get(symbol, {"score": 0, "label": "neutral", "count": 0})

            # Dynamic weighting based on data availability
            av_score = av.get("score")
            supp_score = supp.get("score")
            av_has_data = av.get("count", 0) > 0 or (av_score is not None and abs(av_score) > 0.01)
            supp_has_data = supp.get("count", 0) > 0 or (supp_score is not None and abs(supp_score) > 0.01)
            
            if av_has_data and supp_has_data:
                # Both have data: use weighted blend (AV 60%, supplemental 40%)
                score = av.get("score", 0) * 0.6 + supp.get("score", 0) * 0.4
                weight_method = "blended"
            elif av_has_data:
                # Only AV has data: use it at 100%
                score = av.get("score", 0)
                weight_method = "av_only"
            elif supp_has_data:
                # Only supplemental has data: use it at 100%
                score = supp.get("score", 0)
                weight_method = "supp_only"
            else:
                # Neither has data: neutral
                score = 0
                weight_method = "none"

            # Derive label with same thresholds used elsewhere
            if score > 0.12:
                label = "bullish"
            elif score < -0.12:
                label = "bearish"
            else:
                label = "neutral"

            # Track which methods were used
            methods_used = []
            if av.get("count", 0) > 0:
                methods_used.append("alpha_vantage")
            if supp.get("method") == "nlp":
                methods_used.append("nlp")
            elif supp.get("count", 0) > 0:
                methods_used.append(supp_source.split("_")[-1] if "_" in supp_source else supp_source)

            merged[symbol] = {
                "score": score,
                "label": label,
                "count": (av.get("count", 0) or 0) + (supp.get("count", 0) or 0),
                "method": ", ".join(methods_used) if methods_used else "unknown",
                "articles_analyzed": supp.get("articles_analyzed", 0),
                "weight_method": weight_method,
                "components": {
                    "alpha_vantage": av,
                    supp_source: supp,
                },
            }

        return {
            "source": f"merged_av_{supp_source}",
            "symbols": merged,
            "timestamp": datetime.now().isoformat(),
        }
    
    def _generate_insights(self, sentiment_data: Dict[str, Any]) -> List[str]:
        """Generate actionable insights from sentiment analysis"""
        insights = []
        
        # Identify most bullish symbols
        bullish_symbols = [
            (symbol, data["overall_score"])
            for symbol, data in sentiment_data.items()
            if data["sentiment"] == "bullish"
        ]
        bullish_symbols.sort(key=lambda x: x[1], reverse=True)
        
        if bullish_symbols:
            top_bullish = bullish_symbols[0]
            insights.append(
                f"{top_bullish[0]} shows strongest positive sentiment "
                f"(score: {top_bullish[1]:.2f})"
            )
        
        # Identify most bearish symbols
        bearish_symbols = [
            (symbol, data["overall_score"])
            for symbol, data in sentiment_data.items()
            if data["sentiment"] == "bearish"
        ]
        bearish_symbols.sort(key=lambda x: x[1])
        
        if bearish_symbols:
            top_bearish = bearish_symbols[0]
            insights.append(
                f"{top_bearish[0]} shows most negative sentiment "
                f"(score: {top_bearish[1]:.2f})"
            )
        
        # Count sentiment distribution
        bullish_count = sum(1 for d in sentiment_data.values() if d["sentiment"] == "bullish")
        bearish_count = sum(1 for d in sentiment_data.values() if d["sentiment"] == "bearish")
        neutral_count = sum(1 for d in sentiment_data.values() if d["sentiment"] == "neutral")
        
        insights.append(
            f"Sentiment distribution: {bullish_count} bullish, "
            f"{neutral_count} neutral, {bearish_count} bearish"
        )
        
        return insights
    
    def _generate_recommendations(self, sentiment_data: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on sentiment"""
        recommendations = []
        
        for symbol, data in sentiment_data.items():
            score = data["overall_score"]
            sentiment = data["sentiment"]
            confidence = data.get("confidence", 0)
            news_score = data.get("news_score")
            data_quality = data.get("data_quality", "unknown")
            articles_analyzed = data.get("articles_analyzed", 0)
            
            # Handle INSUFFICIENT_DATA case
            if sentiment == "INSUFFICIENT_DATA":
                recommendations.append(
                    f"Monitor {symbol}: insufficient news data (0 articles analyzed); unable to determine sentiment"
                )
            # More lenient thresholds to give actionable recommendations
            elif sentiment == "bullish":
                if confidence > 0.5:  # Slightly higher gate to reduce noise
                    recommendations.append(
                        f"BUY bias for {symbol}: news sentiment bullish (score {score:.2f}, news {news_score:.2f}); consider adding/averaging up"
                    )
                else:
                    recommendations.append(
                        f"Hold {symbol}: mildly positive news sentiment (score {score:.2f}), monitor for confirmation"
                    )
            elif sentiment == "bearish":
                if confidence > 0.5:  # Slightly higher gate to reduce noise
                    recommendations.append(
                        f"SELL bias for {symbol}: news sentiment bearish (score {score:.2f}, news {news_score:.2f}); consider trimming"
                    )
                else:
                    recommendations.append(
                        f"Hold {symbol}: slightly negative news sentiment (score {score:.2f}), monitor for deterioration"
                    )
            elif sentiment == "neutral":
                recommendations.append(
                    f"Hold {symbol}: neutral news sentiment (score {score:.2f}), watch for catalysts"
                )
        
        return recommendations
    
    def _create_summary(self, sentiment_data: Dict[str, Any]) -> str:
        """Create summary of sentiment analysis"""
        total_symbols = len(sentiment_data)
        
        # Calculate average score excluding INSUFFICIENT_DATA entries
        scores = [d["overall_score"] for d in sentiment_data.values() if d["overall_score"] is not None]
        avg_score = sum(scores) / len(scores) if scores else 0
        
        bullish_count = sum(1 for d in sentiment_data.values() if d["sentiment"] == "bullish")
        bearish_count = sum(1 for d in sentiment_data.values() if d["sentiment"] == "bearish")
        insufficient_count = sum(1 for d in sentiment_data.values() if d["sentiment"] == "INSUFFICIENT_DATA")
        
        if avg_score > 0.2:
            market_mood = "positive"
        elif avg_score < -0.2:
            market_mood = "negative"
        else:
            market_mood = "neutral"
        
        summary = (
            f"Market sentiment analysis across {total_symbols} portfolio holdings shows "
            f"a {market_mood} overall mood (score: {avg_score:.2f}). "
            f"{bullish_count} positions have bullish sentiment, {bearish_count} are bearish."
        )
        
        if insufficient_count > 0:
            summary += f" Note: {insufficient_count} position(s) have insufficient news data."
        
        return summary
    
    async def _analyze_news_with_ai(self, articles: List[Dict]) -> Dict[str, Any]:
        """Use AI to analyze sentiment from news articles"""
        if not articles:
            return {"score": None, "label": "INSUFFICIENT_DATA", "count": 0, "articles_analyzed": 0}
        
        # Prepare articles summary for AI
        articles_text = "\n\n".join([
            f"Title: {article.get('title', 'N/A')}\nDescription: {article.get('description', 'N/A')}"
            for article in articles[:10]  # Top 10 articles
        ])
        
        prompt = f"""Analyze the sentiment of these financial news articles and provide a sentiment score.

Articles:
{articles_text}

Based on these articles, provide:
1. A sentiment score from -1.0 (very bearish) to +1.0 (very bullish)
2. A label: "bullish", "bearish", or "neutral"
3. Brief reasoning

Respond in this exact format:
SCORE: [number]
LABEL: [bullish/bearish/neutral]
REASONING: [brief explanation]"""

        try:
            response = await self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a financial analyst expert at analyzing market sentiment from news."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=200
            )
            
            content = response.choices[0].message.content
            
            # Parse the response
            score = 0.0
            label = "neutral"
            
            for line in content.split('\n'):
                if line.startswith('SCORE:'):
                    try:
                        score = float(line.split(':')[1].strip())
                    except:
                        pass
                elif line.startswith('LABEL:'):
                    label = line.split(':')[1].strip().lower()
            
            return {
                "score": score,
                "label": label,
                "count": len(articles),
                "ai_analysis": content
            }
            
        except Exception as e:
            self.log(f"AI sentiment analysis failed: {str(e)}", "ERROR")
            return {"score": None, "label": "INSUFFICIENT_DATA", "count": len(articles), "articles_analyzed": 0}
    
    async def _generate_sentiment_ai_reasoning(self, results: Dict[str, Any], symbols: List[str]) -> str:
        """Generate AI reasoning for sentiment analysis results with data quality verification"""
        
        # Extract key metrics
        overall = results.get("overall_sentiment", {})
        bullish = [s for s, d in overall.items() if d.get("sentiment") == "bullish"]
        bearish = [s for s, d in overall.items() if d.get("sentiment") == "bearish"]
        insufficient_data = [s for s, d in overall.items() if d.get("sentiment") == "INSUFFICIENT_DATA"]
        
        # Calculate data quality
        analyzed_count = sum(1 for d in overall.values() if d.get("articles_analyzed", 0) > 0)
        data_quality_pct = (analyzed_count / len(symbols) * 100) if symbols else 0
        
        prompt = f"""Analyze this portfolio sentiment data and provide concise investment insights.

Portfolio: {len(symbols)} positions
Data Quality: {data_quality_pct:.0f}% of positions have article data
Bullish signals: {', '.join(bullish) if bullish else 'none'}
Bearish signals: {', '.join(bearish) if bearish else 'none'}
Insufficient Data: {', '.join(insufficient_data) if insufficient_data else 'none'}

Top sentiment scores:
{json.dumps({s: {'score': d.get('overall_score'), 'sentiment': d.get('sentiment'), 'articles': d.get('articles_analyzed', 0)} for s, d in list(overall.items())[:5]}, indent=2)}

Provide:
1. What the sentiment tells us about market perception (noting any data gaps)
2. Which positions warrant attention and why (flag positions with insufficient data)
3. Key risks or opportunities
4. Actionable takeaway

Be concise (3-4 sentences max)."""
        
        return await self.generate_ai_reasoning(results, prompt)



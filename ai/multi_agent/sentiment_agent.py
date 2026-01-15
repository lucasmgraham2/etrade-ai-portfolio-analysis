"""
Sentiment Agent
Analyzes market sentiment from news, social media, and forums
"""

from typing import Dict, Any, List
from datetime import datetime, timedelta
import asyncio
import aiohttp
import json
import os
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

    @staticmethod
    def _insufficient(reason: str = "none") -> Dict[str, Any]:
        """Standardized shape for missing or inadequate sentiment data."""
        return {
            "score": None,
            "label": "INSUFFICIENT_DATA",
            "count": 0,
            "articles_analyzed": 0,
            "method": reason
        }
        
    async def analyze(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform sentiment analysis
        
        Args:
            context: Contains portfolio data with symbols
            
        Returns:
            Sentiment analysis results
        """
        try:
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
        except Exception as e:
            import traceback
            self.log(f"Sentiment analysis error: {str(e)}", "ERROR")
            self.log(f"Traceback: {traceback.format_exc()}", "DEBUG")
            # Return empty results structure with error
            return {
                "error": f"Sentiment analysis failed: {str(e)}",
                "overall_sentiment": {},
                "summary": "Sentiment analysis unavailable"
            }
    
    
    async def _analyze_news_sentiment(self, symbols: List[str]) -> Dict[str, Any]:
        """
        Analyze news sentiment for portfolio symbols using NewsAPI
        
        Note: Alpha Vantage NEWS_SENTIMENT endpoint is deprecated and no longer used.
        Alpha Vantage is still used for sector performance data in the sector agent.
        """
        self.log("Fetching news sentiment...")
        
        has_newsapi = "newsapi" in self.api_keys
        has_finnhub = "finnhub" in self.api_keys
        
        # Use NewsAPI as primary source
        if has_newsapi:
            return await self._fetch_newsapi_data(symbols)
        
        # Finnhub as fallback
        if has_finnhub:
            return await self._fetch_finnhub_news(symbols)

        # No news API available
        self.log("No news API key found", "WARNING")
        return {
            "source": "none",
            "symbols": {s: self._insufficient("no_sources") for s in symbols},
            "timestamp": datetime.now().isoformat()
        }
    
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
                score = symbol_info.get("score", 0) or 0  # Handle None scores
                if symbol_info.get("count", 0) > 0 or abs(score) > 0.05:
                    return True
        return False
    
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
                                symbol_sentiments[symbol] = self._insufficient("finnhub_empty")
                        else:
                            self.log(f"Finnhub error for {symbol}: HTTP {response.status}", "WARNING")
                            symbol_sentiments[symbol] = self._insufficient("finnhub_error")
                except Exception as e:
                    self.log(f"Error fetching from Finnhub for {symbol}: {str(e)}", "WARNING")
                    symbol_sentiments[symbol] = self._insufficient("finnhub_error")
                
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

        # Daily cache and rate-limit lock
        date_str = datetime.now().strftime('%Y%m%d')
        news_cache: Dict[str, Any] = {}
        lock_filename = f"newsapi_lock_{date_str}.txt"

        try:
            full_cache_path = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "analysis_cache", f"newsapi_cache_{date_str}.json"))
            if os.path.exists(full_cache_path):
                with open(full_cache_path, 'r', encoding='utf-8') as f:
                    news_cache = json.load(f)

            lock_path = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "analysis_cache", lock_filename))
            if os.path.exists(lock_path):
                self.log("NewsAPI daily cap previously hit; skipping NewsAPI calls today", "INFO")
                return {"source": "newsapi", "symbols": {}, "timestamp": datetime.now().isoformat(), "adherence": "daily_cap_lock"}
        except Exception as e:
            self.log(f"Failed to load NewsAPI cache: {e}", "WARNING")

        def save_cache():
            try:
                full_cache_path_local = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "analysis_cache", f"newsapi_cache_{date_str}.json"))
                os.makedirs(os.path.dirname(full_cache_path_local), exist_ok=True)
                with open(full_cache_path_local, 'w', encoding='utf-8') as f:
                    json.dump(news_cache, f, ensure_ascii=False, indent=2)
            except Exception as e:
                self.log(f"Failed to save NewsAPI cache: {e}", "WARNING")

        symbol_sentiments: Dict[str, Any] = {}
        rate_limit_hit = False
        calls_made = 0

        mode = self.newsapi_mode
        if mode == "disabled":
            self.log("NewsAPI disabled by configuration", "INFO")
            return {"source": "newsapi", "symbols": {}, "timestamp": datetime.now().isoformat()}

        symbols_to_query = symbols
        if mode == "free":
            symbols_to_query = symbols[:max(1, min(len(symbols), self.newsapi_max_calls))]
            if len(symbols_to_query) < len(symbols):
                self.log(f"NewsAPI free mode: limiting symbol queries to {len(symbols_to_query)}/{len(symbols)}", "INFO")

        async with aiohttp.ClientSession() as session:
            skip_logged = False
            for symbol in symbols_to_query:
                try:
                    if rate_limit_hit:
                        if not skip_logged:
                            self.log("Skipping remaining symbols due to NewsAPI rate limit", "INFO")
                            skip_logged = True
                        symbol_sentiments[symbol] = self._insufficient("newsapi_rate_limit")
                        continue

                    if mode == "free" and calls_made >= self.newsapi_max_calls:
                        self.log("NewsAPI free mode cap reached; switching to headlines fallback", "INFO")
                        rate_limit_hit = True
                        symbol_sentiments[symbol] = self._insufficient("newsapi_cap")
                        continue

                    end_date = datetime.now()
                    start_date = end_date - timedelta(days=self.lookback_days)

                    cached = news_cache.get(symbol)
                    if cached and cached.get("from") == start_date.strftime('%Y-%m-%d'):
                        articles = cached.get("articles", [])
                        self.log(f"NewsAPI cache hit: {len(articles)} articles for {symbol}", "DEBUG")
                        sentiment = self._analyze_news_articles(articles) if articles else self._insufficient("newsapi_cache_empty")
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

                            calls_made += 1
                            news_cache[symbol] = {
                                "from": start_date.strftime('%Y-%m-%d'),
                                "to": end_date.strftime('%Y-%m-%d'),
                                "count": len(articles),
                                "articles": articles,
                            }
                            save_cache()

                            if articles:
                                sentiment = self._analyze_news_articles(articles)
                                score_val = sentiment.get("score", 0) if sentiment.get("score") is not None else 0
                                self.log(f"{symbol}: NLP analysis - method={sentiment.get('method')}, articles_analyzed={sentiment.get('articles_analyzed', 0)}, score={score_val:.3f}", "DEBUG")

                                if sentiment.get("score") in (0, None) and self.openai_client:
                                    self.log(f"NLP inconclusive for {symbol}, trying OpenAI analysis", "INFO")
                                    ai_sentiment = await self._analyze_news_with_ai(articles)
                                    if ai_sentiment.get("score") not in (None, 0):
                                        sentiment = ai_sentiment
                            else:
                                sentiment = self._insufficient("newsapi_no_articles")

                            symbol_sentiments[symbol] = sentiment
                        elif response.status == 429:
                            self.log(f"NewsAPI rate limit (429) - free tier exhausted", "WARNING")
                            rate_limit_hit = True
                            try:
                                lock_path_write = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "analysis_cache", lock_filename))
                                os.makedirs(os.path.dirname(lock_path_write), exist_ok=True)
                                with open(lock_path_write, 'w', encoding='utf-8') as f:
                                    f.write(f"429 hit at {datetime.now().isoformat()}\n")
                            except Exception as e:
                                self.log(f"Failed to write NewsAPI lock file: {e}", "WARNING")
                            symbol_sentiments[symbol] = self._insufficient("newsapi_rate_limit")
                        else:
                            self.log(f"NewsAPI error for {symbol}: HTTP {response.status}", "WARNING")
                            symbol_sentiments[symbol] = self._insufficient("newsapi_error")
                except Exception as e:
                    self.log(f"Error fetching from NewsAPI for {symbol}: {str(e)}", "ERROR")
                    symbol_sentiments[symbol] = self._insufficient("newsapi_error")

                await asyncio.sleep(1.0)

        if rate_limit_hit and mode in ("free", "everything"):
            try:
                self.log("Attempting top-headlines batch fallback", "INFO")
                batch_sentiment = await self._fetch_newsapi_top_headlines(api_key)
                if batch_sentiment:
                    for symbol in symbols:
                        if symbol not in symbol_sentiments:
                            symbol_sentiments[symbol] = batch_sentiment if batch_sentiment else self._insufficient("newsapi_top_headlines")
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
                    sentiment = self._analyze_news_articles(articles) if articles else self._insufficient("newsapi_top_headlines_empty")
                    return sentiment
                elif response.status == 429:
                    self.log("Top-headlines rate limit hit (429)", "WARNING")
                    return self._insufficient("newsapi_top_headlines_rate")
                else:
                    self.log(f"Top-headlines error: HTTP {response.status}", "WARNING")
                    return self._insufficient("newsapi_top_headlines_error")
    
    def _analyze_news_articles(self, articles: List[Dict]) -> Dict[str, Any]:
        """Analyze sentiment from news articles using NLP transformers"""
        if not articles:
            return self._insufficient("no_articles")

        if NLP_AVAILABLE and sentiment_analyzer:
            self.log(f"Using NLP transformer for {len(articles)} articles", "DEBUG")
            nlp_result = self._analyze_with_nlp(articles)
            self.log(f"NLP analysis complete: {nlp_result.get('articles_analyzed', 0)} articles analyzed, score {nlp_result.get('score', 0):.3f}", "INFO")
            return nlp_result
        else:
            self.log("NLP transformers not available", "WARNING")
            return {"score": None, "label": "INSUFFICIENT_DATA", "articles_analyzed": len(articles)}
    
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
            self.log(f"NLP sentiment analysis failed: {str(e)}", "ERROR")
            return self._insufficient("nlp_error")
    
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

    def _generate_insights(self, sentiment_data: Dict[str, Any]) -> List[str]:
        """Generate actionable insights from sentiment analysis"""
        insights = []
        
        # Identify most bullish symbols (with valid scores)
        bullish_symbols = [
            (symbol, data["overall_score"])
            for symbol, data in sentiment_data.items()
            if data["sentiment"] == "bullish" and data["overall_score"] is not None
        ]
        bullish_symbols.sort(key=lambda x: x[1], reverse=True)
        
        if bullish_symbols:
            top_bullish = bullish_symbols[0]
            insights.append(
                f"{top_bullish[0]} shows strongest positive sentiment "
                f"(score: {top_bullish[1]:.2f})"
            )
        
        # Identify most bearish symbols (with valid scores)
        bearish_symbols = [
            (symbol, data["overall_score"])
            for symbol, data in sentiment_data.items()
            if data["sentiment"] == "bearish" and data["overall_score"] is not None
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
                    f"Monitor {symbol}: insufficient news data ({articles_analyzed} articles analyzed); unable to determine sentiment"
                )
            # More lenient thresholds to give actionable recommendations
            elif sentiment == "bullish":
                if score is None or confidence <= 0.5:  # Handle None scores
                    recommendations.append(
                        f"Hold {symbol}: mildly positive news sentiment, monitor for confirmation"
                    )
                else:
                    recommendations.append(
                        f"BUY bias for {symbol}: news sentiment bullish (score {score:.2f}, news {news_score:.2f}); consider adding/averaging up"
                    )
            elif sentiment == "bearish":
                if score is None or confidence <= 0.5:  # Handle None scores
                    recommendations.append(
                        f"Hold {symbol}: mildly negative news sentiment, monitor for weakness"
                    )
                else:
                    recommendations.append(
                        f"SELL bias for {symbol}: news sentiment bearish (score {score:.2f}, news {news_score:.2f}); consider trimming"
                    )
            elif sentiment == "neutral":
                if score is not None:
                    recommendations.append(
                        f"Hold {symbol}: neutral news sentiment (score {score:.2f}), watch for catalysts"
                    )
                else:
                    recommendations.append(
                        f"Hold {symbol}: neutral news sentiment, watch for catalysts"
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



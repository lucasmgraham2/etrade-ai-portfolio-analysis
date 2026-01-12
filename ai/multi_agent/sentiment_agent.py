"""
Sentiment Agent
Analyzes market sentiment from news, social media, and forums
"""

from typing import Dict, Any, List
from datetime import datetime, timedelta
import asyncio
import aiohttp
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
        
        return {
            "summary": self._create_summary(overall_sentiment),
            "overall_sentiment": overall_sentiment,
            "news_sentiment": news_sentiment,
            "insights": insights,
            "recommendations": self._generate_recommendations(overall_sentiment),
            "data_sources": ["news"],
            "analysis_period_days": self.lookback_days
        }
    
    async def _analyze_news_sentiment(self, symbols: List[str]) -> Dict[str, Any]:
        """
        Analyze news sentiment for portfolio symbols
        
        Uses APIs like:
        - Alpha Vantage News Sentiment
        - NewsAPI
        - Finnhub
        """
        self.log("Fetching news sentiment...")
        
        has_av = "alpha_vantage" in self.api_keys
        has_newsapi = "newsapi" in self.api_keys
        
        # If both APIs are available, fetch in parallel and merge for higher fidelity
        if has_av and has_newsapi:
            av_data, newsapi_data = await asyncio.gather(
                self._fetch_alpha_vantage_news(symbols),
                self._fetch_newsapi_data(symbols)
            )
            return self._merge_news_sources(symbols, av_data, newsapi_data)
        
        # Single-source fallbacks
        if has_av:
            return await self._fetch_alpha_vantage_news(symbols)
        if has_newsapi:
            return await self._fetch_newsapi_data(symbols)
        
        # Simulated only if no keys provided
        self.log("No news API key found, using simulated data", "WARNING")
        return
    
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
                            symbol_sentiments[symbol] = {"score": 0, "label": "neutral"}
                except Exception as e:
                    self.log(f"Error fetching news for {symbol}: {str(e)}", "ERROR")
                    symbol_sentiments[symbol] = {"score": 0, "label": "neutral"}
                    
                # Rate limiting
                await asyncio.sleep(0.5)
        
        return {
            "source": "alpha_vantage",
            "symbols": symbol_sentiments,
            "timestamp": datetime.now().isoformat()
        }
    
    async def _fetch_newsapi_data(self, symbols: List[str]) -> Dict[str, Any]:
        """Fetch news from NewsAPI and analyze sentiment with NLP"""
        api_key = self.api_keys.get("newsapi")
        
        if not api_key:
            self.log("NewsAPI key not configured, skipping NewsAPI", "WARNING")
            return {"source": "newsapi", "symbols": {}, "timestamp": datetime.now().isoformat()}
        
        symbol_sentiments = {}
        rate_limit_hit = False
        async with aiohttp.ClientSession() as session:
            for symbol in symbols:
                try:
                    # Skip if we've already hit rate limit
                    if rate_limit_hit:
                        self.log(f"Skipping {symbol} due to API rate limit", "WARNING")
                        symbol_sentiments[symbol] = {"score": 0, "label": "neutral", "count": 0, "method": "none", "articles_analyzed": 0}
                        continue
                    
                    # Calculate date range
                    end_date = datetime.now()
                    start_date = end_date - timedelta(days=self.lookback_days)
                    
                    url = (
                        f"https://newsapi.org/v2/everything?"
                        f"q={symbol}&"
                        f"from={start_date.strftime('%Y-%m-%d')}&"
                        f"to={end_date.strftime('%Y-%m-%d')}&"
                        f"sortBy=relevancy&"
                        f"language=en&"
                        f"pageSize=100&"
                        f"apiKey={api_key}"
                    )
                    
                    async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                        if response.status == 200:
                            data = await response.json()
                            articles = data.get("articles", [])
                            self.log(f"NewsAPI returned {len(articles)} articles for {symbol}", "DEBUG")
                            
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
                            self.log(f"NewsAPI rate limit (429) for {symbol} - free tier exhausted", "WARNING")
                            rate_limit_hit = True
                            symbol_sentiments[symbol] = {"score": 0, "label": "neutral", "count": 0, "method": "none", "articles_analyzed": 0}
                        else:
                            self.log(f"NewsAPI error for {symbol}: HTTP {response.status}", "WARNING")
                            symbol_sentiments[symbol] = {"score": 0, "label": "neutral", "count": 0, "method": "none", "articles_analyzed": 0}
                except Exception as e:
                    self.log(f"Error fetching from NewsAPI for {symbol}: {str(e)}", "ERROR")
                    symbol_sentiments[symbol] = {"score": 0, "label": "neutral", "count": 0, "method": "none", "articles_analyzed": 0}
                    
                # Add delay to respect rate limits
                await asyncio.sleep(1.0)
        
        self.log(f"NewsAPI analysis complete: {len([s for s in symbol_sentiments.values() if s.get('method') == 'nlp'])} symbols with NLP analysis", "INFO")
        
        return {
            "source": "newsapi",
            "symbols": symbol_sentiments,
            "timestamp": datetime.now().isoformat()
        }
    
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
            self.log("NLP not available, transformer model not loaded - using keyword fallback", "WARNING")
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
                    text = f"{article.get('title', '')} {article.get('description', '')}"
                    
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
                self.log("NLP analysis produced no sentiment scores, returning neutral", "WARNING")
                return {"score": 0, "label": "neutral", "count": len(articles), "articles_analyzed": 0, "method": "nlp"}
            
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
            title = (article.get("title", "") + " " + article.get("description", "")).lower()
            
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
            "count": len(articles)
        }
    
    def _parse_alpha_vantage_sentiment(self, data: Dict) -> Dict[str, Any]:
        """Parse Alpha Vantage sentiment response with improved thresholds"""
        feed = data.get("feed", [])
        
        if not feed:
            return {"score": 0, "label": "neutral", "count": 0}
        
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
        """Aggregate sentiment from news source"""
        
        aggregated = {}
        
        for symbol in symbols:
            # Get sentiment from news source
            news_sentiment = news.get("symbols", {}).get(symbol, {})
            
            # Use news score directly
            score = news_sentiment.get("score", 0)
            
            # Determine overall label
            if score > 0.12:
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
                "method": news_sentiment.get("method", "unknown"),  # Include NLP vs keyword method
                "articles_analyzed": news_sentiment.get("articles_analyzed", 0)
            }
        
        return aggregated

    def _merge_news_sources(
        self,
        symbols: List[str],
        av_data: Dict[str, Any],
        newsapi_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Merge Alpha Vantage and NewsAPI sentiment into a single feed.
        Weights default to AV 0.6 (broad coverage) and NewsAPI+OpenAI 0.4 (deeper reasoning).
        """

        merged = {}
        av_symbols = av_data.get("symbols", {}) if av_data else {}
        na_symbols = newsapi_data.get("symbols", {}) if newsapi_data else {}

        for symbol in symbols:
            av = av_symbols.get(symbol, {"score": 0, "label": "neutral", "count": 0})
            na = na_symbols.get(symbol, {"score": 0, "label": "neutral", "count": 0})

            # Weighted score blend
            score = av.get("score", 0) * 0.6 + na.get("score", 0) * 0.4

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
            if na.get("method") == "nlp":
                methods_used.append("nlp")
            elif na.get("count", 0) > 0:
                methods_used.append("newsapi")

            merged[symbol] = {
                "score": score,
                "label": label,
                "count": (av.get("count", 0) or 0) + (na.get("count", 0) or 0),
                "method": ", ".join(methods_used) if methods_used else "unknown",
                "articles_analyzed": na.get("articles_analyzed", 0),
                "components": {
                    "alpha_vantage": av,
                    "newsapi": na,
                },
            }

        return {
            "source": "merged_alphaVantage_newsapi",
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
            confidence = data["confidence"]
            news_score = data.get("news_score", 0)
            
            # More lenient thresholds to give actionable recommendations
            if sentiment == "bullish":
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
        
        avg_score = sum(d["overall_score"] for d in sentiment_data.values()) / total_symbols
        
        bullish_count = sum(1 for d in sentiment_data.values() if d["sentiment"] == "bullish")
        bearish_count = sum(1 for d in sentiment_data.values() if d["sentiment"] == "bearish")
        
        if avg_score > 0.2:
            market_mood = "positive"
        elif avg_score < -0.2:
            market_mood = "negative"
        else:
            market_mood = "neutral"
        
        return (
            f"Market sentiment analysis across {total_symbols} portfolio holdings shows "
            f"a {market_mood} overall mood (score: {avg_score:.2f}). "
            f"{bullish_count} positions have bullish sentiment, {bearish_count} are bearish."
        )
    
    async def _analyze_news_with_ai(self, articles: List[Dict]) -> Dict[str, Any]:
        """Use AI to analyze sentiment from news articles"""
        if not articles:
            return {"score": 0, "label": "neutral", "count": 0}
        
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
            self.log(f"AI analysis failed: {str(e)}, falling back to keyword matching", "WARNING")
            # Fallback to simple keyword matching
            return {"score": 0, "label": "neutral", "count": len(articles)}

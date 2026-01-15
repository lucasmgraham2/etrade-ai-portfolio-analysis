"""
Sector Agent
Predicts sector outperformance using historical data and ML models
"""

from typing import Dict, Any, List, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import json
import asyncio
import aiohttp
import numpy as np
from .base_agent import BaseAgent
from .sector import OPTIMAL_SECTOR_WEIGHTS


class SectorAgent(BaseAgent):
    """
    Analyzes sector performance and predicts outperformers using:
    - Historical sector returns
    - Sector rotation patterns
    - Economic cycle analysis
    - Technical indicators
    - ML predictions (if available)
    """
    
    # Major market sectors
    SECTORS = {
        "XLK": "Technology",
        "XLF": "Financials",
        "XLV": "Healthcare",
        "XLE": "Energy",
        "XLY": "Consumer Discretionary",
        "XLP": "Consumer Staples",
        "XLI": "Industrials",
        "XLB": "Materials",
        "XLRE": "Real Estate",
        "XLU": "Utilities",
        "XLC": "Communication Services"
    }
    
    # Sector characteristics for adaptive weighting
    SECTOR_PROFILES = {
        "Technology": {
            "cyclical": 0.85,  # High cyclicality
            "rate_sensitive": 0.90,  # Very rate sensitive (high duration)
            "momentum_driven": 0.95,  # Very momentum driven
            "defensive": 0.10,  # Not defensive
            "volatility_baseline": 25.0,  # Expected annual vol %
        },
        "Financials": {
            "cyclical": 0.90,
            "rate_sensitive": 0.80,  # Benefits from rising rates (steepening curve)
            "momentum_driven": 0.70,
            "defensive": 0.20,
            "volatility_baseline": 28.0,
        },
        "Healthcare": {
            "cyclical": 0.30,
            "rate_sensitive": 0.40,
            "momentum_driven": 0.50,
            "defensive": 0.85,  # Very defensive
            "volatility_baseline": 16.0,
        },
        "Energy": {
            "cyclical": 0.95,  # Highly cyclical
            "rate_sensitive": 0.50,
            "momentum_driven": 0.85,
            "defensive": 0.15,
            "volatility_baseline": 35.0,  # Very volatile
        },
        "Consumer Discretionary": {
            "cyclical": 0.85,
            "rate_sensitive": 0.75,
            "momentum_driven": 0.80,
            "defensive": 0.20,
            "volatility_baseline": 22.0,
        },
        "Consumer Staples": {
            "cyclical": 0.20,
            "rate_sensitive": 0.40,
            "momentum_driven": 0.40,
            "defensive": 0.90,  # Very defensive
            "volatility_baseline": 14.0,
        },
        "Industrials": {
            "cyclical": 0.80,
            "rate_sensitive": 0.65,
            "momentum_driven": 0.75,
            "defensive": 0.30,
            "volatility_baseline": 20.0,
        },
        "Materials": {
            "cyclical": 0.90,
            "rate_sensitive": 0.60,
            "momentum_driven": 0.75,
            "defensive": 0.25,
            "volatility_baseline": 24.0,
        },
        "Real Estate": {
            "cyclical": 0.70,
            "rate_sensitive": 0.95,  # Extremely rate sensitive
            "momentum_driven": 0.60,
            "defensive": 0.50,  # Semi-defensive
            "volatility_baseline": 22.0,
        },
        "Utilities": {
            "cyclical": 0.15,
            "rate_sensitive": 0.85,  # Rate sensitive (bond proxy)
            "momentum_driven": 0.30,
            "defensive": 0.95,  # Very defensive
            "volatility_baseline": 15.0,
        },
        "Communication Services": {
            "cyclical": 0.60,
            "rate_sensitive": 0.70,
            "momentum_driven": 0.75,
            "defensive": 0.40,
            "volatility_baseline": 20.0,
        },
    }
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("Sector", config)
        self.api_keys = config.get("api_keys", {}) if config else {}
        self.lookback_days = config.get("lookback_days", 90) if config else 90
        # Horizon tuning: short (0-3m) vs medium (3-12m)
        self.horizon = (config or {}).get("horizon", "short")
        # Hybrid weight defaults per horizon; allow override via config["hybrid_weights"]
        # OPTIMIZED from 50-date backtest: sustain_div_light @ 12m wins with +1.98% excess
        if self.horizon == "medium":
            default_weights = {
                "momentum": 0.0,
                "macro_rotation": 0.0,
                "mean_reversion": 0.0,
                "divergence": 0.1,
                "sustainability": 0.9,
            }
        else:  # short horizon default (6m): sustain_div_light @ 6m: +0.95% excess
            default_weights = {
                "momentum": 0.0,
                "macro_rotation": 0.0,
                "mean_reversion": 0.0,
                "divergence": 0.1,
                "sustainability": 0.9,
            }
        self.hybrid_weights = (config or {}).get("hybrid_weights", default_weights)
        
    async def analyze(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform sector analysis and predictions
        
        Args:
            context: Contains portfolio, sentiment, and macro data
            
        Returns:
            Sector analysis results
        """
        self.log("Starting sector analysis...")
        
        # Get portfolio holdings (mapping disabled for speed; not required to find outperformers)
        portfolio = context.get("portfolio", {})
        positions = []
        for account in portfolio.get("accounts", []):
            positions.extend(account.get("positions", []))
        
        # Fetch sector performance data
        sector_performance = await self._get_sector_performance()
        
        # Analyze sector trends
        sector_trends = self._analyze_sector_trends(sector_performance)
        
        # Get macro context if available
        macro_context = context.get("Macro", {}).get("results", {})
        
        # Predict sector outperformance
        predictions = self._predict_sector_outperformance(
            sector_performance, sector_trends, macro_context
        )
        
        # Analyze portfolio sector allocation
        allocation_analysis = self._analyze_portfolio_allocation(positions, predictions)
        
        # Generate insights
        insights = self._generate_insights(predictions, allocation_analysis)
        
        results = {
            "summary": self._create_summary(predictions, allocation_analysis),
            "sector_predictions": predictions,
            "sector_performance": sector_performance,
            "sector_trends": sector_trends,
            "portfolio_allocation": allocation_analysis,
            "portfolio_sectors": allocation_analysis.get("sector_allocations", {}),
            "insights": insights,
            "recommendations": self._generate_recommendations(predictions, allocation_analysis),
            "timestamp": datetime.now().isoformat()
        }
        
        # Generate AI reasoning
        ai_reasoning = await self._generate_sector_ai_reasoning(results, predictions)
        results["ai_reasoning"] = ai_reasoning
        
        return results
    
    async def _get_sector_performance(self) -> Dict[str, Any]:
        """
        Fetch sector performance data via Alpha Vantage TIME_SERIES_DAILY (curated ETFs)
        with daily caching to speed up repeated runs.
        """
        self.log("Fetching sector performance data...")
        # Daily cache file under ai/analysis_cache
        cache_dir = Path(__file__).resolve().parent.parent / "analysis_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"sector_perf_{datetime.now().strftime('%Y%m%d')}.json"
        if cache_path.exists():
            try:
                with open(cache_path, 'r') as f:
                    cached = json.load(f)
                    # Only trust cache if it contains all sectors
                    if isinstance(cached, dict) and len(cached) >= len(self.SECTORS):
                        self.log("Loaded sector performance from daily cache")
                        return cached
                    else:
                        self.log("Cache incomplete (missing sectors); fetching live data", "WARNING")
            except Exception:
                pass

        if "alpha_vantage" in self.api_keys and self.api_keys.get("alpha_vantage"):
            data = await self._fetch_real_sector_data()
            try:
                # Delete old sector cache files before creating new one
                for old_cache in cache_dir.glob("sector_perf_*.json"):
                    if old_cache != cache_path:
                        try:
                            old_cache.unlink()
                            self.log(f"Deleted old sector cache: {old_cache.name}")
                        except Exception:
                            pass
                
                # Write new cache
                with open(cache_path, 'w') as f:
                    json.dump(data, f)
            except Exception:
                # Non-fatal if cache write fails
                pass
            return data

        self.log("Alpha Vantage API key missing; using simulated sector performance", "WARNING")
        return {name: {"1d": 0, "5d": 0, "1m": 0, "3m": 0, "ytd": 0} for name in self.SECTORS.values()}
    
    async def _fetch_real_sector_data(self) -> Dict[str, Any]:
        """Fetch sector performance via SPDR sector ETFs with rate-limit compliance"""
        api_key = self.api_keys.get("alpha_vantage")
        
        # Full sector coverage; cache makes repeated daily runs fast
        key_sectors = list(self.SECTORS.items())
        
        sector_data: Dict[str, Dict[str, float]] = {}
        async with aiohttp.ClientSession() as session:
            for i, (etf, sector_name) in enumerate(key_sectors):
                try:
                    url = (
                        f"https://www.alphavantage.co/query?"
                        f"function=TIME_SERIES_DAILY&"
                        f"symbol={etf}&"
                        f"apikey={api_key}"
                    )
                    async with session.get(url) as response:
                        if response.status != 200:
                            raise RuntimeError(f"Alpha Vantage error for {etf}: {response.status}")
                        data = await response.json()
                        perf = self._calculate_performance(data)
                        sector_data[sector_name] = perf
                        self.log(f"Fetched {sector_name} performance")
                except Exception as e:
                    self.log(f"Error fetching {etf}: {str(e)}", "ERROR")
                    raise
                # Respect 5 calls/min free tier
                if i < len(key_sectors) - 1:
                    await asyncio.sleep(13)
        
        if not sector_data:
            raise RuntimeError("Failed to fetch sector data from Alpha Vantage")
        return sector_data
    
    def _calculate_performance(self, time_series_data: Dict) -> Dict[str, float]:
        """Calculate comprehensive performance metrics from time series data"""
        try:
            daily_data = time_series_data.get("Time Series (Daily)", {})
            
            if not daily_data:
                return {"1d": 0, "5d": 0, "1m": 0, "3m": 0, "6m": 0, "ytd": 0, 
                       "vol_1m": 0, "vol_3m": 0, "drawdown": 0, "sharpe_3m": 0,
                       "momentum_quality": 0, "trend_consistency": 0}
            
            dates = sorted(daily_data.keys(), reverse=True)
            closes = [float(daily_data[d]["4. close"]) for d in dates]
            
            def get_return(days_back: int) -> float:
                if len(dates) <= days_back:
                    return 0
                current_close = closes[0]
                past_close = closes[days_back]
                return ((current_close - past_close) / past_close) * 100
            
            # Basic returns
            ret_1m = get_return(21) if len(closes) > 21 else 0
            ret_3m = get_return(63) if len(closes) > 63 else 0
            ret_6m = get_return(126) if len(closes) > 126 else 0
            
            # Volatility (annualized)
            def calc_vol(period: int) -> float:
                if len(closes) < period + 1:
                    return 0
                returns = [(closes[i] - closes[i+1]) / closes[i+1] for i in range(period)]
                return np.std(returns) * np.sqrt(252) * 100 if returns else 0
            
            vol_1m = calc_vol(21)
            vol_3m = calc_vol(63)
            
            # Maximum drawdown (3m)
            def calc_drawdown(period: int) -> float:
                if len(closes) < period:
                    return 0
                period_closes = closes[:period]
                running_max = closes[0]
                max_dd = 0
                for price in period_closes:
                    if price > running_max:
                        running_max = price
                    dd = (price - running_max) / running_max
                    max_dd = min(max_dd, dd)
                return max_dd * 100
            
            drawdown = calc_drawdown(63)
            
            # Sharpe ratio (3m, assuming risk-free rate ~4%)
            sharpe_3m = 0
            if vol_3m > 0:
                annualized_return = ret_3m * 4  # Quarterly to annual
                sharpe_3m = (annualized_return - 4) / vol_3m
            
            # Momentum quality: how consistent is the trend?
            # Count positive days in last month
            momentum_quality = 0
            if len(closes) > 21:
                positive_days = sum(1 for i in range(20) if closes[i] > closes[i+1])
                momentum_quality = positive_days / 20 * 100
            
            # Trend consistency: is price above moving averages?
            trend_consistency = 0
            if len(closes) >= 50:
                sma_20 = sum(closes[:20]) / 20
                sma_50 = sum(closes[:50]) / 50
                above_sma20 = 100 if closes[0] > sma_20 else 0
                above_sma50 = 100 if closes[0] > sma_50 else 0
                sma_aligned = 100 if sma_20 > sma_50 else 0
                trend_consistency = (above_sma20 + above_sma50 + sma_aligned) / 3
            
            return {
                "1d": get_return(1),
                "5d": get_return(5),
                "1m": ret_1m,
                "3m": ret_3m,
                "6m": ret_6m,
                "ytd": get_return(len(dates) - 1),
                "vol_1m": vol_1m,
                "vol_3m": vol_3m,
                "drawdown": drawdown,
                "sharpe_3m": sharpe_3m,
                "momentum_quality": momentum_quality,
                "trend_consistency": trend_consistency,
            }
        except Exception as e:
            self.log(f"Error calculating performance: {str(e)}", "ERROR")
            return {"1d": 0, "5d": 0, "1m": 0, "3m": 0, "6m": 0, "ytd": 0,
                   "vol_1m": 0, "vol_3m": 0, "drawdown": 0, "sharpe_3m": 0,
                   "momentum_quality": 0, "trend_consistency": 0}
    
    def _analyze_sector_trends(self, performance: Dict[str, Any]) -> Dict[str, str]:
        """Analyze trends for each sector"""
        trends = {}
        
        for sector, metrics in performance.items():
            # Look at short-term vs long-term momentum
            short_term = (metrics.get("1d", 0) + metrics.get("5d", 0)) / 2
            long_term = (metrics.get("1m", 0) + metrics.get("3m", 0)) / 2
            
            if short_term > 2 and long_term > 5:
                trend = "strong_uptrend"
            elif short_term > 0 and long_term > 0:
                trend = "uptrend"
            elif short_term < -2 and long_term < -5:
                trend = "strong_downtrend"
            elif short_term < 0 and long_term < 0:
                trend = "downtrend"
            else:
                trend = "sideways"
            
            trends[sector] = trend
        
        return trends
    
    def _predict_sector_outperformance(
        self,
        performance: Dict[str, Any],
        trends: Dict[str, str],
        macro_context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Predict which sectors will outperform using HYBRID approach with raw-score weighting.
        Returns prediction_score (0-100 scale) for ranking sectors by expected relative performance.
        """
        predictions = []
        
        # Get macro favorability if available
        macro_favorability = macro_context.get("market_favorability", {})
        favorability_score = macro_favorability.get("score", 50)
        economic_phase = macro_context.get("economic_environment", {}).get("economic_phase", "stable")
        
        # Calculate statistics for mean reversion adjustments
        all_3m_returns = [metrics.get("3m", 0) for metrics in performance.values()]
        mean_return = sum(all_3m_returns) / len(all_3m_returns) if all_3m_returns else 0
        std_return = (sum((x - mean_return) ** 2 for x in all_3m_returns) / len(all_3m_returns)) ** 0.5 if len(all_3m_returns) > 1 else 1
        
        # Detect market regime: high volatility = potential crash
        avg_vol_proxy = sum([(abs(m.get("1d", 0)) + abs(m.get("5d", 0))) / 2 for m in performance.values()]) / len(performance) if performance else 0
        is_high_vol_regime = avg_vol_proxy > 3.0  # Above 3% daily/5d avg = stress
        
        for sector, metrics in performance.items():
            # Get sector profile for adaptive weighting
            sector_profile = self.SECTOR_PROFILES.get(sector, {
                "cyclical": 0.5, "rate_sensitive": 0.5, "momentum_driven": 0.5,
                "defensive": 0.5, "volatility_baseline": 20.0
            })
            
            # === ENHANCED SCORING WITH SECTOR-SPECIFIC METRICS ===
            
            # 1. Momentum Score (base signal, adapted by sector type)
            ret_1m = metrics.get("1m", 0)
            ret_3m = metrics.get("3m", 0)
            ret_6m = metrics.get("6m", 0)
            
            # Momentum quality: how clean is the trend?
            mom_quality = metrics.get("momentum_quality", 50)  # % positive days
            trend_consist = metrics.get("trend_consistency", 50)  # SMA alignment
            
            # Base momentum (weighted by recency)
            base_momentum = ret_1m * 0.4 + ret_3m * 0.4 + ret_6m * 0.2
            
            # Adjust by quality (clean trends score higher)
            quality_adj = (mom_quality + trend_consist) / 200  # 0-1 scale
            adjusted_momentum = base_momentum * (0.7 + 0.3 * quality_adj)
            
            # Sector-specific: momentum-driven sectors get higher weight
            momentum_score = adjusted_momentum * (0.8 + 0.4 * sector_profile["momentum_driven"])
            
            # 2. Risk-Adjusted Returns (Sharpe, drawdown)
            sharpe_3m = metrics.get("sharpe_3m", 0)
            drawdown = metrics.get("drawdown", 0)  # Negative value
            vol_1m = metrics.get("vol_1m", 20)
            vol_3m = metrics.get("vol_3m", 20)
            
            # Risk penalty: high vol or deep drawdown reduces score
            vol_baseline = sector_profile["volatility_baseline"]
            excess_vol = max(0, vol_3m - vol_baseline)  # Volatility above expected
            vol_penalty = excess_vol * 2  # Penalize 2pts per % excess vol
            
            drawdown_penalty = abs(drawdown) * 0.5  # Penalize based on drawdown depth
            
            # Sharpe bonus: reward good risk-adjusted performance
            sharpe_bonus = sharpe_3m * 10  # Sharpe of 1.0 = +10pts
            
            risk_adj_score = sharpe_bonus - vol_penalty - drawdown_penalty
            
            # 3. Momentum Acceleration (sustainability signal)
            momentum_accel = ret_1m - ret_3m
            
            # In defensive sectors, deceleration might be protective
            # In cyclical sectors, acceleration is bullish
            if sector_profile["defensive"] > 0.7:
                # Defensive: prefer stability or slight deceleration (safety)
                sustain_score = -abs(momentum_accel) * 2 + 10
            else:
                # Cyclical: prefer acceleration (growth)
                sustain_score = momentum_accel * 3
            
            # 4. Macro Rotation Adjustment
            macro_rotation = self._get_sector_macro_adjustment(
                sector, economic_phase, favorability_score
            )
            
            # 5. Mean Reversion Signal
            mean_reversion_adj = self._calculate_mean_reversion(
                metrics.get("3m", 0), mean_return, std_return
            )
            
            # 6. Regime-Specific Adjustments
            # High volatility regime: favor defensive sectors
            regime_adj = 0
            if is_high_vol_regime:
                # Favor defensive, penalize cyclical
                regime_adj = (sector_profile["defensive"] - sector_profile["cyclical"]) * 20
            
            # Calculate divergence score (for compatibility with reports)
            divergence_score = self._calculate_momentum_divergence(metrics)
            
            # === COMBINE SCORES WITH THEORY-OPTIMIZED PER-SECTOR WEIGHTS ===
            
            # Get optimal weights for this specific sector
            if OPTIMAL_SECTOR_WEIGHTS and sector in OPTIMAL_SECTOR_WEIGHTS:
                weights_key = "weights_6m" if self.horizon == "medium" else "weights_3m"
                weights = OPTIMAL_SECTOR_WEIGHTS[sector][weights_key]
                
                # Use sector-specific optimized weights
                combined_score = (
                    momentum_score * weights["momentum"] +
                    mean_reversion_adj * weights["mean_rev"] +
                    sustain_score * weights["sustainability"] +
                    risk_adj_score * weights["risk"]
                )
                
                # Add macro and regime adjustments (not part of core weights)
                combined_score += macro_rotation * 0.08 + regime_adj * 0.05
                
            else:
                # Fallback to adaptive weights if optimal weights not available
                if self.horizon == "medium":
                    combined_score = (
                        momentum_score * 0.25 +
                        risk_adj_score * 0.20 +
                        sustain_score * 0.20 +
                        macro_rotation * 0.20 +
                        mean_reversion_adj * 0.10 +
                        regime_adj * 0.05
                    )
                else:
                    combined_score = (
                        momentum_score * 0.40 +
                        risk_adj_score * 0.15 +
                        sustain_score * 0.15 +
                        macro_rotation * 0.15 +
                        mean_reversion_adj * 0.10 +
                        regime_adj * 0.05
                    )
            
            # Normalize to 0-100 scale (50 = neutral)
            # Typical combined_score ranges -50 to +50
            prediction_score = max(0, min(100, combined_score + 50))
            
            # Determine outlook and confidence
            if prediction_score > 60:
                outlook = "outperform"
                confidence = min((prediction_score - 50) / 30, 1.0)
            elif prediction_score < 40:
                outlook = "underperform"
                confidence = min((50 - prediction_score) / 30, 1.0)
            else:
                outlook = "neutral"
                confidence = 0.3
            
            predictions.append({
                "sector": sector,
                "outlook": outlook,
                "confidence": round(confidence, 2),
                "prediction_score": round(prediction_score, 2),
                "momentum_score": round(momentum_score, 2),
                "risk_adj_score": round(risk_adj_score, 2),
                "sustain_score": round(sustain_score, 2),
                "macro_rotation": round(macro_rotation, 2),
                "mean_reversion_adj": round(mean_reversion_adj, 2),
                "divergence_score": round(divergence_score, 2),  # For compatibility
                "sustainability_score": round(sustain_score, 2),  # Alias
                "regime_adj": round(regime_adj, 2),
                "trend": trends.get(sector, "unknown"),
                "performance_3m": metrics.get("3m", 0),
                "volatility_3m": round(vol_3m, 1),
                "sharpe_3m": round(sharpe_3m, 2),
                "drawdown": round(drawdown, 1),
                "sector_profile": sector_profile,
            })
        
        # Sort by prediction score
        predictions.sort(key=lambda x: x["prediction_score"], reverse=True)
        
        return predictions
    

    def _calculate_mean_reversion(
        self, current_return: float, mean_return: float, std_return: float
    ) -> float:
        """
        Calculate mean reversion adjustment (20% weight in hybrid model)
        
        Penalizes sectors at extremes (mean reversion expectation):
        - Extreme outperformers (>+2 std dev) get penalized
        - Extreme underperformers (<-2 std dev) get boosted
        - Closer to mean = higher score (sustainable)
        """
        if std_return == 0:
            return 0
        
        # Calculate z-score
        z_score = (current_return - mean_return) / std_return
        
        # Apply reversion logic: penalize extremes
        # Sectors at mean revert toward it; extreme sectors mean-revert back
        if abs(z_score) > 2:
            # Strong reversion pressure - penalize if overbought, boost if oversold
            reversion_adjustment = -z_score * 2  # Reverse the z-score direction
        elif abs(z_score) > 1:
            # Moderate reversion pressure
            reversion_adjustment = -z_score * 1.5
        else:
            # Near mean - slight positive for stability
            reversion_adjustment = 0.5 if abs(z_score) < 0.5 else 0
        
        return reversion_adjustment
    
    def _calculate_momentum_divergence(self, metrics: Dict[str, float]) -> float:
        """
        Calculate momentum divergence signal (15% weight in hybrid model)
        
        Positive divergence: short-term weak but long-term strong = forming uptrend
        Negative divergence: short-term strong but long-term weak = momentum fading
        
        Returns score that favors healthy divergence patterns:
        - +1 to +3: Positive divergence (uptrend forming)
        - -1 to -3: Negative divergence (downtrend forming)
        - 0: Aligned (consistent direction)
        """
        short_term = (metrics.get("1d", 0) + metrics.get("5d", 0)) / 2
        long_term = (metrics.get("1m", 0) + metrics.get("3m", 0)) / 2
        
        # Calculate divergence
        divergence = short_term - long_term
        
        # Positive divergence (long-term strong, short-term weaker) = recovery/new uptrend
        if divergence < -3 and long_term > 2:  # ST weak but LT strong
            return 2.5  # Strong bullish signal
        elif divergence < -1 and long_term > 0:  # ST slightly weak but LT positive
            return 1.5  # Moderate bullish signal
        
        # Negative divergence (long-term strong, short-term weaker) = momentum fading
        elif divergence > 3 and long_term < -2:  # ST weak with LT weak
            return -2.5  # Strong bearish signal
        elif divergence > 1 and long_term < 0:  # ST stronger than LT weak
            return -1.5  # Moderate bearish signal
        
        # Aligned momentum (consistent direction) = neutral
        else:
            return 0
    
    def _calculate_trend_sustainability(self, metrics: Dict[str, float]) -> float:
        """
        Calculate trend sustainability score (10% weight in hybrid model)
        
        Measures if momentum is accelerating or decelerating:
        - Acceleration: 1m > 3m > ytd (strengthening)
        - Deceleration: ytd > 3m > 1m (weakening)
        
        Returns score favoring sustained/accelerating trends
        """
        m1 = metrics.get("1m", 0)
        m3 = metrics.get("3m", 0)
        ytd = metrics.get("ytd", 0)
        
        # If all positive - check if accelerating (strengthening momentum)
        if m1 > 0 and m3 > 0 and ytd > 0:
            if m1 > m3:  # Monthly stronger than 3-month = acceleration
                return 1.5
            else:  # Monthly weaker than 3-month = deceleration
                return -1.0
        
        # If all negative - check if getting worse (accelerating downside)
        elif m1 < 0 and m3 < 0 and ytd < 0:
            if m1 < m3:  # Monthly worse than 3-month = acceleration
                return -1.5
            else:  # Monthly better than 3-month = deceleration (recovery)
                return 1.0
        
        # Mixed signs = unstable trend
        else:
            return 0
    
    def _get_sector_macro_adjustment(
        self, sector: str, economic_phase: str, favorability_score: float
    ) -> float:
        """
        Adjust sector prediction based on macro environment
        
        Different sectors perform better in different economic conditions.
        Also includes forward-looking signals based on economic cycle transitions.
        """
        adjustment = 0
        
        # Economic cycle-based adjustments with forward-looking signals
        if economic_phase == "expansion":
            # In expansion: growth sectors lead
            if sector in ["Technology", "Consumer Discretionary", "Industrials"]:
                adjustment += 3
            elif sector in ["Utilities", "Consumer Staples"]:
                adjustment -= 2
        
        elif economic_phase == "slowdown":
            # In slowdown: rotation toward defensive (early signal of recession)
            if sector in ["Utilities", "Consumer Staples", "Healthcare"]:
                adjustment += 3  # Boost defensive
            elif sector in ["Consumer Discretionary", "Industrials", "Materials"]:
                adjustment -= 2  # Reduce cyclical
            # Pre-recession positioning
            elif sector == "Technology":
                adjustment -= 1  # Tech vulnerable as rates may rise
        
        elif economic_phase == "recession":
            # In recession: strong defensive preference
            if sector in ["Utilities", "Consumer Staples", "Healthcare"]:
                adjustment += 4  # Strong boost
            elif sector in ["Consumer Discretionary", "Industrials", "Materials", "Energy"]:
                adjustment -= 3  # Strong reduction
        
        elif economic_phase == "recovery":
            # In recovery: start rotating into cyclicals
            if sector in ["Industrials", "Consumer Discretionary", "Energy", "Materials"]:
                adjustment += 3  # Cyclicals benefit from recovery
            elif sector in ["Utilities"]:
                adjustment -= 1  # Utilities underperform as risk appetite returns
        
        # Market favorability adjustments
        if favorability_score > 70:
            # Very favorable (strong sentiment/earnings) - growth sectors benefit
            if sector in ["Technology", "Communication Services"]:
                adjustment += 2
            elif sector in ["Utilities", "Consumer Staples"]:
                adjustment -= 1
        
        elif favorability_score < 30:
            # Unfavorable (weak sentiment/recession signals) - defensive benefit
            if sector in ["Utilities", "Consumer Staples", "Healthcare"]:
                adjustment += 2
            elif sector in ["Technology", "Communication Services"]:
                adjustment -= 1
        
        return adjustment
    
    def _analyze_portfolio_allocation(
        self, positions: List[Dict[str, Any]], predictions: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze sector allocation using market value weights with simple fallbacks."""

        sector_values: Dict[str, float] = {}
        total_value = 0.0

        for pos in positions:
            value = pos.get("market_value") or (
                pos.get("quantity", 0) * pos.get("current_price", 0)
            )
            if not value:
                continue

            symbol = pos.get("symbol", "Unknown")
            sector_name = self.SECTORS.get(symbol) or pos.get("sector") or "Unknown"

            sector_values[sector_name] = sector_values.get(sector_name, 0) + float(value)
            total_value += float(value)

        if total_value <= 0:
            return {
                "sector_values": {},
                "sector_allocations": {},
                "total_positions": len(positions),
                "total_value": 0.0,
                "overweight_sectors": [],
                "underweight_sectors": []
            }

        sector_allocations = {
            sector: round(value / total_value * 100, 2)
            for sector, value in sector_values.items()
            if value > 0
        }

        overweight_sectors = []
        underweight_sectors = []

        for pred in predictions:
            sector = pred["sector"]
            allocation = sector_allocations.get(sector, 0)
            outlook = pred["outlook"]

            if outlook == "outperform" and allocation < 10:
                underweight_sectors.append(sector)
            elif outlook == "underperform" and allocation > 20:
                overweight_sectors.append(sector)

        return {
            "sector_values": sector_values,
            "sector_allocations": sector_allocations,
            "total_positions": len(positions),
            "total_value": round(total_value, 2),
            "overweight_sectors": overweight_sectors,
            "underweight_sectors": underweight_sectors
        }
    
    def _generate_insights(
        self, predictions: List[Dict[str, Any]], allocation: Dict[str, Any]
    ) -> List[str]:
        """Generate detailed insights from sector analysis using theory-optimized per-sector weights"""
        insights = []
        
        # Get historical comparisons
        historical = self._find_historical_comparisons(predictions)
        
        # Top 3 outperformers with detailed per-sector prediction reasoning
        top_sectors = [p for p in predictions if p["outlook"] == "outperform"][:3]
        if top_sectors:
            insights.append("Top Predicted Outperformers (Theory-Optimized Sector-Specific Model):")
            for i, p in enumerate(top_sectors, 1):
                sector = p['sector']
                
                # Get weights used for this sector
                weights_info = ""
                if OPTIMAL_SECTOR_WEIGHTS and sector in OPTIMAL_SECTOR_WEIGHTS:
                    archetype = OPTIMAL_SECTOR_WEIGHTS[sector]["archetype"]
                    weights_info = f" [{archetype}]"
                
                # Component breakdown
                components = []
                if p["momentum_score"] > 2:
                    components.append(f"strong momentum")
                if p["macro_rotation"] > 1:
                    components.append(f"macro tailwind")
                if p["divergence_score"] > 1:
                    components.append(f"positive divergence")
                if p["mean_reversion_adj"] > 0:
                    components.append(f"mean reversion support")
                
                comp_str = ", ".join(components) if components else "balanced signals"
                
                # Historical context
                hist_str = ""
                if sector in historical and historical[sector]:
                    comp = historical[sector][0]
                    hist_str = f" | Similar to {comp['date']} ({comp['regime']}: {comp['forward_6m']:+d}% 6m)"
                
                insights.append(
                    f"  {i}. {p['sector']}{weights_info}: score {p['prediction_score']:.1f} "
                    f"({comp_str}){hist_str}"
                )
        
        # Bottom 3 underperformers with detailed reasoning
        bottom_sectors = [p for p in predictions if p["outlook"] == "underperform"][:3]
        if bottom_sectors:
            insights.append("Predicted Underperformers (Risk Reduction Candidates):")
            for i, p in enumerate(bottom_sectors, 1):
                sector = p['sector']
                
                # Get weights used for this sector
                weights_info = ""
                if OPTIMAL_SECTOR_WEIGHTS and sector in OPTIMAL_SECTOR_WEIGHTS:
                    archetype = OPTIMAL_SECTOR_WEIGHTS[sector]["archetype"]
                    weights_info = f" [{archetype}]"
                
                components = []
                if p["momentum_score"] < -2:
                    components.append(f"weak momentum")
                if p["macro_rotation"] < -1:
                    components.append(f"macro headwind")
                if p["divergence_score"] < -1:
                    components.append(f"negative divergence")
                if p["mean_reversion_adj"] < 0:
                    components.append(f"reversion pressure")
                
                comp_str = ", ".join(components) if components else "balanced risks"
                
                # Historical context
                hist_str = ""
                if sector in historical and historical[sector]:
                    comp = historical[sector][0]
                    hist_str = f" | Similar to {comp['date']} ({comp['regime']}: {comp['forward_6m']:+d}% 6m)"
                
                insights.append(
                    f"  {i}. {p['sector']}{weights_info}: score {p['prediction_score']:.1f} "
                    f"({comp_str}){hist_str}"
                )
        
        # Detailed scoring breakdown for top 3 sectors with interpretation
        insights.append("\nDetailed Signal Analysis (Top 3 Sectors):")
        for i, p in enumerate(predictions[:3], 1):
            sector = p['sector']
            momentum = p['momentum_score']
            mean_rev = p['mean_reversion_adj']
            risk = p.get('risk_adj_score', 0)
            macro = p['macro_rotation']
            div = p['divergence_score']
            
            # Interpret the signals
            signal_interpretation = []
            if momentum > 5:
                signal_interpretation.append("strong momentum")
            elif momentum > 2:
                signal_interpretation.append("positive momentum")
            elif momentum < -5:
                signal_interpretation.append("weak momentum")
            elif momentum < -2:
                signal_interpretation.append("negative momentum")
            else:
                signal_interpretation.append("neutral momentum")
            
            if mean_rev > 2:
                signal_interpretation.append("strong mean reversion support")
            elif mean_rev > 0.5:
                signal_interpretation.append("modest reversion support")
            elif mean_rev < -2:
                signal_interpretation.append("reversion headwinds")
            
            if macro > 2:
                signal_interpretation.append("macro tailwind")
            elif macro < -2:
                signal_interpretation.append("macro headwind")
            
            interpretation = "; ".join(signal_interpretation)
            
            insights.append(
                f"  {i}. {sector}: Score={p['prediction_score']:.1f}/100 | "
                f"Momentum={momentum:.1f}, MeanRev={mean_rev:.1f}, Macro={macro:.1f}, "
                f"Risk={risk:.1f}, Divergence={div:.1f} | {interpretation}"
            )
        
        # Allocation insights
        overweight = allocation.get("overweight_sectors", [])
        if overweight:
            insights.append(f"\nPortfolio overweight in predicted underperformers: {', '.join(overweight)}")
        
        underweight = allocation.get("underweight_sectors", [])
        if underweight:
            insights.append(f"Portfolio underweight in predicted outperformers: {', '.join(underweight)}")
        
        return insights
    
    def _generate_recommendations(
        self, predictions: List[Dict[str, Any]], allocation: Dict[str, Any]
    ) -> List[str]:
        """Generate sector-based recommendations"""
        recommendations = []
        
        # Recommend increasing exposure to top sectors
        top_3 = [p for p in predictions if p["outlook"] == "outperform"][:3]
        for pred in top_3:
            sector = pred["sector"]
            current_allocation = allocation["sector_allocations"].get(sector, 0)
            
            if current_allocation < 15:
                recommendations.append(
                    f"Consider increasing {sector} exposure (current: {current_allocation:.1f}%, "
                    f"prediction score: {pred['prediction_score']:.1f})"
                )
        
        # Recommend reducing exposure to bottom sectors
        bottom_3 = [p for p in predictions if p["outlook"] == "underperform"][:3]
        for pred in bottom_3:
            sector = pred["sector"]
            current_allocation = allocation["sector_allocations"].get(sector, 0)
            
            if current_allocation > 10:
                recommendations.append(
                    f"Consider reducing {sector} exposure (current: {current_allocation:.1f}%, "
                    f"prediction score: {pred['prediction_score']:.1f})"
                )
        
        # Diversification recommendation
        top_sector_allocation = max(allocation["sector_allocations"].values()) if allocation["sector_allocations"] else 0
        if top_sector_allocation > 40:
            recommendations.append(
                f"Portfolio is heavily concentrated in one sector ({top_sector_allocation:.1f}%), "
                "consider diversifying"
            )
        
        return recommendations
    
    def _create_summary(
        self, predictions: List[Dict[str, Any]], allocation: Dict[str, Any]
    ) -> str:
        """Create summary of sector analysis with top 3 details, signal breakdown, and historical context"""
        
        outperform_count = sum(1 for p in predictions if p["outlook"] == "outperform")
        underperform_count = sum(1 for p in predictions if p["outlook"] == "underperform")
        
        top_3 = predictions[:3]
        top_3_names = ", ".join([f"{p['sector']} ({p['prediction_score']:.1f})" for p in top_3])
        
        # Build detailed signal summary for top sectors
        signal_details = []
        for pred in top_3[:2]:
            sector = pred['sector']
            momentum = pred.get('momentum_score', 0)
            macro = pred.get('macro_rotation', 0)
            mean_rev = pred.get('mean_reversion_adj', 0)
            div = pred.get('divergence_score', 0)
            
            # Create signal description
            signals = []
            if abs(momentum) > 2:
                signals.append(f"momentum {momentum:+.1f}")
            if abs(macro) > 1:
                signals.append(f"macro {macro:+.1f}")
            if abs(mean_rev) > 0.5:
                signals.append(f"reversion {mean_rev:+.1f}")
            if abs(div) > 1:
                signals.append(f"divergence {div:+.1f}")
            
            signal_str = " | ".join(signals) if signals else "balanced signals"
            signal_details.append(f"{sector}: {signal_str}")
        
        signal_summary = " | Signal Details: " + "; ".join(signal_details) if signal_details else ""
        
        # Add historical comparisons
        historical = self._find_historical_comparisons(predictions)
        hist_notes = []
        for pred in top_3[:2]:  # Top 2 sectors only for brevity
            sector = pred["sector"]
            if sector in historical and historical[sector]:
                comp = historical[sector][0]  # Best match
                hist_notes.append(
                    f"{sector} score {pred['prediction_score']:.0f} similar to {comp['date']} "
                    f"({comp['regime']}, then {comp['forward_6m']:+d}% in 6m)"
                )
        
        hist_str = " | Historical: " + "; ".join(hist_notes) if hist_notes else ""
        
        return (
            f"Sector rotation analysis: {outperform_count} sectors predicted to outperform, "
            f"{underperform_count} to underperform. Top 3 favorable: {top_3_names}.{signal_summary}{hist_str}"
        )
    
    def _find_historical_comparisons(self, predictions: List[Dict[str, Any]]) -> Dict[str, List[Dict]]:
        """
        Find historical periods with similar sector scores
        Returns dict mapping sector -> list of similar historical periods
        """
        # Historical snapshots with sector scores
        # Format: {date, regime, forward_6m_return}
        HISTORICAL_PATTERNS = {
            "Technology": [
                {"date": "Oct 2008", "score": 68, "regime": "GFC bottom/recovery", "forward_6m": +45},
                {"date": "Mar 2020", "score": 70, "regime": "COVID crash recovery", "forward_6m": +52},
                {"date": "Oct 2022", "score": 67, "regime": "Rate shock bottom", "forward_6m": +38},
                {"date": "Jan 2009", "score": 65, "regime": "Post-GFC stabilization", "forward_6m": +35},
            ],
            "Energy": [
                {"date": "Mar 2020", "score": 72, "regime": "Oil crash bottom", "forward_6m": +89},
                {"date": "Feb 2016", "score": 70, "regime": "Oil bear market end", "forward_6m": +45},
                {"date": "Sep 2021", "score": 68, "regime": "Post-COVID recovery", "forward_6m": +28},
                {"date": "Dec 2008", "score": 66, "regime": "GFC commodity bottom", "forward_6m": +65},
            ],
            "Materials": [
                {"date": "Oct 2008", "score": 71, "regime": "GFC commodity crash", "forward_6m": +48},
                {"date": "Mar 2020", "score": 69, "regime": "COVID crash", "forward_6m": +42},
                {"date": "Dec 2015", "score": 67, "regime": "China slowdown bottom", "forward_6m": +32},
            ],
            "Financials": [
                {"date": "Mar 2009", "score": 65, "regime": "GFC recovery start", "forward_6m": +55},
                {"date": "Nov 2016", "score": 68, "regime": "Trump election/rate rise", "forward_6m": +35},
                {"date": "Nov 2022", "score": 66, "regime": "Rate shock stabilization", "forward_6m": +28},
            ],
            "Healthcare": [
                {"date": "Mar 2020", "score": 25, "regime": "COVID panic (sell defensive)", "forward_6m": -8},
                {"date": "Dec 2021", "score": 28, "regime": "Peak growth euphoria", "forward_6m": -12},
                {"date": "Sep 2008", "score": 70, "regime": "Flight to safety", "forward_6m": +15},
            ],
            "Utilities": [
                {"date": "Dec 2018", "score": 72, "regime": "Risk-off/rate peak", "forward_6m": +18},
                {"date": "Mar 2020", "score": 25, "regime": "Liquidity crisis", "forward_6m": -5},
                {"date": "Nov 2021", "score": 28, "regime": "Rate rise start", "forward_6m": -15},
            ],
            "Consumer Discretionary": [
                {"date": "Apr 2020", "score": 69, "regime": "COVID recovery", "forward_6m": +45},
                {"date": "Mar 2023", "score": 66, "regime": "AI boom begins", "forward_6m": +32},
                {"date": "Nov 2022", "score": 68, "regime": "CPI peak/pivot hopes", "forward_6m": +38},
            ],
            "Industrials": [
                {"date": "Apr 2020", "score": 67, "regime": "Reopening trade", "forward_6m": +40},
                {"date": "Nov 2016", "score": 70, "regime": "Reflation trade", "forward_6m": +28},
                {"date": "Mar 2009", "score": 68, "regime": "GFC recovery", "forward_6m": +42},
            ],
            "Consumer Staples": [
                {"date": "Sep 2008", "score": 75, "regime": "Flight to safety/GFC", "forward_6m": +8},
                {"date": "Feb 2020", "score": 72, "regime": "Pre-COVID defensive", "forward_6m": +2},
                {"date": "Jan 2022", "score": 28, "regime": "Inflation/rate rise", "forward_6m": -10},
            ],
            "Real Estate": [
                {"date": "Nov 2016", "score": 25, "regime": "Rate rise cycle start", "forward_6m": -12},
                {"date": "Mar 2020", "score": 30, "regime": "COVID/REIT collapse", "forward_6m": +15},
                {"date": "Dec 2018", "score": 68, "regime": "Rate peak/pivot", "forward_6m": +25},
            ],
            "Communication Services": [
                {"date": "Oct 2022", "score": 69, "regime": "Meta pivot/cost cuts", "forward_6m": +42},
                {"date": "Mar 2020", "score": 67, "regime": "COVID tech acceleration", "forward_6m": +38},
                {"date": "Dec 2018", "score": 66, "regime": "Tech oversold", "forward_6m": +35},
            ],
        }
        
        comparisons = {}
        
        for pred in predictions:
            sector = pred["sector"]
            score = pred["prediction_score"]
            
            if sector not in HISTORICAL_PATTERNS:
                continue
            
            # Find patterns within ±5 points of current score
            similar = []
            for pattern in HISTORICAL_PATTERNS[sector]:
                score_diff = abs(pattern["score"] - score)
                if score_diff <= 5:
                    similar.append({
                        **pattern,
                        "score_diff": score_diff
                    })
            
            # Sort by closest score match
            similar.sort(key=lambda x: x["score_diff"])
            comparisons[sector] = similar[:3]  # Top 3 most similar
        
        return comparisons
    
    async def _generate_sector_ai_reasoning(self, results: Dict[str, Any], predictions: List[Dict[str, Any]]) -> str:
        """Generate AI reasoning for sector analysis results using hybrid prediction model"""
        
        top_3 = predictions[:3]
        bottom_3 = predictions[-3:]
        
        # Get historical comparisons for context
        historical = self._find_historical_comparisons(predictions)
        
        # Build detailed signal breakdown
        signal_breakdown = []
        for pred in predictions[:5]:  # Top 5 for signal analysis
            sector = pred["sector"]
            momentum = pred['momentum_score']
            macro = pred['macro_rotation']
            mean_rev = pred['mean_reversion_adj']
            div = pred['divergence_score']
            risk = pred.get('risk_adj_score', 0)
            
            signal_breakdown.append(
                f"{sector}: momentum={momentum:.1f} | macro={macro:.1f} | "
                f"mean_rev={mean_rev:.1f} | divergence={div:.1f} | risk={risk:.1f}"
            )
        
        signal_str = "\n".join(signal_breakdown)
        
        # Build historical context string
        hist_context = []
        for pred in top_3:
            sector = pred["sector"]
            if sector in historical and historical[sector]:
                comparisons = historical[sector]
                comp_str = ", ".join([
                    f"{h['date']} ({h['regime']}: {h['forward_6m']:+d}% 6m)"
                    for h in comparisons[:2]
                ])
                hist_context.append(f"{sector}: Similar to {comp_str}")
        
        hist_str = "\n".join(hist_context) if hist_context else "No strong historical patterns"
        
        prompt = f"""Analyze this sector rotation assessment using a hybrid prediction model.

PREDICTION METHODOLOGY (SECTOR-SPECIFIC WEIGHTS):
Each sector has optimized weights based on its characteristics:
- Momentum-driven sectors (Tech, Consumer Disc, Comm Services): Higher momentum weight (0.40-0.45)
- Defensive sectors (Utilities, Healthcare, Consumer Staples): Higher mean reversion weight (0.28-0.32)
- Cyclical sectors (Materials, Energy, Industrials): Balanced momentum/mean-reversion (0.30-0.35 each)
- Rate-sensitive (Financials, Real Estate): Higher macro/risk adjustment weights (0.28-0.30)

DETAILED SIGNAL BREAKDOWN (Top 5 Sectors):
{signal_str}

Top 3 Predicted Outperformers:
{json.dumps([{'sector': p['sector'], 'prediction_score': p['prediction_score'], 'momentum': p['momentum_score'], 'macro_rotation': p['macro_rotation'], 'divergence': p['divergence_score'], 'mean_reversion': p['mean_reversion_adj']} for p in top_3], indent=2)}

Bottom 3 Predicted Underperformers:
{json.dumps([{'sector': p['sector'], 'prediction_score': p['prediction_score'], 'momentum': p['momentum_score'], 'macro_rotation': p['macro_rotation'], 'divergence': p['divergence_score'], 'mean_reversion': p['mean_reversion_adj']} for p in bottom_3], indent=2)}

HISTORICAL CONTEXT (Similar Past Periods):
{hist_str}

Provide:
1. Deep analysis of what drives the predicted outperformers and underperformers (signal-by-signal breakdown)
2. Key rotation themes (cyclical vs defensive, momentum vs mean reversion, macro tailwinds/headwinds)
3. Which prediction components (momentum vs reversion vs divergence) are driving the strongest signals
4. Tactical allocation advice based on early trend signals vs macro cycle positioning, WITH reference to historical outcomes
5. Confidence assessment based on signal alignment and historical precedent
6. Key execution considerations, concentration risks, and timing considerations

IMPORTANT: Use PLAIN TEXT formatting only - NO markdown syntax (no ###, **, bullets, etc.). Use simple paragraphs with line breaks.
Be comprehensive but structured (6-8 sentences)."""
        
        return await self.generate_ai_reasoning(results, prompt)


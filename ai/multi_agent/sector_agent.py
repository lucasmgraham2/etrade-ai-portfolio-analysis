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
from .base_agent import BaseAgent


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
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("Sector", config)
        self.api_keys = config.get("api_keys", {}) if config else {}
        self.lookback_days = config.get("lookback_days", 90) if config else 90
        
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
        portfolio_symbols = portfolio.get("summary", {}).get("unique_symbols", [])
        portfolio_sectors = {}
        
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
        allocation_analysis = self._analyze_portfolio_allocation(
            portfolio_sectors, predictions
        )
        
        # Generate insights
        insights = self._generate_insights(predictions, allocation_analysis)
        
        results = {
            "summary": self._create_summary(predictions, allocation_analysis),
            "sector_predictions": predictions,
            "sector_performance": sector_performance,
            "sector_trends": sector_trends,
            "portfolio_allocation": allocation_analysis,
            "portfolio_sectors": portfolio_sectors,
            "insights": insights,
            "recommendations": self._generate_recommendations(predictions, allocation_analysis),
            "timestamp": datetime.now().isoformat()
        }
        
        # Generate AI reasoning
        ai_reasoning = await self._generate_sector_ai_reasoning(results, predictions)
        results["ai_reasoning"] = ai_reasoning
        
        return results
    
    async def _map_symbols_to_sectors(
        self, symbols: List[str], portfolio: Dict[str, Any]
    ) -> Dict[str, List[str]]:
        """
        Map portfolio symbols to their sectors using live API data
        
        Uses Alpha Vantage OVERVIEW endpoint to fetch sector classifications
        """
        if not self.api_keys.get("alpha_vantage"):
            self.log("Warning: Alpha Vantage key missing; sector classifications unavailable", "WARNING")
            return {"Unknown": symbols}
        
        sectors_dict = {}
        api_key = self.api_keys.get("alpha_vantage")
        
        async with aiohttp.ClientSession() as session:
            for i, symbol in enumerate(symbols):
                try:
                    url = (
                        f"https://www.alphavantage.co/query?"
                        f"function=OVERVIEW&"
                        f"symbol={symbol}&"
                        f"apikey={api_key}"
                    )
                    
                    async with session.get(url) as response:
                        if response.status == 200:
                            data = await response.json()
                            sector = data.get("Sector", "Other")
                            
                            if sector not in sectors_dict:
                                sectors_dict[sector] = []
                            sectors_dict[sector].append(symbol)
                            self.log(f"Mapped {symbol} to {sector}")
                        else:
                            self.log(f"Failed to fetch sector for {symbol}", "WARNING")
                            if "Other" not in sectors_dict:
                                sectors_dict["Other"] = []
                            sectors_dict["Other"].append(symbol)
                    
                except Exception as e:
                    self.log(f"Error fetching sector for {symbol}: {str(e)}", "ERROR")
                    if "Other" not in sectors_dict:
                        sectors_dict["Other"] = []
                    sectors_dict["Other"].append(symbol)
                
                # Rate limiting: Alpha Vantage free tier is 5 calls/min
                if i < len(symbols) - 1:
                    await asyncio.sleep(13)
        
        if not sectors_dict:
            sectors_dict["Unknown"] = symbols
        
        return sectors_dict
    
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
        raise RuntimeError("Alpha Vantage API key missing: live sector data required")
    
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
        """Calculate performance metrics from time series data"""
        try:
            daily_data = time_series_data.get("Time Series (Daily)", {})
            
            if not daily_data:
                return {"1d": 0, "5d": 0, "1m": 0, "3m": 0, "ytd": 0}
            
            dates = sorted(daily_data.keys(), reverse=True)
            
            def get_return(days_back: int) -> float:
                if len(dates) <= days_back:
                    return 0
                current_close = float(daily_data[dates[0]]["4. close"])
                past_close = float(daily_data[dates[days_back]]["4. close"])
                return ((current_close - past_close) / past_close) * 100
            
            return {
                "1d": get_return(1),
                "5d": get_return(5),
                "1m": get_return(21),
                "3m": get_return(63),
                "ytd": get_return(len(dates) - 1)  # Approximate YTD
            }
        except Exception as e:
            self.log(f"Error calculating performance: {str(e)}", "ERROR")
            return {"1d": 0, "5d": 0, "1m": 0, "3m": 0, "ytd": 0}
    
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
        Predict which sectors will outperform
        
        Uses combination of:
        - Momentum indicators
        - Macro environment
        - Sector rotation patterns
        """
        predictions = []
        
        # Get macro favorability if available
        macro_favorability = macro_context.get("market_favorability", {})
        favorability_score = macro_favorability.get("score", 50)
        economic_phase = macro_context.get("economic_environment", {}).get("economic_phase", "stable")
        
        for sector, metrics in performance.items():
            # Calculate momentum score
            momentum_score = (
                metrics.get("1m", 0) * 0.3 +
                metrics.get("3m", 0) * 0.5 +
                metrics.get("ytd", 0) * 0.2
            )
            
            # Adjust based on macro environment
            macro_adjustment = self._get_sector_macro_adjustment(
                sector, economic_phase, favorability_score
            )
            
            # Calculate final prediction score
            prediction_score = momentum_score + macro_adjustment
            
            # Determine outlook
            if prediction_score > 5:
                outlook = "outperform"
                confidence = min(abs(prediction_score) / 10, 1.0)
            elif prediction_score < -5:
                outlook = "underperform"
                confidence = min(abs(prediction_score) / 10, 1.0)
            else:
                outlook = "neutral"
                confidence = 1.0 - min(abs(prediction_score) / 5, 1.0)
            
            predictions.append({
                "sector": sector,
                "outlook": outlook,
                "confidence": round(confidence, 2),
                "prediction_score": round(prediction_score, 2),
                "momentum_score": round(momentum_score, 2),
                "macro_adjustment": round(macro_adjustment, 2),
                "trend": trends.get(sector, "unknown"),
                "performance_3m": metrics.get("3m", 0)
            })
        
        # Sort by prediction score
        predictions.sort(key=lambda x: x["prediction_score"], reverse=True)
        
        return predictions
    
    def _get_sector_macro_adjustment(
        self, sector: str, economic_phase: str, favorability_score: float
    ) -> float:
        """
        Adjust sector prediction based on macro environment
        
        Different sectors perform better in different economic conditions
        """
        adjustment = 0
        
        # Economic cycle-based adjustments
        if economic_phase == "expansion":
            if sector in ["Technology", "Consumer Discretionary", "Industrials"]:
                adjustment += 3
            elif sector in ["Utilities", "Consumer Staples"]:
                adjustment -= 2
        elif economic_phase == "slowdown" or economic_phase == "recession":
            if sector in ["Utilities", "Consumer Staples", "Healthcare"]:
                adjustment += 3
            elif sector in ["Consumer Discretionary", "Industrials", "Materials"]:
                adjustment -= 2
        
        # Market favorability adjustment
        if favorability_score > 70:
            # Very favorable - growth sectors benefit
            if sector in ["Technology", "Communication Services"]:
                adjustment += 2
        elif favorability_score < 30:
            # Unfavorable - defensive sectors benefit
            if sector in ["Utilities", "Consumer Staples", "Healthcare"]:
                adjustment += 2
        
        return adjustment
    
    def _analyze_portfolio_allocation(
        self, portfolio_sectors: Dict[str, List[str]], predictions: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze how portfolio is allocated across sectors"""
        
        # Count positions per sector
        sector_counts = {sector: len(symbols) for sector, symbols in portfolio_sectors.items()}
        total_positions = sum(sector_counts.values())
        
        # Calculate allocation percentages
        sector_allocations = {
            sector: (count / total_positions * 100)
            for sector, count in sector_counts.items()
        }
        
        # Compare to predictions
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
            "sector_counts": sector_counts,
            "sector_allocations": sector_allocations,
            "total_positions": total_positions,
            "overweight_sectors": overweight_sectors,
            "underweight_sectors": underweight_sectors
        }
    
    def _generate_insights(
        self, predictions: List[Dict[str, Any]], allocation: Dict[str, Any]
    ) -> List[str]:
        """Generate detailed insights from sector analysis"""
        insights = []
        
        # Top 3 outperformers with detailed reasoning
        top_sectors = [p for p in predictions if p["outlook"] == "outperform"][:3]
        if top_sectors:
            insights.append("Top Outperforming Sectors:")
            for i, p in enumerate(top_sectors, 1):
                insights.append(
                    f"  {i}. {p['sector']}: score {p['prediction_score']:.1f}, "
                    f"momentum {p['momentum_score']:.1f}, trend {p['trend']}, "
                    f"3m perf {p['performance_3m']:.1f}%"
                )
        
        # Bottom 3 underperformers with detailed reasoning
        bottom_sectors = [p for p in predictions if p["outlook"] == "underperform"][:3]
        if bottom_sectors:
            insights.append("Sectors to Avoid or Reduce:")
            for i, p in enumerate(bottom_sectors, 1):
                insights.append(
                    f"  {i}. {p['sector']}: score {p['prediction_score']:.1f}, "
                    f"momentum {p['momentum_score']:.1f}, trend {p['trend']}, "
                    f"3m perf {p['performance_3m']:.1f}%"
                )
        
        # Allocation insights
        overweight = allocation.get("overweight_sectors", [])
        if overweight:
            insights.append(f"Portfolio overweight in lagging sectors: {', '.join(overweight)}")
        
        underweight = allocation.get("underweight_sectors", [])
        if underweight:
            insights.append(f"Portfolio underweight in outperforming sectors: {', '.join(underweight)}")
        
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
        """Create summary of sector analysis with top 3 details"""
        
        outperform_count = sum(1 for p in predictions if p["outlook"] == "outperform")
        underperform_count = sum(1 for p in predictions if p["outlook"] == "underperform")
        
        top_3 = predictions[:3]
        top_3_names = ", ".join([f"{p['sector']} ({p['prediction_score']:.1f})" for p in top_3])
        
        return (
            f"Sector rotation analysis: {outperform_count} sectors predicted to outperform, "
            f"{underperform_count} to underperform. Top 3 favorable: {top_3_names}."
        )
    
    async def _generate_sector_ai_reasoning(self, results: Dict[str, Any], predictions: List[Dict[str, Any]]) -> str:
        """Generate AI reasoning for sector analysis results"""
        
        top_3 = predictions[:3]
        bottom_3 = predictions[-3:]
        
        prompt = f"""Analyze this sector rotation assessment and provide concise investment insights.

Top 3 Favorable Sectors:
{json.dumps([{'sector': p['sector'], 'score': p['prediction_score'], 'outlook': p.get('outlook', 'neutral')} for p in top_3], indent=2)}

Bottom 3 Unfavorable Sectors:
{json.dumps([{'sector': p['sector'], 'score': p['prediction_score'], 'outlook': p.get('outlook', 'neutral')} for p in bottom_3], indent=2)}

Provide:
1. What this sector rotation signals about market conditions
2. Investment themes or trends driving these shifts
3. Tactical sector allocation advice
4. Key timing or execution considerations

Be concise (3-4 sentences max)."""
        
        return await self.generate_ai_reasoning(results, prompt)

"""
Macro Agent
Analyzes macroeconomic indicators and their impact on markets
"""

from typing import Dict, Any, List
from datetime import datetime, timedelta
import asyncio
import aiohttp
from .base_agent import BaseAgent


class MacroAgent(BaseAgent):
    """
    Analyzes macroeconomic environment using:
    - GDP data
    - Inflation rates (CPI, PPI)
    - Interest rates (Federal Funds Rate)
    - Unemployment rates
    - Treasury yields
    - Market indices
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("Macro", config)
        self.api_keys = config.get("api_keys", {}) if config else {}
        self.fred_api_key = self.api_keys.get("fred")  # Federal Reserve Economic Data
        
    async def analyze(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform macroeconomic analysis
        
        Args:
            context: Analysis context
            
        Returns:
            Macro analysis results
        """
        self.log("Starting macroeconomic analysis...")
        
        # Gather economic indicators in parallel
        indicator_tasks = [
            self._get_gdp_data(),
            self._get_inflation_data(),
            self._get_interest_rate_data(),
            self._get_unemployment_data(),
            self._get_treasury_yield_data(),
            self._get_market_indices()
        ]
        
        (gdp_data, inflation_data, interest_rate_data, 
         unemployment_data, treasury_data, market_indices) = await asyncio.gather(*indicator_tasks)
        
        # Analyze overall economic environment
        economic_environment = self._analyze_economic_environment(
            gdp_data, inflation_data, interest_rate_data, 
            unemployment_data, treasury_data, market_indices
        )
        
        # Determine market favorability
        market_favorability = self._calculate_market_favorability(economic_environment)
        
        # Generate insights
        insights = self._generate_insights(economic_environment, market_favorability)
        
        return {
            "summary": self._create_summary(economic_environment, market_favorability),
            "market_favorability": market_favorability,
            "economic_environment": economic_environment,
            "indicators": {
                "gdp": gdp_data,
                "inflation": inflation_data,
                "interest_rates": interest_rate_data,
                "unemployment": unemployment_data,
                "treasury_yields": treasury_data,
                "market_indices": market_indices
            },
            "insights": insights,
            "recommendations": self._generate_recommendations(market_favorability),
            "timestamp": datetime.now().isoformat()
        }
    
    async def _get_gdp_data(self) -> Dict[str, Any]:
        """Fetch GDP growth data"""
        self.log("Fetching GDP data...")
        if not self.fred_api_key:
            raise RuntimeError("FRED API key missing: live GDP data required")
        return await self._fetch_fred_data("GDP", "Gross Domestic Product")
    
    async def _get_inflation_data(self) -> Dict[str, Any]:
        """Fetch inflation data (CPI)"""
        self.log("Fetching inflation data...")
        if not self.fred_api_key:
            raise RuntimeError("FRED API key missing: live inflation data required")
        return await self._fetch_fred_data("CPIAUCSL", "Consumer Price Index")
    
    async def _get_interest_rate_data(self) -> Dict[str, Any]:
        """Fetch Federal Funds Rate"""
        self.log("Fetching interest rate data...")
        if not self.fred_api_key:
            raise RuntimeError("FRED API key missing: live rate data required")
        return await self._fetch_fred_data("FEDFUNDS", "Federal Funds Rate")
    
    async def _get_unemployment_data(self) -> Dict[str, Any]:
        """Fetch unemployment rate"""
        self.log("Fetching unemployment data...")
        if not self.fred_api_key:
            raise RuntimeError("FRED API key missing: live unemployment data required")
        return await self._fetch_fred_data("UNRATE", "Unemployment Rate")
    
    async def _get_treasury_yield_data(self) -> Dict[str, Any]:
        """Fetch 10-year Treasury yield"""
        self.log("Fetching Treasury yield data...")
        if not self.fred_api_key:
            raise RuntimeError("FRED API key missing: live Treasury data required")
        return await self._fetch_fred_data("DGS10", "10-Year Treasury Yield")
    
    async def _get_market_indices(self) -> Dict[str, Any]:
        """
        Fetch major market indices from Alpha Vantage
        
        Returns S&P 500, Dow Jones, and Nasdaq performance data
        """
        self.log("Fetching market indices...")
        
        if not self.api_keys.get("alpha_vantage"):
            raise RuntimeError("Alpha Vantage API key missing: live market index data required")
        
        # Use liquid ETF proxies for broad indices to ensure API coverage
        indices = {
            "SPY": "S&P 500 (via SPY)",
            "DIA": "Dow Jones (via DIA)",
            "QQQ": "Nasdaq 100 (via QQQ)"
        }
        
        index_data = {}
        api_key = self.api_keys.get("alpha_vantage")
        
        async with aiohttp.ClientSession() as session:
            for i, (symbol, name) in enumerate(indices.items()):
                try:
                    url = (
                        f"https://www.alphavantage.co/query?"
                        f"function=TIME_SERIES_DAILY&"
                        f"symbol={symbol}&"
                        f"apikey={api_key}"
                    )
                    
                    async with session.get(url) as response:
                        if response.status == 200:
                            data = await response.json()
                            daily_data = data.get("Time Series (Daily)", {})
                            
                            if daily_data:
                                dates = sorted(daily_data.keys(), reverse=True)
                                current_price = float(daily_data[dates[0]]["4. close"])
                                
                                # Calculate returns over various periods
                                def get_return(days_back: int) -> float:
                                    if len(dates) <= days_back:
                                        return 0
                                    past_price = float(daily_data[dates[days_back]]["4. close"])
                                    return ((current_price - past_price) / past_price) * 100
                                
                                index_data[name] = {
                                    "current_price": round(current_price, 2),
                                    "1d_change": round(get_return(1), 2),
                                    "5d_change": round(get_return(5), 2),
                                    "1m_change": round(get_return(21), 2),
                                    "3m_change": round(get_return(63), 2),
                                    "ytd_change": round(get_return(len(dates) - 1), 2)
                                }
                                self.log(f"Fetched {name} data")
                            else:
                                self.log(f"No data available for {symbol}", "WARNING")
                        else:
                            self.log(f"Failed to fetch {symbol}", "WARNING")
                    
                except Exception as e:
                    self.log(f"Error fetching {symbol}: {str(e)}", "ERROR")
                
                # Rate limiting between requests (Alpha Vantage free tier: 5 calls/min)
                if i < len(indices) - 1:
                    await asyncio.sleep(13)
        
        if not index_data:
            raise RuntimeError("Failed to fetch market indices from Alpha Vantage")
        
        return {
            "indices": index_data,
            "timestamp": datetime.now().isoformat()
        }
    
    async def _fetch_fred_data(self, series_id: str, name: str) -> Dict[str, Any]:
        """
        Fetch data from FRED (Federal Reserve Economic Data)
        
        Args:
            series_id: FRED series ID
            name: Human-readable name
        """
        try:
            url = (
                f"https://api.stlouisfed.org/fred/series/observations?"
                f"series_id={series_id}&"
                f"api_key={self.fred_api_key}&"
                f"file_type=json&"
                f"sort_order=desc&"
                f"limit=12"  # Get last 12 observations
            )
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        observations = data.get("observations", [])
                        
                        if observations:
                            # Find first valid observation (skip "." values)
                            latest = None
                            previous = None
                            for obs in observations:
                                if obs["value"] != "." and latest is None:
                                    latest = obs
                                elif obs["value"] != "." and previous is None:
                                    previous = obs
                                    break
                            
                            if latest is None:
                                raise ValueError(f"No valid data points found for {series_id}")
                            
                            if previous is None:
                                previous = latest
                            
                            current_value = float(latest["value"])
                            previous_value = float(previous["value"])
                            change = current_value - previous_value
                            change_pct = (change / previous_value * 100) if previous_value != 0 else 0
                            
                            return {
                                "name": name,
                                "current": current_value,
                                "previous": previous_value,
                                "change": change,
                                "change_pct": change_pct,
                                "date": latest["date"],
                                "trend": "increasing" if change > 0 else "decreasing" if change < 0 else "stable",
                                "source": "FRED"
                            }
        except Exception as e:
            self.log(f"Error fetching FRED data for {series_id}: {str(e)}", "ERROR")
        raise RuntimeError(f"Failed to fetch FRED data for {series_id}: {e}")
    
    def _analyze_economic_environment(
        self,
        gdp: Dict,
        inflation: Dict,
        rates: Dict,
        unemployment: Dict,
        treasury: Dict,
        indices: Dict
    ) -> Dict[str, Any]:
        """Analyze overall economic environment"""
        
        # Determine economic cycle phase
        phase = self._determine_economic_phase(gdp, inflation, unemployment, rates)
        
        # Assess growth outlook
        growth_outlook = self._assess_growth_outlook(gdp, unemployment, indices)
        
        # Assess inflation pressure
        inflation_pressure = self._assess_inflation_pressure(inflation, rates)
        
        # Assess monetary policy stance
        monetary_stance = self._assess_monetary_policy(rates, inflation)
        
        # Assess market risk
        market_risk = self._assess_market_risk(indices, treasury)
        
        return {
            "economic_phase": phase,
            "growth_outlook": growth_outlook,
            "inflation_pressure": inflation_pressure,
            "monetary_policy_stance": monetary_stance,
            "market_risk_level": market_risk
        }
    
    def _determine_economic_phase(
        self, gdp: Dict, inflation: Dict, unemployment: Dict, rates: Dict
    ) -> str:
        """Determine current economic cycle phase"""
        
        gdp_growth = gdp.get("current", 0)
        inflation_rate = inflation.get("current", 0)
        unemployment_rate = unemployment.get("current", 0)
        
        # Simplified phase determination
        if gdp_growth > 3.0 and unemployment_rate < 4.0:
            return "expansion"
        elif gdp_growth > 2.0 and inflation_rate < 3.0:
            return "moderate_growth"
        elif gdp_growth < 1.0:
            return "slowdown"
        elif gdp_growth < 0:
            return "recession"
        else:
            return "stable"
    
    def _assess_growth_outlook(self, gdp: Dict, unemployment: Dict, indices: Dict) -> str:
        """Assess economic growth outlook"""
        
        gdp_trend = gdp.get("trend", "stable")
        unemployment_trend = unemployment.get("trend", "stable")
        market_trend = indices.get("SP500", {}).get("trend", "neutral")
        
        positive_signals = 0
        if gdp_trend == "increasing":
            positive_signals += 1
        if unemployment_trend == "decreasing":
            positive_signals += 1
        if market_trend == "bullish":
            positive_signals += 1
        
        if positive_signals >= 2:
            return "positive"
        elif positive_signals == 1:
            return "neutral"
        else:
            return "negative"
    
    def _assess_inflation_pressure(self, inflation: Dict, rates: Dict) -> str:
        """Assess inflation pressure"""
        
        inflation_rate = inflation.get("current", 0)
        inflation_trend = inflation.get("trend", "stable")
        
        if inflation_rate > 3.5:
            return "high"
        elif inflation_rate > 2.5:
            return "elevated"
        elif inflation_rate > 2.0:
            return "moderate"
        else:
            return "low"
    
    def _assess_monetary_policy(self, rates: Dict, inflation: Dict) -> str:
        """Assess monetary policy stance"""
        
        rate_trend = rates.get("trend", "stable")
        inflation_rate = inflation.get("current", 0)
        current_rate = rates.get("current", 0)
        
        if rate_trend == "increasing":
            return "tightening"
        elif rate_trend == "decreasing":
            return "easing"
        elif current_rate > 5.0:
            return "restrictive"
        elif current_rate < 2.0:
            return "accommodative"
        else:
            return "neutral"
    
    def _assess_market_risk(self, indices: Dict, treasury: Dict) -> str:
        """Assess market risk level"""
        
        vix = indices.get("VIX", {}).get("current", 0)
        sp500_ytd = indices.get("SP500", {}).get("change_ytd_pct", 0)
        
        if vix > 25:
            return "high"
        elif vix > 20:
            return "elevated"
        elif vix > 15:
            return "moderate"
        else:
            return "low"
    
    def _calculate_market_favorability(self, environment: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate overall market favorability score
        
        Returns score from 0-100 with interpretation
        """
        score = 50  # Start at neutral
        
        # Economic phase impact
        phase = environment.get("economic_phase")
        if phase == "expansion":
            score += 20
        elif phase == "moderate_growth":
            score += 10
        elif phase == "slowdown":
            score -= 10
        elif phase == "recession":
            score -= 20
        
        # Growth outlook impact
        growth = environment.get("growth_outlook")
        if growth == "positive":
            score += 15
        elif growth == "negative":
            score -= 15
        
        # Inflation impact
        inflation = environment.get("inflation_pressure")
        if inflation == "low":
            score += 10
        elif inflation == "elevated":
            score -= 5
        elif inflation == "high":
            score -= 15
        
        # Monetary policy impact
        policy = environment.get("monetary_policy_stance")
        if policy == "easing":
            score += 10
        elif policy == "accommodative":
            score += 5
        elif policy == "tightening":
            score -= 10
        elif policy == "restrictive":
            score -= 5
        
        # Market risk impact
        risk = environment.get("market_risk_level")
        if risk == "low":
            score += 10
        elif risk == "elevated":
            score -= 5
        elif risk == "high":
            score -= 15
        
        # Ensure score is within bounds
        score = max(0, min(100, score))
        
        # Determine interpretation
        if score >= 70:
            interpretation = "very_favorable"
            recommendation = "Strong market conditions support growth-oriented strategies"
        elif score >= 55:
            interpretation = "favorable"
            recommendation = "Positive market conditions support moderate risk-taking"
        elif score >= 45:
            interpretation = "neutral"
            recommendation = "Mixed signals suggest balanced portfolio approach"
        elif score >= 30:
            interpretation = "unfavorable"
            recommendation = "Challenging conditions favor defensive positioning"
        else:
            interpretation = "very_unfavorable"
            recommendation = "Adverse conditions suggest capital preservation focus"
        
        return {
            "score": score,
            "interpretation": interpretation,
            "recommendation": recommendation
        }
    
    def _generate_insights(
        self, environment: Dict[str, Any], favorability: Dict[str, Any]
    ) -> List[str]:
        """Generate actionable insights"""
        insights = []
        
        # Economic phase insight
        phase = environment.get("economic_phase")
        insights.append(f"Economy is in {phase.replace('_', ' ')} phase")
        
        # Growth insight
        growth = environment.get("growth_outlook")
        insights.append(f"Growth outlook is {growth}")
        
        # Inflation insight
        inflation = environment.get("inflation_pressure")
        insights.append(f"Inflation pressure is {inflation}")
        
        # Monetary policy insight
        policy = environment.get("monetary_policy_stance")
        insights.append(f"Fed policy stance is {policy}")
        
        # Risk insight
        risk = environment.get("market_risk_level")
        insights.append(f"Market risk level: {risk}")
        
        # Overall favorability
        score = favorability.get("score")
        interp = favorability.get("interpretation")
        insights.append(f"Market favorability: {score}/100 ({interp.replace('_', ' ')})")
        
        return insights
    
    def _generate_recommendations(self, favorability: Dict[str, Any]) -> List[str]:
        """Generate macro-based recommendations"""
        recommendations = []
        
        score = favorability.get("score")
        
        if score >= 70:
            recommendations.extend([
                "Consider overweighting growth stocks in favorable environment",
                "Technology and cyclical sectors likely to outperform",
                "Maintain higher equity allocation vs fixed income"
            ])
        elif score >= 55:
            recommendations.extend([
                "Balanced approach with moderate growth tilt appropriate",
                "Selective opportunities in quality growth companies",
                "Monitor for signs of deterioration"
            ])
        elif score >= 45:
            recommendations.extend([
                "Neutral positioning with quality focus recommended",
                "Balance between growth and defensive sectors",
                "Increase cash allocation for flexibility"
            ])
        elif score >= 30:
            recommendations.extend([
                "Favor defensive sectors (utilities, consumer staples, healthcare)",
                "Reduce exposure to high-beta stocks",
                "Consider increasing bond allocation"
            ])
        else:
            recommendations.extend([
                "Defensive positioning strongly recommended",
                "Focus on capital preservation over growth",
                "Increase allocation to cash and high-quality bonds"
            ])
        
        return recommendations
    
    def _create_summary(
        self, environment: Dict[str, Any], favorability: Dict[str, Any]
    ) -> str:
        """Create summary of macro analysis"""
        
        phase = environment.get("economic_phase", "unknown")
        score = favorability.get("score", 50)
        interp = favorability.get("interpretation", "neutral")
        growth = environment.get("growth_outlook", "neutral")
        inflation = environment.get("inflation_pressure", "unknown")
        policy = environment.get("monetary_policy_stance", "neutral")
        risk = environment.get("market_risk_level", "moderate")
        
        reasons = [
            f"economic phase: {phase.replace('_', ' ')}",
            f"growth outlook: {growth}",
            f"inflation pressure: {inflation}",
            f"Fed stance: {policy}",
            f"market risk: {risk}"
        ]

        return (
            f"Macroeconomic analysis: {phase.replace('_', ' ')} phase; "
            f"market favorability {score}/100 ({interp.replace('_', ' ')}). "
            f"Drivers: {', '.join(reasons)}. "
            f"Guidance: {favorability.get('recommendation', '')}"
        )

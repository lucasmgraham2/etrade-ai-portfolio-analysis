"""
Popular Macroeconomic Metrics Analyzer
Analyzes mainstream economic indicators to assess market conditions
"""

from typing import Dict, Any
import asyncio
import aiohttp
from datetime import datetime


class PopularMetricsAnalyzer:
    """
    Analyzes popular macroeconomic indicators:
    - GDP Growth
    - Unemployment Rate
    - CPI/PPI Inflation
    - Federal Funds Rate
    - 10-Year Treasury Yield
    - S&P 500 Performance
    - ISM Manufacturing PMI
    - ISM Services PMI
    - Retail Sales
    - Consumer Confidence Index
    """
    
    def __init__(self, api_keys: Dict[str, str], weights: Dict[str, float]):
        """
        Initialize Popular Metrics Analyzer
        
        Args:
            api_keys: Dictionary with 'fred' and 'alpha_vantage' keys
            weights: Dictionary of metric weights from config
        """
        self.fred_api_key = api_keys.get("fred")
        self.alpha_vantage_key = api_keys.get("alpha_vantage")
        self.weights = weights
        
        if not self.fred_api_key:
            raise ValueError("FRED API key is required for popular metrics")
        if not self.alpha_vantage_key:
            raise ValueError("Alpha Vantage API key is required for popular metrics")
    
    async def analyze(self) -> Dict[str, Any]:
        """
        Perform full analysis of popular metrics
        
        Returns:
            Dictionary with scores, data, and overall assessment
        """
        print("[Popular Metrics] Starting analysis...")
        
        # Fetch all metrics in parallel
        tasks = [
            self._fetch_gdp(),
            self._fetch_unemployment(),
            self._fetch_cpi(),
            self._fetch_ppi(),
            self._fetch_fed_funds_rate(),
            self._fetch_treasury_10y(),
            self._fetch_sp500(),
            self._fetch_ism_manufacturing(),
            self._fetch_ism_services(),
            self._fetch_retail_sales(),
            self._fetch_consumer_confidence()
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Unpack results
        (gdp_data, unemployment_data, cpi_data, ppi_data, fed_funds_data,
         treasury_10y_data, sp500_data, ism_mfg_data, ism_services_data,
         retail_sales_data, consumer_conf_data) = results
        
        # Calculate individual scores
        scores = {
            "gdp_growth": self._score_gdp(gdp_data),
            "unemployment": self._score_unemployment(unemployment_data),
            "cpi_inflation": self._score_cpi(cpi_data),
            "ppi_inflation": self._score_ppi(ppi_data),
            "fed_funds_rate": self._score_fed_funds(fed_funds_data, cpi_data),
            "treasury_10y": self._score_treasury(treasury_10y_data),
            "sp500_performance": self._score_sp500(sp500_data),
            "ism_manufacturing": self._score_ism(ism_mfg_data),
            "ism_services": self._score_ism(ism_services_data),
            "retail_sales": self._score_retail_sales(retail_sales_data),
            "consumer_confidence": self._score_consumer_confidence(consumer_conf_data)
        }
        
        # Calculate weighted composite score
        composite_score = self._calculate_composite_score(scores)
        
        return {
            "composite_score": composite_score,
            "individual_scores": scores,
            "raw_data": {
                "gdp": gdp_data,
                "unemployment": unemployment_data,
                "cpi": cpi_data,
                "ppi": ppi_data,
                "fed_funds_rate": fed_funds_data,
                "treasury_10y": treasury_10y_data,
                "sp500": sp500_data,
                "ism_manufacturing": ism_mfg_data,
                "ism_services": ism_services_data,
                "retail_sales": retail_sales_data,
                "consumer_confidence": consumer_conf_data
            },
            "timestamp": datetime.now().isoformat()
        }
    
    async def _fetch_fred_data(self, series_id: str, name: str) -> Dict[str, Any]:
        """Fetch data from FRED API"""
        try:
            url = (
                f"https://api.stlouisfed.org/fred/series/observations?"
                f"series_id={series_id}&"
                f"api_key={self.fred_api_key}&"
                f"file_type=json&"
                f"sort_order=desc&"
                f"limit=24"  # Get last 24 observations for trend analysis
            )
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        observations = data.get("observations", [])
                        
                        # Filter out invalid values
                        valid_obs = [obs for obs in observations if obs["value"] != "."]
                        
                        if not valid_obs:
                            raise ValueError(f"No valid data for {series_id}")
                        
                        current = float(valid_obs[0]["value"])
                        previous = float(valid_obs[1]["value"]) if len(valid_obs) > 1 else current
                        year_ago = float(valid_obs[12]["value"]) if len(valid_obs) > 12 else current
                        
                        return {
                            "name": name,
                            "current": current,
                            "previous": previous,
                            "year_ago": year_ago,
                            "mom_change": current - previous,
                            "yoy_change": current - year_ago,
                            "yoy_change_pct": ((current - year_ago) / year_ago * 100) if year_ago != 0 else 0,
                            "date": valid_obs[0]["date"],
                            "trend": "increasing" if current > previous else "decreasing" if current < previous else "stable"
                        }
                    else:
                        raise RuntimeError(f"FRED API error: {response.status}")
        except Exception as e:
            print(f"[Popular Metrics] Error fetching {series_id}: {e}")
            raise
    
    async def _fetch_gdp(self) -> Dict[str, Any]:
        """Fetch GDP data"""
        return await self._fetch_fred_data("GDP", "Real GDP")
    
    async def _fetch_unemployment(self) -> Dict[str, Any]:
        """Fetch unemployment rate"""
        return await self._fetch_fred_data("UNRATE", "Unemployment Rate")
    
    async def _fetch_cpi(self) -> Dict[str, Any]:
        """Fetch CPI (Consumer Price Index)"""
        return await self._fetch_fred_data("CPIAUCSL", "Consumer Price Index")
    
    async def _fetch_ppi(self) -> Dict[str, Any]:
        """Fetch PPI (Producer Price Index)"""
        return await self._fetch_fred_data("PPIACO", "Producer Price Index")
    
    async def _fetch_fed_funds_rate(self) -> Dict[str, Any]:
        """Fetch Federal Funds Rate"""
        return await self._fetch_fred_data("FEDFUNDS", "Federal Funds Rate")
    
    async def _fetch_treasury_10y(self) -> Dict[str, Any]:
        """Fetch 10-Year Treasury Yield"""
        return await self._fetch_fred_data("DGS10", "10-Year Treasury Yield")
    
    async def _fetch_ism_manufacturing(self) -> Dict[str, Any]:
        """Fetch ISM Manufacturing PMI"""
        return await self._fetch_fred_data("NAPM", "ISM Manufacturing PMI")
    
    async def _fetch_ism_services(self) -> Dict[str, Any]:
        """Fetch ISM Services PMI"""
        return await self._fetch_fred_data("NMFBPI", "ISM Services PMI")
    
    async def _fetch_retail_sales(self) -> Dict[str, Any]:
        """Fetch Retail Sales"""
        return await self._fetch_fred_data("RSAFS", "Retail Sales")
    
    async def _fetch_consumer_confidence(self) -> Dict[str, Any]:
        """Fetch Consumer Confidence Index"""
        return await self._fetch_fred_data("UMCSENT", "Consumer Confidence")
    
    async def _fetch_sp500(self) -> Dict[str, Any]:
        """Fetch S&P 500 data from Alpha Vantage"""
        try:
            url = (
                f"https://www.alphavantage.co/query?"
                f"function=TIME_SERIES_DAILY&"
                f"symbol=SPY&"
                f"apikey={self.alpha_vantage_key}"
            )
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        daily_data = data.get("Time Series (Daily)", {})
                        
                        if not daily_data:
                            raise ValueError("No S&P 500 data available")
                        
                        dates = sorted(daily_data.keys(), reverse=True)
                        current_price = float(daily_data[dates[0]]["4. close"])
                        
                        # Calculate returns
                        month_ago_price = float(daily_data[dates[21]]["4. close"]) if len(dates) > 21 else current_price
                        three_month_price = float(daily_data[dates[63]]["4. close"]) if len(dates) > 63 else current_price
                        year_ago_price = float(daily_data[dates[252]]["4. close"]) if len(dates) > 252 else current_price
                        
                        return {
                            "name": "S&P 500",
                            "current": current_price,
                            "1m_return": ((current_price - month_ago_price) / month_ago_price * 100),
                            "3m_return": ((current_price - three_month_price) / three_month_price * 100),
                            "1y_return": ((current_price - year_ago_price) / year_ago_price * 100),
                            "date": dates[0]
                        }
                    else:
                        raise RuntimeError(f"Alpha Vantage API error: {response.status}")
        except Exception as e:
            print(f"[Popular Metrics] Error fetching S&P 500: {e}")
            raise
    
    def _score_gdp(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Score GDP growth (0-100 scale)
        - Above 3%: Very positive (70-100)
        - 2-3%: Positive (55-70)
        - 1-2%: Neutral (45-55)
        - 0-1%: Slightly negative (30-45)
        - Below 0%: Negative (0-30)
        """
        if isinstance(data, Exception):
            return {"score": 50, "reasoning": "Data unavailable", "error": str(data)}
        
        yoy_change = data.get("yoy_change_pct", 0)
        
        if yoy_change >= 3.0:
            score = 70 + min(30, (yoy_change - 3.0) * 10)
        elif yoy_change >= 2.0:
            score = 55 + (yoy_change - 2.0) * 15
        elif yoy_change >= 1.0:
            score = 45 + (yoy_change - 1.0) * 10
        elif yoy_change >= 0:
            score = 30 + yoy_change * 15
        else:
            score = max(0, 30 + yoy_change * 10)
        
        return {
            "score": round(score, 1),
            "reasoning": f"GDP growth at {yoy_change:.1f}% YoY",
            "data": data
        }
    
    def _score_unemployment(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Score unemployment rate (0-100 scale)
        - Below 4%: Very positive (70-100)
        - 4-5%: Positive (55-70)
        - 5-6%: Neutral (45-55)
        - 6-7%: Negative (30-45)
        - Above 7%: Very negative (0-30)
        """
        if isinstance(data, Exception):
            return {"score": 50, "reasoning": "Data unavailable", "error": str(data)}
        
        rate = data.get("current", 5.0)
        trend = data.get("trend", "stable")
        
        if rate < 4.0:
            score = 70 + (4.0 - rate) * 15
        elif rate < 5.0:
            score = 55 + (5.0 - rate) * 15
        elif rate < 6.0:
            score = 45 + (6.0 - rate) * 10
        elif rate < 7.0:
            score = 30 + (7.0 - rate) * 15
        else:
            score = max(0, 30 - (rate - 7.0) * 10)
        
        # Adjust for trend
        if trend == "decreasing":
            score += 5
        elif trend == "increasing":
            score -= 5
        
        return {
            "score": round(max(0, min(100, score)), 1),
            "reasoning": f"Unemployment at {rate:.1f}%, {trend}",
            "data": data
        }
    
    def _score_cpi(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Score CPI inflation (0-100 scale)
        - 1.5-2.5%: Optimal (70-100)
        - 2.5-3.5%: Acceptable (55-70)
        - 3.5-5%: Elevated (30-55)
        - Above 5%: High (0-30)
        - Below 1.5%: Low/deflationary risk (40-70)
        """
        if isinstance(data, Exception):
            return {"score": 50, "reasoning": "Data unavailable", "error": str(data)}
        
        yoy_change = data.get("yoy_change_pct", 2.0)
        
        if 1.5 <= yoy_change <= 2.5:
            score = 70 + (2.0 - abs(yoy_change - 2.0)) * 30
        elif 2.5 < yoy_change <= 3.5:
            score = 55 + (3.5 - yoy_change) * 15
        elif 3.5 < yoy_change <= 5.0:
            score = 30 + (5.0 - yoy_change) * 16.7
        elif yoy_change > 5.0:
            score = max(0, 30 - (yoy_change - 5.0) * 10)
        else:  # Below 1.5%
            score = 40 + yoy_change * 20
        
        return {
            "score": round(score, 1),
            "reasoning": f"CPI inflation at {yoy_change:.1f}% YoY",
            "data": data
        }
    
    def _score_ppi(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Score PPI inflation (similar to CPI but leading indicator)"""
        if isinstance(data, Exception):
            return {"score": 50, "reasoning": "Data unavailable", "error": str(data)}
        
        yoy_change = data.get("yoy_change_pct", 2.0)
        
        # Similar scoring to CPI
        if 1.5 <= yoy_change <= 3.0:
            score = 70 + (2.25 - abs(yoy_change - 2.25)) * 20
        elif 3.0 < yoy_change <= 5.0:
            score = 50 + (5.0 - yoy_change) * 10
        elif yoy_change > 5.0:
            score = max(0, 50 - (yoy_change - 5.0) * 10)
        else:  # Below 1.5%
            score = 40 + yoy_change * 20
        
        return {
            "score": round(score, 1),
            "reasoning": f"PPI inflation at {yoy_change:.1f}% YoY",
            "data": data
        }
    
    def _score_fed_funds(self, data: Dict[str, Any], cpi_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Score Fed Funds Rate (0-100 scale)
        Consider both absolute level and real rate (vs inflation)
        """
        if isinstance(data, Exception):
            return {"score": 50, "reasoning": "Data unavailable", "error": str(data)}
        
        rate = data.get("current", 5.0)
        trend = data.get("trend", "stable")
        
        # Calculate real rate if CPI data available
        if not isinstance(cpi_data, Exception):
            cpi_rate = cpi_data.get("yoy_change_pct", 2.0)
            real_rate = rate - cpi_rate
        else:
            real_rate = rate - 2.0  # Assume 2% inflation
        
        # Score based on real rate and trend
        if real_rate < 0:  # Negative real rates = accommodative
            score = 70 + min(30, abs(real_rate) * 10)
        elif real_rate < 1.0:
            score = 55 + (1.0 - real_rate) * 15
        elif real_rate < 2.0:
            score = 45 + (2.0 - real_rate) * 10
        else:  # Restrictive policy
            score = max(20, 45 - (real_rate - 2.0) * 12.5)
        
        # Adjust for trend (cutting = positive, hiking = negative)
        if trend == "decreasing":
            score += 10
        elif trend == "increasing":
            score -= 10
        
        return {
            "score": round(max(0, min(100, score)), 1),
            "reasoning": f"Fed Funds at {rate:.2f}%, real rate ~{real_rate:.1f}%, {trend}",
            "data": data
        }
    
    def _score_treasury(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Score 10Y Treasury Yield
        Lower yields generally better for equities (easier financing)
        """
        if isinstance(data, Exception):
            return {"score": 50, "reasoning": "Data unavailable", "error": str(data)}
        
        yield_rate = data.get("current", 4.0)
        trend = data.get("trend", "stable")
        
        if yield_rate < 3.0:
            score = 70 + (3.0 - yield_rate) * 10
        elif yield_rate < 4.0:
            score = 55 + (4.0 - yield_rate) * 15
        elif yield_rate < 5.0:
            score = 40 + (5.0 - yield_rate) * 15
        else:
            score = max(20, 40 - (yield_rate - 5.0) * 10)
        
        # Rising yields = headwind for stocks
        if trend == "increasing":
            score -= 5
        elif trend == "decreasing":
            score += 5
        
        return {
            "score": round(max(0, min(100, score)), 1),
            "reasoning": f"10Y Treasury at {yield_rate:.2f}%, {trend}",
            "data": data
        }
    
    def _score_sp500(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Score S&P 500 performance (momentum indicator)
        """
        if isinstance(data, Exception):
            return {"score": 50, "reasoning": "Data unavailable", "error": str(data)}
        
        three_month_return = data.get("3m_return", 0)
        one_year_return = data.get("1y_return", 0)
        
        # Weight 3-month more heavily for medium-term outlook
        composite_return = three_month_return * 0.6 + one_year_return * 0.4
        
        if composite_return > 15:
            score = 80 + min(20, (composite_return - 15) * 2)
        elif composite_return > 5:
            score = 60 + (composite_return - 5) * 2
        elif composite_return > -5:
            score = 40 + (composite_return + 5) * 2
        elif composite_return > -15:
            score = 20 + (composite_return + 15) * 2
        else:
            score = max(0, 20 + (composite_return + 15))
        
        return {
            "score": round(score, 1),
            "reasoning": f"S&P 500: 3M {three_month_return:.1f}%, 1Y {one_year_return:.1f}%",
            "data": data
        }
    
    def _score_ism(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Score ISM PMI (both Manufacturing and Services)
        - Above 55: Strong expansion (70-100)
        - 50-55: Expansion (55-70)
        - 45-50: Slight contraction (40-55)
        - Below 45: Contraction (0-40)
        """
        if isinstance(data, Exception):
            return {"score": 50, "reasoning": "Data unavailable", "error": str(data)}
        
        pmi = data.get("current", 50)
        name = data.get("name", "ISM PMI")
        
        if pmi >= 55:
            score = 70 + min(30, (pmi - 55) * 3)
        elif pmi >= 50:
            score = 55 + (pmi - 50) * 3
        elif pmi >= 45:
            score = 40 + (pmi - 45) * 3
        else:
            score = max(0, pmi * 0.89)  # Linear scale to 0
        
        return {
            "score": round(score, 1),
            "reasoning": f"{name} at {pmi:.1f} ({'expansion' if pmi >= 50 else 'contraction'})",
            "data": data
        }
    
    def _score_retail_sales(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Score Retail Sales growth
        Strong consumer spending = positive for economy
        """
        if isinstance(data, Exception):
            return {"score": 50, "reasoning": "Data unavailable", "error": str(data)}
        
        yoy_change = data.get("yoy_change_pct", 3.0)
        
        if yoy_change > 6:
            score = 75 + min(25, (yoy_change - 6) * 5)
        elif yoy_change > 3:
            score = 55 + (yoy_change - 3) * 6.7
        elif yoy_change > 0:
            score = 40 + yoy_change * 5
        else:
            score = max(10, 40 + yoy_change * 10)
        
        return {
            "score": round(score, 1),
            "reasoning": f"Retail sales growth at {yoy_change:.1f}% YoY",
            "data": data
        }
    
    def _score_consumer_confidence(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Score Consumer Confidence Index
        - Above 100: Strong confidence (70-100)
        - 90-100: Good (55-70)
        - 80-90: Moderate (45-55)
        - Below 80: Weak (0-45)
        """
        if isinstance(data, Exception):
            return {"score": 50, "reasoning": "Data unavailable", "error": str(data)}
        
        confidence = data.get("current", 90)
        trend = data.get("trend", "stable")
        
        if confidence >= 100:
            score = 70 + min(30, (confidence - 100) * 1.5)
        elif confidence >= 90:
            score = 55 + (confidence - 90) * 1.5
        elif confidence >= 80:
            score = 45 + (confidence - 80) * 1
        else:
            score = max(10, confidence * 0.56)
        
        # Adjust for trend
        if trend == "increasing":
            score += 5
        elif trend == "decreasing":
            score -= 5
        
        return {
            "score": round(max(0, min(100, score)), 1),
            "reasoning": f"Consumer confidence at {confidence:.1f}, {trend}",
            "data": data
        }
    
    def _calculate_composite_score(self, scores: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate weighted composite score from all individual scores
        
        Returns:
            Dictionary with composite score and breakdown
        """
        total_weight = 0
        weighted_sum = 0
        score_breakdown = []
        
        for metric, score_data in scores.items():
            weight = self.weights.get(metric, 0)
            score = score_data.get("score", 50)
            
            weighted_sum += score * weight
            total_weight += weight
            
            score_breakdown.append({
                "metric": metric,
                "score": score,
                "weight": weight,
                "weighted_contribution": round(score * weight, 2),
                "reasoning": score_data.get("reasoning", "")
            })
        
        composite = weighted_sum / total_weight if total_weight > 0 else 50
        
        # Determine direction
        if composite >= 60:
            direction = "bullish"
        elif composite >= 40:
            direction = "neutral"
        else:
            direction = "bearish"
        
        return {
            "score": round(composite, 1),
            "direction": direction,
            "breakdown": score_breakdown,
            "total_weight": total_weight
        }

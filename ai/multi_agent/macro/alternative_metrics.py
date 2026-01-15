"""
Alternative Macroeconomic Metrics Analyzer
Analyzes less mainstream but valuable economic indicators
"""

from typing import Dict, Any, List
import asyncio
import aiohttp
from datetime import datetime


class AlternativeMetricsAnalyzer:
    """
    Analyzes alternative macroeconomic indicators:
    - Yield Curve (2Y-10Y spread)
    - M2 Money Supply Growth
    - MOVE Index (Bond Volatility)
    - High Yield Credit Spreads
    - Leading Economic Index (LEI)
    - Sahm Rule (Recession Indicator)
    - Dollar Index (DXY)
    - Architecture Billings Index (ABI)
    - Copper Prices (Dr. Copper)
    - Luxury Item Sales
    - Gold/Treasury Ratio
    - Corporate Debt to GDP
    """
    
    def __init__(self, api_keys: Dict[str, str], weights: Dict[str, float], analysis_date: str = None):
        """
        Initialize Alternative Metrics Analyzer
        
        Args:
            api_keys: Dictionary with 'fred' and 'alpha_vantage' keys
            weights: Dictionary of metric weights from config
            analysis_date: Optional date for historical analysis (YYYY-MM-DD)
        """
        self.fred_api_key = api_keys.get("fred")
        self.alpha_vantage_key = api_keys.get("alpha_vantage")
        self.weights = weights
        self.analysis_date = analysis_date
        
        if not self.fred_api_key:
            raise ValueError("FRED API key is required for alternative metrics")
        if not self.alpha_vantage_key:
            raise ValueError("Alpha Vantage API key is required for alternative metrics")
    
    async def analyze(self) -> Dict[str, Any]:
        """
        Perform full analysis of alternative metrics
        
        Returns:
            Dictionary with scores, data, and overall assessment
        """
        print("[Alternative Metrics] Starting analysis...")

        # Fetch FRED-backed metrics in parallel (fast, lenient rate limits)
        fred_tasks = [
            self._fetch_yield_curve(),
            self._fetch_m2_money_supply(),
            self._fetch_high_yield_spreads(),
            self._fetch_lei(),
            self._fetch_sahm_rule(),
            self._fetch_dollar_index(),
            self._fetch_abi(),
            self._fetch_corporate_debt()
        ]

        (yield_curve_data, m2_data, hy_spreads_data, lei_data, sahm_data,
         dxy_data, abi_data, corp_debt_data) = await asyncio.gather(*fred_tasks, return_exceptions=True)

        # Fetch Alpha Vantage-backed metrics sequentially to respect rate limits
        copper_data = await self._fetch_copper()
        await asyncio.sleep(15)  # Respect free-tier pacing
        luxury_data = await self._fetch_luxury_sales()
        await asyncio.sleep(15)
        gold_ratio_data = await self._fetch_gold_ratio()
        
        # Calculate individual scores
        scores = {
            "yield_curve": self._score_yield_curve(yield_curve_data),
            "m2_growth": self._score_m2(m2_data),
            "high_yield_spreads": self._score_hy_spreads(hy_spreads_data),
            "leading_economic_index": self._score_lei(lei_data),
            "sahm_rule": self._score_sahm(sahm_data),
            "dollar_index": self._score_dollar(dxy_data),
            "building_permits": self._score_building_permits(abi_data),
            "copper_prices": self._score_copper(copper_data),
            "luxury_sales": self._score_luxury(luxury_data),
            "gold_treasury_ratio": self._score_gold_ratio(gold_ratio_data),
            "corporate_debt_gdp": self._score_corp_debt(corp_debt_data)
        }
        
        # Calculate weighted composite score
        composite_score = self._calculate_composite_score(scores)
        
        return {
            "composite_score": composite_score,
            "individual_scores": scores,
            "raw_data": {
                "yield_curve": yield_curve_data,
                "m2_money_supply": m2_data,
                "high_yield_spreads": hy_spreads_data,
                "lei": lei_data,
                "sahm_rule": sahm_data,
                "dollar_index": dxy_data,
                "abi": abi_data,  # Now building permits
                "copper": copper_data,
                "luxury_sales": luxury_data,
                "gold_ratio": gold_ratio_data,
                "corporate_debt": corp_debt_data
            },
            "timestamp": datetime.now().isoformat()
        }
    
    async def _fetch_fred_data(self, series_id: str, name: str) -> Dict[str, Any]:
        """Fetch data from FRED API with a single retry for transient issues"""
        url = (
            f"https://api.stlouisfed.org/fred/series/observations?"
            f"series_id={series_id}&"
            f"api_key={self.fred_api_key}&"
            f"file_type=json&"
            f"sort_order=desc&"
            f"limit=24"
        )
        if self.analysis_date:
            url += f"&observation_end={self.analysis_date}"
        last_err = None
        for attempt in range(2):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(url) as response:
                        if response.status == 200:
                            data = await response.json()
                            observations = data.get("observations", [])
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
                        last_err = RuntimeError(f"FRED API error: {response.status}")
            except Exception as e:
                last_err = e
            await asyncio.sleep(1)
        print(f"[Alternative Metrics] Error fetching {series_id}: {last_err}")
        raise last_err

    async def _fetch_fred_candidates(self, series_ids: List[str], name: str) -> Dict[str, Any]:
        """Try multiple FRED series IDs until one succeeds."""
        last_err = None
        for sid in series_ids:
            try:
                return await self._fetch_fred_data(sid, name)
            except Exception as e:
                last_err = e
                continue
        # Fallback: try FRED series search by name
        search_id = await self._search_fred_series(name)
        if search_id:
            try:
                return await self._fetch_fred_data(search_id, name)
            except Exception as e:
                last_err = e
        # Graceful fallback: return unavailable placeholder without raising
        return {"name": name, "unavailable": True, "error": str(last_err) if last_err else "Series not found"}

    async def _search_fred_series(self, search_text: str) -> str | None:
        """Search FRED for a series ID by text and return the best match."""
        url = (
            f"https://api.stlouisfed.org/fred/series/search?"
            f"search_text={search_text}&"
            f"api_key={self.fred_api_key}&"
            f"file_type=json"
        )
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        series = data.get("seriess", [])
                        # Prefer monthly frequency and titles containing key words
                        def score(item):
                            title = item.get("title", "").lower()
                            freq = item.get("frequency_short", "").lower()
                            s = 0
                            if "architecture billings" in title or "abi" in title:
                                s += 3
                            if freq == "m":
                                s += 1
                            return s
                        if series:
                            best = sorted(series, key=score, reverse=True)[0]
                            return best.get("id")
        except Exception:
            return None
        return None
    
    async def _fetch_yield_curve(self) -> Dict[str, Any]:
        """
        Fetch Yield Curve (10Y - 2Y spread)
        Inverted curve = recession warning
        """
        try:
            # Fetch both yields in parallel
            url_10y = (
                f"https://api.stlouisfed.org/fred/series/observations?"
                f"series_id=DGS10&api_key={self.fred_api_key}&"
                f"file_type=json&sort_order=desc&limit=5"
            )
            url_2y = (
                f"https://api.stlouisfed.org/fred/series/observations?"
                f"series_id=DGS2&api_key={self.fred_api_key}&"
                f"file_type=json&sort_order=desc&limit=5"
            )
            
            async with aiohttp.ClientSession() as session:
                resp_10y, resp_2y = await asyncio.gather(
                    session.get(url_10y),
                    session.get(url_2y)
                )
                
                data_10y = await resp_10y.json()
                data_2y = await resp_2y.json()
                
                obs_10y = [o for o in data_10y.get("observations", []) if o["value"] != "."]
                obs_2y = [o for o in data_2y.get("observations", []) if o["value"] != "."]
                
                if not obs_10y or not obs_2y:
                    raise ValueError("Missing yield curve data")
                
                y10 = float(obs_10y[0]["value"])
                y2 = float(obs_2y[0]["value"])
                spread = y10 - y2
                
                return {
                    "name": "Yield Curve Spread (10Y-2Y)",
                    "current": spread,
                    "10y_yield": y10,
                    "2y_yield": y2,
                    "inverted": spread < 0,
                    "date": obs_10y[0]["date"]
                }
        except Exception as e:
            print(f"[Alternative Metrics] Error fetching yield curve: {e}")
            raise
    
    async def _fetch_m2_money_supply(self) -> Dict[str, Any]:
        """Fetch M2 Money Supply"""
        return await self._fetch_fred_data("M2SL", "M2 Money Supply")
    
    async def _fetch_high_yield_spreads(self) -> Dict[str, Any]:
        """Fetch High Yield Credit Spreads (ICE BofA)"""
        return await self._fetch_fred_data("BAMLH0A0HYM2", "High Yield Spreads")
    
    async def _fetch_lei(self) -> Dict[str, Any]:
        """Fetch Leading Economic Index"""
        return await self._fetch_fred_data("USSLIND", "Leading Economic Index")
    
    async def _fetch_sahm_rule(self) -> Dict[str, Any]:
        """
        Fetch Sahm Rule Recession Indicator
        Value >= 0.5 signals recession
        """
        return await self._fetch_fred_data("SAHMREALTIME", "Sahm Rule Indicator")
    
    async def _fetch_dollar_index(self) -> Dict[str, Any]:
        """Fetch Dollar Index (DXY via FRED)"""
        return await self._fetch_fred_data("DTWEXBGS", "Trade Weighted Dollar Index")
    
    async def _fetch_abi(self) -> Dict[str, Any]:
        """
        Fetch Building Permits (ABI alternative)
        Leading indicator for construction (6-9 months ahead)
        Rising permits = economic expansion
        """
        # Building permits are a reliable leading indicator for construction
        return await self._fetch_fred_data("PERMIT", "Building Permits")
    
    async def _fetch_copper(self) -> Dict[str, Any]:
        """Fetch Copper Prices (Alpha Vantage) with rate-limit retry"""
        url = (
            f"https://www.alphavantage.co/query?"
            f"function=TIME_SERIES_DAILY&"
            f"symbol=CPER&"
            f"apikey={self.alpha_vantage_key}"
        )
        last_err = None
        for attempt in range(3):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(url) as response:
                        if response.status == 200:
                            data = await response.json()
                            daily_data = data.get("Time Series (Daily)", {})
                            if not daily_data:
                                raise ValueError("No copper data available")
                            dates = sorted(daily_data.keys(), reverse=True)
                            current_price = float(daily_data[dates[0]]["4. close"])
                            month_ago = float(daily_data[dates[21]]["4. close"]) if len(dates) > 21 else current_price
                            three_month = float(daily_data[dates[63]]["4. close"]) if len(dates) > 63 else current_price
                            return {
                                "name": "Copper Prices",
                                "current": current_price,
                                "1m_change_pct": ((current_price - month_ago) / month_ago * 100),
                                "3m_change_pct": ((current_price - three_month) / three_month * 100),
                                "trend": "increasing" if current_price > month_ago else "decreasing",
                                "date": dates[0]
                            }
                        if response.status == 429:
                            last_err = RuntimeError("Alpha Vantage rate limit hit (429)")
                            await asyncio.sleep(15)
                            continue
                        last_err = RuntimeError(f"Alpha Vantage API error: {response.status}")
            except Exception as e:
                last_err = e
            await asyncio.sleep(2)
        print(f"[Alternative Metrics] Error fetching copper: {last_err}")
        raise last_err
    
    async def _fetch_luxury_sales(self) -> Dict[str, Any]:
        """Fetch luxury goods proxy (Alpha Vantage) with rate-limit retry"""
        url = (
            f"https://www.alphavantage.co/query?"
            f"function=TIME_SERIES_DAILY&"
            f"symbol=RL&"
            f"apikey={self.alpha_vantage_key}"
        )
        last_err = None
        for attempt in range(3):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(url) as response:
                        if response.status == 200:
                            data = await response.json()
                            daily_data = data.get("Time Series (Daily)", {})
                            if not daily_data:
                                raise ValueError("No luxury sales data available")
                            dates = sorted(daily_data.keys(), reverse=True)
                            current_price = float(daily_data[dates[0]]["4. close"])
                            three_month = float(daily_data[dates[63]]["4. close"]) if len(dates) > 63 else current_price
                            return {
                                "name": "Luxury Sales Indicator",
                                "current": current_price,
                                "3m_change_pct": ((current_price - three_month) / three_month * 100),
                                "trend": "increasing" if current_price > three_month else "decreasing",
                                "date": dates[0]
                            }
                        if response.status == 429:
                            last_err = RuntimeError("Alpha Vantage rate limit hit (429)")
                            await asyncio.sleep(15)
                            continue
                        last_err = RuntimeError(f"Alpha Vantage API error: {response.status}")
            except Exception as e:
                last_err = e
            await asyncio.sleep(2)
        print(f"[Alternative Metrics] Error fetching luxury sales: {last_err}")
        raise last_err
    
    async def _fetch_gold_ratio(self) -> Dict[str, Any]:
        """Fetch Gold trend (Alpha Vantage) with rate-limit retry"""
        url = (
            f"https://www.alphavantage.co/query?"
            f"function=TIME_SERIES_DAILY&"
            f"symbol=GLD&"
            f"apikey={self.alpha_vantage_key}"
        )
        last_err = None
        for attempt in range(3):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(url) as response:
                        if response.status == 200:
                            data = await response.json()
                            daily_data = data.get("Time Series (Daily)", {})
                            if not daily_data:
                                raise ValueError("No gold data available")
                            dates = sorted(daily_data.keys(), reverse=True)
                            current_price = float(daily_data[dates[0]]["4. close"])
                            month_ago = float(daily_data[dates[21]]["4. close"]) if len(dates) > 21 else current_price
                            return {
                                "name": "Gold Price Trend",
                                "current": current_price,
                                "1m_change_pct": ((current_price - month_ago) / month_ago * 100),
                                "trend": "increasing" if current_price > month_ago else "decreasing",
                                "date": dates[0]
                            }
                        if response.status == 429:
                            last_err = RuntimeError("Alpha Vantage rate limit hit (429)")
                            await asyncio.sleep(15)
                            continue
                        last_err = RuntimeError(f"Alpha Vantage API error: {response.status}")
            except Exception as e:
                last_err = e
            await asyncio.sleep(2)
        print(f"[Alternative Metrics] Error fetching gold: {last_err}")
        raise last_err
    
    async def _fetch_corporate_debt(self) -> Dict[str, Any]:
        """Fetch Corporate Debt to GDP ratio"""
        return await self._fetch_fred_data("NCBDBIQ027S", "Corporate Debt to GDP")
    
    def _score_yield_curve(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Score Yield Curve
        - Steep curve (>1%): Very positive (70-100)
        - Normal (0.5-1%): Positive (55-70)
        - Flat (0-0.5%): Neutral (45-55)
        - Inverted (<0): Negative (0-45) - recession warning
        """
        if isinstance(data, Exception):
            return {"score": 50, "reasoning": "Data unavailable", "error": str(data)}
        
        spread = data.get("current", 0)
        
        if spread > 1.0:
            score = 70 + min(30, (spread - 1.0) * 15)
        elif spread > 0.5:
            score = 55 + (spread - 0.5) * 30
        elif spread > 0:
            score = 45 + spread * 20
        else:  # Inverted
            score = max(0, 45 + spread * 30)  # Heavily penalize inversion
        
        status = "inverted" if spread < 0 else "flat" if spread < 0.5 else "normal"
        
        return {
            "score": round(score, 1),
            "reasoning": f"Yield curve spread at {spread:.2f}% ({status})",
            "data": data
        }
    
    def _score_m2(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Score M2 Money Supply Growth
        - 5-8%: Optimal (70-100)
        - 3-5% or 8-12%: Acceptable (50-70)
        - <3% or >12%: Concerning (0-50)
        """
        if isinstance(data, Exception):
            return {"score": 50, "reasoning": "Data unavailable", "error": str(data)}
        
        growth = data.get("yoy_change_pct", 5)
        
        if 5 <= growth <= 8:
            score = 70 + min(30, (7 - abs(growth - 6.5)) * 10)
        elif 3 <= growth < 5 or 8 < growth <= 12:
            if growth < 5:
                score = 50 + (growth - 3) * 10
            else:
                score = 50 + (12 - growth) * 5
        else:
            if growth < 3:
                score = max(20, 50 + (growth - 3) * 10)
            else:
                score = max(10, 50 - (growth - 12) * 5)
        
        return {
            "score": round(score, 1),
            "reasoning": f"M2 growth at {growth:.1f}% YoY",
            "data": data
        }
    
    def _score_hy_spreads(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Score High Yield Spreads
        - <3%: Low risk, bullish (70-100)
        - 3-5%: Normal (50-70)
        - 5-8%: Elevated (30-50)
        - >8%: Stress (0-30)
        """
        if isinstance(data, Exception):
            return {"score": 50, "reasoning": "Data unavailable", "error": str(data)}
        
        spread = data.get("current", 4)
        trend = data.get("trend", "stable")
        
        if spread < 3:
            score = 70 + (3 - spread) * 10
        elif spread < 5:
            score = 50 + (5 - spread) * 10
        elif spread < 8:
            score = 30 + (8 - spread) * 6.7
        else:
            score = max(0, 30 - (spread - 8) * 5)
        
        # Widening spreads = deteriorating conditions
        if trend == "increasing":
            score -= 10
        elif trend == "decreasing":
            score += 5
        
        return {
            "score": round(max(0, min(100, score)), 1),
            "reasoning": f"HY spreads at {spread:.2f}%, {trend}",
            "data": data
        }
    
    def _score_lei(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Score Leading Economic Index
        Rising = expansion ahead, Falling = contraction
        """
        if isinstance(data, Exception):
            return {"score": 50, "reasoning": "Data unavailable", "error": str(data)}
        
        yoy_change = data.get("yoy_change_pct", 0)
        trend = data.get("trend", "stable")
        
        if yoy_change > 3:
            score = 70 + min(30, (yoy_change - 3) * 7.5)
        elif yoy_change > 0:
            score = 50 + yoy_change * 6.7
        else:
            score = max(10, 50 + yoy_change * 10)
        
        if trend == "increasing":
            score += 5
        elif trend == "decreasing":
            score -= 5
        
        return {
            "score": round(max(0, min(100, score)), 1),
            "reasoning": f"LEI change {yoy_change:.1f}% YoY, {trend}",
            "data": data
        }
    
    def _score_sahm(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Score Sahm Rule
        - <0.3: Safe (70-100)
        - 0.3-0.5: Caution (50-70)
        - >=0.5: Recession signal (0-50)
        """
        if isinstance(data, Exception):
            return {"score": 50, "reasoning": "Data unavailable", "error": str(data)}
        
        value = data.get("current", 0)
        
        if value < 0.3:
            score = 70 + (0.3 - value) * 100
        elif value < 0.5:
            score = 50 + (0.5 - value) * 100
        else:
            score = max(0, 50 - (value - 0.5) * 50)
        
        status = "recession signal" if value >= 0.5 else "elevated" if value >= 0.3 else "safe"
        
        return {
            "score": round(score, 1),
            "reasoning": f"Sahm Rule at {value:.2f} ({status})",
            "data": data
        }
    
    def _score_dollar(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Score Dollar Index
        Strong dollar can hurt US multinationals and commodities
        """
        if isinstance(data, Exception):
            return {"score": 50, "reasoning": "Data unavailable", "error": str(data)}
        
        yoy_change = data.get("yoy_change_pct", 0)
        
        # Moderate dollar strength is neutral
        # Very strong dollar (>5% YoY) = headwind for stocks
        # Weak dollar (<-5% YoY) = supportive but inflation risk
        
        if -2 <= yoy_change <= 2:
            score = 60 + (2 - abs(yoy_change)) * 10
        elif yoy_change > 2:
            score = max(30, 60 - (yoy_change - 2) * 10)
        else:  # yoy_change < -2
            score = 50 + (yoy_change + 2) * 5
        
        return {
            "score": round(score, 1),
            "reasoning": f"Dollar index {yoy_change:+.1f}% YoY",
            "data": data
        }
    
    def _score_building_permits(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Score Building Permits
        - Strong growth (>10% YoY): Strong expansion (70-100)
        - Moderate growth (0-10%): Expansion (50-70)
        - Decline (<0%): Contraction (0-50)
        """
        if isinstance(data, Exception):
            return {"score": 50, "reasoning": "Data unavailable", "error": str(data)}
        
        yoy_change = data.get("yoy_change_pct", 0)
        current = data.get("current", 0)
        
        if yoy_change > 10:
            score = 70 + min(30, (yoy_change - 10) * 3)
        elif yoy_change > 0:
            score = 50 + yoy_change * 2
        else:
            score = max(10, 50 + yoy_change * 2.5)
        
        return {
            "score": round(score, 1),
            "reasoning": f"Building permits {yoy_change:+.1f}% YoY ({'expansion' if yoy_change >= 0 else 'contraction'})",
            "data": data
        }
    
    def _score_abi(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Score Architecture Billings Index
        - >52: Strong expansion (70-100)
        - 50-52: Expansion (55-70)
        - 48-50: Neutral (45-55)
        - <48: Contraction (0-45)
        """
        if isinstance(data, Exception) or data.get("unavailable"):
            return {"score": 50, "reasoning": "ABI data unavailable", "data": data}
        
        abi = data.get("current", 50)
        
        if abi > 52:
            score = 70 + min(30, (abi - 52) * 7.5)
        elif abi > 50:
            score = 55 + (abi - 50) * 7.5
        elif abi > 48:
            score = 45 + (abi - 48) * 5
        else:
            score = max(10, abi * 0.94)
        
        return {
            "score": round(score, 1),
            "reasoning": f"ABI at {abi:.1f} ({'expansion' if abi >= 50 else 'contraction'})",
            "data": data
        }
    
    def _score_copper(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Score Copper Prices (Dr. Copper)
        Rising copper = strong global growth expectations
        """
        if isinstance(data, Exception):
            return {"score": 50, "reasoning": "Data unavailable", "error": str(data)}
        
        three_month_change = data.get("3m_change_pct", 0)
        
        if three_month_change > 10:
            score = 75 + min(25, (three_month_change - 10) * 2.5)
        elif three_month_change > 0:
            score = 50 + three_month_change * 2.5
        else:
            score = max(15, 50 + three_month_change * 2.5)
        
        return {
            "score": round(score, 1),
            "reasoning": f"Copper {three_month_change:+.1f}% over 3 months",
            "data": data
        }
    
    def _score_luxury(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Score Luxury Sales
        Strong luxury sales = healthy high-end consumer
        """
        if isinstance(data, Exception):
            return {"score": 50, "reasoning": "Data unavailable", "error": str(data)}
        
        three_month_change = data.get("3m_change_pct", 0)
        
        if three_month_change > 8:
            score = 70 + min(30, (three_month_change - 8) * 3)
        elif three_month_change > 0:
            score = 50 + three_month_change * 2.5
        else:
            score = max(20, 50 + three_month_change * 3)
        
        return {
            "score": round(score, 1),
            "reasoning": f"Luxury sales {three_month_change:+.1f}% over 3 months",
            "data": data
        }
    
    def _score_gold_ratio(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Score Gold trend
        Rising gold = fear/uncertainty (negative for stocks)
        """
        if isinstance(data, Exception):
            return {"score": 50, "reasoning": "Data unavailable", "error": str(data)}
        
        month_change = data.get("1m_change_pct", 0)
        
        # Inverse relationship: falling gold = positive for risk assets
        if month_change < -3:
            score = 70 + min(30, abs(month_change + 3) * 7.5)
        elif month_change < 0:
            score = 55 + abs(month_change) * 5
        elif month_change < 5:
            score = 55 - month_change * 5
        else:
            score = max(20, 30 - (month_change - 5) * 2)
        
        return {
            "score": round(score, 1),
            "reasoning": f"Gold {month_change:+.1f}% over 1 month",
            "data": data
        }
    
    def _score_corp_debt(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Score Corporate Debt to GDP
        - <70%: Healthy (70-100)
        - 70-80%: Acceptable (50-70)
        - >80%: Elevated (0-50)
        """
        if isinstance(data, Exception):
            return {"score": 50, "reasoning": "Data unavailable", "error": str(data)}
        
        ratio = data.get("current", 75)
        
        if ratio < 70:
            score = 70 + (70 - ratio) * 0.5
        elif ratio < 80:
            score = 50 + (80 - ratio) * 2
        else:
            score = max(20, 50 - (ratio - 80) * 2)
        
        return {
            "score": round(score, 1),
            "reasoning": f"Corporate debt at {ratio:.1f}% of GDP",
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

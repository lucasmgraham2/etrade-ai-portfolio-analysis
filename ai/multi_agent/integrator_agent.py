"""
Integrator Agent
Combines all agent outputs with portfolio data to generate actionable recommendations
"""

from typing import Dict, Any, List
from datetime import datetime
import json
from .base_agent import BaseAgent


class IntegratorAgent(BaseAgent):
    """
    Synthesizes outputs from all agents to provide:
    - Portfolio rebalancing recommendations
    - Buy/sell/hold decisions for each position
    - Risk assessment
    - Action priorities
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("Integrator", config)
        self.risk_tolerance = config.get("risk_tolerance", "moderate") if config else "moderate"
        
    async def analyze(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Integrate all analysis and generate recommendations
        
        Args:
            context: Contains portfolio and all agent outputs
            
        Returns:
            Integrated analysis with actionable recommendations
        """
        try:
            self.log("Integrating multi-agent analysis...")
            
            # Validate all required data is present
            required_keys = ["portfolio", "Sentiment", "Sector"]
            if not self.validate_context(context, required_keys):
                return {"error": "Missing required agent outputs (portfolio, sentiment, sector)"}
            
            # Extract data from context
            portfolio = context["portfolio"]
            
            # Get results from other agents - handle potential errors
            sentiment_results = context.get("Sentiment", {})
            if not sentiment_results or "error" in sentiment_results:
                return {"error": "Sentiment analysis failed or returned no data"}
            
            # Macro is optional - if it fails, use defaults
            macro_results = context.get("Macro", {})
            if not macro_results or "error" in macro_results:
                self.log("Macro analysis not available, using neutral defaults", "WARNING")
                macro_results = {
                    "confidence_score": {
                        "score": 50,
                        "direction": "NEUTRAL",
                        "interpretation": "Macro data unavailable - using neutral stance",
                        "crisis_detection": {"flags": {}}
                    },
                    "summary": "Macro analysis unavailable"
                }
            
            sector_results = context.get("Sector", {})
            if not sector_results or "error" in sector_results:
                return {"error": "Sector analysis failed or returned no data"}
            
            # Get portfolio positions
            positions = self._extract_positions(portfolio)
            
            # Analyze each position
            position_analyses = self._analyze_positions(
                positions, sentiment_results, macro_results, sector_results
            )
            
            # Generate portfolio-level recommendations
            portfolio_recommendations = self._generate_portfolio_recommendations(
                portfolio, position_analyses, macro_results, sector_results
            )
            
            # Prioritize actions
            action_priorities = self._prioritize_actions(position_analyses, portfolio_recommendations)
            
            # Calculate risk assessment
            risk_assessment = self._assess_portfolio_risk(
                portfolio, position_analyses, macro_results
            )

            # Bucket recommendations for quick view
            recommendation_buckets = {"buy": [], "hold": [], "sell": []}
            for p in position_analyses:
                rec = p.get("recommendation")
                symbol = p.get("symbol")
                entry = f"{symbol} ({rec})"
                if rec in ["STRONG_BUY", "BUY"]:
                    recommendation_buckets["buy"].append(entry)
                elif rec in ["HOLD"]:
                    recommendation_buckets["hold"].append(entry)
                elif rec in ["SELL", "TAKE_PROFIT", "CUT_LOSS", "STRONG_SELL"]:
                    recommendation_buckets["sell"].append(entry)
            
            portfolio_recommendations["recommendation_buckets"] = recommendation_buckets
            
            # Generate executive summary
            executive_summary = self._create_executive_summary(
                portfolio, position_analyses, portfolio_recommendations,
                macro_results, sentiment_results, sector_results
            )
            
            results = {
                "summary": executive_summary,
                "position_analyses": position_analyses,
                "portfolio_recommendations": portfolio_recommendations,
                "action_priorities": action_priorities,
                "risk_assessment": risk_assessment,
                "recommendation_buckets": recommendation_buckets,
                "recommendations": [
                    f"[Priority {a['priority']}] {a['action']}" + (f" — {a['rationale']}" if a.get('rationale') else "")
                    for a in action_priorities
                ],
                "timestamp": datetime.now().isoformat()
            }
            
            # Generate AI reasoning
            ai_reasoning = await self._generate_integrator_ai_reasoning(
                results, position_analyses, macro_results, sentiment_results, sector_results
            )
            results["ai_reasoning"] = ai_reasoning
            
            # Generate new position suggestions
            new_position_suggestions = await self._suggest_new_positions(
                portfolio,
                position_analyses,
                macro_results,
                sentiment_results,
                sector_results,
                risk_assessment
            )
            results["new_position_suggestions"] = new_position_suggestions
            
            return results
        except Exception as e:
            import traceback
            self.log(f"Integrator error: {str(e)}", "ERROR")
            self.log(f"Traceback: {traceback.format_exc()}", "DEBUG")
            return {"error": f"Integration failed: {str(e)}"}
    
    
    def _extract_positions(self, portfolio: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract all positions from portfolio"""
        positions = []
        
        for account in portfolio.get("accounts", []):
            for position in account.get("positions", []):
                # Add account info to position
                position_copy = position.copy()
                position_copy["account_id"] = account.get("account_id")
                position_copy["account_type"] = account.get("description", "Unknown")
                positions.append(position_copy)
        
        return positions
    
    def _analyze_positions(
        self,
        positions: List[Dict[str, Any]],
        sentiment: Dict[str, Any],
        macro: Dict[str, Any],
        sector: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Analyze each position with all agent insights"""
        
        analyses = []
        
        # Get sentiment data
        overall_sentiment = sentiment.get("overall_sentiment", {})
        
        # Get macro data
        confidence_score = macro.get("confidence_score", {})
        macro_score = confidence_score.get("score", 50) if isinstance(confidence_score, dict) else 50
        
        # Get sector predictions (handle case where it might not be a list)
        sector_predictions = sector.get("sector_predictions", [])
        if not isinstance(sector_predictions, list):
            sector_predictions = []
        sector_dict = {p["sector"]: p for p in sector_predictions if isinstance(p, dict) and "sector" in p}
        
        # Get portfolio sectors
        portfolio_sectors = sector.get("portfolio_sectors", {})
        
        for position in positions:
            symbol = position.get("symbol")
            
            # Get position metrics
            total_gain_pct = position.get("total_gain_pct", 0)
            market_value = position.get("market_value", 0)
            quantity = position.get("quantity", 0)
            
            # Get sentiment for this symbol
            symbol_sentiment = overall_sentiment.get(symbol, {})
            sentiment_score = symbol_sentiment.get("overall_score") or 0  # Handle None values
            sentiment_label = symbol_sentiment.get("sentiment", "neutral")
            
            # Determine sector
            position_sector = self._get_position_sector(symbol, portfolio_sectors)
            
            # Get sector prediction
            sector_pred = sector_dict.get(position_sector, {})
            sector_outlook = sector_pred.get("outlook", "neutral")
            sector_score = sector_pred.get("prediction_score", 0)
            
            # Calculate composite score
            composite_score = self._calculate_composite_score(
                sentiment_score, macro_score, sector_score, total_gain_pct
            )
            
            # Generate recommendation
            recommendation = self._generate_position_recommendation(
                composite_score, total_gain_pct, sentiment_label, 
                sector_outlook, macro_score
            )
            
            # Calculate confidence
            confidence = self._calculate_recommendation_confidence(
                sentiment_score, macro_score, sector_score
            )
            
            analysis = {
                "symbol": symbol,
                "account_type": position.get("account_type"),
                "market_value": market_value,
                "quantity": quantity,
                "total_gain_pct": total_gain_pct,
                "sector": position_sector,
                "sentiment": {
                    "score": sentiment_score,
                    "label": sentiment_label
                },
                "sector_outlook": sector_outlook,
                "sector_score": sector_score,
                "macro_favorability": macro_score,
                "composite_score": composite_score,
                "recommendation": recommendation,
                "confidence": confidence,
                "rationale": self._create_rationale(
                    symbol, recommendation, sentiment_label, 
                    sector_outlook, macro_score, total_gain_pct
                )
            }
            
            analyses.append(analysis)
        
        # Sort by composite score (best opportunities first)
        analyses.sort(key=lambda x: x["composite_score"], reverse=True)
        
        return analyses
    
    def _get_position_sector(self, symbol: str, portfolio_sectors: Dict[str, Any]) -> str:
        """Find which sector a symbol belongs to"""
        for sector, value in portfolio_sectors.items():
            # Handle both dict (sector: {symbols: [...]}) and list/set structures
            if isinstance(value, dict) and "symbols" in value:
                if symbol in value.get("symbols", []):
                    return sector
            elif isinstance(value, (list, set, tuple)):
                if symbol in value:
                    return sector
            # Skip non-iterable values (floats, etc.)
        return "Other"
    
    def _calculate_composite_score(
        self,
        sentiment_score: float,
        macro_score: float,
        sector_score: float,
        current_gain_pct: float
    ) -> float:
        """
        Calculate composite score for position
        
        Weights:
        - Sentiment: 25%
        - Macro: 20%
        - Sector: 30%
        - Current performance: 25%
        """
        # Normalize macro score to -1 to 1 range
        macro_normalized = (macro_score - 50) / 50
        
        # Normalize current gain to -1 to 1 range (capped at ±50%)
        gain_normalized = max(-1, min(1, current_gain_pct / 50))
        
        # Calculate weighted score
        composite = (
            sentiment_score * 0.25 +
            macro_normalized * 0.20 +
            (sector_score / 10) * 0.30 +  # Sector scores typically range -20 to 20
            gain_normalized * 0.25
        )
        
        # Scale to 0-100
        scaled_score = (composite + 1) * 50
        
        return round(scaled_score, 2)
    
    def _generate_position_recommendation(
        self,
        composite_score: float,
        current_gain_pct: float,
        sentiment: str,
        sector_outlook: str,
        macro_score: float
    ) -> str:
        """Generate recommendation for a position"""
        
        # Strong buy signals
        if composite_score > 70 and sentiment == "bullish" and sector_outlook == "outperform":
            return "STRONG_BUY"
        
        # Buy signals
        if composite_score > 60:
            return "BUY"
        
        # Hold signals
        if 40 <= composite_score <= 60:
            return "HOLD"
        
        # Sell signals
        if composite_score < 40:
            # Check if taking profits or cutting losses
            if current_gain_pct > 15:
                return "TAKE_PROFIT"
            elif current_gain_pct < -15:
                return "CUT_LOSS"
            else:
                return "SELL"
        
        # Strong sell signals
        if composite_score < 30 and sentiment == "bearish" and sector_outlook == "underperform":
            return "STRONG_SELL"
        
        return "HOLD"
    
    def _calculate_recommendation_confidence(
        self, sentiment_score: float, macro_score: float, sector_score: float
    ) -> float:
        """Calculate confidence in recommendation"""
        
        # Confidence is higher when signals align
        signals = []
        
        # Sentiment signal - handle None/0 scores
        if sentiment_score and abs(sentiment_score) > 0.3:
            signals.append(abs(sentiment_score))
        
        # Macro signal
        macro_strength = abs(macro_score - 50) / 50
        if macro_strength > 0.2:
            signals.append(macro_strength)
        
        # Sector signal
        sector_strength = abs(sector_score) / 20
        if sector_strength > 0.2:
            signals.append(sector_strength)
        
        if not signals:
            return 0.5
        
        # Average signal strength
        confidence = sum(signals) / len(signals)
        
        # Boost confidence if all signals agree
        if len(signals) == 3:
            confidence *= 1.2
        
        return round(min(1.0, confidence), 2)
    
    def _create_rationale(
        self,
        symbol: str,
        recommendation: str,
        sentiment: str,
        sector_outlook: str,
        macro_score: float,
        gain_pct: float
    ) -> str:
        """Create human-readable rationale for recommendation"""
        
        reasons = []
        
        # Sentiment reason
        if sentiment == "bullish":
            reasons.append("positive market sentiment")
        elif sentiment == "bearish":
            reasons.append("negative market sentiment")
        
        # Sector reason
        if sector_outlook == "outperform":
            reasons.append("sector predicted to outperform")
        elif sector_outlook == "underperform":
            reasons.append("sector predicted to underperform")
        
        # Macro reason
        if macro_score > 70:
            reasons.append("very favorable macro environment")
        elif macro_score > 55:
            reasons.append("favorable macro environment")
        elif macro_score < 30:
            reasons.append("unfavorable macro environment")
        elif macro_score < 45:
            reasons.append("challenging macro environment")
        
        # Performance reason
        if gain_pct > 20:
            reasons.append(f"strong gains of {gain_pct:.1f}%")
        elif gain_pct < -20:
            reasons.append(f"significant losses of {gain_pct:.1f}%")
        
        if reasons:
            return f"{recommendation} based on: " + ", ".join(reasons)
        else:
            return f"{recommendation} - mixed signals suggest neutral stance"

    def _extract_sector_highlights(
        self,
        sector_predictions: List[Dict[str, Any]],
        allocation: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Summarize sector leaders/laggards and allocation gaps."""

        # Sort by strength for clear top/bottom groupings
        outperformers = [p for p in sector_predictions if p.get("outlook") == "outperform"]
        underperformers = [p for p in sector_predictions if p.get("outlook") == "underperform"]
        outperformers.sort(key=lambda x: x.get("prediction_score", 0), reverse=True)
        underperformers.sort(key=lambda x: x.get("prediction_score", 0))

        top_favorable = [
            {
                "sector": p.get("sector", "Unknown"),
                "score": p.get("prediction_score", 0),
                "confidence": p.get("confidence", 0)
            }
            for p in outperformers[:3]
        ]

        least_favorable = [
            {
                "sector": p.get("sector", "Unknown"),
                "score": p.get("prediction_score", 0),
                "confidence": p.get("confidence", 0)
            }
            for p in underperformers[:3]
        ]

        return {
            "top_favorable": top_favorable,
            "least_favorable": least_favorable,
            "underweight": allocation.get("underweight_sectors", []),
            "overweight": allocation.get("overweight_sectors", [])
        }
    
    def _generate_portfolio_recommendations(
        self,
        portfolio: Dict[str, Any],
        position_analyses: List[Dict[str, Any]],
        macro: Dict[str, Any],
        sector: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate portfolio-level recommendations"""
        
        total_value = portfolio.get("summary", {}).get("total_portfolio_value", 0)
        total_cash = portfolio.get("summary", {}).get("total_cash", 0)
        cash_pct = (total_cash / total_value * 100) if total_value > 0 else 0
        
        # Count recommendations
        strong_sells = sum(1 for p in position_analyses if p["recommendation"] in ["STRONG_SELL", "CUT_LOSS"])
        sells = sum(1 for p in position_analyses if p["recommendation"] in ["SELL", "TAKE_PROFIT"])
        holds = sum(1 for p in position_analyses if p["recommendation"] == "HOLD")
        buys = sum(1 for p in position_analyses if p["recommendation"] == "BUY")
        strong_buys = sum(1 for p in position_analyses if p["recommendation"] == "STRONG_BUY")
        
        # Get macro context
        macro_score = macro.get("market_favorability", {}).get("score", 50)
        
        # Sector diversification
        sectors_represented = len(set(p["sector"] for p in position_analyses))
        
        recommendations = []
        
        # Cash allocation recommendation
        if macro_score > 70:
            if cash_pct > 20:
                recommendations.append({
                    "type": "CASH_ALLOCATION",
                    "action": "Deploy excess cash into high-conviction positions",
                    "priority": "HIGH",
                    "rationale": "Strong macro environment supports equity investment"
                })
        elif macro_score < 30:
            if cash_pct < 15:
                recommendations.append({
                    "type": "CASH_ALLOCATION",
                    "action": "Increase cash allocation to 15-20% for protection",
                    "priority": "HIGH",
                    "rationale": "Weak macro environment suggests defensive positioning"
                })
        
        # Rebalancing recommendation
        if strong_sells > 0 or sells > 2:
            recommendations.append({
                "type": "REBALANCING",
                "action": f"Consider selling {strong_sells + sells} positions showing weakness",
                "priority": "MEDIUM",
                "rationale": "Multiple positions showing negative signals"
            })
        
        # Opportunity recommendation
        if strong_buys > 0 or buys > 2:
            recommendations.append({
                "type": "OPPORTUNITY",
                "action": f"Consider adding to {strong_buys + buys} positions showing strength",
                "priority": "MEDIUM",
                "rationale": "Multiple positions showing positive signals"
            })
        
        # Diversification recommendation
        if sectors_represented < 5:
            recommendations.append({
                "type": "DIVERSIFICATION",
                "action": "Increase sector diversification (currently only {0} sectors)".format(sectors_represented),
                "priority": "LOW",
                "rationale": "Portfolio lacks sector diversification"
            })
        
        # Sector rotation recommendation with richer context
        sector_predictions = sector.get("sector_predictions", [])
        allocation = sector.get("portfolio_allocation", {})
        sector_highlights = self._extract_sector_highlights(sector_predictions, allocation)

        top_favorable = sector_highlights.get("top_favorable", [])
        least_favorable = sector_highlights.get("least_favorable", [])
        underweight_sectors = sector_highlights.get("underweight", [])
        overweight_sectors = sector_highlights.get("overweight", [])

        if top_favorable:
            formatted_top = ", ".join([
                f"{p['sector']} (score {p['score']:.1f})" for p in top_favorable
            ])
            recommendations.append({
                "type": "SECTOR_OUTLOOK",
                "action": f"Top favorable sectors: {formatted_top}",
                "priority": "HIGH",
                "rationale": "Top 3 sectors expected to outperform based on momentum and macro context"
            })

        if least_favorable:
            formatted_bottom = ", ".join([
                f"{p['sector']} (score {p['score']:.1f})" for p in least_favorable
            ])
            recommendations.append({
                "type": "SECTOR_RISK",
                "action": f"Not favorable sectors: {formatted_bottom}",
                "priority": "MEDIUM",
                "rationale": "Avoid adding exposure where signals point to underperformance"
            })

        if underweight_sectors:
            recommendations.append({
                "type": "SECTOR_ROTATION",
                "action": f"Consider adding exposure to: {', '.join(underweight_sectors[:3])}",
                "priority": "MEDIUM",
                "rationale": "Sectors predicted to outperform are underrepresented"
            })

        if overweight_sectors:
            recommendations.append({
                "type": "SECTOR_TRIM",
                "action": f"Trim exposure in: {', '.join(overweight_sectors[:3])}",
                "priority": "MEDIUM",
                "rationale": "Overweight in sectors expected to lag increases downside risk"
            })
        
        return {
            "portfolio_metrics": {
                "total_value": total_value,
                "cash_allocation": cash_pct,
                "total_positions": len(position_analyses),
                "sectors_represented": sectors_represented
            },
            "recommendation_summary": {
                "strong_buy": strong_buys,
                "buy": buys,
                "hold": holds,
                "sell": sells,
                "strong_sell": strong_sells
            },
            "recommendations": recommendations,
            "sector_highlights": sector_highlights
        }
    
    def _prioritize_actions(
        self,
        position_analyses: List[Dict[str, Any]],
        portfolio_recommendations: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Prioritize all recommended actions"""
        
        actions = []
        
        # Add high-priority portfolio actions
        for rec in portfolio_recommendations.get("recommendations", []):
            if rec["priority"] == "HIGH":
                actions.append({
                    "priority": 1,
                    "type": rec["type"],
                    "action": rec["action"],
                    "rationale": rec["rationale"]
                })
        
        # Add positions showing weakness summary
        weak_positions = [a for a in position_analyses if a["recommendation"] in ["STRONG_SELL", "CUT_LOSS", "SELL"]]
        if weak_positions:
            weak_symbols = [p["symbol"] for p in weak_positions[:5]]
            actions.append({
                "priority": 2,
                "type": "WEAKNESS_ALERT",
                "action": f"Positions showing weakness: {', '.join(weak_symbols)}",
                "rationale": f"{len(weak_positions)} position(s) with negative signals - review for potential exit"
            })
        
        # Add strong sell/buy signals
        for analysis in position_analyses:
            if analysis["recommendation"] in ["STRONG_SELL", "CUT_LOSS"]:
                actions.append({
                    "priority": 2,
                    "type": "POSITION_ACTION",
                    "symbol": analysis["symbol"],
                    "action": f"{analysis['recommendation']}: {analysis['symbol']}",
                    "rationale": analysis["rationale"],
                    "market_value": analysis["market_value"]
                })
            elif analysis["recommendation"] == "STRONG_BUY":
                actions.append({
                    "priority": 2,
                    "type": "POSITION_ACTION",
                    "symbol": analysis["symbol"],
                    "action": f"STRONG_BUY: {analysis['symbol']}",
                    "rationale": analysis["rationale"],
                    "market_value": analysis["market_value"]
                })
        
        # Add medium-priority portfolio actions
        for rec in portfolio_recommendations.get("recommendations", []):
            if rec["priority"] == "MEDIUM":
                actions.append({
                    "priority": 3,
                    "type": rec["type"],
                    "action": rec["action"],
                    "rationale": rec["rationale"]
                })
        
        # Add regular buy/sell signals for larger positions
        for analysis in position_analyses:
            if analysis["market_value"] > 1000:  # Focus on material positions
                if analysis["recommendation"] in ["BUY", "SELL", "TAKE_PROFIT"]:
                    actions.append({
                        "priority": 4,
                        "type": "POSITION_ACTION",
                        "symbol": analysis["symbol"],
                        "action": f"{analysis['recommendation']}: {analysis['symbol']}",
                        "rationale": analysis["rationale"],
                        "market_value": analysis["market_value"]
                    })
        
        # Sort by priority
        actions.sort(key=lambda x: x["priority"])
        
        return actions[:10]  # Return top 10 actions
    
    def _assess_portfolio_risk(
        self,
        portfolio: Dict[str, Any],
        position_analyses: List[Dict[str, Any]],
        macro: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Assess overall portfolio risk"""
        
        total_value = portfolio.get("summary", {}).get("total_portfolio_value", 0)
        
        # Calculate concentration risk
        position_values = [p["market_value"] for p in position_analyses]
        largest_position = max(position_values) if position_values else 0
        concentration_pct = (largest_position / total_value * 100) if total_value > 0 else 0
        
        if concentration_pct > 40:
            concentration_risk = "HIGH"
        elif concentration_pct > 25:
            concentration_risk = "MEDIUM"
        else:
            concentration_risk = "LOW"
        
        # Calculate sentiment risk
        bearish_positions = sum(1 for p in position_analyses if p["sentiment"]["label"] == "bearish")
        sentiment_risk_pct = (bearish_positions / len(position_analyses) * 100) if position_analyses else 0
        
        if sentiment_risk_pct > 50:
            sentiment_risk = "HIGH"
        elif sentiment_risk_pct > 30:
            sentiment_risk = "MEDIUM"
        else:
            sentiment_risk = "LOW"
        
        # Calculate underperformance risk
        losing_positions = sum(1 for p in position_analyses if p["total_gain_pct"] < -10)
        underperformance_risk_pct = (losing_positions / len(position_analyses) * 100) if position_analyses else 0
        
        if underperformance_risk_pct > 40:
            underperformance_risk = "HIGH"
        elif underperformance_risk_pct > 25:
            underperformance_risk = "MEDIUM"
        else:
            underperformance_risk = "LOW"
        
        # Overall risk assessment
        risk_factors = [concentration_risk, sentiment_risk, underperformance_risk]
        high_risks = risk_factors.count("HIGH")
        medium_risks = risk_factors.count("MEDIUM")
        
        if high_risks >= 2:
            overall_risk = "HIGH"
        elif high_risks == 1 or medium_risks >= 2:
            overall_risk = "MEDIUM"
        else:
            overall_risk = "LOW"
        
        return {
            "overall_risk": overall_risk,
            "concentration_risk": concentration_risk,
            "largest_position_pct": round(concentration_pct, 1),
            "sentiment_risk": sentiment_risk,
            "bearish_positions_pct": round(sentiment_risk_pct, 1),
            "underperformance_risk": underperformance_risk,
            "losing_positions_pct": round(underperformance_risk_pct, 1),
            "risk_factors": {
                "concentration": concentration_risk,
                "sentiment": sentiment_risk,
                "underperformance": underperformance_risk
            }
        }
    
    def _create_executive_summary(
        self,
        portfolio: Dict[str, Any],
        position_analyses: List[Dict[str, Any]],
        portfolio_recommendations: Dict[str, Any],
        macro: Dict[str, Any],
        sentiment: Dict[str, Any],
        sector: Dict[str, Any]
    ) -> str:
        """Create executive summary synthesizing macro, sentiment, and sector into actionable insights"""
        
        total_value = portfolio.get("summary", {}).get("total_portfolio_value", 0)
        total_positions = len(position_analyses)
        
        # Extract macro data (using correct keys from confidence_score)
        confidence_score = macro.get("confidence_score", {})
        macro_score = confidence_score.get("score", 50)
        macro_direction = confidence_score.get("direction", "neutral")
        macro_interpretation = confidence_score.get("interpretation", "mixed signals")
        
        # Extract sector highlights
        sector_highlights = portfolio_recommendations.get("sector_highlights", {}) or {}
        if not sector_highlights:
            sector_highlights = self._extract_sector_highlights(
                sector.get("sector_predictions", []),
                sector.get("portfolio_allocation", {})
            )
        
        # Extract sentiment data
        overall_sentiment = sentiment.get("overall_sentiment", {})
        sentiment_scores = [
            d.get("overall_score")
            for d in overall_sentiment.values()
            if d.get("overall_score") is not None
        ]
        avg_sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0
        bullish_count = sum(1 for d in overall_sentiment.values() if d.get("sentiment") == "bullish")
        bearish_count = sum(1 for d in overall_sentiment.values() if d.get("sentiment") == "bearish")
        
        if avg_sentiment > 0.2:
            sentiment_label = "bullish"
        elif avg_sentiment < -0.2:
            sentiment_label = "bearish"
        else:
            sentiment_label = "neutral"
        
        # Extract portfolio action summary
        rec_summary = portfolio_recommendations.get("recommendation_summary", {})
        strong_buys = rec_summary.get("strong_buy", 0)
        buys = rec_summary.get("buy", 0)
        holds = rec_summary.get("hold", 0)
        sells = rec_summary.get("sell", 0)
        strong_sells = rec_summary.get("strong_sell", 0)
        
        top_fav_sectors = sector_highlights.get("top_favorable", [])
        least_fav_sectors = sector_highlights.get("least_favorable", [])
        
        # Build comprehensive summary
        summary = (
            f"Portfolio (${total_value:,.0f} across {total_positions} holdings) sits in a {macro_direction.lower()} "
            f"macro environment (score {macro_score}/100) with {sentiment_label} sentiment ({avg_sentiment:+.2f}). "
        )
        
        if top_fav_sectors:
            top_names = ", ".join([p['sector'] for p in top_fav_sectors[:2]])
            summary += (
                f"The market is being pushed by favorable sector rotation toward {top_names}. "
            )
        
        if least_fav_sectors:
            low_names = ", ".join([p['sector'] for p in least_fav_sectors[:2]])
            summary += (
                f"Caution warranted in {low_names} which show weakness. "
            )
        
        # Add action recommendations
        if strong_buys > 0 or strong_sells > 0:
            summary += f"Key signals: {strong_buys} strong buy and {strong_sells} cut-loss signals in current holdings. "
        
        if macro_score > 60 and sentiment_label == "bullish":
            summary += "Macro tailwinds combined with positive sentiment support increased equity exposure and selective growth positioning. "
        elif macro_score < 40 or sentiment_label == "bearish":
            summary += "Mixed macro/sentiment backdrop suggests defensive positioning and reducing concentration risk. "
        else:
            summary += "Mixed signals warrant balanced approach with focus on quality and conviction positions. "
        
        summary += (
            f"Consider deploying cash into favorable sectors while trimming {bearish_count} bearish positions. "
            f"New position ideas are generated below across macro, sentiment, and sector filters."
        )
        
        return summary
    
    async def _generate_integrator_ai_reasoning(
        self,
        results: Dict[str, Any],
        position_analyses: List[Dict[str, Any]],
        macro: Dict[str, Any],
        sentiment: Dict[str, Any],
        sector: Dict[str, Any]
    ) -> str:
        """Generate AI reasoning for integrated portfolio recommendations"""
        
        # Extract key metrics
        buys = [p['symbol'] for p in position_analyses if p.get('recommendation') == 'BUY'][:3]
        sells = [p['symbol'] for p in position_analyses if p.get('recommendation') in ['SELL', 'CUT_LOSS']][:3]
        holds = [p['symbol'] for p in position_analyses if p.get('recommendation') == 'HOLD'][:3]
        
        macro_score = macro.get('confidence_score', {}).get('score', 50)
        macro_regime = macro.get('market_regime', {}).get('regime', 'neutral')
        macro_direction = macro.get('confidence_score', {}).get('direction', 'stable')
        crisis_flags = macro.get('confidence_score', {}).get('crisis_detection', {}).get('flags', {})
        agreement_level = macro.get('confidence_score', {}).get('agreement', {}).get('level', 'moderate')
        risk_level = results.get('risk_assessment', {}).get('overall_risk', 'MEDIUM')
        
        # Build crisis context
        active_crises = []
        if crisis_flags.get('credit_stress_cliff'):
            active_crises.append('CREDIT STRESS (HY spreads deteriorating)')
        if crisis_flags.get('recession_detector'):
            active_crises.append('RECESSION RISK (2+ recession indicators triggered)')
        if crisis_flags.get('inflation_confidence_combo'):
            active_crises.append('STAGFLATION WARNING (CPI>3% + confidence collapse)')
        crisis_text = ', '.join(active_crises) if active_crises else 'None'
        
        # Build macro interpretation guidance
        macro_context = f"""
MACRO SCORE INTERPRETATION (CRITICAL - READ CAREFULLY):
Score: {macro_score}/100 ({macro_direction}, {agreement_level} agreement)
Regime: {macro_regime.upper()}
Active Crisis Flags: {crisis_text}

WHAT THIS SCORE MEANS (6-12 MONTH FORWARD OUTLOOK):
- The macro score is a FORWARD-LOOKING indicator (6-12 months ahead), NOT a current market assessment
- Score is weighted: 55% alternative metrics (leading indicators like LEI, spreads, Sahm Rule) + 45% popular metrics (lagging indicators like GDP, CPI, unemployment)
- Higher alternative vs popular score means leading indicators are ahead of traditional metrics

SCORE RANGES:
- 0-20: CRASH/CRISIS - Severe credit stress, recession, or confidence collapse. REDUCE RISK immediately.
- 20-40: BEARISH/INFLECTION - Economic headwinds present or peak inflection with warning signals. DEFENSIVE positioning.
- 40-55: NEUTRAL/TROUGH - Transition period OR bull run entry point if leading indicators surge. WAIT for clarity or ACCUMULATE if recovery signals.
- 55-70: MID-RALLY - Continuation rally with leading indicators sustaining advance. HOLD/ADD on dips.
- 70-85: PEAK STRENGTH - All signals aligned bullish, elevated valuation risk. PROFIT TAKE partially.

CRISIS FLAGS MEANING:
- Credit Stress Cliff: HY bond spreads >400bps or spiking +50bps = credit markets freezing, REDUCE RISK
- Recession Detector: 2+ indicators (Sahm Rule, LEI<30, Jobless Claims, Consumer Confidence) = downturn confirmed, GO DEFENSIVE
- Stagflation Warning: CPI>3% + confidence down >10% YoY = inflation + weak growth combo, REDUCE EQUITY exposure

RECOVERY DETECTOR (Bull Run Entry):
- Triggers when LEI>50 + Sahm Rule safe + HY spreads stable + alternative score>60 + NO crisis flags
- This signals bull run entry point - ACCUMULATE equity exposure despite bearish traditional metrics

AGREEMENT INTERPRETATION:
- High Agreement: Popular and alternative metrics aligned, higher confidence in direction
- High Divergence (Alt >> Pop): Leading indicators ahead of lagging - often bull run entry signal
- Low Agreement: Metrics conflicting, suggests CAUTION and WAIT for clarity
"""
        
        prompt = f"""{macro_context}

Now synthesize this multi-agent portfolio analysis into actionable investment guidance using the macro interpretation above.

Risk Level: {risk_level}

Position Recommendations:
- Buy candidates: {', '.join(buys) if buys else 'none'}
- Hold positions: {', '.join(holds) if holds else 'none'}  
- Sell candidates: {', '.join(sells) if sells else 'none'}

Top Action Priorities:
{json.dumps(results.get('action_priorities', [])[:3], indent=2)}

Provide:
1. Overall portfolio positioning assessment (considering the FORWARD-LOOKING macro outlook)
2. How sentiment + macro + sector signals align or conflict
3. Top 2-3 specific actions to take now
4. Key risk factors to monitor (especially crisis flags if active)

IMPORTANT: Use PLAIN TEXT formatting only - NO markdown syntax (no ###, **, bullets, etc.). Use simple paragraphs.
Be concise (4-5 sentences max). Remember: macro score reflects 6-12 month outlook, NOT current conditions."""
        
        return await self.generate_ai_reasoning(results, prompt)
    
    async def _suggest_new_positions(
        self,
        portfolio: Dict[str, Any],
        position_analyses: List[Dict[str, Any]],
        macro: Dict[str, Any],
        sentiment: Dict[str, Any],
        sector: Dict[str, Any],
        risk_assessment: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Use AI to suggest 2-5 new positions outside current portfolio"""
        
        # Extract current portfolio symbols
        current_symbols = [p['symbol'] for p in position_analyses]
        
        # Extract favorable sectors
        sector_predictions = sector.get('sector_predictions', [])
        favorable_sectors = [p for p in sector_predictions if p.get('prediction_score', 0) > 0]
        favorable_sectors = sorted(favorable_sectors, key=lambda x: x.get('prediction_score', 0), reverse=True)[:3]
        
        unfavorable_sectors = [p for p in sector_predictions if p.get('prediction_score', 0) < 0]
        unfavorable_sectors = sorted(unfavorable_sectors, key=lambda x: x.get('prediction_score', 0))[:3]
        
        # Extract macro stance
        macro_score = macro.get('confidence_score', {}).get('score', 50)
        macro_direction = macro.get('confidence_score', {}).get('direction', 'neutral')
        macro_interpretation = macro.get('confidence_score', {}).get('interpretation', 'neutral')

        # Extract sentiment context
        overall_sentiment = sentiment.get('overall_sentiment', {})
        sentiment_scores = [
            d.get('overall_score')
            for d in overall_sentiment.values()
            if d.get('overall_score') is not None
        ]
        avg_sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0
        bullish_count = sum(1 for d in overall_sentiment.values() if d.get('sentiment') == 'bullish')
        bearish_count = sum(1 for d in overall_sentiment.values() if d.get('sentiment') == 'bearish')
        insufficient_count = sum(1 for d in overall_sentiment.values() if d.get('sentiment') == 'INSUFFICIENT_DATA')
        if avg_sentiment > 0.2:
            sentiment_tilt = 'positive'
        elif avg_sentiment < -0.2:
            sentiment_tilt = 'negative'
        else:
            sentiment_tilt = 'neutral'
        
        # Extract risk tolerance
        overall_risk = risk_assessment.get('overall_risk', 'MEDIUM')
        
        # Prepare sector data for prompt
        fav_sectors_data = [{'sector': p['sector'], 'score': p['prediction_score'], 'outlook': p.get('outlook', 'neutral')} for p in favorable_sectors]
        unfav_sectors_data = [{'sector': p['sector'], 'score': p['prediction_score']} for p in unfavorable_sectors]
        
        # Build detailed prompt with all analysis data
        prompt = f"""Based on this comprehensive portfolio analysis, suggest specific stock tickers or ETFs to BUY as new positions (not currently held).

CURRENT PORTFOLIO:
- Symbols held: {', '.join(current_symbols)}
- Risk level: {overall_risk}
- Total positions: {len(current_symbols)}

MACRO ENVIRONMENT:
- Confidence score: {macro_score}/100 ({macro_direction})
- Interpretation: {macro_interpretation}
- Stance: {'Bullish - favor growth/cyclicals' if macro_score > 60 else 'Bearish - favor defensives' if macro_score < 40 else 'Neutral - balanced approach'}

SENTIMENT CONTEXT (portfolio news-derived):
- Average sentiment score: {avg_sentiment:.2f} ({sentiment_tilt})
- Bullish positions: {bullish_count} | Bearish positions: {bearish_count} | Insufficient data: {insufficient_count}

FAVORABLE SECTORS (to prioritize):
{json.dumps(fav_sectors_data, indent=2)}

UNFAVORABLE SECTORS (to avoid):
{json.dumps(unfav_sectors_data, indent=2)}

RISK TOLERANCE: Match portfolio's {overall_risk} risk profile

Requirements:
1. Provide THREE sections of ideas: macro-driven, sentiment-driven, and sector-driven
2. Each section should list 3-5 tickers (stocks or ETFs)
3. Focus on favorable sectors and align with macro stance (growth vs defensive)
4. Do NOT suggest any symbols already in portfolio: {', '.join(current_symbols)}
5. Match {overall_risk} risk tolerance
6. For each suggestion provide: ticker, company/fund name, sector, and 1-2 sentence rationale

Format your response as a JSON object with three arrays. Be specific and actionable. Only suggest real, liquid securities."""
        
        try:
            # Call OpenAI for suggestions
            self.log("Generating new position suggestions with AI...")
            context_data = {"macro_score": macro_score, "risk_level": overall_risk}
            response_text = await self.generate_ai_reasoning(context_data, prompt)
            
            # Parse JSON response
            import re
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                try:
                    suggestions_payload = json.loads(json_match.group())
                except json.JSONDecodeError:
                    self.log("JSON parsing failed for new position suggestions", level="WARNING")
                    suggestions_payload = None
            else:
                suggestions_payload = None

            if suggestions_payload:
                macro_ideas = []
                sentiment_ideas = []
                sector_ideas = []
                overall_ideas = []

                if isinstance(suggestions_payload, list):
                    overall_ideas = suggestions_payload
                elif isinstance(suggestions_payload, dict):
                    macro_ideas = suggestions_payload.get("macro_ideas", []) or []
                    sentiment_ideas = suggestions_payload.get("sentiment_ideas", []) or []
                    sector_ideas = suggestions_payload.get("sector_ideas", []) or []
                    overall_ideas = suggestions_payload.get("overall_ideas", []) or []

                combined = []
                for group in [macro_ideas, sentiment_ideas, sector_ideas, overall_ideas]:
                    for item in group:
                        if item not in combined:
                            combined.append(item)

                return {
                    "suggestions": combined,
                    "macro_ideas": macro_ideas,
                    "sentiment_ideas": sentiment_ideas,
                    "sector_ideas": sector_ideas,
                    "overall_ideas": overall_ideas,
                    "count": len(combined),
                    "based_on": {
                        "favorable_sectors": [p['sector'] for p in favorable_sectors],
                        "macro_score": macro_score,
                        "macro_direction": macro_direction,
                        "sentiment_tilt": sentiment_tilt,
                        "risk_level": overall_risk
                    }
                }
        except Exception as e:
            self.log(f"Error generating AI position suggestions: {e}", level="WARNING")
        
        # FALLBACK: Generate sensible stock ideas when AI fails or times out
        self.log("Using fallback stock suggestions...", level="INFO")
        macro_ideas = self._generate_fallback_macro_ideas(macro_score, overall_risk, current_symbols)
        sentiment_ideas = self._generate_fallback_sentiment_ideas(sentiment_tilt, bearish_count, overall_risk, current_symbols)
        sector_ideas = self._generate_fallback_sector_ideas(favorable_sectors, overall_risk, current_symbols)
        
        combined = []
        for group in [macro_ideas, sentiment_ideas, sector_ideas]:
            for item in group:
                if item not in combined:
                    combined.append(item)
        
        return {
            "suggestions": combined,
            "macro_ideas": macro_ideas,
            "sentiment_ideas": sentiment_ideas,
            "sector_ideas": sector_ideas,
            "overall_ideas": [],
            "count": len(combined),
            "based_on": {
                "favorable_sectors": [p['sector'] for p in favorable_sectors],
                "macro_score": macro_score,
                "macro_direction": macro_direction,
                "sentiment_tilt": sentiment_tilt,
                "risk_level": overall_risk
            },
            "note": "Using fallback suggestions - consider running with API key for AI-optimized recommendations"
        }
    
    def _generate_fallback_macro_ideas(self, macro_score: float, risk_level: str, current_symbols: List[str]) -> List[Dict[str, Any]]:
        """Generate reasonable stock ideas based on macro score when AI unavailable"""
        ideas = []
        
        if macro_score > 65:
            # Bullish macro - suggest growth/cyclical exposure
            suggestion_universe = [
                {"ticker": "XLG", "name": "SPDR S&P 500 Growth ETF", "sector": "Large-Cap Growth", "rationale": "Bullish macro backdrop favors growth equities with strong near-term catalysts.", "signal": "BUY"},
                {"ticker": "ARKK", "name": "ARK Innovation ETF", "sector": "Growth/Technology", "rationale": "Favorable GDP growth and declining rates create tailwinds for innovation-driven companies.", "signal": "BUY"},
                {"ticker": "VGT", "name": "Vanguard Information Technology ETF", "sector": "Technology", "rationale": "Tech sector benefits from bullish macro momentum and strong corporate earnings outlook.", "signal": "BUY"},
                {"ticker": "XLF", "name": "Financial Select Sector SPDR ETF", "sector": "Financials", "rationale": "Rising rates and strong economic growth support financial sector profitability.", "signal": "BUY"},
            ]
        elif macro_score < 45:
            # Bearish macro - suggest defensive exposure
            suggestion_universe = [
                {"ticker": "XLP", "name": "Consumer Staples Select Sector SPDR ETF", "sector": "Consumer Staples", "rationale": "Defensive positioning with non-cyclical demand holds up well during economic slowdown.", "signal": "BUY"},
                {"ticker": "XLU", "name": "Utilities Select Sector SPDR ETF", "sector": "Utilities", "rationale": "Stable dividend yields and essential services provide downside protection.", "signal": "BUY"},
                {"ticker": "XLV", "name": "Health Care Select Sector SPDR ETF", "sector": "Healthcare", "rationale": "Healthcare demand remains steady regardless of economic cycle; recession-resistant.", "signal": "BUY"},
                {"ticker": "AGG", "name": "iShares Core US Aggregate Bond ETF", "sector": "Fixed Income", "rationale": "Bond exposure may benefit from potential rate cuts ahead", "signal": "WATCH"},
            ]
        else:
            # Neutral macro - suggest balanced exposure
            suggestion_universe = [
                {"ticker": "VTI", "name": "Vanguard Total Stock Market ETF", "sector": "Broad Market", "rationale": "Broad market exposure balances growth and stability in neutral macro environment.", "signal": "BUY"},
                {"ticker": "XLV", "name": "Health Care Select Sector SPDR ETF", "sector": "Healthcare", "rationale": "Defensive characteristics with steady dividends suit mixed market backdrop.", "signal": "BUY"},
                {"ticker": "XLE", "name": "Energy Select Sector SPDR ETF", "sector": "Energy", "rationale": "Energy sector offers valuation support with modest recovery potential.", "signal": "WATCH"},
                {"ticker": "IWM", "name": "iShares Russell 2000 ETF", "sector": "Small-Cap", "rationale": "Small-cap exposure provides diversification and upside participation.", "signal": "WATCH"},
            ]
        
        # Filter out current holdings and apply risk level
        filtered = [x for x in suggestion_universe if x["ticker"] not in current_symbols]
        
        if risk_level == "LOW":
            filtered = [x for x in filtered if x["signal"] == "BUY"]  # Only strongest ideas
        
        return filtered[:3]
    
    def _generate_fallback_sentiment_ideas(self, sentiment_tilt: str, bearish_pct: float, risk_level: str, current_symbols: List[str]) -> List[Dict[str, Any]]:
        """Generate ideas counter-trending bearish sentiment (value play) or aligned with bullish sentiment"""
        ideas = []
        
        if sentiment_tilt == "negative" and bearish_pct > 50:
            # Market is very bearish - look for contrarian opportunity (value)
            suggestion_universe = [
                {"ticker": "VTV", "name": "Vanguard Value ETF", "sector": "Value", "rationale": "Bearish sentiment often creates value opportunities in quality undervalued names; potential mean reversion play.", "signal": "BUY"},
                {"ticker": "TNA", "name": "Direxion Daily Small Cap Bull 3x ETF", "sector": "Small-Cap Value", "rationale": "Contrarian bet when sentiment is most negative; historically precedes strong reversals.", "signal": "WATCH"},
                {"ticker": "SCHV", "name": "Schwab US Value ETF", "sector": "Value", "rationale": "Value stocks trading at depressed multiples amid bearish sentiment offer asymmetric upside.", "signal": "BUY"},
                {"ticker": "BAC", "name": "Bank of America", "sector": "Financials", "rationale": "Quality financials oversold due to macro concerns; fundamental value support remains intact.", "signal": "WATCH"},
            ]
        elif sentiment_tilt == "positive":
            # Market is bullish - ride momentum in quality names
            suggestion_universe = [
                {"ticker": "MSFT", "name": "Microsoft", "sector": "Technology", "rationale": "Dominant AI and cloud exposure; bullish sentiment driven by strong secular growth.", "signal": "BUY"},
                {"ticker": "AAPL", "name": "Apple", "sector": "Technology", "rationale": "Quality mega-cap with positive momentum; benefits from bullish risk-on sentiment.", "signal": "BUY"},
                {"ticker": "NVDA", "name": "NVIDIA", "sector": "Technology", "rationale": "AI chip leader; bullish sentiment justified by structural growth drivers.", "signal": "BUY"},
                {"ticker": "QQQ", "name": "Invesco QQQ Trust", "sector": "Growth Tech", "rationale": "Tech-heavy index to capitalize on bullish momentum in growth equities.", "signal": "BUY"},
            ]
        else:
            # Neutral sentiment - quality dividend/stable
            suggestion_universe = [
                {"ticker": "JNJ", "name": "Johnson & Johnson", "sector": "Healthcare", "rationale": "Dividend aristocrat with defensive characteristics suit neutral sentiment backdrop.", "signal": "BUY"},
                {"ticker": "KO", "name": "The Coca-Cola Company", "sector": "Consumer Staples", "rationale": "Stable consumer staple with consistent dividends provides sentiment-neutral income.", "signal": "BUY"},
                {"ticker": "PG", "name": "Procter & Gamble", "sector": "Consumer Staples", "rationale": "Essential products with pricing power; neutral sentiment environment doesn't threaten fundamentals.", "signal": "BUY"},
                {"ticker": "SCHD", "name": "Schwab US Dividend Equity ETF", "sector": "Dividends", "rationale": "High-quality dividend ETF balances growth and income for mixed sentiment.", "signal": "BUY"},
            ]
        
        filtered = [x for x in suggestion_universe if x["ticker"] not in current_symbols]
        
        if risk_level == "LOW":
            filtered = [x for x in filtered if x["signal"] == "BUY"]
        
        return filtered[:3]
    
    def _generate_fallback_sector_ideas(self, favorable_sectors: List[Dict[str, Any]], risk_level: str, current_symbols: List[str]) -> List[Dict[str, Any]]:
        """Generate ideas focused on favorable predicted sectors"""
        ideas = []
        
        # Map sectors to popular stocks/ETFs with rationales
        sector_mapping = {
            "Consumer Staples": [
                {"ticker": "XLP", "name": "Consumer Staples Select Sector SPDR ETF", "sector": "Consumer Staples", "rationale": "Predicted outperformer with strong momentum; exposure to defensive sector shows favorable outlook.", "signal": "BUY"},
                {"ticker": "SPLG", "name": "SPDR Portfolio S&P 500 Consumer Staples ETF", "sector": "Consumer Staples", "rationale": "Consumer staples sector rotation thesis supports increased allocation to defensive names.", "signal": "BUY"},
                {"ticker": "PEP", "name": "PepsiCo", "sector": "Consumer Staples", "rationale": "Quality staples stock with price-setting power aligns with favorable sector outlook.", "signal": "BUY"},
            ],
            "Healthcare": [
                {"ticker": "XLV", "name": "Health Care Select Sector SPDR ETF", "sector": "Healthcare", "rationale": "Defensive healthcare sector offers steady growth independent of economic cycles.", "signal": "BUY"},
                {"ticker": "VHT", "name": "Vanguard Health Care ETF", "sector": "Healthcare", "rationale": "Broad healthcare exposure with favorable long-term demographics and innovation trends.", "signal": "BUY"},
                {"ticker": "JNJ", "name": "Johnson & Johnson", "sector": "Healthcare", "rationale": "Dividend aristocrat with diversified healthcare portfolio in growth sector.", "signal": "BUY"},
            ],
            "Technology": [
                {"ticker": "XLK", "name": "Technology Select Sector SPDR ETF", "sector": "Technology", "rationale": "Tech sector offers growth exposure in bullish macro environment with AI tailwinds.", "signal": "BUY"},
                {"ticker": "QQQ", "name": "Invesco QQQ Trust", "sector": "Technology", "rationale": "Nasdaq-weighted ETF provides concentrated exposure to outperforming tech/growth names.", "signal": "BUY"},
                {"ticker": "MSFT", "name": "Microsoft", "sector": "Technology", "rationale": "Dominant position in AI and cloud; leading beneficiary of tech sector momentum.", "signal": "BUY"},
            ],
            "Energy": [
                {"ticker": "XLE", "name": "Energy Select Sector SPDR ETF", "sector": "Energy", "rationale": "Energy sector mean reversion play; valuation support on stabilizing commodity backdrop.", "signal": "BUY"},
                {"ticker": "CVX", "name": "Chevron Corporation", "sector": "Energy", "rationale": "Quality energy producer with strong dividend; positioned well for cyclical recovery.", "signal": "WATCH"},
                {"ticker": "MPC", "name": "Marathon Petroleum", "sector": "Energy", "rationale": "Midstream/downstream exposure offers diversified energy sector participation.", "signal": "WATCH"},
            ],
            "Financials": [
                {"ticker": "XLF", "name": "Financial Select Sector SPDR ETF", "sector": "Financials", "rationale": "Financial sector benefits from rising/stable rates and improving credit conditions.", "signal": "BUY"},
                {"ticker": "JPM", "name": "JPMorgan Chase", "sector": "Financials", "rationale": "Strongest financial institution; leadership quality aligns with favorable macro backdrop.", "signal": "BUY"},
                {"ticker": "BAC", "name": "Bank of America", "sector": "Financials", "rationale": "Systemically important bank with attractive valuation in stable rate environment.", "signal": "WATCH"},
            ],
            "Industrials": [
                {"ticker": "XLI", "name": "Industrial Select Sector SPDR ETF", "sector": "Industrials", "rationale": "Industrial sector momentum and mean reversion dynamics support favorable outlook.", "signal": "BUY"},
                {"ticker": "BA", "name": "Boeing", "sector": "Industrials", "rationale": "Aerospace leader positioned for recovery in strong macro environment", "signal": "WATCH"},
                {"ticker": "CAT", "name": "Caterpillar", "sector": "Industrials", "rationale": "Equipment manufacturer benefits from infrastructure spending and global growth.", "signal": "BUY"},
            ],
        }
        
        for fav_sector in favorable_sectors[:2]:  # Top 2 favorable sectors
            sector_name = fav_sector.get("sector")
            if sector_name in sector_mapping:
                sector_options = sector_mapping[sector_name]
                filtered = [x for x in sector_options if x["ticker"] not in current_symbols]
                if filtered:
                    ideas.append(filtered[0])
        
        if risk_level == "LOW":
            ideas = [x for x in ideas if x["signal"] == "BUY"]
        
        return ideas[:3]

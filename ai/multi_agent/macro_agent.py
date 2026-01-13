"""
Macro Agent - Comprehensive Macroeconomic Analysis
Combines popular and alternative metrics to assess market direction with configurable weights
"""

from typing import Dict, Any, List
from datetime import datetime
import asyncio
import json
import os
from .base_agent import BaseAgent
from .macro.popular_metrics import PopularMetricsAnalyzer
from .macro.alternative_metrics import AlternativeMetricsAnalyzer


class MacroAgent(BaseAgent):
    """
    Comprehensive macroeconomic analysis agent that:
    1. Analyzes popular/mainstream economic indicators
    2. Analyzes alternative/less common indicators
    3. Combines both with configurable weights to produce market direction confidence score
    
    Score Scale: 0 = Fully Bearish, 50 = Neutral, 100 = Fully Bullish
    Timeframe: Medium-term (3-12 months)
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("Macro", config)
        self.api_keys = config.get("api_keys", {}) if config else {}
        
        # Load weights configuration
        self.weights_config = self._load_weights_config()
        
        # Initialize analyzers
        popular_weights = self.weights_config["popular_metrics"]["individual_weights"]
        alternative_weights = self.weights_config["alternative_metrics"]["individual_weights"]
        
        self.popular_analyzer = PopularMetricsAnalyzer(self.api_keys, popular_weights)
        self.alternative_analyzer = AlternativeMetricsAnalyzer(self.api_keys, alternative_weights)
        
        # Get category weights for final combination
        self.popular_weight = self.weights_config["popular_metrics"]["total_weight_in_final_score"]
        self.alternative_weight = self.weights_config["alternative_metrics"]["total_weight_in_final_score"]
    
    def _load_weights_config(self) -> Dict[str, Any]:
        """Load weights configuration from JSON file"""
        config_path = os.path.join(
            os.path.dirname(__file__),
            "macro",
            "macro_weights_config.json"
        )
        
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            self.log(f"Error loading weights config: {e}. Using defaults.", "WARNING")
            # Return default weights if config file not found
            return {
                "popular_metrics": {
                    "total_weight_in_final_score": 0.60,
                    "individual_weights": {
                        "gdp_growth": 0.15, "unemployment": 0.12, "cpi_inflation": 0.12,
                        "ppi_inflation": 0.08, "fed_funds_rate": 0.12, "treasury_10y": 0.10,
                        "sp500_performance": 0.10, "ism_manufacturing": 0.08,
                        "ism_services": 0.08, "retail_sales": 0.05, "consumer_confidence": 0.05
                    }
                },
                "alternative_metrics": {
                    "total_weight_in_final_score": 0.40,
                    "individual_weights": {
                        "yield_curve": 0.15, "m2_growth": 0.08, "high_yield_spreads": 0.12,
                        "leading_economic_index": 0.12, "sahm_rule": 0.10, "dollar_index": 0.08,
                        "architecture_billings": 0.06, "copper_prices": 0.10, "luxury_sales": 0.06,
                        "gold_treasury_ratio": 0.08, "corporate_debt_gdp": 0.05
                    }
                },
                "scoring_thresholds": {
                    "very_bullish": 70, "bullish": 60, "neutral_high": 55,
                    "neutral_low": 45, "bearish": 40, "very_bearish": 30
                },
                "confidence_adjustments": {
                    "high_agreement_bonus": 5, "low_agreement_penalty": -5,
                    "agreement_threshold": 10
                }
            }
    
    async def analyze(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform comprehensive macroeconomic analysis
        
        Args:
            context: Analysis context (not heavily used, for compatibility)
            
        Returns:
            Complete macro analysis with confidence score and detailed breakdown
        """
        self.log("="*60)
        self.log("COMPREHENSIVE MACROECONOMIC ANALYSIS")
        self.log("="*60)
        
        try:
            # Run analyses sequentially to respect shared API rate limits (Alpha Vantage)
            self.log("Fetching popular metrics...")
            popular_results = await self.popular_analyzer.analyze()
            await asyncio.sleep(1)  # brief pause between API families
            self.log("Fetching alternative metrics...")
            alternative_results = await self.alternative_analyzer.analyze()
            
            # Check for errors
            if isinstance(popular_results, Exception):
                self.log(f"Popular metrics error: {popular_results}", "ERROR")
                raise popular_results
            
            if isinstance(alternative_results, Exception):
                self.log(f"Alternative metrics error: {alternative_results}", "ERROR")
                raise alternative_results
            
            # Calculate final confidence score
            final_score = self._calculate_final_confidence_score(
                popular_results, alternative_results
            )
            
            # Generate insights and recommendations
            insights = self._generate_insights(popular_results, alternative_results, final_score)
            recommendations = self._generate_recommendations(final_score)
            summary = self._create_summary(final_score, popular_results, alternative_results)
            
            self.log("="*60)
            self.log(f"FINAL CONFIDENCE SCORE: {final_score['score']}/100 ({final_score['direction'].upper()})")
            self.log("="*60)
            
            return {
                "summary": summary,
                "confidence_score": final_score,
                "popular_metrics": popular_results,
                "alternative_metrics": alternative_results,
                "insights": insights,
                "recommendations": recommendations,
                "weights_used": {
                    "popular_weight": self.popular_weight,
                    "alternative_weight": self.alternative_weight
                },
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.log(f"Macro analysis failed: {str(e)}", "ERROR")
            raise
            self.log(f"Macro analysis failed: {str(e)}", "ERROR")
            raise
    
    def _calculate_final_confidence_score(
        self,
        popular_results: Dict[str, Any],
        alternative_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Calculate final confidence score combining popular and alternative metrics
        
        Returns:
            Dictionary with final score, direction, and analysis
        """
        popular_score = popular_results["composite_score"]["score"]
        alternative_score = alternative_results["composite_score"]["score"]
        
        # Calculate weighted average
        final_score = (
            popular_score * self.popular_weight +
            alternative_score * self.alternative_weight
        )
        
        # Check agreement between popular and alternative metrics
        agreement = abs(popular_score - alternative_score)
        thresholds = self.weights_config["confidence_adjustments"]
        
        if agreement <= thresholds["agreement_threshold"]:
            # High agreement - boost confidence
            final_score += thresholds["high_agreement_bonus"]
            agreement_level = "high"
        elif agreement > 20:
            # Low agreement - reduce confidence slightly
            final_score += thresholds["low_agreement_penalty"]
            agreement_level = "low"
        else:
            agreement_level = "moderate"
        
        # Ensure score stays within bounds
        final_score = max(0, min(100, final_score))
        
        # Determine direction and interpretation
        direction, interpretation = self._interpret_score(final_score)
        
        return {
            "score": round(final_score, 1),
            "direction": direction,
            "interpretation": interpretation,
            "components": {
                "popular_metrics_score": round(popular_score, 1),
                "alternative_metrics_score": round(alternative_score, 1),
                "popular_weight": self.popular_weight,
                "alternative_weight": self.alternative_weight
            },
            "agreement": {
                "level": agreement_level,
                "difference": round(agreement, 1),
                "note": self._get_agreement_note(agreement_level, popular_score, alternative_score)
            }
        }
    
    def _interpret_score(self, score: float) -> tuple:
        """
        Interpret confidence score and return direction and detailed interpretation
        
        Returns:
            Tuple of (direction, interpretation)
        """
        thresholds = self.weights_config["scoring_thresholds"]
        
        if score >= thresholds["very_bullish"]:
            return "BULLISH", "Very Bullish - Strong positive signals across metrics"
        elif score >= thresholds["bullish"]:
            return "BULLISH", "Bullish - Favorable macroeconomic conditions"
        elif score >= thresholds["neutral_high"]:
            return "NEUTRAL", "Neutral with Bullish Lean - Mixed signals, slight positive tilt"
        elif score >= thresholds["neutral_low"]:
            return "NEUTRAL", "Neutral - Balanced signals, no clear direction"
        elif score >= thresholds["bearish"]:
            return "BEARISH", "Neutral with Bearish Lean - Mixed signals, slight negative tilt"
        elif score >= thresholds["very_bearish"]:
            return "BEARISH", "Bearish - Challenging macroeconomic conditions"
        else:
            return "BEARISH", "Very Bearish - Strong negative signals across metrics"
    
    def _get_agreement_note(self, agreement_level: str, popular: float, alternative: float) -> str:
        """Generate note about agreement between metric categories"""
        if agreement_level == "high":
            return f"Strong consensus between popular ({popular:.1f}) and alternative ({alternative:.1f}) metrics increases confidence"
        elif agreement_level == "low":
            return f"Divergence between popular ({popular:.1f}) and alternative ({alternative:.1f}) metrics suggests caution"
        else:
            return f"Moderate agreement between popular ({popular:.1f}) and alternative ({alternative:.1f}) metrics"
    
    def _generate_insights(
        self,
        popular_results: Dict[str, Any],
        alternative_results: Dict[str, Any],
        final_score: Dict[str, Any]
    ) -> List[str]:
        """Generate detailed insights from the analysis"""
        insights = []
        
        # Overall assessment
        score = final_score["score"]
        direction = final_score["direction"]
        interpretation = final_score["interpretation"]
        
        insights.append(f"{'='*80}")
        insights.append(f"COMPREHENSIVE MACROECONOMIC ANALYSIS REPORT")
        insights.append(f"{'='*80}")
        insights.append("")
        
        insights.append(f"=== OVERALL MARKET DIRECTION ===")
        insights.append(f"Confidence Score: {score}/100 - {direction}")
        insights.append(f"Interpretation: {interpretation}")
        insights.append(f"Timeframe: Medium-term (3-12 months)")
        insights.append("")
        
        # Agreement analysis
        agreement = final_score["agreement"]
        insights.append(f"=== METRIC AGREEMENT ANALYSIS ===")
        insights.append(f"Agreement Level: {agreement['level'].upper()}")
        insights.append(f"Score Difference: {agreement['difference']:.1f} points")
        insights.append(f"{agreement['note']}")
        insights.append("")
        
        # Detailed popular metrics review
        insights.append(f"{'='*80}")
        insights.append(f"POPULAR METRICS DETAILED REVIEW")
        insights.append(f"Category Score: {final_score['components']['popular_metrics_score']}/100")
        insights.append(f"Weight in Final Score: {final_score['components']['popular_weight']*100:.0f}%")
        insights.append(f"{'='*80}")
        insights.append("")
        
        popular_breakdown = popular_results["composite_score"]["breakdown"]
        popular_raw_data = popular_results["raw_data"]
        
        # Sort by metric name for organized presentation
        sorted_popular = sorted(popular_breakdown, key=lambda x: x["metric"])
        
        for item in sorted_popular:
            metric_name = item['metric']
            metric_title = metric_name.replace('_', ' ').title()
            metric_score = item['score']
            metric_weight = item['weight']
            contribution = item['weighted_contribution']
            
            # Determine sentiment indicator
            if metric_score >= 70:
                sentiment = "🟢 VERY BULLISH"
            elif metric_score >= 60:
                sentiment = "🟢 BULLISH"
            elif metric_score >= 55:
                sentiment = "🟡 SLIGHTLY BULLISH"
            elif metric_score >= 45:
                sentiment = "🟡 NEUTRAL"
            elif metric_score >= 40:
                sentiment = "🟠 SLIGHTLY BEARISH"
            elif metric_score >= 30:
                sentiment = "🔴 BEARISH"
            else:
                sentiment = "🔴 VERY BEARISH"
            
            insights.append(f"┌─ {metric_title}")
            insights.append(f"│  Score: {metric_score:.1f}/100 {sentiment}")
            insights.append(f"│  Weight: {metric_weight*100:.1f}% | Contribution: {contribution:.2f} points")
            insights.append(f"│  Analysis: {item['reasoning']}")
            
            # Add raw data details if available
            raw = popular_raw_data.get(metric_name, {})
            if raw and not isinstance(raw, Exception):
                insights.append(f"│  Data Details:")
                if 'current' in raw:
                    insights.append(f"│    • Current Value: {raw['current']:.2f}")
                if 'yoy_change_pct' in raw:
                    insights.append(f"│    • YoY Change: {raw['yoy_change_pct']:+.2f}%")
                if 'trend' in raw:
                    insights.append(f"│    • Trend: {raw['trend'].title()}")
                if '3m_return' in raw:
                    insights.append(f"│    • 3-Month Return: {raw['3m_return']:+.2f}%")
                if '1y_return' in raw:
                    insights.append(f"│    • 1-Year Return: {raw['1y_return']:+.2f}%")
                if 'date' in raw:
                    insights.append(f"│    • Data Date: {raw['date']}")
            
            insights.append(f"└─")
            insights.append("")
        
        # Detailed alternative metrics review
        insights.append(f"{'='*80}")
        insights.append(f"ALTERNATIVE METRICS DETAILED REVIEW")
        insights.append(f"Category Score: {final_score['components']['alternative_metrics_score']}/100")
        insights.append(f"Weight in Final Score: {final_score['components']['alternative_weight']*100:.0f}%")
        insights.append(f"{'='*80}")
        insights.append("")
        
        alt_breakdown = alternative_results["composite_score"]["breakdown"]
        alt_raw_data = alternative_results["raw_data"]
        
        sorted_alt = sorted(alt_breakdown, key=lambda x: x["metric"])
        
        for item in sorted_alt:
            metric_name = item['metric']
            metric_title = metric_name.replace('_', ' ').title()
            metric_score = item['score']
            metric_weight = item['weight']
            contribution = item['weighted_contribution']
            
            # Determine sentiment indicator
            if metric_score >= 70:
                sentiment = "🟢 VERY BULLISH"
            elif metric_score >= 60:
                sentiment = "🟢 BULLISH"
            elif metric_score >= 55:
                sentiment = "🟡 SLIGHTLY BULLISH"
            elif metric_score >= 45:
                sentiment = "🟡 NEUTRAL"
            elif metric_score >= 40:
                sentiment = "🟠 SLIGHTLY BEARISH"
            elif metric_score >= 30:
                sentiment = "🔴 BEARISH"
            else:
                sentiment = "🔴 VERY BEARISH"
            
            insights.append(f"┌─ {metric_title}")
            insights.append(f"│  Score: {metric_score:.1f}/100 {sentiment}")
            insights.append(f"│  Weight: {metric_weight*100:.1f}% | Contribution: {contribution:.2f} points")
            insights.append(f"│  Analysis: {item['reasoning']}")
            
            # Add raw data details if available
            raw = alt_raw_data.get(metric_name, {})
            if raw and not isinstance(raw, Exception):
                insights.append(f"│  Data Details:")
                if 'current' in raw:
                    insights.append(f"│    • Current Value: {raw['current']:.2f}")
                if 'inverted' in raw:
                    insights.append(f"│    • Yield Curve: {'INVERTED ⚠' if raw['inverted'] else 'Normal ✓'}")
                if '10y_yield' in raw and '2y_yield' in raw:
                    insights.append(f"│    • 10Y Yield: {raw['10y_yield']:.2f}% | 2Y Yield: {raw['2y_yield']:.2f}%")
                if 'yoy_change_pct' in raw:
                    insights.append(f"│    • YoY Change: {raw['yoy_change_pct']:+.2f}%")
                if 'trend' in raw:
                    insights.append(f"│    • Trend: {raw['trend'].title()}")
                if '3m_change_pct' in raw:
                    insights.append(f"│    • 3-Month Change: {raw['3m_change_pct']:+.2f}%")
                if '1m_change_pct' in raw:
                    insights.append(f"│    • 1-Month Change: {raw['1m_change_pct']:+.2f}%")
                if 'date' in raw:
                    insights.append(f"│    • Data Date: {raw['date']}")
            
            insights.append(f"└─")
            insights.append("")
        
        # Summary of strongest and weakest signals
        insights.append(f"{'='*80}")
        insights.append(f"SUMMARY: TOP SIGNALS")
        insights.append(f"{'='*80}")
        insights.append("")
        
        all_scores = popular_breakdown + alt_breakdown
        
        insights.append("🟢 TOP 5 BULLISH SIGNALS (Highest Scores):")
        sorted_all = sorted(all_scores, key=lambda x: x["score"], reverse=True)
        for i, item in enumerate(sorted_all[:5], 1):
            insights.append(f"  {i}. {item['metric'].replace('_', ' ').title()}: {item['score']:.1f}/100")
            insights.append(f"     └─ {item['reasoning']}")
        
        insights.append("")
        insights.append("🔴 TOP 5 BEARISH SIGNALS (Lowest Scores):")
        for i, item in enumerate(sorted_all[-5:][::-1], 1):
            insights.append(f"  {i}. {item['metric'].replace('_', ' ').title()}: {item['score']:.1f}/100")
            insights.append(f"     └─ {item['reasoning']}")
        
        insights.append("")
        
        # Key risk factors
        insights.append(f"⚠ CRITICAL RISK FACTORS (Score < 40):")
        risk_factors = [item for item in all_scores if item["score"] < 40]
        
        if risk_factors:
            for item in sorted(risk_factors, key=lambda x: x["score"]):
                insights.append(f"  • {item['metric'].replace('_', ' ').title()}: {item['score']:.1f}/100")
                insights.append(f"    └─ {item['reasoning']}")
        else:
            insights.append("  ✓ No critical risk factors identified (all metrics above 40)")
        
        insights.append("")
        
        # Positive catalysts
        insights.append(f"✓ STRONG POSITIVE CATALYSTS (Score ≥ 70):")
        catalysts = [item for item in all_scores if item["score"] >= 70]
        
        if catalysts:
            for item in sorted(catalysts, key=lambda x: x["score"], reverse=True):
                insights.append(f"  • {item['metric'].replace('_', ' ').title()}: {item['score']:.1f}/100")
                insights.append(f"    └─ {item['reasoning']}")
        else:
            insights.append("  • No metrics showing very strong positive signals (≥70)")
        
        insights.append("")
        insights.append(f"{'='*80}")
        insights.append(f"END OF DETAILED INDICATOR REVIEW")
        insights.append(f"{'='*80}")
        
        return insights
    
    def _generate_recommendations(self, final_score: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations based on confidence score"""
        score = final_score["score"]
        direction = final_score["direction"]
        agreement = final_score["agreement"]["level"]
        
        recommendations = []
        
        # Strategic positioning
        recommendations.append("=== STRATEGIC POSITIONING ===")
        
        if score >= 70:
            recommendations.append("• Portfolio Stance: AGGRESSIVE GROWTH")
            recommendations.append("• Equity Allocation: Consider overweight (70-85%)")
            recommendations.append("• Sector Focus: Growth, Technology, Cyclicals")
            recommendations.append("• Risk Tolerance: Can take higher beta positions")
        elif score >= 60:
            recommendations.append("• Portfolio Stance: MODERATE GROWTH")
            recommendations.append("• Equity Allocation: Slight overweight (60-70%)")
            recommendations.append("• Sector Focus: Quality growth, selective cyclicals")
            recommendations.append("• Risk Tolerance: Moderate risk acceptable")
        elif score >= 55:
            recommendations.append("• Portfolio Stance: NEUTRAL WITH GROWTH TILT")
            recommendations.append("• Equity Allocation: Market weight (55-60%)")
            recommendations.append("• Sector Focus: Balanced across growth and defensive")
            recommendations.append("• Risk Tolerance: Maintain quality bias")
        elif score >= 45:
            recommendations.append("• Portfolio Stance: BALANCED NEUTRAL")
            recommendations.append("• Equity Allocation: Balanced (50-55%)")
            recommendations.append("• Sector Focus: Diversified, quality companies")
            recommendations.append("• Risk Tolerance: Conservative approach")
        elif score >= 40:
            recommendations.append("• Portfolio Stance: NEUTRAL WITH DEFENSIVE TILT")
            recommendations.append("• Equity Allocation: Slight underweight (45-50%)")
            recommendations.append("• Sector Focus: Defensive sectors, dividends")
            recommendations.append("• Risk Tolerance: Low beta preferred")
        elif score >= 30:
            recommendations.append("• Portfolio Stance: DEFENSIVE")
            recommendations.append("• Equity Allocation: Underweight (35-45%)")
            recommendations.append("• Sector Focus: Utilities, Healthcare, Consumer Staples")
            recommendations.append("• Risk Tolerance: Minimize volatility")
        else:
            recommendations.append("• Portfolio Stance: CAPITAL PRESERVATION")
            recommendations.append("• Equity Allocation: Minimal (20-35%)")
            recommendations.append("• Sector Focus: Only highest quality defensives")
            recommendations.append("• Risk Tolerance: Avoid risk, preserve capital")
        
        recommendations.append("")
        recommendations.append("=== TACTICAL CONSIDERATIONS ===")
        
        # Adjust for agreement level
        if agreement == "low":
            recommendations.append("⚠ LOW AGREEMENT WARNING:")
            recommendations.append("  • Popular and alternative metrics diverging")
            recommendations.append("  • Consider smaller position sizes")
            recommendations.append("  • Increase diversification")
            recommendations.append("  • Monitor for resolution of divergence")
        elif agreement == "high":
            recommendations.append("✓ HIGH AGREEMENT CONFIRMATION:")
            recommendations.append("  • Both metric sets aligned")
            recommendations.append("  • Higher confidence in directional view")
            recommendations.append("  • Can size positions more aggressively")
        
        recommendations.append("")
        
        # Specific tactical moves
        if direction == "BULLISH":
            recommendations.append("• Consider increasing duration in equities")
            recommendations.append("• Look for breakout opportunities")
            recommendations.append("• Use pullbacks as buying opportunities")
        elif direction == "BEARISH":
            recommendations.append("• Consider raising cash levels")
            recommendations.append("• Tighten stop-losses on existing positions")
            recommendations.append("• Avoid chasing rallies")
        else:
            recommendations.append("• Maintain current allocation")
            recommendations.append("• Wait for clearer signals before major moves")
            recommendations.append("• Focus on company-specific fundamentals")
        
        recommendations.append("")
        recommendations.append("=== MONITORING ===")
        recommendations.append("• Re-evaluate monthly or when major data releases occur")
        recommendations.append("• Watch for changes in metric agreement levels")
        recommendations.append("• Pay attention to threshold crossings (e.g., score moving from 62 to 58)")
        
        return recommendations
    
    def _create_summary(
        self,
        final_score: Dict[str, Any],
        popular_results: Dict[str, Any],
        alternative_results: Dict[str, Any]
    ) -> str:
        """Create concise summary of macro analysis"""
        score = final_score["score"]
        direction = final_score["direction"]
        popular_score = final_score["components"]["popular_metrics_score"]
        alt_score = final_score["components"]["alternative_metrics_score"]
        agreement = final_score["agreement"]["level"]
        
        summary = (
            f"MACRO CONFIDENCE SCORE: {score}/100 ({direction}) | "
            f"Popular Metrics: {popular_score} | Alternative Metrics: {alt_score} | "
            f"Agreement: {agreement.upper()} | "
            f"Medium-term outlook (3-12 months): {final_score['interpretation']}"
        )
        
        return summary


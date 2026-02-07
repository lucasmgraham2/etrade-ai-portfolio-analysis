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
        self.analysis_date = config.get("analysis_date") if config else None
        
        # Load weights configuration
        self.weights_config = self._load_weights_config()
        
        # Initialize analyzers
        popular_weights = self.weights_config["popular_metrics"]["individual_weights"]
        alternative_weights = self.weights_config["alternative_metrics"]["individual_weights"]
        
        self.popular_analyzer = PopularMetricsAnalyzer(self.api_keys, popular_weights, self.analysis_date)
        self.alternative_analyzer = AlternativeMetricsAnalyzer(self.api_keys, alternative_weights, self.analysis_date)
        
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
        
        with open(config_path, 'r') as f:
            return json.load(f)
    
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
            
            # Detect market regime
            regime = self._detect_market_regime(
                final_score, popular_results, alternative_results
            )
            
            # Generate insights and recommendations
            insights = self._generate_insights(popular_results, alternative_results, final_score, regime)
            recommendations = self._generate_recommendations(final_score)
            summary = self._create_summary(final_score, popular_results, alternative_results, regime)
            
            self.log("="*60)
            self.log(f"FINAL CONFIDENCE SCORE: {final_score['score']}/100 ({final_score['direction'].upper()})")
            self.log(f"MARKET REGIME: {regime['regime'].upper()} - {regime['interpretation']}")
            self.log("="*60)
            
            return {
                "summary": summary,
                "confidence_score": final_score,
                "market_regime": regime,
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
    
    def _calculate_final_confidence_score(
        self,
        popular_results: Dict[str, Any],
        alternative_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Calculate final confidence score combining popular and alternative metrics
        Applies crisis detection multipliers for recession/credit stress warnings
        Applies recovery detection boost for bull run entry signals
        
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
            # Large divergence - need to check if crisis active before applying recovery logic
            # First, run crisis detectors to see if any crisis flags are active
            _, preliminary_flags = self._apply_crisis_multipliers(
                popular_results, alternative_results
            )
            crisis_detected = any([
                preliminary_flags.get("credit_stress_cliff", False),
                preliminary_flags.get("recession_detector", False),
                preliminary_flags.get("inflation_confidence_combo", False)
            ])
            
            # If alternatives >> popular AND alternatives > 60 AND NO crisis, it's recovery entry
            if alternative_score > popular_score and alternative_score > 60 and not crisis_detected:
                # Recovery entry detected - boost confidence instead of penalizing
                final_score += 8  # Boost for forward-looking recovery signal
                agreement_level = "high_divergence_recovery"
            else:
                # Traditional caution on divergence or crisis override
                final_score += thresholds["low_agreement_penalty"]
                agreement_level = "low"
        else:
            agreement_level = "moderate"
        
        # Apply crisis detection multipliers AND recovery boost
        multiplier_factors, crisis_flags = self._apply_crisis_multipliers(
            popular_results, alternative_results
        )
        
        final_score = final_score * multiplier_factors
        
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
            },
            "crisis_detection": {
                "multiplier_applied": multiplier_factors,
                "flags": crisis_flags
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
    
    def _apply_crisis_multipliers(
        self,
        popular_results: Dict[str, Any],
        alternative_results: Dict[str, Any]
    ) -> tuple:
        """
        Apply crisis detection multipliers to identify recession/credit stress conditions.
        
        Returns:
            Tuple of (multiplier_factor, crisis_flags_dict)
        """
        multiplier = 1.0
        flags = {
            "credit_stress_cliff": False,
            "recession_detector": False,
            "inflation_confidence_combo": False,
            "recovery_detector": False
        }
        
        multipliers_config = self.weights_config["confidence_adjustments"]["crisis_multipliers"]
        
        # Extract metric values for analysis
        pop_breakdown = {item["metric"]: item for item in popular_results["composite_score"]["breakdown"]}
        alt_breakdown = {item["metric"]: item for item in alternative_results["composite_score"]["breakdown"]}
        pop_data = popular_results["raw_data"]
        alt_data = alternative_results["raw_data"]
        
        # CRISIS DETECTOR 1: Credit Stress Cliff
        # Triggers when HY spreads > 5% AND LEI < 25
        hy_score = alt_breakdown.get("high_yield_spreads", {}).get("score", 100)
        lei_score = alt_breakdown.get("leading_economic_index", {}).get("score", 100)
        
        if hy_score < 50 and lei_score < 30:  # Both indicate stress
            multiplier *= multipliers_config["credit_stress_cliff"]["multiplier"]
            flags["credit_stress_cliff"] = True
            self.log(f"CREDIT STRESS CLIFF DETECTED: HY Spreads score={hy_score:.1f}, LEI score={lei_score:.1f}", "WARNING")
        
        # CRISIS DETECTOR 2: Recession Detector
        # Triggers when 2+ of the following:
        # - Sahm Rule > 0.4
        # - LEI < 30
        # - Jobless Claims +10% YoY
        # - Consumer Confidence -10% YoY
        recession_triggers = 0
        sahm_score = alt_breakdown.get("sahm_rule", {}).get("score", 60)
        if sahm_score < 50:  # Elevated
            recession_triggers += 1
        
        lei_score = alt_breakdown.get("leading_economic_index", {}).get("score", 100)
        if lei_score < 30:
            recession_triggers += 1
        
        jobless_score = pop_breakdown.get("jobless_claims", {}).get("score", 50)
        if jobless_score < 40:  # Deteriorating
            recession_triggers += 1
        
        consumer_conf_score = pop_breakdown.get("consumer_confidence", {}).get("score", 50)
        if consumer_conf_score < 40:  # Declining sharply
            recession_triggers += 1
        
        if recession_triggers >= 2:
            multiplier *= multipliers_config["recession_detector"]["multiplier"]
            flags["recession_detector"] = True
            self.log(f"RECESSION WARNING: {recession_triggers} recession indicators triggered", "WARNING")
        
        # CRISIS DETECTOR 3: Inflation + Confidence Collapse (Stagflation)
        # Triggers when CPI > 3% AND Consumer Confidence down >10% YoY
        cpi_score = pop_breakdown.get("cpi_inflation", {}).get("score", 50)
        cpi_data = pop_data.get("cpi", {})
        cpi_yoy = cpi_data.get("yoy_change_pct", 0)
        
        conf_data = pop_data.get("consumer_confidence", {})
        conf_yoy = conf_data.get("yoy_change_pct", 0)
        
        if cpi_yoy > 3.0 and conf_yoy < -10.0:
            multiplier *= multipliers_config["inflation_confidence_combo"]["multiplier"]
            flags["inflation_confidence_combo"] = True
            self.log(f"STAGFLATION RISK: CPI +{cpi_yoy:.1f}% YoY, Confidence {conf_yoy:.1f}% YoY", "WARNING")
        
        # RECOVERY DETECTOR: Bull Run Entry Signal
        # Triggers when LEI strong + Sahm safe + HY spreads stable + alternatives > 60
        # BUT only if NO crisis flags are active (stagflation, recession, credit stress)
        recovery_config = multipliers_config["recovery_detector"]
        alt_score = alternative_results["composite_score"]["score"]
        
        crisis_active = flags["credit_stress_cliff"] or flags["recession_detector"] or flags["inflation_confidence_combo"]
        
        lei_recovery = lei_score >= recovery_config["lei_score_threshold"]
        sahm_safe = sahm_score >= recovery_config["sahm_rule_safe_threshold"]
        hy_stable = hy_score >= recovery_config["hy_spreads_threshold"]
        alt_bullish = alt_score >= recovery_config["alternative_score_threshold"]
        
        if lei_recovery and sahm_safe and hy_stable and alt_bullish and not crisis_active:
            boost = 1.0 + recovery_config["boost_amount"]
            multiplier *= boost
            flags["recovery_detector"] = True
            self.log(f"RECOVERY DETECTOR ACTIVE: LEI={lei_score:.0f}, Sahm={sahm_score:.0f}, HY={hy_score:.0f}, Alt Score={alt_score:.0f}", "INFO")
        
        return multiplier, flags

    
    def _get_agreement_note(self, agreement_level: str, popular: float, alternative: float) -> str:
        """Generate note about agreement between metric categories"""
        if agreement_level == "high":
            return f"Strong consensus between popular ({popular:.1f}) and alternative ({alternative:.1f}) metrics increases confidence"
        elif agreement_level == "high_divergence_recovery":
            return f"Leading indicators ({alternative:.1f}) ahead of traditional metrics ({popular:.1f}) - bull run entry signal"
        elif agreement_level == "low":
            return f"Divergence between popular ({popular:.1f}) and alternative ({alternative:.1f}) metrics suggests caution"
        else:
            return f"Moderate agreement between popular ({popular:.1f}) and alternative ({alternative:.1f}) metrics"
    
    def _detect_market_regime(
        self,
        final_score: Dict[str, Any],
        popular_results: Dict[str, Any],
        alternative_results: Dict[str, Any]
    ) -> Dict[str, str]:
        """
        Detect current market regime based on score, agreement, and crisis flags.
        
        Regimes:
        - CRASH/CRISIS: Credit stress cliff or recession detected, very low scores (<35)
        - PEAK/INFLECTION: High agreement + very bullish OR stagflation warning + high alt score
        - TROUGH/RECOVERY: Large divergence (alt >> pop) + recovery detector + 55-65 range
        - MID-RALLY: Sustained divergence with climbing score (55-70), no crisis
        - PEAK_STRENGTH: High agreement + very bullish (>70) + no crisis warnings
        """
        score = final_score["score"]
        direction = final_score["direction"]
        agreement = final_score["agreement"]
        crisis = final_score["crisis_detection"]
        
        popular_score = final_score["components"]["popular_metrics_score"]
        alternative_score = final_score["components"]["alternative_metrics_score"]
        agreement_diff = agreement["difference"]
        
        # Check for crisis flags
        has_crisis = any([
            crisis["flags"].get("credit_stress_cliff", False),
            crisis["flags"].get("recession_detector", False),
            crisis["flags"].get("inflation_confidence_combo", False)
        ])
        
        # REGIME 1: CRASH/CRISIS
        if has_crisis and score < 35:
            return {
                "regime": "crash",
                "interpretation": "Market crisis - credit stress, recession, or major confidence collapse",
                "action": "REDUCE RISK - De-risk portfolio, raise cash",
                "confidence": "HIGH"
            }
        
        # REGIME 2: PEAK/INFLECTION
        # High agreement at bullish levels OR stagflation warning blocking recovery
        if has_crisis and score >= 45 and score < 55:
            return {
                "regime": "inflection",
                "interpretation": "Peak inflection point - warning signals present despite bullish metrics",
                "action": "REBALANCE - Reduce exposure, lock in gains",
                "confidence": "HIGH"
            }
        
        # REGIME 3: TROUGH/RECOVERY ENTRY
        # Large divergence with recovery detector active, OR approaching recovery (high alt score with bearish pop)
        if agreement_diff > 20 and not has_crisis:
            if alternative_score > 60 and crisis["flags"].get("recovery_detector", False):
                return {
                    "regime": "trough_recovery",
                    "interpretation": "Bull run entry point - leading indicators surging while traditional metrics lag",
                    "action": "ACCUMULATE - Buy weakness, increase equity exposure",
                    "confidence": "HIGH"
                }
            elif alternative_score > 65 and popular_score < 45 and score >= 45:
                # Recovery approaching - alternatives very strong, populars weak, but stabilizing
                return {
                    "regime": "trough_recovery",
                    "interpretation": "Bull run entry point - leading indicators surging while traditional metrics lag",
                    "action": "ACCUMULATE - Buy weakness, increase equity exposure",
                    "confidence": "MEDIUM"
                }
        
        # REGIME 4: MID-RALLY
        # Sustained divergence (alt >> pop) with score climbing, no crisis, OR moderate divergence with bullish score
        if not has_crisis and score >= 55 and score < 70:
            if agreement_diff > 15 and alternative_score > 60:
                # Clear divergence mid-rally
                return {
                    "regime": "mid_rally",
                    "interpretation": "Continuation rally - leading indicators sustaining advance",
                    "action": "HOLD/ADD - Ride the trend, add on dips",
                    "confidence": "HIGH"
                }
            elif agreement["level"] == "moderate" and score >= 60:
                # Moderate divergence but bullish - transition into full rally
                return {
                    "regime": "mid_rally",
                    "interpretation": "Rally strengthening - traditional metrics catching up to leading indicators",
                    "action": "HOLD/ADD - Ride the trend, add on dips",
                    "confidence": "MEDIUM"
                }
        
        # REGIME 5: PEAK_STRENGTH
        # High agreement + very bullish + no crisis
        if agreement["level"] == "high" and score >= 70 and not has_crisis:
            return {
                "regime": "peak_strength",
                "interpretation": "Peak market euphoria - all signals aligned bullish, elevated valuation risk",
                "action": "PROFIT TAKE - Consider partial profits, increase discipline",
                "confidence": "MEDIUM"
            }
        
        # REGIME 6: NEUTRAL/TRANSITION
        # Moderate agreement or mixed signals
        if 40 <= score < 55 and not has_crisis:
            return {
                "regime": "neutral_transition",
                "interpretation": "Transition period - mixed signals, direction unclear",
                "action": "WAIT - Wait for clarity, maintain balanced positioning",
                "confidence": "MEDIUM"
            }
        
        # REGIME 7: BEARISH_CAUTION
        # Bearish without full crisis
        if score < 40 and not has_crisis:
            return {
                "regime": "bearish_caution",
                "interpretation": "Bearish environment - economic headwinds present",
                "action": "DEFENSIVE - Reduce equity, increase defensives",
                "confidence": "MEDIUM"
            }
        
        # DEFAULT
        return {
            "regime": "neutral",
            "interpretation": f"Neutral sentiment - Score {score}/100, {direction}",
            "action": "MAINTAIN - Hold current allocation",
            "confidence": "LOW"
        }
    def _generate_insights(
        self,
        popular_results: Dict[str, Any],
        alternative_results: Dict[str, Any],
        final_score: Dict[str, Any],
        regime: Dict[str, Any]
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
        insights.append(f"Regime: {regime.get('regime', 'unknown').upper()} - {regime.get('interpretation', 'N/A')}")
        insights.append(f"Suggested stance: {regime.get('action', 'N/A')} (confidence: {regime.get('confidence', 'N/A')})")

        # Add context on agreement and crisis flags for direction depth
        popular_score = final_score["components"]["popular_metrics_score"]
        alt_score = final_score["components"]["alternative_metrics_score"]
        agreement = final_score["agreement"]["level"]
        agreement_diff = final_score["agreement"]["difference"]
        crisis_flags = final_score.get("crisis_detection", {}).get("flags", {})
        active_flags = [
            name.replace("_", " ")
            for name, active in crisis_flags.items()
            if active
        ]
        if alt_score > popular_score:
            insights.append(
                f"Leading indicators are ahead of lagging metrics by {agreement_diff:.1f} points, which can signal early-cycle strength."
            )
        elif popular_score > alt_score:
            insights.append(
                f"Lagging indicators are ahead of leading metrics by {agreement_diff:.1f} points, which can signal late-cycle risk."
            )
        else:
            insights.append("Leading and lagging indicators are closely aligned, reinforcing the directional signal.")

        if active_flags:
            insights.append(f"Active crisis flags: {', '.join(active_flags)} (risk-off bias until cleared).")
        else:
            insights.append("No crisis flags active; risk posture can follow the score trend.")
        insights.append("")
        
        # Agreement analysis
        agreement = final_score["agreement"]
        insights.append(f"=== METRIC AGREEMENT ANALYSIS ===")
        insights.append(f"Agreement Level: {agreement['level'].upper()}")
        insights.append(f"Score Difference: {agreement['difference']:.1f} points")
        insights.append(f"{agreement['note']}")
        insights.append("")
        
        # Summary of strongest and weakest signals
        insights.append(f"{'='*80}")
        insights.append(f"SUMMARY: TOP SIGNALS")
        insights.append(f"{'='*80}")
        insights.append("")

        popular_breakdown = popular_results["composite_score"]["breakdown"]
        alt_breakdown = alternative_results["composite_score"]["breakdown"]
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
        insights.append(f"[WARNING] CRITICAL RISK FACTORS (Score < 40):")
        risk_factors = [item for item in all_scores if item["score"] < 40]
        
        if risk_factors:
            for item in sorted(risk_factors, key=lambda x: x["score"]):
                insights.append(f"  • {item['metric'].replace('_', ' ').title()}: {item['score']:.1f}/100")
                insights.append(f"    └─ {item['reasoning']}")
        else:
            insights.append("  [OK] No critical risk factors identified (all metrics above 40)")
        
        insights.append("")
        
        # Positive catalysts
        insights.append(f"[OK] STRONG POSITIVE CATALYSTS (Score ≥ 70):")
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
            recommendations.append("[WARNING] LOW AGREEMENT WARNING:")
            recommendations.append("  • Popular and alternative metrics diverging")
            recommendations.append("  • Consider smaller position sizes")
            recommendations.append("  • Increase diversification")
            recommendations.append("  • Monitor for resolution of divergence")
        elif agreement == "high":
            recommendations.append("[OK] HIGH AGREEMENT CONFIRMATION:")
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
        alternative_results: Dict[str, Any],
        regime: Dict[str, Any]
    ) -> str:
        """Create detailed summary of macro analysis"""
        score = final_score["score"]
        direction = final_score["direction"]
        popular_score = final_score["components"]["popular_metrics_score"]
        alt_score = final_score["components"]["alternative_metrics_score"]
        agreement = final_score["agreement"]["level"]
        agreement_diff = final_score["agreement"]["difference"]
        
        # Get crisis flags
        crisis_flags = final_score.get("crisis_detection", {}).get("flags", {})
        
        # Build crisis detection section
        crisis_lines = []
        crisis_active = False
        
        if crisis_flags.get("credit_stress_cliff"):
            crisis_lines.append("  [!] CREDIT STRESS CLIFF: HY bond spreads elevated or spiking rapidly")
            crisis_active = True
        else:
            crisis_lines.append("  [OK] Credit Stress: Not detected")
        
        if crisis_flags.get("recession_detector"):
            crisis_lines.append("  [!] RECESSION RISK: 2+ recession indicators triggered")
            crisis_active = True
        else:
            crisis_lines.append("  [OK] Recession Risk: Not detected")
        
        if crisis_flags.get("inflation_confidence_combo"):
            # Get actual CPI and confidence numbers if available
            pop_data = popular_results.get("raw_data", {})
            cpi_data = pop_data.get("cpi", {})
            conf_data = pop_data.get("consumer_confidence", {})
            cpi_yoy = cpi_data.get("yoy_change_pct", 0)
            conf_yoy = conf_data.get("yoy_change_pct", 0)
            crisis_lines.append(f"  [!] STAGFLATION WARNING: CPI +{cpi_yoy:.1f}% YoY, Confidence {conf_yoy:.1f}% YoY")
            crisis_active = True
        else:
            crisis_lines.append("  [OK] Stagflation Risk: Not detected")
        
        if crisis_flags.get("recovery_detector"):
            crisis_lines.append("  [+] RECOVERY DETECTOR ACTIVE: Bull run entry signal detected")
        
        # Build metric breakdown - top 3 from each category
        popular_breakdown = popular_results["composite_score"]["breakdown"]
        alt_breakdown = alternative_results["composite_score"]["breakdown"]
        
        # Sort by score and get top 3
        popular_sorted = sorted(popular_breakdown, key=lambda x: x["score"], reverse=True)
        alt_sorted = sorted(alt_breakdown, key=lambda x: x["score"], reverse=True)
        
        popular_lines = []
        for i, item in enumerate(popular_sorted[:3], 1):
            metric_name = item['metric'].replace('_', ' ').title()
            metric_score = item['score']
            status = "Strong" if metric_score >= 60 else "Weak" if metric_score < 40 else "Moderate"
            popular_lines.append(f"  {i}. {metric_name}: {metric_score:.0f}/100 ({status})")
        
        alt_lines = []
        for i, item in enumerate(alt_sorted[:3], 1):
            metric_name = item['metric'].replace('_', ' ').title()
            metric_score = item['score']
            status = "Strong" if metric_score >= 60 else "Weak" if metric_score < 40 else "Moderate"
            alt_lines.append(f"  {i}. {metric_name}: {metric_score:.0f}/100 ({status})")
        
        # Build agreement analysis
        if agreement_diff > 20:
            if alt_score > popular_score:
                agreement_note = (
                    f"Leading indicators ({alt_score:.1f}) are significantly ahead of traditional metrics ({popular_score:.1f}) "
                    f"by {agreement_diff:.1f} points - suggests early stage strength not yet reflected in backward-looking data."
                )
            else:
                agreement_note = (
                    f"Traditional metrics ({popular_score:.1f}) are ahead of leading indicators ({alt_score:.1f}) "
                    f"by {agreement_diff:.1f} points - caution warranted as forward signals weaker."
                )
        elif agreement_diff <= 10:
            agreement_note = (
                f"High agreement between popular ({popular_score:.1f}) and alternative ({alt_score:.1f}) metrics "
                f"increases confidence in direction."
            )
        else:
            agreement_note = (
                f"Moderate divergence between popular ({popular_score:.1f}) and alternative ({alt_score:.1f}) metrics - "
                f"mixed signals suggest transitional phase."
            )
        
        # Assemble full summary
        summary_parts = [
            f"CONFIDENCE SCORE: {score:.1f}/100 ({direction.upper()})",
            f"Popular Metrics: {popular_score:.1f} | Alternative Metrics: {alt_score:.1f} | Agreement: {agreement.upper()}",
            "",
            f"[FORWARD OUTLOOK: 6-12 month projection]",
            "",
            f"MARKET REGIME: {regime.get('regime', 'unknown').upper()}",
            f"  Interpretation: {regime.get('interpretation', 'N/A')}",
            f"  Recommended Action: {regime.get('action', 'N/A')}",
            f"  Confidence: {regime.get('confidence', 'N/A')}",
            "",
            "CRISIS DETECTION:",
            "\n".join(crisis_lines),
            "",
            "METRIC BREAKDOWN:",
            f"Popular Metrics (Lagging indicators, 45% weight):",
            "\n".join(popular_lines),
            "",
            f"Alternative Metrics (Leading indicators, 55% weight):",
            "\n".join(alt_lines),
            "",
            f"AGREEMENT ANALYSIS: {agreement_note}",
            "",
            f"MEDIUM-TERM OUTLOOK: {final_score['interpretation']}"
        ]
        
        return "\n".join(summary_parts)


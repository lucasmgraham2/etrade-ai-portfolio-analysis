"""
Agent Orchestrator
Coordinates execution of multiple agents and combines their outputs
"""

import asyncio
from typing import Dict, Any, List
from datetime import datetime
import json
from .base_agent import BaseAgent


class AgentOrchestrator:
    """Orchestrates multiple agents for comprehensive portfolio analysis"""
    
    def __init__(self, portfolio_data: Dict[str, Any]):
        """
        Initialize the orchestrator
        
        Args:
            portfolio_data: Portfolio data from E*TRADE
        """
        self.portfolio_data = portfolio_data
        self.agents: List[BaseAgent] = []
        self.results = {}
        self.execution_order = []
        
    def register_agent(self, agent: BaseAgent):
        """
        Register an agent with the orchestrator
        
        Args:
            agent: Agent instance to register
        """
        self.agents.append(agent)
        self.execution_order.append(agent.name)
        print(f"Registered agent: {agent.name}")
        
    async def run_sequential(self) -> Dict[str, Any]:
        """
        Run agents sequentially, passing results to next agents
        
        Returns:
            Combined results from all agents
        """
        print("\n" + "="*60)
        print("Starting Portfolio Analysis")
        print("="*60 + "\n")
        
        start_time = datetime.now()
        context = {"portfolio": self.portfolio_data}
        
        for agent in self.agents:
            # Execute agent with accumulated context
            agent_output = await agent.execute(context)
            
            # Store results
            self.results[agent.name] = agent_output
            self.execution_order.append(agent.name)
            
            # Add agent results to context for next agents
            context[agent.name] = agent_output["results"]
            
        end_time = datetime.now()
        total_time = (end_time - start_time).total_seconds()
        
        print("\n" + "="*60)
        print(f"Analysis Complete ({total_time:.2f}s)")
        print("="*60 + "\n")
        
        return self._compile_results(total_time)
    
    async def run_parallel(self, parallel_groups: List[List[str]] = None) -> Dict[str, Any]:
        """
        Run agents in parallel groups for better performance
        
        Args:
            parallel_groups: List of agent name groups to run in parallel
                           e.g., [["Sentiment", "Macro"], ["Sector"], ["Integrator"]]
                           
        Returns:
            Combined results from all agents
        """
        print("\n" + "="*60)
        print("Starting Multi-Agent Portfolio Analysis (Parallel Mode)")
        print("="*60 + "\n")
        
        start_time = datetime.now()
        context = {"portfolio": self.portfolio_data}
        
        if not parallel_groups:
            # Default: Run all independent agents in parallel, then integrator
            independent_agents = [a.name for a in self.agents if a.name != "Integrator"]
            parallel_groups = [independent_agents, ["Integrator"]]
        
        # Execute each parallel group
        for group_idx, group_names in enumerate(parallel_groups):
            print(f"\n--- Parallel Group {group_idx + 1}: {', '.join(group_names)} ---\n")
            
            # Get agents in this group
            group_agents = [a for a in self.agents if a.name in group_names]
            
            # Run agents in parallel
            tasks = [agent.execute(context) for agent in group_agents]
            group_results = await asyncio.gather(*tasks)
            
            # Update context with results
            for agent, output in zip(group_agents, group_results):
                self.results[agent.name] = output
                self.execution_order.append(agent.name)
                context[agent.name] = output["results"]
        
        end_time = datetime.now()
        total_time = (end_time - start_time).total_seconds()
        
        print("\n" + "="*60)
        print(f"Analysis Complete ({total_time:.2f}s)")
        print("="*60 + "\n")
        
        return self._compile_results(total_time)
    
    def _compile_results(self, total_time: float) -> Dict[str, Any]:
        """
        Compile all agent results into final output
        
        Args:
            total_time: Total execution time in seconds
            
        Returns:
            Compiled results dictionary
        """
        # Count successes and failures
        successful = sum(1 for r in self.results.values() if r["status"] == "completed")
        failed = sum(1 for r in self.results.values() if r["status"] == "failed")
        
        compiled = {
            "timestamp": datetime.now().isoformat(),
            "total_execution_time": total_time,
            "agents_executed": len(self.agents),
            "execution_order": self.execution_order,
            "successful_agents": successful,
            "failed_agents": failed,
            "portfolio_summary": self._get_portfolio_summary(),
            "agent_results": self.results
        }
        
        return compiled
    
    def _get_portfolio_summary(self) -> Dict[str, Any]:
        """Extract key portfolio metrics"""
        summary = self.portfolio_data.get("summary", {})
        return {
            "total_value": summary.get("total_portfolio_value", 0),
            "total_positions": summary.get("total_positions", 0),
            "symbols": summary.get("unique_symbols", [])
        }
    
    def save_results(self, filepath: str = None):
        """
        Save analysis results to file
        
        Args:
            filepath: Optional custom filepath
        """
        if not filepath:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"multi_agent_analysis_{timestamp}.json"
            
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2)
            
        print(f"Results saved to: {filepath}")
        
    def generate_report(self) -> str:
        """
        Generate human-readable report from analysis
        
        Returns:
            Formatted text report
        """
        report_lines = [
            "="*80,
            "MULTI-AGENT PORTFOLIO ANALYSIS REPORT",
            "="*80,
            f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "\n" + "-"*80,
            "\nPORTFOLIO OVERVIEW",
            "-"*80
        ]
        
        # Portfolio summary
        summary = self._get_portfolio_summary()
        report_lines.extend([
            f"Total Portfolio Value: ${summary['total_value']:,.2f}",
            f"Total Positions: {summary['total_positions']}",
            f"Symbols: {', '.join(summary['symbols'])}",
        ])
        
        # Agent results
        for agent_name in self.execution_order:
            agent_result = self.results[agent_name]
            report_lines.extend([
                f"\n{'-'*80}",
                f"\n{agent_name.upper()}",
                f"{'-'*80}",
                f"Status: {agent_result['status']}",
                f"Execution Time: {agent_result['execution_time']:.2f}s"
            ])
            
            # Add agent-specific results
            if agent_result['status'] == 'completed':
                results = agent_result['results']
                if 'summary' in results and agent_name != "Integrator":
                    report_lines.append(f"\n{results['summary']}")
                
                # Add AI reasoning if available
                if 'ai_reasoning' in results and results['ai_reasoning']:
                    report_lines.append("\nAI Analysis:")
                    report_lines.append(f"  {results['ai_reasoning']}")
                
                if 'recommendations' in results:
                    report_lines.append("\nRecommendations:")
                    for rec in results['recommendations']:
                        report_lines.append(f"  • {rec}")

                # Provide richer detail for integrator output
                if agent_name == "Integrator":
                    report_lines.append("\nExecutive Summary:")
                    report_lines.append(f"  {results.get('summary', 'No summary available')}")

                    risk = results.get('risk_assessment', {})
                    if risk:
                        report_lines.append("\nRisk Overview:")
                        report_lines.append(f"  Overall: {risk.get('overall_risk', 'UNKNOWN')}")
                        report_lines.append(
                            f"  Concentration: {risk.get('concentration_risk', 'UNKNOWN')} (largest position {risk.get('largest_position_pct', 0)}%)"
                        )
                        report_lines.append(
                            f"  Sentiment: {risk.get('sentiment_risk', 'UNKNOWN')} ({risk.get('bearish_positions_pct', 0)}% bearish)"
                        )
                        report_lines.append(
                            f"  Underperformance: {risk.get('underperformance_risk', 'UNKNOWN')} ({risk.get('losing_positions_pct', 0)}% losing)"
                        )

                    sector_highlights = (
                        results.get('portfolio_recommendations', {})
                        .get('sector_highlights', {})
                    )
                    if sector_highlights:
                        report_lines.append("\nSector Outlook:")
                        top_fav = sector_highlights.get('top_favorable', [])
                        if top_fav:
                            top_fav_text = ", ".join(
                                [f"{p['sector']} (score {p['score']:.1f})" for p in top_fav]
                            )
                            report_lines.append(f"  Top favorable: {top_fav_text}")
                        least_fav = sector_highlights.get('least_favorable', [])
                        if least_fav:
                            least_fav_text = ", ".join(
                                [f"{p['sector']} (score {p['score']:.1f})" for p in least_fav]
                            )
                            report_lines.append(f"  Not favorable: {least_fav_text}")
                        underweight = sector_highlights.get('underweight', [])
                        if underweight:
                            report_lines.append(
                                f"  Underweight vs outlook: {', '.join(underweight)}"
                            )
                        overweight = sector_highlights.get('overweight', [])
                        if overweight:
                            report_lines.append(
                                f"  Overweight in lagging sectors: {', '.join(overweight)}"
                            )

                    actions = results.get('action_priorities', [])[:5]
                    if actions:
                        report_lines.append("\nTop Priority Actions:")
                        for action in actions:
                            report_lines.append(
                                f"  • [P{action.get('priority')}] {action.get('action')} ({action.get('rationale')})"
                            )

                    buckets = results.get('recommendation_buckets', {})
                    if buckets:
                        report_lines.append("\nBuy/Hold/Sell Summary:")
                        report_lines.append(
                            f"  Buy: {', '.join(buckets.get('buy', [])[:8]) or 'none'}"
                        )
                        report_lines.append(
                            f"  Hold: {', '.join(buckets.get('hold', [])[:8]) or 'none'}"
                        )
                        report_lines.append(
                            f"  Sell/Trim: {', '.join(buckets.get('sell', [])[:8]) or 'none'}"
                        )
            else:
                report_lines.append(f"\nErrors: {', '.join(agent_result['errors'])}")
        
        report_lines.extend([
            "\n" + "="*80,
            "END OF REPORT",
            "="*80
        ])
        
        return "\n".join(report_lines)

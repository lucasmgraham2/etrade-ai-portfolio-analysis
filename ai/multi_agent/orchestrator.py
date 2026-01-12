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
        print(f"✓ Registered agent: {agent.name}")
        
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
        print(f"✓ Analysis Complete ({total_time:.2f}s)")
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
        print("🚀 Starting Multi-Agent Portfolio Analysis (Parallel Mode)")
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
        print(f"✓ Analysis Complete ({total_time:.2f}s)")
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
            
        print(f"✓ Results saved to: {filepath}")
        
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
                if 'summary' in results:
                    report_lines.append(f"\n{results['summary']}")
                if 'recommendations' in results:
                    report_lines.append("\nRecommendations:")
                    for rec in results['recommendations']:
                        report_lines.append(f"  • {rec}")
            else:
                report_lines.append(f"\nErrors: {', '.join(agent_result['errors'])}")
        
        report_lines.extend([
            "\n" + "="*80,
            "END OF REPORT",
            "="*80
        ])
        
        return "\n".join(report_lines)

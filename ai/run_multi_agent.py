"""
Multi-Agent Portfolio Analysis Runner
Main script to execute all agents and generate comprehensive portfolio analysis
"""

import asyncio
import sys
import os
import json
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv(Path(__file__).parent.parent / ".env")

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from multi_agent.orchestrator import AgentOrchestrator
from multi_agent.sentiment_agent import SentimentAgent
from multi_agent.macro_agent import MacroAgent
from multi_agent.sector_agent import SectorAgent
from multi_agent.integrator_agent import IntegratorAgent


def load_portfolio_data(filepath: str = None) -> dict:
    """Load portfolio data from JSON file, preferring canonical folder"""
    
    if not filepath:
        # Check both potential locations and choose newest file overall
        candidate_dirs = [
            Path(__file__).parent.parent / "etrade" / "etrade_reports",  # canonical
            Path(__file__).parent.parent / "etrade_reports"               # legacy
        ]
        json_candidates = []
        for d in candidate_dirs:
            if d.exists():
                json_candidates.extend(list(d.glob("etrade_data_*.json")))
        
        if not json_candidates:
            raise FileNotFoundError("No portfolio data files found in etrade/etrade_reports or legacy etrade_reports")
        
        filepath = max(json_candidates, key=os.path.getmtime)
        chosen_dir = Path(filepath).parent
        print(f"📂 Loading portfolio data from: {chosen_dir.name}/{Path(filepath).name}")
    
    with open(filepath, 'r') as f:
        return json.load(f)


def load_config() -> dict:
    """
    Load configuration from environment or config file
    
    Config should include:
    - API keys for various services
    - Risk tolerance
    - Analysis parameters
    """
    config = {
        "api_keys": {
            "alpha_vantage": os.getenv("ALPHA_VANTAGE_API_KEY"),
            "newsapi": os.getenv("NEWSAPI_KEY"),
            "fred": os.getenv("FRED_API_KEY"),
            "openai": os.getenv("OPENAI_API_KEY"),
        },
        "risk_tolerance": os.getenv("RISK_TOLERANCE", "moderate"),
        "sentiment": {
            "lookback_days": int(os.getenv("SENTIMENT_LOOKBACK_DAYS", "7"))
        },
        "sector": {
            "lookback_days": int(os.getenv("SECTOR_LOOKBACK_DAYS", "90"))
        }
    }
    
    # Log which APIs are configured
    configured_apis = [k for k, v in config["api_keys"].items() if v]
    if configured_apis:
        print(f"Configured APIs: {', '.join(configured_apis)}")
    else:
        print("No API keys configured - using simulated data")
        print("Add API keys to .env for live market data")
    
    return config


async def run_analysis(portfolio_filepath: str = None, parallel: bool = True):
    """
    Run multi-agent portfolio analysis
    
    Args:
        portfolio_filepath: Path to portfolio JSON file (optional)
        parallel: Whether to run agents in parallel (default: True)
    """
    
    print("\n" + "="*80)
    print("MULTI-AGENT PORTFOLIO ANALYSIS SYSTEM")
    print("="*80)
    
    # Load portfolio data
    try:
        portfolio_data = load_portfolio_data(portfolio_filepath)
        print(f"✓ Portfolio loaded: ${portfolio_data['summary']['total_portfolio_value']:,.2f}")
        print(f"✓ Positions: {portfolio_data['summary']['total_positions']}")
        print(f"✓ Symbols: {', '.join(portfolio_data['summary']['unique_symbols'])}")
    except Exception as e:
        print(f"✗ Error loading portfolio: {str(e)}")
        return
    
    # Load configuration
    config = load_config()
    
    # Create orchestrator
    orchestrator = AgentOrchestrator(portfolio_data)
    
    # Register agents
    print("\n📋 Registering agents...")
    orchestrator.register_agent(SentimentAgent({**config.get("sentiment", {}), "api_keys": config.get("api_keys", {})}))
    orchestrator.register_agent(MacroAgent(config))
    orchestrator.register_agent(SectorAgent({**config.get("sector", {}), "api_keys": config.get("api_keys", {})}))
    orchestrator.register_agent(IntegratorAgent(config))
    
    # Run analysis
    try:
        if parallel:
            # Run independent agents in parallel, then integrator
            results = await orchestrator.run_parallel([
                ["Sentiment", "Macro", "Sector"],
                ["Integrator"]
            ])
        else:
            results = await orchestrator.run_sequential()
        
        # Generate and save report
        print("\nGenerating report...")
        report = orchestrator.generate_report()
        
        # Create output directory and consolidate stray reports from ai/ root
        output_dir = Path(__file__).parent / "analysis_reports"
        output_dir.mkdir(exist_ok=True)
        
        # Move any stray multi-agent reports in ai/ into analysis_reports/
        ai_root = Path(__file__).parent
        for pattern in ["multi_agent_analysis_*.json", "multi_agent_report_*.txt"]:
            for stray in ai_root.glob(pattern):
                try:
                    target = output_dir / stray.name
                    stray.rename(target)
                    print(f"   Moved: {stray.name} → analysis_reports/{stray.name}")
                except Exception as e:
                    print(f"   Warning: Could not move {stray.name}: {e}")
        
        # Delete old analysis files
        print("Cleaning up old analysis files...")
        old_json_files = list(output_dir.glob("multi_agent_analysis_*.json"))
        old_txt_files = list(output_dir.glob("multi_agent_report_*.txt"))
        
        for old_file in old_json_files + old_txt_files:
            try:
                old_file.unlink()
                print(f"   Deleted: {old_file.name}")
            except Exception as e:
                print(f"   Warning: Could not delete {old_file.name}: {e}")
        
        # Save new results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save JSON results (make them JSON-safe by stringifying Exceptions)
        def _make_json_safe(obj):
            """Recursively convert the results tree into JSON-serializable values."""
            if isinstance(obj, Exception):
                return {"error": str(obj), "type": obj.__class__.__name__}
            if isinstance(obj, dict):
                return {k: _make_json_safe(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [ _make_json_safe(v) for v in obj ]
            return obj

        json_file = output_dir / f"multi_agent_analysis_{timestamp}.json"
        with open(json_file, 'w') as f:
            json.dump(_make_json_safe(results), f, indent=2)
        print(f"✓ JSON results saved: {json_file.relative_to(Path(__file__).parent)}")
        
        # Save text report
        report_file = output_dir / f"multi_agent_report_{timestamp}.txt"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"✓ Text report saved: {report_file.relative_to(Path(__file__).parent)}")
        
        # Print executive summary
        print("\n" + "="*80)
        print("📌 EXECUTIVE SUMMARY")
        print("="*80)
        
        integrator_output = results["agent_results"].get("Integrator", {})
        integrator_results = integrator_output.get("results", {})
        
        # Check if integrator had an error
        if integrator_output.get("status") == "failed":
            print(f"\nIntegrator agent encountered an error: {integrator_output.get('errors', ['Unknown error'])}")
            print("     Analysis from other agents is still available in the reports.")
        elif integrator_results:
            print(f"\n{integrator_results.get('summary', 'No summary available')}")
            
            print("\nTOP PRIORITY ACTIONS:")
            for i, action in enumerate(integrator_results.get("action_priorities", [])[:5], 1):
                print(f"  {i}. {action.get('action')}")
                print(f"     → {action.get('rationale')}")
            
            risk = integrator_results.get("risk_assessment", {})
            print(f"\nRISK ASSESSMENT: {risk.get('overall_risk', 'UNKNOWN')}")
            print(f"   • Concentration Risk: {risk.get('concentration_risk', 'UNKNOWN')} ({risk.get('largest_position_pct', 0)}% in largest position)")
            print(f"   • Sentiment Risk: {risk.get('sentiment_risk', 'UNKNOWN')} ({risk.get('bearish_positions_pct', 0)}% bearish)")
            print(f"   • Underperformance Risk: {risk.get('underperformance_risk', 'UNKNOWN')} ({risk.get('losing_positions_pct', 0)}% losing)")
        
        print("\n" + "="*80)
        print(f"Analysis complete! Review full report in:")
        print(f"   {report_file.relative_to(Path(__file__).parent.parent)}")
        print("="*80 + "\n")
        
        return results
        
    except Exception as e:
        print(f"\n✗ Error during analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Main entry point"""
    
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Multi-Agent Portfolio Analysis")
    parser.add_argument(
        "--portfolio",
        help="Path to portfolio JSON file (optional, will use most recent if not specified)"
    )
    parser.add_argument(
        "--sequential",
        action="store_true",
        help="Run agents sequentially instead of parallel"
    )
    
    args = parser.parse_args()
    
    # Run analysis
    asyncio.run(run_analysis(
        portfolio_filepath=args.portfolio,
        parallel=not args.sequential
    ))


if __name__ == "__main__":
    main()

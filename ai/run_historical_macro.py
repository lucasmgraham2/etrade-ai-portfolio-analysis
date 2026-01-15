"""
Historical Macro Analysis Runner
Runs macro analysis as of a specific historical date
"""

import asyncio
import sys
import os
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
import json

# Load environment variables
load_dotenv(Path(__file__).parent.parent / ".env")

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from multi_agent.macro_agent import MacroAgent


async def run_historical_analysis(target_date: str):
    """
    Run macro analysis as of a specific historical date
    
    Args:
        target_date: Date in YYYY-MM-DD format (e.g., "2021-12-31")
    """
    
    print("\n" + "="*80)
    print(f"HISTORICAL MACROECONOMIC ANALYSIS - AS OF {target_date}")
    print("="*80)
    
    # Load configuration
    config = {
        "api_keys": {
            "alpha_vantage": os.getenv("ALPHA_VANTAGE_API_KEY"),
            "fred": os.getenv("FRED_API_KEY"),
        },
        "analysis_date": target_date  # Pass target date to agent
    }
    
    # Check API keys
    if not config["api_keys"]["fred"]:
        print("ERROR: FRED API key not found in .env file")
        print("Please add FRED_API_KEY to your .env file")
        return
    
    if not config["api_keys"]["alpha_vantage"]:
        print("WARNING: Alpha Vantage API key not found")
        print("Some metrics may not be available")
    
    # Create macro agent
    macro_agent = MacroAgent(config)
    
    # Run analysis
    try:
        print(f"\nFetching economic data as of {target_date}...")
        results = await macro_agent.analyze({})
        
        # Save results
        output_dir = Path(__file__).parent / "historical_analysis"
        output_dir.mkdir(exist_ok=True)
        
        date_str = target_date.replace("-", "")
        output_file = output_dir / f"macro_analysis_{date_str}.json"
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✓ Results saved to: {output_file}")
        
        # Print summary
        print("\n" + "="*80)
        print("ANALYSIS RESULTS")
        print("="*80)
        
        confidence = results.get("confidence_score", {})
        print(f"\nMarket Confidence Score: {confidence.get('score', 'N/A')}/100")
        print(f"Direction: {confidence.get('direction', 'N/A')}")
        print(f"Interpretation: {confidence.get('interpretation', 'N/A')}")
        
        agreement = confidence.get("agreement", {})
        print(f"\nMetric Agreement: {agreement.get('level', 'N/A')}")
        print(f"Note: {agreement.get('note', 'N/A')}")
        
        components = confidence.get("components", {})
        print(f"\nPopular Metrics Score: {components.get('popular_metrics_score', 'N/A')}/100")
        print(f"Alternative Metrics Score: {components.get('alternative_metrics_score', 'N/A')}/100")
        
        print("\n" + "="*80)
        print(f"Full analysis saved to: {output_file}")
        print("="*80 + "\n")
        
        return results
        
    except Exception as e:
        print(f"\nERROR: Analysis failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Main entry point"""
    
    # Default to December 31, 2021
    target_date = "2021-12-31"
    
    # Allow custom date from command line
    if len(sys.argv) > 1:
        target_date = sys.argv[1]
    
    # Validate date format
    try:
        datetime.strptime(target_date, "%Y-%m-%d")
    except ValueError:
        print(f"ERROR: Invalid date format: {target_date}")
        print("Please use YYYY-MM-DD format (e.g., 2021-12-31)")
        sys.exit(1)
    
    # Run analysis
    asyncio.run(run_historical_analysis(target_date))


if __name__ == "__main__":
    main()

"""
Master Script - E*TRADE AI Portfolio Analysis
Runs the complete analysis workflow in order:
1. Fetches fresh portfolio data from E*TRADE
2. Runs AI analysis with your chosen provider
3. Displays results and saves to file
"""
import os
import sys
import subprocess
from datetime import datetime

def print_header(title):
    """Print a nice section header"""
    print("\n" + "="*80)
    print(f" {title}")
    print("="*80 + "\n")

def run_script(script_path, description, working_dir=None, interactive=False):
    """Run a Python script and handle errors"""
    print_header(description)
    
    if working_dir is None:
        working_dir = os.path.dirname(script_path)
    
    try:
        if interactive:
            # Run interactively - allows user input and shows output in real-time
            result = subprocess.run(
                [sys.executable, script_path],
                cwd=working_dir,
                check=True
            )
        else:
            # Run silently
            result = subprocess.run(
                [sys.executable, script_path],
                cwd=working_dir,
                check=True,
                text=True
            )
        print(f"\n✓ {description} - COMPLETE\n")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ {description} - FAILED")
        print(f"Error: {e}\n")
        return False
    except Exception as e:
        print(f"\n✗ {description} - ERROR")
        print(f"Error: {e}\n")
        return False

def main():
    """Run the complete portfolio analysis workflow"""
    
    print("\n" + "="*80)
    print(" E*TRADE AI PORTFOLIO ANALYSIS - COMPLETE WORKFLOW")
    print(f" {datetime.now().strftime('%B %d, %Y at %I:%M %p')}")
    print("="*80)
    
    print("\nThis script will:")
    print("  1. ✓ Fetch your latest portfolio data from E*TRADE")
    print("  2. ✓ Generate AI analysis with your chosen provider")
    print("  3. ✓ Display recommendations")
    print("  4. ✓ Save results to timestamped file\n")
    
    response = input("Ready to start? (y/n): ").strip().lower()
    
    if response != 'y':
        print("\n❌ Analysis cancelled\n")
        return
    
    # Get the root directory
    root_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Step 1: Fetch portfolio data
    etrade_script = os.path.join(root_dir, "etrade", "get_all_data.py")
    if not run_script(etrade_script, "STEP 1: Fetching Portfolio Data", root_dir, interactive=True):
        print("⚠️  Failed to fetch portfolio data. Cannot continue.")
        return
    
    # Step 2: Run AI analysis
    ai_script = os.path.join(root_dir, "ai", "portfolio_advisor.py")
    if not run_script(ai_script, "STEP 2: AI Portfolio Analysis", root_dir, interactive=True):
        print("⚠️  AI analysis failed.")
        return
    
    # Complete!
    print_header("ANALYSIS COMPLETE!")
    print("Your portfolio analysis has been saved.")
    print("Check the ai/ folder for the latest analysis file.\n")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n❌ Analysis interrupted by user\n")
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}\n")

"""
Daily Portfolio Analysis Automation
Run this script daily to get fresh AI analysis of your portfolio
"""
import os
import sys
import subprocess
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def main():
    """Run daily portfolio analysis workflow"""
    
    print("\n" + "="*80)
    print(f" DAILY PORTFOLIO ANALYSIS - {datetime.now().strftime('%B %d, %Y')}")
    print("="*80 + "\n")
    
    print("This script will:")
    print("  1. Fetch fresh portfolio data from E*TRADE")
    print("  2. Analyze with AI")
    print("  3. Save analysis to timestamped file")
    print("  4. Display recommendations\n")
    
    response = input("Continue with analysis? (y/n): ").strip().lower()
    
    if response != 'y':
        print("\n[CANCELLED] Analysis cancelled\n")
        return
    
    # Get root directory
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Step 1: Fetch fresh portfolio data
    print("\n" + "="*80)
    print(" STEP 1: Fetching Fresh Portfolio Data")
    print("="*80 + "\n")
    
    etrade_script = os.path.join(root_dir, "etrade", "get_all_data.py")
    try:
        subprocess.run([sys.executable, etrade_script], cwd=root_dir, check=True)
        print("\n[OK] Portfolio data fetched successfully\n")
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] Failed to fetch portfolio data: {e}\n")
        return
    
    # Step 2: Run Multi-Agent AI analysis
    print("="*80)
    print(" STEP 2: Running Multi-Agent AI Analysis")
    print("="*80 + "\n")
    
    ai_script = os.path.join(root_dir, "ai", "run_multi_agent.py")
    try:
        subprocess.run([sys.executable, ai_script], cwd=os.path.join(root_dir, "ai"), check=True)
        print("\n[OK] Multi-agent AI analysis completed\n")
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] Multi-agent AI analysis failed: {e}\n")
        return
    
    print("="*80)
    print(" DAILY ANALYSIS COMPLETE!")
    print("="*80 + "\n")
    print("[SUCCESS] Your daily portfolio analysis is ready!")
    print("[FILES] View reports in ai/analysis_reports/\n")

if __name__ == "__main__":
    main()

"""
Complete Portfolio Analysis Pipeline
Fetches fresh E*TRADE data and runs multi-agent analysis
"""

import subprocess
import sys
import os
from pathlib import Path
import shutil
import glob


def run_command(command, cwd=None):
    """Run a command and return success status"""
    print(f"\n{'='*80}")
    print(f"Running: {' '.join(command)}")
    print(f"{'='*80}\n")
    
    result = subprocess.run(
        command,
        cwd=cwd,
        capture_output=False,
        text=True
    )
    
    return result.returncode == 0

def consolidate_etrade_reports(root_dir: Path):
    """Move legacy root etrade_reports files into etrade/etrade_reports"""
    legacy_dir = root_dir / "etrade_reports"
    canonical_dir = root_dir / "etrade" / "etrade_reports"
    
    canonical_dir.mkdir(parents=True, exist_ok=True)
    
    if legacy_dir.exists():
        moved = 0
        for f in legacy_dir.glob("etrade_*.*"):
            try:
                shutil.move(str(f), str(canonical_dir / f.name))
                moved += 1
            except Exception:
                pass
        if moved:
            print(f"   Moved {moved} legacy report(s) into etrade/etrade_reports")
        try:
            if not any(legacy_dir.iterdir()):
                legacy_dir.rmdir()
                print("   Removed empty legacy etrade_reports folder")
        except Exception:
            pass

def main():
    """Run complete analysis pipeline"""
    
    # Get paths
    root_dir = Path(__file__).parent
    etrade_dir = root_dir / "etrade"
    ai_dir = root_dir / "ai"
    
    print("\n" + "="*80)
    print("COMPLETE PORTFOLIO ANALYSIS PIPELINE")
    print("="*80)
    print("\nThis will:")
    print("  1. Fetch fresh portfolio data from E*TRADE")
    print("  2. Run multi-agent AI analysis")
    print("  3. Generate comprehensive reports")
    print("\n" + "="*80 + "\n")
    
    # Step 1: Fetch E*TRADE data
    print("\nSTEP 1: Fetching E*TRADE Portfolio Data")
    print("-"*80)
    
    if not etrade_dir.exists():
        print(f"Error: etrade directory not found at {etrade_dir}")
        return False
    
    success = run_command(
        [sys.executable, "get_all_data.py"],
        cwd=str(etrade_dir)
    )
    
    if not success:
        print("\nFailed to fetch portfolio data")
        return False
    
    print("\nPortfolio data fetched successfully")
    
    # Consolidate E*TRADE reports into canonical folder
    print("\n🧹 Consolidating E*TRADE reports...")
    consolidate_etrade_reports(root_dir)
    
    # Step 2: Run multi-agent analysis
    print("\nSTEP 2: Running Multi-Agent Analysis")
    print("-"*80)
    
    if not ai_dir.exists():
        print(f"[ERROR] ai directory not found at {ai_dir}")
        return False
    
    success = run_command(
        [sys.executable, "run_multi_agent.py"],
        cwd=str(ai_dir)
    )
    
    if not success:
        print("\n[ERROR] Failed to run multi-agent analysis")
        return False
    
    print("\n[SUCCESS] Multi-agent analysis completed successfully")
    
    # Summary
    print("\n" + "="*80)
    print("[SUCCESS] PIPELINE COMPLETE!")
    print("="*80)
    print("\nGenerated files:")
    print("  etrade/etrade_reports/etrade_data_*.json (latest portfolio data)")
    print("  📂 ai/analysis_reports/ (multi-agent analysis reports)")
    print("\nView reports in the ai/analysis_reports/ folder")
    print("="*80 + "\n")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

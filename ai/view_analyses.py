"""
View all saved AI portfolio analyses
Lists all analysis files with portfolio values and dates
"""
import os
import glob
from datetime import datetime

def list_analyses():
    """List all saved multi-agent analysis reports"""
    reports_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'analysis_reports')
    text_reports = glob.glob(os.path.join(reports_dir, 'multi_agent_report_*.txt'))
    
    if not text_reports:
        print("\n📭 No multi-agent reports found yet.")
        print("Run run_multi_agent.py or the pipeline to generate your first report!\n")
        return
    
    # Sort newest first
    text_reports.sort(reverse=True)
    
    print("\n" + "="*80)
    print(" SAVED MULTI-AGENT ANALYSIS REPORTS")
    print("="*80 + "\n")
    
    for i, filepath in enumerate(text_reports, 1):
        filename = os.path.basename(filepath)
        timestamp_str = filename.replace('multi_agent_report_', '').replace('.txt', '')
        try:
            dt = datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S')
            date_formatted = dt.strftime('%B %d, %Y at %I:%M %p')
        except:
            date_formatted = timestamp_str
        
        # Extract portfolio value line from report
        portfolio_line = "Portfolio value: Unknown"
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip().startswith('Portfolio value:'):
                        portfolio_line = line.strip()
                        break
        except:
            pass
        
        size_kb = os.path.getsize(filepath) / 1024
        print(f"{i}. {filename}")
        print(f"   Generated: {date_formatted}")
        print(f"   {portfolio_line}")
        print(f"   Size: {size_kb:.1f} KB\n")
    
    print("="*80)
    print(f"Total reports saved: {len(text_reports)}\n")
    
    if text_reports:
        print("To view the latest report:")
        print(f"  notepad {text_reports[0]}")
        print("\nOr open it from the file explorer in ai/analysis_reports\n")

if __name__ == "__main__":
    list_analyses()

"""
View all saved AI portfolio analyses
Lists all analysis files with portfolio values and dates
"""
import os
import glob
from datetime import datetime

def list_analyses():
    """List all saved AI analysis files"""
    
    # Get all analysis files
    analysis_files = glob.glob('ai_analysis_*.txt')
    
    if not analysis_files:
        print("\n📭 No analysis files found yet.")
        print("Run quick_start_openai.py to generate your first analysis!\n")
        return
    
    # Sort by date (newest first)
    analysis_files.sort(reverse=True)
    
    print("\n" + "="*80)
    print(" 📊 SAVED AI PORTFOLIO ANALYSES")
    print("="*80 + "\n")
    
    for i, filepath in enumerate(analysis_files, 1):
        # Extract timestamp from filename
        timestamp_str = filepath.replace('ai_analysis_', '').replace('.txt', '')
        try:
            dt = datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S')
            date_formatted = dt.strftime('%B %d, %Y at %I:%M %p')
        except:
            date_formatted = timestamp_str
        
        # Try to extract portfolio value from file
        portfolio_value = "Unknown"
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    if 'Portfolio Value:' in line:
                        portfolio_value = line.split('Portfolio Value:')[1].strip()
                        break
        except:
            pass
        
        print(f"{i}. {filepath}")
        print(f"   Generated: {date_formatted}")
        print(f"   Portfolio: {portfolio_value}")
        
        # Get file size
        size_kb = os.path.getsize(filepath) / 1024
        print(f"   Size: {size_kb:.1f} KB\n")
    
    print("="*80)
    print(f"Total analyses saved: {len(analysis_files)}\n")
    
    # Show latest analysis preview
    if analysis_files:
        print("To view the latest analysis:")
        print(f"  notepad {analysis_files[0]}")
        print("\nOr open it from the file explorer in the ai/ folder\n")

if __name__ == "__main__":
    list_analyses()

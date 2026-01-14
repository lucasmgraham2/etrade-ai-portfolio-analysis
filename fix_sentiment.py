#!/usr/bin/env python3
"""Fix script to replace all neutral defaults with INSUFFICIENT_DATA"""

import os

file_path = os.path.join(os.path.dirname(__file__), 'ai', 'multi_agent', 'sentiment_agent.py')

with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# Replace patterns from most specific to least specific
patterns = [
    ('"score": 0, "label": "neutral", "count": 0, "method": "none", "articles_analyzed": 0', 
     '"score": None, "label": "INSUFFICIENT_DATA", "count": 0, "method": "none", "articles_analyzed": 0'),
    ('"score": 0, "label": "neutral", "count": 0, "method": "none"',
     '"score": None, "label": "INSUFFICIENT_DATA", "count": 0, "method": "none"'),
    ('"score": 0, "label": "neutral", "count": 0',
     '"score": None, "label": "INSUFFICIENT_DATA", "count": 0'),
    ('"score": 0, "label": "neutral"',
     '"score": None, "label": "INSUFFICIENT_DATA"'),
]

for old, new in patterns:
    count = content.count(old)
    if count > 0:
        content = content.replace(old, new)
        print(f"Replaced {count} occurrences of: {old[:50]}...")

with open(file_path, 'w', encoding='utf-8') as f:
    f.write(content)

print('\nAll neutral defaults successfully replaced with INSUFFICIENT_DATA')

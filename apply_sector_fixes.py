#!/usr/bin/env python3
"""Update screener.py with correct JSE_TOP_40 and JSE_SECTORS from Excel"""

import pandas as pd
import re

# Read Excel
df = pd.read_excel('/home/red/Downloads/constituent_details.xlsx', 
                   skiprows=2, 
                   names=['Code', 'Company', 'Industry', 'Shares', 'DivYield', 'MarketCap', 'Price'])

# Sector mapping
SECTOR_MAP = {
    'Banks': 'Financials',
    'Insurance': 'Financials',
    'Real Estate': 'Real Estate',
    'Basic Resources': 'Materials',
    'Food Beverage and Tobacco': 'Consumer',
    'Personal Care Drug and Grocery Stores': 'Consumer',
    'Industrial Goods & Sevices': 'Consumer',
    'Consumer Products and Services': 'Consumer',
    'Retail': 'Consumer',
    'Telecommunications': 'Telecom',
    'Technology': 'Technology',
    'Financial Services': 'Financials',
    'Chemicals': 'Materials',
}

# Generate JSE_TOP_40
tickers = sorted([f"{row['Code']}.JO" for _, row in df.iterrows()])
top40_str = "JSE_TOP_40 = [\n" + "\n".join([f'    "{t}",' for t in tickers]) + "\n]"

# Generate JSE_SECTORS
sectors_str = "JSE_SECTORS = {\n"
for _, row in df.iterrows():
    ticker = f"{row['Code']}.JO"
    sector = SECTOR_MAP.get(row['Industry'], 'Other')
    sectors_str += f'    "{ticker}": "{sector}",  # {row["Company"]}\n'
sectors_str += "}"

# Read current screener.py
with open('/home/red/projects/active/fin_fuz_sol/backend/screener.py', 'r') as f:
    content = f.read()

# Replace JSE_TOP_40
content = re.sub(
    r'JSE_TOP_40 = \[.*?\]',
    top40_str,
    content,
    flags=re.DOTALL
)

# Replace JSE_SECTORS
content = re.sub(
    r'JSE_SECTORS = \{.*?\}',
    sectors_str,
    content,
    flags=re.DOTALL
)

# Write back
with open('/home/red/projects/active/fin_fuz_sol/backend/screener.py', 'w') as f:
    f.write(content)

print("✅ Updated screener.py with:")
print(f"  - {len(tickers)} tickers in JSE_TOP_40")
print(f"  - {len(set(sectors_str.split(': ')[1::2]))} unique sectors in JSE_SECTORS")

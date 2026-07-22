#!/usr/bin/env python3
"""Update screener.py with exact 12 industries from Satrix spreadsheet"""

import pandas as pd
import re

# Read Excel
df = pd.read_excel('/home/red/Downloads/constituent_details.xlsx', 
                   skiprows=2, 
                   names=['Code', 'Company', 'Industry', 'Shares', 'DivYield', 'MarketCap', 'Price'])

# Drop header row that got included
df = df[df['Industry'] != 'INDUSTRY']

# Map to 12 main industries (grouping similar ones)
INDUSTRY_MAP = {
    'Banks': 'Banks',
    'Basic Resources': 'Basic Resources',
    'Chemicals': 'Chemicals',
    'Consumer Products and Services': 'Consumer Products & Services',
    'Financial Services': 'Financial Services',
    'Food Beverage and Tobacco': 'Food, Beverage & Tobacco',
    'Industrial Goods & Sevices': 'Industrial Goods & Services',
    'Insurance': 'Insurance',
    'Personal Care Drug and Grocery Stores': 'Personal Care, Drug & Grocery',
    'Real Estate': 'Real Estate',
    'Retail': 'Retail',
    'Technology': 'Technology',
    'Telecommunications': 'Telecommunications',
}

# Generate JSE_TOP_40
tickers = sorted([f"{row['Code']}.JO" for _, row in df.iterrows()])
top40_str = "JSE_TOP_40 = [\n" + "\n".join([f'    "{t}",' for t in tickers]) + "\n]"

# Generate JSE_SECTORS with exact industry names
sectors_str = "# Sector mappings from Satrix constituent_details.xlsx (12 industries)\nJSE_SECTORS = {\n"
for _, row in df.iterrows():
    ticker = f"{row['Code']}.JO"
    industry = row['Industry']
    sector = INDUSTRY_MAP.get(industry, 'Other')
    sectors_str += f'    "{ticker}": "{sector}",  # {row["Company"]} - {industry}\n'
sectors_str += "}"

# Read current screener.py
with open('/home/red/projects/active/fin_fuz_sol/backend/screener.py', 'r') as f:
    content = f.read()

# Replace JSE_TOP_40
content = re.sub(
    r'# === JSE Top 40 Tickers.*?\]',
    top40_str,
    content,
    flags=re.DOTALL
)

# Replace JSE_SECTORS
content = re.sub(
    r'# Sector mappings from constituent_details.xlsx.*?}',
    sectors_str,
    content,
    flags=re.DOTALL
)

# Write back
with open('/home/red/projects/active/fin_fuz_sol/backend/screener.py', 'w') as f:
    f.write(content)

print("✅ Updated screener.py with 12 industries from Satrix spreadsheet:")
print(f"  - {len(tickers)} tickers in JSE_TOP_40")
print(f"  - {len(set(sectors_str.split(': ')[1::2]))} unique industries")
print("\nIndustries:")
for i, ind in enumerate(sorted(set(INDUSTRY_MAP.values())), 1):
    print(f"  {i}. {ind}")

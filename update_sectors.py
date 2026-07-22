#!/usr/bin/env python3
"""Update JSE_TOP_40 and JSE_SECTORS in screener.py based on constituent_details.xlsx"""

import pandas as pd

# Read the Excel file
df = pd.read_excel('/home/red/Downloads/constituent_details.xlsx', 
                   skiprows=2, 
                   names=['Code', 'Company', 'Industry', 'Shares', 'DivYield', 'MarketCap', 'Price'])

# Create sector mapping
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

# Generate JSE_TOP_40 list
tickers = [f"{row['Code']}.JO" for _, row in df.iterrows()]

# Generate sector mapping
sectors = {}
for _, row in df.iterrows():
    ticker = f"{row['Code']}.JO"
    industry = row['Industry']
    sector = SECTOR_MAP.get(industry, 'Other')
    sectors[ticker] = sector

# Print results
print("JSE_TOP_40 = [")
for ticker in tickers:
    print(f'    "{ticker}",')
print("]")
print("\nJSE_SECTORS = {")
for ticker, sector in sectors.items():
    print(f'    "{ticker}": "{sector}",')
print("}")
print(f"\nTotal tickers: {len(tickers)}")
print(f"Total sectors: {len(set(sectors.values()))}")

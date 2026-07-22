#!/usr/bin/env python3
"""Comprehensive backend testing script."""

import sys
sys.path.append('/home/red/projects/active/fin_fuz_sol/backend')

from fundamentals import (
    get_financial_statements, 
    get_ratio_trends, 
    get_analyst_estimates,
    get_segment_data,
    get_fair_value_comparison
)
from screener import get_sector_performance, screen_stocks

print("=" * 70)
print("🧪 COMPREHENSIVE BACKEND TEST SUITE")
print("=" * 70)

# Test 1: Sector Performance
print("\n1. SECTOR PERFORMANCE TEST")
print("-" * 70)
sectors = get_sector_performance()
if sectors:
    print(f"✅ Total sectors: {len(sectors)}")
    for s in sorted(sectors, key=lambda x: x['stock_count'], reverse=True):
        print(f"   - {s['name']}: {s['stock_count']} stocks")
    
    # Verify Real Estate
    real_estate = next((s for s in sectors if s['name'] == 'Real Estate'), None)
    if real_estate:
        print(f"✅ Real Estate sector exists with {len(real_estate['stocks'])} stocks:")
        for stock in real_estate['stocks']:
            print(f"     • {stock['ticker']} - {stock['name']}")
else:
    print("❌ Sector performance failed")

# Test 2: Financial Statements
print("\n2. FINANCIAL STATEMENTS TEST (NPN.JO)")
print("-" * 70)
stmts = get_financial_statements('NPN.JO')
if stmts:
    print(f"✅ Years available: {len(stmts.get('years', []))}")
    print(f"✅ Income statement items: {len(stmts.get('income_statement', {}))}")
    print(f"✅ Balance sheet items: {len(stmts.get('balance_sheet', {}))}")
    print(f"✅ Cash flow items: {len(stmts.get('cash_flow', {}))}")
else:
    print("❌ Financial statements failed")

# Test 3: Ratio Trends
print("\n3. RATIO TRENDS TEST (NPN.JO)")
print("-" * 70)
ratios = get_ratio_trends('NPN.JO')
if ratios:
    current = ratios.get('current', {})
    print(f"✅ P/E Ratio: {current.get('pe_ratio', 'N/A')}")
    print(f"✅ ROE: {current.get('return_on_equity', 'N/A')}")
    print(f"✅ Profit Margin: {current.get('profit_margin', 'N/A')}")
else:
    print("❌ Ratio trends failed")

# Test 4: Analyst Estimates
print("\n4. ANALYST ESTIMATES TEST (NPN.JO)")
print("-" * 70)
analyst = get_analyst_estimates('NPN.JO')
if analyst:
    pt = analyst.get('price_targets', {})
    print(f"✅ Mean Target: R{pt.get('mean', 'N/A')}")
    print(f"✅ High Target: R{pt.get('high', 'N/A')}")
    print(f"✅ Low Target: R{pt.get('low', 'N/A')}")
    rec = analyst.get('recommendations', {})
    if rec:
        print(f"✅ Latest Recommendations: {rec}")
else:
    print("❌ Analyst estimates failed")

# Test 5: Segment Data
print("\n5. SEGMENT DATA TEST (NPN.JO)")
print("-" * 70)
segments = get_segment_data('NPN.JO')
if segments:
    print(f"✅ Segment data: {segments.get('message', 'No message')}")
    print(f"   Note: {segments.get('note', '')}")
else:
    print("❌ Segment data failed")

# Test 6: Fair Value Comparison
print("\n6. FAIR VALUE COMPARISON TEST (NPN.JO)")
print("-" * 70)
fair_value = get_fair_value_comparison('NPN.JO', 500.00)
if fair_value:
    print(f"✅ Current Price: R{fair_value.get('current_price', 'N/A')}")
    print(f"✅ Your DCF: R{fair_value.get('your_dcf', 'N/A')}")
    print(f"✅ DCF Upside: {fair_value.get('dcf_upside', 'N/A')}%")
    print(f"✅ Analyst Target: R{fair_value.get('analyst_target', 'N/A')}")
    print(f"✅ Verdict: {fair_value.get('verdict', 'N/A')}")
else:
    print("❌ Fair value comparison failed")

# Test 7: Stock Screener
print("\n7. STOCK SCREENER TEST")
print("-" * 70)
results = screen_stocks(max_pe=15, min_dividend_yield=0.03)
if results:
    print(f"✅ Found {len(results)} stocks with P/E < 15 and Div Yield > 3%")
    print("   Top 5 results:")
    for r in results[:5]:
        print(f"     • {r['ticker']}: P/E={r['pe_ratio']:.1f}, Div={r['dividend_yield']*100:.1f}%")
else:
    print("❌ Stock screener failed")

print("\n" + "=" * 70)
print("✅ ALL TESTS COMPLETED")
print("=" * 70)

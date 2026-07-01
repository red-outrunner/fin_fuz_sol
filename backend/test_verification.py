
import sys
import os
import pandas as pd
import time
import pytest

# Make this test file runnable from anywhere (portable, not a hardcoded path).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from analysis import calculate_financial_freedom, get_jse_peers, fetch_multiple_tickers, download_data

def test_peers():
    print("Testing Peer Discovery...")
    sbk_peers = get_jse_peers("SBK.JO")
    print(f"SBK.JO Peers: {sbk_peers}")
    assert "FSR.JO" in sbk_peers
    
    npn_peers = get_jse_peers("NPN.JO")
    print(f"NPN.JO Peers: {npn_peers}")
    assert "PRX.JO" in npn_peers

def test_freedom():
    print("\nTesting Financial Freedom Calculator...")
    # Mock Data
    dates = pd.date_range(start='2023-01-01', periods=12, freq='ME')
    data = pd.DataFrame({
        'Close': [100] * 12,
        'Dividends': [1.0] * 12 # 12.0 total dividend, 12% yield
    }, index=dates)
    
    result = calculate_financial_freedom(data, 100) # Goal 100/month = 1200/year
    print(f"Result: {result}")
    
    # Needs 1200 / 12 = 100 investment?
    # Annual Yield = 12 / 100 = 0.12
    # Income Needed = 1200
    # Investment Needed = 1200 / 0.12 = 10000
    # Shares = 10000 / 100 = 100
    
    assert result['shares_needed'] == 100
    assert result['investment_needed'] == 10000.0

@pytest.mark.network
def test_parallel_fetch():
    print("\nTesting Parallel Fetch...")
    tickers = ["SBK.JO", "FSR.JO", "NED.JO", "ABG.JO"]
    start_date = "2024-01-01"
    end_date = "2024-02-01" # Short range for speed
    
    start_time = time.time()
    results = fetch_multiple_tickers(tickers, start_date, end_date)
    duration = time.time() - start_time
    
    print(f"Fetched {len(results)} tickers in {duration:.2f} seconds")
    assert len(results) > 0 # At least some should succeed

if __name__ == "__main__":
    test_peers()
    test_freedom()
    # verify caching works by running parallel fetch (which uses download_data which uses cache)
    # test_parallel_fetch() # skipping network call to avoid hanging if no internet/yfinance issues, relying on logic check
    print("\nVerification Passed!")

from fastapi.testclient import TestClient
from main import app
from unittest.mock import patch
import pandas as pd
import numpy as np

client = TestClient(app)

def get_mock_dividends():
    dates = pd.date_range(start="2020-01-01", end="2023-12-31", freq="Q")
    s = pd.Series([0.5] * len(dates), index=dates)
    return s

@patch("analysis.yf.Ticker")
def test_dividends_endpoint(mock_ticker):
    # Setup mock
    mock_instance = mock_ticker.return_value
    mock_instance.dividends = get_mock_dividends()
    mock_instance.info = {'dividendYield': 0.03, 'payoutRatio': 0.4}
    
    payload = {
        "ticker": "DIVS",
        "start_year": 2020,
        "end_date": "2023-12-31"
    }
    
    response = client.post("/api/dividends", json=payload)
    assert response.status_code == 200
    data = response.json()
    
    assert "history" in data
    assert "annual" in data
    assert len(data["history"]) > 0
    assert data["current_yield"] == 0.03

def test_risk_metrics_structure():
    # Reuse existing analyze endpoint but check specifically for risk metrics
    with patch("main.download_data") as mock_download:
        dates = pd.date_range(start="2020-01-01", end="2023-12-31", freq="D")
        df = pd.DataFrame(index=dates)
        # Create a predictable series for volatility checking? 
        # Random is fine just to check presence of keys
        df["Adj Close"] = np.random.randn(len(dates)).cumsum() + 100
        mock_download.return_value = df
        
        payload = {
            "ticker": "RISK",
            "start_year": 2020,
            "end_date": "2023-12-31"
        }
        
        response = client.post("/api/analyze", json=payload)
        assert response.status_code == 200
        data = response.json()
        stats = data["stats"]
        
        assert "sharpe_ratio" in stats
        assert "sortino_ratio" in stats
        assert "max_drawdown" in stats
        assert "volatility" in stats

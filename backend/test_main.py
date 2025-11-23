from fastapi.testclient import TestClient
from main import app
from unittest.mock import patch
import pandas as pd
import numpy as np

client = TestClient(app)

# Mock data
def get_mock_data():
    dates = pd.date_range(start="2020-01-01", end="2023-12-31", freq="D")
    df = pd.DataFrame(index=dates)
    df["Adj Close"] = np.random.randn(len(dates)).cumsum() + 100
    return df

@patch("main.download_data")
def test_analyze_ticker(mock_download):
    mock_download.return_value = get_mock_data()
    
    payload = {
        "ticker": "TEST",
        "start_year": 2020,
        "end_date": "2023-12-31"
    }
    response = client.post("/api/analyze", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "stats" in data
    assert "pivot_data" in data
    assert "monthly_returns" in data
    assert data["ticker"] == "TEST"

@patch("main.download_data")
def test_ml_analysis(mock_download):
    mock_download.return_value = get_mock_data()
    
    payload = {
        "ticker": "TEST",
        "start_year": 2010,
        "end_date": "2023-12-31"
    }
    response = client.post("/api/ml", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "pca_components" in data
    assert "clusters" in data
    assert "anomalies" in data

@patch("main.download_data")
def test_stats_analysis(mock_download):
    mock_download.return_value = get_mock_data()
    
    payload = {
        "ticker": "TEST",
        "start_year": 2010,
        "end_date": "2023-12-31"
    }
    response = client.post("/api/stats", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "f_stat" in data
    assert "p_value" in data

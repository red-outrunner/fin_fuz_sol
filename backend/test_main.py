from fastapi.testclient import TestClient
from main import app
import pytest

client = TestClient(app)

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Global Index Analyzer API is running"}

def test_analyze_ticker():
    # Use a ticker that is likely to have data
    payload = {
        "ticker": "^GSPC", # S&P 500
        "start_year": 2020,
        "end_date": "2023-12-31"
    }
    response = client.post("/api/analyze", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "stats" in data
    assert "pivot_data" in data
    assert "monthly_returns" in data
    assert data["ticker"] == "^GSPC"

def test_ml_analysis():
    payload = {
        "ticker": "^GSPC",
        "start_year": 2010, # Need enough data for ML
        "end_date": "2023-12-31"
    }
    response = client.post("/api/ml", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "pca_components" in data
    assert "clusters" in data
    assert "anomalies" in data

def test_stats_analysis():
    payload = {
        "ticker": "^GSPC",
        "start_year": 2010,
        "end_date": "2023-12-31"
    }
    response = client.post("/api/stats", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "f_stat" in data
    assert "p_value" in data

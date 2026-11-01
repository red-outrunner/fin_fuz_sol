"""Regression coverage for projections from dividend-adjusted total returns."""
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from analytics import core_math
from main import app


@pytest.mark.parametrize("method", ["gbm", "bootstrap"])
@pytest.mark.parametrize("monthly_return", [0.0, 0.01])
def test_total_returns_compound_once(method, monthly_return):
    # Constant total returns make all percentile paths analytically predictable.
    returns = pd.Series([monthly_return] * 24)
    result = core_math.run_monte_carlo(returns, years=1, n_sims=10, method=method)
    assert len(result) == 13
    for month, point in enumerate(result):
        expected = 10000 * (1 + monthly_return) ** month
        for percentile in ("p10", "p50", "p90"):
            assert point[percentile] == pytest.approx(expected)


@pytest.mark.parametrize("endpoint", ["/api/projection", "/api/export/pdf"])
def test_adjusted_history_never_adds_current_dividend_yield(endpoint):
    # Daily adjusted Close implies exactly 1% total return each calendar month.
    dates = pd.date_range("2020-01-01", "2022-12-31", freq="D")
    month_index = (dates.year - 2020) * 12 + dates.month - 1
    history = pd.DataFrame({"Close": 100 * np.power(1.01, month_index)}, index=dates)
    from analytics import run_monte_carlo
    with patch("main.download_data", return_value=history), \
         patch("market_data.yf.Ticker") as ticker, \
         patch("main.run_monte_carlo", wraps=run_monte_carlo) as simulate, \
         patch("main.get_jse_peers", return_value=[]), \
         patch("main.calculate_summary_stats", return_value={}), \
         patch("main.PDFReportGenerator"):
        ticker.return_value.info = {"dividendYield": 0.12}
        response = TestClient(app).post(endpoint, json={
            "ticker": "DIVS", "start_year": 2020, "end_date": "2022-12-31",
        })
        assert response.status_code == 200
        ticker.assert_not_called()
        assert simulate.call_count == 1
        if endpoint == "/api/projection":
            final = response.json()[-1]
            for percentile in ("p10", "p50", "p90"):
                assert final[percentile] == pytest.approx(10000 * 1.01 ** 120)

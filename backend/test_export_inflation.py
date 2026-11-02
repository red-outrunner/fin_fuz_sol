"""Export summaries must receive the same inflation setting as the dashboard."""
import io
from unittest.mock import patch

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from main import app


@pytest.mark.parametrize("endpoint", ["excel", "sheets", "pdf"])
@pytest.mark.parametrize("rate", [0.0, .05])
def test_export_summary_uses_requested_inflation(endpoint, rate):
    dates = pd.date_range("2025-01-31", periods=12, freq="ME")
    returns = pd.Series([.01] * 12, index=dates)
    processed = {
        "monthly_ret": returns,
        "df": pd.DataFrame({"Close": range(100, 112)}, index=dates),
        "pivot": pd.DataFrame({1: [.01]}, index=[2025]),
    }
    summary = {"cagr": .12 if rate == 0 else .07}
    with patch("main.download_data", return_value=processed["df"]), \
            patch("main.process_data", return_value=processed), \
            patch("main.calculate_summary_stats", return_value=summary) as stats, \
            patch("main.get_jse_peers", return_value=[]), \
            patch("main.run_monte_carlo", return_value=None), \
            patch("main.calculate_dca", return_value=None), \
            patch("main.PDFReportGenerator") as report:
        response = TestClient(app).post(f"/api/export/{endpoint}", json={
            "ticker": "TEST.JO", "start_year": 2025, "end_date": "2026-01-01",
            "inflation_rate": rate,
        })
    assert response.status_code == 200
    assert stats.call_args.kwargs["inflation_rate"] == rate
    if endpoint == "excel":
        frame = pd.read_excel(io.BytesIO(response.content), sheet_name="Summary Stats")
        assert frame["cagr"].iloc[0] == pytest.approx(summary["cagr"])
    elif endpoint == "sheets":
        assert str(summary["cagr"]) in response.json()["tsv"]
    else:
        report.return_value.add_executive_summary.assert_called_once_with(summary)

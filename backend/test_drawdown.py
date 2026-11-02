"""Drawdown must include losses from the original investment, before any gains."""
import io
from unittest.mock import patch

import pandas as pd
import pytest

from analytics import core_math, calculate_summary_stats
from reports import PDFReportGenerator


@pytest.mark.parametrize("returns, expected", [
    ([-.5, 0], [-.5, -.5]),
    ([-.5], [-.5]),
    ([-.2, -.25], [-.2, -.4]),
    ([-.5, 1, -.2], [-.5, 0, -.2]),
    ([.5, -.5], [0, -.5]),
    ([.1, .1], [0, 0]),
    ([0, 0], [0, 0]),
    ([-1, 0], [-1, -1]),
    ([.5, -1], [0, -1]),
])
def test_drawdown_includes_starting_wealth(returns, expected):
    series = pd.Series(returns, index=pd.date_range("2025-01-31", periods=len(returns), freq="ME"))
    drawdown = core_math.calculate_drawdown(series)
    assert drawdown.index.equals(series.index)
    assert drawdown.tolist() == pytest.approx(expected)
    assert core_math.calculate_max_drawdown(series) == pytest.approx(min(expected))


def test_empty_drawdown():
    series = pd.Series(dtype=float)
    assert core_math.calculate_drawdown(series).empty
    assert core_math.calculate_max_drawdown(series) == 0


@pytest.mark.parametrize("inflation", [0, 1.01 ** 12 - 1])
def test_summary_metric_and_chart_agree_on_initial_loss(inflation):
    returns = pd.Series([-.5] + [0.] * 11, index=pd.date_range("2025-01-31", periods=12, freq="ME"))
    pivot = pd.DataFrame([returns.values], index=[2025], columns=range(1, 13))
    stats = calculate_summary_stats(returns, pivot, inflation_rate=inflation)
    expected_min = .5 / (1 + inflation) - 1
    assert stats["max_drawdown"] == pytest.approx(expected_min)
    chart = stats["drawdown_series"]
    assert chart[0]["value"] == pytest.approx(.5 / (1 + inflation) ** (1 / 12) - 1)
    assert min(point["value"] for point in chart) == pytest.approx(expected_min)


def test_pdf_chart_shows_loss_from_starting_wealth():
    returns = pd.Series([-.5, 0.], index=pd.date_range("2025-01-31", periods=2, freq="ME"))
    report = PDFReportGenerator(io.BytesIO(), "TEST", 2025, "2025-12-31")
    # Inspect actual plotted values; normal image rendering closes the figure.
    with patch.object(report, "_fig_to_image", wraps=report._fig_to_image) as render:
        report.add_drawdown_chart({"monthly_ret": returns})
    figure = render.call_args.args[0]
    assert figure.axes[0].lines[0].get_ydata().tolist() == pytest.approx([-.5, -.5])

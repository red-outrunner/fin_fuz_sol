"""Unit tests for the sector_manager module."""
import os
import sys
import tempfile
import pytest

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from sector_manager import (
    load_sector_data,
    get_jse_top_40,
    get_jse_sectors,
    get_sector_for_ticker,
    import_from_excel,
    sync_sector_data
)
import screener


def test_load_sector_data():
    data = load_sector_data()
    assert isinstance(data, dict)
    assert "top_40" in data
    assert "sectors" in data
    assert len(data["top_40"]) > 0
    assert len(data["sectors"]) > 0


def test_get_jse_top_40():
    top40 = get_jse_top_40()
    assert isinstance(top40, list)
    assert "NPN.JO" in top40
    assert "SBK.JO" in top40
    assert "ABG.JO" in top40


def test_get_jse_sectors():
    sectors = get_jse_sectors()
    assert isinstance(sectors, dict)
    assert sectors.get("ABG.JO") == "Banks"
    assert sectors.get("NPN.JO") == "Technology"
    assert sectors.get("SOL.JO") == "Chemicals"


def test_get_sector_for_ticker():
    assert get_sector_for_ticker("ABG.JO") == "Banks"
    assert get_sector_for_ticker("NPN.JO") == "Technology"
    assert get_sector_for_ticker("GRT.JO") == "Real Estate"
    # Unknown ticker fallback
    assert get_sector_for_ticker("UNKNOWN_XYZ.JO", fallback_yfinance=False) == "Other"


def test_screener_backward_compatibility():
    assert screener.JSE_TOP_40 == get_jse_top_40()
    assert screener.JSE_SECTORS == get_jse_sectors()
    assert "NPN.JO" in screener.JSE_TOP_40
    assert screener.JSE_SECTORS["ABG.JO"] == "Banks"


def test_import_from_excel_invalid_path():
    assert import_from_excel("/non/existent/path/constituent_details.xlsx") is False


def test_sync_sector_data():
    res = sync_sector_data(auto_search_excel=False)
    assert res["status"] == "success"
    assert res["total_tickers"] > 0
    assert res["total_sectors"] > 0

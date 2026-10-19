"""Unit tests for JSE ZAc/ZAR normalization — Capitec/Naspers must not be /100'd when currency is ZAR."""
from currency_utils import normalize_price, display_currency, BRAND_ATTRIBUTION
from currency_utils import history_quote_mode, normalize_history_value


def test_normalize_zac_cents():
    assert normalize_price(250000, "ZAc") == 2500.0


def test_normalize_zar_capitec_like():
    # Capitec-scale price already in rands — must NOT divide
    assert normalize_price(2500, "ZAR") == 2500.0


def test_normalize_zar_naspers_like():
    assert normalize_price(3500, "ZAR") == 3500.0


def test_normalize_gbp_pence():
    assert normalize_price(1234, "GBp") == 12.34


def test_display_currency_maps_zac():
    assert display_currency("ZAc", "CPI.JO") == "ZAR"
    assert display_currency("ZAR", "CPI.JO") == "ZAR"
    assert display_currency("", "CPI.JO") == "ZAR"
    assert display_currency("", "AAPL") == "USD"


def test_brand_attribution():
    assert "ubomvu.com" in BRAND_ATTRIBUTION
    assert "Analyzed with Ubomvu" in BRAND_ATTRIBUTION


def test_history_quote_mode_suffixes():
    assert history_quote_mode("CPI.JO") == "ZAc"
    assert history_quote_mode("bp.L") == "GBp"
    assert history_quote_mode("AZN.IL") == "GBp"
    assert history_quote_mode("AAPL") is None
    assert history_quote_mode("SAP.DE") is None  # Euros already major units


def test_normalize_history_value_zac():
    # JSE history Close/Adj Close arrives in ZAc — 448,802 ZAc → R4,488.02
    assert normalize_history_value(448802, "CPI.JO") == 4488.02


def test_normalize_history_value_gbp():
    assert normalize_history_value(12345, "XYZ.L") == 123.45


def test_normalize_history_value_pass_through():
    assert normalize_history_value(150.5, "AAPL") == 150.5
    assert normalize_history_value(None, "CPI.JO") is None

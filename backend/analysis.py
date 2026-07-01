"""Backward-compatible facade.

The former monolith has been split into focused modules:
  - serialization.py : clean_data (JSON-safe serialization)
  - market_data.py   : yfinance fetching, parquet caching, news/SSRF, fundamentals
  - analytics.py     : return processing, stats, ML, Monte Carlo, DCA, peers

This module re-exports the public surface so existing imports
(`from analysis import download_data, process_data, ...`) keep working unchanged.
"""
from serialization import clean_data

from market_data import (
    download_data,
    fetch_multiple_tickers,
    search_tickers,
    get_company_profile,
    generate_fun_stats,
    get_key_stats,
    get_news,
    is_safe_public_url,
    get_article_content,
    get_calendar,
    get_dividend_history,
    get_financials,
    normalize_dividend_yield,
    get_dividend_yield,
)

from analytics import (
    process_data,
    calculate_summary_stats,
    run_ml_analysis,
    run_anova_test,
    calculate_dca,
    calculate_correlation,
    run_monte_carlo,
    calculate_financial_freedom,
    get_jse_peers,
)

__all__ = [
    "clean_data",
    "download_data",
    "fetch_multiple_tickers",
    "search_tickers",
    "get_company_profile",
    "generate_fun_stats",
    "get_key_stats",
    "get_news",
    "is_safe_public_url",
    "get_article_content",
    "get_calendar",
    "get_dividend_history",
    "get_financials",
    "normalize_dividend_yield",
    "get_dividend_yield",
    "process_data",
    "calculate_summary_stats",
    "run_ml_analysis",
    "run_anova_test",
    "calculate_dca",
    "calculate_correlation",
    "run_monte_carlo",
    "calculate_financial_freedom",
    "get_jse_peers",
]

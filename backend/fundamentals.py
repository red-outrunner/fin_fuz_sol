"""Financial statements and fundamental analysis data."""
import os
import logging
import yfinance as yf
import pandas as pd
from typing import Optional, Dict, Any, List

from serialization import clean_data

logger = logging.getLogger(__name__)


def get_financial_statements(ticker: str) -> Optional[Dict[str, Any]]:
    """
    Get 5-year financial statements for a ticker.
    Returns income statement, balance sheet, and cash flow.
    """
    try:
        t = yf.Ticker(ticker)
        
        # Get financial statements
        income_stmt = t.income_stmt
        balance_sheet = t.balance_sheet
        cash_flow = t.cashflow
        
        if income_stmt.empty and balance_sheet.empty and cash_flow.empty:
            return None
        
        # Convert to dict with years as keys
        def stmt_to_dict(stmt, name):
            if stmt.empty:
                return {}
            # Convert to millions for readability
            result = {}
            for idx in stmt.index:
                result[idx] = {}
                for col in stmt.columns:
                    year = col.strftime('%Y') if hasattr(col, 'strftime') else str(col)
                    value = stmt.loc[idx, col]
                    # Convert to millions if value is large
                    if abs(value) > 1e9:
                        value = value / 1e6  # Convert to millions
                    result[idx][year] = round(value, 2) if pd.notna(value) else None
            return result
        
        return {
            "income_statement": stmt_to_dict(income_stmt, "income"),
            "balance_sheet": stmt_to_dict(balance_sheet, "balance"),
            "cash_flow": stmt_to_dict(cash_flow, "cash"),
            "years": list(set([col.strftime('%Y') if hasattr(col, 'strftime') else str(col) 
                              for col in income_stmt.columns] + 
                             [col.strftime('%Y') if hasattr(col, 'strftime') else str(col) 
                              for col in balance_sheet.columns] +
                             [col.strftime('%Y') if hasattr(col, 'strftime') else str(col) 
                              for col in cash_flow.columns]))
        }
    except Exception as e:
        logger.error(f"Error fetching financial statements for {ticker}: {e}")
        return None


def get_ratio_trends(ticker: str) -> Optional[Dict[str, Any]]:
    """
    Get 5-year ratio trends for fundamental analysis.
    """
    try:
        t = yf.Ticker(ticker)
        info = t.info
        
        # Get historical data for trend analysis
        hist = t.history(period="5y")
        
        if hist.empty:
            return None
        
        # Current ratios from info
        current_ratios = {
            "pe_ratio": info.get('trailingPE'),
            "forward_pe": info.get('forwardPE'),
            "price_to_book": info.get('priceToBook'),
            "debt_to_equity": info.get('debtToEquity'),
            "return_on_equity": info.get('returnOnEquity'),
            "return_on_assets": info.get('returnOnAssets'),
            "profit_margin": info.get('profitMargins'),
            "operating_margin": info.get('operatingMargins'),
            "gross_margin": info.get('grossMargins'),
            "current_ratio": info.get('currentRatio'),
            "quick_ratio": info.get('quickRatio'),
        }
        
        # Calculate historical ratios from financials
        # (Simplified - in production would calculate from actual financial statements)
        return {
            "current": current_ratios,
            "trend": "Historical trend data would be calculated from 5-year financial statements"
        }
    except Exception as e:
        logger.error(f"Error fetching ratio trends for {ticker}: {e}")
        return None


def get_analyst_estimates(ticker: str) -> Optional[Dict[str, Any]]:
    """
    Get analyst estimates: price targets, EPS estimates, recommendations.
    """
    try:
        t = yf.Ticker(ticker)
        
        # Get analyst price targets
        price_targets = t.info.get('targetMeanPrice')
        target_high = t.info.get('targetHighPrice')
        target_low = t.info.get('targetLowPrice')
        target_median = t.info.get('targetMedianPrice')
        
        # Get recommendations
        recommendations = t.recommendations
        
        # Format recommendations
        rec_summary = {}
        if recommendations is not None and not recommendations.empty:
            latest_rec = recommendations.iloc[0] if len(recommendations) > 0 else None
            if latest_rec is not None:
                rec_summary = {
                    "period": str(latest_rec.name) if hasattr(latest_rec.name, 'strftime') else str(latest_rec.name),
                    "strong_buy": int(latest_rec.get('Strong Buy', 0)),
                    "buy": int(latest_rec.get('Buy', 0)),
                    "hold": int(latest_rec.get('Hold', 0)),
                    "sell": int(latest_rec.get('Sell', 0)),
                    "strong_sell": int(latest_rec.get('Strong Sell', 0)),
                }
        
        # Get recommendation trend safely
        rec_trend = None
        try:
            if hasattr(t, 'recommendation_trend') and t.recommendation_trend is not None:
                rec_trend = t.recommendation_trend.to_dict()
        except Exception:
            pass
        
        # Get earnings estimate safely
        earnings_est = None
        try:
            if hasattr(t, 'earnings_estimate') and t.earnings_estimate is not None:
                earnings_est = t.earnings_estimate.to_dict()
        except Exception:
            pass
        
        return {
            "price_targets": {
                "mean": price_targets,
                "median": target_median,
                "high": target_high,
                "low": target_low,
            },
            "recommendations": rec_summary,
            "recommendation_trend": rec_trend,
            "earnings_estimate": earnings_est,
        }
    except Exception as e:
        logger.error(f"Error fetching analyst estimates for {ticker}: {e}")
        return None


def get_segment_data(ticker: str) -> Optional[Dict[str, Any]]:
    """
    Get segment revenue breakdown for conglomerates.
    """
    try:
        t = yf.Ticker(ticker)
        
        # yfinance doesn't reliably provide segment data for all tickers
        # For now, return placeholder - in production would scrape from annual reports
        return {
            "available": False,
            "message": "Segment data not available via yfinance. Would need to scrape from company annual reports.",
            "note": "For NPN.JO (Naspers), segments typically include: Classifieds, Food Delivery, Payments, E-commerce"
        }
    except Exception as e:
        logger.error(f"Error fetching segment data for {ticker}: {e}")
        return None


def get_fair_value_comparison(ticker: str, dcf_value: float) -> Optional[Dict[str, Any]]:
    """
    Compare DCF valuation to analyst consensus and peers.
    """
    try:
        t = yf.Ticker(ticker)
        info = t.info
        
        current_price = info.get('currentPrice', info.get('regularMarketPrice'))
        target_mean = info.get('targetMeanPrice')
        
        if not current_price:
            return None
        
        # Calculate upside/downside
        dcf_upside = ((dcf_value - current_price) / current_price) * 100 if current_price else 0
        analyst_upside = ((target_mean - current_price) / current_price) * 100 if target_mean else 0
        
        return {
            "current_price": current_price,
            "your_dcf": dcf_value,
            "dcf_upside": round(dcf_upside, 2),
            "analyst_target": target_mean,
            "analyst_upside": round(analyst_upside, 2),
            "verdict": "Undervalued" if dcf_value > current_price else "Overvalued",
            "confidence": "High" if abs(dcf_upside) > 20 else "Medium" if abs(dcf_upside) > 10 else "Low"
        }
    except Exception as e:
        logger.error(f"Error fetching fair value comparison for {ticker}: {e}")
        return None

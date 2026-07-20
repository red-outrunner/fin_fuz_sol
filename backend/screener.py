"""Stock screener and discovery features: filtering, heatmap data, stock ideas feed."""
import os
import logging
import concurrent.futures
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf

from market_data import normalize_dividend_yield, download_data
from serialization import clean_data

logger = logging.getLogger(__name__)

# === JSE Top 40 Tickers (as of 2025) ===
# Source: Satrix Top 40 constituents
JSE_TOP_40 = [
    "NPN.JO",  # Naspers
    "PRX.JO",  # Prosus
    "AGL.JO",  # Anglo American
    "ANG.JO",  # AngloGold Ashanti
    "ANH.JO",  # Anheuser-Busch InBev
    "ABG.JO",  # Absa Group
    "BHG.JO",  # BHP Group
    "BID.JO",  # Bid Corporation
    "BTI.JO",  # British American Tobacco
    "CPI.JO",  # Capitec Bank
    "CLS.JO",  # Clicks Group
    "DSY.JO",  # Discovery
    "EXX.JO",  # Exxaro Resources
    "FSR.JO",  # FirstRand
    "GFI.JO",  # Gold Fields
    "GRT.JO",  # Growthpoint Properties
    "IMP.JO",  # Impala Platinum
    "INL.JO",  # Investec
    "INP.JO",  # Investec
    "MCG.JO",  # MultiChoice Group
    "MRP.JO",  # Mr Price Group
    "MTN.JO",  # MTN Group
    "NED.JO",  # Nedbank
    "NPH.JO",  # Northam Platinum
    "OMU.JO",  # Old Mutual
    "RNI.JO",  # Remgro
    "REM.JO",  # Remgro
    "SBK.JO",  # Standard Bank
    "SHP.JO",  # Shoprite Holdings
    "SLM.JO",  # Sanlam
    "SOL.JO",  # Sasol
    "VOD.JO",  # Vodacom
    "WHL.JO",  # Woolworths Holdings
    "BVT.JO",  # Bidvest
    "APN.JO",  # Aspen Pharmacare
    "VAL.JO",  # Vulcan Materials
    "MNP.JO",  # Murray & Roberts
    "GLN.JO",  # Glencore
]

# Sector mappings for JSE stocks
JSE_SECTORS = {
    # Technology
    "NPN.JO": "Technology",
    "PRX.JO": "Technology",
    "MCG.JO": "Technology",
    # Financials
    "ABG.JO": "Financials",
    "CPI.JO": "Financials",
    "FSR.JO": "Financials",
    "NED.JO": "Financials",
    "SBK.JO": "Financials",
    "INL.JO": "Financials",
    "INP.JO": "Financials",
    "OMU.JO": "Financials",
    "SLM.JO": "Financials",
    "GRT.JO": "Financials",
    # Materials / Mining
    "AGL.JO": "Materials",
    "ANG.JO": "Materials",
    "BHG.JO": "Materials",
    "EXX.JO": "Materials",
    "GFI.JO": "Materials",
    "IMP.JO": "Materials",
    "NPH.JO": "Materials",
    "SOL.JO": "Materials",
    "VAL.JO": "Materials",
    "GLN.JO": "Materials",
    "MNP.JO": "Materials",
    # Consumer
    "ANH.JO": "Consumer",
    "BID.JO": "Consumer",
    "BTI.JO": "Consumer",
    "CLS.JO": "Consumer",
    "MRP.JO": "Consumer",
    "SHP.JO": "Consumer",
    "WHL.JO": "Consumer",
    "BVT.JO": "Consumer",
    "APN.JO": "Consumer",
    # Telecom
    "MTN.JO": "Telecom",
    "VOD.JO": "Telecom",
    # Healthcare
    "DSY.JO": "Healthcare",
    "RNI.JO": "Healthcare",
    "REM.JO": "Healthcare",
}


def _fetch_ticker_info(ticker: str) -> Optional[Dict[str, Any]]:
    """Fetches fundamental info for a single ticker."""
    try:
        t = yf.Ticker(ticker)
        info = t.info
        
        # Skip if no essential data
        if not info.get('marketCap') and not info.get('currentPrice'):
            return None
        
        # Get recent price data for 52-week calculation
        try:
            hist = t.history(period="1y")
            if hist.empty:
                high_52w = None
                low_52w = None
                current_price = info.get('currentPrice', info.get('regularMarketPrice'))
            else:
                high_52w = hist['High'].max()
                low_52w = hist['Low'].min()
                current_price = hist['Close'].iloc[-1] if not hist.empty else None
        except Exception:
            high_52w = None
            low_52w = None
            current_price = info.get('currentPrice', info.get('regularMarketPrice'))
        
        # Calculate distance from 52-week high/low
        pct_from_high = None
        pct_from_low = None
        if current_price and high_52w:
            pct_from_high = ((current_price - high_52w) / high_52w) * 100
        if current_price and low_52w:
            pct_from_low = ((current_price - low_52w) / low_52w) * 100
        
        # Get dividend yield
        div_yield = normalize_dividend_yield(info.get('dividendYield'))
        
        # Get sector
        sector = JSE_SECTORS.get(ticker, info.get('sector', 'Other'))
        
        # Calculate valuation metrics
        trailing_pe = info.get('trailingPE')
        forward_pe = info.get('forwardPE')
        pb_ratio = info.get('priceToBook')
        peg_ratio = info.get('pegRatio')
        
        # Financial health
        debt_equity = info.get('debtToEquity')
        roe = info.get('returnOnEquity')
        roa = info.get('returnOnAssets')
        profit_margin = info.get('profitMargins')
        
        # Growth
        revenue_growth = info.get('revenueGrowth')
        earnings_growth = info.get('earningsGrowth')
        
        # Trading
        beta = info.get('beta')
        avg_volume = info.get('averageVolume')
        market_cap = info.get('marketCap')
        
        return {
            "ticker": ticker,
            "name": info.get('shortName', info.get('longName', ticker)),
            "sector": sector,
            "industry": info.get('industry', 'N/A'),
            "current_price": current_price,
            "currency": info.get('currency', 'ZAR'),
            "market_cap": market_cap,
            "pe_ratio": trailing_pe,
            "forward_pe": forward_pe,
            "peg_ratio": peg_ratio,
            "price_to_book": pb_ratio,
            "dividend_yield": div_yield,
            "debt_to_equity": debt_equity,
            "return_on_equity": roe,
            "return_on_assets": roa,
            "profit_margin": profit_margin,
            "revenue_growth": revenue_growth,
            "earnings_growth": earnings_growth,
            "beta": beta,
            "avg_volume": avg_volume,
            "high_52w": high_52w,
            "low_52w": low_52w,
            "pct_from_high": pct_from_high,
            "pct_from_low": pct_from_low,
        }
    except Exception as e:
        logger.debug(f"Error fetching info for {ticker}: {e}")
        return None


def get_jse_universe() -> List[str]:
    """Returns the list of JSE Top 40 tickers."""
    return JSE_TOP_40.copy()


def screen_stocks(
    min_market_cap: Optional[float] = None,
    max_market_cap: Optional[float] = None,
    min_pe: Optional[float] = None,
    max_pe: Optional[float] = None,
    min_dividend_yield: Optional[float] = None,
    max_pe_ratio: Optional[float] = None,
    min_roe: Optional[float] = None,
    max_debt_equity: Optional[float] = None,
    min_beta: Optional[float] = None,
    max_beta: Optional[float] = None,
    sectors: Optional[List[str]] = None,
    min_revenue_growth: Optional[float] = None,
    min_profit_margin: Optional[float] = None,
    undervalued_only: bool = False,
    dividend_growers_only: bool = False,
) -> List[Dict[str, Any]]:
    """
    Screens JSE stocks based on fundamental criteria.
    
    Args:
        min_market_cap: Minimum market cap (in currency units)
        max_market_cap: Maximum market cap
        min_pe: Minimum P/E ratio
        max_pe: Maximum P/E ratio (for value screening)
        min_dividend_yield: Minimum dividend yield (as decimal, e.g., 0.03 for 3%)
        max_pe_ratio: Alias for max_pe
        min_roe: Minimum return on equity (as decimal)
        max_debt_equity: Maximum debt-to-equity ratio
        min_beta: Minimum beta
        max_beta: Maximum beta (for low volatility)
        sectors: List of sectors to include
        min_revenue_growth: Minimum revenue growth (as decimal)
        min_profit_margin: Minimum profit margin (as decimal)
        undervalued_only: If True, only stocks trading >20% below 52-week high
        dividend_growers_only: If True, only stocks with dividend yield > 0
        
    Returns:
        List of stock data dictionaries matching criteria
    """
    logger.info(f"Screening stocks with criteria: P/E max={max_pe or max_pe_ratio}, "
                f"div yield min={min_dividend_yield}, sectors={sectors}")
    
    results = []
    max_pe = max_pe or max_pe_ratio  # Support both parameter names
    
    # Fetch data for all JSE Top 40 in parallel
    tickers_to_screen = JSE_TOP_40
    logger.info(f"Screening {len(tickers_to_screen)} JSE Top 40 stocks...")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        future_to_ticker = {
            executor.submit(_fetch_ticker_info, ticker): ticker
            for ticker in tickers_to_screen
        }
        
        for future in concurrent.futures.as_completed(future_to_ticker):
            try:
                data = future.result()
                if data is None:
                    continue
                
                # Apply filters
                # Market cap
                if min_market_cap and data.get('market_cap') and data['market_cap'] < min_market_cap:
                    continue
                if max_market_cap and data.get('market_cap') and data['market_cap'] > max_market_cap:
                    continue
                
                # P/E ratio
                if min_pe and data.get('pe_ratio') and data['pe_ratio'] < min_pe:
                    continue
                if max_pe and data.get('pe_ratio') and data['pe_ratio'] > max_pe:
                    continue
                
                # Dividend yield
                if min_dividend_yield and data.get('dividend_yield') and data['dividend_yield'] < min_dividend_yield:
                    continue
                if dividend_growers_only and (not data.get('dividend_yield') or data['dividend_yield'] <= 0):
                    continue
                
                # ROE
                if min_roe and data.get('return_on_equity') and data['return_on_equity'] < min_roe:
                    continue
                
                # Debt/Equity
                if max_debt_equity and data.get('debt_to_equity') and data['debt_to_equity'] > max_debt_equity:
                    continue
                
                # Beta
                if min_beta and data.get('beta') and data['beta'] < min_beta:
                    continue
                if max_beta and data.get('beta') and data['beta'] > max_beta:
                    continue
                
                # Sector
                if sectors and data.get('sector') not in sectors:
                    continue
                
                # Revenue growth
                if min_revenue_growth and data.get('revenue_growth') and data['revenue_growth'] < min_revenue_growth:
                    continue
                
                # Profit margin
                if min_profit_margin and data.get('profit_margin') and data['profit_margin'] < min_profit_margin:
                    continue
                
                # Undervalued (trading significantly below 52-week high)
                if undervalued_only:
                    pct_from_high = data.get('pct_from_high')
                    if pct_from_high is None or pct_from_high > -20:  # Not at least 20% below high
                        continue
                
                results.append(data)
                
            except Exception as e:
                logger.error(f"Error processing ticker: {e}")
    
    # Sort by market cap descending (largest first)
    results.sort(key=lambda x: x.get('market_cap') or 0, reverse=True)
    
    return clean_data(results)


def get_sector_performance() -> List[Dict[str, Any]]:
    """
    Calculates sector performance for JSE heatmap.
    Returns performance data by sector for treemap visualization.
    """
    logger.info("Calculating JSE sector performance...")
    
    sector_data = {}
    
    # Fetch data for all sectors in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        future_to_ticker = {
            executor.submit(_fetch_ticker_info, ticker): ticker
            for ticker in JSE_TOP_40
        }
        
        for future in concurrent.futures.as_completed(future_to_ticker):
            try:
                data = future.result()
                if data is None:
                    continue
                
                sector = data.get('sector', 'Other')
                
                if sector not in sector_data:
                    sector_data[sector] = {
                        "stocks": [],
                        "total_market_cap": 0,
                        "performance_sum": 0,
                        "count": 0
                    }
                
                # Calculate 1-day change (approximate from recent data)
                try:
                    t = yf.Ticker(data['ticker'])
                    hist = t.history(period="5d")
                    if len(hist) >= 2:
                        day_change = ((hist['Close'].iloc[-1] - hist['Close'].iloc[0]) / hist['Close'].iloc[0]) * 100
                    else:
                        day_change = 0
                except Exception:
                    day_change = 0
                
                stock_info = {
                    "ticker": data['ticker'],
                    "name": data.get('name', data['ticker']),
                    "market_cap": data.get('market_cap') or 0,
                    "change_percent": day_change,
                    "current_price": data.get('current_price'),
                }
                
                sector_data[sector]["stocks"].append(stock_info)
                sector_data[sector]["total_market_cap"] += data.get('market_cap') or 0
                sector_data[sector]["performance_sum"] += day_change
                sector_data[sector]["count"] += 1
                
            except Exception as e:
                logger.error(f"Error in sector performance: {e}")
    
    # Build sector summaries
    sectors = []
    for sector_name, data in sector_data.items():
        avg_performance = data["performance_sum"] / data["count"] if data["count"] > 0 else 0
        
        sectors.append({
            "name": sector_name,
            "market_cap": data["total_market_cap"],
            "change_percent": avg_performance,
            "stock_count": data["count"],
            "stocks": data["stocks"]
        })
    
    # Sort sectors by market cap
    sectors.sort(key=lambda x: x["market_cap"], reverse=True)
    
    return clean_data(sectors)


def get_stock_ideas() -> Dict[str, List[Dict[str, Any]]]:
    """
    Generates curated stock ideas feed:
    - Undervalued stocks (low P/E, high dividend)
    - 52-week lows (potential bargains)
    - High dividend growers
    - Strong momentum (near 52-week high)
    """
    logger.info("Generating stock ideas feed...")
    
    # Fetch all stock data
    all_stocks = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        future_to_ticker = {
            executor.submit(_fetch_ticker_info, ticker): ticker
            for ticker in JSE_TOP_40
        }
        
        for future in concurrent.futures.as_completed(future_to_ticker):
            try:
                data = future.result()
                if data:
                    all_stocks.append(data)
            except Exception as e:
                logger.error(f"Error fetching stock data: {e}")
    
    ideas = {
        "undervalued": [],
        "52_week_lows": [],
        "dividend_stars": [],
        "momentum_leaders": [],
        "growth_stocks": []
    }
    
    for stock in all_stocks:
        pe = stock.get('pe_ratio')
        div_yield = stock.get('dividend_yield') or 0
        pct_from_high = stock.get('pct_from_high')
        pct_from_low = stock.get('pct_from_low')
        revenue_growth = stock.get('revenue_growth')
        roe = stock.get('return_on_equity')
        
        # Undervalued: P/E < 12 and dividend yield > 3%
        if pe and pe < 12 and div_yield > 0.03:
            ideas["undervalued"].append(stock)
        
        # 52-week lows: Trading within 10% of 52-week low
        if pct_from_low is not None and pct_from_low < 0.10:
            ideas["52_week_lows"].append(stock)
        
        # Dividend stars: Yield > 5%
        if div_yield > 0.05:
            ideas["dividend_stars"].append(stock)
        
        # Momentum leaders: Trading within 10% of 52-week high
        if pct_from_high is not None and pct_from_high > -10:
            ideas["momentum_leaders"].append(stock)
        
        # Growth stocks: Revenue growth > 10% and ROE > 15%
        if revenue_growth and revenue_growth > 0.10 and roe and roe > 0.15:
            ideas["growth_stocks"].append(stock)
    
    # Sort each category
    ideas["undervalued"].sort(key=lambda x: x.get('pe_ratio') or 999)
    ideas["52_week_lows"].sort(key=lambda x: x.get('pct_from_low') or 999)
    ideas["dividend_stars"].sort(key=lambda x: x.get('dividend_yield') or 0, reverse=True)
    ideas["momentum_leaders"].sort(key=lambda x: x.get('pct_from_high') or -999, reverse=True)
    ideas["growth_stocks"].sort(key=lambda x: x.get('revenue_growth') or 0, reverse=True)
    
    # Limit to top 10 per category
    for key in ideas:
        ideas[key] = ideas[key][:10]
    
    return clean_data(ideas)


def get_ticker_details(ticker: str) -> Optional[Dict[str, Any]]:
    """Gets detailed info for a single ticker."""
    return _fetch_ticker_info(ticker)

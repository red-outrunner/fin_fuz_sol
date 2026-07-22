"""Quantitative analytics layer: return processing, summary statistics, ML,
seasonality tests, DCA, Monte Carlo, financial-freedom, and peer lookup. Operates
on data already fetched by the market_data layer (no network access here)."""
import os
import sys
import logging

import numpy as np
import pandas as pd
from scipy import stats

# core_math lives in the repo root.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import core_math
from serialization import clean_data

logger = logging.getLogger(__name__)


def process_data(data: pd.DataFrame):
    """Processes raw data into monthly returns and pivot tables."""
    try:
        data = core_math.validate_and_clean_data(data)

        # Determine price column
        if "Adj Close" in data.columns:
            price_col = "Adj Close"
        elif "Close" in data.columns:
            price_col = "Close"
        else:
            raise ValueError("No valid price column found")

        # Resample to monthly
        monthly = data[price_col].resample('ME').last()
        monthly_ret = monthly.pct_change().dropna()

        if isinstance(monthly_ret, pd.DataFrame):
            monthly_ret = monthly_ret.iloc[:, 0]

        if monthly_ret.empty:
            return None

        # Create DataFrame for pivot
        df = monthly_ret.to_frame(name='ret')
        df['year'] = df.index.year
        df['month'] = df.index.month

        pivot = df.pivot_table(index='year', columns='month', values='ret')

        # Ensure all months are present
        for i in range(1, 13):
            if i not in pivot.columns:
                pivot[i] = np.nan

        # Do NOT replace NaN with None here, as it converts to object dtype and breaks stats calculation
        # pivot = pivot.where(pd.notnull(pivot), None)

        raw_monthly_ret = monthly_ret.copy()
        raw_df = df.copy()
        raw_pivot = pivot.copy()

        # Apply winsorization for plotting series
        winsorized_ret = core_math.apply_outlier_filtering(monthly_ret)
        win_df = winsorized_ret.to_frame(name='ret')
        win_df['year'] = win_df.index.year
        win_df['month'] = win_df.index.month
        winsorized_pivot = win_df.pivot_table(index='year', columns='month', values='ret')

        for i in range(1, 13):
            if i not in winsorized_pivot.columns:
                winsorized_pivot[i] = np.nan

        # Calculate Moving Averages (on monthly close prices)
        ma_12 = monthly.rolling(window=12).mean()
        ma_60 = monthly.rolling(window=60).mean()

        return {
            "monthly_ret": raw_monthly_ret,
            "pivot": raw_pivot,
            "df": raw_df,
            "winsorized_ret": winsorized_ret,
            "winsorized_pivot": winsorized_pivot,
            "prices": monthly,
            "ma_12": ma_12,
            "ma_60": ma_60
        }
    except Exception as e:
        logger.error(f"Error processing data: {e}")
        return None


def calculate_summary_stats(monthly_ret: pd.Series, pivot: pd.DataFrame, ticker: str = None, inflation_rate: float = 0.0):
    """Calculates summary statistics using raw returns (fat tail risk intact).

    ticker        - used to pick a country-aware risk-free rate for Sharpe/Sortino.
    inflation_rate - annual rate (e.g. 0.05). When > 0 all return-based metrics
                     (CAGR, vol, Sharpe, Sortino, drawdown, wealth, annual returns)
                     are computed on the REAL return series and the risk-free rate is
                     converted to real terms, so "inflation-adjusted" is consistent.
    """
    try:

        # Real (inflation-adjusted) return series when requested; otherwise nominal.
        # When inflation_rate == 0 this equals monthly_ret exactly, so the default
        # behaviour is unchanged.
        if inflation_rate and inflation_rate > 0:
            monthly_inflation = (1 + inflation_rate) ** (1/12) - 1
            calc_series = (1 + monthly_ret) / (1 + monthly_inflation) - 1
        else:
            calc_series = monthly_ret

        # Seasonality descriptors stay on the nominal monthly pivot.
        month_avg = pivot.mean()
        month_median = pivot.median()
        overall_avg = calc_series.mean()

        best_month = month_avg.idxmax()
        worst_month = month_avg.idxmin()

        months_map = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
                      7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}

        # Country-aware risk-free rate (e.g. ~10% for .JO, ~4.2% US) instead of a flat
        # 4%. If inflation-adjusting, convert the nominal rate to a real rate so it is
        # comparable with the real return series.
        nominal_rf = core_math.get_risk_free_rate(ticker) if ticker else 0.04
        if inflation_rate and inflation_rate > 0:
            risk_free_rate = (1 + nominal_rf) / (1 + inflation_rate) - 1
        else:
            risk_free_rate = nominal_rf

        # Advanced Metrics Calculation (on the chosen real/nominal series)
        cagr = core_math.calculate_cagr(calc_series)
        volatility = core_math.calculate_volatility(calc_series)

        sharpe_ratio = core_math.calculate_sharpe(cagr, volatility, risk_free_rate)
        sortino_ratio = core_math.calculate_sortino(calc_series, cagr, risk_free_rate)
        max_drawdown = core_math.calculate_max_drawdown(calc_series)

        cumulative_returns = (1 + calc_series).cumprod()
        peak = cumulative_returns.expanding(min_periods=1).max()
        drawdown = (cumulative_returns / peak) - 1

        # 5. Wealth Index (Growth of 10,000)
        wealth_index = 10000 * (1 + calc_series).cumprod()
        # Prepend starting value
        wealth_data = [{"date": str(calc_series.index[0] - pd.DateOffset(months=1)).split(" ")[0], "value": 10000}]
        wealth_data.extend([{"date": str(d).split(" ")[0], "value": v} for d, v in wealth_index.items()])

        # 6. Drawdown Series
        drawdown_series = drawdown.to_dict()
        drawdown_data = [{"date": str(d).split(" ")[0], "value": v} for d, v in drawdown_series.items()]

        # 7. Annual Returns
        # Group monthly returns by year and calc product
        annual_ret = (1 + calc_series).groupby(calc_series.index.year).prod() - 1
        annual_returns_data = [{"year": y, "value": v} for y, v in annual_ret.items()]

        stats_dict = {
            "overall_avg": overall_avg,
            "month_avg": month_avg.to_dict(),
            "month_median": month_median.to_dict(),
            "best_month": {"index": int(best_month), "name": months_map.get(best_month), "value": month_avg[best_month]},
            "worst_month": {"index": int(worst_month), "name": months_map.get(worst_month), "value": month_avg[worst_month]},
            "std_dev": pivot.std().to_dict(),
            "positive_rate": ((pivot > 0).sum() / pivot.count()).to_dict(),
            "cagr": cagr,
            "volatility": volatility,
            "risk_free_rate": risk_free_rate,
            "sharpe_ratio": sharpe_ratio,
            "sortino_ratio": sortino_ratio,
            "max_drawdown": max_drawdown,
            "wealth_index": wealth_data,
            "drawdown_series": drawdown_data,
            "annual_returns": annual_returns_data
        }
        return clean_data(stats_dict)
    except Exception as e:
        logger.error(f"Error calculating stats: {e}")
        return None


def run_ml_analysis(monthly_ret: pd.Series):
    """Runs PCA, GMM, and Isolation Forest.

    Adds a `dates` list ("YYYY-MM") aligned with the results so the frontend can
    label each point with its month (same alignment as the ML CSV export: each
    12-month window is labeled with the month it leads into).
    """
    try:
        results = core_math.run_ml_clusters(monthly_ret)
        if results is None:
            return None

        dates = monthly_ret.index[12:]
        if len(dates) != len(results["clusters"]):
            dates = dates[-len(results["clusters"]):]
        results["dates"] = [d.strftime("%Y-%m") for d in dates]
        return results
    except Exception as e:
        logger.error(f"Error in ML analysis: {e}")
        return None


def run_anova_test(pivot: pd.DataFrame):
    """Runs ANOVA test for seasonality."""
    try:
        month_groups = [pivot[c].dropna().values for c in pivot.columns if not pivot[c].dropna().empty]
        if len(month_groups) < 2:
            return {"error": "Not enough data for ANOVA"}

        f_stat, p_val = stats.f_oneway(*month_groups)

        return {
            "f_stat": float(f_stat) if not np.isnan(f_stat) else None,
            "p_value": float(p_val) if not np.isnan(p_val) else None,
            "significant": bool(p_val < 0.05) if not np.isnan(p_val) else False
        }
    except Exception as e:
        return {"error": str(e)}


def calculate_dca(monthly_ret: pd.Series, monthly_contribution: float):
    """
    Simulates Dollar Cost Averaging.
    """
    try:
        return core_math.calculate_dca(monthly_ret, monthly_contribution)
    except Exception as e:
        logger.error(f"Error calculating DCA: {e}")
        return None


def calculate_correlation(pivot: pd.DataFrame):
     # Placeholder if needed, but we can do it in main logic using pandas corr() directly on cleaned data
     pass


def run_monte_carlo(monthly_ret: pd.Series, years: int = 10, n_sims: int = 1000, dividend_yield: float = 0.0):
    """
    Runs Monte Carlo simulation for future wealth projection.
    dividend_yield is an annual fraction (e.g. 0.03) reinvested into the total return.
    """
    try:
        return core_math.run_monte_carlo(monthly_ret, years, n_sims, method='gbm', dividend_yield=dividend_yield)
    except Exception as e:
        logger.error(f"Error in Monte Carlo: {e}")
        return None


def calculate_financial_freedom(data, monthly_income_goal, ticker: str = None):
    """
    Calculates the number of shares and investment required to generate a target monthly income.
    Uses actual dividend data from yfinance with proper currency conversion to ZAR.

    Args:
        data: Price data DataFrame (from download_data)
        monthly_income_goal: Target monthly dividend income (assumed ZAR)
        ticker: Optional ticker symbol to fetch actual dividend data
    """
    try:
        import yfinance as yf

        ticker_obj = yf.Ticker(ticker) if ticker else None
        info = ticker_obj.info if ticker_obj else {}

        # Get last price
        if "Close" in data.columns:
            current_price = data["Close"].iloc[-1]
        elif "Adj Close" in data.columns:
            current_price = data["Adj Close"].iloc[-1]
        else:
            return None

        # Detect currency and convert to ZAR
        # yfinance sometimes misreports currency, so we use ticker suffix as primary indicator
        ticker_upper = (ticker or "").upper()
        
        if ticker_upper.endswith(".JO"):
            # JSE stocks: prices in ZAR (Rand), dividends in ZAR
            # Note: Some older yfinance versions return ZAc (cents), so we detect by magnitude
            price_in_zar = current_price
            # If price looks like cents (>500 suggests it's ZAc, since few JSE stocks trade above R500)
            if current_price > 500:
                price_in_zar = current_price / 100.0
        elif ticker_upper.endswith(".US") or "." not in ticker_upper:
            # US stocks: convert USD to ZAR
            try:
                zar_per_usd = yf.Ticker("ZAR=X").history(period="1d")["Close"].iloc[-1]
            except:
                zar_per_usd = 18.0
            price_in_zar = current_price * zar_per_usd
        else:
            # Other exchanges (UK, EU, AU, etc.) - use info currency or default to no conversion
            currency = info.get("currency", "USD")
            if currency == "ZAc":
                price_in_zar = current_price / 100.0
            elif currency == "USD":
                try:
                    zar_per_usd = yf.Ticker("ZAR=X").history(period="1d")["Close"].iloc[-1]
                except:
                    zar_per_usd = 18.0
                price_in_zar = current_price * zar_per_usd
            else:
                # GBP, EUR, AUD, etc. - would need additional FX conversion (TODO)
                price_in_zar = current_price

        # Fetch actual dividend data from yfinance
        annual_yield = 0.0
        dividends_12m_raw = 0.0

        if ticker and ticker_obj:
            try:
                dividends = ticker_obj.dividends

                if dividends is not None and not dividends.empty:
                    # Sum last 12 months of dividends
                    last_year = dividends.index.max() - pd.DateOffset(months=12)
                    dividends_12m_raw = dividends[dividends.index >= last_year].sum()

                    # Convert dividends to ZAR using same logic as price
                    dividends_in_zar = dividends_12m_raw
                    
                    if ticker_upper.endswith(".JO"):
                        # JSE: check if dividends are in cents (same magnitude logic as price)
                        if dividends_12m_raw > 500:  # Likely in ZAc
                            dividends_in_zar = dividends_12m_raw / 100.0
                    elif ticker_upper.endswith(".US") or "." not in ticker_upper:
                        # US: convert USD dividends to ZAR
                        try:
                            zar_per_usd = yf.Ticker("ZAR=X").history(period="1d")["Close"].iloc[-1]
                        except:
                            zar_per_usd = 18.0
                        dividends_in_zar = dividends_12m_raw * zar_per_usd
                    else:
                        # Other currencies - use info currency
                        currency = info.get("currency", "USD")
                        if currency == "ZAc":
                            dividends_in_zar = dividends_12m_raw / 100.0
                        elif currency == "USD":
                            try:
                                zar_per_usd = yf.Ticker("ZAR=X").history(period="1d")["Close"].iloc[-1]
                            except:
                                zar_per_usd = 18.0
                            dividends_in_zar = dividends_12m_raw * zar_per_usd

                    if price_in_zar > 0:
                        annual_yield = dividends_in_zar / price_in_zar
                        dividends_12m = dividends_in_zar  # Store the ZAR-converted value
            except Exception as e:
                logger.warning(f"Could not fetch dividend data for {ticker}: {e}")

        # Fallback: if still no yield, return zeros (frontend should warn user)
        if annual_yield <= 0:
            return {
                "current_price": price_in_zar,
                "annual_yield": 0.0,
                "shares_needed": 0,
                "investment_needed": 0,
                "monthly_income_goal": monthly_income_goal,
                "warning": "No dividend data available for this stock"
            }

        estimated_annual_income_needed = monthly_income_goal * 12

        investment_needed = estimated_annual_income_needed / annual_yield
        shares_needed = int(investment_needed / price_in_zar)

        return {
            "current_price": price_in_zar,
            "annual_yield": annual_yield,
            "dividends_12m": dividends_12m,
            "shares_needed": shares_needed,
            "investment_needed": investment_needed,
            "monthly_income_goal": monthly_income_goal,
            "currency": "ZAR"
        }
    except Exception as e:
        logger.error(f"Error calculating financial freedom: {e}")
        return None


def get_jse_peers(ticker):
    """
    Returns a list of valid JSE competitor tickers based on the input ticker's sector.
    Now uses the 12-13 Satrix industries for accurate peer matching.
    ETFs compete with ETFs, Tech with Tech, Banks with Banks, etc.
    """
    # Normalize ticker
    ticker = ticker.upper().strip()
    
    # Import sector mappings from screener
    try:
        from screener import JSE_SECTORS
    except ImportError:
        JSE_SECTORS = {}
    
    # Get the sector/industry for the input ticker
    ticker_sector = JSE_SECTORS.get(ticker, None)
    
    if not ticker_sector:
        # Fallback to old method if sector not found
        sector_map = {
            "BANKS": ["SBK.JO", "FSR.JO", "NED.JO", "ABG.JO", "CPI.JO"],
            "RETAIL": ["SHP.JO", "PIK.JO", "WHL.JO", "SPP.JO", "MRP.JO"],
            "MINING": ["ANG.JO", "SOL.JO", "IMP.JO", "SSW.JO", "GFI.JO", "BHP.JO"],
            "TECH/PROSUS": ["NPN.JO", "PRX.JO"],
            "TELCO": ["MTN.JO", "VOD.JO", "TKG.JO"],
            "INSURANCE": ["SLM.JO", "OMU.JO", "DSY.JO"],
            "PROPERTY": ["GRT.JO", "NEP.JO", "RDF.JO"]
        }
        
        # Find which list the ticker belongs to
        for sector, members in sector_map.items():
            if ticker in members:
                peers = [m for m in members if m != ticker]
                return peers[:3]
        
        # Default fallback
        defaults = ["STX40.JO", "NPN.JO", "SBK.JO"]
        return [d for d in defaults if d != ticker]
    
    # NEW: Use 12-13 Satrix industries for accurate peer matching
    # Find all tickers in the same industry
    peers = [t for t, sector in JSE_SECTORS.items() if sector == ticker_sector and t != ticker]
    
    # Return top 3 peers (could be enhanced with market cap filtering)
    return peers[:4] if len(peers) > 3 else peers

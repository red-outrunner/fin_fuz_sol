import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import IsolationForest
from scipy import stats
import logging
import requests
from bs4 import BeautifulSoup
import re
import urllib.parse

import os
import json
import hashlib
import time
import concurrent.futures

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CACHE_DIR = "cache_v2_8_1"
os.makedirs(CACHE_DIR, exist_ok=True)

def clean_data(data):
    """Recursively replace NaN and Infinity with None for JSON serialization."""
    if isinstance(data, dict):
        return {k: clean_data(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [clean_data(v) for v in data]
    elif isinstance(data, (float, np.float64, np.float32)):
        if np.isnan(data) or np.isinf(data):
            return None
        return float(data)
    elif isinstance(data, (int, np.int64, np.int32)):
        return int(data)
    elif isinstance(data, pd.Series):
        return clean_data(data.to_dict())
    elif isinstance(data, pd.DataFrame):
        return clean_data(data.to_dict(orient='records'))
    return data

def _get_cache_path(ticker, start_date, end_date):
    """Generates a unique cache filename based on request parameters."""
    raw = f"{ticker}_{start_date}_{end_date}"
    hashed = hashlib.md5(raw.encode()).hexdigest()
    return os.path.join(CACHE_DIR, f"{hashed}.pkl")

def download_data(ticker: str, start_date: str, end_date: str):
    """Downloads data from yfinance with disk caching."""
    cache_path = _get_cache_path(ticker, start_date, end_date)
    
    # Check Cache
    if os.path.exists(cache_path):
        try:
            # Check if cache is older than 24 hours
            if time.time() - os.path.getmtime(cache_path) < 86400:
                return pd.read_pickle(cache_path)
        except Exception as e:
            logger.warning(f"Cache read error: {e}")

    try:
        # Use Ticker object to avoid shared state/caching issues with yf.download in threads
        ticker_obj = yf.Ticker(ticker)
        data = ticker_obj.history(start=start_date, end=end_date, auto_adjust=True)
        
        if data is None or data.empty:
            return None
            
        # Ensure index is datetime
        data.index = pd.to_datetime(data.index)
        
        # Save to Cache
        try:
            data.to_pickle(cache_path)
        except Exception as e:
            logger.warning(f"Cache write error: {e}")
        
        return data
    except Exception as e:
        logger.error(f"Error downloading data for {ticker}: {e}")
        return None

def fetch_multiple_tickers(tickers: list, start_date: str, end_date: str):
    """Fetches data for multiple tickers in parallel."""
    results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        future_to_ticker = {
            executor.submit(download_data, ticker, start_date, end_date): ticker 
            for ticker in tickers
        }
        for future in concurrent.futures.as_completed(future_to_ticker):
            ticker = future_to_ticker[future]
            try:
                data = future.result()
                if data is not None:
                    results[ticker] = data
            except Exception as e:
                logger.error(f"Parallel fetch error for {ticker}: {e}")
    return results

def process_data(data: pd.DataFrame):
    """Processes raw data into monthly returns and pivot tables."""
    try:
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
        
        # Calculate Moving Averages (on monthly close prices)
        # We use the monthly series which is already resampled
        ma_12 = monthly.rolling(window=12).mean()
        ma_60 = monthly.rolling(window=60).mean()

        return {
            "monthly_ret": monthly_ret,
            "pivot": pivot,
            "df": df,
            "prices": monthly,
            "ma_12": ma_12,
            "ma_60": ma_60
        }
    except Exception as e:
        logger.error(f"Error processing data: {e}")
        return None

def calculate_summary_stats(monthly_ret: pd.Series, pivot: pd.DataFrame, inflation_rate: float = 0.0):
    """Calculates summary statistics."""
    try:
        # Adjust for inflation if needed
        # inflation_rate is annual percentage (e.g. 0.05 for 5%)
        # Convert to monthly inflation
        if inflation_rate > 0:
            monthly_inflation = (1 + inflation_rate)**(1/12) - 1
            # Real Return = (1 + Nominal) / (1 + Inflation) - 1
            # Approximation: Nominal - Inflation
            real_ret = (1 + monthly_ret) / (1 + monthly_inflation) - 1
            calc_series = real_ret
        else:
            calc_series = monthly_ret

        month_avg = pivot.mean()
        month_median = pivot.median()
        overall_avg = monthly_ret.mean()
        
        best_month = month_avg.idxmax()
        worst_month = month_avg.idxmin()
        
        months_map = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun', 
                      7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}

        # Advanced Metrics Calculation
        # 1. CAGR
        total_return = (1 + monthly_ret).prod() - 1
        n_years = len(monthly_ret) / 12
        cagr = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0

        # 2. Volatility (Annualized)
        volatility = monthly_ret.std() * np.sqrt(12)

        # 3. Sharpe Ratio (Assume Rf=0.02 for simplicity)
        risk_free_rate = 0.02
        sharpe_ratio = (cagr - risk_free_rate) / volatility if volatility != 0 else 0
        
        # 3.5 Sortino Ratio
        # Downside deviation: std dev of negative returns only
        negative_returns = monthly_ret[monthly_ret < 0]
        downside_deviation = negative_returns.std() * np.sqrt(12)
        sortino_ratio = (cagr - risk_free_rate) / downside_deviation if downside_deviation != 0 else 0

        # 4. Max Drawdown
        cumulative_returns = (1 + monthly_ret).cumprod()
        peak = cumulative_returns.expanding(min_periods=1).max()
        drawdown = (cumulative_returns / peak) - 1
        max_drawdown = drawdown.min()

        # 5. Wealth Index (Growth of 10,000)
        wealth_index = 10000 * (1 + calc_series).cumprod()
        # Prepend starting value
        wealth_data = [{"date": str(monthly_ret.index[0] - pd.DateOffset(months=1)).split(" ")[0], "value": 10000}]
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
    """Runs PCA, GMM, and Isolation Forest."""
    try:
        data = monthly_ret.values
        X = []
        # Create sequences of 12 months
        if len(data) < 24: # Need at least some data
            return None
            
        for i in range(12, len(data)):
            X.append(data[i-12:i])
        X = np.array(X)
        
        if len(X) == 0:
            return None

        # PCA
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        
        # GMM
        gmm = GaussianMixture(n_components=3, random_state=42)
        labels = gmm.fit_predict(X_pca)
        
        # Isolation Forest
        iso = IsolationForest(contamination=0.05, random_state=42)
        anomalies = iso.fit_predict(X_pca)
        
        return {
            "pca_components": X_pca.tolist(),
            "clusters": labels.tolist(),
            "anomalies": anomalies.tolist()
        }
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
    Returns:
        dca_data: List of dicts with date, total_invested, portfolio_value
        summary: Dict with final stats
    """
    try:
        if monthly_ret.empty:
            return None
            
        dates = monthly_ret.index
        values = []
        
        total_invested = 0
        current_holdings = 0 # In dollars initially? No, we need to track value.
        # Simpler approach: 
        # Month 0: Invest $X. 
        # Month 1: (Old Value * (1+Ret)) + $X
        
        current_value = 0
        
        # We need a starting point before the first return?
        # Typically DCA implies buying at the *start* or *end* of the period.
        # Let's assume we contribute at the start of each month (before that month's return).
        
        data_points = []
        
        for date, ret in monthly_ret.items():
            # Contribute
            total_invested += monthly_contribution
            current_value += monthly_contribution
            
            # Grow
            current_value *= (1 + ret)
            
            # Store
            data_points.append({
                "date": str(date).split(" ")[0],
                "invested": total_invested,
                "value": current_value
            })
            
        total_profit = current_value - total_invested
        roi = (total_profit / total_invested) if total_invested > 0 else 0
        
        return {
            "dca_series": data_points,
            "summary": {
                "total_invested": total_invested,
                "final_value": current_value,
                "total_profit": total_profit,
                "roi": roi
            }
        }
    except Exception as e:
        logger.error(f"Error calculating DCA: {e}")
        return None

def calculate_correlation(pivot: pd.DataFrame):
     # Placeholder if needed, but we can do it in main logic using pandas corr() directly on cleaned data
     pass

def run_monte_carlo(monthly_ret: pd.Series, years: int = 10, n_sims: int = 1000):
    """
    Runs Monte Carlo simulation for future wealth projection.
    Returns: 10th, 50th, 90th percentile paths.
    """
    try:
        if monthly_ret.empty or len(monthly_ret) < 12:
            return None

        # Calculate parameters from history
        mu = monthly_ret.mean()
        sigma = monthly_ret.std()
        
        last_val = 10000 # Start projection at 10k or just relative 1.0
        months = years * 12
        
        # Simulation
        # Result shape: (months, n_sims)
        # Generate random returns: normal distribution
        sim_rets = np.random.normal(mu, sigma, (months, n_sims))
        
        # Calculate cumulative paths
        # (1 + r).cumprod()
        sim_growth = (1 + sim_rets).cumprod(axis=0)
        sim_paths = last_val * sim_growth
        
        # Insert start value
        sim_paths = np.vstack([np.full((1, n_sims), last_val), sim_paths])
        
        # Calculate percentiles at each time step
        p10 = np.percentile(sim_paths, 10, axis=1)
        p50 = np.percentile(sim_paths, 50, axis=1)
        p90 = np.percentile(sim_paths, 90, axis=1)
        
        # Format for chart
        # We need dates. Start from "Today" + 1 month
        start_date = pd.Timestamp.now()
        dates = [start_date + pd.DateOffset(months=i) for i in range(months + 1)]
        
        projection_data = []
        for i in range(len(dates)):
            projection_data.append({
                "date": str(dates[i]).split(" ")[0],
                "p10": p10[i],
                "p50": p50[i],
                "p90": p90[i]
            })
            
        return projection_data
    except Exception as e:
        logger.error(f"Error in Monte Carlo: {e}")
        return None



def get_company_profile(ticker: str):
    """
    Fetches company profile: major shareholder and sentiment.
    Returns None if data unavailable (e.g. indices).
    """
    try:
        t = yf.Ticker(ticker)
        
        # 1. Biggest Shareholder
        biggest_holder = None
        try:
             # institutional_holders returns a DataFrame
             ih = t.institutional_holders
             if ih is not None and not ih.empty:
                 # Usually columns are "Holder", "Shares", "Date Reported", "% Out", "Value"
                 # Sort by "% Out" or "Shares" just to be safe, though usually sorted by default
                 # yfinance column names can vary slightly by version, but 'Holder' is standard
                 # Let's take the first row
                 top_row = ih.iloc[0]
                 # specific handling for different column names
                 holder_name = top_row.get(0, top_row.iloc[0]) if isinstance(top_row.keys()[0], int) else top_row.get('Holder', top_row.get('Holder Name'))
                 pct_held = top_row.get(3, top_row.iloc[3]) if isinstance(top_row.keys()[0], int) else top_row.get('% Out', top_row.get('% Held'))
                 
                 biggest_holder = {
                     "name": str(holder_name),
                     "percent": str(pct_held) # keep as string (e.g. 0.08 or '8%') or float
                 }
        except Exception as e:
            # logger.warning(f"Could not fetch holders for {ticker}: {e}")
            pass

        # 2. Sentiment / Recommendation
        sentiment = None
        try:
            info = t.info
            rec_key = info.get('recommendationKey') # 'buy', 'hold', 'sell', 'strong_buy'
            rec_mean = info.get('recommendationMean') # 1.0 - 5.0 typically
            
            if rec_key:
                sentiment = {
                    "key": rec_key,
                    "score": rec_mean
                }
        except Exception as e:
            pass

        return {
            "biggest_shareholder": biggest_holder,
            "sentiment": sentiment,
            "sector": t.info.get('sector'),
            "industry": t.info.get('industry'),
            "summary": t.info.get('longBusinessSummary')
        }

    except Exception as e:
        logger.error(f"Error fetching profile: {e}")
        return None


def generate_fun_stats(info):
    """Generates entertaining statistics and estimated rankings."""
    try:
        current_price = info.get('currentPrice', info.get('regularMarketPrice', 0))
        market_cap = info.get('marketCap', 0)
        currency = info.get('currency', 'USD')
        
        # 1. Market Cap Ranking (Estimated)
        # Thresholds in Billions (USD approx)
        rank_badge = "Unranked"
        if market_cap > 2000_000_000_000: # 2T
            rank_badge = "Top 5 Global 🌍"
        elif market_cap > 1000_000_000_000: # 1T
            rank_badge = "Top 10 Global 🏆"
        elif market_cap > 500_000_000_000: # 500B
            rank_badge = "Top 20 Global 🚀"
        elif market_cap > 200_000_000_000:
            rank_badge = "Blue Chip Titan 🏛️"
        elif market_cap > 50_000_000_000:
            rank_badge = "Large Cap Leader 🏢"
        elif market_cap > 10_000_000_000:
            rank_badge = "Mid Cap Mover 🚤"
        elif market_cap > 2_000_000_000:
            rank_badge = "Small Cap Challenger 🧗"
        else:
            rank_badge = "Micro Cap Gem 💎"

        # 2. Burger Index (Big Mac Index proxy)
        # Approx Big Mac price: $5.69 (USD)
        burger_price = 5.69
        burgers = 0
        if current_price and currency == 'USD':
             burgers = current_price / burger_price
             
        burger_text = f"1 Share = {int(burgers)} Big Macs 🍔" if burgers > 0 else "N/A"

        # 3. Market Mood
        beta = info.get('beta', 1)
        mood = "Neutral 😐"
        if beta > 1.5:
             mood = "Wild Ride 🎢 (High Volatility)"
        elif beta > 1.1:
             mood = "Aggressive 🐂"
        elif beta < 0.8:
             mood = "Defensive 🛡️"
        elif beta < 0:
             mood = "Contrarian 🐻"
             
        return {
            "rank_badge": rank_badge,
            "burger_index": burger_text,
            "market_mood": mood
        }
    except Exception as e:
        logger.error(f"Error generating fun stats: {e}")
        return {}

def get_key_stats(ticker: str):
    """
    Fetches fundamental statistics for the company.
    """
    try:
        t = yf.Ticker(ticker)
        info = t.info
        
        # Helper to safely get value or formatting
        def fmt(key, is_pct=False, is_currency=False):
            val = info.get(key)
            if val is None: return None
            return val

        fun_stats = generate_fun_stats(info)

        stats = {
            "valuation": {
                "market_cap": fmt("marketCap"),
                "pe_ratio": fmt("trailingPE"),
                "forward_pe": fmt("forwardPE"),
                "peg_ratio": fmt("pegRatio"),
                "price_to_book": fmt("priceToBook"),
                "dividend_yield": fmt("dividendYield"),
            },
            "financials": {
                "revenue": fmt("totalRevenue"),
                "revenue_growth": fmt("revenueGrowth"),
                "gross_margins": fmt("grossMargins"),
                "operating_margins": fmt("operatingMargins"),
                "profit_margins": fmt("profitMargins"),
                "ebitda": fmt("ebitda"),
            },
            "trading": {
                "beta": fmt("beta"),
                "short_ratio": fmt("shortRatio"),
                "target_high": fmt("targetHighPrice"),
                "target_low": fmt("targetLowPrice"),
                "target_mean": fmt("targetMeanPrice"),
                "recommendation_mean": fmt("recommendationMean"),
            },
            "insight": {
                "rank": fun_stats.get("rank_badge"),
                "burgers": fun_stats.get("burger_index"),
                "mood": fun_stats.get("market_mood")
            }
        }
        return clean_data(stats)
    except Exception as e:
        logger.error(f"Error fetching stats for {ticker}: {e}")
        return None

def get_news(ticker: str):
    """
    Fetches latest news for the company using DuckDuckGo HTML search.
    """
    try:
        # Construct query
        query = f"{ticker} stock news"
        url = f"https://html.duckduckgo.com/html?q={urllib.parse.quote(query)}"
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        results = []
        
        # DDG HTML results are usually in .result__body
        for result in soup.select('.result'):
            try:
                title_tag = result.select_one('.result__a')
                if not title_tag: continue
                
                title = title_tag.get_text()
                link = title_tag.get('href')
                
                # DDG links are redirects, e.g. //duckduckgo.com/l/?uddg=...
                # We can try to extract 'uddg' param
                if 'duckduckgo.com/l/' in link:
                     parsed = urllib.parse.urlparse(link)
                     qs = urllib.parse.parse_qs(parsed.query)
                     if 'uddg' in qs:
                         link = qs['uddg'][0]

                snippet_tag = result.select_one('.result__snippet')
                snippet = snippet_tag.get_text() if snippet_tag else ""
                
                # Extract source from snippet or url
                domain = urllib.parse.urlparse(link).netloc.replace('www.', '')
                
                results.append({
                    "title": title,
                    "publisher": domain,
                    "link": link,
                    "date": "Recent", # DDG HTML doesn't reliably give dates
                    "thumbnail": None,
                    "summary": snippet
                })
                
                if len(results) >= 4: break
            except Exception:
                continue
                
        return results
    except Exception as e:
        logger.error(f"Error fetching news for {ticker}: {e}")
        return []

def get_article_content(url: str):
    """
    Fetches and extracts text content from a news URL.
    """
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Heuristic to find main content
        # 1. Look for <article>
        article = soup.find('article')
        
        # 2. If no article, look for common main content divs
        if not article:
            for cls in ['main-content', 'story-content', 'article-body', 'post-content']:
                article = soup.find(class_=re.compile(cls))
                if article: break
        
        # 3. Fallback to extracting all paragraphs if reasonable count
        if not article:
            paras = soup.find_all('p')
            # Filter distinct content paragraphs (simple heuristic: length > 50 chars)
            content_paras = [p.get_text() for p in paras if len(p.get_text()) > 50]
            text = "\n\n".join(content_paras)
        else:
             # Extract text from found container
             text = ""
             for p in article.find_all(['p', 'h2', 'h3']):
                 text += p.get_text() + "\n\n"

        return {"content": text if len(text) > 100 else "Could not extract article content automatically. Please visit the link."}

    except Exception as e:
        logger.error(f"Error extracting content from {url}: {e}")
        return {"content": f"Error loading article: {str(e)}"}


def get_calendar(ticker: str):
    """
    Fetches upcoming earnings and dividend events.
    """
    try:
        t = yf.Ticker(ticker)
        
        # Calendar returns a dict with keys like 'Dividend Date', 'Earnings Date', etc.
        cal = t.calendar
        
        events = []
        
        if cal:
            # Handle difference in return structure (sometimes dataframe, sometimes dict)
            # Recent yfinance returns simple dict
            
            # Earnings
            earnings_date = cal.get('Earnings Date')
            if earnings_date:
                # specific handling if it's a list
                if isinstance(earnings_date, list):
                     earnings_date = earnings_date[0]
                events.append({
                    "event": "Earnings Release",
                    "date": str(earnings_date).split(" ")[0]
                })

            # Dividends
            div_date = cal.get('Dividend Date')
            ex_div = cal.get('Ex-Dividend Date')
            
            if div_date:
                events.append({
                    "event": "Dividend Date",
                    "date": str(div_date).split(" ")[0]
                })
            if ex_div:
                events.append({
                    "event": "Ex-Dividend Date",
                    "date": str(ex_div).split(" ")[0]
                })

        return events
    except Exception as e:
        logger.error(f"Error fetching calendar for {ticker}: {e}")
        return []

def get_dividend_history(ticker: str, start_year: int = 2010):
    """
    Fetches historical dividend data.
    """
    try:
        t = yf.Ticker(ticker)
        dividends = t.dividends
        
        if dividends is None or dividends.empty:
            return None
            
        # Filter by date
        start_date = f"{start_year}-01-01"
        dividends = dividends[dividends.index >= start_date]
        
        if dividends.empty:
            return None
            
        # Group by year for annual growth
        annual_div = dividends.resample('Y').sum()
        
        # Calculate Growth
        growth = annual_div.pct_change().dropna()
        
        div_data = []
        for date, val in dividends.items():
             div_data.append({
                 "date": str(date).split(" ")[0],
                 "value": val
             })
             
        annual_data = []
        for date, val in annual_div.items():
            annual_data.append({
                "year": date.year,
                "value": val,
                "growth": growth.get(date, 0)
            })
            
        return {
            "history": div_data,
            "annual": annual_data,
            "current_yield": t.info.get('dividendYield', 0),
            "payout_ratio": t.info.get('payoutRatio', 0)
        }
    except Exception as e:
        logger.error(f"Error fetching dividends for {ticker}: {e}")
        return None

def search_tickers(query: str):
    """
    Searches for tickers using Yahoo Finance Autocomplete API.
    """
    try:
        url = "https://query2.finance.yahoo.com/v1/finance/search"
        params = {
            "q": query,
            "quotesCount": 10,
            "newsCount": 0,
            "enableFuzzyQuery": "true",
            "enableCb": "false"
        }
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        response = requests.get(url, params=params, headers=headers, timeout=5)
        data = response.json()
        
        quotes = data.get('quotes', [])
        
        results = []
        for quote in quotes:
            # Filter out non-equity if desired, currently keeping most valid types
            if 'symbol' not in quote: continue
            
            results.append({
                "symbol": quote['symbol'],
                "shortname": quote.get('shortname', quote['symbol']),
                "longname": quote.get('longname', ''),
                "exchange": quote.get('exchange', ''),
                "typeDisp": quote.get('typeDisp', '')
            })
            
        return results
    except Exception as e:
        logger.error(f"Error searching tickers for {query}: {e}")
        return []

def get_financials(ticker: str):
    """
    Fetches financial data needed for Valuation (DCF).
    """
    try:
        t = yf.Ticker(ticker)
        info = t.info
        
        # 1. Free Cash Flow
        # yfinance often returns cashflow as a DataFrame in t.cashflow
        fcf = None
        try:
             cf = t.cashflow
             if cf is not None and not cf.empty:
                 # Look for 'Free Cash Flow' or calculate 'Total Cash From Operating Activities' - 'Capital Expenditures'
                 # Note: yfinance rows are localized, usually "Free Cash Flow" exists in recent versions
                 if "Free Cash Flow" in cf.index:
                     fcf = cf.loc["Free Cash Flow"].iloc[0] # Most recent
                 elif "Total Cash From Operating Activities" in cf.index and "Capital Expenditures" in cf.index:
                     fcf = cf.loc["Total Cash From Operating Activities"].iloc[0] + cf.loc["Capital Expenditures"].iloc[0] # CapEx is usually negative
        except Exception:
            pass
            
        # Fallback to info provided FCF if available (often not reliable/present)
        if fcf is None:
             fcf = info.get("freeCashflow")
             
        # 2. Shares Outstanding
        shares = info.get("sharesOutstanding")
        
        # 3. Beta
        beta = info.get("beta")
        
        # 4. WACC components (Simplified)
        # We need Risk Free Rate (assume 4%), Market Risk Premium (assume 5%)
        # Cost of Equity = Rf + Beta * (Rm - Rf)
        # Cost of Debt = (Interest Expense / Total Debt) * (1 - Tax Rate) -> Hard to get accurately automatically
        # For MVP, we will let user edit the Discount Rate, but calculate a suggested one mostly based on Equity
        
        risk_free = 0.04
        market_premium = 0.05
        suggested_discount_rate = 0.10 # Default
        
        if beta:
             suggested_discount_rate = risk_free + beta * market_premium
             
        return {
            "fcf": fcf,
            "shares_outstanding": shares,
            "beta": beta,
            "price": info.get("currentPrice", info.get("regularMarketPreviousClose")),
            "suggested_discount_rate": suggested_discount_rate,
            "currency": info.get("currency", "USD")
        }
    except Exception as e:
        logger.error(f"Error fetching financials for {ticker}: {e}")
        return None

def calculate_financial_freedom(data, monthly_income_goal):
    """
    Calculates the number of shares and investment required to generate a target monthly income.
    Assumes dividend yield based on the last year's payout.
    """
    try:
        # Get last price
        if "Close" in data.columns:
            current_price = data["Close"].iloc[-1]
        elif "Adj Close" in data.columns:
            current_price = data["Adj Close"].iloc[-1]
        else:
            return None

        # Calculate approximate dividend yield (using a robust heuristic if actual div data is missing)
        # Ideally, we should fetch actual dividends. 
        # For this MVP, we will assume a "Yield" based on standard sector averages if not found, 
        # or calculate it if the dataframe contains 'Dividends' column.
        
        annual_yield = 0.0
        
        if "Dividends" in data.columns:
            # Sum last 12 months of dividends
            last_year = data.index.max() - pd.DateOffset(years=1)
            dividends_12m = data[data.index > last_year]["Dividends"].sum()
            if current_price > 0:
                annual_yield = dividends_12m / current_price
        
        # Fallback if no dividends found (or it's 0) - maybe user selected a non-paying stock
        # We return the data assuming calculated yield. If 0, frontend should warn.
        
        estimated_annual_income_needed = monthly_income_goal * 12
        
        if annual_yield > 0:
            investment_needed = estimated_annual_income_needed / annual_yield
            shares_needed = int(investment_needed / current_price)
        else:
            investment_needed = 0
            shares_needed = 0

        return {
            "current_price": current_price,
            "annual_yield": annual_yield,
            "shares_needed": shares_needed,
            "investment_needed": investment_needed,
            "monthly_income_goal": monthly_income_goal
        }
    except Exception as e:
        logger.error(f"Error calculating financial freedom: {e}")
        return None

def get_jse_peers(ticker):
    """
    Returns a list of valid JSE competitor tickers based on the input ticker's sector.
    This is a simplified lookup for the MVP.
    """
    # Normalize ticker
    ticker = ticker.upper().strip()
    
    # Simple Sector Map for common JSE stocks
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
    found_peers = []
    
    for sector, members in sector_map.items():
        if ticker in members:
            # Return all members except the ticker itself, limit to top 3
            peers = [m for m in members if m != ticker]
            found_peers = peers[:3]
            break
            
    # Default fallback if not found in map (Generic Top 40)
    if not found_peers:
        # Avoid self-reference in fallback
        defaults = ["STX40.JO", "NPN.JO", "SBK.JO"]
        found_peers = [d for d in defaults if d != ticker]
        
    return found_peers

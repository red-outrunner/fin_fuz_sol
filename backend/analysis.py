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

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

def download_data(ticker: str, start_date: str, end_date: str):
    """Downloads data from yfinance."""
    try:
        data = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=False)
        if data is None or data.empty:
            return None
        return data
    except Exception as e:
        logger.error(f"Error downloading data for {ticker}: {e}")
        return None

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
        
        return {
            "monthly_ret": monthly_ret,
            "pivot": pivot,
            "df": df
        }
    except Exception as e:
        logger.error(f"Error processing data: {e}")
        return None

def calculate_summary_stats(monthly_ret: pd.Series, pivot: pd.DataFrame):
    """Calculates summary statistics."""
    try:
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

        # 5. Wealth Index (Growth of $10,000)
        wealth_index = 10000 * (1 + monthly_ret).cumprod()
        # Prepend starting value
        wealth_data = [{"date": str(monthly_ret.index[0] - pd.DateOffset(months=1)).split(" ")[0], "value": 10000}]
        wealth_data.extend([{"date": str(d).split(" ")[0], "value": v} for d, v in wealth_index.items()])

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
            "wealth_index": wealth_data
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


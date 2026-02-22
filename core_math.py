import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import IsolationForest

class DataQualityError(Exception):
    """Exception raised when data quality does not meet institutional standards."""
    pass

def validate_and_clean_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Validates data right after yfinance fetches.
    Raises DataQualityError if missing >5% of expected trading days.
    Uses forward-fill to impute up to 5 days.
    """
    if data is None or data.empty:
        raise DataQualityError("No data fetched.")

    # Forward fill isolated missing daily data (limit to 5 days)
    data = data.ffill(limit=5)

    # Check for missing percentage of the remaining NaNs
    missing_pct = data.isna().sum().max() / len(data)
    if missing_pct > 0.05:
        raise DataQualityError(f"Data quality failure: Missing >5% of expected trading days. Percent missing: {missing_pct:.2%}")

    # Drop any remaining NaNs after ffill and validate again
    data = data.dropna()
    if data.empty:
        raise DataQualityError("Data consists entirely of NaNs after cleaning.")
        
    # Check for excessive gaps in time (e.g., from long trading halts or dropped NaN blocks)
    # We expect daily data, so a gap of 10+ calendar days is irregular and dangerous for pct_change
    time_deltas = data.index.to_series().diff()
    if not time_deltas.empty and pd.notna(time_deltas.max()):
        max_gap_days = time_deltas.max().days
        if max_gap_days > 10:
            raise DataQualityError(f"Data quality failure: Time gap of {max_gap_days} days detected. Data is corrupted by trading halts or excessive missing values.")
            
    return data

def apply_outlier_filtering(monthly_ret: pd.Series) -> pd.Series:
    """Uses Winsorization (1st and 99th percentiles) to cap severe outliers strictly for plotting/visual smoothing."""
    if len(monthly_ret) < 12:
        return monthly_ret
        
    p01 = np.percentile(monthly_ret, 1)
    p99 = np.percentile(monthly_ret, 99)
    
    clean_series = monthly_ret.copy()
    clean_series = np.clip(clean_series, p01, p99)
    
    return clean_series

def get_risk_free_rate(ticker: str) -> float:
    """
    Dynamic risk-free rates estimation based on ticker suffix.
    Uses South African metrics for .JO (approx 10% currently), etc.
    """
    ticker_upper = ticker.upper()
    if ticker_upper.endswith('.JO'):
         return 0.10  # SA 10Y Yield approx 10%
    elif ticker_upper.endswith('.AX'):
         return 0.04  # Aus 10Y Yield approx 4%
    elif ticker_upper.endswith('.L'):
         return 0.04  # UK
    elif ticker_upper.endswith('.DE'):
         return 0.025 # German 
    elif ticker_upper.endswith('.SS') or ticker_upper.endswith('.HK'):
         return 0.025 # China/HK
    else:
         return 0.042 # US 10Y approx 4.2%

def get_equity_risk_premium(ticker: str) -> float:
    """
    Dynamic Equity Risk Premium (ERP) mapping based on country suffix.
    """
    ticker_upper = ticker.upper()
    if ticker_upper.endswith('.JO'):
         return 0.075  # South Africa ~ 7.5%
    elif ticker_upper.endswith('.AX'):
         return 0.05   # Australia ~ 5%
    elif ticker_upper.endswith('.L'):
         return 0.055  # UK ~ 5.5%
    elif ticker_upper.endswith('.DE'):
         return 0.055  # Germany ~ 5.5%
    elif ticker_upper.endswith('.SS') or ticker_upper.endswith('.HK'):
         return 0.06   # China/HK ~ 6%
    else:
         return 0.045  # Default Global/US ~ 4.5%

def calculate_wacc(ticker: str, beta: float) -> float:
    """
    Multi-Factor Discount Rate calculation.
    WACC = Rf + Beta * ERP(Equity Risk Premium) + Size Premium
    """
    rf = get_risk_free_rate(ticker)
    erp = get_equity_risk_premium(ticker)
    size_premium = 0.01 
    beta_val = beta if (beta is not None and beta > 0) else 1.0
    return rf + (beta_val * erp) + size_premium

def calculate_cagr(monthly_ret: pd.Series) -> float:
    if monthly_ret.empty: return 0.0
    total_return = (1 + monthly_ret).prod() - 1
    n_years = len(monthly_ret) / 12
    return (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0.0

def calculate_volatility(monthly_ret: pd.Series) -> float:
    if monthly_ret.empty: return 0.0
    return monthly_ret.std() * np.sqrt(12)

def calculate_sharpe(cagr: float, volatility: float, risk_free_rate: float) -> float:
    if volatility == 0: return 0.0
    return (cagr - risk_free_rate) / volatility

def calculate_sortino(monthly_ret: pd.Series, cagr: float, risk_free_rate: float) -> float:
    if monthly_ret.empty: return 0.0
    
    # Calculate the target return (monthly risk-free rate)
    target_monthly = risk_free_rate / 12.0
    
    # Identify deviations below the target return
    downside_diff = np.minimum(monthly_ret - target_monthly, 0)
    
    # Square the deviations, average them, and take the square root
    downside_dev_monthly = np.sqrt(np.mean(downside_diff ** 2))
    
    # Annualize the downside deviation
    downside_dev = downside_dev_monthly * np.sqrt(12)
    
    return (cagr - risk_free_rate) / downside_dev if downside_dev > 0 else 0.0

def calculate_max_drawdown(monthly_ret: pd.Series) -> float:
    if monthly_ret.empty: return 0.0
    cumulative = (1 + monthly_ret).cumprod()
    peak = cumulative.expanding(min_periods=1).max()
    drawdowns = (cumulative / peak) - 1
    return drawdowns.min()

def calculate_dca(monthly_ret: pd.Series, monthly_contribution: float) -> dict:
    if monthly_ret.empty: return None
    total_invested = 0
    current_value = 0
    data_points = []
    
    for date, ret in monthly_ret.items():
        total_invested += monthly_contribution
        current_value += monthly_contribution
        current_value *= (1 + ret)
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

def run_monte_carlo(monthly_ret: pd.Series, years: int = 10, n_sims: int = 1000, method: str = 'gbm', dividend_yield: float = 0.0) -> list:
    """
    Runs Monte Carlo simulation using Geometric Brownian Motion (GBM) or Historical Bootstrapping.
    Incorporates dividend reinvestment into total return.
    Returns: 10th, 50th, 90th percentile paths.
    """
    if monthly_ret.empty or len(monthly_ret) < 12: return None
    months = years * 12
    last_val = 10000 
    
    # Adjust monthly yield
    monthly_yield = dividend_yield / 12.0
    
    if method == 'gbm':
        # GBM: dS = mu*S*dt + sigma*S*dW
        mu = monthly_ret.mean() + monthly_yield
        sigma = monthly_ret.std()
        
        sim_paths = np.zeros((months + 1, n_sims))
        sim_paths[0] = last_val
        
        for t in range(1, months + 1):
            # GBM ensuring price cannot go below zero mathematically since exp() is positive
            drift = (mu - 0.5 * sigma**2)
            shock = sigma * np.random.normal(0, 1, n_sims)
            sim_paths[t] = sim_paths[t-1] * np.exp(drift + shock)
        
    elif method == 'bootstrap':
        # Historical Bootstrapping (sampling with replacement) to naturally capture fat tails
        sim_paths = np.zeros((months + 1, n_sims))
        sim_paths[0] = last_val
        ret_vals = monthly_ret.values
        
        for t in range(1, months + 1):
            sampled_rets = np.random.choice(ret_vals, n_sims, replace=True) + monthly_yield
            # Ensure price >= 0
            growth = np.maximum(sampled_rets + 1, 0)
            sim_paths[t] = sim_paths[t-1] * growth
    else:
        return None

    p10 = np.percentile(sim_paths, 10, axis=1)
    p50 = np.percentile(sim_paths, 50, axis=1)
    p90 = np.percentile(sim_paths, 90, axis=1)
    
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

def run_ml_clusters(monthly_ret: pd.Series) -> dict:
    if len(monthly_ret) < 24: return None
    data = monthly_ret.values
    X = np.array([data[i-12:i] for i in range(12, len(data))])
    if len(X) == 0: return None
    
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    gmm = GaussianMixture(n_components=3, random_state=42)
    labels = gmm.fit_predict(X_pca)
    
    iso = IsolationForest(contamination=0.05, random_state=42)
    anomalies = iso.fit_predict(X_pca)
    
    return {
        "pca_components": X_pca.tolist(),
        "clusters": labels.tolist(),
        "anomalies": anomalies.tolist()
    }

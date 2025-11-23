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
        
        pivot = pivot.reindex(sorted(pivot.columns), axis=1)

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

        return {
            "overall_avg": overall_avg,
            "month_avg": month_avg.to_dict(),
            "month_median": month_median.to_dict(),
            "best_month": {"index": int(best_month), "name": months_map.get(best_month), "value": month_avg[best_month]},
            "worst_month": {"index": int(worst_month), "name": months_map.get(worst_month), "value": month_avg[worst_month]},
            "std_dev": pivot.std().to_dict(),
            "positive_rate": ((pivot > 0).sum() / pivot.count()).to_dict()
        }
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
            "f_stat": f_stat,
            "p_value": p_val,
            "significant": p_val < 0.05
        }
    except Exception as e:
        logger.error(f"Error in ANOVA test: {e}")
        return {"error": str(e)}

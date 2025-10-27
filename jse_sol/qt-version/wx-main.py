# jse_analyzer_wx.py
# Global Index Monthly Return Analyzer (wxPython Version)
# Enhanced ML analysis with PCA, GMM, Isolation Forest, cluster visualization,
# plain-English summary, upcoming month forecast, and comprehensive logging.

import sys
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import matplotlib
import os
import pickle
import json
from scipy import stats
import threading # Standard Python threading
import re
import concurrent.futures
import logging
import time
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import IsolationForest
from statsmodels.tsa.arima.model import ARIMA
import warnings
import matplotlib.patches as patches

# wxPython Imports
import wx
import wx.adv # For SpinCtrlDouble

# Matplotlib-wxPython Integration
matplotlib.use('WXAgg')
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
from matplotlib.figure import Figure

# Try importing pmdarima, with fallback if it fails
try:
    from pmdarima import auto_arima
    PMDARIMA_AVAILABLE = True
except (ImportError, ValueError) as e:
    PMDARIMA_AVAILABLE = False
    warnings.warn(f"Failed to import pmdarima ({str(e)}). Using fixed-order ARIMA for forecasting.")

warnings.filterwarnings('ignore')

# --- Setup Logging ---
def setup_logging():
    """Configures the root logger to write to a file."""
    log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('jse_analyzer_wx.log', mode='w')
    file_handler.setFormatter(log_formatter)
    
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    if logger.hasHandlers():
        logger.handlers.clear()
        
    logger.addHandler(file_handler)
    return logger

# --- wxPython Worker for Non-Blocking Analysis ---
class DataAnalysisWorker(threading.Thread):
    """Handles data fetching and heavy processing in a separate thread."""
    
    def __init__(self, frame, settings, logger, cache_dir, version):
        super().__init__(daemon=True)
        self.frame = frame # Main window instance for wx.CallAfter
        self.settings = settings
        self.logger = logger
        self.cache_dir = cache_dir
        self.VERSION = version
        self._is_running = True
        self.task_type = None
        self.primary_data = None

    def stop(self):
        """Allows external stopping of the worker."""
        self._is_running = False

    def set_task(self, task_type, settings, primary_data=None):
        """Sets the task and relevant data/settings for the next run."""
        self.task_type = task_type
        self.settings = settings
        self.primary_data = primary_data

    def run(self):
        """The main execution method, dispatching based on task_type."""
        try:
            if self.task_type == 'primary':
                self.run_primary_analysis()
            elif self.task_type == 'comparison':
                # primary_data is expected to be passed via set_task
                self.run_comparison_analysis(self.primary_data['comparison_tickers'], self.primary_data['primary_data_for_compare'])
            elif self.task_type == 'ml':
                # primary_data is expected to be passed via set_task
                self.run_ml_analysis(self.primary_data['monthly_ret'], self.primary_data['pivot'])
            else:
                wx.CallAfter(self.frame.handle_error, "Worker started without a valid task type.")
        
        except Exception as e:
            self.logger.exception("An error occurred during data analysis.")
            wx.CallAfter(self.frame.handle_error, f"Analysis failed: {str(e)}")
        finally:
            wx.CallAfter(self.frame.handle_finished)
            
    # --- Helper functions (identical to original) ---
    
    def get_price_column(self, data):
        if "Adj Close" in data.columns: return "Adj Close"
        elif "Close" in data.columns: return "Close"
        else: raise ValueError("Neither 'Adj Close' nor 'Close' column found in data")

    def get_cache_filename(self, ticker, start_year, end_date):
        return f"{self.cache_dir}/{ticker.replace('^', '').replace('.', '_')}_{start_year}_{end_date[:4]}_v{self.VERSION.replace('.', '_')}.pkl"

    def load_cached_data(self, ticker, start_year, end_date):
        cache_file = self.get_cache_filename(ticker, start_year, end_date)
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    data = pickle.load(f)
                    self.logger.info(f"Loaded cached data for {ticker} from {cache_file}")
                    return data
            except Exception as e:
                self.logger.error(f"Error loading cache for {ticker}: {e}")
        return None

    def save_cached_data(self, ticker, start_year, end_date, data):
        cache_file = self.get_cache_filename(ticker, start_year, end_date)
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f)
            self.logger.info(f"Saved data for {ticker} to cache: {cache_file}")
        except Exception as e:
            self.logger.error(f"Error saving cache for {ticker}: {e}")
            
    def process_ticker_data(self, data, ticker):
        if data is None or data.empty:
            raise ValueError(f"No data returned for ticker {ticker}.")

        price_col = self.get_price_column(data)
        monthly = data[price_col].resample('ME').last()
        monthly_ret = monthly.pct_change().dropna()

        if isinstance(monthly_ret, pd.DataFrame):
            if monthly_ret.empty:
                raise ValueError(f"Monthly returns calculation for {ticker} resulted in an empty DataFrame.")
            monthly_ret = monthly_ret.iloc[:, 0]

        if monthly_ret.empty:
            raise ValueError(f"Monthly returns for {ticker} are empty after processing.")

        monthly_ret.name = 'ret'
        df = monthly_ret.to_frame()
        df['year'] = df.index.year
        df['month'] = df.index.month
        pivot = df.pivot_table(index='year', columns='month', values='ret')
        month_avg = pivot.mean().sort_index()
        month_median = pivot.median().sort_index()
        overall_avg = monthly_ret.mean()
        
        return {
            'monthly_ret': monthly_ret, 'pivot': pivot, 'month_avg': month_avg,
            'month_median': month_median, 'overall_avg': overall_avg, 'data': data
        }

    # --- Analysis routines (using wx.CallAfter for signals) ---

    def run_primary_analysis(self):
        """Primary analysis routine for the main ticker."""
        ticker = self.settings['ticker']
        start_year = self.settings['start_year']
        end_date = self.settings['end_date']
        
        wx.CallAfter(self.frame.SetStatusText, f"📥 Downloading or loading cache for {ticker}...")
        
        data = self.load_cached_data(ticker, start_year, end_date)
        
        if data is None:
            max_retries = 3
            for attempt in range(max_retries):
                if not self._is_running: return
                try:
                    wx.CallAfter(self.frame.SetStatusText, f"Downloading {ticker} (attempt {attempt+1}/{max_retries})...")
                    data = yf.download(
                        ticker, start=f"{start_year}-01-01", end=end_date,
                        progress=False, auto_adjust=False
                    )
                    if not data.empty:
                        self.save_cached_data(ticker, start_year, end_date, data)
                        break
                except Exception as e:
                    if attempt == max_retries - 1: raise
                    time.sleep(2)
            
        if data is None or data.empty:
            raise ValueError(f"No data returned from Yahoo Finance for {ticker}.")

        wx.CallAfter(self.frame.SetStatusText, "⚙️ Processing data...")
        results = self.process_ticker_data(data, ticker)
        
        wx.CallAfter(self.frame.handle_primary_data_ready, results)

    def run_comparison_analysis(self, comparison_tickers, primary_data):
        """Comparison analysis routine for comparison tickers."""
        wx.CallAfter(self.frame.SetStatusText, "Starting parallel comparison analysis...")
        self.logger.info("Starting parallel analysis for comparison tickers.")
        
        comparison_data_results = {}
        if primary_data is not None:
             comparison_data_results[self.settings['ticker']] = primary_data

        tickers_to_fetch = [t for t in comparison_tickers if t != self.settings['ticker']]
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = {
                executor.submit(self._fetch_and_process_single_ticker, ticker): ticker
                for ticker in tickers_to_fetch
            }
            
            for future in concurrent.futures.as_completed(futures):
                if not self._is_running: break
                ticker = futures[future]
                try:
                    result = future.result()
                    if result is not None:
                        comparison_data_results[ticker] = result
                    else:
                        self.logger.warning(f"No data processed for comparison ticker {ticker}.")
                except Exception as e:
                    self.logger.error(f"Error processing comparison ticker {ticker}: {str(e)}")
                    wx.CallAfter(self.frame.SetStatusText, f"⚠️ Error processing {ticker}: {str(e)}")
        
        wx.CallAfter(self.frame.handle_comparison_data_ready, comparison_data_results)

    def _fetch_and_process_single_ticker(self, ticker):
        """Helper to fetch and process a single comparison ticker."""
        start_year = self.settings['start_year']
        end_date = self.settings['end_date']
        data = self.load_cached_data(ticker, start_year, end_date)
        if data is None:
            data = yf.download(
                ticker, start=f"{start_year}-01-01", end=end_date,
                progress=False, auto_adjust=False
            )
            if data is not None and not data.empty:
                self.save_cached_data(ticker, start_year, end_date, data)
        if data is None or data.empty:
            return None
        return self.process_ticker_data(data, ticker)
        
    def run_ml_analysis(self, monthly_ret, pivot):
        """Runs the ML analysis (KMeans, PCA, ARIMA) in the worker thread."""
        wx.CallAfter(self.frame.SetStatusText, "🤖 Starting ML Analysis...")
        
        n_clusters = self.settings.get('ml_n_clusters', 3)
        contamination = self.settings.get('ml_contamination', 0.1)
        arima_order_p = self.settings.get('ml_arima_p', 5)
        arima_order_d = self.settings.get('ml_arima_d', 1)
        arima_order_q = self.settings.get('ml_arima_q', 0)
        
        X = pivot.fillna(0).values.T
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        wx.CallAfter(self.frame.SetStatusText, "Running PCA...")
        pca = PCA(n_components=2)
        features_reduced = pca.fit_transform(X_scaled)
        
        wx.CallAfter(self.frame.SetStatusText, f"Running KMeans Clustering (k={n_clusters})...")
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
        clusters = kmeans.fit_predict(X_scaled)
        
        wx.CallAfter(self.frame.SetStatusText, f"Running Isolation Forest Anomaly Detection (Contamination={contamination})...")
        iso_forest = IsolationForest(contamination=contamination, random_state=42)
        anom_pred = iso_forest.fit_predict(X_scaled)
        
        wx.CallAfter(self.frame.SetStatusText, "Running Time Series Forecast...")
        
        arima_order = None
        if PMDARIMA_AVAILABLE:
            wx.CallAfter(self.frame.SetStatusText, "Using AutoARIMA for optimal order selection...")
            model = auto_arima(
                monthly_ret, seasonal=False, stepwise=True,
                suppress_warnings=True, error_action='ignore', max_p=5, max_q=5
            )
            arima_order = model.order
        else:
            order = (arima_order_p, arima_order_d, arima_order_q)
            wx.CallAfter(self.frame.SetStatusText, f"Using fixed ARIMA{order}...")
            model = ARIMA(monthly_ret, order=order)
            model = model.fit()
            arima_order = order

        forecast_steps = 1
        if PMDARIMA_AVAILABLE:
            forecast_result = model.predict(n_periods=forecast_steps)
            forecast_mean = forecast_result.iloc[0] if isinstance(forecast_result, pd.Series) else forecast_result[0]
        else:
            forecast = model.get_forecast(steps=forecast_steps)
            forecast_mean = forecast.predicted_mean.iloc[0]

        wx.CallAfter(self.frame.SetStatusText, "Generating ML results...")
        
        ml_results = {
            'features_reduced': features_reduced, 'clusters': clusters, 'anom_pred': anom_pred,
            'pca_explained_variance': pca.explained_variance_ratio_,
            'forecast_mean': forecast_mean,
            'forecast_date': monthly_ret.index[-1] + timedelta(days=31),
            'n_clusters': n_clusters, 'contamination': contamination, 'arima_order': arima_order
        }
        
        wx.CallAfter(self.frame.handle_ml_data_ready, ml_results)


# --- Main Application Frame (wxPython) ---
class JSEAnalyzer(wx.Frame):
    """A wxPython application for analyzing monthly returns of global financial indices."""
    VERSION = "2.9.4 (wxPython Port)"

    def __init__(self):
        super().__init__(None, title=f"📊 Global Index Monthly Return Analyzer (v{self.VERSION})", size=(1400, 900))
        self.logger = setup_logging()
        self.logger.info(f"--- Application Start: Global Index Analyzer v{self.VERSION} (wxPython) ---")

        # Data and State
        self.data = None
        self.monthly_ret = None
        self.pivot = None
        self.month_avg = None
        self.month_median = None
        self.overall_avg = 0
        self.comparison_data = {}
        self.comparison_tickers = []
        self.months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        self.ml_results = None
        self.stat_results = None
        self.current_ticker = ""
        self.worker_thread = None

        self.CreateStatusBar()
        self.SetStatusText("✨ Initializing Application...")

        self.cache_dir = f"cache_v{self.VERSION.replace('.', '_').replace(' ', '_')}"
        os.makedirs(self.cache_dir, exist_ok=True)
        self.logger.info(f"Cache directory set to: {self.cache_dir}")

        self.setup_ui()
        self.logger.info("Application initialized successfully.")
        
        self.toggle_dark_mode(self.dark_mode_check.IsChecked())
        self.SetStatusText("✨ Ready - Select an index and click Analyze")
        self.Center()
        self.Show()

    # --- UI Setup ---
    def setup_ui(self):
        """Set up the main wxPython GUI components."""
        
        main_panel = wx.Panel(self)
        main_sizer = wx.BoxSizer(wx.VERTICAL)

        self.ticker_options = {
            "🇿🇦 JSE All Share (^J203.JO)": "^J203.JO", "🇺🇸 S&P 500 (^GSPC)": "^GSPC",
            "🇬🇧 FTSE 100 (^FTSE)": "^FTSE", "🇩🇪 DAX (^GDAXI)": "^GDAXI",
            "🇯🇵 Nikkei 225 (^N225)": "^N225", "🇭🇰 Hang Seng (^HSI)": "^HSI",
            "🇨🇳 Shanghai Composite (000001.SS)": "000001.SS", "🌍 MSCI World (^MXWO)": "^MXWO",
            "🌍 MSCI Emerging Markets (^MXEF)": "^MXEF"
        }
        
        # --- Header and Config ---
        header_sizer = wx.BoxSizer(wx.HORIZONTAL)
        title_label = wx.StaticText(main_panel, label=f"📈 Global Index Monthly Return Analyzer (v{self.VERSION})")
        title_font = wx.Font(12, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD)
        title_label.SetFont(title_font)
        
        self.dark_mode_check = wx.CheckBox(main_panel, label="🌙 Dark Mode")
        
        header_sizer.Add(title_label, 1, wx.ALIGN_CENTER_VERTICAL | wx.LEFT, 10)
        header_sizer.AddStretchSpacer()
        header_sizer.Add(self.dark_mode_check, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 10)
        main_sizer.Add(header_sizer, 0, wx.EXPAND | wx.ALL, 5)

        # Configuration Panel (StaticBoxSizer)
        config_box = wx.StaticBox(main_panel, label="⚙️ Configuration Panel")
        config_sizer = wx.StaticBoxS


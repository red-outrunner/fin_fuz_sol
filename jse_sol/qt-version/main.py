# jse_analyzer_kivy.py
# Global Index Monthly Return Analyzer (Kivy Version)
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

# Kivy Imports
import kivy
kivy.require('2.1.0') # Example version, adjust as needed
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.popup import Popup
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.filechooser import FileChooserListView
from kivy.properties import (
    StringProperty, ObjectProperty, BooleanProperty,
    NumericProperty, ColorProperty, ListProperty
)
from kivy.clock import mainthread

# Matplotlib-Kivy Integration
# REQUIRES: kivy-garden install matplotlib
from kivy_garden.matplotlib.widgets import FigureCanvasKivyAgg

# Try importing pmdarima, with fallback if it fails
try:
    from pmdarima import auto_arima
    PMDARIMA_AVAILABLE = True
except (ImportError, ValueError) as e:
    PMDARIMA_AVAILABLE = False
    warnings.warn(f"Failed to import pmdarima ({str(e)}). Using fixed-order ARIMA for forecasting.")

warnings.filterwarnings('ignore')
# Matplotlib backend is set by the Kivy garden widget

# --- Setup Logging ---
def setup_logging():
    """Configures the root logger to write to a file."""
    log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('jse_analyzer_kivy.log', mode='w')
    file_handler.setFormatter(log_formatter)
    
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    if logger.hasHandlers():
        logger.handlers.clear()
        
    logger.addHandler(file_handler)
    return logger

# --- Kivy-compatible Worker for Non-Blocking Analysis ---
class DataAnalysisWorker:
    """Handles data fetching and heavy processing in a separate thread."""
    
    def __init__(self, settings, logger, cache_dir, version, callbacks, parent=None):
        self.settings = settings
        self.logger = logger
        self.cache_dir = cache_dir
        self.VERSION = version
        self._is_running = True
        self.task_type = None
        self.primary_data = None
        self.callbacks = callbacks # Dict of callback functions

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
                self.run_comparison_analysis(self.primary_data['comparison_tickers'], self.primary_data['primary_data_for_compare'])
            elif self.task_type == 'ml':
                self.run_ml_analysis(self.primary_data['monthly_ret'], self.primary_data['pivot'])
            else:
                self.callbacks['error']("Worker started without a valid task type.")
        except Exception as e:
            self.logger.exception(f"An error occurred during task {self.task_type}.")
            self.callbacks['error'](f"Analysis failed: {str(e)}")
        finally:
            self.callbacks['finished']()
            
    # ... (All other methods from DataAnalysisWorker remain IDENTICAL) ...
    # get_price_column, get_cache_filename, load_cached_data, save_cached_data,
    # process_ticker_data
            
    def get_price_column(self, data):
        """Determine the correct price column to use (Close or Adj Close)."""
        if "Adj Close" in data.columns:
            return "Adj Close"
        elif "Close" in data.columns:
            return "Close"
        else:
            raise ValueError("Neither 'Adj Close' nor 'Close' column found in data")

    def get_cache_filename(self, ticker, start_year, end_date):
        """Generate cache filename with versioning."""
        return f"{self.cache_dir}/{ticker.replace('^', '').replace('.', '_')}_{start_year}_{end_date[:4]}_v{self.VERSION.replace('.', '_')}.pkl"

    def load_cached_data(self, ticker, start_year, end_date):
        """Load data from cache if available."""
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
        """Cache downloaded data locally."""
        cache_file = self.get_cache_filename(ticker, start_year, end_date)
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f)
            self.logger.info(f"Saved data for {ticker} to cache: {cache_file}")
        except Exception as e:
            self.logger.error(f"Error saving cache for {ticker}: {e}")
            
    def process_ticker_data(self, data, ticker):
        """Calculates monthly returns, pivot tables, and summary statistics."""
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
            'monthly_ret': monthly_ret,
            'pivot': pivot,
            'month_avg': month_avg,
            'month_median': month_median,
            'overall_avg': overall_avg,
            'data': data # Original data for full export
        }


    def run_primary_analysis(self):
        """Primary analysis routine for the main ticker."""
        # Note: No 'try...except' here, it's handled in run()
        ticker = self.settings['ticker']
        start_year = self.settings['start_year']
        end_date = self.settings['end_date']
        
        self.callbacks['progress'](f"📥 Downloading or loading cache for {ticker}...")
        
        data = self.load_cached_data(ticker, start_year, end_date)
        
        if data is None:
            max_retries = 3
            for attempt in range(max_retries):
                if not self._is_running: return
                try:
                    self.callbacks['progress'](f"Downloading {ticker} (attempt {attempt+1}/{max_retries})...")
                    data = yf.download(
                        ticker,
                        start=f"{start_year}-01-01",
                        end=end_date,
                        progress=False,
                        auto_adjust=False
                    )
                    if not data.empty:
                        self.save_cached_data(ticker, start_year, end_date, data)
                        break
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise
                    time.sleep(2)
            
        if data is None or data.empty:
            raise ValueError(f"No data returned from Yahoo Finance for {ticker}.")

        self.callbacks['progress']("⚙️ Processing data...")
        results = self.process_ticker_data(data, ticker)
        
        self.callbacks['primary_data_ready'](results)


    def run_comparison_analysis(self, comparison_tickers, primary_data):
        """Comparison analysis routine for comparison tickers."""
        self.callbacks['progress']("Starting parallel comparison analysis...")
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
                        self.logger.info(f"Successfully processed comparison data for {ticker}.")
                    else:
                        self.logger.warning(f"No data processed for comparison ticker {ticker}.")
                except Exception as e:
                    self.logger.error(f"Error processing comparison ticker {ticker}: {str(e)}")
                    self.callbacks['progress'](f"⚠️ Error processing {ticker}: {str(e)}")
        
        self.callbacks['comparison_data_ready'](comparison_data_results)


    def _fetch_and_process_single_ticker(self, ticker):
        """Helper to fetch and process a single comparison ticker."""
        start_year = self.settings['start_year']
        end_date = self.settings['end_date']
        
        data = self.load_cached_data(ticker, start_year, end_date)
        
        if data is None:
            data = yf.download(
                ticker,
                start=f"{start_year}-01-01",
                end=end_date,
                progress=False,
                auto_adjust=False
            )
            if data is not None and not data.empty:
                self.save_cached_data(ticker, start_year, end_date, data)
        
        if data is None or data.empty:
            return None

        return self.process_ticker_data(data, ticker)
        
    def run_ml_analysis(self, monthly_ret, pivot):
        """Runs the ML analysis (KMeans, PCA, ARIMA) in the worker thread, using user settings."""
        self.callbacks['progress']("🤖 Starting ML Analysis...")
        
        n_clusters = self.settings.get('ml_n_clusters', 3)
        contamination = self.settings.get('ml_contamination', 0.1)
        arima_order_p = self.settings.get('ml_arima_p', 5)
        arima_order_d = self.settings.get('ml_arima_d', 1)
        arima_order_q = self.settings.get('ml_arima_q', 0)
        
        X = pivot.fillna(0).values.T 
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        self.callbacks['progress']("Running PCA...")
        pca = PCA(n_components=2)
        features_reduced = pca.fit_transform(X_scaled)
        
        self.callbacks['progress'](f"Running KMeans Clustering (k={n_clusters})...")
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
        clusters = kmeans.fit_predict(X_scaled)
        
        self.callbacks['progress'](f"Running Isolation Forest Anomaly Detection (Contamination={contamination})...")
        iso_forest = IsolationForest(contamination=contamination, random_state=42)
        anom_pred = iso_forest.fit_predict(X_scaled)
        
        self.callbacks['progress']("Running Time Series Forecast...")
        
        arima_order = None
        if PMDARIMA_AVAILABLE:
            self.callbacks['progress']("Using AutoARIMA for optimal order selection...")
            model = auto_arima(
                monthly_ret, seasonal=False, stepwise=True,
                suppress_warnings=True, error_action='ignore', max_p=5, max_q=5
            )
            arima_order = model.order
        else:
            order = (arima_order_p, arima_order_d, arima_order_q)
            self.callbacks['progress'](f"Using fixed ARIMA{order}...")
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

        self.callbacks['progress']("Generating ML results...")
        
        ml_results = {
            'features_reduced': features_reduced,
            'clusters': clusters,
            'anom_pred': anom_pred,
            'pca_explained_variance': pca.explained_variance_ratio_,
            'forecast_mean': forecast_mean,
            'forecast_date': monthly_ret.index[-1] + timedelta(days=31),
            'n_clusters': n_clusters,
            'contamination': contamination,
            'arima_order': arima_order
        }
        
        self.callbacks['ml_data_ready'](ml_results)


# --- Main Kivy Layout Class ---
class MainLayout(BoxLayout):
    """Holds all UI logic and state."""
    VERSION = "2.9.4 (Kivy Port)"
    
    # --- Kivy Properties for UI Binding ---
    status_bar_text = StringProperty("✨ Ready - Select an index and click Analyze")
    analysis_in_progress = BooleanProperty(False)
    
    # Primary Data
    primary_data = ObjectProperty(None, allownone=True) # Holds dict from process_ticker_data
    comparison_data = ObjectProperty({}, allownone=True)
    ml_results = ObjectProperty(None, allownone=True)
    stat_results = ObjectProperty(None, allownone=True)
    
    # UI State Properties
    bar_metric = StringProperty("Mean")
    show_benchmark = BooleanProperty(True)
    custom_ticker_enabled = BooleanProperty(False)
    current_ticker = StringProperty("")
    
    # Ticker Mapping
    ticker_options = {
        "🇿🇦 JSE All Share (^J203.JO)": "^J203.JO",
        "🇺🇸 S&P 500 (^GSPC)": "^GSPC",
        "🇬🇧 FTSE 100 (^FTSE)": "^FTSE",
        "🇩🇪 DAX (^GDAXI)": "^GDAXI",
        "🇯🇵 Nikkei 225 (^N225)": "^N225",
        "🇭🇰 Hang Seng (^HSI)": "^HSI",
        "🇨🇳 Shanghai Composite (000001.SS)": "000001.SS",
        "🌍 MSCI World (^MXWO)": "^MXWO",
        "🌍 MSCI Emerging Markets (^MXEF)": "^MXEF"
    }
    
    # Date Range Mapping
    date_range_options = ["Custom", "Last 5 Years", "Last 10 Years", "Last 20 Years", "All Data"]
    
    # Comparison Tickers
    comparison_tickers_list = ListProperty([]) # List of ticker strings
    
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    # Dark Mode Colors
    bg_color = ColorProperty([1, 1, 1, 1])
    fg_color = ColorProperty([0.17, 0.24, 0.31, 1]) # Dark text
    card_color = ColorProperty([0.95, 0.95, 0.95, 1])
    text_input_bg = ColorProperty([1, 1, 1, 1])
    
    # Plot Colors
    plot_fig_bg = ColorProperty([1, 1, 1, 1])
    plot_ax_bg = ColorProperty([0.97, 0.97, 0.97, 1])
    plot_fg = ColorProperty([0.17, 0.24, 0.31, 1])
    plot_grid_color = ColorProperty([0.8, 0.8, 0.8, 1])

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.logger = App.get_running_app().logger
        self.cache_dir = App.get_running_app().cache_dir
        self.analysis_worker = None
        self.worker_thread = None

    def toggle_dark_mode(self, active):
        """Toggle between light and dark mode."""
        self.logger.info(f"Toggling dark mode to {'ON' if active else 'OFF'}.")
        if active:
            self.bg_color = [0.17, 0.24, 0.31, 1] # Dark Blue/Gray
            self.fg_color = [0.93, 0.94, 0.94, 1] # Light Gray
            self.card_color = [0.2, 0.29, 0.37, 1]
            self.text_input_bg = [0.2, 0.29, 0.37, 1]
            # Plot
            self.plot_fig_bg = [0.2, 0.29, 0.37, 1]
            self.plot_ax_bg = [0.2, 0.29, 0.37, 1]
            self.plot_fg = [0.93, 0.94, 0.94, 1]
            self.plot_grid_color = [0.4, 0.4, 0.4, 1]
        else:
            self.bg_color = [1, 1, 1, 1]
            self.fg_color = [0.17, 0.24, 0.31, 1]
            self.card_color = [0.95, 0.95, 0.95, 1]
            self.text_input_bg = [1, 1, 1, 1]
            # Plot
            self.plot_fig_bg = [1, 1, 1, 1]
            self.plot_ax_bg = [0.97, 0.97, 0.97, 1]
            self.plot_fg = [0.17, 0.24, 0.31, 1]
            self.plot_grid_color = [0.8, 0.8, 0.8, 1]
        
        self.update_charts() # Redraw charts with new colors

    def on_date_range_change(self, text):
        """Update date inputs based on selected range."""
        self.logger.info(f"Date range changed to: {text}")
        current_year = datetime.now().year
        
        # Access UI elements by id
        start_year_input = self.ids.start_year_spin
        end_date_input = self.ids.end_date_entry
        
        is_custom = text == "Custom"
        start_year_input.disabled = not is_custom
        end_date_input.disabled = not is_custom
        
        if text == "Last 5 Years":
            start_year_input.text = str(current_year - 5)
            end_date_input.text = datetime.today().strftime("%Y-%m-%d")
        elif text == "Last 10 Years":
            start_year_input.text = str(current_year - 10)
            end_date_input.text = datetime.today().strftime("%Y-%m-%d")
        elif text == "Last 20 Years":
            start_year_input.text = str(current_year - 20)
            end_date_input.text = datetime.today().strftime("%Y-%m-%d")
        elif text == "All Data":
            start_year_input.text = "1990"
            end_date_input.text = datetime.today().strftime("%Y-%m-%d")

    def add_comparison_ticker(self):
        """Add a ticker to the comparison list."""
        ticker_name = self.ids.compare_spinner.text
        if not ticker_name:
            self.show_popup("Warning", "Please select a ticker to compare.")
            return

        selected_ticker = self.ticker_options.get(ticker_name)
        if selected_ticker is None:
            self.show_popup("Warning", "Invalid ticker selection.")
            return
            
        if selected_ticker not in self.comparison_tickers_list:
            self.comparison_tickers_list.append(selected_ticker)
            self.logger.info(f"Added '{ticker_name}' ({selected_ticker}) to comparison list.")
            self.status_bar_text = f"✅ Added {ticker_name} to comparison list."
            
            if self.primary_data:
                self.analyze_comparison_data()
        else:
            self.show_popup("Warning", "Ticker already added for comparison.")

    def get_current_settings(self):
        """Retrieves and validates current user inputs."""
        settings = {}
        try:
            settings['start_year'] = int(self.ids.start_year_spin.text)
            settings['end_date'] = self.ids.end_date_entry.text.strip()
            
            if self.ids.custom_ticker_check.active:
                ticker = self.ids.custom_ticker_entry.text.strip()
                settings['ticker'] = ticker
            else:
                ticker_name = self.ids.ticker_spinner.text
                settings['ticker'] = self.ticker_options.get(ticker_name, "^J203.JO")
            
            self.current_ticker = settings['ticker']
            settings['end_year'] = settings['end_date'][:4]
            
            settings['ml_n_clusters'] = int(self.ids.ml_n_clusters_spin.text)
            settings['ml_contamination'] = float(self.ids.ml_contamination_spin.text)
            settings['ml_arima_p'] = int(self.ids.ml_arima_p_spin.text)
            settings['ml_arima_d'] = int(self.ids.ml_arima_d_spin.text)
            settings['ml_arima_q'] = int(self.ids.ml_arima_q_spin.text)
            
            return settings
        except Exception as e:
             self.show_popup("Input Error", f"Input validation failed: {e}")
             return None

    def build_callbacks(self):
        """Builds the callback dict for the worker."""
        return {
            'finished': self.on_worker_finished,
            'error': self.on_worker_error,
            'progress': self.on_worker_progress,
            'primary_data_ready': self.on_primary_data_ready,
            'comparison_data_ready': self.on_comparison_data_ready,
            'ml_data_ready': self.on_ml_data_ready,
        }

    def analyze_data(self):
        """Initiates primary data analysis."""
        settings = self.get_current_settings()
        if settings is None: return
        
        if self.analysis_in_progress:
            self.show_popup("Busy", "Worker thread is currently running. Please wait.")
            return

        self.primary_data = None
        self.ml_results = None
        self.stat_results = None
        self.analysis_in_progress = True

        callbacks = self.build_callbacks()
        self.analysis_worker = DataAnalysisWorker(settings, self.logger, self.cache_dir, self.VERSION, callbacks)
        self.analysis_worker.set_task('primary', settings)
        
        self.worker_thread = threading.Thread(target=self.analysis_worker.run, daemon=True)
        self.worker_thread.start()
        self.status_bar_text = f"🔍 Analyzing {self.current_ticker}..."

    @mainthread
    def on_primary_data_ready(self, results):
        """Processes primary analysis results on the main thread."""
        self.primary_data = results
        self.update_charts(force_redraw=True)
        self.update_summary()
        self.status_bar_text = f"✅ Analysis complete for {self.current_ticker} ({self.primary_data['pivot'].index[0]}-{self.primary_data['pivot'].index[-1]})"
        
        if self.comparison_tickers_list:
            self.analyze_comparison_data()

    def analyze_comparison_data(self):
        """Initiates comparison analysis."""
        if not self.comparison_tickers_list: return
        
        settings = self.get_current_settings()
        if settings is None: return

        if self.analysis_in_progress:
            self.show_popup("Busy", "Worker thread is currently running. Please wait.")
            return
            
        primary_data_for_compare = {
            'month_avg': self.primary_data['month_avg'], 'pivot': self.primary_data['pivot'], 
            'monthly_ret': self.primary_data['monthly_ret'],
            'month_median': self.primary_data['month_median'], 
            'overall_avg': self.primary_data['overall_avg'], 'data': self.primary_data['data']
        }
        worker_data = {
            'comparison_tickers': self.comparison_tickers_list,
            'primary_data_for_compare': primary_data_for_compare
        }
        
        self.analysis_in_progress = True

        callbacks = self.build_callbacks()
        self.analysis_worker = DataAnalysisWorker(settings, self.logger, self.cache_dir, self.VERSION, callbacks)
        self.analysis_worker.set_task('comparison', settings, primary_data=worker_data)

        self.worker_thread = threading.Thread(target=self.analysis_worker.run, daemon=True)
        self.worker_thread.start()
        self.status_bar_text = f"🔄 Analyzing {len(self.comparison_tickers_list)} comparison tickers..."

    @mainthread
    def on_comparison_data_ready(self, comparison_results):
        """Processes comparison analysis results on the main thread."""
        self.comparison_data = comparison_results
        self.update_charts(chart_only=['comparison'])
        self.status_bar_text = f"✅ Comparison analysis complete."
    
    @mainthread
    def on_worker_finished(self):
        self.analysis_in_progress = False
        self.analysis_worker = None
        self.worker_thread = None
        self.logger.info("Worker thread finished.")

    @mainthread
    def on_worker_error(self, e):
        self.analysis_in_progress = False
        self.logger.error(f"Worker Error: {e}")
        self.show_popup("Worker Error", e)

    @mainthread
    def on_worker_progress(self, message):
        self.status_bar_text = message

    # --- Visualization Methods ---
    def get_plot_style(self):
        """Returns the appropriate colors for Matplotlib based on the dark mode state."""
        return {
            'fig_bg': self.plot_fig_bg,
            'ax_bg': self.plot_ax_bg,
            'fg': self.plot_fg,
            'grid_color': self.plot_grid_color,
        }

    def update_charts(self, chart_only=None, force_redraw=False):
        """Update selected or all visualization tabs."""
        if not self.primary_data: return

        style = self.get_plot_style()
        self.logger.info(f"Updating charts. Dark mode: {self.ids.dark_mode_check.active}")
        
        charts_to_update = chart_only or ['bar', 'heatmap', 'scatter', 'comparison']

        if 'bar' in charts_to_update:
            self.update_bar_chart(style)
        if 'heatmap' in charts_to_update:
            self.update_heatmap(style)
        if 'scatter' in charts_to_update:
            self.update_scatter_plot(style)
        if 'comparison' in charts_to_update:
            self.update_comparison_chart(style)
            
    def clear_plot_canvas(self, canvas_id):
        """Helper to clear a Kivy layout (canvas)."""
        canvas = self.ids[canvas_id]
        canvas.clear_widgets()

    def update_bar_chart(self, style):
        self.clear_plot_canvas('bar_chart_canvas')
        fig, ax = plt.subplots(figsize=(10, 4), facecolor=style['fig_bg'])
        ax.set_facecolor(style['ax_bg'])

        metric_data = self.primary_data['month_avg'] if self.bar_metric == "Mean" else self.primary_data['month_median']
        
        bars = ax.bar(range(1, 13), metric_data*100, alpha=0.8, color=style['fg'], edgecolor=style['fg'], linewidth=1)
        
        ax.tick_params(colors=style['fg'])
        ax.yaxis.label.set_color(style['fg'])
        ax.title.set_color(style['fg'])
        ax.set_xticks(range(1, 13))
        ax.set_xticklabels(self.months, fontsize=9, color=style['fg'])
        ax.set_ylabel(f'{self.bar_metric} Monthly Return (%)', fontsize=9, fontweight='bold')
        ax.set_title(f'{self.bar_metric} Monthly Returns for {self.current_ticker}', fontsize=10, fontweight='bold', pad=15)
        ax.grid(axis='y', alpha=0.3, linestyle='--', color=style['grid_color'])
        ax.axhline(y=0, color=style['fg'], linestyle='-', linewidth=0.8, alpha=0.5)

        for bar in bars:
            height = bar.get_height()
            bar.set_color('#e74c3c' if height < 0 else '#27ae60')
            bar.set_edgecolor('#c0392b' if height < 0 else '#229954')
            ax.text(bar.get_x() + bar.get_width()/2., height + (0.05 if height >= 0 else -0.2),
                    f'{height:.1f}%', ha='center', va='bottom' if height >= 0 else 'top',
                    fontsize=6, fontweight='bold', color=style['fg'])
        
        if self.show_benchmark:
            overall_avg_pct = self.primary_data['overall_avg'] * 100
            ax.axhline(y=overall_avg_pct, color='#f1c40f', linestyle='--', linewidth=2,
                       label=f'Overall Average: {overall_avg_pct:.2f}%', alpha=0.8)
            ax.legend(loc='upper right', framealpha=0.9, facecolor=style['ax_bg'], labelcolor=style['fg'])

        self.ids.bar_chart_canvas.add_widget(FigureCanvasKivyAgg(fig))

    def update_heatmap(self, style):
        self.clear_plot_canvas('heatmap_canvas')
        fig, ax = plt.subplots(figsize=(14, 8), facecolor=style['fig_bg'])
        ax.set_facecolor(style['ax_bg'])
        
        sns.heatmap(self.primary_data['pivot']*100, center=0, cmap='RdYlGn', cbar_kws={'label':'Monthly Return (%)'},
                    linewidths=.5, ax=ax, annot=True, fmt='.1f', cbar=True, annot_kws={"color": style['fg']})
        
        ax.tick_params(colors=style['fg'])
        ax.title.set_color(style['fg'])
        ax.set_xlabel('Month', fontsize=11, fontweight='bold', color=style['fg'])
        ax.set_ylabel('Year', fontsize=11, fontweight='bold', color=style['fg'])
        ax.set_title(f'Month-by-Year Returns for {self.current_ticker}', fontsize=12, fontweight='bold', pad=20, color=style['fg'])
        ax.set_xticklabels(self.months, rotation=0, fontsize=9, color=style['fg'])
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8, color=style['fg'])
        
        cbar = ax.collections[0].colorbar
        cbar.set_label('Monthly Return (%)', fontsize=10, fontweight='bold', color=style['fg'])
        cbar.ax.yaxis.set_tick_params(color=style['fg'])
        cbar.ax.yaxis.set_ticklabels(cbar.ax.get_yticklabels(), color=style['fg'])
        
        self.ids.heatmap_canvas.add_widget(FigureCanvasKivyAgg(fig))

    def update_scatter_plot(self, style):
        self.clear_plot_canvas('scatter_canvas')
        pivot = self.primary_data['pivot']
        month_avg = self.primary_data['month_avg']
        
        monthly_stats = pd.DataFrame({
            'month': range(1, 13),
            'avg_return': month_avg * 100,
            'std_dev': pivot.std() * 100,
            'positive_rate': (pivot > 0).sum() / pivot.count() * 100
        })
        
        fig, ax = plt.subplots(figsize=(12, 8), facecolor=style['fig_bg'])
        ax.set_facecolor(style['ax_bg'])
        
        scatter = ax.scatter(monthly_stats['std_dev'], monthly_stats['avg_return'],
                             c=monthly_stats['positive_rate'], cmap='RdYlGn', s=250, 
                             alpha=0.8, edgecolors=style['fg'], linewidth=1)
        
        for i, month in enumerate(self.months):
            ax.annotate(month, (monthly_stats['std_dev'].iloc[i], monthly_stats['avg_return'].iloc[i]),
                        xytext=(5, 5), textcoords='offset points', fontsize=9, fontweight='bold',
                        bbox=dict(boxstyle="round,pad=0.2", fc=style['fig_bg'], alpha=0.7, ec=style['fg']),
                        color=style['fg'])
        
        ax.tick_params(colors=style['fg'])
        ax.yaxis.label.set_color(style['fg'])
        ax.xaxis.label.set_color(style['fg'])
        ax.title.set_color(style['fg'])
        ax.set_xlabel('Monthly Return Standard Deviation (%)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Average Monthly Return (%)', fontsize=11, fontweight='bold')
        ax.set_title(f'Risk vs Return by Month for {self.current_ticker}\n(Color = Positive Return Rate)',
                     fontsize=12, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3, linestyle='--', color=style['grid_color'])
        
        cbar = plt.colorbar(scatter, ax=ax, label='Positive Return Rate (%)')
        cbar.set_label('Positive Return Rate (%)', fontsize=10, fontweight='bold', color=style['fg'])
        cbar.ax.yaxis.set_tick_params(color=style['fg'])
        cbar.ax.yaxis.set_ticklabels(cbar.ax.get_yticklabels(), color=style['fg'])

        if self.show_benchmark:
            overall_avg_pct = self.primary_data['overall_avg'] * 100
            ax.axhline(y=overall_avg_pct, color='#f1c40f', linestyle='--', linewidth=2, alpha=0.8,
                       label=f'Overall Avg Return: {overall_avg_pct:.2f}%')
            ax.legend(loc='upper right', framealpha=0.9, facecolor=style['ax_bg'], labelcolor=style['fg'])
            
        self.ids.scatter_canvas.add_widget(FigureCanvasKivyAgg(fig))

    def update_comparison_chart(self, style):
        self.clear_plot_canvas('comparison_canvas')
        if not self.comparison_data: return

        fig, ax = plt.subplots(figsize=(12, 6), facecolor=style['fig_bg'])
        ax.set_facecolor(style['ax_bg'])
        
        line_styles = ['-', '--', ':', '-.']
        markers = ['o', 's', '^', 'D']
        
        for i, (ticker, data) in enumerate(self.comparison_data.items()):
            month_avg_pct = data['month_avg'] * 100
            ax.plot(range(1, 13), month_avg_pct, label=ticker, 
                    linewidth=2, marker=markers[i % len(markers)], linestyle=line_styles[i % len(line_styles)])
        
        ax.tick_params(colors=style['fg'])
        ax.yaxis.label.set_color(style['fg'])
        ax.title.set_color(style['fg'])
        ax.set_xticks(range(1, 13))
        ax.set_xticklabels(self.months, fontsize=10, color=style['fg'])
        ax.set_ylabel('Average Monthly Return (%)', fontsize=11, fontweight='bold', color=style['fg'])
        ax.set_title('Comparative Monthly Returns', fontsize=12, fontweight='bold', pad=20, color=style['fg'])
        ax.grid(True, alpha=0.3, linestyle='--', color=style['grid_color'])
        ax.legend(loc='upper right', framealpha=0.9, facecolor=style['ax_bg'], labelcolor=style['fg'])
        ax.axhline(y=0, color=style['fg'], linestyle='-', linewidth=0.8, alpha=0.5)

        self.ids.comparison_canvas.add_widget(FigureCanvasKivyAgg(fig))

    # --- Output and Utility Methods ---
    def update_summary(self):
        """Update the summary tab with analysis results."""
        if not self.primary_data: return
        
        pivot = self.primary_data['pivot']
        month_avg = self.primary_data['month_avg']
        month_median = self.primary_data['month_median']
        monthly_ret = self.primary_data['monthly_ret']
        overall_avg = self.primary_data['overall_avg']
        
        summary = f"📊 MONTHLY RETURN ANALYSIS SUMMARY\n"
        summary += f"{'='*50}\n"
        summary += f"Index/ETF: {self.current_ticker}\n"
        summary += f"Analysis Period: {pivot.index[0]} to {pivot.index[-1]} ({len(pivot)} years)\n"
        summary += f"Total Months Analyzed: {len(monthly_ret)}\n"
        summary += f"{'='*50}\n"
        summary += "📈 MONTHLY AVERAGE RETURNS:\n"
        summary += "-" * 30 + "\n"
        
        for month_num, month_name in enumerate(self.months, 1):
            if month_num in month_avg.index:
                avg_ret = month_avg[month_num] * 100
                summary += f"{month_name:3}: {avg_ret:+6.2f}%\n"

        summary += f"\n{'='*50}\n"
        overall_avg_pct = overall_avg * 100
        summary += f"🎯 OVERALL AVERAGE RETURN: {overall_avg_pct:+6.2f}%\n"
        best_month_idx = month_avg.idxmax()
        worst_month_idx = month_avg.idxmin()
        best_month_name = self.months[best_month_idx - 1]
        worst_month_name = self.months[worst_month_idx - 1]
        summary += f"🏆 BEST MONTH: {best_month_name} ({month_avg[best_month_idx]*100:+.2f}%)\n"
        summary += f"⚠️  WORST MONTH: {worst_month_name} ({month_avg[worst_month_idx]*100:+.2f}%)\n"
        
        self.ids.summary_text.text = summary
        self.logger.info("Summary tab updated.")

    def export_to_excel(self):
        """Export analysis data to Excel."""
        if not self.primary_data:
            self.show_popup("Warning", "No data to export. Please analyze data first.")
            return

        settings = self.get_current_settings()
        filename = f"{self.current_ticker.replace('^', '').replace('.JO', '').replace('.SS', '')}_monthly_analysis_{settings['start_year']}_{settings['end_year']}.xlsx"
        
        self.show_save_dialog(filename, self._save_excel)

    def _save_excel(self, path, filename):
        """Callback to save excel data."""
        filepath = os.path.join(path, filename)
        try:
            with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
                self.primary_data['pivot'].to_excel(writer, sheet_name='Year_Month_Returns')
                summary_stats = pd.DataFrame({
                    'Month': self.months, 
                    'Average_Return_%': self.primary_data['month_avg'].values * 100,
                    'Median_Return_%': self.primary_data['month_median'].values * 100,
                    'Std_Dev_%': self.primary_data['pivot'].std().values * 100,
                })
                summary_stats.to_excel(writer, sheet_name='Monthly_Summary', index=False)
            
            self.show_popup("Success", f"✅ Data exported to {filepath}")
            self.status_bar_text = f"💾 Data exported to {filepath}"
        except Exception as e:
            self.logger.exception("Error exporting data to Excel.")
            self.show_popup("Error", f"Error exporting data: {e}")
            self.status_bar_text = "❌ Export failed"

    def generate_report(self):
        """Generates a Markdown report summarizing all analysis tabs."""
        if not self.primary_data:
            self.show_popup("Warning", "Please run primary analysis first.")
            return

        if not self.stat_results and not self.ml_results:
             self.show_popup("Warning", "Run Statistical Tests or ML Analysis first.")
             return

        report_content = self._generate_markdown_report_content()
        filename = f"{self.current_ticker.replace('^', '').replace('.JO', '').replace('.SS', '')}_Report_{datetime.now().strftime('%Y%m%d_%H%M')}.md"
        
        self.show_save_dialog(filename, self._save_report)

    def _save_report(self, path, filename):
        """Callback to save markdown report."""
        filepath = os.path.join(path, filename)
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(report_content)
            
            self.show_popup("Success", f"✅ Report exported to {filepath}")
            self.status_bar_text = f"📄 Report exported to {filepath}"
        except Exception as e:
            self.logger.exception("Error exporting report.")
            self.show_popup("Error", f"Error exporting report: {e}")
            self.status_bar_text = "❌ Report export failed"

    def _generate_markdown_report_content(self):
        """Helper to compile report content from analysis results."""
        date_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        settings = self.get_current_settings()
        
        # --- 1. Header & General Summary ---
        report = f"# Monthly Return Analysis Report: {self.current_ticker}\n\n"
        report += f"**Generated On:** {date_str}\n"
        report += f"**Analysis Period:** {settings['start_year']} to {settings['end_year']} ({len(self.primary_data['pivot'])} years)\n"
        report += f"**Total Months Analyzed:** {len(self.primary_data['monthly_ret'])}\n\n"
        report += "---\n\n"

        # --- 2. Summary Statistics ---
        report += "## 📈 Monthly Return Summary\n\n"
        
        month_avg = self.primary_data['month_avg']
        overall_avg_pct = self.primary_data['overall_avg'] * 100
        best_month_idx = month_avg.idxmax()
        worst_month_idx = month_avg.idxmin()
        best_month_name = self.months[best_month_idx - 1]
        worst_month_name = self.months[worst_month_idx - 1]
        
        report += f"| Metric | Value |\n"
        report += f"| :--- | :--- |\n"
        report += f"| **Overall Average Monthly Return** | {overall_avg_pct:+.2f}% |\n"
        report += f"| **Best Month** | {best_month_name} ({month_avg[best_month_idx]*100:+.2f}%) |\n"
        report += f"| **Worst Month** | {worst_month_name} ({month_avg[worst_month_idx]*100:+.2f}%) |\n\n"
        
        report += "### Monthly Breakdown (Averages, Medians, and Risk)\n\n"
        report += "| Month | Avg. Return (%) | Median Return (%) | Standard Deviation (%) |\n"
        report += "| :---: | :---: | :---: | :---: |\n"
        for month_num, month_name in enumerate(self.months, 1):
             if month_num in month_avg.index:
                avg_ret = month_avg[month_num] * 100
                med_ret = self.primary_data['month_median'][month_num] * 100
                std_dev = self.primary_data['pivot'].std()[month_num] * 100
                report += f"| {month_name} | {avg_ret:+.2f} | {med_ret:+.2f} | {std_dev:.2f} |\n"
        report += "\n"

        # --- 3. Statistical Tests ---
        report += "## 🧮 Statistical Significance\n\n"
        if self.stat_results:
            report += "```\n"
            report += self.stat_results.strip()
            report += "\n```\n\n"
        else:
            report += "*Statistical tests were not run.*\n\n"

        # --- 4. Machine Learning Analysis ---
        report += "## 🤖 Machine Learning Insights\n\n"
        if self.ml_results:
            # ... (Content from old _generate_markdown_report_content) ...
            forecast_date_str = self.ml_results['forecast_date'].strftime('%Y-%m')
            forecast_mean_pct = self.ml_results['forecast_mean'] * 100
            pca_variance = self.ml_results['pca_explained_variance']
            n_clusters = self.ml_results['n_clusters']
            contamination = self.ml_results['contamination']
            arima_order = self.ml_results['arima_order']
            
            report += "### Time Series Forecast\n\n"
            report += f"The next forecasted monthly return (for {forecast_date_str}) is **{forecast_mean_pct:+.4f}%**.\n"
            report += f"Model Used: {'AutoARIMA' if PMDARIMA_AVAILABLE else 'ARIMA'} with order {arima_order}.\n\n"
            
            report += "### Clustering and Anomaly Detection\n\n"
            report += f"- **KMeans Clusters (k):** {n_clusters}\n"
            report += f"- **PCA Components (PC1+PC2):** These components capture {pca_variance[0]*100:.2f}% and {pca_variance[1]*100:.2f}% of the monthly return pattern variance, respectively.\n"
            
            anomalies = np.where(self.ml_results['anom_pred'] == -1)[0]
            if len(anomalies) > 0:
                anomaly_months = [self.months[i] for i in anomalies]
                report += f"- **Anomalous Months (Isolation Forest, Contamination {contamination:.2f}):** {', '.join(anomaly_months)}\n\n"
            else:
                report += f"- **Anomalies:** No significant anomalies detected (Contamination {contamination:.2f}).\n\n"
        else:
            report += "*ML analysis was not run.*\n\n"
            
        return report

    # --- ML and Stats ---
    def run_statistical_tests(self):
        """Run statistical significance tests."""
        if not self.primary_data:
            self.show_popup("Warning", "No data available. Please analyze data first.")
            return
            
        self.ids.stats_text.text = ""
        self.status_bar_text = "🧮 Running statistical tests..."
        
        try:
            results = "--- Statistical Test Results ---\n\n"
            monthly_ret = self.primary_data['monthly_ret']
            
            jb_test = stats.jarque_bera(monthly_ret)
            results += f"1. Normality Test (Jarque-Bera):\n"
            results += f"  Statistic: {jb_test.statistic:.4f}\n  p-value: {jb_test.pvalue:.4f}\n"
            results += "  Conclusion: Monthly returns are likely NOT normally distributed (p < 0.05).\n" if jb_test.pvalue < 0.05 else "  Conclusion: Monthly returns are likely normally distributed (p >= 0.05).\n"
                
            results += "\n--- Mean Return Significance ---\n\n"
            
            t_test = stats.ttest_1samp(monthly_ret, 0)
            results += f"2. Mean Return Test (T-Test vs 0):\n"
            results += f"  Mean Monthly Return: {self.primary_data['overall_avg'] * 100:.4f}%\n"
            results += f"  T-Statistic: {t_test.statistic:.4f}\n  p-value: {t_test.pvalue:.4f}\n"
            results += "  Conclusion: The mean return is statistically SIGNIFICANTLY different from zero (p < 0.05).\n" if t_test.pvalue < 0.05 else "  Conclusion: The mean return is NOT statistically different from zero (p >= 0.05).\n"

            self.stat_results = results
            self.ids.stats_text.text = results
            self.status_bar_text = "🧮 Statistical tests completed"
        except Exception as e:
            self.logger.exception("Error running statistical tests.")
            self.show_popup("Error", f"Error running statistical tests: {e}")
            self.status_bar_text = "❌ Statistical tests failed"
            self.stat_results = None

    def run_ml_analysis(self):
        """Initiates ML analysis."""
        if not self.primary_data:
            self.show_popup("Warning", "No data available. Please analyze data first.")
            return

        settings = self.get_current_settings()
        if settings is None: return

        if self.analysis_in_progress:
            self.show_popup("Busy", "Worker thread is currently running. Please wait.")
            return

        self.ids.ml_text.text = ""
        self.clear_plot_canvas('ml_plot_frame')
        self.analysis_in_progress = True
        
        worker_data = {
            'monthly_ret': self.primary_data['monthly_ret'],
            'pivot': self.primary_data['pivot']
        }

        callbacks = self.build_callbacks()
        self.analysis_worker = DataAnalysisWorker(settings, self.logger, self.cache_dir, self.VERSION, callbacks)
        self.analysis_worker.set_task('ml', settings, primary_data=worker_data)

        self.worker_thread = threading.Thread(target=self.analysis_worker.run, daemon=True)
        self.worker_thread.start()
        self.status_bar_text = "🤖 Starting ML Analysis..."

    @mainthread
    def on_ml_data_ready(self, results):
        self.ml_results = results
        self.update_ml_tab()
        self.status_bar_text = "🤖 Advanced ML analysis completed"

    def update_ml_tab(self):
        """Updates the ML tab content and plot based on results."""
        if not self.ml_results: return
        
        style = self.get_plot_style()
        
        # --- Text Summary ---
        forecast_date_str = self.ml_results['forecast_date'].strftime('%Y-%m')
        forecast_mean_pct = self.ml_results['forecast_mean'] * 100
        pca_variance = self.ml_results['pca_explained_variance']
        n_clusters = self.ml_results['n_clusters']
        contamination = self.ml_results['contamination']
        arima_order = self.ml_results['arima_order']
        
        summary = "--- ML Analysis Summary ---\n\n"
        summary += f"🔮 Forecast for {forecast_date_str}:\n"
        summary += f"  Predicted Mean Monthly Return: {forecast_mean_pct:+.4f}%\n"
        summary += f"  (Model: {'AutoARIMA' if PMDARIMA_AVAILABLE else 'ARIMA'}{arima_order})\n\n"
        
        summary += f"📊 PCA Components:\n"
        summary += f"  PC1 Explains: {pca_variance[0]*100:.2f}%\n"
        summary += f"  PC2 Explains: {pca_variance[1]*100:.2f}%\n"
        summary += "  (The plot below visualizes the months using these two main components)\n\n"
        
        summary += f"🧩 KMeans Clustering (k={n_clusters}):\n"
        summary += f"  The 12 months are grouped into {n_clusters} clusters based on their historical yearly returns.\n\n"
        
        summary += f"🚨 Isolation Forest Anomaly Detection:\n"
        summary += f"  Contamination parameter used: {contamination:.2f}\n"
        anomalies = np.where(self.ml_results['anom_pred'] == -1)[0]
        if len(anomalies) > 0:
            anomaly_months = [self.months[i] for i in anomalies]
            summary += f"  Detected Anomalous Months: {', '.join(anomaly_months)}\n"
        else:
            summary += "  No anomalies detected in monthly return patterns.\n"
        
        self.ids.ml_text.text = summary
        
        # --- Plot Visualization (PCA + Clustering) ---
        self.clear_plot_canvas('ml_plot_frame')
        fig, ax = plt.subplots(figsize=(10, 6), facecolor=style['fig_bg'])
        ax.set_facecolor(style['ax_bg'])
        
        features_reduced = self.ml_results['features_reduced']
        clusters = self.ml_results['clusters']
        anom_pred = self.ml_results['anom_pred']
        
        base_cols = ['#3498db', '#e74c3c', '#2ecc71', '#f1c40f', '#9b59b6', '#1abc9c', '#e67e22', '#34495e', '#c0392b', '#7f8c8d']
        cols = base_cols[:n_clusters] 
        
        for cl in np.unique(clusters):
            idx = np.where(clusters == cl)
            norm_idx = np.intersect1d(idx, np.where(anom_pred != -1))
            
            ax.scatter(features_reduced[norm_idx, 0], features_reduced[norm_idx, 1],
                       s=150, alpha=0.7, color=cols[cl], label=f'Cluster {cl+1}')
                       
            for i in norm_idx:
                ax.annotate(self.months[i], (features_reduced[i, 0], features_reduced[i, 1]),
                           xytext=(3, 3), textcoords='offset points', fontsize=9, fontweight='bold',
                           bbox=dict(boxstyle="round,pad=0.2", fc=style['fig_bg'], alpha=0.7, ec=style['fg']),
                           color=style['fg'])

        anom_idx = np.where(anom_pred == -1)
        if np.any(anom_idx):
            anom_color = '#f1c40f' 
            ax.scatter(features_reduced[anom_idx, 0], features_reduced[anom_idx, 1],
                       s=200, marker='X', c=anom_color, label='Anomaly', linewidths=2, edgecolors=style['fg'])
            for i in anom_idx[0]:
                 ax.annotate(self.months[i], (features_reduced[i, 0], features_reduced[i, 1]),
                           xytext=(3, 3), textcoords='offset points', fontsize=9, fontweight='bold',
                           bbox=dict(boxstyle="round,pad=0.2", fc=style['fig_bg'], alpha=0.7, ec=style['fg']),
                           color=style['fg'])
                           
        ax.tick_params(colors=style['fg'])
        ax.yaxis.label.set_color(style['fg'])
        ax.xaxis.label.set_color(style['fg'])
        ax.title.set_color(style['fg'])
        ax.set_xlabel('PC1', color=style['fg']); ax.set_ylabel('PC2', color=style['fg'])
        ax.set_title(f'Monthly Clusters – PCA space for {self.current_ticker}', color=style['fg'])
        ax.grid(True, alpha=0.3, color=style['grid_color']); 
        ax.legend(loc='upper right', framealpha=0.9, facecolor=style['ax_bg'], labelcolor=style['fg'])

        self.ids.ml_plot_frame.add_widget(FigureCanvasKivyAgg(fig))

    # --- Kivy Popup and Dialog Helpers ---
    def show_popup(self, title, text):
        """Shows a simple popup message."""
        content = BoxLayout(orientation='vertical', padding=10, spacing=10)
        content.add_widget(Label(text=text, color=self.fg_color))
        btn = Button(text='OK', size_hint_y=None, height=44)
        content.add_widget(btn)
        
        popup = Popup(title=title, content=content, size_hint=(0.6, 0.4))
        btn.bind(on_press=popup.dismiss)
        popup.open()

    def show_save_dialog(self, default_filename, save_callback):
        """Shows a save file dialog popup."""
        content = BoxLayout(orientation='vertical', spacing=10)
        
        # Use home directory or current directory
        try:
            default_path = os.path.expanduser('~')
        except Exception:
            default_path = os.path.abspath(os.path.dirname(__file__))
            
        file_chooser = FileChooserListView(path=default_path, filters=['*.xlsx', '*.md'])
        filename_input = TextInput(text=default_filename, size_hint_y=None, height=44)
        
        btn_box = BoxLayout(size_hint_y=None, height=44, spacing=10)
        save_btn = Button(text='Save')
        cancel_btn = Button(text='Cancel')
        btn_box.add_widget(save_btn)
        btn_box.add_widget(cancel_btn)
        
        content.add_widget(file_chooser)
        content.add_widget(filename_input)
        content.add_widget(btn_box)
        
        popup = Popup(title='Save File As...', content=content, size_hint=(0.8, 0.8))
        
        def do_save(instance):
            path = file_chooser.path
            filename = filename_input.text
            if not filename:
                self.show_popup("Error", "Filename cannot be empty.")
                return
            save_callback(path, filename)
            popup.dismiss()
            
        save_btn.bind(on_press=do_save)
        cancel_btn.bind(on_press=popup.dismiss)
        popup.open()

# --- Kivy App Class ---
class JSEAnalyzerApp(App):
    """Main Kivy Application."""
    VERSION = "2.9.4 (Kivy Port)"
    
    def build(self):
        self.title = f"📊 Global Index Monthly Return Analyzer (v{self.VERSION})"
        self.logger = setup_logging()
        self.logger.info(f"--- Application Start: Global Index Analyzer v{self.VERSION} (Kivy) ---")
        
        # Initialize Cache Directory
        self.cache_dir = f"cache_v{self.VERSION.replace('.', '_').replace(' ', '_')}"
        os.makedirs(self.cache_dir, exist_ok=True)
        self.logger.info(f"Cache directory set to: {self.cache_dir}")
        
        return MainLayout()

if __name__ == "__main__":
    # Ensure Matplotlib uses the Kivy backend
    matplotlib.use('module://kivy_garden.matplotlib.backend_kivy')
    JSEAnalyzerApp().run()


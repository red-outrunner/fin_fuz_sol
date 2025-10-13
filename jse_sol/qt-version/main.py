# jse_analyzer_pyqt.py
# Global Index Monthly Return Analyzer (PyQt6 Version)
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
import threading # Still used for general locking, but QThread for main tasks
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

# PyQt6 Imports
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QLineEdit, QComboBox, QPushButton, QCheckBox, QTextEdit,
    QTabWidget, QGroupBox, QMessageBox, QFileDialog, QSizePolicy, QSpinBox,
    QFrame
)
from PyQt6.QtCore import (
    QObject, QThread, pyqtSignal, QDate, Qt, QTimer
)
from PyQt6.QtGui import QPalette, QColor, QFont

# Matplotlib-PyQt Integration
# FIX: Use the generic backend_qtagg import path for FigureCanvasQTAgg
# This works regardless of whether Matplotlib is configured for PyQt6 or PySide6.
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# Try importing pmdarima, with fallback if it fails
try:
    from pmdarima import auto_arima
    PMDARIMA_AVAILABLE = True
except (ImportError, ValueError) as e:
    PMDARIMA_AVAILABLE = False
    warnings.warn(f"Failed to import pmdarima ({str(e)}). Using fixed-order ARIMA for forecasting.")

warnings.filterwarnings('ignore')
matplotlib.use('QtAgg')

# --- Setup Logging ---
def setup_logging():
    """Configures the root logger to write to a file."""
    log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('jse_analyzer_pyqt.log', mode='w')
    file_handler.setFormatter(log_formatter)
    
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    if logger.hasHandlers():
        logger.handlers.clear()
        
    logger.addHandler(file_handler)
    return logger

# --- QThread Worker for Non-Blocking Analysis ---
class DataAnalysisWorker(QObject):
    """Handles data fetching and heavy processing in a separate thread."""
    
    # Signals to communicate back to the main thread
    finished = pyqtSignal()
    error = pyqtSignal(str)
    progress = pyqtSignal(str)
    data_ready = pyqtSignal(dict) # Primary analysis results
    comparison_data_ready = pyqtSignal(dict) # Comparison analysis results

    def __init__(self, settings, logger, cache_dir, version, parent=None):
        super().__init__(parent)
        self.settings = settings
        self.logger = logger
        self.cache_dir = cache_dir
        self.VERSION = version
        self._is_running = True

    def stop(self):
        """Allows external stopping of the worker."""
        self._is_running = False

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
            # Cache check improvement: Don't check age, check file integrity/existence.
            # Assuming historical data is immutable.
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
        try:
            ticker = self.settings['ticker']
            start_year = self.settings['start_year']
            end_date = self.settings['end_date']
            
            self.progress.emit(f"📥 Downloading or loading cache for {ticker}...")
            
            data = self.load_cached_data(ticker, start_year, end_date)
            
            if data is None:
                max_retries = 3
                for attempt in range(max_retries):
                    if not self._is_running: return
                    try:
                        self.progress.emit(f"Downloading {ticker} (attempt {attempt+1}/{max_retries})...")
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

            self.progress.emit("⚙️ Processing data...")
            results = self.process_ticker_data(data, ticker)
            
            self.data_ready.emit(results)
            self.finished.emit()
            
        except Exception as e:
            self.logger.exception("An error occurred during primary data analysis.")
            self.error.emit(f"Analysis failed for {ticker}: {str(e)}")
            self.finished.emit()

    def run_comparison_analysis(self, comparison_tickers, primary_data):
        """Comparison analysis routine for comparison tickers."""
        self.progress.emit("Starting parallel comparison analysis...")
        self.logger.info("Starting parallel analysis for comparison tickers.")
        
        comparison_data_results = {}
        tickers_to_fetch = [t for t in comparison_tickers if t != self.settings['ticker']]
        
        # Add primary data results to comparison list to avoid re-download
        if primary_data is not None:
             comparison_data_results[self.settings['ticker']] = primary_data

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
                    self.progress.emit(f"⚠️ Error processing {ticker}: {str(e)}")
        
        self.comparison_data_ready.emit(comparison_data_results)
        self.finished.emit()


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
        """Runs the ML analysis (KMeans, PCA, ARIMA) in the worker thread."""
        try:
            self.progress.emit("🤖 Starting ML Analysis...")
            
            # --- 1. Data Preparation ---
            # Use the monthly returns for clustering/PCA
            X = pivot.fillna(0).values.T # Months (12 rows) x Years (N columns)
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # --- 2. PCA for Visualization ---
            self.progress.emit("Running PCA...")
            pca = PCA(n_components=2)
            features_reduced = pca.fit_transform(X_scaled)
            
            # --- 3. Clustering (KMeans) ---
            self.progress.emit("Running KMeans Clustering (k=3)...")
            kmeans = KMeans(n_clusters=3, random_state=42, n_init='auto')
            clusters = kmeans.fit_predict(X_scaled)
            
            # --- 4. Anomaly Detection (Isolation Forest) ---
            self.progress.emit("Running Isolation Forest Anomaly Detection...")
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            anom_pred = iso_forest.fit_predict(X_scaled)
            
            # --- 5. Time Series Forecast (ARIMA/AutoARIMA) ---
            self.progress.emit("Running Time Series Forecast...")
            
            # Use AutoARIMA if available, otherwise fixed ARIMA(5, 1, 0)
            if PMDARIMA_AVAILABLE:
                model = auto_arima(
                    monthly_ret, seasonal=False, stepwise=True,
                    suppress_warnings=True, error_action='ignore', max_p=5, max_q=5
                )
            else:
                model = ARIMA(monthly_ret, order=(5, 1, 0))
                model = model.fit()

            # Forecast for the next month (next period after last data point)
            forecast_steps = 1
            if PMDARIMA_AVAILABLE:
                forecast_result = model.predict(n_periods=forecast_steps)
                forecast_mean = forecast_result.iloc[0] if isinstance(forecast_result, pd.Series) else forecast_result[0]
            else:
                forecast = model.get_forecast(steps=forecast_steps)
                forecast_mean = forecast.predicted_mean.iloc[0]

            self.progress.emit("Generating ML results...")
            
            ml_results = {
                'features_reduced': features_reduced,
                'clusters': clusters,
                'anom_pred': anom_pred,
                'pca_explained_variance': pca.explained_variance_ratio_,
                'forecast_mean': forecast_mean,
                'forecast_date': monthly_ret.index[-1] + timedelta(days=31) # Approx next month end
            }
            
            self.data_ready.emit(ml_results)
            self.finished.emit()
            
        except Exception as e:
            self.logger.exception("ML analysis failed.")
            self.error.emit(f"ML analysis failed: {str(e)}")
            self.finished.emit()


# --- Main Application Class (PyQt6) ---
class JSEAnalyzer(QMainWindow):
    """A PyQt6 application for analyzing monthly returns of global financial indices."""
    VERSION = "2.9.0 (PyQt6)"

    def __init__(self):
        super().__init__()
        self.logger = setup_logging()
        self.logger.info(f"--- Application Start: Global Index Analyzer v{self.VERSION} (PyQt6) ---")

        # Data and State
        self.data = None # Original data
        self.monthly_ret = None # Monthly returns (Series)
        self.pivot = None # Monthly pivot table
        self.month_avg = None
        self.month_median = None
        self.overall_avg = 0
        self.comparison_data = {}
        self.comparison_tickers = [] # List of ticker strings
        self.months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        self.ml_results = None

        # Thread management
        self.worker_thread = None
        self.current_worker = None
        
        # Cache directory
        self.cache_dir = f"cache_v{self.VERSION.replace('.', '_').replace(' ', '_')}"
        os.makedirs(self.cache_dir, exist_ok=True)
        self.logger.info(f"Cache directory set to: {self.cache_dir}")
        
        self.setup_ui()
        self.logger.info("Application initialized successfully.")
        
        # Apply default theme
        self.toggle_dark_mode(self.dark_mode_check.isChecked())


    # --- UI Setup ---
    def setup_ui(self):
        """Set up the main PyQt6 GUI components."""
        self.setWindowTitle(f"📊 Global Index Monthly Return Analyzer (v{self.VERSION})")
        self.setGeometry(100, 100, 1400, 900)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # Ticker options map
        self.ticker_options = {
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
        
        # --- Header and Config ---
        header_frame = QFrame()
        header_layout = QHBoxLayout(header_frame)
        
        title_label = QLabel(f"📈 Global Index Monthly Return Analyzer (v{self.VERSION})")
        title_font = QFont("Arial", 12, QFont.Weight.Bold)
        title_label.setFont(title_font)
        header_layout.addWidget(title_label)
        header_layout.addStretch(1)

        self.dark_mode_check = QCheckBox("🌙 Dark Mode")
        self.dark_mode_check.setChecked(False)
        self.dark_mode_check.stateChanged.connect(lambda state: self.toggle_dark_mode(state == Qt.CheckState.Checked.value))
        header_layout.addWidget(self.dark_mode_check)
        
        main_layout.addWidget(header_frame)

        # Configuration Panel (QGroupBox)
        config_group = QGroupBox("⚙️ Configuration Panel")
        config_layout = QVBoxLayout(config_group)
        
        # Row 1: Ticker Selection
        ticker_row = QHBoxLayout()
        ticker_row.addWidget(QLabel("Index/ETF:"))
        
        self.ticker_combo = QComboBox()
        self.ticker_combo.addItems(list(self.ticker_options.keys()))
        self.ticker_combo.setCurrentText("🇿🇦 JSE All Share (^J203.JO)")
        self.ticker_combo.setToolTip("Select a predefined index or ETF for analysis.")
        ticker_row.addWidget(self.ticker_combo)
        
        self.custom_ticker_check = QCheckBox("Custom Ticker")
        self.custom_ticker_check.stateChanged.connect(self.toggle_custom_ticker)
        ticker_row.addWidget(self.custom_ticker_check)
        
        self.custom_ticker_entry = QLineEdit()
        self.custom_ticker_entry.setPlaceholderText("e.g., AAPL")
        self.custom_ticker_entry.setVisible(False)
        ticker_row.addWidget(self.custom_ticker_entry)
        ticker_row.addStretch(1)
        config_layout.addLayout(ticker_row)

        # Row 2: Date Controls
        date_row = QHBoxLayout()
        date_row.addWidget(QLabel("Date Range:"))
        
        self.date_range_combo = QComboBox()
        self.date_range_combo.addItems(["Custom", "Last 5 Years", "Last 10 Years", "Last 20 Years", "All Data"])
        self.date_range_combo.setCurrentText("Custom")
        self.date_range_combo.currentTextChanged.connect(self.on_date_range_change)
        date_row.addWidget(self.date_range_combo)
        
        date_row.addWidget(QLabel("Start Year:"))
        self.start_year_spin = QSpinBox()
        self.start_year_spin.setRange(1900, datetime.now().year)
        self.start_year_spin.setValue(1990)
        date_row.addWidget(self.start_year_spin)
        
        date_row.addWidget(QLabel("End Date (YYYY-MM-DD):"))
        self.end_date_entry = QLineEdit(datetime.today().strftime("%Y-%m-%d"))
        self.end_date_entry.setToolTip("Enter the end date in YYYY-MM-DD format.")
        date_row.addWidget(self.end_date_entry)
        date_row.addStretch(1)
        config_layout.addLayout(date_row)
        
        # Row 3: Comparison and Actions
        action_row = QHBoxLayout()
        action_row.addWidget(QLabel("Compare With:"))
        
        self.compare_combo = QComboBox()
        self.compare_combo.addItems([""] + list(self.ticker_options.keys()))
        self.compare_combo.setToolTip("Select an index to compare with the primary ticker.")
        action_row.addWidget(self.compare_combo)
        
        self.add_compare_btn = QPushButton("➕ Add Comparison")
        self.add_compare_btn.clicked.connect(self.add_comparison_ticker)
        action_row.addWidget(self.add_compare_btn)
        
        self.analyze_btn = QPushButton("🔍 Analyze Data")
        self.analyze_btn.clicked.connect(self.analyze_data)
        self.analyze_btn.setStyleSheet("background-color: #3498db; color: white;")
        action_row.addWidget(self.analyze_btn)
        
        self.export_btn = QPushButton("💾 Export to Excel")
        self.export_btn.clicked.connect(self.export_to_excel)
        self.export_btn.setStyleSheet("background-color: #27ae60; color: white;")
        self.export_btn.setEnabled(False)
        action_row.addWidget(self.export_btn)
        
        self.pdf_btn = QPushButton("📄 Export PDF")
        self.pdf_btn.clicked.connect(self.generate_pdf_report)
        self.pdf_btn.setStyleSheet("background-color: #f39c12; color: white;")
        self.pdf_btn.setEnabled(False)
        action_row.addWidget(self.pdf_btn)
        
        self.ml_btn = QPushButton("🤖 ML Analysis")
        self.ml_btn.clicked.connect(self.run_ml_analysis)
        self.ml_btn.setEnabled(False)
        action_row.addWidget(self.ml_btn)
        
        self.stats_btn = QPushButton("🧮 Significance Test")
        self.stats_btn.clicked.connect(self.run_statistical_tests)
        self.stats_btn.setEnabled(False)
        action_row.addWidget(self.stats_btn)
        
        action_row.addStretch(1)
        config_layout.addLayout(action_row)
        main_layout.addWidget(config_group)

        # --- Tab Widget for Results ---
        self.notebook = QTabWidget()
        main_layout.addWidget(self.notebook)

        # Tabs
        self.bar_frame = self.create_chart_tab("bar_chart_canvas")
        self.heatmap_frame = self.create_chart_tab("heatmap_canvas")
        self.scatter_frame = self.create_chart_tab("scatter_canvas")
        self.comparison_frame = self.create_chart_tab("comparison_canvas")
        
        self.summary_text = QTextEdit()
        self.summary_text.setReadOnly(True)
        
        self.stats_content = QWidget()
        self.stats_layout = QVBoxLayout(self.stats_content)
        self.stats_text = QTextEdit()
        self.stats_text.setReadOnly(True)
        self.stats_layout.addWidget(self.stats_text)
        
        self.ml_content = QWidget()
        self.ml_layout = QVBoxLayout(self.ml_content)
        self.ml_text = QTextEdit()
        self.ml_text.setReadOnly(True)
        self.ml_layout.addWidget(self.ml_text)
        self.ml_plot_frame = QWidget()
        self.ml_layout.addWidget(self.ml_plot_frame)
        
        self.notebook.addTab(self.bar_frame, "📊 Average Returns")
        self.notebook.addTab(self.heatmap_frame, "🌡️ Year-Month Heatmap")
        self.notebook.addTab(self.scatter_frame, "⚖️ Risk vs Return")
        self.notebook.addTab(self.summary_text, "📋 Summary Statistics")
        self.notebook.addTab(self.comparison_frame, "🔄 Comparison")
        self.notebook.addTab(self.stats_content, "🧮 Statistical Tests")
        self.notebook.addTab(self.ml_content, "🤖 ML Analysis")
        
        # Bar Chart Controls (inside the Bar Frame layout)
        bar_controls = QHBoxLayout()
        self.bar_metric_combo = QComboBox()
        self.bar_metric_combo.addItems(["Mean", "Median"])
        self.bar_metric_combo.currentTextChanged.connect(self.update_charts)
        bar_controls.addWidget(QLabel("Metric:"))
        bar_controls.addWidget(self.bar_metric_combo)
        
        self.toggle_benchmark_btn = QCheckBox("📊 Show Benchmark")
        self.toggle_benchmark_btn.setChecked(True)
        self.toggle_benchmark_btn.stateChanged.connect(lambda: self.update_charts(chart_only=['bar', 'scatter']))
        bar_controls.addWidget(self.toggle_benchmark_btn)
        bar_controls.addStretch(1)
        self.bar_frame.layout().insertLayout(0, bar_controls) # Insert above the chart

        # --- Status Bar ---
        self.status_bar = self.statusBar()
        self.status_bar.showMessage("✨ Ready - Select an index and click Analyze")
        
        self.on_date_range_change() # Initial date setting

    def create_chart_tab(self, canvas_name):
        """Helper to create a standard frame for charts."""
        frame = QWidget()
        layout = QVBoxLayout(frame)
        
        # Placeholder for the canvas (will be replaced by Matplotlib)
        setattr(self, canvas_name, None)
        
        return frame
        
    # --- UI Handlers ---
    def toggle_dark_mode(self, is_dark):
        """Toggle between light and dark mode using QPalette and QSS."""
        self.logger.info(f"Toggling dark mode to {'ON' if is_dark else 'OFF'}.")
        
        palette = QPalette()
        if is_dark:
            dark_color = QColor(44, 62, 80) # Dark Blue/Gray
            light_color = QColor(236, 240, 241) # Light Gray/Off-white
            
            palette.setColor(QPalette.ColorRole.Window, dark_color)
            palette.setColor(QPalette.ColorRole.WindowText, light_color)
            palette.setColor(QPalette.ColorRole.Base, QColor(52, 73, 94))
            palette.setColor(QPalette.ColorRole.AlternateBase, dark_color)
            palette.setColor(QPalette.ColorRole.ToolTipBase, dark_color)
            palette.setColor(QPalette.ColorRole.ToolTipText, light_color)
            palette.setColor(QPalette.ColorRole.Text, light_color)
            palette.setColor(QPalette.ColorRole.Button, dark_color)
            palette.setColor(QPalette.ColorRole.ButtonText, light_color)
            palette.setColor(QPalette.ColorRole.BrightText, QColor(255, 0, 0))
            palette.setColor(QPalette.ColorRole.Link, QColor(41, 128, 185))
            palette.setColor(QPalette.ColorRole.Highlight, QColor(52, 152, 219))
            palette.setColor(QPalette.ColorRole.HighlightedText, QColor(0, 0, 0))
        else:
            app = QApplication.instance()
            app.setPalette(QApplication.instance().style().standardPalette())
            
        QApplication.instance().setPalette(palette)
        self.summary_text.setStyleSheet(f"background-color: {dark_color.name() if is_dark else 'white'}; color: {light_color.name() if is_dark else '#2c3e50'};")
        self.stats_text.setStyleSheet(f"background-color: {dark_color.name() if is_dark else 'white'}; color: {light_color.name() if is_dark else '#2c3e50'};")
        self.ml_text.setStyleSheet(f"background-color: {dark_color.name() if is_dark else 'white'}; color: {light_color.name() if is_dark else '#2c3e50'};")

        self.update_charts(force_redraw=True) # Redraw Matplotlib charts

    def toggle_custom_ticker(self, state):
        """Toggle custom ticker visibility."""
        is_checked = state == Qt.CheckState.Checked.value
        self.custom_ticker_entry.setVisible(is_checked)
        self.ticker_combo.setEnabled(not is_checked)
        self.logger.info(f"Custom ticker mode: {is_checked}")

    def on_date_range_change(self):
        """Update date inputs based on selected range."""
        selection = self.date_range_combo.currentText()
        self.logger.info(f"Date range changed to: {selection}")
        current_year = datetime.now().year
        
        is_custom = selection == "Custom"
        self.start_year_spin.setEnabled(is_custom)
        self.end_date_entry.setEnabled(is_custom)
        
        if selection == "Last 5 Years":
            self.start_year_spin.setValue(current_year - 5)
            self.end_date_entry.setText(datetime.today().strftime("%Y-%m-%d"))
        elif selection == "Last 10 Years":
            self.start_year_spin.setValue(current_year - 10)
            self.end_date_entry.setText(datetime.today().strftime("%Y-%m-%d"))
        elif selection == "Last 20 Years":
            self.start_year_spin.setValue(current_year - 20)
            self.end_date_entry.setText(datetime.today().strftime("%Y-%m-%d"))
        elif selection == "All Data":
            self.start_year_spin.setValue(1990)
            self.end_date_entry.setText(datetime.today().strftime("%Y-%m-%d"))

    def add_comparison_ticker(self):
        """Add a ticker to the comparison list."""
        ticker_name = self.compare_combo.currentText()
        if not ticker_name:
            QMessageBox.warning(self, "Warning", "Please select a ticker to compare.")
            return

        selected_ticker = self.ticker_options.get(ticker_name)
        if selected_ticker is None:
            QMessageBox.warning(self, "Warning", "Invalid ticker selection.")
            return
            
        if selected_ticker not in self.comparison_tickers:
            self.comparison_tickers.append(selected_ticker)
            self.logger.info(f"Added '{ticker_name}' ({selected_ticker}) to comparison list.")
            self.status_bar.showMessage(f"✅ Added {ticker_name} to comparison list.")
            
            # Auto-run comparison analysis if primary data is available
            if self.pivot is not None:
                self.analyze_comparison_data()
        else:
            QMessageBox.warning(self, "Warning", "Ticker already added for comparison.")

    # --- Data and Analysis Control ---
    def validate_inputs(self):
        """Validate user inputs and set self.settings."""
        settings = {}
        try:
            settings['start_year'] = self.start_year_spin.value()
            if settings['start_year'] < 1900 or settings['start_year'] > datetime.now().year:
                raise ValueError(f"Start year must be between 1900 and {datetime.now().year}.")

            settings['end_date'] = self.end_date_entry.text().strip()
            datetime.strptime(settings['end_date'], "%Y-%m-%d")

            if self.custom_ticker_check.isChecked():
                ticker = self.custom_ticker_entry.text().strip()
                if not ticker: raise ValueError("Custom ticker cannot be empty.")
                settings['ticker'] = ticker
            else:
                ticker_name = self.ticker_combo.currentText()
                settings['ticker'] = self.ticker_options.get(ticker_name, "^J203.JO")
            
            self.current_ticker = settings['ticker']
            settings['end_year'] = settings['end_date'][:4]
            self.logger.info(f"Input validation successful: {settings}")
            return settings
        except ValueError as e:
            QMessageBox.critical(self, "Input Error", str(e))
            self.status_bar.showMessage(f"❌ Input validation failed: {str(e)}")
            return None

    def analyze_data(self):
        """Initiates primary data analysis in a QThread."""
        settings = self.validate_inputs()
        if settings is None: return

        # Reset state and disable buttons
        self.data = None
        self.ml_results = None
        self.set_buttons_enabled(False)

        # Create worker and thread
        self.worker_thread = QThread()
        self.current_worker = DataAnalysisWorker(settings, self.logger, self.cache_dir, self.VERSION)
        self.current_worker.moveToThread(self.worker_thread)

        # Connect signals/slots
        self.worker_thread.started.connect(self.current_worker.run_primary_analysis)
        self.current_worker.data_ready.connect(self.handle_primary_data_ready)
        self.current_worker.progress.connect(self.status_bar.showMessage)
        self.current_worker.error.connect(lambda e: QMessageBox.critical(self, "Analysis Error", e))
        self.current_worker.finished.connect(self.worker_thread.quit)
        self.current_worker.finished.connect(self.current_worker.deleteLater)
        self.worker_thread.finished.connect(self.worker_thread.deleteLater)

        # Start the thread
        self.worker_thread.start()
        self.status_bar.showMessage(f"🔍 Analyzing {self.current_ticker}...")

    def handle_primary_data_ready(self, results):
        """Processes primary analysis results on the main thread."""
        self.data = results['data']
        self.monthly_ret = results['monthly_ret']
        self.pivot = results['pivot']
        self.month_avg = results['month_avg']
        self.month_median = results['month_median']
        self.overall_avg = results['overall_avg']

        self.update_charts(force_redraw=True)
        self.update_summary()
        self.set_buttons_enabled(True)
        
        self.status_bar.showMessage(f"✅ Analysis complete for {self.current_ticker} ({self.start_year_spin.value()}-{self.end_date_entry.text()[:4]})")
        
        # Immediately run comparison analysis if comparison tickers exist
        if self.comparison_tickers:
            QTimer.singleShot(100, self.analyze_comparison_data)

    def analyze_comparison_data(self):
        """Initiates comparison analysis in a QThread."""
        if not self.comparison_tickers: return

        settings = self.validate_inputs()
        if settings is None: return

        # Reuse existing primary data for comparison
        primary_data_for_compare = {
            'month_avg': self.month_avg, 'pivot': self.pivot, 'monthly_ret': self.monthly_ret,
            'month_median': self.month_median, 'overall_avg': self.overall_avg, 'data': self.data
        }
        
        self.set_buttons_enabled(False)

        # Create worker and thread for comparison (can reuse settings)
        self.worker_thread = QThread()
        self.current_worker = DataAnalysisWorker(settings, self.logger, self.cache_dir, self.VERSION)
        self.current_worker.moveToThread(self.worker_thread)

        # Connect signals/slots
        self.worker_thread.started.connect(lambda: self.current_worker.run_comparison_analysis(self.comparison_tickers, primary_data_for_compare))
        self.current_worker.comparison_data_ready.connect(self.handle_comparison_data_ready)
        self.current_worker.progress.connect(self.status_bar.showMessage)
        self.current_worker.finished.connect(self.worker_thread.quit)
        self.current_worker.finished.connect(self.current_worker.deleteLater)
        self.worker_thread.finished.connect(self.worker_thread.deleteLater)

        # Start the thread
        self.worker_thread.start()
        self.status_bar.showMessage(f"🔄 Analyzing {len(self.comparison_tickers)} comparison tickers...")

    def handle_comparison_data_ready(self, comparison_results):
        """Processes comparison analysis results on the main thread."""
        self.comparison_data = comparison_results
        self.update_charts(chart_only=['comparison'])
        self.set_buttons_enabled(True)
        self.status_bar.showMessage(f"✅ Comparison analysis complete.")
    
    def set_buttons_enabled(self, enabled):
        """Enables/disables action buttons based on analysis status."""
        self.analyze_btn.setEnabled(enabled)
        self.export_btn.setEnabled(enabled and self.pivot is not None)
        self.pdf_btn.setEnabled(enabled and self.pivot is not None)
        self.ml_btn.setEnabled(enabled and self.pivot is not None)
        self.stats_btn.setEnabled(enabled and self.pivot is not None)
        self.add_compare_btn.setEnabled(enabled)
        # Ticker selection controls remain enabled for configuration changes
        self.ticker_combo.setEnabled(enabled and not self.custom_ticker_check.isChecked())
        self.custom_ticker_check.setEnabled(enabled)


    # --- Visualization Methods ---
    def get_plot_style(self):
        """Returns the appropriate colors for Matplotlib based on the dark mode state."""
        is_dark = self.dark_mode_check.isChecked()
        return {
            'fig_bg': '#34495e' if is_dark else 'white',
            'ax_bg': '#34495e' if is_dark else '#f7f7f7',
            'fg': '#ecf0f1' if is_dark else '#2c3e50',
            'grid_color': '#5a778e' if is_dark else '#ccc',
        }

    def update_charts(self, chart_only=None, force_redraw=False):
        """Update selected or all visualization tabs."""
        if self.pivot is None: return

        style = self.get_plot_style()
        self.logger.info(f"Updating charts. Dark mode: {self.dark_mode_check.isChecked()}")
        
        charts_to_update = chart_only or ['bar', 'heatmap', 'scatter', 'comparison']

        if 'bar' in charts_to_update:
            self.update_bar_chart(style, force_redraw)
        if 'heatmap' in charts_to_update:
            self.update_heatmap(style, force_redraw)
        if 'scatter' in charts_to_update:
            self.update_scatter_plot(style, force_redraw)
        if 'comparison' in charts_to_update:
            self.update_comparison_chart(style, force_redraw)
            
    def clear_layout(self, layout):
        """Helper to clear all widgets from a layout."""
        if layout is not None:
            while layout.count():
                child = layout.takeAt(0)
                if child.widget() is not None:
                    child.widget().deleteLater()
                elif child.layout() is not None:
                    self.clear_layout(child.layout())

    def update_bar_chart(self, style, force_redraw):
        """Update the bar chart visualization."""
        
        # Clear existing plot, preserving controls
        layout = self.bar_frame.layout()
        if layout.count() > 1: # Index 1 onwards are charts
            for i in range(1, layout.count()):
                item = layout.itemAt(i)
                if item:
                    if item.widget(): item.widget().deleteLater()
                    if item.layout(): self.clear_layout(item.layout())

        # Create Figure and Canvas
        fig, ax = plt.subplots(figsize=(10, 4), facecolor=style['fig_bg'])
        ax.set_facecolor(style['ax_bg'])

        metric_data = self.month_avg if self.bar_metric_combo.currentText() == "Mean" else self.month_median
        metric_label = self.bar_metric_combo.currentText()
        
        bars = ax.bar(range(1, 13), metric_data*100, alpha=0.8, color=style['fg'], edgecolor=style['fg'], linewidth=1)
        
        # Styling
        ax.tick_params(colors=style['fg'])
        ax.yaxis.label.set_color(style['fg'])
        ax.title.set_color(style['fg'])
        ax.set_xticks(range(1, 13))
        ax.set_xticklabels(self.months, fontsize=9, color=style['fg'])
        ax.set_ylabel(f'{metric_label} Monthly Return (%)', fontsize=9, fontweight='bold')
        ax.set_title(f'{metric_label} Monthly Returns for {self.current_ticker}', fontsize=10, fontweight='bold', pad=15)
        ax.grid(axis='y', alpha=0.3, linestyle='--', color=style['grid_color'])
        ax.axhline(y=0, color=style['fg'], linestyle='-', linewidth=0.8, alpha=0.5)

        for bar in bars:
            height = bar.get_height()
            if height < 0:
                bar.set_color('#e74c3c') # Red for negative
                bar.set_edgecolor('#c0392b')
            else:
                bar.set_color('#27ae60') # Green for positive
                bar.set_edgecolor('#229954')
            ax.text(bar.get_x() + bar.get_width()/2., height + (0.05 if height >= 0 else -0.2),
                    f'{height:.1f}%', ha='center', va='bottom' if height >= 0 else 'top',
                    fontsize=6, fontweight='bold', color=style['fg'])
        
        if self.toggle_benchmark_btn.isChecked():
            overall_avg_pct = self.overall_avg * 100
            ax.axhline(y=overall_avg_pct, color='#f1c40f', linestyle='--', linewidth=2,
                       label=f'Overall Average: {overall_avg_pct:.2f}%', alpha=0.8)
            ax.legend(loc='upper right', framealpha=0.9, facecolor=style['ax_bg'], labelcolor=style['fg'])

        canvas = FigureCanvas(fig)
        layout.addWidget(canvas)
        canvas.draw()


    def update_heatmap(self, style, force_redraw):
        """Update the year-month heatmap visualization."""
        self.clear_layout(self.heatmap_frame.layout())
        
        fig, ax = plt.subplots(figsize=(14, 8), facecolor=style['fig_bg'])
        ax.set_facecolor(style['ax_bg'])
        
        sns.heatmap(self.pivot*100, center=0, cmap='RdYlGn', cbar_kws={'label':'Monthly Return (%)'},
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
        
        canvas = FigureCanvas(fig)
        self.heatmap_frame.layout().addWidget(canvas)
        canvas.draw()


    def update_scatter_plot(self, style, force_redraw):
        """Update the risk vs return scatter plot."""
        self.clear_layout(self.scatter_frame.layout())
        
        monthly_stats = pd.DataFrame({
            'month': range(1, 13),
            'avg_return': self.month_avg * 100,
            'std_dev': self.pivot.std() * 100,
            'positive_rate': (self.pivot > 0).sum() / self.pivot.count() * 100
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
        
        # Styling
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

        if self.toggle_benchmark_btn.isChecked():
            overall_avg_pct = self.overall_avg * 100
            ax.axhline(y=overall_avg_pct, color='#f1c40f', linestyle='--', linewidth=2, alpha=0.8,
                       label=f'Overall Avg Return: {overall_avg_pct:.2f}%')
            ax.legend(loc='upper right', framealpha=0.9, facecolor=style['ax_bg'], labelcolor=style['fg'])
            
        canvas = FigureCanvas(fig)
        self.scatter_frame.layout().addWidget(canvas)
        canvas.draw()


    def update_comparison_chart(self, style, force_redraw):
        """Update the comparison chart with multi-ticker data."""
        self.clear_layout(self.comparison_frame.layout())

        if not self.comparison_data:
            return

        fig, ax = plt.subplots(figsize=(12, 6), facecolor=style['fig_bg'])
        ax.set_facecolor(style['ax_bg'])
        
        line_styles = ['-', '--', ':', '-.']
        markers = ['o', 's', '^', 'D']
        
        # Plot all tickers in comparison_data
        for i, (ticker, data) in enumerate(self.comparison_data.items()):
            month_avg_pct = data['month_avg'] * 100
            ax.plot(range(1, 13), month_avg_pct, label=ticker, 
                    linewidth=2, marker=markers[i % len(markers)], linestyle=line_styles[i % len(line_styles)])
        
        # Styling
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

        canvas = FigureCanvas(fig)
        self.comparison_frame.layout().addWidget(canvas)
        canvas.draw()

    # --- Output and Utility Methods ---
    def update_summary(self):
        """Update the summary tab with analysis results."""
        if self.pivot is None: return
        
        summary = f"📊 MONTHLY RETURN ANALYSIS SUMMARY\n"
        summary += f"{'='*50}\n"
        summary += f"Index/ETF: {self.current_ticker}\n"
        summary += f"Analysis Period: {self.start_year_spin.value()} to {self.end_date_entry.text()[:4]} ({len(self.pivot)} years)\n"
        summary += f"Total Months Analyzed: {len(self.monthly_ret)}\n"
        summary += f"{'='*50}\n"
        summary += "📈 MONTHLY AVERAGE RETURNS:\n"
        summary += "-" * 30 + "\n"
        for month_num, month_name in enumerate(self.months, 1):
            if month_num in self.month_avg.index:
                avg_ret = self.month_avg[month_num] * 100
                summary += f"{month_name:3}: {avg_ret:+6.2f}%\n"
        summary += f"\n{'='*50}\n"
        overall_avg_pct = self.overall_avg * 100
        summary += f"🎯 OVERALL AVERAGE RETURN: {overall_avg_pct:+6.2f}%\n"
        best_month_idx = self.month_avg.idxmax()
        worst_month_idx = self.month_avg.idxmin()
        best_month_name = self.months[best_month_idx - 1]
        worst_month_name = self.months[worst_month_idx - 1]
        summary += f"🏆 BEST MONTH: {best_month_name} ({self.month_avg[best_month_idx]*100:+.2f}%)\n"
        summary += f"⚠️  WORST MONTH: {worst_month_name} ({self.month_avg[worst_month_idx]*100:+.2f}%)\n"
        
        self.summary_text.setText(summary)
        self.logger.info("Summary tab updated.")

    def export_to_excel(self):
        """Export analysis data to Excel."""
        if self.pivot is None:
            QMessageBox.warning(self, "Warning", "No data to export. Please analyze data first.")
            return

        filename = f"{self.current_ticker.replace('^', '').replace('.JO', '').replace('.SS', '')}_monthly_analysis_{self.start_year_spin.value()}_{self.end_date_entry.text()[:4]}.xlsx"
        filepath, _ = QFileDialog.getSaveFileName(self, "Export to Excel", filename, "Excel files (*.xlsx)")
        
        if not filepath: return

        try:
            with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
                self.pivot.to_excel(writer, sheet_name='Year_Month_Returns')
                summary_stats = pd.DataFrame({
                    'Month': self.months, 'Average_Return_%': self.month_avg.values * 100,
                    'Median_Return_%': self.month_median.values * 100,
                    'Std_Dev_%': self.pivot.std().values * 100,
                })
                summary_stats.to_excel(writer, sheet_name='Monthly_Summary', index=False)
            
            QMessageBox.information(self, "Success", f"✅ Data exported to {filepath}")
            self.status_bar.showMessage(f"💾 Data exported to {filepath}")
        except Exception as e:
            self.logger.exception("Error exporting data to Excel.")
            QMessageBox.critical(self, "Error", f"Error exporting data: {e}")
            self.status_bar.showMessage("❌ Export failed")

    def generate_pdf_report(self):
        """Placeholder for PDF generation (requires reportlab setup not included in this migration)."""
        QMessageBox.information(self, "Info", "PDF Report generation is a complex feature that needs dedicated reportlab integration. Functionality is disabled in this version.")
        
    # --- ML and Stats ---
    def run_statistical_tests(self):
        """Run statistical significance tests."""
        if self.pivot is None:
            QMessageBox.warning(self, "Warning", "No data available. Please analyze data first.")
            return
            
        self.stats_text.clear()
        self.status_bar.showMessage("🧮 Running statistical tests...")
        
        try:
            results = "--- Statistical Test Results ---\n\n"
            
            # 1. Normality Test (Jarque-Bera on monthly returns)
            jb_test = stats.jarque_bera(self.monthly_ret)
            results += f"1. Normality Test (Jarque-Bera):\n"
            results += f"  Statistic: {jb_test.statistic:.4f}\n"
            results += f"  p-value: {jb_test.pvalue:.4f}\n"
            if jb_test.pvalue < 0.05:
                results += "  Conclusion: Monthly returns are likely NOT normally distributed (p < 0.05).\n"
            else:
                results += "  Conclusion: Monthly returns are likely normally distributed (p >= 0.05).\n"
                
            results += "\n--- Annualized Returns Significance ---\n\n"
            
            # 2. T-Test against Zero (Check if mean return is significantly different from zero)
            t_test = stats.ttest_1samp(self.monthly_ret, 0)
            results += f"2. Mean Return Test (T-Test vs 0):\n"
            results += f"  Mean Monthly Return: {self.overall_avg * 100:.4f}%\n"
            results += f"  T-Statistic: {t_test.statistic:.4f}\n"
            results += f"  p-value: {t_test.pvalue:.4f}\n"
            if t_test.pvalue < 0.05:
                results += "  Conclusion: The mean return is statistically SIGNIFICANTLY different from zero (p < 0.05).\n"
            else:
                results += "  Conclusion: The mean return is NOT statistically different from zero (p >= 0.05).\n"

            self.stats_text.setText(results)
            self.status_bar.showMessage("🧮 Statistical tests completed")
        except Exception as e:
            self.logger.exception("Error running statistical tests.")
            QMessageBox.critical(self, "Error", f"Error running statistical tests: {e}")
            self.status_bar.showMessage("❌ Statistical tests failed")

    def run_ml_analysis(self):
        """Initiates ML analysis in a QThread."""
        if self.pivot is None:
            QMessageBox.warning(self, "Warning", "No data available. Please analyze data first.")
            return

        settings = self.validate_inputs()
        if settings is None: return

        self.ml_text.clear()
        self.clear_layout(self.ml_plot_frame.layout())
        self.set_buttons_enabled(False)
        
        # Create worker and thread
        self.worker_thread = QThread()
        self.current_worker = DataAnalysisWorker(settings, self.logger, self.cache_dir, self.VERSION)
        self.current_worker.moveToThread(self.worker_thread)

        # Connect signals/slots
        self.worker_thread.started.connect(lambda: self.current_worker.run_ml_analysis(self.monthly_ret, self.pivot))
        self.current_worker.data_ready.connect(self.handle_ml_data_ready)
        self.current_worker.progress.connect(self.status_bar.showMessage)
        self.current_worker.error.connect(lambda e: QMessageBox.critical(self, "ML Analysis Error", e))
        self.current_worker.finished.connect(self.worker_thread.quit)
        self.current_worker.finished.connect(self.current_worker.deleteLater)
        self.worker_thread.finished.connect(self.worker_thread.deleteLater)

        # Start the thread
        self.worker_thread.start()
        self.status_bar.showMessage("🤖 Starting ML Analysis...")

    def handle_ml_data_ready(self, results):
        """Processes ML analysis results on the main thread."""
        self.ml_results = results
        self.set_buttons_enabled(True)
        self.update_ml_tab()
        self.status_bar.showMessage("🤖 Advanced ML analysis completed")

    def update_ml_tab(self):
        """Updates the ML tab content and plot based on results."""
        if not self.ml_results: return
        
        style = self.get_plot_style()
        
        # --- Text Summary ---
        forecast_date_str = self.ml_results['forecast_date'].strftime('%Y-%m')
        forecast_mean_pct = self.ml_results['forecast_mean'] * 100
        pca_variance = self.ml_results['pca_explained_variance']
        
        summary = "--- ML Analysis Summary ---\n\n"
        summary += f"🔮 Forecast for {forecast_date_str}:\n"
        summary += f"  Predicted Mean Monthly Return: {forecast_mean_pct:+.4f}%\n"
        summary += f"  (Based on {'AutoARIMA' if PMDARIMA_AVAILABLE else 'ARIMA(5,1,0)'})\n\n"
        
        summary += f"📊 PCA Components:\n"
        summary += f"  PC1 Explains: {pca_variance[0]*100:.2f}%\n"
        summary += f"  PC2 Explains: {pca_variance[1]*100:.2f}%\n"
        summary += "  (The plot below visualizes the months using these two main components)\n\n"
        
        summary += f"🧩 KMeans Clustering (k=3):\n"
        summary += "  The 12 months are grouped into 3 clusters based on their historical yearly returns.\n"
        summary += "  Cluster 1, 2, 3 details are visualized in the plot.\n\n"
        
        summary += f"🚨 Isolation Forest Anomaly Detection:\n"
        anomalies = np.where(self.ml_results['anom_pred'] == -1)[0]
        if len(anomalies) > 0:
            anomaly_months = [self.months[i] for i in anomalies]
            summary += f"  Detected Anomalous Months (10% contamination): {', '.join(anomaly_months)}\n"
        else:
            summary += "  No anomalies detected in monthly return patterns.\n"
        
        self.ml_text.setText(summary)
        
        # --- Plot Visualization (PCA + Clustering) ---
        self.clear_layout(self.ml_plot_frame.layout())
        
        fig, ax = plt.subplots(figsize=(10, 6), facecolor=style['fig_bg'])
        ax.set_facecolor(style['ax_bg'])
        
        features_reduced = self.ml_results['features_reduced']
        clusters = self.ml_results['clusters']
        anom_pred = self.ml_results['anom_pred']
        
        cols = ['#3498db', '#e74c3c', '#2ecc71'] # Blue, Red, Green
        
        for cl in np.unique(clusters):
            idx = np.where(clusters == cl)
            # Filter for non-anomalies first
            norm_idx = np.intersect1d(idx, np.where(anom_pred != -1))
            
            ax.scatter(features_reduced[norm_idx, 0], features_reduced[norm_idx, 1],
                       s=150, alpha=0.7, color=cols[cl], label=f'Cluster {cl+1}')
                       
            # Annotate non-anomalies
            for i in norm_idx:
                ax.annotate(self.months[i], (features_reduced[i, 0], features_reduced[i, 1]),
                           xytext=(3, 3), textcoords='offset points', fontsize=9, fontweight='bold',
                           bbox=dict(boxstyle="round,pad=0.2", fc=style['fig_bg'], alpha=0.7, ec=style['fg']),
                           color=style['fg'])

        # Plot Anomalies
        anom_idx = np.where(anom_pred == -1)
        if np.any(anom_idx):
            ax.scatter(features_reduced[anom_idx, 0], features_reduced[anom_idx, 1],
                       s=200, marker='X', c='#f1c40f', label='Anomaly', linewidths=2, edgecolors=style['fg'])
            for i in anom_idx[0]:
                 ax.annotate(self.months[i], (features_reduced[i, 0], features_reduced[i, 1]),
                           xytext=(3, 3), textcoords='offset points', fontsize=9, fontweight='bold',
                           bbox=dict(boxstyle="round,pad=0.2", fc=style['fig_bg'], alpha=0.7, ec=style['fg']),
                           color=style['fg'])
                           
        # Styling
        ax.tick_params(colors=style['fg'])
        ax.yaxis.label.set_color(style['fg'])
        ax.xaxis.label.set_color(style['fg'])
        ax.title.set_color(style['fg'])
        ax.set_xlabel('PC1', color=style['fg']); ax.set_ylabel('PC2', color=style['fg'])
        ax.set_title(f'Monthly Clusters – PCA space for {self.current_ticker}', color=style['fg'])
        ax.grid(True, alpha=0.3, color=style['grid_color']); 
        ax.legend(loc='upper right', framealpha=0.9, facecolor=style['ax_bg'], labelcolor=style['fg'])

        canvas = FigureCanvas(fig)
        self.ml_plot_frame.setLayout(QVBoxLayout())
        self.ml_plot_frame.layout().addWidget(canvas)
        canvas.draw()


if __name__ == "__main__":
    # Ensure Matplotlib uses the QtAgg backend
    matplotlib.use('QtAgg') 
    
    app = QApplication(sys.argv)
    # Set a default style/theme for better appearance on all systems
    app.setStyle("Fusion") 

    main_window = JSEAnalyzer()
    main_window.show()
    sys.exit(app.exec())

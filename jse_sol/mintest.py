# jse.py
# Global Index Monthly Return Analyzer (Version 2.9.12)
# Enhanced ML analysis with PCA, GMM, Isolation Forest, cluster visualization,
# plain-English summary, upcoming month forecast, and comprehensive logging.
# v2.9.12: Fixed custom ticker UI placement and chart refresh issues.

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib
import os
import pickle
import threading
import json
from scipy import stats
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.animation as animation
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
import matplotlib.patches as patches
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import IsolationForest
from statsmodels.tsa.arima.model import ARIMA
import warnings
import re
import concurrent.futures
import logging
import time

# Try importing pmdarima, with fallback if it fails
try:
    from pmdarima import auto_arima
    PMDARIMA_AVAILABLE = True
except (ImportError, ValueError) as e:
    PMDARIMA_AVAILABLE = False
    logging.warning(f"Failed to import pmdarima ({str(e)}). Using fixed-order ARIMA for forecasting.")

warnings.filterwarnings('ignore')
matplotlib.use('TkAgg')

# --- Setup Logging ---
def setup_logging():
    """Configures the root logger to write to a file."""
    log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('jse_analyzer.log', mode='w')
    file_handler.setFormatter(log_formatter)
    
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Remove existing handlers to avoid duplicate logs in console
    if logger.hasHandlers():
        logger.handlers.clear()
        
    logger.addHandler(file_handler)
    return logger

class Tooltip:
    """Simple tooltip class for Tkinter widgets."""
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tooltip = None
        self.widget.bind("<Enter>", self.show_tooltip)
        self.widget.bind("<Leave>", self.hide_tooltip)

    def show_tooltip(self, event=None):
        x, y, _, _ = self.widget.bbox("insert")
        x += self.widget.winfo_rootx() + 25
        y += self.widget.winfo_rooty() + 25
        self.tooltip = tk.Toplevel(self.widget)
        self.tooltip.wm_overrideredirect(True)
        self.tooltip.wm_geometry(f"+{x}+{y}")
        label = tk.Label(self.tooltip, text=self.text, background="#ffffe0", relief="solid", borderwidth=1, padx=5, pady=3)
        label.pack()

    def hide_tooltip(self, event=None):
        if self.tooltip:
            self.tooltip.destroy()
            self.tooltip = None

class JSEAnalyzer:
    """A GUI application for analyzing monthly returns of global financial indices."""
    VERSION = "2.9.12"

    def __init__(self):
        self.logger = setup_logging()
        self.logger.info(f"--- Application Start: Global Index Analyzer v{self.VERSION} ---")

        self.root = tk.Tk()
        self.root.title(f"📊 Global Index Monthly Return Analyzer (v{self.VERSION})")
        self.root.geometry("1300x850")
        self.root.configure(bg='#f0f0f0')
        
        # Cache directory with versioned prefix
        self.cache_dir = f"cache_v{self.VERSION.replace('.', '_')}"
        os.makedirs(self.cache_dir, exist_ok=True)
        self.logger.info(f"Cache directory set to: {self.cache_dir}")

        # Default values and ticker options
        self.ticker_options = {
            "🇿🇦 JSE All Share (^J203.JO)": "^J203.JO",
            "🇿🇦 JSE Financials (^J258.JO)": "^J258.JO",
            "🇿🇦 JSE Industrials (^J252.JO)": "^J252.JO",
            "🇿🇦 JSE Resources (^J250.JO)": "^J250.JO",
            "🇦🇺 ASX 200 (^AXJO)": "^AXJO",
            "🇺🇸 S&P 500 (^GSPC)": "^GSPC",
            "🇬🇧 FTSE 100 (^FTSE)": "^FTSE",
            "🇩🇪 DAX (^GDAXI)": "^GDAXI",
            "🇯🇵 Nikkei 225 (^N225)": "^N225",
            "🇭🇰 Hang Seng (^HSI)": "^HSI",
            "🇨🇳 Shanghai Composite (000001.SS)": "000001.SS",
            "🌍 MSCI World (^MXWO)": "^MXWO",
            "🌍 MSCI Emerging Markets (^MXEF)": "^MXEF"
        }
        self.ticker = "^J203.JO"
        self.start_year = 1990
        self.end_date = datetime.today().strftime("%Y-%m-%d")
        
        # Initialize data attributes
        self.data = None
        self.monthly = None
        self.monthly_ret = None
        self.df = None
        self.pivot = None
        self.month_avg = None
        self.month_median = None
        self.months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        self.show_benchmark = tk.BooleanVar(value=True)
        self.dark_mode = tk.BooleanVar(value=False)
        self.comparison_tickers = []
        self.comparison_data = {}
        self.bar_metric = tk.StringVar(value="Mean")
        
        # Data ready flag to prevent premature chart updates
        self.data_ready = False

        # Style configuration
        self.style = ttk.Style()
        self.style.theme_use('clam')
        self.configure_styles()
        self.setup_ui()
        self.logger.info("Application initialized successfully.")

    # --- UI Configuration ---
    def configure_styles(self):
        """Configure custom styles for the GUI."""
        self.style.configure('Title.TLabel', font=('Arial', 12, 'bold'), foreground='#2c3e50')
        self.style.configure('Header.TFrame', background='#3498db')
        self.style.configure('Config.TLabelframe', background='#f8f9fa', foreground='#2c3e50')
        self.style.configure('Config.TLabelframe.Label', font=('Arial', 10, 'bold'), foreground='#2c3e50')
        self.style.configure('Primary.TButton', font=('Arial', 9, 'bold'), background='#3498db', foreground='white')
        self.style.map('Primary.TButton', background=[('active', '#2980b9')])
        self.style.configure('Secondary.TButton', font=('Arial', 9), background='#95a5a6', foreground='white')
        self.style.map('Secondary.TButton', background=[('active', '#7f8c8d')])
        self.style.configure('Success.TButton', font=('Arial', 9, 'bold'), background='#27ae60', foreground='white')
        self.style.map('Success.TButton', background=[('active', '#229954')])
        self.style.configure('Warning.TButton', font=('Arial', 9, 'bold'), background='#f39c12', foreground='white')
        self.style.map('Warning.TButton', background=[('active', '#d35400')])
        self.logger.info("UI styles configured.")

    def setup_ui(self):
        """Set up the main GUI components with improved layout and tooltips."""
        # Main container
        main_container = tk.Frame(self.root, bg='#f0f0f0')
        main_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # Header with title
        header_frame = ttk.Frame(main_container, style='Header.TFrame')
        header_frame.pack(fill=tk.X, pady=(0, 20))
        title_label = ttk.Label(header_frame, text=f"📈 Global Index Monthly Return Analyzer (v{self.VERSION})",
                               style='Title.TLabel')
        title_label.pack(side=tk.LEFT, padx=10, pady=10)
        dark_mode_check = ttk.Checkbutton(header_frame, text="🌙 Dark Mode",
                                         variable=self.dark_mode,
                                         command=self.toggle_dark_mode)
        dark_mode_check.pack(side=tk.RIGHT, padx=10, pady=10)
        Tooltip(dark_mode_check, "Toggle dark mode for better visibility in low-light environments.")

        # --- Collapsible Configuration Panel ---
        # Toggle button frame (always visible)
        self.config_toggle_frame = ttk.Frame(main_container)
        self.config_toggle_frame.pack(fill=tk.X, pady=(0, 5))

        self.config_visible = tk.BooleanVar(value=True)
        self.toggle_config_btn = ttk.Button(self.config_toggle_frame, text="❌ Hide Configuration",
                                           command=self.toggle_config_panel, style='Secondary.TButton')
        self.toggle_config_btn.pack(side=tk.LEFT)
        Tooltip(self.toggle_config_btn, "Show/hide the configuration panel.")

        # Content frame (what gets shown/hidden)
        self.config_content_frame = ttk.Frame(main_container)
        self.config_content_frame.pack(fill=tk.X, pady=(0, 20))

        # Configuration frame (visual grouping)
        config_frame = ttk.LabelFrame(self.config_content_frame, text="⚙️ Configuration Panel",
                                     style='Config.TLabelframe', padding="5")
        config_frame.pack(fill=tk.BOTH, expand=True)

        # Ticker selection row
        ticker_row = ttk.Frame(config_frame)
        ticker_row.pack(fill=tk.X, pady=(0, 15))
        ticker_label = ttk.Label(ticker_row, text="Index/ETF:")
        ticker_label.pack(side=tk.LEFT, padx=(0, 10))
        Tooltip(ticker_label, "Select a predefined index or ETF for analysis.")
        self.ticker_var = tk.StringVar(value="🇿🇦 JSE All Share (^J203.JO)")
        ticker_combo = ttk.Combobox(ticker_row, textvariable=self.ticker_var,
                                   values=list(self.ticker_options.keys()),
                                   state="readonly", width=35, font=('Arial', 9))
        ticker_combo.pack(side=tk.LEFT, padx=(0, 15))

        # Custom ticker controls
        self.use_custom_var = tk.BooleanVar()
        custom_check = ttk.Checkbutton(ticker_row, text="Custom Ticker",
                                      variable=self.use_custom_var,
                                      command=self.toggle_custom_ticker)
        custom_check.pack(side=tk.LEFT, padx=(0, 10))
        Tooltip(custom_check, "Enable to enter a custom ticker symbol not in the list.")
        self.custom_ticker_var = tk.StringVar()
        self.custom_ticker_entry = ttk.Entry(ticker_row, textvariable=self.custom_ticker_var,
                                            width=15, font=('Arial', 9))
        # FIX: Pack and then hide using pack_forget for consistency
        self.custom_ticker_entry.pack(side=tk.LEFT, padx=(0, 15))
        self.custom_ticker_entry.pack_forget()

        # Comparison ticker selection
        compare_label = ttk.Label(ticker_row, text="Compare With:")
        compare_label.pack(side=tk.LEFT, padx=(15, 10))
        Tooltip(compare_label, "Select an index to compare with the primary ticker.")
        self.compare_var = tk.StringVar(value="")
        compare_combo = ttk.Combobox(ticker_row, textvariable=self.compare_var,
                                    values=list(self.ticker_options.keys()),
                                    state="readonly", width=35, font=('Arial', 9))
        compare_combo.pack(side=tk.LEFT, padx=(0, 15))
        add_compare_btn = ttk.Button(ticker_row, text="➕ Add Comparison",
                                    command=self.add_comparison_ticker, style='Secondary.TButton')
        add_compare_btn.pack(side=tk.LEFT)
        Tooltip(add_compare_btn, "Add the selected ticker for comparative analysis.")

        # Date controls row
        date_row = ttk.Frame(config_frame)
        date_row.pack(fill=tk.X, pady=(0, 15))
        date_range_label = ttk.Label(date_row, text="Date Range:")
        date_range_label.pack(side=tk.LEFT, padx=(0, 10))
        Tooltip(date_range_label, "Choose a predefined date range or custom dates.")
        self.date_range_var = tk.StringVar(value="Custom")
        date_range_combo = ttk.Combobox(date_row, textvariable=self.date_range_var,
                                       values=["Custom", "Last 5 Years", "Last 10 Years",
                                              "Last 20 Years", "All Data"],
                                       state="readonly", width=15, font=('Arial', 9))
        date_range_combo.pack(side=tk.LEFT, padx=(0, 15))
        date_range_combo.bind('<<ComboboxSelected>>', self.on_date_range_change)

        start_year_label = ttk.Label(date_row, text="Start Year:")
        start_year_label.pack(side=tk.LEFT, padx=(15, 10))
        Tooltip(start_year_label, "Enter the starting year for custom date range.")
        self.start_year_var = tk.IntVar(value=self.start_year)
        self.start_year_entry = ttk.Entry(date_row, textvariable=self.start_year_var,
                                         width=8, font=('Arial', 9))
        self.start_year_entry.pack(side=tk.LEFT, padx=(0, 15))

        end_date_label = ttk.Label(date_row, text="End Date:")
        end_date_label.pack(side=tk.LEFT, padx=(15, 10))
        Tooltip(end_date_label, "Enter the end date in YYYY-MM-DD format for custom range.")
        self.end_date_var = tk.StringVar(value=self.end_date)
        self.end_date_entry = ttk.Entry(date_row, textvariable=self.end_date_var,
                                       width=12, font=('Arial', 9))
        self.end_date_entry.pack(side=tk.LEFT, padx=(0, 15))

        # Action buttons row
        button_row = ttk.Frame(config_frame)
        button_row.pack(fill=tk.X, pady=(15, 0))
        analyze_btn = ttk.Button(button_row, text="🔍 Analyze Data",
                                command=self.analyze_data, style='Primary.TButton')
        analyze_btn.pack(side=tk.LEFT, padx=(0, 10))
        Tooltip(analyze_btn, "Start the analysis with the selected parameters.")

        self.export_btn = ttk.Button(button_row, text="💾 Export to Excel",
                                    command=self.export_to_excel, state=tk.DISABLED,
                                    style='Success.TButton')
        self.export_btn.pack(side=tk.LEFT, padx=(0, 10))
        Tooltip(self.export_btn, "Export the analysis results to an Excel file.")

        pdf_btn = ttk.Button(button_row, text="📄 Export PDF",
                            command=self.generate_pdf_report, style='Warning.TButton')
        pdf_btn.pack(side=tk.LEFT, padx=(0, 10))
        Tooltip(pdf_btn, "Generate and export a PDF report of the analysis.")

        ml_btn = ttk.Button(button_row, text="🤖 ML Analysis",
                           command=self.run_ml_analysis, style='Secondary.TButton')
        ml_btn.pack(side=tk.LEFT, padx=(0, 10))
        Tooltip(ml_btn, "Perform advanced machine learning analysis with cluster visualization and forecasting.")

        stats_btn = ttk.Button(button_row, text="🧮 Significance Test",
                              command=self.run_statistical_tests, style='Secondary.TButton')
        stats_btn.pack(side=tk.LEFT)
        Tooltip(stats_btn, "Run statistical significance tests on the data.")

        # Results notebook
        self.notebook_frame = ttk.Frame(main_container)
        self.notebook_frame.pack(fill=tk.BOTH, expand=True)

        self.notebook = ttk.Notebook(self.notebook_frame, padding=5)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # Tabs
        self.bar_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.bar_frame, text="📊 Average Returns")
        
        # FIX: Store reference to control frame to avoid destroying it
        self.bar_control_frame = ttk.Frame(self.bar_frame)
        self.bar_control_frame.pack(fill=tk.X, pady=(5, 0))
        
        bar_metric_label = ttk.Label(self.bar_control_frame, text="Metric:")
        bar_metric_label.pack(side=tk.LEFT, padx=(5, 5))
        Tooltip(bar_metric_label, "Choose between mean or median for the bar chart.")
        
        # Disable metric combo initially until data is ready
        self.metric_combo = ttk.Combobox(self.bar_control_frame, textvariable=self.bar_metric,
                                        values=["Mean", "Median"], state="disabled", width=10, font=('Arial', 9))
        self.metric_combo.pack(side=tk.LEFT, padx=(0, 5))
        self.metric_combo.bind('<<ComboboxSelected>>', lambda event: self.update_charts())

        # Toggle benchmark button in control frame
        self.benchmark_toggle_btn = ttk.Button(self.bar_control_frame, text="📊 Toggle Benchmark",
                                             command=self.toggle_benchmark_line,
                                             style='Secondary.TButton')
        self.benchmark_toggle_btn.pack(side=tk.LEFT, padx=(5, 0))
        Tooltip(self.benchmark_toggle_btn, "Toggle the benchmark line in the bar chart.")

        self.heatmap_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.heatmap_frame, text="🌡️ Year-Month Heatmap")

        self.scatter_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.scatter_frame, text="⚖️ Risk vs Return")

        self.summary_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.summary_frame, text="📋 Summary Statistics")
        self.summary_text = tk.Text(self.summary_frame, wrap=tk.WORD, font=('Consolas', 10),
                                   bg='#ffffff', fg='#2c3e50')
        self.summary_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        scrollbar = ttk.Scrollbar(self.summary_frame, orient=tk.VERTICAL, command=self.summary_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.summary_text.configure(yscrollcommand=scrollbar.set)

        self.comparison_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.comparison_frame, text="🔄 Comparison")

        self.stats_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.stats_frame, text="🧮 Statistical Tests")

        self.ml_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.ml_frame, text="🤖 ML Analysis")

        # Status bar
        self.status_var = tk.StringVar(value="✨ Ready - Select an index and click Analyze")
        status_bar = ttk.Label(main_container, textvariable=self.status_var,
                              relief=tk.SUNKEN, padding=5, font=('Arial', 9),
                              background='#ecf0f1', foreground='#7f8c8d')
        status_bar.pack(fill=tk.X, pady=(10, 0))

        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        self.logger.info("UI setup complete.")

    # --- UI Handlers ---
    def toggle_config_panel(self):
        """Show or hide the configuration panel to maximize graph space."""
        if self.config_visible.get():
            # Currently visible, so hide it
            self.config_content_frame.pack_forget()
            self.toggle_config_btn.config(text="⚙️ Show Configuration")
            self.config_visible.set(False)
            self.logger.info("Configuration panel hidden.")
            self.status_var.set("✅ Configuration panel hidden")
        else:
            # Currently hidden, so show it - pack it before notebook_frame to maintain order
            self.config_content_frame.pack(before=self.notebook_frame, fill=tk.X, pady=(0, 20))
            self.toggle_config_btn.config(text="❌ Hide Configuration")
            self.config_visible.set(True)
            self.logger.info("Configuration panel shown.")
            self.status_var.set("✅ Configuration panel shown")
        
        # Force layout update to prevent visual glitches
        self.root.update()

    def toggle_dark_mode(self):
        """Toggle between light and dark mode with improved color application."""
        is_dark = self.dark_mode.get()
        self.logger.info(f"Toggling dark mode to {'ON' if is_dark else 'OFF'}.")
        if is_dark:
            self.root.configure(bg='#2c3e50')
            self.summary_text.configure(bg='#34495e', fg='#ecf0f1')
            self.style.configure('Config.TLabelframe', background='#34495e', foreground='#ecf0f1')
            self.style.configure('Config.TLabelframe.Label', foreground='#ecf0f1')
            self.style.configure('Title.TLabel', foreground='#ecf0f1')
            self.style.configure('Header.TFrame', background='#1f618d')
        else:
            self.root.configure(bg='#f0f0f0')
            self.summary_text.configure(bg='#ffffff', fg='#2c3e50')
            self.style.configure('Config.TLabelframe', background='#f8f9fa', foreground='#2c3e50')
            self.style.configure('Config.TLabelframe.Label', foreground='#2c3e50')
            self.style.configure('Title.TLabel', foreground='#2c3e50')
            self.style.configure('Header.TFrame', background='#3498db')

    def toggle_custom_ticker(self):
        """Toggle between predefined ticker dropdown and custom ticker entry."""
        if self.use_custom_var.get():
            self.logger.info("Switched to custom ticker input.")
            # FIX: Show entry next to checkbox using pack
            self.custom_ticker_entry.pack(side=tk.LEFT, padx=(0, 15))
            self.custom_ticker_entry.delete(0, tk.END)
        else:
            self.logger.info("Switched to predefined ticker dropdown.")
            # FIX: Hide entry using pack_forget
            self.custom_ticker_entry.pack_forget()

    def on_date_range_change(self, event=None):
        """Update date inputs based on selected range."""
        selection = self.date_range_var.get()
        self.logger.info(f"Date range changed to: {selection}")
        current_year = datetime.now().year
        if selection == "Last 5 Years":
            self.start_year_var.set(current_year - 5)
            self.end_date_var.set(datetime.today().strftime("%Y-%m-%d"))
            self.start_year_entry.config(state=tk.DISABLED)
            self.end_date_entry.config(state=tk.DISABLED)
        elif selection == "Last 10 Years":
            self.start_year_var.set(current_year - 10)
            self.end_date_var.set(datetime.today().strftime("%Y-%m-%d"))
            self.start_year_entry.config(state=tk.DISABLED)
            self.end_date_entry.config(state=tk.DISABLED)
        elif selection == "Last 20 Years":
            self.start_year_var.set(current_year - 20)
            self.end_date_var.set(datetime.today().strftime("%Y-%m-%d"))
            self.start_year_entry.config(state=tk.DISABLED)
            self.end_date_entry.config(state=tk.DISABLED)
        elif selection == "All Data":
            self.start_year_var.set(1990)
            self.end_date_var.set(datetime.today().strftime("%Y-%m-%d"))
            self.start_year_entry.config(state=tk.DISABLED)
            self.end_date_entry.config(state=tk.DISABLED)
        else: # Custom
            self.start_year_entry.config(state=tk.NORMAL)
            self.end_date_entry.config(state=tk.NORMAL)

    def add_comparison_ticker(self):
        """Add a ticker to the comparison list."""
        ticker_name = self.compare_var.get()
        if not ticker_name:
            self.logger.warning("Add comparison ticker attempted with no selection.")
            messagebox.showwarning("Warning", "Please select a ticker to compare.")
            return
        
        ticker = self.ticker_options.get(ticker_name)
        if ticker and ticker not in self.comparison_tickers and ticker != self.ticker:
            self.comparison_tickers.append(ticker)
            self.logger.info(f"Added '{ticker_name}' ({ticker}) to comparison list.")
            self.status_var.set(f"✅ Added {ticker_name} to comparison")
            if self.data_ready:
                self.analyze_comparison_data()
        elif ticker == self.ticker:
            self.logger.warning(f"Attempted to add primary ticker '{ticker}' for comparison.")
            messagebox.showwarning("Warning", "Cannot compare the same ticker.")
        elif ticker in self.comparison_tickers:
            self.logger.warning(f"Attempted to add duplicate comparison ticker '{ticker}'.")
            messagebox.showwarning("Warning", "Ticker already added for comparison.")

    # --- Data Handling ---
    def validate_inputs(self):
        """Validate user inputs before analysis with clear error messages."""
        self.logger.info("Validating user inputs.")
        try:
            start_year = self.start_year_var.get()
            if start_year < 1900 or start_year > datetime.now().year:
                raise ValueError(f"Start year must be between 1900 and {datetime.now().year}.")

            end_date = self.end_date_var.get()
            if not re.match(r"\d{4}-\d{2}-\d{2}", end_date):
                raise ValueError("End date must be in YYYY-MM-DD format (e.g., 2023-12-31).")
            datetime.strptime(end_date, "%Y-%m-%d")

            if self.use_custom_var.get():
                ticker = self.custom_ticker_var.get().strip()
                if not ticker:
                    raise ValueError("Custom ticker cannot be empty.")
                if not re.match(r"^[A-Za-z0-9^.-]+$", ticker):
                    raise ValueError("Invalid ticker format. Tickers can contain letters, numbers, ^, ., and -.")
                self.ticker = ticker
            else:
                ticker_name = self.ticker_var.get()
                self.ticker = self.ticker_options.get(ticker_name, "^J203.JO")

            self.logger.info(f"Input validation successful. Ticker: {self.ticker}, Start: {start_year}, End: {end_date}")
            return True
        except ValueError as e:
            self.logger.error(f"Input validation failed: {str(e)}")
            messagebox.showerror("Input Validation Error", str(e))
            self.status_var.set(f"❌ Input validation failed: {str(e)}")
            return False

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
        """Load data from cache if available and recent."""
        cache_file = self.get_cache_filename(ticker, start_year, end_date)
        if os.path.exists(cache_file):
            if datetime.now() - datetime.fromtimestamp(os.path.getmtime(cache_file)) < timedelta(days=1):
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

    def analyze_data_async(self):
        """Run data analysis in background thread with improved error handling."""
        def worker():
            try:
                if not self.validate_inputs():
                    return

                self.status_var.set("📥 Downloading data...")
                self.root.update()
                self.logger.info(f"Starting data download for ticker: {self.ticker}")

                start_year = self.start_year_var.get()
                end_date = self.end_date_var.get()

                # Download data with retry logic
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        self.data = yf.download(
                            self.ticker,
                            start=f"{start_year}-01-01",
                            end=end_date,
                            progress=False,
                            auto_adjust=False
                        )
                        if not self.data.empty:
                            break
                    except Exception as e:
                        if attempt == max_retries - 1:
                            raise
                        self.status_var.set(f"⚠️ Network error (attempt {attempt+1}/{max_retries})")
                        self.logger.warning(f"Network error on attempt {attempt+1}/{max_retries} for {self.ticker}: {str(e)}")
                        self.root.update()
                        time.sleep(2)

                self.status_var.set("⚙️ Processing data...")
                self.root.update()
                self.logger.info("Data download complete. Processing data...")

                if self.data is None or self.data.empty:
                    raise ValueError("No data returned from Yahoo Finance. Check ticker symbol or internet connection.")

                price_col = self.get_price_column(self.data)
                self.monthly = self.data[price_col].resample('ME').last()
                self.monthly_ret = self.monthly.pct_change().dropna()

                if isinstance(self.monthly_ret, pd.DataFrame):
                    if self.monthly_ret.empty:
                        raise ValueError("Monthly returns calculation resulted in an empty DataFrame.")
                    self.monthly_ret = self.monthly_ret.iloc[:, 0]

                if self.monthly_ret.empty:
                    raise ValueError("Monthly returns are empty after processing.")

                self.monthly_ret.name = 'ret'
                self.df = self.monthly_ret.to_frame()
                self.df['year'] = self.df.index.year
                self.df['month'] = self.df.index.month
                self.pivot = self.df.pivot_table(index='year', columns='month', values='ret')
                self.month_avg = self.pivot.mean().sort_index()
                self.month_median = self.pivot.median().sort_index()
                self.overall_avg = self.monthly_ret.mean()
                
                # Set data ready flag and enable controls
                self.data_ready = True
                self.logger.info("Data processing complete. Updating UI.")

                # Enable metric combo
                self.root.after(0, lambda: self.metric_combo.config(state="readonly"))
                
                # Update UI on main thread
                self.root.after(0, self.update_charts)
                self.root.after(0, self.update_summary)
                self.root.after(0, lambda: self.export_btn.config(state=tk.NORMAL))
                self.root.after(0, lambda: self.status_var.set(f"✅ Analysis complete for {self.ticker} ({start_year}-{end_date[:4]})"))

            except Exception as e:
                self.logger.exception("An error occurred during data analysis.")
                # FIX: Capture exception message as default argument to avoid free variable error
                self.root.after(0, lambda msg=str(e): messagebox.showerror("Analysis Error", msg))
                self.root.after(0, lambda msg=str(e): self.status_var.set(f"❌ Analysis failed: {msg}"))

        thread = threading.Thread(target=worker)
        thread.daemon = True
        thread.start()

    def analyze_data(self):
        """Wrapper for async analysis."""
        self.logger.info("'Analyze Data' button clicked.")
        # Reset data ready flag
        self.data_ready = False
        self.metric_combo.config(state="disabled")
        self.analyze_data_async()

    def analyze_comparison_data(self):
        """Analyze data for comparison tickers with parallel downloads."""
        self.logger.info("Starting analysis for comparison tickers.")
        self.comparison_data = {}
        comparison_tickers = self.comparison_tickers.copy()
        if self.ticker not in comparison_tickers:
            comparison_tickers.append(self.ticker)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(self.download_and_process_ticker, ticker, self.start_year_var.get(), self.end_date_var.get()): ticker
                for ticker in comparison_tickers
            }
            for future in concurrent.futures.as_completed(futures):
                ticker = futures[future]
                try:
                    result = future.result()
                    if result is not None:
                        self.comparison_data[ticker] = result
                        self.logger.info(f"Successfully processed comparison data for {ticker}.")
                except Exception as e:
                    self.logger.error(f"Error processing comparison ticker {ticker}: {str(e)}")
                    # FIX: Capture exception message as default argument to avoid free variable error
                    self.root.after(0, lambda msg=str(e): self.status_var.set(f"❌ Error processing {ticker}: {msg}"))

        if self.data_ready:
            self.update_comparison_chart()

    def download_and_process_ticker(self, ticker, start_year, end_date):
        """Download and process data for a single ticker."""
        try:
            cached_data = self.load_cached_data(ticker, start_year, end_date)
            if cached_data is not None:
                data = cached_data
            else:
                self.logger.info(f"No cache found for {ticker}. Downloading from yfinance.")
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
                self.logger.warning(f"No data returned for ticker {ticker}.")
                return None

            price_col = self.get_price_column(data)
            monthly = data[price_col].resample('ME').last()
            monthly_ret = monthly.pct_change().dropna()

            if isinstance(monthly_ret, pd.DataFrame):
                monthly_ret = monthly_ret.iloc[:, 0]
            if monthly_ret.empty:
                return None

            df = monthly_ret.to_frame()
            df['year'] = df.index.year
            df['month'] = df.index.month
            pivot = df.pivot_table(index='year', columns='month', values=monthly_ret.name)
            month_avg = pivot.mean().sort_index()
            return {'month_avg': month_avg, 'pivot': pivot}
        except Exception as e:
            self.logger.error(f"Failed to download/process ticker {ticker}: {e}")
            return None

    # --- Visualization Methods ---
    def update_charts(self):
        """Update all visualization tabs only if data is ready."""
        if not self.data_ready or self.pivot is None:
            self.logger.warning("update_charts() called before data is ready. Skipping.")
            return
        
        self.logger.info("Updating all charts.")
        self.update_bar_chart()
        self.update_heatmap()
        self.update_scatter_plot()
        self.update_comparison_chart()
        self.update_summary()

    def update_bar_chart(self):
        """Update the bar chart with proper metrics and benchmark."""
        if not self.data_ready or self.month_avg is None or self.month_median is None:
            return
            
        # FIX: Clear existing chart frames but keep control frame
        for widget in self.bar_frame.winfo_children():
            if widget != self.bar_control_frame:  # Don't destroy the control frame
                widget.destroy()
        
        # Create new chart frame
        chart_frame = ttk.Frame(self.bar_frame)
        chart_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        fig1, ax1 = plt.subplots(figsize=(10, 4))
        metric_data = self.month_avg if self.bar_metric.get() == "Mean" else self.month_median
        metric_label = "Average" if self.bar_metric.get() == "Mean" else "Median"
        self.logger.info(f"Updating bar chart with metric: {metric_label}")

        bars = ax1.bar(range(1, 13), metric_data*100, alpha=0.8, color='#3498db', edgecolor='#2980b9', linewidth=1)
        ax1.set_xticks(range(1, 13))
        ax1.set_xticklabels(self.months, fontsize=9)
        ax1.set_ylabel(f'{metric_label} Monthly Return (%)', fontsize=9, fontweight='bold')
        ax1.set_title(f'{metric_label} Monthly Returns for {self.ticker}\nPeriod: {self.start_year_var.get()} to {self.end_date_var.get()[:4]}',
                     fontsize=10, fontweight='bold', pad=15)
        ax1.grid(axis='y', alpha=0.3, linestyle='--')
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)

        if self.show_benchmark.get():
            overall_avg_pct = self.overall_avg * 100
            ax1.axhline(y=overall_avg_pct, color='#e74c3c', linestyle='--', linewidth=2,
                       label=f'Overall Average: {overall_avg_pct:.2f}%', alpha=0.8)
            ax1.legend(loc='upper right', framealpha=0.9)

        ax1.axhline(y=0, color='#2c3e50', linestyle='-', linewidth=0.8, alpha=0.5)

        for bar in bars:
            height = bar.get_height()
            if height < 0:
                bar.set_color('#e74c3c')
                bar.set_edgecolor('#c0392b')
            ax1.text(bar.get_x() + bar.get_width()/2., height + (0.05 if height >= 0 else -0.2),
                    f'{height:.1f}%', ha='center', va='bottom' if height >= 0 else 'top',
                    fontsize=6, fontweight='bold')
        
        canvas1 = FigureCanvasTkAgg(fig1, chart_frame)
        canvas1.draw()
        canvas1.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.status_var.set(f"📊 Showing {metric_label} Returns for {self.ticker}")

    def update_heatmap(self):
        """Update the year-month heatmap visualization."""
        if not self.data_ready or self.pivot is None:
            return
            
        self.logger.info("Updating heatmap.")
        for widget in self.heatmap_frame.winfo_children():
            widget.destroy()

        heatmap_frame = ttk.Frame(self.heatmap_frame)
        heatmap_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        fig2, ax2 = plt.subplots(figsize=(14, 8))
        sns.heatmap(self.pivot*100, center=0, cmap='RdYlGn', cbar_kws={'label':'Monthly Return (%)'},
                    linewidths=.5, ax=ax2, annot=True, fmt='.1f', cbar=True)
        ax2.set_xlabel('Month', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Year', fontsize=11, fontweight='bold')
        ax2.set_title(f'Month-by-Year Returns for {self.ticker}\nPeriod: {self.start_year_var.get()} to {self.end_date_var.get()[:4]}',
                     fontsize=12, fontweight='bold', pad=20)
        ax2.set_xticks(np.arange(12) + 0.5)
        ax2.set_xticklabels(self.months, rotation=0, fontsize=9)
        ax2.set_yticklabels(ax2.get_yticklabels(), rotation=0, fontsize=8)
        cbar = ax2.collections[0].colorbar
        cbar.set_label('Monthly Return (%)', fontsize=10, fontweight='bold')
        canvas2 = FigureCanvasTkAgg(fig2, heatmap_frame)
        canvas2.draw()
        canvas2.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def update_scatter_plot(self):
        """Update the risk vs return scatter plot."""
        if not self.data_ready or self.pivot is None:
            return
            
        self.logger.info("Updating risk vs return scatter plot.")
        for widget in self.scatter_frame.winfo_children():
            widget.destroy()

        scatter_chart_frame = ttk.Frame(self.scatter_frame)
        scatter_chart_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        monthly_stats = pd.DataFrame({
            'month': range(1, 13),
            'avg_return': self.month_avg * 100,
            'std_dev': self.pivot.std() * 100,
            'positive_rate': (self.pivot > 0).sum() / self.pivot.count() * 100
        })
        fig3, ax3 = plt.subplots(figsize=(12, 8))
        scatter = ax3.scatter(monthly_stats['std_dev'], monthly_stats['avg_return'],
                             c=monthly_stats['positive_rate'],
                             cmap='RdYlGn', s=250, alpha=0.8, edgecolors='black', linewidth=1)
        for i, month in enumerate(self.months):
            ax3.annotate(month, (monthly_stats['std_dev'].iloc[i], monthly_stats['avg_return'].iloc[i]),
                        xytext=(5, 5), textcoords='offset points', fontsize=9, fontweight='bold',
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))
        ax3.set_xlabel('Monthly Return Standard Deviation (%)', fontsize=11, fontweight='bold')
        ax3.set_ylabel('Average Monthly Return (%)', fontsize=11, fontweight='bold')
        ax3.set_title(f'Risk vs Return by Month for {self.ticker}\n(Color = Positive Return Rate)\nPeriod: {self.start_year_var.get()} to {self.end_date_var.get()[:4]}',
                     fontsize=12, fontweight='bold', pad=20)
        ax3.grid(True, alpha=0.3, linestyle='--')
        plt.colorbar(scatter, ax=ax3, label='Positive Return Rate (%)')
        if self.show_benchmark.get():
            overall_avg_pct = self.overall_avg * 100
            ax3.axhline(y=overall_avg_pct, color='#e74c3c', linestyle='--', linewidth=2, alpha=0.8,
                       label=f'Overall Avg Return: {overall_avg_pct:.2f}%')
            ax3.legend(loc='upper right', framealpha=0.9)
        canvas3 = FigureCanvasTkAgg(fig3, scatter_chart_frame)
        canvas3.draw()
        canvas3.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def update_comparison_chart(self):
        """Update the comparison tab with multi-ticker data."""
        if not self.data_ready or self.pivot is None:
            return
            
        self.logger.info("Updating comparison chart.")
        for widget in self.comparison_frame.winfo_children():
            widget.destroy()

        if not self.comparison_data or self.pivot is None:
            self.logger.info("No comparison data to display.")
            return

        fig, ax = plt.subplots(figsize=(12, 6))
        x = range(1, 13)
        ax.plot(x, self.month_avg*100, label=self.ticker, linewidth=2, marker='o')
        for ticker, data in self.comparison_data.items():
            if ticker == self.ticker:
                continue
            ax.plot(x, data['month_avg']*100, label=ticker, linewidth=2, marker='o', linestyle='--')
        ax.set_xticks(range(1, 13))
        ax.set_xticklabels(self.months, fontsize=10)
        ax.set_ylabel('Average Monthly Return (%)', fontsize=11, fontweight='bold')
        ax.set_title(f'Comparative Monthly Returns\nPeriod: {self.start_year_var.get()} to {self.end_date_var.get()[:4]}',
                     fontsize=12, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='upper right', framealpha=0.9)
        ax.axhline(y=0, color='#2c3e50', linestyle='-', linewidth=0.8, alpha=0.5)
        canvas = FigureCanvasTkAgg(fig, self.comparison_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # --- Output Methods ---
    def update_summary(self):
        """Update the summary tab with analysis results."""
        if not self.data_ready or self.pivot is None:
            return
            
        self.logger.info("Updating summary tab.")
        self.summary_text.delete(1.0, tk.END)
        
        summary = f"📊 MONTHLY RETURN ANALYSIS SUMMARY\n"
        summary += f"{'='*50}\n"
        summary += f"Index/ETF: {self.ticker}\n"
        summary += f"Analysis Period: {self.start_year_var.get()} to {self.end_date_var.get()[:4]} ({len(self.pivot)} years)\n"
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
        self.summary_text.insert(1.0, summary)
        self.logger.info("Summary tab updated.")

    def export_to_excel(self):
        """Export analysis data to Excel with improved error handling."""
        self.logger.info("'Export to Excel' button clicked.")
        if not self.data_ready or self.pivot is None:
            self.logger.warning("Export to Excel attempted with no data.")
            messagebox.showwarning("Warning", "No data to export. Please analyze data first.")
            return

        try:
            filename = f"{self.ticker.replace('^', '').replace('.JO', '').replace('.SS', '').replace('.AX', '')}_monthly_analysis_{self.start_year_var.get()}_{self.end_date_var.get()[:4]}.xlsx"
            filepath = filedialog.asksaveasfilename(defaultextension=".xlsx", initialfile=filename,
                                                      filetypes=[("Excel files", "*.xlsx")])
            if not filepath:
                self.logger.info("Excel export cancelled by user.")
                return

            with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
                self.pivot.to_excel(writer, sheet_name='Year_Month_Returns')
                summary_stats = pd.DataFrame({
                    'Month': self.months, 'Average_Return_%': self.month_avg.values * 100,
                    'Median_Return_%': self.month_median.values * 100,
                    'Std_Dev_%': self.pivot.std().values * 100,
                })
                summary_stats.to_excel(writer, sheet_name='Monthly_Summary', index=False)
            
            messagebox.showinfo("Success", f"✅ Data exported to {filepath}")
            self.status_var.set(f"💾 Data exported to {filepath}")
            self.logger.info(f"Data successfully exported to {filepath}")
        except Exception as e:
            self.logger.exception("Error exporting data to Excel.")
            messagebox.showerror("Error", f"Error exporting data: {e}")
            self.status_var.set("❌ Export failed")

    def generate_pdf_report(self):
        """Generate professional PDF report with improved error handling."""
        self.logger.info("'Export PDF' button clicked.")
        if not self.data_ready or self.pivot is None:
            self.logger.warning("PDF export attempted with no data.")
            messagebox.showwarning("Warning", "No data to export. Please analyze data first.")
            return

        try:
            filename = filedialog.asksaveasfilename(defaultextension=".pdf", filetypes=[("PDF files", "*.pdf")])
            if not filename:
                self.logger.info("PDF export cancelled by user.")
                return

            doc = SimpleDocTemplate(filename, pagesize=A4)
            story = []
            styles = getSampleStyleSheet()
            title = Paragraph(f"Monthly Return Analysis: {self.ticker}", styles['h1'])
            story.append(title)
            story.append(Paragraph(f"Analysis Period: {self.start_year_var.get()} to {self.end_date_var.get()[:4]}", styles['Normal']))
            story.append(Paragraph(f"Overall Average Return: {self.overall_avg*100:.2f}%", styles['Normal']))
            # Add summary table
            summary_data = [['Month', 'Average Return (%)']]
            for i, month in enumerate(self.months):
                if i+1 in self.month_avg.index:
                    summary_data.append([month, f"{self.month_avg[i+1]*100:.2f}"])
            summary_table = Table(summary_data)
            summary_table.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,0), colors.grey),
                ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
                ('ALIGN', (0,0), (-1,-1), 'CENTER'),
                ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
                ('BOTTOMPADDING', (0,0), (-1,0), 12),
                ('BACKGROUND', (0,1), (-1,-1), colors.beige),
            ]))
            story.append(summary_table)
            
            doc.build(story)
            messagebox.showinfo("Success", f"✅ PDF report generated: {filename}")
            self.status_var.set(f"📄 PDF report generated: {filename}")
            self.logger.info(f"PDF report successfully generated: {filename}")
        except Exception as e:
            self.logger.exception("Error generating PDF report.")
            messagebox.showerror("Error", f"Error generating PDF: {e}")
            self.status_var.set("❌ PDF generation failed")

    # --- Analysis Methods ---
    def toggle_benchmark_line(self):
        """Toggle the benchmark line on/off."""
        self.show_benchmark.set(not self.show_benchmark.get())
        status = "ON" if self.show_benchmark.get() else "OFF"
        self.logger.info(f"Benchmark line toggled to {status}.")
        if self.data_ready:
            self.update_charts()
        self.status_var.set(f"📊 Benchmark line is now {status}")

    def run_statistical_tests(self):
        """Run statistical significance tests with threading and progress updates."""
        self.logger.info("'Significance Test' button clicked.")
        if not self.data_ready or self.pivot is None:
            self.logger.warning("Statistical test run attempted with no data.")
            messagebox.showwarning("Warning", "No data available. Please analyze data first.")
            return

        def worker():
            try:
                self.status_var.set("🧮 Running statistical tests...")
                self.root.update()
                
                for widget in self.stats_frame.winfo_children():
                    widget.destroy()

                stats_text = tk.Text(self.stats_frame, wrap=tk.WORD, font=('Consolas', 10))
                stats_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
                
                results = "📊 STATISTICAL SIGNIFICANCE TESTS\n"
                results += "="*50 + "\n\n"
                
                # Test 1: ANOVA - Are monthly returns significantly different?
                self.status_var.set("🧮 Running ANOVA test...")
                self.root.update()
                month_returns = [self.pivot[col].dropna() for col in self.pivot.columns]
                f_stat, p_value = stats.f_oneway(*month_returns)
                results += f"1. ANOVA TEST (F-statistic: {f_stat:.4f})\n"
                results += f"   H0: All months have same mean return\n"
                results += f"   p-value: {p_value:.6f}\n"
                results += f"   Result: {'REJECT H0' if p_value < 0.05 else 'FAIL TO REJECT H0'}\n"
                results += f"   Interpretation: {'Months ARE significantly different' if p_value < 0.05 else 'NO significant difference between months'}\n\n"
                
                # Test 2: T-tests for each month vs overall mean
                results += "2. ONE-SAMPLE T-TESTS (Each month vs Overall Mean)\n"
                results += "-" * 50 + "\n"
                overall_mean = self.monthly_ret.mean()
                for month_num, month_name in enumerate(self.months, 1):
                    if month_num in self.month_avg.index:
                        month_data = self.pivot[month_num].dropna()
                        if len(month_data) > 1:
                            t_stat, p_val = stats.ttest_1samp(month_data, overall_mean)
                            significance = "SIGNIFICANT" if p_val < 0.05 else "NOT SIGNIFICANT"
                            results += f"   {month_name}: t={t_stat:.3f}, p={p_val:.4f} ({significance})\n"
                
                # Test 3: Normality test
                results += f"\n3. NORMALITY TEST (Shapiro-Wilk)\n"
                results += "-" * 50 + "\n"
                sample_data = self.monthly_ret.sample(min(5000, len(self.monthly_ret)))  # Sample for speed
                shapiro_stat, shapiro_p = stats.shapiro(sample_data)
                results += f"   Statistic: {shapiro_stat:.4f}\n"
                results += f"   p-value: {shapiro_p:.6f}\n"
                results += f"   Result: {'Normal' if shapiro_p > 0.05 else 'NOT Normal'} distribution\n"
                
                # Test 4: Seasonality detection
                results += f"\n4. SEASONALITY CHECK\n"
                results += "-" * 50 + "\n"
                best_month = self.months[self.month_avg.idxmax() - 1]
                worst_month = self.months[self.month_avg.idxmin() - 1]
                results += f"   Best month: {best_month} ({self.month_avg.max()*100:.2f}%)\n"
                results += f"   Worst month: {worst_month} ({self.month_avg.min()*100:.2f}%)\n"
                results += f"   Seasonality effect: {abs(self.month_avg.max() - self.month_avg.min())*100:.2f}% difference\n"
                
                stats_text.insert(1.0, results)
                self.logger.info("Statistical tests completed successfully.")
                self.status_var.set("✅ Statistical tests completed")
                
            except Exception as e:
                self.logger.exception("Error running statistical tests.")
                self.root.after(0, lambda msg=str(e): messagebox.showerror("Error", f"Statistical tests failed: {msg}"))
                self.root.after(0, lambda msg=str(e): self.status_var.set(f"❌ Statistical tests failed: {msg}"))

        # Run in separate thread
        thread = threading.Thread(target=worker)
        thread.daemon = True
        thread.start()

    def run_ml_analysis(self):
        """Run advanced machine learning analysis with threading and progress updates."""
        self.logger.info("'ML Analysis' button clicked.")
        if not self.data_ready or self.pivot is None:
            self.logger.warning("ML analysis run attempted with no data.")
            messagebox.showwarning("Warning", "No data available. Please analyze data first.")
            return

        def worker():
            try:
                self.status_var.set("🤖 Preparing data for ML analysis...")
                self.root.update()
                
                for widget in self.ml_frame.winfo_children():
                    widget.destroy()

                # Prepare data
                returns = self.monthly_ret.dropna()
                if len(returns) < 24:  # Need at least 2 years of data
                    raise ValueError("Insufficient data for ML analysis (need at least 24 months)")

                # Create feature matrix: use rolling window of 12 months to predict next month
                self.status_var.set("🤖 Creating features...")
                self.root.update()
                
                # Reshape data for ML models
                X = []
                y_months = []
                y_returns = []
                
                # Use 12-month sequences
                for i in range(12, len(returns)):
                    X.append(returns.iloc[i-12:i].values)
                    y_months.append(returns.index[i].month)
                    y_returns.append(returns.iloc[i])
                
                X = np.array(X)
                y_months = np.array(y_months)
                y_returns = np.array(y_returns)
                
                if len(X) < 10:
                    raise ValueError("Insufficient sequences for clustering")

                # Tab container for ML results
                ml_notebook = ttk.Notebook(self.ml_frame)
                ml_notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

                # --- 1. PCA Analysis Tab ---
                pca_tab = ttk.Frame(ml_notebook)
                ml_notebook.add(pca_tab, text="📉 PCA Clustering")
                
                self.status_var.set("🤖 Running PCA...")
                self.root.update()
                
                # PCA for dimensionality reduction and visualization
                pca = PCA(n_components=2)
                X_pca = pca.fit_transform(X)
                
                fig_pca, ax_pca = plt.subplots(figsize=(10, 6))
                # FIX: Changed from invalid 'tab12' to valid 'Set3' colormap
                scatter = ax_pca.scatter(X_pca[:, 0], X_pca[:, 1], c=y_months, cmap='Set3', alpha=0.6)
                ax_pca.set_xlabel(f'First Principal Component ({pca.explained_variance_ratio_[0]:.1%} variance)')
                ax_pca.set_ylabel(f'Second Principal Component ({pca.explained_variance_ratio_[1]:.1%} variance)')
                ax_pca.set_title(f'PCA: Returns Patterns Colored by Target Month\nTotal Explained Variance: {pca.explained_variance_ratio_.sum():.1%}')
                # FIX: Update colorbar to use valid colormap
                plt.colorbar(scatter, ax=ax_pca, ticks=range(1, 13)).set_ticklabels(self.months)
                
                canvas_pca = FigureCanvasTkAgg(fig_pca, pca_tab)
                canvas_pca.draw()
                canvas_pca.get_tk_widget().pack(fill=tk.BOTH, expand=True)
                
                # --- 2. GMM Clustering Tab ---
                gmm_tab = ttk.Frame(ml_notebook)
                ml_notebook.add(gmm_tab, text="🎲 Gaussian Mixture")
                
                self.status_var.set("🤖 Running Gaussian Mixture Model...")
                self.root.update()
                
                # GMM clustering
                n_clusters = min(4, len(X))
                gmm = GaussianMixture(n_components=n_clusters, random_state=42)
                cluster_labels = gmm.fit_predict(X_pca)
                
                fig_gmm, ax_gmm = plt.subplots(figsize=(10, 6))
                scatter = ax_gmm.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, cmap='Set3', alpha=0.7)
                ax_gmm.set_xlabel(f'First Principal Component ({pca.explained_variance_ratio_[0]:.1%} variance)')
                ax_gmm.set_ylabel(f'Second Principal Component ({pca.explained_variance_ratio_[1]:.1%} variance)')
                ax_gmm.set_title(f'Gaussian Mixture Clustering (k={n_clusters})\nAIC: {gmm.aic(X_pca):.0f}, BIC: {gmm.bic(X_pca):.0f}')
                plt.colorbar(scatter, ax=ax_gmm)
                
                canvas_gmm = FigureCanvasTkAgg(fig_gmm, gmm_tab)
                canvas_gmm.draw()
                canvas_gmm.get_tk_widget().pack(fill=tk.BOTH, expand=True)
                
                # --- 3. Anomaly Detection Tab ---
                anomaly_tab = ttk.Frame(ml_notebook)
                ml_notebook.add(anomaly_tab, text="🔍 Anomaly Detection")
                
                self.status_var.set("🤖 Running Isolation Forest...")
                self.root.update()
                
                # Isolation Forest for anomaly detection
                iso_forest = IsolationForest(contamination=0.05, random_state=42)
                anomaly_labels = iso_forest.fit_predict(X_pca)
                
                fig_anomaly, ax_anomaly = plt.subplots(figsize=(10, 6))
                normal_mask = anomaly_labels == 1
                anomaly_mask = anomaly_labels == -1
                
                ax_anomaly.scatter(X_pca[normal_mask, 0], X_pca[normal_mask, 1], c='blue', alpha=0.6, label='Normal')
                ax_anomaly.scatter(X_pca[anomaly_mask, 0], X_pca[anomaly_mask, 1], c='red', alpha=0.9, label='Anomaly')
                ax_anomaly.set_xlabel(f'First Principal Component ({pca.explained_variance_ratio_[0]:.1%} variance)')
                ax_anomaly.set_ylabel(f'Second Principal Component ({pca.explained_variance_ratio_[1]:.1%} variance)')
                ax_anomaly.set_title(f'Isolation Forest Anomaly Detection\n{anomaly_mask.sum()} outliers detected ({100*anomaly_mask.sum()/len(X_pca):.1f}%)')
                ax_anomaly.legend()
                
                canvas_anomaly = FigureCanvasTkAgg(fig_anomaly, anomaly_tab)
                canvas_anomaly.draw()
                canvas_anomaly.get_tk_widget().pack(fill=tk.BOTH, expand=True)
                
                # --- 4. Summary Statistics Tab ---
                summary_tab = ttk.Frame(ml_notebook)
                ml_notebook.add(summary_tab, text="📊 ML Summary")
                
                summary_text = tk.Text(summary_tab, wrap=tk.WORD, font=('Consolas', 10))
                summary_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
                
                ml_summary = "🤖 MACHINE LEARNING ANALYSIS SUMMARY\n"
                ml_summary += "="*50 + "\n\n"
                ml_summary += f"📈 Data Points: {len(X)} sequences of 12 months\n"
                ml_summary += f"📊 PCA Explained Variance: {pca.explained_variance_ratio_[0]:.1%} + {pca.explained_variance_ratio_[1]:.1%}\n\n"
                ml_summary += "🔍 FINDINGS:\n"
                ml_summary += "-" * 30 + "\n"
                ml_summary += f"• {n_clusters} distinct return patterns identified by GMM\n"
                ml_summary += f"• {anomaly_mask.sum()} anomalous periods detected ({100*anomaly_mask.sum()/len(X_pca):.1f}%)\n"
                ml_summary += f"• PCA shows {pca.explained_variance_ratio_.sum():.1%} variance in first 2 components\n"
                ml_summary += f"• Suggests returns have {'strong' if pca.explained_variance_ratio_[0] > 0.5 else 'moderate'} seasonal structure\n"
                
                summary_text.insert(1.0, ml_summary)
                
                self.logger.info("ML analysis completed successfully.")
                self.status_var.set("✅ ML analysis completed")
                
            except Exception as e:
                self.logger.exception("Error running ML analysis.")
                self.root.after(0, lambda msg=str(e): messagebox.showerror("Error", f"ML analysis failed: {msg}"))
                self.root.after(0, lambda msg=str(e): self.status_var.set(f"❌ ML analysis failed: {msg}"))

        # Run in separate thread
        thread = threading.Thread(target=worker)
        thread.daemon = True
        thread.start()

    def run(self):
        """Start the application."""
        self.logger.info("Starting Tkinter main loop.")
        self.root.mainloop()
        self.logger.info("--- Application Shutdown ---")

if __name__ == "__main__":
    app = JSEAnalyzer()
    app.run()

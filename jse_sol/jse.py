# jse2.py
# Global Index Monthly Return Analyzer (Version 2.8.1)
# Enhanced ML analysis with PCA, GMM, Isolation Forest, cluster visualization,
# plain-English summary, and upcoming month forecast with fallback for pmdarima errors
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
# Try importing pmdarima, with fallback if it fails
try:
    from pmdarima import auto_arima
    PMDARIMA_AVAILABLE = True
except (ImportError, ValueError) as e:
    PMDARIMA_AVAILABLE = False
    print(f"Warning: Failed to import pmdarima ({str(e)}). Using fixed-order ARIMA for forecasting.")
warnings.filterwarnings('ignore')
matplotlib.use('TkAgg')

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
    VERSION = "2.8.1"

    def __init__(self):
        self.root = tk.Tk()
        self.root.title(f"📊 Global Index Monthly Return Analyzer (v{self.VERSION})")
        self.root.geometry("1300x850")
        self.root.configure(bg='#f0f0f0')
        # Cache directory with versioned prefix
        self.cache_dir = f"cache_v{self.VERSION.replace('.', '_')}"
        os.makedirs(self.cache_dir, exist_ok=True)

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
        self.bar_metric = tk.StringVar(value="Mean")  # Control for mean/median toggle

        # Style configuration
        self.style = ttk.Style()
        self.style.theme_use('clam')
        self.configure_styles()
        self.setup_ui()

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

        # Configuration frame
        config_frame = ttk.LabelFrame(main_container, text="⚙️ Configuration Panel",
                                     style='Config.TLabelframe', padding="20")
        config_frame.pack(fill=tk.X, pady=(0, 20))

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
        self.custom_ticker_entry.pack(side=tk.LEFT)
        self.custom_ticker_entry.grid_remove()

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

        self.toggle_benchmark_btn = ttk.Button(button_row, text="📊 Toggle Benchmark",
                                              command=self.toggle_benchmark_line,
                                              style='Secondary.TButton')
        self.toggle_benchmark_btn.pack(side=tk.LEFT, padx=(0, 10))
        Tooltip(self.toggle_benchmark_btn, "Toggle the overall average benchmark line on charts.")

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
        notebook_frame = ttk.Frame(main_container)
        notebook_frame.pack(fill=tk.BOTH, expand=True)
        self.notebook = ttk.Notebook(notebook_frame, padding=5)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # Tabs
        self.bar_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.bar_frame, text="📊 Average Returns")
        bar_control_frame = ttk.Frame(self.bar_frame)
        bar_control_frame.pack(fill=tk.X, pady=(5, 0))
        bar_metric_label = ttk.Label(bar_control_frame, text="Metric:")
        bar_metric_label.pack(side=tk.LEFT, padx=(5, 5))
        Tooltip(bar_metric_label, "Choose between mean or median for the bar chart.")
        metric_combo = ttk.Combobox(bar_control_frame, textvariable=self.bar_metric,
                                   values=["Mean", "Median"], state="readonly", width=10, font=('Arial', 9))
        metric_combo.pack(side=tk.LEFT, padx=(0, 5))
        metric_combo.bind('<<ComboboxSelected>>', lambda event: self.update_charts())

        self.toggle_benchmark_btn = ttk.Button(bar_control_frame, text="📊 Toggle Benchmark",
                                             command=self.toggle_benchmark_line,
                                             style='Secondary.TButton')
        self.toggle_benchmark_btn.pack(side=tk.LEFT, padx=(5, 0))
        Tooltip(self.toggle_benchmark_btn, "Toggle the benchmark line in the bar chart.")

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

    # --- UI Handlers ---
    def toggle_dark_mode(self):
        """Toggle between light and dark mode with improved color application."""
        if self.dark_mode.get():
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
            self.custom_ticker_entry.pack(side=tk.LEFT, padx=(0, 15))
            self.custom_ticker_entry.delete(0, tk.END)
        else:
            self.custom_ticker_entry.pack_forget()

    def on_date_range_change(self, event=None):
        """Update date inputs based on selected range."""
        selection = self.date_range_var.get()
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
        else:
            self.start_year_entry.config(state=tk.NORMAL)
            self.end_date_entry.config(state=tk.NORMAL)

    def add_comparison_ticker(self):
        """Add a ticker to the comparison list."""
        ticker_name = self.compare_var.get()
        if not ticker_name:
            messagebox.showwarning("Warning", "Please select a ticker to compare.")
            return
        ticker = self.ticker_options.get(ticker_name)
        if ticker and ticker not in self.comparison_tickers and ticker != self.ticker:
            self.comparison_tickers.append(ticker)
            self.status_var.set(f"✅ Added {ticker_name} to comparison")
            self.analyze_comparison_data()
        elif ticker == self.ticker:
            messagebox.showwarning("Warning", "Cannot compare the same ticker.")
        elif ticker in self.comparison_tickers:
            messagebox.showwarning("Warning", "Ticker already added for comparison.")

    # --- Data Handling ---
    def validate_inputs(self):
        """Validate user inputs before analysis with clear error messages."""
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

            return True
        except ValueError as e:
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
                        return pickle.load(f)
                except Exception as e:
                    print(f"Error loading cache for {ticker}: {e}")
        return None

    def save_cached_data(self, ticker, start_year, end_date, data):
        """Cache downloaded data locally."""
        cache_file = self.get_cache_filename(ticker, start_year, end_date)
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            print(f"Error saving cache for {ticker}: {e}")

    def analyze_data_async(self):
        """Run data analysis in background thread with improved error handling."""
        def worker():
            try:
                if not self.validate_inputs():
                    return

                self.status_var.set("📥 Downloading data...")
                self.root.update()

                # Validate date range before download
                start_year = self.start_year_var.get()
                end_date = self.end_date_var.get()

                # Check if end_date is valid
                try:
                    datetime.strptime(end_date, "%Y-%m-%d")
                except ValueError:
                    raise ValueError("End date must be in YYYY-MM-DD format (e.g., 2023-12-31).")

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
                        break
                    except Exception as e:
                        if attempt == max_retries - 1:
                            raise
                        self.status_var.set(f"⚠️ Network error (attempt {attempt+1}/{max_retries}): {str(e)}")
                        self.root.update()
                        time.sleep(2)  # Wait before retry

                # Process data
                self.status_var.set("⚙️ Processing data...")
                self.root.update()

                if self.data is None or self.data.empty:
                    raise ValueError("No data returned from Yahoo Finance. Check ticker symbol or internet connection.")

                # Get correct price column
                price_col = self.get_price_column(self.data)

                # Convert to monthly returns
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

                # Update UI
                self.root.after(0, self.update_charts)
                self.root.after(0, self.update_summary)
                self.root.after(0, lambda: self.export_btn.config(state=tk.NORMAL))
                self.root.after(0, lambda: self.status_var.set(f"✅ Analysis complete for {self.ticker} ({start_year}-{end_date[:4]})"))

            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Analysis Error", str(e)))
                self.root.after(0, lambda: self.status_var.set(f"❌ Analysis failed: {str(e)}"))

        thread = threading.Thread(target=worker)
        thread.daemon = True
        thread.start()

    def analyze_data(self):
        """Wrapper for async analysis."""
        self.analyze_data_async()

    def analyze_comparison_data(self):
        """Analyze data for comparison tickers with parallel downloads."""
        self.comparison_data = {}

        # Download all comparison tickers in parallel
        comparison_tickers = self.comparison_tickers.copy()
        comparison_tickers.append(self.ticker)  # Add primary ticker

        # Use thread pool for parallel data download
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
                except Exception as e:
                    self.status_var.set(f"❌ Error processing {ticker}: {str(e)}")

        self.update_comparison_chart()

    def download_and_process_ticker(self, ticker, start_year, end_date):
        """Download and process data for a single ticker."""
        try:
            cached_data = self.load_cached_data(ticker, start_year, end_date)
            if cached_data is not None:
                data = cached_data
            else:
                data = yf.download(
                    ticker,
                    start=f"{start_year}-01-01",
                    end=end_date,
                    progress=False,
                    auto_adjust=False
                )
                self.save_cached_data(ticker, start_year, end_date, data)

            if data is None or data.empty:
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
            return {
                'month_avg': month_avg,
                'pivot': pivot
            }
        except Exception as e:
            self.status_var.set(f"❌ Error processing {ticker}: {str(e)}")
            return None

    # --- Visualization Methods ---
    def update_charts(self):
        """Update all visualization tabs by breaking into smaller methods."""
        self.update_bar_chart()
        self.update_heatmap()
        self.update_scatter_plot()
        self.update_comparison_chart()
        self.update_summary()

    def update_bar_chart(self):
        """Update the bar chart with proper metrics and benchmark."""
        for widget in self.bar_frame.winfo_children():
            if widget != self.bar_frame.winfo_children()[0]:  # Preserve control frame
                widget.destroy()

        chart_frame = ttk.Frame(self.bar_frame)
        chart_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        fig1, ax1 = plt.subplots(figsize=(12, 6))
        metric_data = self.month_avg if self.bar_metric.get() == "Mean" else self.month_median
        metric_label = "Average" if self.bar_metric.get() == "Mean" else "Median"

        bars = ax1.bar(range(1, 13), metric_data*100, alpha=0.8, color='#3498db',
                      edgecolor='#2980b9', linewidth=1)

        ax1.set_xticks(range(1, 13))
        ax1.set_xticklabels(self.months, fontsize=10)
        ax1.set_ylabel(f'{metric_label} Monthly Return (%)', fontsize=11, fontweight='bold')
        ax1.set_title(f'{metric_label} Monthly Returns for {self.ticker}\nPeriod: {self.start_year_var.get()} to {self.end_date_var.get()[:4]}',
                     fontsize=12, fontweight='bold', pad=20)
        ax1.grid(axis='y', alpha=0.3, linestyle='--')
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)

        if self.show_benchmark.get():
            overall_avg_pct = self.overall_avg * 100
            ax1.axhline(y=overall_avg_pct, color='#e74c3c', linestyle='--', linewidth=2,
                       label=f'Overall Average: {overall_avg_pct:.2f}%', alpha=0.8)
            ax1.legend(loc='upper right', framealpha=0.9)

        ax1.axhline(y=0, color='#2c3e50', linestyle='-', linewidth=0.8, alpha=0.5)

        for i, bar in enumerate(bars):
            height = bar.get_height()
            if height < 0:
                bar.set_color('#e74c3c')
                bar.set_edgecolor('#c0392b')

            ax1.text(bar.get_x() + bar.get_width()/2., height + (0.1 if height >= 0 else -0.3),
                    f'{height:.1f}%', ha='center', va='bottom' if height >= 0 else 'top',
                    fontsize=8, fontweight='bold')

        annot1 = ax1.annotate("", xy=(0,0), xytext=(10,10), textcoords="offset points",
                             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=1, alpha=0.9),
                             arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0"),
                             fontsize=9, fontweight='bold')
        annot1.set_visible(False)

        def update_annot_bar(bar, index):
            x = bar.get_x() + bar.get_width() / 2
            y = bar.get_y() + bar.get_height()
            annot1.xy = (x, y)
            month_name = self.months[index]
            value = metric_data.iloc[index] * 100
            text = f"{month_name}\n{value:.2f}%"
            annot1.set_text(text)
            annot1.get_bbox_patch().set_alpha(0.9)

        def hover_bar(event):
            vis = annot1.get_visible()
            if event.inaxes == ax1:
                for i, bar in enumerate(bars):
                    cont, ind = bar.contains(event)
                    if cont:
                        update_annot_bar(bar, i)
                        annot1.set_visible(True)
                        fig1.canvas.draw_idle()
                        return
            if vis:
                annot1.set_visible(False)
                fig1.canvas.draw_idle()

        fig1.canvas.mpl_connect("motion_notify_event", hover_bar)
        canvas1 = FigureCanvasTkAgg(fig1, chart_frame)
        canvas1.draw()
        canvas1.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.status_var.set(f"📊 Showing {metric_label} Returns for {self.ticker}")

    def update_heatmap(self):
        """Update the year-month heatmap visualization."""
        for widget in self.heatmap_frame.winfo_children():
            widget.destroy()

        heatmap_frame = ttk.Frame(self.heatmap_frame)
        heatmap_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        fig2, ax2 = plt.subplots(figsize=(14, 8))
        im = sns.heatmap(self.pivot*100, center=0, cmap='RdYlGn',
                        cbar_kws={'label':'Monthly Return (%)'},
                        linewidths=.5, ax=ax2, annot=True, fmt='.1f',
                        cbar=True)

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
        ax3.spines['top'].set_visible(False)
        ax3.spines['right'].set_visible(False)

        cbar3 = plt.colorbar(scatter, ax=ax3, label='Positive Return Rate (%)')
        cbar3.set_label('Positive Return Rate (%)', fontsize=10, fontweight='bold')

        if self.show_benchmark.get():
            overall_avg_pct = self.overall_avg * 100
            ax3.axhline(y=overall_avg_pct, color='#e74c3c', linestyle='--', linewidth=2, alpha=0.8,
                       label=f'Overall Avg Return: {overall_avg_pct:.2f}%')
            ax3.legend(loc='upper right', framealpha=0.9)

        ax3.axhline(y=0, color='#2c3e50', linestyle='-', linewidth=0.8, alpha=0.5)
        ax3.axvline(x=0, color='#2c3e50', linestyle='-', linewidth=0.8, alpha=0.5)

        annot3 = ax3.annotate("", xy=(0,0), xytext=(10,10), textcoords="offset points",
                             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=1, alpha=0.9),
                             arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0"),
                             fontsize=9)
        annot3.set_visible(False)

        def update_annot_scatter(ind):
            index = ind["ind"][0]
            pos = scatter.get_offsets()[index]
            annot3.xy = pos
            month_name = self.months[index]
            avg_ret = monthly_stats['avg_return'].iloc[index]
            std_dev = monthly_stats['std_dev'].iloc[index]
            pos_rate = monthly_stats['positive_rate'].iloc[index]
            text = f"{month_name}\nReturn: {avg_ret:.2f}%\nRisk: {std_dev:.2f}%\nPos Rate: {pos_rate:.1f}%"
            annot3.set_text(text)
            annot3.get_bbox_patch().set_alpha(0.9)

        def hover_scatter(event):
            vis = annot3.get_visible()
            if event.inaxes == ax3:
                cont, ind = scatter.contains(event)
                if cont:
                    update_annot_scatter(ind)
                    annot3.set_visible(True)
                    fig3.canvas.draw_idle()
                    return
            if vis:
                annot3.set_visible(False)
                fig3.canvas.draw_idle()

        fig3.canvas.mpl_connect("motion_notify_event", hover_scatter)
        canvas3 = FigureCanvasTkAgg(fig3, scatter_chart_frame)
        canvas3.draw()
        canvas3.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def update_comparison_chart(self):
        """Update the comparison tab with multi-ticker data."""
        for widget in self.comparison_frame.winfo_children():
            widget.destroy()

        if not self.comparison_data or self.pivot is None:
            return

        fig, ax = plt.subplots(figsize=(12, 6))
        x = range(1, 13)
        ax.plot(x, self.month_avg*100, label=self.ticker, linewidth=2, marker='o')

        for ticker, data in self.comparison_data.items():
            if ticker == self.ticker:
                continue
            ax.plot(x, data['month_avg']*100, label=ticker, linewidth=2, marker='o')

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
        summary += f"📊 BENCHMARK LINE STATUS: {'ON' if self.show_benchmark.get() else 'OFF'}\n"

        best_month_idx = self.month_avg.idxmax()
        worst_month_idx = self.month_avg.idxmin()
        best_month_name = self.months[best_month_idx - 1]
        worst_month_name = self.months[worst_month_idx - 1]

        summary += f"🏆 BEST MONTH: {best_month_name} ({self.month_avg[best_month_idx]*100:+.2f}%)\n"
        summary += f"⚠️  WORST MONTH: {worst_month_name} ({self.month_avg[worst_month_idx]*100:+.2f}%)\n"
        summary += f"\n📈 ADDITIONAL STATISTICS:\n"
        summary += "-" * 30 + "\n"
        summary += f"Standard Deviation: {self.monthly_ret.std()*100:.2f}%\n"
        summary += f"Sharpe Ratio: {self.overall_avg/self.monthly_ret.std():.2f}\n"
        summary += f"Positive Months: {(self.monthly_ret > 0).sum()}/{len(self.monthly_ret)} ({(self.monthly_ret > 0).sum()/len(self.monthly_ret)*100:.1f}%)\n"

        self.summary_text.insert(1.0, summary)

    def export_to_excel(self):
        """Export analysis data to Excel with improved error handling."""
        if self.pivot is None:
            messagebox.showwarning("Warning", "No data to export. Please analyze data first.")
            return

        try:
            filename = f"{self.ticker.replace('^', '').replace('.JO', '').replace('.SS', '').replace('.AX', '')}_monthly_analysis_{self.start_year_var.get()}_{self.end_date_var.get()[:4]}.xlsx"
            with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                self.pivot.to_excel(writer, sheet_name='Year_Month_Returns')

                summary_stats = pd.DataFrame({
                    'Month': range(1, 13),
                    'Month_Name': self.months,
                    'Average_Return_%': self.month_avg.values * 100,
                    'Median_Return_%': self.month_median.values * 100,
                    'Std_Dev_%': self.pivot.std().values * 100,
                    'Best_Return_%': self.pivot.max().values * 100,
                    'Worst_Return_%': self.pivot.min().values * 100,
                    'Positive_Months_Count': (self.pivot > 0).sum().values,
                    'Total_Months_Count': self.pivot.count().values,
                    'Positive_Rate_%': ((self.pivot > 0).sum() / self.pivot.count() * 100).values
                })
                summary_stats.to_excel(writer, sheet_name='Monthly_Summary', index=False)

                raw_data = pd.DataFrame({
                    'Date': self.pivot.index,
                    'Year': self.pivot.index,
                    **{f'M{col}': self.pivot[col].values for col in self.pivot.columns}
                })
                raw_data.to_excel(writer, sheet_name='Raw_Data', index=False)

            messagebox.showinfo("Success", f"✅ Data exported to {filename}")
            self.status_var.set(f"💾 Data exported to {filename}")
        except Exception as e:
            messagebox.showerror("Error", f"Error exporting data: {e}")
            self.status_var.set("❌ Export failed")

    def generate_pdf_report(self):
        """Generate professional PDF report with improved error handling."""
        if self.pivot is None:
            messagebox.showwarning("Warning", "No data to export. Please analyze data first.")
            return

        try:
            filename = filedialog.asksaveasfilename(
                defaultextension=".pdf",
                filetypes=[("PDF files", "*.pdf"), ("All files", "*.*")],
                title="Save PDF Report"
            )
            if not filename:
                return

            doc = SimpleDocTemplate(filename, pagesize=A4)
            story = []
            styles = getSampleStyleSheet()
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=18,
                spaceAfter=30,
                alignment=1
            )
            title = Paragraph(f"Monthly Return Analysis Report<br/>{self.ticker}", title_style)
            story.append(title)
            story.append(Spacer(1, 20))

            info_text = f"""
            <b>Analysis Period:</b> {self.start_year_var.get()} to {self.end_date_var.get()[:4]}<br/>
            <b>Total Years Analyzed:</b> {len(self.pivot)}<br/>
            <b>Total Months:</b> {len(self.monthly_ret)}<br/>
            <b>Overall Average Return:</b> {self.overall_avg*100:+.2f}%<br/>
            """
            info_para = Paragraph(info_text, styles['Normal'])
            story.append(info_para)
            story.append(Spacer(1, 20))

            story.append(Paragraph("<b>Monthly Average Returns</b>", styles['Heading2']))
            story.append(Spacer(1, 12))

            table_data = [['Month', 'Average Return (%)']]
            for month_num, month_name in enumerate(self.months, 1):
                if month_num in self.month_avg.index:
                    avg_ret = self.month_avg[month_num] * 100
                    table_data.append([month_name, f"{avg_ret:+.2f}%"])

            table = Table(table_data)
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            story.append(table)

            doc.build(story)
            messagebox.showinfo("Success", f"✅ PDF report generated: {filename}")
            self.status_var.set(f"📄 PDF report generated: {filename}")
        except Exception as e:
            messagebox.showerror("Error", f"Error generating PDF: {e}")
            self.status_var.set("❌ PDF generation failed")

    # --- Analysis Methods ---
    def toggle_benchmark_line(self):
        """Toggle the benchmark line on/off."""
        self.show_benchmark.set(not self.show_benchmark.get())
        self.update_charts()
        status = "ON" if self.show_benchmark.get() else "OFF"
        self.status_var.set(f"📊 Benchmark line is now {status}")

    def run_statistical_tests(self):
        """Run statistical significance tests with improved error handling."""
        if self.pivot is None:
            messagebox.showwarning("Warning", "No data available. Please analyze data first.")
            return

        try:
            for widget in self.stats_frame.winfo_children():
                widget.destroy()

            stats_text = tk.Text(self.stats_frame, wrap=tk.WORD, font=('Consolas', 10))
            stats_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

            results = "🧮 STATISTICAL SIGNIFICANCE TESTS\n"
            results += "=" * 50 + "\n"
            results += "T-TESTS BETWEEN MONTHS (p < 0.05 indicates significant difference):\n"
            results += "-" * 70 + "\n"

            significant_pairs = []
            for i in range(1, 13):
                for j in range(i+1, 13):
                    month1_data = self.pivot[i].dropna()
                    month2_data = self.pivot[j].dropna()
                    if len(month1_data) > 1 and len(month2_data) > 1:
                        t_stat, p_val = stats.ttest_ind(month1_data, month2_data)
                        if p_val < 0.05:
                            significant_pairs.append((i, j, p_val))
                            month1_name = self.months[i-1]
                            month2_name = self.months[j-1]
                            results += f"{month1_name} vs {month2_name}: p = {p_val:.4f}\n"

            if not significant_pairs:
                results += "No statistically significant differences found between months (p < 0.05)\n"

            results += f"\nTotal significant pairs: {len(significant_pairs)}\n"
            results += "DESCRIPTIVE STATISTICS BY MONTH:\n"
            results += "-" * 40 + "\n"

            for i in range(1, 13):
                month_data = self.pivot[i].dropna()
                month_name = self.months[i-1]
                results += f"{month_name}: Mean={month_data.mean()*100:+.2f}%, Std={month_data.std()*100:.2f}%\n"

            stats_text.insert(1.0, results)
            self.status_var.set("🧮 Statistical tests completed")
        except Exception as e:
            messagebox.showerror("Error", f"Error running statistical tests: {e}")
            self.status_var.set("❌ Statistical tests failed")

    def run_ml_analysis(self):
        """Run advanced machine learning analysis with PCA, GMM, Isolation Forest, cluster visualization, and upcoming month forecast."""
        if self.pivot is None:
            messagebox.showwarning("Warning", "No data available. Please analyze data first.")
            return

        try:
            for widget in self.ml_frame.winfo_children():
                widget.destroy()

            # Create frame for text and plot
            ml_frame_container = ttk.Frame(self.ml_frame)
            ml_frame_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

            ml_text = tk.Text(ml_frame_container, wrap=tk.WORD, font=('Consolas', 10), height=15)
            ml_text.pack(fill=tk.X, padx=5, pady=(0, 5))
            scrollbar = ttk.Scrollbar(ml_frame_container, orient=tk.VERTICAL, command=ml_text.yview)
            scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            ml_text.configure(yscrollcommand=scrollbar.set)

            results = "🤖 ADVANCED MACHINE LEARNING ANALYSIS\n"
            results += "=" * 50 + "\n"

            # Prepare data
            monthly_features = pd.DataFrame({
                'month': range(1, 13),
                'avg_return': self.month_avg.values,
                'std_dev': self.pivot.std().values,
                'positive_rate': (self.pivot > 0).sum().values / self.pivot.count().values
            })

            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(monthly_features[['avg_return', 'std_dev', 'positive_rate']])

            # PCA for dimensionality reduction
            pca = PCA(n_components=2)
            features_reduced = pca.fit_transform(features_scaled)
            explained_variance = pca.explained_variance_ratio_.sum()

            results += f"PCA: Reduced to 2 components, explaining {explained_variance:.2%} of variance\n"
            results += "-" * 40 + "\n"

            # Optimize number of clusters using BIC
            bic_values = []
            for k in range(1, 6):
                gmm = GaussianMixture(n_components=k, random_state=42)
                gmm.fit(features_reduced)
                bic_values.append(gmm.bic(features_reduced))

            optimal_k = np.argmin(bic_values) + 1

            results += f"Optimal number of clusters (BIC method): {optimal_k}\n"

            # Gaussian Mixture Model clustering
            gmm = GaussianMixture(n_components=optimal_k, random_state=42)
            clusters = gmm.fit_predict(features_reduced)
            monthly_features['cluster'] = clusters
            probabilities = gmm.predict_proba(features_reduced)

            results += f"GMM CLUSTERING RESULTS (n_components={optimal_k}):\n"
            results += "-" * 40 + "\n"

            for cluster_id in range(optimal_k):
                cluster_months = monthly_features[monthly_features['cluster'] == cluster_id]
                month_names = [self.months[idx - 1] for idx in cluster_months['month']]
                results += f"Cluster {cluster_id + 1}: {', '.join(month_names)}\n"
                results += f"  Avg Return: {cluster_months['avg_return'].mean() * 100:+.2f}%\n"
                results += f"  Avg Risk: {cluster_months['std_dev'].mean() * 100:.2f}%\n"
                results += f"  Avg Pos Rate: {cluster_months['positive_rate'].mean() * 100:.1f}%\n"
                cluster_probs = probabilities[monthly_features['cluster'] == cluster_id].mean(axis=0)
                results += f"  Cluster Probabilities: {', '.join([f'{p:.2f}' for p in cluster_probs])}\n"

            results += "\nISOLATION FOREST ANOMALY DETECTION:\n"
            results += "-" * 40 + "\n"

            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            anomalies = iso_forest.fit_predict(features_scaled)
            anomaly_scores = -iso_forest.decision_function(features_scaled)

            for i, (score, month) in enumerate(zip(anomaly_scores, self.months)):
                if anomalies[i] == -1:
                    results += f"{month}: Anomaly Score = {score:.3f} (Outlier)\n"
                else:
                    results += f"{month}: Anomaly Score = {score:.3f}\n"

            # Plain-English Summary
            results += "\nPLAIN-ENGLISH SUMMARY:\n"
            results += "-" * 40 + "\n"

            variance_text = f"We simplified the data into two main patterns, capturing {explained_variance:.2%} of the variation in monthly returns, risk, and positive performance. "
            variance_text += "This means the scatter plot below shows most of the key trends reliably." if explained_variance > 0.8 else "This means some patterns may not be fully captured in the plot."
            results += variance_text + "\n"

            results += f"The analysis grouped the 12 months into {optimal_k} clusters based on similar return and risk profiles. For example, months like February may group together if they often have strong returns, while volatile months like September may form a separate group.\n"

            if len(np.where(anomalies == -1)[0]) > 0:
                anomaly_months = [self.months[i] for i in np.where(anomalies == -1)[0]]
                results += f"Some months, like {', '.join(anomaly_months)}, stand out as unusual due to their unique return or risk patterns, possibly indicating higher volatility or significant market events.\n"
            else:
                results += "No months were flagged as highly unusual, suggesting consistent patterns across the year.\n"

            # Upcoming Month Forecast using ARIMA
            results += "\nUPCOMING MONTH FORECAST:\n"
            results += "-" * 40 + "\n"

            try:
                # Prepare data for ARIMA
                returns = self.monthly_ret.to_frame()

                # Determine forecast date (next month)
                end_date = pd.to_datetime(self.end_date_var.get())
                current_year = end_date.year
                current_month = end_date.month
                forecast_month = current_month + 1 if current_month < 12 else 1
                forecast_year = current_year if current_month < 12 else current_year + 1
                forecast_month_name = self.months[forecast_month - 1]
                forecast_date = pd.to_datetime(f"{forecast_year}-{forecast_month:02d}-28")  # Approximate end of month

                steps = (forecast_year - returns.index[-1].year) * 12 + (forecast_month - returns.index[-1].month)
                if steps <= 0:
                    steps = 1  # Ensure at least one step ahead

                if PMDARIMA_AVAILABLE:
                    # Use auto_arima to select optimal order
                    model = auto_arima(returns['ret'], seasonal=True, m=12, suppress_warnings=True,
                                     start_p=0, start_q=0, max_p=3, max_q=3, start_P=0, start_Q=0, max_P=2, max_Q=2)
                    arima_order = model.order
                    seasonal_order = model.seasonal_order
                    results += f"ARIMA Model: Order={arima_order}, Seasonal Order={seasonal_order}\n"
                else:
                    # Fallback to fixed-order ARIMA(1,1,1)(1,1,1)[12]
                    arima_order = (1, 1, 1)
                    seasonal_order = (1, 1, 1, 12)
                    results += f"ARIMA Model: Using fallback order=(1,1,1), seasonal_order=(1,1,1,12) due to pmdarima unavailability\n"

                # Fit ARIMA model
                arima_model = ARIMA(returns['ret'], order=arima_order, seasonal_order=seasonal_order)
                arima_fit = arima_model.fit()

                # Forecast
                forecast = arima_fit.get_forecast(steps=steps)
                forecast_mean = forecast.predicted_mean.iloc[-1] * 100
                conf_int = forecast.conf_int(alpha=0.05).iloc[-1] * 100

                results += f"Predicted return for {forecast_month_name} {forecast_year}: {forecast_mean:.2f}% (95% CI: {conf_int.iloc[0]:.2f}% to {conf_int.iloc[1]:.2f}%)\n"

                # Project forecast into PCA space
                forecast_stats = np.array([[forecast_mean / 100, self.pivot[forecast_month].std(), (self.pivot[forecast_month] > 0).sum() / self.pivot[forecast_month].count()]])
                forecast_scaled = scaler.transform(forecast_stats)
                forecast_pca = pca.transform(forecast_scaled)[0]
            except Exception as e:
                results += f"Forecasting failed: {str(e)}\n"
                forecast_pca = None

            ml_text.insert(1.0, results)

            # Cluster visualization
            ml_plot_frame = ttk.Frame(ml_frame_container)
            ml_plot_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

            fig4, ax4 = plt.subplots(figsize=(12, 6))
            colors = plt.cm.Set1(np.linspace(0, 1, optimal_k))

            for cluster_id in range(optimal_k):
                cluster_data = features_reduced[monthly_features['cluster'] == cluster_id]
                ax4.scatter(cluster_data[:, 0], cluster_data[:, 1], s=150, alpha=0.7,
                           c=[colors[cluster_id]], label=f'Cluster {cluster_id + 1}')

            # Highlight anomalies
            anomaly_indices = np.where(anomalies == -1)[0]
            if len(anomaly_indices) > 0:
                ax4.scatter(features_reduced[anomaly_indices, 0], features_reduced[anomaly_indices, 1],
                           s=200, marker='x', c='red', label='Anomalies', linewidths=2)

            # Plot forecasted month
            if forecast_pca is not None:
                ax4.scatter(forecast_pca[0], forecast_pca[1], s=200, marker='*', c='purple',
                           label=f'{forecast_month_name} {forecast_year} Forecast', linewidths=2)

            for i, month in enumerate(self.months):
                ax4.annotate(month, (features_reduced[i, 0], features_reduced[i, 1]),
                            xytext=(5, 5), textcoords='offset points', fontsize=9, fontweight='bold',
                            bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))

            ax4.set_xlabel('Principal Component 1', fontsize=11, fontweight='bold')
            ax4.set_ylabel('Principal Component 2', fontsize=11, fontweight='bold')
            ax4.set_title(f'Monthly Clusters in PCA Space for {self.ticker}\nPeriod: {self.start_year_var.get()} to {self.end_date_var.get()[:4]}',
                         fontsize=12, fontweight='bold', pad=20)
            ax4.grid(True, alpha=0.3, linestyle='--')
            ax4.spines['top'].set_visible(False)
            ax4.spines['right'].set_visible(False)
            ax4.legend(loc='upper right', framealpha=0.9)

            annot4 = ax4.annotate("", xy=(0,0), xytext=(10,10), textcoords="offset points",
                                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=1, alpha=0.9),
                                 arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0"),
                                 fontsize=9)
            annot4.set_visible(False)

            def update_annot_ml(ind):
                index = ind["ind"][0]
                pos = features_reduced[index]
                annot4.xy = pos
                month_name = self.months[index]
                cluster_id = monthly_features['cluster'].iloc[index]
                prob = max(probabilities[index])
                score = anomaly_scores[index]
                status = "Outlier" if anomalies[index] == -1 else "Normal"
                text = f"{month_name}\nCluster: {cluster_id + 1}\nProb: {prob:.2f}\nAnomaly Score: {score:.3f} ({status})"
                annot4.set_text(text)
                annot4.get_bbox_patch().set_alpha(0.9)

            def hover_ml(event):
                vis = annot4.get_visible()
                if event.inaxes == ax4:
                    for i in range(len(self.months)):
                        cont = ax4.contains_point((event.xdata, event.ydata))
                        if cont:
                            update_annot_ml({"ind": [i]})
                            annot4.set_visible(True)
                            fig4.canvas.draw_idle()
                            return
                    scatter = ax4.collections[0]
                    cont, ind = scatter.contains(event)
                    if cont:
                        update_annot_ml(ind)
                        annot4.set_visible(True)
                        fig4.canvas.draw_idle()
                        return
                if vis:
                    annot4.set_visible(False)
                    fig4.canvas.draw_idle()

            fig4.canvas.mpl_connect("motion_notify_event", hover_ml)
            canvas4 = FigureCanvasTkAgg(fig4, ml_plot_frame)
            canvas4.draw()
            canvas4.get_tk_widget().pack(fill=tk.BOTH, expand=True)

            self.status_var.set("🤖 Advanced ML analysis with visualization and forecasting completed")
        except Exception as e:
            messagebox.showerror("Error", f"Error running ML analysis: {e}")
            self.status_var.set("❌ ML analysis failed")

    def run(self):
        """Start the application."""
        self.root.mainloop()

if __name__ == "__main__":
    app = JSEAnalyzer()
    app.run()

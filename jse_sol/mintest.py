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
import logging
import time
import re
import concurrent.futures
from scipy import stats
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import IsolationForest
import warnings

# Suppress warnings for cleaner console output
warnings.filterwarnings('ignore')
matplotlib.use('TkAgg')

# --- Configuration Constants ---
SIDEBAR_BG = '#2c3e50'      # Dark Blue-Grey
SIDEBAR_FG = '#ecf0f1'      # Off-White
MAIN_BG = '#f5f6fa'         # Very Light Grey
ACCENT_COLOR = '#3498db'    # Blue
SUCCESS_COLOR = '#27ae60'   # Green
WARNING_COLOR = '#f39c12'   # Orange

# --- Setup Logging ---
def setup_logging():
    """Configures the root logger."""
    log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('jse_analyzer.log', mode='w')
    file_handler.setFormatter(log_formatter)
    
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
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
        label = tk.Label(self.tooltip, text=self.text, background="#ffffe0", 
                        relief="solid", borderwidth=1, padx=5, pady=3)
        label.pack()

    def hide_tooltip(self, event=None):
        if self.tooltip:
            self.tooltip.destroy()
            self.tooltip = None

class JSEAnalyzer:
    """A modern GUI application for analyzing monthly returns of global financial indices."""
    VERSION = "3.0.2"

    def __init__(self):
        self.logger = setup_logging()
        self.logger.info(f"--- Application Start: Global Index Analyzer v{self.VERSION} ---")

        self.root = tk.Tk()
        self.root.title(f"Global Index Monthly Return Analyzer (v{self.VERSION})")
        self.root.geometry("1400x900")
        self.root.configure(bg=MAIN_BG)
        
        # Cache setup
        self.cache_dir = f"cache_v{self.VERSION.replace('.', '_')}"
        os.makedirs(self.cache_dir, exist_ok=True)

        # Data Definitions
        self.ticker_options = {
            "🇿🇦 JSE All Share (^J203.JO)": "^J203.JO",
            "🇿🇦 JSE Financials (^J258.JO)": "^J258.JO",
            "🇿🇦 JSE Resources (^J250.JO)": "^J250.JO",
            "🇺🇸 S&P 500 (^GSPC)": "^GSPC",
            "🇺🇸 Nasdaq 100 (^NDX)": "^NDX",
            "🇬🇧 FTSE 100 (^FTSE)": "^FTSE",
            "🇩🇪 DAX (^GDAXI)": "^GDAXI",
            "🇯🇵 Nikkei 225 (^N225)": "^N225",
            "🇨🇳 Shanghai Composite (000001.SS)": "000001.SS",
            "🌍 MSCI World (^MXWO)": "^MXWO",
            "🌍 MSCI Emerging Markets (^MXEF)": "^MXEF"
        }
        self.ticker = "^J203.JO"
        self.start_year = 1990
        self.end_date = datetime.today().strftime("%Y-%m-%d")
        
        # State Variables
        self.data = None
        self.monthly_ret = None
        self.pivot = None
        self.month_avg = None
        self.month_median = None
        self.months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        self.show_benchmark = tk.BooleanVar(value=True)
        self.dark_mode = tk.BooleanVar(value=False)
        self.comparison_tickers = []
        self.comparison_data = {}
        self.bar_metric = tk.StringVar(value="Mean")
        self.data_ready = False

        # UI Setup
        self.style = ttk.Style()
        self.style.theme_use('clam')
        self.configure_styles()
        self.setup_ui()

    def configure_styles(self):
        """Configure modern, flat styles for the GUI."""
        # General Ttk styles
        self.style.configure('TFrame', background=MAIN_BG)
        self.style.configure('TLabel', background=MAIN_BG, foreground='#2c3e50', font=('Segoe UI', 9))
        
        # Notebook (Tabs)
        self.style.configure('TNotebook', background=MAIN_BG)
        self.style.configure('TNotebook.Tab', padding=[12, 4], font=('Segoe UI', 10))
        
        # Sidebar Specifics
        self.style.configure('Sidebar.TFrame', background=SIDEBAR_BG)
        self.style.configure('Sidebar.TLabel', background=SIDEBAR_BG, foreground=SIDEBAR_FG, font=('Segoe UI', 9))
        self.style.configure('SidebarTitle.TLabel', background=SIDEBAR_BG, foreground=SIDEBAR_FG, font=('Segoe UI', 14, 'bold'))
        self.style.configure('SidebarHeader.TLabel', background=SIDEBAR_BG, foreground='#bdc3c7', font=('Segoe UI', 8, 'bold'))
        self.style.configure('Sidebar.TCheckbutton', background=SIDEBAR_BG, foreground=SIDEBAR_FG, font=('Segoe UI', 9))

        # Buttons
        self.style.configure('Action.TButton', font=('Segoe UI', 10, 'bold'), background=ACCENT_COLOR, foreground='white', borderwidth=0)
        self.style.map('Action.TButton', background=[('active', '#2980b9')])
        
        self.style.configure('Secondary.TButton', font=('Segoe UI', 9), background='#7f8c8d', foreground='white', borderwidth=0)
        self.style.map('Secondary.TButton', background=[('active', '#95a5a6')])
        
        self.style.configure('Success.TButton', font=('Segoe UI', 9), background=SUCCESS_COLOR, foreground='white', borderwidth=0)
        self.style.map('Success.TButton', background=[('active', '#2ecc71')])

    def setup_ui(self):
        """Constructs the Dashboard Layout: Left Sidebar + Main Content Area."""
        
        # Main Container (Holds Sidebar + Content)
        main_container = ttk.Frame(self.root)
        main_container.pack(fill=tk.BOTH, expand=True)

        # --- LEFT SIDEBAR ---
        self.sidebar_frame = ttk.Frame(main_container, style='Sidebar.TFrame', width=280)
        self.sidebar_frame.pack(side=tk.LEFT, fill=tk.Y)
        self.sidebar_frame.pack_propagate(False) # Maintain fixed width

        # --- MAIN CONTENT AREA ---
        self.content_frame = ttk.Frame(main_container, style='TFrame')
        self.content_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=15, pady=15)

        self.build_sidebar()
        self.build_content_area()

    def build_sidebar(self):
        """Populates the sidebar with controls."""
        pad_x = 20
        
        # 1. App Title
        ttk.Label(self.sidebar_frame, text="MARKET\nANALYZER", style='SidebarTitle.TLabel', justify=tk.LEFT).pack(anchor='w', padx=pad_x, pady=(30, 20))
        
        # 2. Data Settings Section
        ttk.Label(self.sidebar_frame, text="DATA SETTINGS", style='SidebarHeader.TLabel').pack(anchor='w', padx=pad_x, pady=(10, 5))
        
        # Ticker Selection
        ttk.Label(self.sidebar_frame, text="Select Asset:", style='Sidebar.TLabel').pack(anchor='w', padx=pad_x)
        self.ticker_var = tk.StringVar(value="🇿🇦 JSE All Share (^J203.JO)")
        
        # Container for Ticker Input (Combo vs Entry)
        self.ticker_input_frame = ttk.Frame(self.sidebar_frame, style='Sidebar.TFrame')
        self.ticker_input_frame.pack(fill=tk.X, padx=pad_x, pady=(0, 5))
        
        self.ticker_combo = ttk.Combobox(self.ticker_input_frame, textvariable=self.ticker_var, 
                                        values=list(self.ticker_options.keys()), state="readonly", height=15)
        self.ticker_combo.pack(fill=tk.X)
        
        self.custom_ticker_var = tk.StringVar()
        self.custom_ticker_entry = ttk.Entry(self.ticker_input_frame, textvariable=self.custom_ticker_var)
        
        self.use_custom_var = tk.BooleanVar()
        self.custom_check = ttk.Checkbutton(self.sidebar_frame, text="Use Custom Ticker", variable=self.use_custom_var,
                                      command=self.toggle_custom_ticker, style='Sidebar.TCheckbutton')
        self.custom_check.pack(anchor='w', padx=pad_x, pady=(0, 15))

        # Date Range
        ttk.Label(self.sidebar_frame, text="Time Period:", style='Sidebar.TLabel').pack(anchor='w', padx=pad_x)
        self.date_range_var = tk.StringVar(value="Custom")
        
        # Updated values to include 1 Year and 3 Years
        date_combo = ttk.Combobox(self.sidebar_frame, textvariable=self.date_range_var,
                                 values=["Custom", "Last 1 Year", "Last 3 Years", "Last 5 Years", "Last 10 Years", "Last 20 Years", "All Data"], 
                                 state="readonly")
        date_combo.pack(fill=tk.X, padx=pad_x, pady=(0, 5))
        date_combo.bind('<<ComboboxSelected>>', self.on_date_range_change)

        # Custom Date Inputs
        date_row = ttk.Frame(self.sidebar_frame, style='Sidebar.TFrame')
        date_row.pack(fill=tk.X, padx=pad_x, pady=(0, 15))
        
        self.start_year_var = tk.IntVar(value=self.start_year)
        self.start_year_entry = ttk.Entry(date_row, textvariable=self.start_year_var, width=6)
        self.start_year_entry.pack(side=tk.LEFT)
        
        ttk.Label(date_row, text=" to ", style='Sidebar.TLabel').pack(side=tk.LEFT)
        
        self.end_date_var = tk.StringVar(value=self.end_date)
        self.end_date_entry = ttk.Entry(date_row, textvariable=self.end_date_var, width=11)
        self.end_date_entry.pack(side=tk.LEFT)

        # Comparison
        ttk.Label(self.sidebar_frame, text="Compare With:", style='Sidebar.TLabel').pack(anchor='w', padx=pad_x)
        self.compare_var = tk.StringVar()
        comp_row = ttk.Frame(self.sidebar_frame, style='Sidebar.TFrame')
        comp_row.pack(fill=tk.X, padx=pad_x, pady=(0, 20))
        
        ttk.Combobox(comp_row, textvariable=self.compare_var, values=list(self.ticker_options.keys()), 
                    state="readonly").pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(comp_row, text="+", width=3, command=self.add_comparison_ticker, 
                  style='Secondary.TButton').pack(side=tk.LEFT, padx=(5, 0))

        # 3. Action Buttons
        ttk.Separator(self.sidebar_frame, orient='horizontal').pack(fill=tk.X, padx=10, pady=10)
        
        self.analyze_btn = ttk.Button(self.sidebar_frame, text="⚡ RUN ANALYSIS", command=self.analyze_data, style='Action.TButton')
        self.analyze_btn.pack(fill=tk.X, padx=pad_x, pady=10, ipady=5)
        
        # Secondary Analysis Tools
        tools_frame = ttk.Frame(self.sidebar_frame, style='Sidebar.TFrame')
        tools_frame.pack(fill=tk.X, padx=pad_x, pady=10)
        
        ttk.Button(tools_frame, text="🤖 ML Insights", command=self.run_ml_analysis, style='Secondary.TButton').pack(fill=tk.X, pady=2)
        ttk.Button(tools_frame, text="🧮 Stat Tests", command=self.run_statistical_tests, style='Secondary.TButton').pack(fill=tk.X, pady=2)

        # 4. Exports & Footer
        ttk.Label(self.sidebar_frame, text="EXPORTS", style='SidebarHeader.TLabel').pack(anchor='w', padx=pad_x, pady=(20, 5))
        
        export_row = ttk.Frame(self.sidebar_frame, style='Sidebar.TFrame')
        export_row.pack(fill=tk.X, padx=pad_x)
        
        self.export_btn = ttk.Button(export_row, text="Excel", command=self.export_to_excel, state=tk.DISABLED, style='Success.TButton')
        self.export_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 2))
        
        self.pdf_btn = ttk.Button(export_row, text="PDF", command=self.generate_pdf_report, state=tk.DISABLED, style='Success.TButton')
        self.pdf_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(2, 0))

        # Bottom Controls
        self.sidebar_frame.pack_propagate(False) # Ensure footer stays at bottom if using pack side bottom
        footer_frame = ttk.Frame(self.sidebar_frame, style='Sidebar.TFrame')
        footer_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=pad_x, pady=20)
        
        ttk.Checkbutton(footer_frame, text="Dark Mode", variable=self.dark_mode, 
                       command=self.toggle_dark_mode, style='Sidebar.TCheckbutton').pack(anchor='w')

    def build_content_area(self):
        """Populates the main content area with tabs and status."""
        
        # Status Bar (Top)
        self.status_var = tk.StringVar(value="✨ Ready - Select an asset and click Run Analysis")
        self.status_label = ttk.Label(self.content_frame, textvariable=self.status_var,
                                    font=('Segoe UI', 10), background='#e1e1e1', foreground='#555', padding=8, relief='flat')
        self.status_label.pack(fill=tk.X, pady=(0, 15))

        # Notebook (Tabs)
        self.notebook = ttk.Notebook(self.content_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # Tab 0: Price History (New)
        self.history_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.history_frame, text="📈 Price History")

        # Tab 1: Average Returns (Bar Chart)
        self.bar_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.bar_frame, text="📊 Performance")
        
        # Controls specific to Bar Chart (kept inside tab for context)
        self.bar_control_frame = ttk.Frame(self.bar_frame)
        self.bar_control_frame.pack(fill=tk.X, pady=10)
        
        ttk.Label(self.bar_control_frame, text="Metric:").pack(side=tk.LEFT, padx=5)
        self.metric_combo = ttk.Combobox(self.bar_control_frame, textvariable=self.bar_metric, 
                                        values=["Mean", "Median"], state="disabled", width=10)
        self.metric_combo.pack(side=tk.LEFT, padx=5)
        self.metric_combo.bind('<<ComboboxSelected>>', lambda e: self.update_charts())
        
        ttk.Button(self.bar_control_frame, text="Toggle Benchmark", command=self.toggle_benchmark_line, 
                  style='Secondary.TButton').pack(side=tk.LEFT, padx=10)

        # Tab 2: Heatmap
        self.heatmap_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.heatmap_frame, text="🌡️ Heatmap")

        # Tab 3: Risk vs Return
        self.scatter_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.scatter_frame, text="⚖️ Risk/Return")

        # Tab 4: Comparison
        self.comparison_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.comparison_frame, text="🔄 Compare")

        # Tab 5: Summary Stats
        self.summary_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.summary_frame, text="📋 Summary")
        self.summary_text = tk.Text(self.summary_frame, wrap=tk.WORD, font=('Consolas', 10), relief='flat', padx=10, pady=10)
        self.summary_text.pack(fill=tk.BOTH, expand=True)

        # Tab 6: Statistical Tests
        self.stats_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.stats_frame, text="🧮 Significance")

        # Tab 7: ML Analysis
        self.ml_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.ml_frame, text="🤖 Machine Learning")

    # --- Logic Handlers ---
    def toggle_custom_ticker(self):
        """Swaps the Combobox for an Entry widget in the sidebar."""
        if self.use_custom_var.get():
            self.ticker_combo.pack_forget()
            self.custom_ticker_entry.pack(fill=tk.X)
            self.custom_ticker_entry.delete(0, tk.END)
            self.custom_ticker_entry.focus()
        else:
            self.custom_ticker_entry.pack_forget()
            self.ticker_combo.pack(fill=tk.X)

    def toggle_benchmark_line(self):
        """Toggles the benchmark line on charts."""
        current = self.show_benchmark.get()
        self.show_benchmark.set(not current)
        # status = "ON" if not current else "OFF" # Optional log message
        if self.data_ready:
            self.update_charts()

    def toggle_dark_mode(self):
        """Toggles UI colors."""
        is_dark = self.dark_mode.get()
        
        bg_color = '#2c3e50' if is_dark else MAIN_BG
        fg_color = '#ecf0f1' if is_dark else '#2c3e50'
        text_bg = '#34495e' if is_dark else '#ffffff'
        text_fg = '#ecf0f1' if is_dark else '#2c3e50'
        
        self.content_frame.configure(style='TFrame') # Refresh style
        self.style.configure('TFrame', background=bg_color)
        self.style.configure('TLabel', background=bg_color, foreground=fg_color)
        self.style.configure('TNotebook', background=bg_color)
        
        self.summary_text.configure(bg=text_bg, fg=text_fg)
        
        # Update plots if they exist
        if self.data_ready:
            self.update_charts()

    def on_date_range_change(self, event=None):
        selection = self.date_range_var.get()
        current_year = datetime.now().year
        
        if selection == "Custom":
            self.start_year_entry.config(state=tk.NORMAL)
            self.end_date_entry.config(state=tk.NORMAL)
        else:
            self.start_year_entry.config(state=tk.DISABLED)
            self.end_date_entry.config(state=tk.DISABLED)
            
            self.end_date_var.set(datetime.today().strftime("%Y-%m-%d"))
            if selection == "Last 1 Year":
                self.start_year_var.set(current_year - 1)
            elif selection == "Last 3 Years":
                self.start_year_var.set(current_year - 3)
            elif selection == "Last 5 Years":
                self.start_year_var.set(current_year - 5)
            elif selection == "Last 10 Years":
                self.start_year_var.set(current_year - 10)
            elif selection == "Last 20 Years":
                self.start_year_var.set(current_year - 20)
            elif selection == "All Data":
                self.start_year_var.set(1990)

    # --- Data Processing ---
    def validate_inputs(self):
        try:
            start_year = self.start_year_var.get()
            if start_year < 1900 or start_year > datetime.now().year:
                raise ValueError("Invalid Start Year")
            
            # Ticker Logic
            if self.use_custom_var.get():
                t = self.custom_ticker_var.get().strip()
                if not t: raise ValueError("Custom ticker is empty")
                self.ticker = t
            else:
                name = self.ticker_var.get()
                self.ticker = self.ticker_options.get(name, "^J203.JO")
                
            return True
        except ValueError as e:
            messagebox.showerror("Input Error", str(e))
            return False

    def analyze_data(self):
        if not self.validate_inputs(): return
        
        self.status_var.set("📥 Downloading market data...")
        self.analyze_btn.config(state=tk.DISABLED)
        self.data_ready = False
        self.metric_combo.config(state="disabled")
        
        # Run in thread
        threading.Thread(target=self._analysis_worker, daemon=True).start()

    def _analysis_worker(self):
        try:
            start = f"{self.start_year_var.get()}-01-01"
            end = self.end_date_var.get()
            
            # 1. Download
            self.data = yf.download(self.ticker, start=start, end=end, progress=False, auto_adjust=False)
            
            if self.data is None or self.data.empty:
                raise ValueError("No data found. Check ticker or connection.")
            
            # 2. Process
            price_col = 'Adj Close' if 'Adj Close' in self.data.columns else 'Close'
            self.monthly = self.data[price_col].resample('ME').last()
            self.monthly_ret = self.monthly.pct_change().dropna()
            
            # Check if Series or DataFrame and handle
            if isinstance(self.monthly_ret, pd.DataFrame):
                 self.monthly_ret = self.monthly_ret.iloc[:, 0]

            self.df = self.monthly_ret.to_frame(name='ret')
            self.df['year'] = self.df.index.year
            self.df['month'] = self.df.index.month
            
            self.pivot = self.df.pivot_table(index='year', columns='month', values='ret')
            self.month_avg = self.pivot.mean().sort_index()
            self.month_median = self.pivot.median().sort_index()
            self.overall_avg = self.monthly_ret.mean()
            
            self.data_ready = True
            
            # 3. Update UI (Main Thread)
            self.root.after(0, self._post_analysis_ui_update)
            
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Analysis Failed", str(e)))
            self.root.after(0, lambda: self.status_var.set("❌ Analysis failed"))
        finally:
            self.root.after(0, lambda: self.analyze_btn.config(state=tk.NORMAL))

    def _post_analysis_ui_update(self):
        self.metric_combo.config(state="readonly")
        self.export_btn.config(state=tk.NORMAL)
        self.pdf_btn.config(state=tk.NORMAL)
        self.status_var.set(f"✅ Analysis complete for {self.ticker}")
        self.update_charts()
        self.update_summary()

    # --- Visualization ---
    def update_charts(self):
        if not self.data_ready: return
        
        self.update_price_chart()
        self.update_bar_chart()
        self.update_heatmap()
        self.update_scatter()
        self.update_comparison()

    def _clear_frame(self, frame):
        for w in frame.winfo_children():
            if w != self.bar_control_frame: # Preserve controls
                w.destroy()

    def update_price_chart(self):
        """Updates the price history line chart."""
        self._clear_frame(self.history_frame)
        
        fig, ax = plt.subplots(figsize=(10, 5), dpi=100)
        if self.dark_mode.get():
            fig.patch.set_facecolor('#2c3e50')
            ax.set_facecolor('#34495e')
            ax.tick_params(colors='white')
            ax.xaxis.label.set_color('white')
            ax.yaxis.label.set_color('white')
            ax.title.set_color('white')
            ax.spines['bottom'].set_color('white')
            ax.spines['left'].set_color('white')
            
        # Plotting raw price data
        price_col = 'Adj Close' if 'Adj Close' in self.data.columns else 'Close'
        self.data[price_col].plot(ax=ax, color='#2980b9', linewidth=2)
        
        ax.set_title(f"Price History: {self.ticker}")
        ax.set_ylabel(f"Price ({price_col})")
        ax.set_xlabel("Date")
        ax.grid(True, alpha=0.3)
        
        canvas = FigureCanvasTkAgg(fig, self.history_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def update_bar_chart(self):
        self._clear_frame(self.bar_frame)
        
        fig, ax = plt.subplots(figsize=(10, 5), dpi=100)
        if self.dark_mode.get():
            fig.patch.set_facecolor('#2c3e50')
            ax.set_facecolor('#34495e')
            ax.tick_params(colors='white')
            ax.xaxis.label.set_color('white')
            ax.yaxis.label.set_color('white')
            ax.title.set_color('white')
            ax.spines['bottom'].set_color('white')
            ax.spines['left'].set_color('white')
        
        data = self.month_avg if self.bar_metric.get() == "Mean" else self.month_median
        colors_list = ['#e74c3c' if x < 0 else '#3498db' for x in data]
        
        bars = ax.bar(range(1, 13), data*100, color=colors_list, alpha=0.8)
        
        ax.set_xticks(range(1, 13))
        ax.set_xticklabels(self.months)
        ax.set_title(f"{self.bar_metric.get()} Monthly Returns ({self.ticker})")
        ax.set_ylabel("Return (%)")
        ax.grid(axis='y', alpha=0.2)
        
        if self.show_benchmark.get():
            avg = self.overall_avg * 100
            ax.axhline(y=avg, color='#f1c40f', linestyle='--', label=f'Avg: {avg:.2f}%')
            ax.legend()

        canvas = FigureCanvasTkAgg(fig, self.bar_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def update_heatmap(self):
        self._clear_frame(self.heatmap_frame)
        fig, ax = plt.subplots(figsize=(12, 8))
        
        sns.heatmap(self.pivot*100, center=0, cmap='RdYlGn', annot=True, fmt='.1f', ax=ax,
                   cbar_kws={'label': 'Return (%)'})
        ax.set_title(f"Monthly Returns Heatmap: {self.ticker}")
        
        canvas = FigureCanvasTkAgg(fig, self.heatmap_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def update_scatter(self):
        self._clear_frame(self.scatter_frame)
        fig, ax = plt.subplots(figsize=(10, 6))
        
        std = self.pivot.std() * 100
        avg = self.month_avg * 100
        
        ax.scatter(std, avg, s=100, alpha=0.7, c='#3498db')
        for i, txt in enumerate(self.months):
            if i+1 in std.index:
                ax.annotate(txt, (std[i+1], avg[i+1]), xytext=(5, 5), textcoords='offset points')
                
        ax.set_xlabel("Risk (Standard Deviation %)")
        ax.set_ylabel("Return (Average %)")
        ax.set_title("Risk vs Return Profile")
        ax.grid(True, alpha=0.3)
        
        canvas = FigureCanvasTkAgg(fig, self.scatter_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # --- Comparison Logic ---
    def add_comparison_ticker(self):
        name = self.compare_var.get()
        ticker = self.ticker_options.get(name)
        if ticker and ticker not in self.comparison_tickers:
            self.comparison_tickers.append(ticker)
            threading.Thread(target=self._fetch_comparison, args=(ticker,), daemon=True).start()
    
    def _fetch_comparison(self, ticker):
        try:
            data = yf.download(ticker, start=f"{self.start_year_var.get()}-01-01", 
                             end=self.end_date_var.get(), progress=False, auto_adjust=False)
            if not data.empty:
                price_col = 'Adj Close' if 'Adj Close' in data.columns else 'Close'
                monthly = data[price_col].resample('ME').last().pct_change().dropna()
                
                # Handle Series/DataFrame ambiguity
                if isinstance(monthly, pd.DataFrame):
                    monthly = monthly.iloc[:, 0]
                    
                df = monthly.to_frame(name='ret')
                df['month'] = df.index.month
                avg = df.groupby('month')['ret'].mean().sort_index()
                
                self.comparison_data[ticker] = avg
                self.root.after(0, self.update_comparison)
                self.root.after(0, lambda: self.status_var.set(f"✅ Added {ticker} to comparison"))
        except Exception as e:
            self.logger.error(f"Comparison error: {e}")

    def update_comparison(self):
        self._clear_frame(self.comparison_frame)
        if not self.comparison_data: return
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot Main Ticker
        ax.plot(range(1, 13), self.month_avg*100, label=self.ticker, linewidth=3, marker='o')
        
        # Plot Comparisons
        for ticker, data in self.comparison_data.items():
            ax.plot(range(1, 13), data*100, label=ticker, linestyle='--', marker='x')
            
        ax.set_xticks(range(1, 13))
        ax.set_xticklabels(self.months)
        ax.set_ylabel("Average Return (%)")
        ax.set_title("Comparative Analysis")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        canvas = FigureCanvasTkAgg(fig, self.comparison_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # --- Analysis Tools (Stats & ML) ---
    def update_summary(self):
        txt = f"ANALYSIS SUMMARY: {self.ticker}\n"
        txt += "="*40 + "\n"
        txt += f"Overall Average Monthly Return: {self.overall_avg*100:.2f}%\n"
        txt += f"Best Month: {self.months[self.month_avg.idxmax()-1]} ({self.month_avg.max()*100:.2f}%)\n"
        txt += f"Worst Month: {self.months[self.month_avg.idxmin()-1]} ({self.month_avg.min()*100:.2f}%)\n\n"
        txt += "MONTHLY AVERAGES:\n"
        for i, m in enumerate(self.months):
            if i+1 in self.month_avg.index:
                txt += f"{m:<5}: {self.month_avg[i+1]*100:>6.2f}%\n"
        
        self.summary_text.delete(1.0, tk.END)
        self.summary_text.insert(1.0, txt)

    def run_statistical_tests(self):
        if not self.data_ready: return
        
        def worker():
            try:
                # ANOVA
                month_groups = [self.pivot[c].dropna() for c in self.pivot.columns]
                f_stat, p_val = stats.f_oneway(*month_groups)
                
                res = f"STATISTICAL SIGNIFICANCE (ANOVA)\n"
                res += "-"*30 + "\n"
                res += f"F-Score: {f_stat:.4f}\n"
                res += f"P-Value: {p_val:.4f}\n"
                res += "Conclusion: " + ("Significant Seasonality detected." if p_val < 0.05 else "No significant difference between months.")
                
                self.root.after(0, lambda: self._show_text_result(self.stats_frame, res))
            except Exception as e:
                self.logger.error(e)

        threading.Thread(target=worker, daemon=True).start()

    def run_ml_analysis(self):
        if not self.data_ready: return
        
        def worker():
            try:
                # Prep data for ML (Sequences of 12 months)
                data = self.monthly_ret.values
                X = []
                for i in range(12, len(data)):
                    X.append(data[i-12:i])
                X = np.array(X)
                
                # PCA
                pca = PCA(n_components=2)
                X_pca = pca.fit_transform(X)
                
                # GMM Clustering
                gmm = GaussianMixture(n_components=3, random_state=42)
                labels = gmm.fit_predict(X_pca)
                
                # Isolation Forest (Anomalies)
                iso = IsolationForest(contamination=0.05, random_state=42)
                anomalies = iso.fit_predict(X_pca)
                
                # Visualization on Main Thread
                self.root.after(0, lambda: self._plot_ml_results(X_pca, labels, anomalies))
                
            except Exception as e:
                self.logger.error(e)
        
        threading.Thread(target=worker, daemon=True).start()

    def _show_text_result(self, frame, text):
        self._clear_frame(frame)
        t = tk.Text(frame, font=('Consolas', 10), wrap=tk.WORD, padx=20, pady=20)
        t.pack(fill=tk.BOTH, expand=True)
        t.insert(1.0, text)

    def _plot_ml_results(self, X_pca, labels, anomalies):
        self._clear_frame(self.ml_frame)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Clusters
        ax1.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', alpha=0.6)
        ax1.set_title("Market Regimes (GMM Clustering)")
        ax1.set_xlabel("PCA Component 1")
        
        # Anomalies
        colors_anom = ['red' if x == -1 else 'blue' for x in anomalies]
        ax2.scatter(X_pca[:, 0], X_pca[:, 1], c=colors_anom, alpha=0.6)
        ax2.set_title("Anomaly Detection (Isolation Forest)")
        ax2.set_xlabel("PCA Component 1")
        
        canvas = FigureCanvasTkAgg(fig, self.ml_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # --- Export Handlers ---
    def export_to_excel(self):
        if not self.data_ready: return
        f = filedialog.asksaveasfilename(defaultextension=".xlsx", filetypes=[("Excel", "*.xlsx")])
        if f:
            try:
                with pd.ExcelWriter(f) as writer:
                    self.pivot.to_excel(writer, sheet_name='Monthly_Grid')
                    self.df.to_excel(writer, sheet_name='Raw_Data')
                messagebox.showinfo("Success", "Export Complete")
            except Exception as e:
                messagebox.showerror("Error", str(e))

    def generate_pdf_report(self):
        if not self.data_ready: return
        f = filedialog.asksaveasfilename(defaultextension=".pdf", filetypes=[("PDF", "*.pdf")])
        if f:
            try:
                doc = SimpleDocTemplate(f, pagesize=A4)
                elements = []
                styles = getSampleStyleSheet()
                
                elements.append(Paragraph(f"Market Analysis: {self.ticker}", styles['Title']))
                elements.append(Paragraph(f"Period: {self.start_year_var.get()} - {self.end_date_var.get()}", styles['Normal']))
                
                data = [['Month', 'Average Return']]
                for i, m in enumerate(self.months):
                    if i+1 in self.month_avg.index:
                        data.append([m, f"{self.month_avg[i+1]*100:.2f}%"])
                
                t = Table(data)
                t.setStyle(TableStyle([
                    ('BACKGROUND', (0,0), (-1,0), colors.grey),
                    ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
                    ('ALIGN', (0,0), (-1,-1), 'CENTER'),
                    ('GRID', (0,0), (-1,-1), 1, colors.black)
                ]))
                elements.append(t)
                doc.build(elements)
                messagebox.showinfo("Success", "PDF Generated")
            except Exception as e:
                messagebox.showerror("Error", str(e))

if __name__ == "__main__":
    app = JSEAnalyzer()
    app.root.mainloop()

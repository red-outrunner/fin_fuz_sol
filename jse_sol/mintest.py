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
from sklearn.ensemble import IsolationForest
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet

# --- Configuration & Constants ---
matplotlib.use('TkAgg')
# Using a cleaner, more professional style for charts
plt.style.use('seaborn-v0_8-whitegrid')

VERSION = "3.1.0"
CACHE_DIR = f"cache_v{VERSION.replace('.', '_')}"
os.makedirs(CACHE_DIR, exist_ok=True)

# Modern Color Palette
COLORS = {
    'sidebar_bg': '#1e293b',      # Dark Slate
    'sidebar_fg': '#f8fafc',      # Off-white text
    'main_bg': '#f1f5f9',         # Light Grey Blue
    'card_bg': '#ffffff',         # White
    'accent': '#3b82f6',          # Bright Blue
    'accent_hover': '#2563eb',    # Darker Blue
    'success': '#10b981',         # Emerald
    'danger': '#ef4444',          # Red
    'text_primary': '#334155',    # Dark Grey
    'text_secondary': '#64748b'   # Medium Grey
}

TICKER_OPTIONS = {
    "🇺🇸 S&P 500": "^GSPC",
    "🇺🇸 Nasdaq 100": "^NDX",
    "🇿🇦 JSE Top 40": "^J200.JO",
    "🇬🇧 FTSE 100": "^FTSE",
    "🌍 MSCI World": "^MXWO",
    "🥇 Gold": "GC=F",
    "₿ Bitcoin": "BTC-USD",
    "🍎 Apple": "AAPL",
    "🚗 Tesla": "TSLA",
    "🛒 Amazon": "AMZN"
}

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# --- Logic / Data Layer ---
class MarketDataManager:
    """Handles data fetching and financial calculations."""
    
    def __init__(self):
        self.data = None

    def get_cache_path(self, ticker, start):
        # Simple cache key based on ticker and start year
        clean_ticker = ticker.replace('^', '').replace('.', '_')
        return os.path.join(CACHE_DIR, f"{clean_ticker}_{start}.pkl")

    def fetch_data(self, ticker, start_year):
        current_date = datetime.now().strftime("%Y-%m-%d")
        cache_path = self.get_cache_path(ticker, start_year)
        
        # 1. Try Cache (Valid for 24 hours)
        if os.path.exists(cache_path):
            file_age = datetime.now() - datetime.fromtimestamp(os.path.getmtime(cache_path))
            if file_age < timedelta(hours=24):
                try:
                    with open(cache_path, 'rb') as f:
                        logger.info(f"Loaded cache for {ticker}")
                        return pickle.load(f)
                except Exception:
                    pass # Corrupt cache, ignore

        # 2. Fetch Live Data
        start_date = f"{start_year}-01-01"
        # Auto_adjust=True handles splits/dividends better for "Total Return" view
        df = yf.download(ticker, start=start_date, end=current_date, progress=False, auto_adjust=True)
        
        if df.empty:
            raise ValueError(f"No data found for {ticker}")

        # 3. Process Data
        processed_data = self._process_financials(df)
        
        # 4. Save Cache
        with open(cache_path, 'wb') as f:
            pickle.dump(processed_data, f)
            
        return processed_data

    def _process_financials(self, df):
        # Clean column names if multi-index
        if isinstance(df.columns, pd.MultiIndex):
            # Keep only the ticker level if possible, or just the Close
            try:
                s = df['Close'].iloc[:, 0] 
            except:
                s = df.iloc[:, 0]
        else:
            s = df['Close'] if 'Close' in df.columns else df.iloc[:, 0]

        s = s.dropna()

        # 1. Monthly Returns
        monthly_prices = s.resample('ME').last()
        monthly_ret = monthly_prices.pct_change().dropna()

        # 2. Pivot Table (Seasonality)
        df_ret = monthly_ret.to_frame(name='ret')
        df_ret['year'] = df_ret.index.year
        df_ret['month'] = df_ret.index.month
        pivot = df_ret.pivot_table(index='year', columns='month', values='ret')

        # 3. Wealth Index (Growth of $10,000)
        # We use daily returns for a smoother chart
        daily_ret = s.pct_change().fillna(0)
        wealth_index = 10000 * (1 + daily_ret).cumprod()

        # 4. Key Metrics Calculation
        # CAGR
        total_years = (s.index[-1] - s.index[0]).days / 365.25
        total_return = (s.iloc[-1] / s.iloc[0]) - 1
        cagr = (1 + total_return) ** (1 / total_years) - 1 if total_years > 0 else 0

        # Volatility (Annualized)
        volatility = monthly_ret.std() * np.sqrt(12)

        # Sharpe Ratio (Assume Risk Free Rate ~ 3%)
        rf = 0.03
        sharpe = (cagr - rf) / volatility if volatility != 0 else 0

        # Max Drawdown
        rolling_max = s.cummax()
        drawdown = (s - rolling_max) / rolling_max
        max_drawdown = drawdown.min()

        return {
            'monthly_ret': monthly_ret,
            'pivot': pivot,
            'wealth_index': wealth_index,
            'metrics': {
                'cagr': cagr,
                'volatility': volatility,
                'sharpe': sharpe,
                'max_dd': max_drawdown
            },
            'raw_series': s
        }

# --- Visualization Layer ---
class PlottingEngine:
    """Handles all Matplotlib rendering."""
    
    MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    @staticmethod
    def clear_frame(frame):
        for widget in frame.winfo_children():
            widget.destroy()

    @staticmethod
    def plot_wealth_index(frame, wealth_data, ticker_name):
        PlottingEngine.clear_frame(frame)
        fig, ax = plt.subplots(figsize=(10, 4), dpi=100)
        fig.patch.set_facecolor('#f8fafc') # Match app background slightly
        ax.set_facecolor('#ffffff')

        ax.plot(wealth_data.index, wealth_data.values, color=COLORS['accent'], linewidth=2)
        ax.fill_between(wealth_data.index, wealth_data.values, 10000, where=(wealth_data.values > 10000), 
                        color=COLORS['accent'], alpha=0.1)
        
        ax.set_title(f"Growth of $10,000 Investment ({ticker_name})", fontsize=12, fontweight='bold', color=COLORS['text_primary'])
        ax.set_ylabel("Portfolio Value ($)", fontsize=9)
        ax.grid(True, linestyle='--', alpha=0.5)
        
        # Format Y-axis with dollar signs
        ax.yaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter('${x:,.0f}'))

        fig.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    @staticmethod
    def plot_seasonality_heatmap(frame, pivot):
        PlottingEngine.clear_frame(frame)
        fig, ax = plt.subplots(figsize=(10, 5), dpi=100)
        
        sns.heatmap(pivot * 100, ax=ax, cmap="RdYlGn", center=0, annot=True, fmt=".1f",
                    cbar_kws={'label': 'Return (%)'}, linewidths=0.5, linecolor='white')
        
        ax.set_title("Monthly Returns Heatmap (%)", fontsize=12, fontweight='bold')
        ax.set_xticklabels(PlottingEngine.MONTHS)
        ax.set_xlabel("")
        ax.set_ylabel("")
        
        fig.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    @staticmethod
    def plot_monthly_distribution(frame, monthly_ret):
        PlottingEngine.clear_frame(frame)
        fig, ax = plt.subplots(figsize=(10, 4), dpi=100)
        
        sns.histplot(monthly_ret * 100, kde=True, color=COLORS['sidebar_bg'], ax=ax)
        
        ax.set_title("Distribution of Monthly Returns", fontsize=12, fontweight='bold')
        ax.set_xlabel("Return (%)")
        ax.axvline(0, color=COLORS['danger'], linestyle='--')
        
        fig.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

# --- UI Components ---
class KPI_Card(tk.Frame):
    """Reusable UI Component for top stats."""
    def __init__(self, parent, title, value, color_logic=None):
        super().__init__(parent, bg=COLORS['card_bg'], bd=1, relief="solid")
        self.title_lbl = tk.Label(self, text=title.upper(), bg=COLORS['card_bg'], fg=COLORS['text_secondary'], font=("Segoe UI", 8, "bold"))
        self.title_lbl.pack(anchor="w", padx=10, pady=(10, 0))
        
        self.value_lbl = tk.Label(self, text=value, bg=COLORS['card_bg'], fg=COLORS['text_primary'], font=("Segoe UI", 18, "bold"))
        self.value_lbl.pack(anchor="w", padx=10, pady=(5, 10))

        if color_logic:
            self.value_lbl.config(fg=color_logic)

# --- Main Application ---
class DashboardApp:
    def __init__(self, root):
        self.root = root
        self.root.title(f"Investor Pro | Market Analytics v{VERSION}")
        self.root.geometry("1366x850")
        self.root.state('zoomed') # Maximize window
        self.root.configure(bg=COLORS['main_bg'])

        self.data_manager = MarketDataManager()
        self.current_data = None
        
        self._setup_styles()
        self._build_layout()
        
        # Initial State
        self.ticker_var.set(list(TICKER_OPTIONS.values())[0])

    def _setup_styles(self):
        style = ttk.Style()
        style.theme_use('clam')
        
        # Define Custom Styles for Modern Look
        style.configure("Sidebar.TFrame", background=COLORS['sidebar_bg'])
        style.configure("Main.TFrame", background=COLORS['main_bg'])
        
        # Notebook (Tabs) Styling
        style.configure("TNotebook", background=COLORS['main_bg'], borderwidth=0)
        style.configure("TNotebook.Tab", padding=[15, 10], font=("Segoe UI", 10), background="#e2e8f0")
        style.map("TNotebook.Tab", background=[("selected", COLORS['accent'])], foreground=[("selected", "white")])
        
        # Buttons
        style.configure("Action.TButton", font=("Segoe UI", 11, "bold"), background=COLORS['accent'], foreground="white", borderwidth=0)
        style.map("Action.TButton", background=[('active', COLORS['accent_hover'])])

    def _build_layout(self):
        # 1. Main Container
        container = ttk.Frame(self.root)
        container.pack(fill=tk.BOTH, expand=True)

        # 2. Left Sidebar (Fixed Width)
        sidebar = ttk.Frame(container, style="Sidebar.TFrame", width=280)
        sidebar.pack(side=tk.LEFT, fill=tk.Y)
        sidebar.pack_propagate(False)

        self._build_sidebar(sidebar)

        # 3. Main Content Area
        content_area = ttk.Frame(container, style="Main.TFrame")
        content_area.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=20, pady=20)

        self._build_kpi_section(content_area)
        self._build_notebook(content_area)

    def _build_sidebar(self, parent):
        # Logo / Header
        header = tk.Label(parent, text="INVESTOR PRO", bg=COLORS['sidebar_bg'], fg='white', font=("Segoe UI", 16, "bold"))
        header.pack(pady=(30, 10))
        
        ver = tk.Label(parent, text=f"v{VERSION}", bg=COLORS['sidebar_bg'], fg=COLORS['text_secondary'], font=("Segoe UI", 9))
        ver.pack(pady=(0, 30))

        # Controls Container
        controls = tk.Frame(parent, bg=COLORS['sidebar_bg'], padx=20)
        controls.pack(fill=tk.X)

        # Ticker
        tk.Label(controls, text="ASSET SELECTION", bg=COLORS['sidebar_bg'], fg=COLORS['text_secondary'], font=("Segoe UI", 8, "bold")).pack(anchor="w")
        
        self.ticker_var = tk.StringVar()
        self.ticker_combo = ttk.Combobox(controls, textvariable=self.ticker_var, font=("Segoe UI", 11), state="readonly")
        self.ticker_combo['values'] = list(TICKER_OPTIONS.keys())
        self.ticker_combo.current(0)
        self.ticker_combo.pack(fill=tk.X, pady=(5, 20))

        # Date Range
        tk.Label(controls, text="TIMEFRAME (START YEAR)", bg=COLORS['sidebar_bg'], fg=COLORS['text_secondary'], font=("Segoe UI", 8, "bold")).pack(anchor="w")
        self.start_year_var = tk.IntVar(value=2015)
        
        # Year Slider for better UX
        current_year = datetime.now().year
        self.year_scale = tk.Scale(controls, from_=1990, to=current_year-1, variable=self.start_year_var, 
                                   orient=tk.HORIZONTAL, bg=COLORS['sidebar_bg'], fg='white', 
                                   highlightthickness=0, font=("Segoe UI", 10))
        self.year_scale.pack(fill=tk.X, pady=(0, 20))

        # Analyze Button
        ttk.Button(controls, text="RUN ANALYSIS", style="Action.TButton", command=self.run_analysis).pack(fill=tk.X, pady=10, ipady=5)

        # Status Indicator
        self.status_var = tk.StringVar(value="Ready")
        status_lbl = tk.Label(parent, textvariable=self.status_var, bg=COLORS['sidebar_bg'], fg=COLORS['success'], font=("Segoe UI", 9), anchor="w")
        status_lbl.pack(side=tk.BOTTOM, fill=tk.X, padx=20, pady=20)

        # Export Section
        tk.Label(controls, text="EXPORTS", bg=COLORS['sidebar_bg'], fg=COLORS['text_secondary'], font=("Segoe UI", 8, "bold")).pack(anchor="w", pady=(20, 5))
        
        btn_frame = tk.Frame(controls, bg=COLORS['sidebar_bg'])
        btn_frame.pack(fill=tk.X)
        
        ttk.Button(btn_frame, text="Excel", command=self.export_excel).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 2))
        ttk.Button(btn_frame, text="PDF", command=self.generate_pdf).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(2, 0))

    def _build_kpi_section(self, parent):
        # A grid of 4 cards
        self.kpi_frame = tk.Frame(parent, bg=COLORS['main_bg'])
        self.kpi_frame.pack(fill=tk.X, pady=(0, 20))
        
        # Placeholders for cards, accessible via self.cards dict
        self.kpi_widgets = {}
        metrics = ["CAGR", "Volatility", "Sharpe Ratio", "Max Drawdown"]
        
        for i, metric in enumerate(metrics):
            card = KPI_Card(self.kpi_frame, metric, "---")
            card.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0 if i == 0 else 15, 0))
            self.kpi_widgets[metric] = card

    def _build_notebook(self, parent):
        self.notebook = ttk.Notebook(parent)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        self.tab_growth = ttk.Frame(self.notebook, style="Main.TFrame")
        self.tab_heat = ttk.Frame(self.notebook, style="Main.TFrame")
        self.tab_dist = ttk.Frame(self.notebook, style="Main.TFrame")
        self.tab_ai = ttk.Frame(self.notebook, style="Main.TFrame")

        self.notebook.add(self.tab_growth, text="  📈 Performance  ")
        self.notebook.add(self.tab_heat, text="  📅 Seasonality  ")
        self.notebook.add(self.tab_dist, text="  📊 Distribution  ")
        self.notebook.add(self.tab_ai, text="  🤖 AI Insights  ")

    # --- Application Logic ---
    def run_analysis(self):
        # 1. Get UI Inputs
        ticker_name = self.ticker_var.get()
        ticker_symbol = TICKER_OPTIONS.get(ticker_name)
        start_year = self.start_year_var.get()

        self.status_var.set(f"Fetching data for {ticker_name}...")
        self.root.config(cursor="watch")
        
        # 2. Threaded Execution
        threading.Thread(target=self._bg_task, args=(ticker_symbol, start_year, ticker_name), daemon=True).start()

    def _bg_task(self, ticker, start, name):
        try:
            data = self.data_manager.fetch_data(ticker, start)
            self.root.after(0, lambda: self._update_ui(data, name))
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", str(e)))
            self.root.after(0, lambda: self.status_var.set("Error"))
        finally:
            self.root.after(0, lambda: self.root.config(cursor=""))

    def _update_ui(self, data, name):
        self.current_data = data
        self.status_var.set(f"Updated: {name}")
        
        # 1. Update KPIs
        metrics = data['metrics']
        
        # Helper to format colors
        def color_ret(val): return COLORS['success'] if val > 0 else COLORS['danger']
        
        self.kpi_widgets["CAGR"].value_lbl.config(text=f"{metrics['cagr']:.2%}", fg=color_ret(metrics['cagr']))
        self.kpi_widgets["Volatility"].value_lbl.config(text=f"{metrics['volatility']:.2%}", fg=COLORS['text_primary'])
        self.kpi_widgets["Sharpe Ratio"].value_lbl.config(text=f"{metrics['sharpe']:.2f}", fg=color_ret(metrics['sharpe']))
        self.kpi_widgets["Max Drawdown"].value_lbl.config(text=f"{metrics['max_dd']:.2%}", fg=COLORS['danger'])

        # 2. Update Charts
        PlottingEngine.plot_wealth_index(self.tab_growth, data['wealth_index'], name)
        PlottingEngine.plot_seasonality_heatmap(self.tab_heat, data['pivot'])
        PlottingEngine.plot_monthly_distribution(self.tab_dist, data['monthly_ret'])
        
        # 3. Update AI Tab
        self._run_ai_scan(data['monthly_ret'])

    def _run_ai_scan(self, monthly_ret):
        # Simple Anomaly Detection
        PlottingEngine.clear_frame(self.tab_ai)
        
        txt_frame = tk.Frame(self.tab_ai, bg="white", bd=1, relief="solid")
        txt_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        txt = tk.Text(txt_frame, font=("Consolas", 11), relief="flat", padx=20, pady=20)
        txt.pack(fill=tk.BOTH, expand=True)
        
        try:
            # Reshape for sklearn
            X = monthly_ret.values.reshape(-1, 1)
            
            # Isolation Forest for anomalies (Market Crashes/Spikes)
            iso = IsolationForest(contamination=0.05, random_state=42)
            preds = iso.fit_predict(X)
            
            anomalies = monthly_ret[preds == -1]
            
            txt.insert(tk.END, "🤖 ARTIFICIAL INTELLIGENCE INSIGHTS\n")
            txt.insert(tk.END, "==================================\n\n")
            
            txt.insert(tk.END, f"Analysis Mode: Unsupervised Anomaly Detection\n")
            txt.insert(tk.END, f"Model: Isolation Forest\n\n")
            
            txt.insert(tk.END, f"🔍 FINDINGS:\n")
            txt.insert(tk.END, f"The model analyzed {len(monthly_ret)} months of data and identified {len(anomalies)} anomaly events.\n")
            txt.insert(tk.END, "These represent extreme market movements (tail risks) separate from normal volatility.\n\n")
            
            txt.insert(tk.END, "⚠️ EXTREME EVENTS DETECTED:\n")
            for date, ret in anomalies.sort_values().items():
                txt.insert(tk.END, f" • {date.strftime('%Y-%m')}: {ret:>7.2%}\n")
                
        except Exception as e:
            txt.insert(tk.END, f"AI Analysis Failed: {e}")

    def export_excel(self):
        if not self.current_data: return
        f = filedialog.asksaveasfilename(defaultextension=".xlsx", filetypes=[("Excel", "*.xlsx")])
        if f:
            with pd.ExcelWriter(f) as writer:
                self.current_data['pivot'].to_excel(writer, sheet_name="Seasonality")
                self.current_data['monthly_ret'].to_excel(writer, sheet_name="Monthly Returns")
                pd.DataFrame([self.current_data['metrics']]).to_excel(writer, sheet_name="KPIs")
            messagebox.showinfo("Export", "Excel file saved successfully.")

    def generate_pdf(self):
        if not self.current_data: return
        f = filedialog.asksaveasfilename(defaultextension=".pdf", filetypes=[("PDF", "*.pdf")])
        if f:
            doc = SimpleDocTemplate(f, pagesize=A4)
            styles = getSampleStyleSheet()
            elements = []
            
            elements.append(Paragraph(f"Investment Report: {self.ticker_var.get()}", styles['Title']))
            elements.append(Paragraph(f"Generated on {datetime.now().strftime('%Y-%m-%d')}", styles['Normal']))
            
            metrics = self.current_data['metrics']
            elements.append(Paragraph(f"CAGR: {metrics['cagr']:.2%}", styles['Heading2']))
            elements.append(Paragraph(f"Sharpe Ratio: {metrics['sharpe']:.2f}", styles['Heading2']))
            
            doc.build(elements)
            messagebox.showinfo("Export", "PDF Report generated.")

if __name__ == "__main__":
    root = tk.Tk()
    app = DashboardApp(root)
    root.mainloop()
    root.mainloop()

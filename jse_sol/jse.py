# jse_monthly_profile.py
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import tkinter as tk
from tkinter import ttk, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib
matplotlib.use('TkAgg')

class JSEAnalyzer:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("JSE Monthly Return Analyzer")
        self.root.geometry("1200x800")
        
        # Default values
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
        self.months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
        
        self.setup_ui()
        
    def setup_ui(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configuration frame
        config_frame = ttk.LabelFrame(main_frame, text="Configuration", padding="10")
        config_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Ticker input
        ttk.Label(config_frame, text="Ticker:").grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        self.ticker_var = tk.StringVar(value=self.ticker)
        ticker_entry = ttk.Entry(config_frame, textvariable=self.ticker_var, width=15)
        ticker_entry.grid(row=0, column=1, sticky=tk.W, padx=(0, 20))
        
        # Date range options
        ttk.Label(config_frame, text="Date Range:").grid(row=0, column=2, sticky=tk.W, padx=(0, 5))
        self.date_range_var = tk.StringVar(value="Custom")
        date_range_combo = ttk.Combobox(config_frame, textvariable=self.date_range_var, 
                                       values=["Custom", "Last 5 Years", "Last 10 Years", "Last 20 Years", "All Data"], 
                                       state="readonly", width=15)
        date_range_combo.grid(row=0, column=3, sticky=tk.W, padx=(0, 20))
        date_range_combo.bind('<<ComboboxSelected>>', self.on_date_range_change)
        
        # Custom date inputs
        ttk.Label(config_frame, text="Start Year:").grid(row=0, column=4, sticky=tk.W, padx=(0, 5))
        self.start_year_var = tk.IntVar(value=self.start_year)
        self.start_year_entry = ttk.Entry(config_frame, textvariable=self.start_year_var, width=8)
        self.start_year_entry.grid(row=0, column=5, sticky=tk.W, padx=(0, 20))
        
        ttk.Label(config_frame, text="End Date:").grid(row=0, column=6, sticky=tk.W, padx=(0, 5))
        self.end_date_var = tk.StringVar(value=self.end_date)
        self.end_date_entry = ttk.Entry(config_frame, textvariable=self.end_date_var, width=12)
        self.end_date_entry.grid(row=0, column=7, sticky=tk.W, padx=(0, 20))
        
        # Analyze button
        analyze_btn = ttk.Button(config_frame, text="Analyze Data", command=self.analyze_data)
        analyze_btn.grid(row=0, column=8, padx=(10, 0))
        
        # Export button
        export_btn = ttk.Button(config_frame, text="Export to Excel", command=self.export_to_excel, state=tk.DISABLED)
        export_btn.grid(row=0, column=9, padx=(10, 0))
        self.export_btn = export_btn
        
        # Results notebook
        notebook = ttk.Notebook(main_frame)
        notebook.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(10, 0))
        
        # Tab 1: Average Returns Bar Chart
        self.bar_frame = ttk.Frame(notebook)
        notebook.add(self.bar_frame, text="Average Returns")
        
        # Tab 2: Heatmap
        self.heatmap_frame = ttk.Frame(notebook)
        notebook.columnconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(1, weight=1)
        notebook.add(self.heatmap_frame, text="Year-Month Heatmap")
        
        # Tab 3: Risk-Return Scatter
        self.scatter_frame = ttk.Frame(notebook)
        notebook.add(self.scatter_frame, text="Risk vs Return")
        
        # Tab 4: Summary Stats
        self.summary_frame = ttk.Frame(notebook)
        notebook.add(self.summary_frame, text="Summary Statistics")
        self.summary_text = tk.Text(self.summary_frame, wrap=tk.WORD)
        self.summary_text.pack(fill=tk.BOTH, expand=True)
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN)
        status_bar.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        
    def on_date_range_change(self, event=None):
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
        else:  # Custom
            self.start_year_entry.config(state=tk.NORMAL)
            self.end_date_entry.config(state=tk.NORMAL)
    
    def analyze_data(self):
        try:
            self.status_var.set("Downloading data...")
            self.root.update()
            
            self.ticker = self.ticker_var.get()
            self.start_year = self.start_year_var.get()
            self.end_date = self.end_date_var.get()
            
            # Download data
            try:
                self.data = yf.download(self.ticker, start=f"{self.start_year}-01-01", end=self.end_date, 
                                       progress=False, auto_adjust=False)
            except Exception as e:
                messagebox.showerror("Error", f"Error downloading  {e}")
                self.status_var.set("Error downloading data")
                return
            
            if self.data.empty:
                messagebox.showerror("Error", "No data returned — check ticker or internet connection")
                self.status_var.set("No data available")
                return
            
            # Process data
            self.status_var.set("Processing data...")
            self.root.update()
            
            price_col = "Adj Close" if "Adj Close" in self.data.columns else "Close"
            self.monthly = self.data[price_col].resample('ME').last()
            self.monthly_ret = self.monthly.pct_change().dropna()
            
            # Handle case where monthly_ret might be DataFrame instead of Series
            if isinstance(self.monthly_ret, pd.DataFrame):
                self.monthly_ret = self.monthly_ret.iloc[:, 0]  # Take first column if DataFrame
            self.monthly_ret.name = 'ret'
            self.df = self.monthly_ret.to_frame()
            
            self.df['year'] = self.df.index.year
            self.df['month'] = self.df.index.month
            self.pivot = self.df.pivot_table(index='year', columns='month', values='ret')
            
            self.month_avg = self.pivot.mean().sort_index()
            self.month_median = self.pivot.median().sort_index()
            
            # Update UI with results
            self.update_charts()
            self.update_summary()
            
            # Enable export button
            self.export_btn.config(state=tk.NORMAL)
            self.status_var.set(f"Analysis complete. Data from {self.start_year} to {self.end_date[:4]}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error during analysis: {e}")
            self.status_var.set("Analysis failed")
    
    def update_charts(self):
        # Clear existing charts
        for widget in self.bar_frame.winfo_children():
            widget.destroy()
        for widget in self.heatmap_frame.winfo_children():
            widget.destroy()
        for widget in self.scatter_frame.winfo_children():
            widget.destroy()
        
        # Bar chart with hover functionality
        fig1, ax1 = plt.subplots(figsize=(10, 5))
        bars = ax1.bar(range(1, 13), self.month_avg*100)
        ax1.set_xticks(range(1, 13))
        ax1.set_xticklabels(self.months)
        ax1.set_ylabel('Avg monthly return (%)')
        ax1.set_title(f'Average monthly returns for {self.ticker} ({self.start_year} to {self.end_date[:4]})')
        ax1.grid(axis='y', alpha=0.25)
        
        # Add hover functionality to bar chart
        annot1 = ax1.annotate("", xy=(0,0), xytext=(20,20), textcoords="offset points",
                             bbox=dict(boxstyle="round", fc="w"),
                             arrowprops=dict(arrowstyle="->"))
        annot1.set_visible(False)
        
        def update_annot_bar(bar, index):
            x = bar.get_x() + bar.get_width() / 2
            y = bar.get_y() + bar.get_height()
            annot1.xy = (x, y)
            month_name = self.months[index]
            value = self.month_avg.iloc[index] * 100
            text = f"{month_name}: {value:.2f}%"
            annot1.set_text(text)
            annot1.get_bbox_patch().set_alpha(0.8)
        
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
        
        canvas1 = FigureCanvasTkAgg(fig1, self.bar_frame)
        canvas1.draw()
        canvas1.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Heatmap
        fig2, ax2 = plt.subplots(figsize=(12, 8))
        im = sns.heatmap(self.pivot*100, center=0, cmap='vlag', 
                        cbar_kws={'label':'monthly return (%)'}, 
                        linewidths=.5, ax=ax2, annot=True, fmt='.1f')
        ax2.set_xlabel('Month')
        ax2.set_ylabel('Year')
        ax2.set_title(f'Month-by-year returns (%) for {self.ticker}')
        ax2.set_xticks(np.arange(12) + 0.5)
        ax2.set_xticklabels(self.months, rotation=0)
        
        canvas2 = FigureCanvasTkAgg(fig2, self.heatmap_frame)
        canvas2.draw()
        canvas2.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Risk vs Return Scatter with hover functionality
        monthly_stats = pd.DataFrame({
            'month': range(1, 13),
            'avg_return': self.month_avg * 100,
            'std_dev': self.pivot.std() * 100,
            'positive_rate': (self.pivot > 0).sum() / self.pivot.count() * 100
        })
        
        fig3, ax3 = plt.subplots(figsize=(12, 8))
        scatter = ax3.scatter(monthly_stats['std_dev'], monthly_stats['avg_return'], 
                             c=monthly_stats['positive_rate'], 
                             cmap='RdYlGn', s=200, alpha=0.7, edgecolors='black')
        
        # Add month labels
        for i, month in enumerate(self.months):
            ax3.annotate(month, (monthly_stats['std_dev'].iloc[i], monthly_stats['avg_return'].iloc[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=9, weight='bold')
        
        ax3.set_xlabel('Monthly Return Standard Deviation (%)')
        ax3.set_ylabel('Average Monthly Return (%)')
        ax3.set_title(f'Risk vs Return by Month for {self.ticker}\n(Color = Positive Return Rate)')
        cbar = plt.colorbar(scatter, ax=ax3, label='Positive Return Rate (%)')
        ax3.grid(True, alpha=0.3)
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax3.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        
        # Add hover functionality to scatter plot
        annot3 = ax3.annotate("", xy=(0,0), xytext=(20,20), textcoords="offset points",
                             bbox=dict(boxstyle="round", fc="w"),
                             arrowprops=dict(arrowstyle="->"))
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
            annot3.get_bbox_patch().set_alpha(0.8)
        
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
        
        canvas3 = FigureCanvasTkAgg(fig3, self.scatter_frame)
        canvas3.draw()
        canvas3.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def update_summary(self):
        self.summary_text.delete(1.0, tk.END)
        summary = f"Monthly Return Summary for {self.ticker} ({self.start_year}-{self.end_date[:4]}):\n"
        summary += "=" * 50 + "\n"
        for month_num, month_name in enumerate(self.months, 1):
            if month_num in self.month_avg.index:
                avg_ret = self.month_avg[month_num] * 100
                summary += f"{month_name}: {avg_ret:+6.2f}%\n"
        
        self.summary_text.insert(1.0, summary)
    
    def export_to_excel(self):
        if self.pivot is None:
            messagebox.showwarning("Warning", "No data to export. Please analyze data first.")
            return
            
        try:
            filename = f"{self.ticker.replace('^', '').replace('.JO', '')}_monthly_analysis_{self.start_year}_{self.end_date[:4]}.xlsx"
            
            with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                # Sheet 1: Year-by-Month Returns
                self.pivot.to_excel(writer, sheet_name='Year_Month_Returns')
                
                # Sheet 2: Monthly Summary Stats
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
                
                # Sheet 3: Raw Data
                raw_data = pd.DataFrame({
                    'Date': self.pivot.index,
                    'Year': self.pivot.index,
                    **{f'M{col}': self.pivot[col].values for col in self.pivot.columns}
                })
                raw_data.to_excel(writer, sheet_name='Raw_Data', index=False)
            
            messagebox.showinfo("Success", f"Data exported to {filename}")
            self.status_var.set(f"Data exported to {filename}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error exporting  {e}")
            self.status_var.set("Export failed")
    
    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    app = JSEAnalyzer()
    app.run()

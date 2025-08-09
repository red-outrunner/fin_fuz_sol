# jse_monthly_profile.py
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from openpyxl import Workbook

# ----- CONFIG -----
ticker = "^J203.JO"           # FTSE/JSE All Share (Yahoo notation). Change to Top40 TRI if you have it.
start_year = 1990             # adjust to the start year you want
end_date = datetime.today().strftime("%Y-%m-%d")
# ------------------

# 1) download daily data, then resample to month-end
try:
    data = yf.download(ticker, start=f"{start_year}-01-01", end=end_date, 
                       progress=False, auto_adjust=False)
except Exception as e:
    raise SystemExit(f"Error downloading data: {e}")

if data.empty:
    raise SystemExit("No data returned — check ticker or internet connection")

# use 'Adj Close' if available; otherwise use 'Close'
price_col = "Adj Close" if "Adj Close" in data.columns else "Close"
monthly = data[price_col].resample('ME').last()  # Changed 'M' to 'ME'

# 2) compute month returns (percent)
monthly_ret = monthly.pct_change().dropna()

# 3) build a DataFrame indexed by year, columns=month number
# Handle case where monthly_ret might be DataFrame instead of Series
if isinstance(monthly_ret, pd.DataFrame):
    monthly_ret = monthly_ret.iloc[:, 0]  # Take first column if DataFrame
monthly_ret.name = 'ret'
df = monthly_ret.to_frame()

df['year'] = df.index.year
df['month'] = df.index.month
pivot = df.pivot_table(index='year', columns='month', values='ret')

# 4) month average & median (since start_year)
month_avg = pivot.mean().sort_index()
month_median = pivot.median().sort_index()

# 5) Plot: average monthly returns (bar) + heatmap of yearly month returns
months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

plt.figure(figsize=(10,5))
plt.bar(range(1,13), month_avg*100)
plt.xticks(range(1,13), months)
plt.ylabel('Avg monthly return (%)')
plt.title(f'Average monthly returns for {ticker} ({start_year} to {end_date[:4]})')
plt.grid(axis='y', alpha=0.25)
plt.show()

# Heatmap
plt.figure(figsize=(12,8))
sns.heatmap(pivot*100, center=0, cmap='vlag', cbar_kws={'label':'monthly return (%)'}, linewidths=.5)
plt.xlabel('Month')
plt.ylabel('Year')
plt.title(f'Month-by-year returns (%) for {ticker}')
plt.xticks(np.arange(12)+.5, months, rotation=0)
plt.show()

# Scatter plot: Risk vs Return by Month (Option C)
monthly_stats = pd.DataFrame({
    'month': range(1, 13),
    'avg_return': month_avg * 100,
    'std_dev': pivot.std() * 100,
    'positive_rate': (pivot > 0).sum() / pivot.count() * 100
})

plt.figure(figsize=(12, 8))
scatter = plt.scatter(monthly_stats['std_dev'], monthly_stats['avg_return'], 
                     c=monthly_stats['positive_rate'], 
                     cmap='RdYlGn', s=200, alpha=0.7, edgecolors='black')

# Add month labels
for i, month in enumerate(months):
    plt.annotate(month, (monthly_stats['std_dev'].iloc[i], monthly_stats['avg_return'].iloc[i]), 
                xytext=(5, 5), textcoords='offset points', fontsize=9, weight='bold')

plt.xlabel('Monthly Return Standard Deviation (%)')
plt.ylabel('Average Monthly Return (%)')
plt.title(f'Risk vs Return by Month for {ticker}\n(Color = Positive Return Rate)')
plt.colorbar(label='Positive Return Rate (%)')
plt.grid(True, alpha=0.3)
plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
plt.show()

# Export to Excel function
def export_to_excel(pivot, month_avg, month_median, ticker, start_year, end_date):
    """Export monthly return data to Excel with multiple sheets"""
    
    filename = f"{ticker.replace('^', '').replace('.JO', '')}_monthly_analysis_{start_year}_{end_date[:4]}.xlsx"
    
    with pd.ExcelWriter(filename, engine='openpyxl') as writer:
        # Sheet 1: Year-by-Month Returns
        pivot.to_excel(writer, sheet_name='Year_Month_Returns')
        
        # Sheet 2: Monthly Summary Stats
        summary_stats = pd.DataFrame({
            'Month': range(1, 13),
            'Month_Name': months,
            'Average_Return_%': month_avg.values * 100,
            'Median_Return_%': month_median.values * 100,
            'Std_Dev_%': pivot.std().values * 100,
            'Best_Return_%': pivot.max().values * 100,
            'Worst_Return_%': pivot.min().values * 100,
            'Positive_Months_Count': (pivot > 0).sum().values,
            'Total_Months_Count': pivot.count().values,
            'Positive_Rate_%': ((pivot > 0).sum() / pivot.count() * 100).values
        })
        summary_stats.to_excel(writer, sheet_name='Monthly_Summary', index=False)
        
        # Sheet 3: Raw Data
        raw_data = pd.DataFrame({
            'Date': pivot.index,
            'Year': pivot.index,
            **{f'M{col}': pivot[col].values for col in pivot.columns}
        })
        raw_data.to_excel(writer, sheet_name='Raw_Data', index=False)
    
    print(f"Data exported to {filename}")

# Call export function
export_to_excel(pivot, month_avg, month_median, ticker, start_year, end_date)

# Optional: Print summary statistics
print(f"\nMonthly Return Summary for {ticker} ({start_year}-{end_date[:4]}):")
print("=" * 50)
for month_num, month_name in enumerate(months, 1):
    if month_num in month_avg.index:
        avg_ret = month_avg[month_num] * 100
        print(f"{month_name}: {avg_ret:+6.2f}%")

# jse_monthly_profile.py
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# ----- CONFIG -----
ticker = "^J203.JO"           # FTSE/JSE All Share (Yahoo notation). Change to Top40 TRI if you have it.
start_year = 1990             # adjust to the start year you want
end_date = datetime.today().strftime("%Y-%m-%d")
# ------------------

# 1) download daily data, then resample to month-end
data = yf.download(ticker, start=f"{start_year}-01-01", end=end_date, 
                   progress=False, auto_adjust=False)

if data.empty:
    raise SystemExit("No data returned — check ticker or internet connection")

# use 'Adj Close' if available; otherwise use 'Close'
price_col = "Adj Close" if "Adj Close" in data.columns else "Close"
monthly = data[price_col].resample('ME').last()  # Changed 'M' to 'ME'

# 2) compute month returns (percent)
monthly_ret = monthly.pct_change().dropna()

# 3) build a DataFrame indexed by year, columns=month number
# monthly_ret is already a Series, so we can use to_frame()
# df = monthly_ret.to_frame(name='ret')
if isinstance(monthly_ret, pd.Series):
    df = monthly_ret.to_frame(name='ret')
else:
    df = monthly_ret.copy()
    df.columns = ['ret']
df['year'] = df.index.year
df['month'] = df.index.month
pivot = df.pivot_table(index='year', columns='month', values='ret')

# 4) month average & median (since start_year)
month_avg = pivot.mean().sort_index()
month_median = pivot.median().sort_index()

# 5) Plot: average monthly returns (bar) + heatmap of yearly month returns
plt.figure(figsize=(10,5))
plt.bar(range(1,13), month_avg*100)
plt.xticks(range(1,13), ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])
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
plt.xticks(np.arange(12)+.5, ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'], rotation=0)
plt.show()

# Optional: Print summary statistics
print(f"\nMonthly Return Summary for {ticker} ({start_year}-{end_date[:4]}):")
print("=" * 50)
for month_num, month_name in enumerate(['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'], 1):
    if month_num in month_avg.index:
        avg_ret = month_avg[month_num] * 100
        print(f"{month_name}: {avg_ret:+6.2f}%")

Global Index Monthly Return Analyzer README
Overview
The Global Index Monthly Return Analyzer is a GUI-based application for analyzing monthly returns of global financial indices and ETFs. It provides interactive visualizations, statistical tests, and machine learning insights for historical and forecasted returns.
Requirements

Python: 3.8+
Dependencies:
yfinance
pandas
numpy==1.26.4
matplotlib
seaborn
tkinter
reportlab
openpyxl
scipy
statsmodels==0.14.1
pmdarima==2.0.4 (optional, for enhanced forecasting)
scikit-learn



Install dependencies:
pip install yfinance pandas numpy==1.26.4 matplotlib seaborn reportlab openpyxl scipy statsmodels==0.14.1 pmdarima==2.0.4 scikit-learn

How to Use the Application
1. Setup and Running

Save the application file as jse2.py.
Run the application:python jse2.py


The GUI window will open, titled "Global Index Monthly Return Analyzer (v2.8)".

2. Main Interface

Configuration Panel:
Index/ETF: Select a predefined index (e.g., 🇿🇦 JSE All Share (^J203.JO)) from the dropdown or enable "Custom Ticker" to enter a custom ticker (e.g., SPY).
Compare With: Select an index for comparison and click "➕ Add Comparison" to include it in analysis.
Date Range: Choose a predefined range (e.g., Last 5 Years) or "Custom" to specify a start year and end date (YYYY-MM-DD).
Dark Mode: Toggle for a darker theme.


Action Buttons:
🔍 Analyze Data: Start the analysis for the selected ticker and date range.
💾 Export to Excel: Save results to an Excel file (available after analysis).
📊 Toggle Benchmark: Show/hide the overall average return line on charts.
📄 Export PDF: Generate a PDF report of the analysis.
🤖 ML Analysis: Run machine learning analysis with forecasting.
🧮 Significance Test: Perform statistical tests on monthly returns.


Tabs:
Average Returns: Bar chart of mean or median monthly returns.
Year-Month Heatmap: Heatmap of returns by year and month.
Risk vs Return: Scatter plot of risk (std dev) vs. return, colored by positive return rate.
Summary Statistics: Text summary of key metrics.
Comparison: Line chart comparing monthly returns across tickers.
Statistical Tests: Results of t-tests and descriptive statistics.
ML Analysis: Machine learning results and visualizations.



3. Using the Application

Select Ticker and Date Range:
Choose an index (e.g., ^J203.JO) or enter a custom ticker.
Set the date range (e.g., 1990 to 2025-08-20 or select "All Data").


Analyze Data:
Click "🔍 Analyze Data" to fetch and process data.
The status bar at the bottom shows progress (e.g., "Downloading data...").
Once complete, tabs populate with visualizations and summaries.


Explore Tabs:
Average Returns: Toggle between Mean and Median using the dropdown. Hover over bars for details.
Year-Month Heatmap: View returns by year and month, with color intensity indicating performance.
Risk vs Return: Hover over points to see risk, return, and positive rate for each month.
Summary Statistics: Read key metrics like overall average return and best/worst months.
Comparison: Add tickers (e.g., ^GSPC) to compare monthly returns.


Run Advanced Analysis:
Statistical Tests: Click "🧮 Significance Test" to see t-test results and monthly statistics in the Statistical Tests tab.
ML Analysis: Click "🤖 ML Analysis" to run machine learning and view results in the ML Analysis tab, including a forecast for the upcoming month (e.g., September 2025).


Export Results:
Click "💾 Export to Excel" to save data to an Excel file with sheets for raw data, monthly summaries, and year-month returns.
Click "📄 Export PDF" to generate a professional PDF report.


Toggle Benchmark:
Click "📊 Toggle Benchmark" to show/hide the average return line on charts.



4. Machine Learning Algorithms Used

Principal Component Analysis (PCA): Dimensionality reduction.
Gaussian Mixture Model (GMM): Clustering months by return patterns.
Isolation Forest: Anomaly detection.
ARIMA (Auto-Regressive Integrated Moving Average): Time-series forecasting (with pmdarima for auto-selection or fixed-order fallback).

5. Troubleshooting

No Data Returned: Ensure the ticker is valid and the internet connection is active.
pmdarima Errors:
If you see ValueError: numpy.dtype size changed, reinstall dependencies:pip uninstall numpy pmdarima statsmodels -y
pip install numpy==1.26.4 pmdarima==2.0.4 statsmodels==0.14.1


If pmdarima fails, the app uses a fallback ARIMA model, noted in the ML Analysis output.


Custom Ticker: Use standard Yahoo Finance ticker formats (e.g., ^GSPC, SPY).

6. Notes

The app caches data daily to reduce API calls. Cache files are stored in a versioned directory (e.g., cache_v2_8).
The ML Analysis tab forecasts the upcoming month (e.g., September 2025) based on the current date (August 20, 2025).
Hover over charts for interactive tooltips with detailed data.

Screenshot

# Screenshot of the Global Index Monthly Return Analyzer GUI
# Shows the main interface with configuration panel, action buttons, and tabs.
# Configuration panel includes ticker selection, date range, and comparison options.
# Tabs display bar charts, heatmaps, scatter plots, summaries, comparisons, statistical tests, and ML analysis.
# Status bar at the bottom shows analysis progress.

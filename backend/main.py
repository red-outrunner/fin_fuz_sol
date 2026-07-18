from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
import io
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
from datetime import datetime
import logging
import os
from analysis import download_data, process_data, calculate_summary_stats, run_ml_analysis, run_anova_test, clean_data, calculate_dca, run_monte_carlo, get_company_profile, get_key_stats, get_news, get_calendar, get_article_content, search_tickers, get_dividend_history, get_financials, fetch_multiple_tickers, calculate_financial_freedom, get_jse_peers, get_dividend_yield
from reports import PDFReportGenerator

from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

# Structured logging. Level is configurable via LOG_LEVEL (default INFO). force=True
# so this format wins over any basicConfig already applied by an imported module.
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    force=True,
)
logger = logging.getLogger(__name__)

# Optional error tracking. Activates ONLY when SENTRY_DSN is set AND sentry-sdk is
# installed; otherwise it is a silent no-op, so local/dev runs need nothing extra.
SENTRY_DSN = os.getenv("SENTRY_DSN")
if SENTRY_DSN:
    try:
        import sentry_sdk
        sentry_sdk.init(
            dsn=SENTRY_DSN,
            environment=os.getenv("ENVIRONMENT", "development"),
            traces_sample_rate=float(os.getenv("SENTRY_TRACES_SAMPLE_RATE", "0.0")),
            send_default_pii=False,
        )
        logger.info("Sentry error tracking enabled.")
    except ImportError:
        logger.warning("SENTRY_DSN is set but sentry-sdk is not installed; error tracking disabled.")
    except Exception as e:
        logger.warning(f"Sentry initialization failed: {e}")

app = FastAPI(title="Global Index Analyzer API", version="3.0.0")

# Rate limiting (IP-based) to protect the upstream yfinance / DuckDuckGo scrapers
# from getting throttled/banned and to stop the heavy ML/Monte-Carlo endpoints from
# being a free DoS vector. Limits are configurable via env:
#   RATE_LIMIT          - default per-IP limit applied to all routes (e.g. "60/minute")
#   RATE_LIMIT_BURST    - secondary longer-window cap (e.g. "1000/hour")
# NOTE: storage is in-memory (per-process). For multi-worker production set
# RATE_LIMIT_STORAGE_URI to a shared backend such as redis://...
_default_limit = os.getenv("RATE_LIMIT", "60/minute")
_burst_limit = os.getenv("RATE_LIMIT_BURST", "1000/hour")
limiter = Limiter(
    key_func=get_remote_address,
    default_limits=[_default_limit, _burst_limit],
    storage_uri=os.getenv("RATE_LIMIT_STORAGE_URI", "memory://"),
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)

# CORS Setup
# This API is fully open: no auth, no cookies, no credentials of any kind. So we
# default to allowing every origin ("*"), which means the deployed frontend works
# out of the box regardless of its URL. To lock it down to specific origins later,
# set ALLOWED_ORIGINS (comma-separated), e.g.:
#   ALLOWED_ORIGINS="https://your-app.netlify.app,https://www.yourdomain.com"
# IMPORTANT: allow_credentials MUST stay False while origins can be "*" — browsers
# reject "*" together with credentials (that mismatch is what made preflight
# OPTIONS return 400). We send no cookies, so credentials=False is correct.
ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.getenv("ALLOWED_ORIGINS", "*").split(",")
    if origin.strip()
]
logger.info(f"CORS allowed origins: {ALLOWED_ORIGINS}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic Models
class AnalysisRequest(BaseModel):
    ticker: str
    start_year: int
    end_date: str
    inflation_rate: Optional[float] = 0.0
class ComparisonRequest(BaseModel):
    tickers: List[str]
    start_year: int
    end_date: str

class SearchRequest(BaseModel):
    query: str

class FreedomRequest(BaseModel):
    ticker: str
    monthly_income_goal: float


@app.get("/")
def read_root():
    return {"message": "Global Index Analyzer API is running"}

@app.get("/health")
def health_check():
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}

@app.post("/api/search")
def search_handler(request: SearchRequest):
    logger.info(f"Searching for: {request.query}")
    results = search_tickers(request.query)
    return clean_data(results)

@app.post("/api/analyze")
def analyze_ticker(request: AnalysisRequest):
    logger.info(f"Analyzing {request.ticker} from {request.start_year} to {request.end_date}")
    
    start_date = f"{request.start_year}-01-01"
    data = download_data(request.ticker, start_date, request.end_date)
    
    if data is None:
        raise HTTPException(status_code=404, detail="Data not found or download failed")
    
    processed = process_data(data)
    if processed is None:
        raise HTTPException(status_code=500, detail="Error processing data")
    
    stats = calculate_summary_stats(processed['monthly_ret'], processed['pivot'], ticker=request.ticker, inflation_rate=request.inflation_rate)
    
    # Prepare pivot data for JSON (reset index to make year a column)
    # Use the winsorized pivot for UI heatmaps/bar charts
    pivot_reset = processed['winsorized_pivot'].reset_index()
    pivot_data = pivot_reset.to_dict(orient='records')
    
    response_data = {
        "ticker": request.ticker,
        "stats": stats,
        "pivot_data": pivot_data,
        "monthly_returns": processed['winsorized_ret'].to_dict(), # Date -> Return (winsorized for charts)
        "moving_averages": {
            "ma_12": processed['ma_12'].where(pd.notnull(processed['ma_12']), None).to_dict(),
            "ma_60": processed['ma_60'].where(pd.notnull(processed['ma_60']), None).to_dict(),
            "prices": processed['prices'].to_dict()
        }
    }
    
    return clean_data(response_data)

@app.post("/api/ml")
def ml_analysis(request: AnalysisRequest):
    logger.info(f"Running ML Analysis for {request.ticker}")
    start_date = f"{request.start_year}-01-01"
    data = download_data(request.ticker, start_date, request.end_date)
    
    if data is None:
        raise HTTPException(status_code=404, detail="Data not found")
        
    processed = process_data(data)
    if processed is None:
        raise HTTPException(status_code=500, detail="Error processing data")
        
    ml_results = run_ml_analysis(processed['monthly_ret'])
    
    if ml_results is None:
        raise HTTPException(status_code=400, detail="Not enough data for ML analysis (need > 2 years)")
        
    return clean_data(ml_results)

@app.post("/api/projection")
def get_wealth_projection(request: AnalysisRequest):
    logger.info(f"Running Wealth Projection for {request.ticker}")
    start_date = f"{request.start_year}-01-01"
    data = download_data(request.ticker, start_date, request.end_date)
    
    if data is None:
        raise HTTPException(status_code=404, detail="Data not found")
        
    processed = process_data(data)
    if processed is None:
        raise HTTPException(status_code=500, detail="Error processing data")
        
    # Reinvest dividends into the projected total return.
    dividend_yield = get_dividend_yield(request.ticker)
    projection = run_monte_carlo(processed['monthly_ret'], dividend_yield=dividend_yield)

    if projection is None:
        raise HTTPException(status_code=400, detail="Not enough data for projection")
        
    return clean_data(projection)

@app.post("/api/stats")
def statistical_tests(request: AnalysisRequest):
    logger.info(f"Running Statistical Tests for {request.ticker}")
    start_date = f"{request.start_year}-01-01"
    data = download_data(request.ticker, start_date, request.end_date)
    
    if data is None:
        raise HTTPException(status_code=404, detail="Data not found")
        
    processed = process_data(data)
    if processed is None:
        raise HTTPException(status_code=500, detail="Error processing data")
        
    anova_results = run_anova_test(processed['pivot'])
    return clean_data(anova_results)

@app.post("/api/compare")
def compare_tickers(request: ComparisonRequest):
    logger.info(f"Comparing tickers: {request.tickers}")
    start_date = f"{request.start_year}-01-01"
    
    results = {}
    
    # NEW: Parallel Fetch
    fetched_data = fetch_multiple_tickers(request.tickers, start_date, request.end_date)
    
    for ticker, data in fetched_data.items():
        if data is not None:
            processed = process_data(data)
            if processed:
                # Calculate average monthly return for each month (1-12)
                month_avg = processed['pivot'].mean().to_dict()
                results[ticker] = month_avg
    
    if not results:
        raise HTTPException(status_code=404, detail="No data found for any requested tickers")
        
    return clean_data(results)

@app.post("/api/correlation")
def get_correlation(request: ComparisonRequest):
    logger.info(f"Calculating Correlation for: {request.tickers}")
    start_date = f"{request.start_year}-01-01"
    
    # We need a common index to calculate correlation correctly
    
    all_series = {}
    
    # NEW: Parallel Fetch
    fetched_data = fetch_multiple_tickers(request.tickers, start_date, request.end_date)
    
    for ticker, data in fetched_data.items():
        if data is not None:
            processed = process_data(data)
            if processed:
                all_series[ticker] = processed['monthly_ret']
    
    if len(all_series) < 2:
        raise HTTPException(status_code=400, detail="Need at least 2 valid tickers for correlation")
        
    # Combine into DataFrame
    df_corr = pd.DataFrame(all_series)
    
    # Calculate Correlation Matrix
    corr_matrix = df_corr.corr()
    
    # Format for Frontend (heatmap friendy)
    # List of { x: ticker1, y: ticker2, value: 0.9 }
    
    matrix_data = []
    for x in corr_matrix.columns:
        for y in corr_matrix.index:
            matrix_data.append({
                "x": x,
                "y": y,
                "value": corr_matrix.loc[y, x]
            })
            
    return clean_data({
        "matrix": matrix_data,
        "tickers": list(corr_matrix.columns)
    })

@app.post("/api/freedom")
def freedom_calculator(request: FreedomRequest):
    logger.info(f"Calculating Financial Freedom for {request.ticker} with goal {request.monthly_income_goal}")
    
    # We need recent data to get price and dividends
    # Fetch last 2 years to be safe for dividend calc
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_year = int(end_date.split("-")[0]) - 2
    start_date = f"{start_year}-01-01"
    
    data = download_data(request.ticker, start_date, end_date)

    if data is None:
        raise HTTPException(status_code=404, detail="Data not found")

    result = calculate_financial_freedom(data, request.monthly_income_goal, ticker=request.ticker)

    if result is None:
        raise HTTPException(status_code=500, detail="Error calculating freedom metrics")

    return clean_data(result)

@app.post("/api/peers")
def get_peers(request: AnalysisRequest):
    logger.info(f"Fetching peers for {request.ticker}")
    peers = get_jse_peers(request.ticker)
    return {"peers": peers}

@app.post("/api/export/excel")
def export_excel(request: AnalysisRequest):
    logger.info(f"Exporting Excel for {request.ticker}")
    start_date = f"{request.start_year}-01-01"
    data = download_data(request.ticker, start_date, request.end_date)
    
    if data is None:
        raise HTTPException(status_code=404, detail="Data not found")
        
    processed = process_data(data)
    if processed is None:
        raise HTTPException(status_code=500, detail="Error processing data")
        
    # Create Excel file in memory
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        processed['df'].to_excel(writer, sheet_name='Monthly Data')
        processed['pivot'].to_excel(writer, sheet_name='Yearly Pivot')
        
        # Add summary stats if desired
        stats = calculate_summary_stats(processed['monthly_ret'], processed['pivot'], ticker=request.ticker, inflation_rate=request.inflation_rate)
        if stats:
            # Flatten stats for dataframe
            flat_stats = {k: v for k, v in stats.items() if isinstance(v, (int, float, str))}
            pd.DataFrame([flat_stats]).to_excel(writer, sheet_name='Summary Stats', index=False)
            
    output.seek(0)
    
    headers = {
        'Content-Disposition': f'attachment; filename="{request.ticker}_analysis.xlsx"'
    }
    return StreamingResponse(output, headers=headers, media_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

@app.post("/api/export/csv")
def export_csv(request: AnalysisRequest):
    logger.info(f"Exporting CSV for {request.ticker}")
    start_date = f"{request.start_year}-01-01"
    data = download_data(request.ticker, start_date, request.end_date)
    
    if data is None:
        raise HTTPException(status_code=404, detail="Data not found")
        
    processed = process_data(data)
    if processed is None:
        raise HTTPException(status_code=500, detail="Error processing data")
        
    output = io.StringIO()
    processed['df'].to_csv(output)
    
    # Convert string to bytes for StreamingResponse
    mem = io.BytesIO()
    mem.write(output.getvalue().encode())
    mem.seek(0)
    
    headers = {
        'Content-Disposition': f'attachment; filename="{request.ticker}_data.csv"'
    }
    return StreamingResponse(mem, headers=headers, media_type='text/csv')

@app.post("/api/export/pdf")
def export_pdf(request: AnalysisRequest):
    logger.info(f"Exporting PDF for {request.ticker}")
    start_date = f"{request.start_year}-01-01"
    data = download_data(request.ticker, start_date, request.end_date)
    
    if data is None:
        raise HTTPException(status_code=404, detail="Data not found")
        
    processed = process_data(data)
    if processed is None:
        raise HTTPException(status_code=500, detail="Error processing data")
        
    stats = calculate_summary_stats(processed['monthly_ret'], processed['pivot'], ticker=request.ticker, inflation_rate=request.inflation_rate)
    
    # --- Generate Additional Data for Report ---
    
    # 1. Peers Analysis
    peers = get_jse_peers(request.ticker)
    peer_data_map = {}
    if peers:
        # Limit to top 5 peers for report clarity
        top_peers = peers[:5]
        fetched_peers = fetch_multiple_tickers(top_peers, start_date, request.end_date)
        for p_ticker, p_data in fetched_peers.items():
            if p_data is not None:
                p_processed = process_data(p_data)
                if p_processed:
                    # Store mean monthly returns for chart
                    peer_data_map[p_ticker] = p_processed['pivot'].mean().to_list()

    # 2. Monte Carlo (dividends reinvested into total return)
    monte_carlo = run_monte_carlo(processed['monthly_ret'], dividend_yield=get_dividend_yield(request.ticker))
    
    # 3. DCA Analysis (Default: R1000/month)
    dca_results = calculate_dca(processed['monthly_ret'], 1000.0)
    
    # --- Build PDF ---
    buffer = io.BytesIO()
    report = PDFReportGenerator(buffer, request.ticker, request.start_year, request.end_date)
    
    report.add_title_page()
    report.add_executive_summary(stats)
    report.add_wealth_chart(processed)
    report.add_drawdown_chart(processed)
    report.add_monthly_table(processed)
    
    if peer_data_map:
        report.add_peer_battle(peer_data_map)
        
    if monte_carlo:
        report.add_monte_carlo(monte_carlo)
        
    if dca_results:
        report.add_dca_analysis(dca_results)
        
    report.build_pdf()
    
    buffer.seek(0)
    
    headers = {
        'Content-Disposition': f'attachment; filename="{request.ticker}_Ubomvu_Report.pdf"'
    }
    return StreamingResponse(buffer, headers=headers, media_type='application/pdf')



@app.post("/api/export/ml")
def export_ml(request: AnalysisRequest):
    logger.info(f"Exporting ML Data for {request.ticker}")
    start_date = f"{request.start_year}-01-01"
    data = download_data(request.ticker, start_date, request.end_date)
    
    if data is None:
        raise HTTPException(status_code=404, detail="Data not found")
        
    processed = process_data(data)
    if processed is None:
        raise HTTPException(status_code=500, detail="Error processing data")
        
    ml_results = run_ml_analysis(processed['monthly_ret'])
    
    if ml_results is None:
        raise HTTPException(status_code=400, detail="Not enough data for ML analysis (need > 2 years)")
    
    # Construct CSV Data
    # Align dates with results. 
    # run_ml_analysis transforms data into windows of 12. 
    # If len(data) = N, results have len = N - 12.
    # The result corresponds to the *end* of the window? 
    # Yes, typically we predict/classify the state at time T based on T-12 to T.
    
    dates = processed['monthly_ret'].index[12:]
    
    # Ensure lengths match
    if len(dates) != len(ml_results['clusters']):
         # Fallback if alignment is tricky, though it should match by logic in run_ml_analysis
         logger.warning(f"Date length {len(dates)} != Result length {len(ml_results['clusters'])}")
         # Slice dates to match results from the end
         dates = dates[-len(ml_results['clusters']):]
    
    export_data = []
    for i, date in enumerate(dates):
        export_data.append({
            "Date": str(date).split(" ")[0],
            "PCA_1": ml_results['pca_components'][i][0],
            "PCA_2": ml_results['pca_components'][i][1],
            "Cluster": ml_results['clusters'][i],
            "Anomaly": "Yes" if ml_results['anomalies'][i] == -1 else "No"
        })
        
    df_export = pd.DataFrame(export_data)
    
    output = io.StringIO()
    df_export.to_csv(output, index=False)
    
    mem = io.BytesIO()
    mem.write(output.getvalue().encode())
    mem.seek(0)
    
    headers = {
        'Content-Disposition': f'attachment; filename="{request.ticker}_ml_analysis.csv"'
    }
    return StreamingResponse(mem, headers=headers, media_type='text/csv')


@app.post("/api/profile")
def company_profile(request: AnalysisRequest):
    logger.info(f"Fetching Profile for {request.ticker}")
    profile = get_company_profile(request.ticker)
    if profile is None:
         # Return empty structure instead of error for UI smoothness
         return {"biggest_shareholder": None, "sentiment": None}
    return clean_data(profile)

@app.post("/api/fundamentals")
def get_fundamentals(request: AnalysisRequest):
    logger.info(f"Fetching Fundamentals for {request.ticker}")
    stats = get_key_stats(request.ticker)
    if stats is None:
        raise HTTPException(status_code=404, detail="Fundamentals not found")
    return stats

@app.post("/api/news")
def get_company_news(request: AnalysisRequest):
    logger.info(f"Fetching News for {request.ticker}")
    news = get_news(request.ticker)
    return clean_data(news)

@app.post("/api/calendar")
def get_company_calendar(request: AnalysisRequest):
    logger.info(f"Fetching Calendar for {request.ticker}")
    calendar = get_calendar(request.ticker)
    return clean_data(calendar)

@app.post("/api/dividends")
def get_dividends(request: AnalysisRequest):
    logger.info(f"Fetching Dividends for {request.ticker}")
    dividends = get_dividend_history(request.ticker, request.start_year)
    if dividends is None:
        return {"history": [], "annual": [], "current_yield": 0, "payout_ratio": 0}
    return clean_data(dividends)

@app.post("/api/valuation")
def get_valuation_data(request: AnalysisRequest):
    logger.info(f"Fetching Financials for Valuation: {request.ticker}")
    financials = get_financials(request.ticker)
    if financials is None:
        raise HTTPException(status_code=404, detail="Financial data for valuation not found")
    return clean_data(financials)

class ArticleRequest(BaseModel):
    url: str

@app.post("/api/news/read")
def read_news_article(request: ArticleRequest):
    logger.info(f"Reading article: {request.url}")
    content = get_article_content(request.url)
    return clean_data(content)

class DcaRequest(BaseModel):
    ticker: str
    start_year: int
    end_date: str
    monthly_contribution: float

@app.post("/api/dca")
def run_dca_simulation(request: DcaRequest):
    logger.info(f"Running DCA Simulation for {request.ticker}")
    start_date = f"{request.start_year}-01-01"
    data = download_data(request.ticker, start_date, request.end_date)
    
    if data is None:
        raise HTTPException(status_code=404, detail="Data not found")
        
    processed = process_data(data)
    if processed is None:
         raise HTTPException(status_code=500, detail="Error processing data")
         
    from analysis import calculate_dca # Import here or at top
    dca_results = calculate_dca(processed['monthly_ret'], request.monthly_contribution)
    
    return clean_data(dca_results)


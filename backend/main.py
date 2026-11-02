from fastapi import FastAPI, HTTPException, BackgroundTasks, Request, WebSocket, WebSocketDisconnect, Depends
from fastapi.responses import Response, StreamingResponse, JSONResponse
from fastapi.exceptions import RequestValidationError
from fastapi.encoders import jsonable_encoder
from starlette.concurrency import run_in_threadpool
import io
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, StringConstraints, field_validator
from typing import Annotated, List, Optional
import pandas as pd
from datetime import datetime
import logging
import os
import asyncio
import json
from analysis import download_data, process_data, calculate_summary_stats, run_ml_analysis, run_anova_test, clean_data, calculate_dca, run_monte_carlo, get_company_profile, get_key_stats, get_news, get_calendar, get_article_content, search_tickers, get_dividend_history, get_financials, fetch_multiple_tickers, calculate_financial_freedom, get_jse_peers
import yfinance as yf
from reports import PDFReportGenerator
from screener import screen_stocks, get_sector_performance, get_stock_ideas, get_ticker_details, get_jse_universe, get_stock_of_the_day
from sector_manager import sync_sector_data
from fundamentals import get_financial_statements, get_ratio_trends, get_analyst_estimates, get_segment_data, get_fair_value_comparison
from technical import build_technical_snapshot, run_backtest
from realtime import get_live_quotes, get_live_quote, evaluate_alerts, quote_provider
from quote_stream import QuoteConnectionLimiter, SharedQuoteCache
from currency_utils import normalize_price, display_currency, normalize_history_value, BRAND_ATTRIBUTION
from fx_rates import fx_rate_to_zar as _fx_rate_to_zar, positive_number as _positive_number
from database import get_db, engine, Base
import models
from portfolio_access import require_portfolio_key
from admin_access import require_sector_admin
from alert_access import require_alert_key
from alert_rules import AlertRule, AlertCreate
from sqlalchemy.orm import Session

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

# Each worker owns a bounded cache and connection budget. Size deployment-wide
# budgets by worker count; HTTP and WebSocket clients share these cached quotes.
quote_cache = SharedQuoteCache(
    ttl_seconds=int(os.getenv("QUOTE_CACHE_TTL_SECONDS", "5")),
    max_entries=int(os.getenv("QUOTE_CACHE_MAX_ENTRIES", "512")),
)
quote_connections = QuoteConnectionLimiter(
    max_connections=int(os.getenv("WS_QUOTES_MAX_CONNECTIONS", "100")),
    max_per_ip=int(os.getenv("WS_QUOTES_MAX_PER_IP", "3")),
    starts_per_minute=int(os.getenv("WS_QUOTES_STARTS_PER_MINUTE", "10")),
    global_starts_per_minute=int(os.getenv("WS_QUOTES_GLOBAL_STARTS_PER_MINUTE", "120")),
)

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


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError):
    # Nonfinite JSON numbers must produce a validation response, not a second
    # exception while serializing the rejected input in FastAPI's error details.
    return JSONResponse(status_code=422, content={
        "detail": clean_data(jsonable_encoder(exc.errors())),
    })

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


def rate_limit_exceeded_handler(request: Request, exc: Exception) -> Response:
    """Adapt SlowAPI's narrower exception type to Starlette's handler contract."""
    if not isinstance(exc, RateLimitExceeded):
        raise exc
    return _rate_limit_exceeded_handler(request, exc)


app.add_exception_handler(RateLimitExceeded, rate_limit_exceeded_handler)
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


@app.on_event("startup")
def startup_event():
    logger.info("Initializing JSE sector & ticker metadata...")
    try:
        res = sync_sector_data(auto_search_excel=False)
        logger.info(f"Sector metadata initialized: {res}")
    except Exception as e:
        logger.error(f"Failed to initialize sector metadata: {e}")


@app.get("/")
def read_root():
    return {"message": "Global Index Analyzer API is running"}

@app.get("/health")
def health_check():
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}

@app.post("/api/admin/sync-sectors", dependencies=[Depends(require_sector_admin)])
def admin_sync_sectors(request: Request):
    """Reload sector JSON; spreadsheet imports are local CLI operations only."""
    if request.query_params:
        raise HTTPException(status_code=400, detail="This endpoint accepts no query parameters")
    try:
        return sync_sector_data(auto_search_excel=False)
    except Exception:
        logger.exception("Error reloading sector metadata")
        raise HTTPException(status_code=500, detail="Failed to reload sector metadata")

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
        
    # Adjusted historical returns already include dividends; do not add yield again.
    projection = run_monte_carlo(processed['monthly_ret'])

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
    meta = get_jse_peers(request.ticker, return_meta=True)
    return meta

def _build_export_frames(request: AnalysisRequest):
    """Shared data prep for Excel / Sheets exports."""
    start_date = f"{request.start_year}-01-01"
    data = download_data(request.ticker, start_date, request.end_date)
    if data is None:
        raise HTTPException(status_code=404, detail="Data not found")
    processed = process_data(data)
    if processed is None:
        raise HTTPException(status_code=500, detail="Error processing data")
    stats = calculate_summary_stats(
        processed['monthly_ret'], processed['pivot'],
        ticker=request.ticker, inflation_rate=request.inflation_rate
    )
    flat_stats = None
    if isinstance(stats, dict) and stats:
        flat_stats = {k: v for k, v in stats.items() if isinstance(v, (int, float, str))}
    return processed, flat_stats


def _make_excel_safe(df):
    """Strip timezone info from index/columns so openpyxl can write them."""
    df = df.copy()
    if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    for col in df.columns:
        if isinstance(df[col].dtype, pd.DatetimeTZDtype):
            df[col] = df[col].dt.tz_localize(None)
    return df


@app.post("/api/export/excel")
def export_excel(request: AnalysisRequest):
    logger.info(f"Exporting Excel for {request.ticker}")
    processed, flat_stats = _build_export_frames(request)

    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        _make_excel_safe(processed['df']).to_excel(writer, sheet_name='Monthly Data')
        _make_excel_safe(processed['pivot']).to_excel(writer, sheet_name='Yearly Pivot')
        if flat_stats:
            pd.DataFrame([flat_stats]).to_excel(writer, sheet_name='Summary Stats', index=False)
        pd.DataFrame([
            [BRAND_ATTRIBUTION],
            [f"Ticker: {request.ticker}"],
            [f"Period: {request.start_year} → {request.end_date}"],
        ]).to_excel(writer, sheet_name='About', index=False, header=False)

    output.seek(0)
    headers = {
        'Content-Disposition': f'attachment; filename="{request.ticker}_analysis.xlsx"'
    }
    return StreamingResponse(
        output,
        headers=headers,
        media_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
    )


@app.post("/api/export/sheets")
def export_sheets(request: AnalysisRequest):
    """
    Returns tab-separated analysis data for one-click Google Sheets sync.
    Frontend copies the TSV to the clipboard and opens sheets.new.
    """
    logger.info(f"Exporting Sheets TSV for {request.ticker}")
    processed, flat_stats = _build_export_frames(request)

    monthly = processed['df'].reset_index()
    monthly_tsv = monthly.to_csv(sep='\t', index=False)
    pivot_tsv = processed['pivot'].reset_index().to_csv(sep='\t', index=False)
    stats_tsv = ""
    if flat_stats:
        stats_tsv = pd.DataFrame([flat_stats]).to_csv(sep='\t', index=False)

    payload = (
        f"{BRAND_ATTRIBUTION}\n"
        f"UBOMVU ANALYSIS\t{request.ticker}\n"
        f"Period\t{request.start_year} → {request.end_date}\n\n"
        f"=== SUMMARY ===\n{stats_tsv}\n"
        f"=== MONTHLY DATA ===\n{monthly_tsv}\n"
        f"=== YEARLY PIVOT ===\n{pivot_tsv}"
    )
    return {
        "ticker": request.ticker,
        "tsv": payload,
        "sheets_url": "https://sheets.new",
        "hint": "Data copied — paste into the new Google Sheet with Ctrl+V / Cmd+V",
    }

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
    output.write(f"# {BRAND_ATTRIBUTION}\n")
    output.write(f"# Ticker: {request.ticker} | Period: {request.start_year} → {request.end_date}\n")
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

    # 2. Monte Carlo (adjusted historical returns already include dividends)
    monte_carlo = run_monte_carlo(processed['monthly_ret'])
    
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
    output.write(f"# {BRAND_ATTRIBUTION}\n")
    output.write(f"# Ticker: {request.ticker} | ML cluster export\n")
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


# === Stock Screener & Discovery Endpoints ===

class ScreenerRequest(BaseModel):
    min_market_cap: Optional[float] = None
    max_market_cap: Optional[float] = None
    min_pe: Optional[float] = None
    max_pe: Optional[float] = None
    min_dividend_yield: Optional[float] = None
    min_roe: Optional[float] = None
    max_debt_equity: Optional[float] = None
    min_beta: Optional[float] = None
    max_beta: Optional[float] = None
    sectors: Optional[List[str]] = None
    min_revenue_growth: Optional[float] = None
    min_profit_margin: Optional[float] = None
    undervalued_only: bool = False
    dividend_growers_only: bool = False

@app.post("/api/screener")
@limiter.limit("30/minute")
def stock_screener(request: Request, screener_request: ScreenerRequest):
    """Screen JSE stocks based on fundamental criteria."""
    logger.info(f"Running stock screener with filters: {screener_request}")
    
    results = screen_stocks(
        min_market_cap=screener_request.min_market_cap,
        max_market_cap=screener_request.max_market_cap,
        min_pe=screener_request.min_pe,
        max_pe=screener_request.max_pe,
        min_dividend_yield=screener_request.min_dividend_yield,
        min_roe=screener_request.min_roe,
        max_debt_equity=screener_request.max_debt_equity,
        min_beta=screener_request.min_beta,
        max_beta=screener_request.max_beta,
        sectors=screener_request.sectors,
        min_revenue_growth=screener_request.min_revenue_growth,
        min_profit_margin=screener_request.min_profit_margin,
        undervalued_only=screener_request.undervalued_only,
        dividend_growers_only=screener_request.dividend_growers_only,
    )
    
    return {"results": results, "count": len(results)}

@app.get("/api/screener/heatmap")
@limiter.limit("30/minute")
def sector_heatmap(request: Request, period: str = "1d"):
    """Get sector performance data for heatmap visualization."""
    logger.info(f"Fetching sector heatmap data for period: {period}")
    sectors = get_sector_performance(period=period)
    return {"sectors": sectors}

@app.get("/api/screener/ideas")
@limiter.limit("30/minute")
def stock_ideas(request: Request):
    """Get curated stock ideas: undervalued, 52-week lows, dividend stars, etc."""
    logger.info("Fetching stock ideas")
    ideas = get_stock_ideas()
    return ideas

@app.get("/api/screener/universe")
@limiter.limit("60/minute")
def get_universe(request: Request):
    """Get list of all available JSE Top 40 tickers."""
    logger.info("Fetching JSE universe")
    tickers = get_jse_universe()
    return {"tickers": tickers}

@app.get("/api/screener/ticker/{ticker}")
@limiter.limit("60/minute")
def ticker_detail(request: Request, ticker: str):
    """Get detailed info for a single ticker."""
    logger.info(f"Fetching details for {ticker}")
    # Normalize ticker to uppercase
    ticker = ticker.upper()
    details = get_ticker_details(ticker)
    if details is None:
        raise HTTPException(status_code=404, detail=f"Ticker {ticker} not found")
    return details


class SparklineRequest(BaseModel):
    tickers: List[str]


@app.post("/api/watchlist/sparklines")
@limiter.limit("30/minute")
def watchlist_sparklines(request: Request, body: SparklineRequest):
    """Return compact 30-day close series for watchlist mini-charts."""
    tickers = [t.upper().strip() for t in body.tickers if t][:20]
    if not tickers:
        return {"series": {}}

    end = datetime.now().strftime("%Y-%m-%d")
    start = (datetime.now() - pd.Timedelta(days=45)).strftime("%Y-%m-%d")
    series = {}

    for ticker in tickers:
        try:
            data = download_data(ticker, start, end)
            if data is None or data.empty:
                series[ticker] = {"prices": [], "change_pct": None}
                continue
            closes = data["Adj Close"] if "Adj Close" in data.columns else data["Close"]
            closes = closes.dropna().tail(30)
            raw = [float(p) for p in closes.tolist()]
            prices = [round(v, 4) if v is not None else 0.0 for v in (normalize_history_value(p, ticker) for p in raw)]
            change_pct = None
            if len(prices) >= 2 and prices[0] != 0:
                change_pct = round(((prices[-1] - prices[0]) / prices[0]) * 100, 2)
            series[ticker] = {
                "prices": prices,
                "change_pct": change_pct,
                "last": prices[-1] if prices else None,
            }
        except Exception as e:
            logger.error(f"Sparkline failed for {ticker}: {e}")
            series[ticker] = {"prices": [], "change_pct": None}

    return {"series": series}


@app.get("/api/stock-of-the-day")
@limiter.limit("60/minute")
def stock_of_the_day(request: Request):
    """
    Featured JSE equity for the Africa/Johannesburg calendar day.
    Day-cached; includes thesis bullets, theme, countdown target, and yesterday's pick.
    """
    result = get_stock_of_the_day()
    if result is None:
        raise HTTPException(status_code=503, detail="No stocks are currently available for the daily feature")
    return result


# === Portfolio Tracking Endpoints ===

class HoldingRequest(BaseModel):
    ticker: Annotated[str, StringConstraints(
        strip_whitespace=True, min_length=1, max_length=64,
        pattern=r"^[A-Za-z0-9^][A-Za-z0-9.^=_-]*$",
    )]
    quantity: float = Field(gt=0, allow_inf_nan=False)
    # Zero cost is valid for gifted shares; costs are in native major units.
    avg_cost: float = Field(ge=0, allow_inf_nan=False)

    @field_validator("ticker")
    @classmethod
    def normalize_ticker(cls, value: str) -> str:
        return value.upper()


_SUFFIX_CURRENCY = {
    "JO": "ZAR", "L": "GBP", "IL": "GBP", "DE": "EUR", "F": "EUR",
    "PA": "EUR", "AS": "EUR", "BR": "EUR", "LS": "EUR", "MI": "EUR",
    "MC": "EUR", "SW": "CHF", "T": "JPY", "HK": "HKD", "SS": "CNY",
    "SZ": "CNY", "AX": "AUD", "TO": "CAD", "V": "CAD", "SA": "BRL",
    "MX": "MXN", "SI": "SGD", "KS": "KRW", "NS": "INR", "BO": "INR",
    "NZ": "NZD",
}

def _infer_holding_currency(ticker: str, yahoo_currency: str = "") -> str:
    return display_currency(yahoo_currency, ticker)


def _holding_quote(ticker: str):
    """Missing/invalid provider quotes remain unavailable, never zero prices."""
    try:
        info = yf.Ticker(ticker).info or {}
    except Exception:
        info = {}
    yahoo_cur = info.get("currency") or ""
    raw_price = _positive_number(info.get("currentPrice"))
    if raw_price is None:
        raw_price = _positive_number(info.get("regularMarketPrice"))
    # Without quote-unit metadata a JSE quote could be cents or rands.
    price = normalize_price(raw_price, yahoo_cur) if yahoo_cur else None
    return {
        "current_price": _positive_number(price),
        "currency": _infer_holding_currency(ticker, yahoo_cur),
        "sector": info.get("sector") or "Other",
        "name": info.get("shortName") or info.get("longName") or ticker,
    }


def _holding_valuation(holding, fx_cache):
    quote = _holding_quote(holding.ticker)
    price = quote["current_price"]
    fx = _fx_rate_to_zar(quote["currency"], fx_cache)
    value = holding.quantity * price if price is not None else None
    cost = holding.quantity * holding.avg_cost
    pnl = value - cost if value is not None else None
    missing = []
    if price is None:
        missing.append("price")
    if fx is None:
        missing.append("fx")
    return {
        **quote,
        "id": holding.id,
        "ticker": holding.ticker,
        "quantity": holding.quantity,
        "avg_cost": holding.avg_cost,
        "current_value": value,
        "cost_basis": cost,
        "pnl": pnl,
        "pnl_pct": pnl / cost * 100 if pnl is not None and cost > 0 else None,
        "value_zar": value * fx if value is not None and fx is not None else None,
        "cost_zar": cost * fx if fx is not None else None,
        "pnl_zar": pnl * fx if pnl is not None and fx is not None else None,
        "valuation_status": "unavailable" if missing else "available",
        "missing_data": missing,
        "created_at": holding.created_at.isoformat() if holding.created_at else None,
    }


@app.get("/api/portfolio/holdings")
@limiter.limit("60/minute")
def get_holdings(request: Request, client_key: str = Depends(require_portfolio_key), db: Session = Depends(get_db)):
    """Get all portfolio holdings for a client (prices in each stock's native currency)."""
    logger.info("Fetching portfolio holdings")
    holdings = db.query(models.PortfolioHolding).filter(
        models.PortfolioHolding.client_key == client_key
    ).all()

    fx_cache = {}
    result = [_holding_valuation(h, fx_cache) for h in holdings]
    return clean_data({"holdings": result, "count": len(result)})


@app.post("/api/portfolio/holdings")
@limiter.limit("30/minute")
def add_holding(request: Request, holding_req: HoldingRequest, client_key: str = Depends(require_portfolio_key), db: Session = Depends(get_db)):
    """Add or update a portfolio holding. avg_cost is in the stock's native currency."""
    logger.info("Adding portfolio holding")
    ticker = holding_req.ticker.strip().upper()

    existing = db.query(models.PortfolioHolding).filter(
        models.PortfolioHolding.client_key == client_key,
        models.PortfolioHolding.ticker == ticker
    ).first()

    if existing:
        total_qty = existing.quantity + holding_req.quantity
        total_cost = (existing.quantity * existing.avg_cost) + (holding_req.quantity * holding_req.avg_cost)
        existing.avg_cost = total_cost / total_qty if total_qty > 0 else 0
        existing.quantity = total_qty
        db.commit()
        db.refresh(existing)
        return {
            "status": "updated",
            "holding_id": existing.id,
            "currency": _infer_holding_currency(ticker, ""),
        }
    else:
        new_holding = models.PortfolioHolding(
            client_key=client_key,
            ticker=ticker,
            quantity=holding_req.quantity,
            avg_cost=holding_req.avg_cost
        )
        db.add(new_holding)
        db.commit()
        db.refresh(new_holding)
        return {
            "status": "created",
            "holding_id": new_holding.id,
            "currency": _infer_holding_currency(ticker, ""),
        }


@app.delete("/api/portfolio/holdings/{holding_id}")
@limiter.limit("30/minute")
def delete_holding(request: Request, holding_id: int, client_key: str = Depends(require_portfolio_key), db: Session = Depends(get_db)):
    """Delete a portfolio holding."""
    logger.info(f"Deleting holding {holding_id}")
    holding = db.query(models.PortfolioHolding).filter(
        models.PortfolioHolding.id == holding_id,
        models.PortfolioHolding.client_key == client_key
    ).first()

    if not holding:
        raise HTTPException(status_code=404, detail="Holding not found")

    db.delete(holding)
    db.commit()
    return {"status": "deleted"}


@app.get("/api/portfolio/performance")
@limiter.limit("30/minute")
def portfolio_performance(request: Request, client_key: str = Depends(require_portfolio_key), db: Session = Depends(get_db)):
    """Portfolio totals in ZAR (FX-converted) plus sector allocation."""
    logger.info("Fetching portfolio performance")
    holdings = db.query(models.PortfolioHolding).filter(
        models.PortfolioHolding.client_key == client_key
    ).all()

    fx_cache = {}
    rows = [_holding_valuation(h, fx_cache) for h in holdings]
    missing = [{"ticker": row["ticker"], "missing_data": row["missing_data"]}
               for row in rows if row["missing_data"]]
    complete = not missing
    # Suppress complete-portfolio figures rather than presenting a partial sum.
    total_value = sum(row["value_zar"] for row in rows) if complete else None
    total_cost = (sum(row["cost_zar"] for row in rows)
                  if all(row["cost_zar"] is not None for row in rows) else None)
    total_pnl = total_value - total_cost if total_value is not None and total_cost is not None else None
    allocation = {}
    if total_value is not None and total_value > 0:
        for row in rows:
            sector = row["sector"]
            allocation[sector] = allocation.get(sector, 0) + row["value_zar"] / total_value * 100
    return clean_data({
        "total_value": total_value,
        "total_cost": total_cost,
        "total_pnl": total_pnl,
        "total_pnl_pct": total_pnl / total_cost * 100 if total_pnl is not None and total_cost is not None and total_cost > 0 else None,
        "holdings_count": len(rows),
        "sector_allocation": allocation,
        "display_currency": "ZAR",
        "currencies": sorted({row["currency"] for row in rows}),
        "valuation_status": "available" if complete else "incomplete",
        "unavailable_holdings": missing,
    })


# === Enhanced Fundamental Analysis Endpoints ===

class FairValueRequest(BaseModel):
    ticker: str
    dcf_value: float

@app.get("/api/fundamentals/{ticker}")
@limiter.limit("30/minute")
def fundamentals(request: Request, ticker: str):
    """Get 5-year financial statements for a ticker."""
    logger.info(f"Fetching financial statements for {ticker}")
    ticker = ticker.upper()
    statements = get_financial_statements(ticker)
    if statements is None:
        raise HTTPException(status_code=404, detail=f"Financial statements not found for {ticker}")
    return statements

@app.get("/api/fundamentals/{ticker}/ratios")
@limiter.limit("30/minute")
def ratio_trends(request: Request, ticker: str):
    """Get ratio trends for fundamental analysis."""
    logger.info(f"Fetching ratio trends for {ticker}")
    ticker = ticker.upper()
    ratios = get_ratio_trends(ticker)
    if ratios is None:
        raise HTTPException(status_code=404, detail=f"Ratio data not found for {ticker}")
    return ratios

@app.get("/api/fundamentals/{ticker}/analyst")
@limiter.limit("30/minute")
def analyst_data(request: Request, ticker: str):
    """Get analyst estimates and recommendations."""
    logger.info(f"Fetching analyst data for {ticker}")
    ticker = ticker.upper()
    estimates = get_analyst_estimates(ticker)
    if estimates is None:
        raise HTTPException(status_code=404, detail=f"Analyst data not found for {ticker}")
    return estimates

@app.get("/api/fundamentals/{ticker}/segments")
@limiter.limit("30/minute")
def segment_data(request: Request, ticker: str):
    """Get segment revenue breakdown."""
    logger.info(f"Fetching segment data for {ticker}")
    ticker = ticker.upper()
    segments = get_segment_data(ticker)
    return segments or {"available": False, "message": "Segment data not available"}

@app.post("/api/fundamentals/fair-value")
@limiter.limit("30/minute")
def fair_value(request: Request, req: FairValueRequest):
    """Compare DCF valuation to analyst consensus."""
    logger.info(f"Fetching fair value comparison for {req.ticker}")
    ticker = req.ticker.upper()
    comparison = get_fair_value_comparison(ticker, req.dcf_value)
    if comparison is None:
        raise HTTPException(status_code=404, detail=f"Fair value data not found for {ticker}")
    return comparison


# === Technical Analysis ===

class TechnicalRequest(BaseModel):
    ticker: str
    timeframe: Optional[str] = "daily"  # daily | weekly | monthly


class BacktestRequest(BaseModel):
    ticker: str
    timeframe: Optional[str] = "daily"
    fast: Optional[int] = 20
    slow: Optional[int] = 50


@app.post("/api/technical")
@limiter.limit("20/minute")
def technical_analysis(request: Request, body: TechnicalRequest):
    """OHLCV + RSI/MACD/Bollinger/Fibonacci + pattern detection + multi-TF."""
    tf = (body.timeframe or "daily").lower()
    if tf not in ("daily", "weekly", "monthly"):
        raise HTTPException(status_code=400, detail="timeframe must be daily, weekly, or monthly")
    snap = build_technical_snapshot(body.ticker, tf)
    if snap is None:
        raise HTTPException(status_code=404, detail=f"No technical data for {body.ticker}")
    return snap


@app.post("/api/technical/backtest")
@limiter.limit("15/minute")
def technical_backtest(request: Request, body: BacktestRequest):
    """SMA crossover strategy backtest."""
    fast = int(body.fast or 20)
    slow = int(body.slow or 50)
    if fast < 2 or slow < 3 or fast >= slow:
        raise HTTPException(status_code=400, detail="Require 2 ≤ fast < slow")
    return run_backtest(body.ticker, body.timeframe or "daily", fast=fast, slow=slow)


# === Live quotes & alerts ===

class QuotesRequest(BaseModel):
    tickers: List[str]


class AlertEvaluateRequest(BaseModel):
    rules: List[AlertRule] = Field(max_length=50)


@app.post("/api/quotes/live")
@limiter.limit("60/minute")
def live_quotes(request: Request, body: QuotesRequest):
    return quote_cache.get_quotes(body.tickers, get_live_quotes)


@app.get("/api/quotes/provider")
def quotes_provider_info():
    return {
        "provider": quote_provider(),
        "supports_streaming": True,
        "note": "WebSocket at /ws/quotes?tickers=NPN.JO,SBK.JO — polls provider every few seconds.",
    }


@app.post("/api/alerts/evaluate")
@limiter.limit("30/minute")
def alerts_evaluate(request: Request, body: AlertEvaluateRequest):
    """Evaluate client-supplied alert rules (also works without DB)."""
    return evaluate_alerts([rule.model_dump() for rule in body.rules])


@app.get("/api/alerts")
def list_alerts(client_key: str = Depends(require_alert_key), db: Session = Depends(get_db)):
    rows = (
        db.query(models.Alert)
        .filter(models.Alert.client_key == client_key, models.Alert.active == True)  # noqa: E712
        .order_by(models.Alert.id.desc())
        .all()
    )
    return {
        "alerts": [
            {
                "id": a.id,
                "ticker": a.ticker,
                "type": a.alert_type,
                "condition": a.condition,
                "threshold": a.threshold,
                "label": a.label,
                "triggered": a.triggered,
                "armed": not a.triggered if a.alert_type == "price" else None,
                "last_triggered_at": a.last_triggered_at.isoformat() if a.last_triggered_at else None,
                "created_at": a.created_at.isoformat() if a.created_at else None,
            }
            for a in rows
        ]
    }


@app.post("/api/alerts")
def create_alert(body: AlertCreate, client_key: str = Depends(require_alert_key), db: Session = Depends(get_db)):
    alert = models.Alert(
        client_key=client_key,
        ticker=body.ticker.upper().strip(),
        alert_type=body.type,
        condition=(body.condition or "above").lower(),
        threshold=body.threshold,
        label=body.label,
        active=True,
    )
    db.add(alert)
    db.commit()
    db.refresh(alert)
    return {"id": alert.id, "ok": True}


@app.delete("/api/alerts/{alert_id}")
def delete_alert(alert_id: int, client_key: str = Depends(require_alert_key), db: Session = Depends(get_db)):
    row = db.query(models.Alert).filter(models.Alert.id == alert_id, models.Alert.client_key == client_key).first()
    if not row:
        raise HTTPException(status_code=404, detail="Alert not found")
    row.active = False
    db.commit()
    return {"ok": True}


@app.post("/api/alerts/check")
@limiter.limit("20/minute")
def check_stored_alerts(request: Request, client_key: str = Depends(require_alert_key), db: Session = Depends(get_db)):
    """Evaluate stored alerts; price rules latch until an opposite-side quote."""
    rows = (
        db.query(models.Alert)
        .filter(models.Alert.client_key == client_key, models.Alert.active == True)  # noqa: E712
        .with_for_update()
        .all()
    )
    rules = [
        {
            "id": a.id,
            "ticker": a.ticker,
            "type": a.alert_type,
            "condition": a.condition,
            "threshold": a.threshold,
            "was_triggered": bool(a.triggered),
        }
        for a in rows
    ]
    result = evaluate_alerts(rules)
    fired_ids = {f.get("id") for f in result.get("fired", [])}
    evaluations = {item.get("id"): item for item in result.get("results", [])}
    for a in rows:
        evaluation = evaluations.get(a.id)
        if evaluation is not None and evaluation.get("type") == "price":
            met = evaluation.get("condition_met", bool(evaluation.get("triggered")))
            # PostgreSQL locks the rows above. Compare-and-set also prevents two
            # SQLite checks that read the same state from publishing the same edge.
            changed = (db.query(models.Alert)
                       .filter(models.Alert.id == a.id, models.Alert.triggered.is_(a.triggered))
                       .update({
                           "triggered": met,
                           **({"last_triggered_at": datetime.utcnow()} if a.id in fired_ids else {}),
                       }, synchronize_session=False))
            if not changed:
                result["fired"] = [item for item in result["fired"] if item.get("id") != a.id]
                evaluation["triggered"] = False
                evaluation["message"] = None
        elif a.id in fired_ids:
            a.triggered = True
            a.last_triggered_at = datetime.utcnow()
    db.commit()
    return result


@app.websocket("/ws/quotes")
async def ws_quotes(websocket: WebSocket):
    """
    Near-real-time quote stream. Polls the configured provider on an interval.
    Query: ?tickers=NPN.JO,SBK.JO&interval=5
    """
    params = websocket.query_params
    tickers = list(dict.fromkeys(t.strip().upper() for t in (params.get("tickers") or "").split(",") if t.strip()))[:20]
    try:
        interval = max(3, min(60, int(params.get("interval") or "5")))
    except ValueError:
        interval = 5
    if not tickers:
        await websocket.accept()
        await websocket.send_json({"error": "Provide tickers query param"})
        await websocket.close()
        return
    ip = websocket.client.host if websocket.client else "unknown"
    if not quote_connections.acquire(ip):
        # Reject before accepting, without allocating any provider work.
        await websocket.close(code=1013, reason="Quote connection limit exceeded")
        return
    try:
        await websocket.accept()
        while True:
            payload = await run_in_threadpool(quote_cache.get_quotes, tickers, get_live_quotes)
            await websocket.send_json(payload)
            # Observe disconnects while waiting, rather than discovering them on
            # the next send. Client messages must not accelerate provider polling.
            deadline = asyncio.get_running_loop().time() + interval
            while True:
                remaining = deadline - asyncio.get_running_loop().time()
                if remaining <= 0:
                    break
                try:
                    message = await asyncio.wait_for(websocket.receive(), timeout=remaining)
                except asyncio.TimeoutError:
                    break
                if message["type"] == "websocket.disconnect":
                    return
    except (WebSocketDisconnect, OSError):
        logger.info("Quote WebSocket disconnected")
    except Exception:
        logger.exception("Quote WebSocket failed")
        try:
            await websocket.close()
        except Exception:
            pass
    finally:
        quote_connections.release(ip)


# Ensure new tables exist in local/dev even before migrations are applied
try:
    Base.metadata.create_all(bind=engine)
except Exception as e:
    logger.warning(f"create_all skipped: {e}")


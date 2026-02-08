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
from analysis import download_data, process_data, calculate_summary_stats, run_ml_analysis, run_anova_test, clean_data, calculate_dca, run_monte_carlo, get_company_profile, get_key_stats, get_news, get_calendar, get_article_content, search_tickers, get_dividend_history, get_financials, fetch_multiple_tickers, calculate_financial_freedom, get_jse_peers
import models, schemas, auth, database
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from fastapi import Depends, status

# Initialise DB
models.Base.metadata.create_all(bind=database.engine)


# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Global Index Analyzer API", version="3.0.0")

# CORS Setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development, allow all. In production, specify frontend URL.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic Models
class AnalysisRequest(BaseModel):
    ticker: str
    start_year: int
    end_date: str
class ComparisonRequest(BaseModel):
    tickers: List[str]
    start_year: int
    end_date: str

class SearchRequest(BaseModel):
    query: str

class FreedomRequest(BaseModel):
    ticker: str
    monthly_income_goal: float


# --- Auth Endpoints ---
@app.post("/api/auth/register", response_model=schemas.User)
def register(user: schemas.UserCreate, db: Session = Depends(database.get_db)):
    db_user = db.query(models.User).filter(models.User.email == user.email).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    hashed_password = auth.get_password_hash(user.password)
    # Automatically assign "pro" tier for demonstration, or default to "free"
    # Using "free" as default, but let's make the FIRST user admin/institutional? No, keep simple.
    db_user = models.User(email=user.email, hashed_password=hashed_password, tier="free")
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

@app.post("/api/auth/token", response_model=schemas.Token)
def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(database.get_db)):
    user = db.query(models.User).filter(models.User.email == form_data.username).first()
    if not user or not auth.verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = auth.timedelta(minutes=auth.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = auth.create_access_token(
        data={"sub": user.email}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer", "tier": user.tier}

@app.get("/api/auth/me", response_model=schemas.User)
def read_users_me(current_user: schemas.User = Depends(auth.get_current_active_user)):
    return current_user

@app.post("/api/auth/upgrade", response_model=schemas.User)
def upgrade_user(upgrade: schemas.UserUpgrade, db: Session = Depends(database.get_db), current_user: models.User = Depends(auth.get_current_active_user)):
    current_user.tier = upgrade.tier
    db.commit()
    db.refresh(current_user)
    return current_user
# ----------------------

@app.get("/")
def read_root():
    return {"message": "Global Index Analyzer API is running"}

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
    
    stats = calculate_summary_stats(processed['monthly_ret'], processed['pivot'])
    
    # Prepare pivot data for JSON (reset index to make year a column)
    pivot_reset = processed['pivot'].reset_index()
    pivot_data = pivot_reset.to_dict(orient='records')
    
    response_data = {
        "ticker": request.ticker,
        "stats": stats,
        "pivot_data": pivot_data,
        "monthly_returns": processed['monthly_ret'].to_dict(), # Date -> Return
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
        
    result = calculate_financial_freedom(data, request.monthly_income_goal)
    
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
        stats = calculate_summary_stats(processed['monthly_ret'], processed['pivot'])
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
        
    stats = calculate_summary_stats(processed['monthly_ret'], processed['pivot'])
    
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    elements = []
    
    styles = getSampleStyleSheet()
    elements.append(Paragraph(f"Analysis Report for {request.ticker}", styles['Title']))
    elements.append(Paragraph(f"Period: {request.start_year} to {request.end_date}", styles['Normal']))
    elements.append(Spacer(1, 12))
    
    # Summary Metrics Table
    if stats:
        data_table = [
            ["Metric", "Value"],
            ["CAGR", f"{stats['cagr']:.2%}" if stats['cagr'] else "N/A"],
            ["Volatility", f"{stats['volatility']:.2%}" if stats['volatility'] else "N/A"],
            ["Sharpe Ratio", f"{stats['sharpe_ratio']:.2f}" if stats['sharpe_ratio'] else "N/A"],
            ["Max Drawdown", f"{stats['max_drawdown']:.2%}" if stats['max_drawdown'] else "N/A"],
        ]
        
        t = Table(data_table)
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (0, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ]))
        elements.append(t)
        
    doc.build(elements)
    buffer.seek(0)
    
    headers = {
        'Content-Disposition': f'attachment; filename="{request.ticker}_report.pdf"'
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


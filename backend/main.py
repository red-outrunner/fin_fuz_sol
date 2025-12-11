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
from analysis import download_data, process_data, calculate_summary_stats, run_ml_analysis, run_anova_test, clean_data, calculate_dca, run_monte_carlo, get_company_profile


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

@app.get("/")
def read_root():
    return {"message": "Global Index Analyzer API is running"}

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
    
    for ticker in request.tickers:
        data = download_data(ticker, start_date, request.end_date)
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
    # Strategy: Download all, merge on Date (monthly)
    
    all_series = {}
    
    for ticker in request.tickers:
        data = download_data(ticker, start_date, request.end_date)
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


@app.post("/api/profile")
def company_profile(request: AnalysisRequest):
    logger.info(f"Fetching Profile for {request.ticker}")
    profile = get_company_profile(request.ticker)
    if profile is None:
         # Return empty structure instead of error for UI smoothness
         return {"biggest_shareholder": None, "sentiment": None}
    return clean_data(profile)

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


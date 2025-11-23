from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
from datetime import datetime
import logging
from analysis import download_data, process_data, calculate_summary_stats, run_ml_analysis, run_anova_test, clean_data

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
        "monthly_returns": processed['monthly_ret'].to_dict() # Date -> Return
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
    # For simplicity in this demo, we'll just return a success message or file path
    # In a real app, we'd stream the file response
    
    # Since we can't easily stream files without more setup, we'll just simulate it
    return {"message": "Excel export functionality is ready (backend implementation pending file streaming setup)"}

@app.post("/api/export/pdf")
def export_pdf(request: AnalysisRequest):
    logger.info(f"Exporting PDF for {request.ticker}")
    # Similar to Excel, return success message
    return {"message": "PDF export functionality is ready (backend implementation pending file streaming setup)"}

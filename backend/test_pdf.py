import sys
import os
sys.path.append(os.getcwd())

import io
from reports import PDFReportGenerator
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Mock Data
def get_mock_data():
    dates = pd.date_range(start='2020-01-01', periods=36, freq='M')
    data = pd.DataFrame({
        'Close': np.random.normal(100, 10, 36).cumsum()
    }, index=dates)
    processed = {
        'monthly_ret': data['Close'].pct_change().dropna(),
        'pivot': pd.DataFrame(np.random.rand(3, 12), columns=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'], index=[2020, 2021, 2022])
    }
    stats = {
        'cagr': 0.15,
        'volatility': 0.20,
        'sharpe_ratio': 1.5,
        'max_drawdown': -0.10,
        'best_year': 0.30,
        'worst_year': -0.05
    }
    return processed, stats

print("Generating PDF...")
try:
    processed, stats = get_mock_data()
    buffer = io.BytesIO()
    
    gen = PDFReportGenerator(buffer, "TEST.TICKER", 2020, "2023-01-01")
    gen.add_title_page()
    gen.add_executive_summary(stats)
    gen.add_wealth_chart(processed)
    gen.add_drawdown_chart(processed)
    gen.add_monthly_table(processed)
    
    # Mock Peer Data
    gen.add_peer_battle({"PEER1": [0.01]*12, "PEER2": [0.02]*12})
    
    # Mock Monte Carlo
    gen.add_monte_carlo({'10th': [100]*12, '50th': [110]*12, '90th': [120]*12})
    
    # Mock DCA
    dca_data = []
    for i in range(12):
        dca_data.append({'Date': f"2022-{i+1:02d}-01", 'Portfolio Value': 1000*(i+1)*1.1, 'Total Invested': 1000*(i+1)})
    gen.add_dca_analysis({'dca_data': dca_data, 'summary': {'total_invested': 12000, 'final_value': 15000, 'profit': 3000, 'return_pct': 0.25}})
    
    gen.build_pdf()
    print("PDF Generated Successfully!")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

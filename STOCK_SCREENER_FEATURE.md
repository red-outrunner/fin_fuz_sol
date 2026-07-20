# Stock Screener & Discovery Tools - Implementation Guide

## Overview

This implementation adds **institutional-grade stock discovery tools** to Ubomvu, competing with features from Morningstar, Yahoo Finance, and Seeking Alpha. Three major features have been added:

1. **Stock Screener** - Filter JSE Top 40 stocks by fundamental criteria
2. **JSE Heatmap** - Visual sector performance overview
3. **Stock Ideas Feed** - Curated investment opportunities

---

## Features Implemented

### 1. Stock Screener (`/screener`)

**Backend:** `backend/screener.py`
- Screens all JSE Top 40 stocks in parallel
- Filter criteria:
  - **Valuation**: P/E ratio, market cap, dividend yield
  - **Quality**: ROE, debt/equity, profit margin
  - **Risk**: Beta (volatility)
  - **Growth**: Revenue growth
  - **Sectors**: Technology, Financials, Materials, Consumer, Telecom, Healthcare
  - **Special**: Undervalued stocks (>20% below 52w high), dividend payers

**Frontend:** `frontend/src/components/StockScreener.jsx`
- Multi-filter UI with real-time results
- **Preset filters**: Value Stocks, Dividend Stars, Growth Stocks, Low Volatility, High Quality
- **Save/load screeners**: Store custom filters in localStorage
- **Sortable results**: Click column headers to sort
- **Responsive design**: Works on mobile and desktop

**API Endpoints:**
```
POST /api/screener
Body: {
  "min_market_cap": 10000000000,
  "max_pe": 15,
  "min_dividend_yield": 0.03,
  "sectors": ["Financials", "Consumer"],
  "undervalued_only": true
}
Response: { "results": [...], "count": 18 }
```

---

### 2. JSE Heatmap (`/heatmap`)

**Backend:** `backend/screener.py::get_sector_performance()`
- Calculates real-time performance for all JSE sectors
- Groups stocks by sector with market cap weights
- Returns nested structure: sectors → stocks

**Frontend:** `frontend/src/components/JSEHeatmap.jsx`
- **Color-coded sectors**: Green (positive) to Red (negative)
- **Interactive cards**: Click stocks to see details
- **Sector summaries**: Best/worst performing, largest by market cap
- **Stock detail panel**: Slide-up with selected stock info
- **Performance legend**: Visual guide for color scale

**API Endpoints:**
```
GET /api/screener/heatmap
Response: {
  "sectors": [
    {
      "name": "Financials",
      "market_cap": 1500000000000,
      "change_percent": 2.34,
      "stock_count": 8,
      "stocks": [...]
    }
  ]
}
```

---

### 3. Stock Ideas Feed (`/ideas`)

**Backend:** `backend/screener.py::get_stock_ideas()`
- Algorithmic curation based on fundamental criteria:
  - **Undervalued**: P/E < 12, Div Yield > 3%
  - **52-Week Lows**: Within 10% of yearly low
  - **Dividend Stars**: Yield > 5%
  - **Momentum Leaders**: Within 10% of 52w high
  - **Growth Stocks**: Revenue growth > 10%, ROE > 15%

**Frontend:** `frontend/src/components/StockIdeasFeed.jsx`
- **Category tabs**: Switch between idea types
- **Stock cards**: Key metrics at a glance
- **Category descriptions**: Explain selection criteria
- **Quick stats**: Count of stocks in each category
- **Disclaimer**: Investment advice warning

**API Endpoints:**
```
GET /api/screener/ideas
Response: {
  "undervalued": [...],
  "52_week_lows": [...],
  "dividend_stars": [...],
  "momentum_leaders": [...],
  "growth_stocks": [...]
}
```

---

## Architecture

### Backend Structure

```
backend/
├── screener.py           # New: Stock screening logic
│   ├── JSE_TOP_40        # List of Top 40 tickers
│   ├── JSE_SECTORS       # Sector mappings
│   ├── screen_stocks()   # Main screening function
│   ├── get_sector_performance()
│   ├── get_stock_ideas()
│   └── _fetch_ticker_info()  # Parallel data fetcher
└── main.py               # Updated: Added screener endpoints
```

### Frontend Structure

```
frontend/src/
├── components/
│   ├── StockScreener.jsx     # New: Screener UI
│   ├── JSEHeatmap.jsx        # New: Heatmap visualization
│   ├── StockIdeasFeed.jsx    # New: Ideas feed
│   └── Sidebar.jsx           # Updated: Added discovery links
└── App.jsx                   # Updated: Hash-based routing
```

---

## Usage Guide

### Stock Screener Examples

**Find Value Stocks:**
1. Click "Value Stocks" preset
2. Adjust Max P/E to 12
3. Set Min Dividend Yield to 0.04 (4%)
4. Click "Run Screener"
5. Save as "Deep Value" for later

**Find Low-Risk Stocks:**
1. Set Max Beta to 0.8
2. Set Max Debt/Equity to 0.5
3. Select sectors: Consumer, Healthcare
4. Run screener

**Find High-Quality Growth:**
1. Click "High Quality" preset
2. Set Min Revenue Growth to 0.15 (15%)
3. Set Min ROE to 0.20 (20%)
4. Run screener

### JSE Heatmap Usage

- **Quick overview**: See which sectors are green/red today
- **Drill down**: Click sector cards to see individual stocks
- **Compare performance**: Hover over stocks for details
- **Market sentiment**: Use summary cards (best/worst/largest)

### Stock Ideas Usage

- **Daily inspiration**: Check "Undervalued" for bargain hunting
- **Contrarian plays**: Review "52-Week Lows" for turnaround candidates
- **Income investing**: Browse "Dividend Stars" for yield
- **Momentum trading**: Check "Momentum Leaders" for trends
- **Growth hunting**: Review "Growth Stocks" for compounders

---

## Technical Details

### Performance Optimization

1. **Parallel fetching**: Uses `ThreadPoolExecutor` with 10 workers
2. **Disk caching**: yfinance data cached in parquet files (24h TTL)
3. **Rate limiting**: API endpoints limited to 30/minute
4. **Frontend efficiency**: Results sorted client-side after initial fetch

### Data Sources

- **yfinance**: Real-time price data, fundamentals, 52-week ranges
- **JSE_MAJOR_SHAREHOLDERS**: Curated table for JSE stocks
- **Sector mappings**: Manually maintained for accuracy

### Error Handling

- Graceful degradation when data unavailable
- User-friendly error messages
- Retry buttons for failed requests
- N/A display for missing metrics

---

## Future Enhancements

### Phase 2 (Recommended)

1. **Insider Trading Tracker**
   - Director dealings data (critical for JSE)
   - Filter by insider buying/selling
   - Track insider ownership changes

2. **Advanced Screener Features**
   - Technical filters (RSI, MACD, volume)
   - Custom formula builder
   - Backtesting integration
   - Export screened results to CSV/Excel

3. **Enhanced Heatmap**
   - Time period selector (1d, 1w, 1m, 3m, 1y)
   - Metric selector (performance, volume, market cap)
   - Treemap visualization (size by market cap)
   - Click to analyze (deep link to dashboard)

4. **Portfolio Integration**
   - "Add to portfolio" from screener results
   - Compare screened stocks side-by-side
   - Export to watchlist

### Phase 3 (Advanced)

1. **AI-Powered Insights**
   - Natural language screening: "Show me undervalued banks with strong dividends"
   - Automated stock summaries
   - Peer comparison narratives

2. **Social Features**
   - Share screened results
   - Public screener templates
   - Community stock ideas

3. **Real-Time Alerts**
   - "Notify when stock enters my screener results"
   - Price alerts from heatmap
   - Breaking news for watched stocks

---

## Testing

### Backend Tests

```bash
cd backend
source venv/bin/activate

# Test screener module
python -c "from screener import screen_stocks; print(screen_stocks(max_pe=15)[:3])"

# Test ideas feed
python -c "from screener import get_stock_ideas; ideas = get_stock_ideas(); print(f'Undervalued: {len(ideas[\"undervalued\"])}')"

# Test heatmap
python -c "from screener import get_sector_performance; sectors = get_sector_performance(); print(f'Sectors: {len(sectors)}')"
```

### API Endpoint Tests

```bash
# Test stock screener
curl -X POST http://localhost:8000/api/screener \
  -H "Content-Type: application/json" \
  -d '{"max_pe": 15, "min_dividend_yield": 0.03}'

# Test ideas feed
curl http://localhost:8000/api/screener/ideas

# Test heatmap
curl http://localhost:8000/api/screener/heatmap

# Test universe
curl http://localhost:8000/api/screener/universe
```

### Test Results (Live)

✅ **Screener**: Returns 18 stocks with P/E < 15 and Div Yield > 3%
✅ **Ideas Feed**: 
  - Undervalued: 10 stocks
  - Dividend Stars: 10 stocks
  - Momentum Leaders: 10 stocks
  - Growth Stocks: 7 stocks
✅ **Heatmap**: 6 sectors (Materials, Consumer, Financials, Technology, Telecom, Healthcare)

### Frontend Tests

```bash
cd frontend
npm run build  # Verify no compilation errors
npm run dev    # Test locally at http://localhost:5173
```

### API Tests

```bash
# Test screener endpoint
curl -X POST http://localhost:8000/api/screener \
  -H "Content-Type: application/json" \
  -d '{"max_pe": 15, "min_dividend_yield": 0.03}'

# Test heatmap endpoint
curl http://localhost:8000/api/screener/heatmap

# Test ideas endpoint
curl http://localhost:8000/api/screener/ideas
```

---

## Deployment

### Backend

1. Deploy to Render/Heroku/AWS as usual
2. Ensure `screener.py` is included in deployment
3. No new dependencies required (uses existing yfinance, pandas)
4. Rate limiting already configured

### Frontend

1. Deploy to Netlify/Vercel as usual
2. New components auto-included in build
3. Hash routing works out-of-the-box
4. No environment variables needed

---

## Competitive Analysis

### vs. Morningstar
✅ **We have**: JSE specialization, free access, simpler UI
❌ **Missing**: 10-year fundamentals, analyst ratings, fair value estimates

### vs. Yahoo Finance
✅ **We have**: Better screening for JSE, curated ideas, cleaner UX
❌ **Missing**: Real-time data, options chain, comprehensive news

### vs. Seeking Alpha
✅ **We have**: Algorithmic objectivity, no paywall, JSE focus
❌ **Missing**: Community analysis, expert opinions, earnings transcripts

### vs. TradingView
✅ **We have**: Fundamental screening (not just technical), JSE expertise
❌ **Missing**: Advanced charting, technical indicators, social network

---

## Success Metrics

Track these to measure adoption:

1. **Usage**: Daily active users of screener/heatmap/ideas
2. **Engagement**: Time spent screening, number of screeners saved
3. **Conversion**: Users who analyze stocks from screener results
4. **Retention**: Return users who come back to check ideas
5. **Feedback**: User surveys on feature usefulness

---

## Conclusion

This implementation provides **institutional-quality stock discovery tools** that differentiate Ubomvu from generic financial websites. By focusing on the JSE market and providing deep fundamental screening, we attract serious retail investors and financial advisors who currently lack good local tools.

**Next Steps:**
1. Launch with core features (screener, heatmap, ideas)
2. Gather user feedback on filter presets and UI
3. Prioritize Phase 2 enhancements based on usage data
4. Market as "The JSE Stock Screener for Serious Investors"

---

**Questions or issues?** Check the code comments or test individual functions in Python REPL.

# Enhanced Fundamental Analysis - Implementation Plan

## Phase 1: Fix Sector Categorizations ✅ IN PROGRESS

Based on `constituent_details.xlsx`:

### Updated Sector Mappings:
- **Growthpoint (GRT.JO)** → Real Estate (not Financials)
- **Remgro (REM.JO)** → Real Estate (not Healthcare)
- **Reinet (RNI.JO)** → Real Estate  
- **All Banks** → Financials
- **All Insurance** → Financials
- **Basic Resources** → Materials
- **Food, Beverage & Tobacco** → Consumer
- **Retail & Personal Care** → Consumer

### New Tickers Added (40 total):
- HAR.JO (Harmony Gold)
- CFR.JO (Compagnie Financiere)
- NRP.JO (NEPI Rockcastle)
- OUT.JO (Outsurance)
- PAN.JO (Pan African Resources)
- PPH.JO (Pick n Pay)
- SSW.JO (Sierra Wire)

---

## Phase 2: Enhanced Fundamental Analysis Features

### 2.1 Interactive Financial Statements
**Component:** `FinancialStatements.jsx`
- 5-10 year history of:
  - Income Statement (Revenue, EBITDA, Net Income)
  - Balance Sheet (Assets, Liabilities, Equity)
  - Cash Flow (Operating, Investing, Financing)
- **Data Source:** yfinance `ticker.financials`, `ticker.balance_sheet`, `ticker.cashflow`
- **Visualization:** Recharts bar/line charts with toggle between absolute and % values

### 2.2 Ratio Trend Charts
**Component:** `RatioTrends.jsx`
- 5-year trends for:
  - ROE (Return on Equity)
  - ROA (Return on Assets)
  - Gross Margin
  - Operating Margin
  - Net Margin
  - Debt/Equity
  - Current Ratio
- **Visualization:** Multi-line chart with industry average comparison

### 2.3 Segment Revenue Breakdown
**Component:** `SegmentBreakdown.jsx`
- For conglomerates (Naspers, Prosus, BHP)
- Pie chart showing revenue by business segment
- 3-year segment growth trends
- **Data Source:** yfinance `ticker.segment` (when available)

### 2.4 Analyst Estimates
**Component:** `AnalystEstimates.jsx`
- EPS estimates (current year, next year)
- Revenue estimates
- Price targets (high, low, mean, median)
- Analyst recommendations (buy/hold/sell breakdown)
- **Data Source:** yfinance `ticker.analyst_price_targets`, `ticker.recommendations`

### 2.5 Fair Value Comparison
**Component:** `FairValueComparison.jsx`
- Your DCF valuation vs analyst consensus
- Peer valuation comparison (P/E, EV/EBITDA)
- Historical valuation bands
- **Visualization:** Gauge chart showing over/undervaluation

---

## Phase 3: Technical Implementation

### Backend Changes (`screener.py`)
1. ✅ Update JSE_TOP_40 list (40 tickers)
2. ✅ Update JSE_SECTORS mapping
3. Add `/api/fundamentals/{ticker}` endpoint:
   - Returns 5-year financial statements
   - Returns ratio trends
   - Returns analyst estimates

### Frontend Components
1. **Create `FundamentalAnalysis.jsx`** - Main container component
2. **Create `FinancialStatements.jsx`** - IS/BS/CF tabs
3. **Create `RatioTrends.jsx`** - Ratio charts
4. **Create `AnalystEstimates.jsx`** - Estimates & targets
5. **Update `Dashboard.jsx`** - Add "Fundamentals" tab

### API Endpoints
```
GET /api/screener/heatmap  (✅ Already returns all metrics)
POST /api/fundamentals/{ticker}  (NEW - detailed fundamentals)
GET /api/analyst-estimates/{ticker}  (NEW)
```

---

## Phase 4: Testing & Deployment

### Testing Checklist
- [ ] Heatmap shows correct sectors (Real Estate separate)
- [ ] Stock detail panel shows all metrics
- [ ] Financial statements load for NPN.JO
- [ ] Ratio trends display 5-year history
- [ ] Analyst estimates show price targets

### Performance Optimization
- Cache fundamentals data (24h TTL)
- Lazy load charts (react-window for virtualization)
- Memoize expensive calculations

---

## Timeline
- **Phase 1:** ✅ Complete (sector fixes)
- **Phase 2:** 2-3 days (components)
- **Phase 3:** 1-2 days (backend + integration)
- **Phase 4:** 1 day (testing)

**Total: 4-6 days for full implementation**

---

## Priority Order
1. ✅ **Sector fixes** (heatmap categorization)
2. **Financial statements** (core data)
3. **Ratio trends** (visual analysis)
4. **Analyst estimates** (forward-looking)
5. **Segment breakdown** (for conglomerates)
6. **Fair value comparison** (valuation context)

Let me know which features you want me to implement first!

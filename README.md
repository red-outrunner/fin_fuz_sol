# Global Index Analyzer (Ubomvu)

A sophisticated financial analysis platform designed for retail investors (with specific focus on JSE) and institutional analysts. Ubomvu provides deep insights into stock performance, risk metrics, and future projections using statistical models and machine learning.

## 🚀 Features

- **Advanced Analytics**: Monthly return heatmaps, volatility analysis, and moving averages.
- **Valuation Lab**: Interactive Discounted Cash Flow (DCF) models.
- **Smart Reports**: Generate professional PDF investment reports with charts and executive summaries.
- **Wealth Projection**: Monte Carlo simulations to forecast portfolio growth.
- **ML Analysis**: Cluster analysis and anomaly detection for market trends.
- **Dividend Analysis**: Deep dive into dividend history and yield metrics.
- **Comparison Engine**: Benchmarking stocks against peers and indices.

## 🛠️ Architecture

The application is built with a decoupled architecture:

- **Frontend**: React (Vite) + Tailwind CSS + Lucide Icons + Recharts.
- **Backend**: FastAPI (Python) + SQLAlchemy + Pydantic.
- **Infrastructure**: Netlify (Frontend) + Render/Any Python Host (Backend).

## 📦 Getting Started

### Prerequisites

- Python 3.9+
- Node.js 18+
- SQLite (local development) / PostgreSQL (production)

### Backend Setup

1. Navigate to the `backend` directory:
   ```bash
   cd backend
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Setup environment variables in a `.env` file:
   ```env
   DATABASE_URL=sqlite:///./users.db
   SECRET_KEY=your_secret_key
   ```
4. Run the API server:
   ```bash
   uvicorn main:app --reload
   ```

### Frontend Setup

1. Navigate to the `frontend` directory:
   ```bash
   cd frontend
   ```
2. Install dependencies:
   ```bash
   npm install
   ```
3. Run the development server:
   ```bash
   npm run dev
   ```

## 🗄️ Database Migrations (Alembic)

Portfolio access is anonymous and private to a browser. The frontend generates a
256-bit key, persists it in localStorage, and sends it in the `X-Portfolio-Key`
header for every portfolio request. The API stores its SHA-256 fingerprint and
checks it when reading, adding, or deleting holdings. Treat the key as a password:
use HTTPS in production. Clearing browser storage loses access; there is no account
recovery or cross-device sync. If storage is blocked, access lasts only for the page
session. Legacy holdings under public client IDs (including `default`) remain in
the database but are inaccessible through these endpoints; they cannot safely be
automatically assigned to a browser. The old `client_key` query parameter no longer
grants portfolio access.

Portfolio valuation never substitutes zero prices or fixed exchange rates for
missing provider data. Holding responses expose `valuation_status` and
`missing_data` (`price` and/or `fx`); dependent values are `null` when unavailable.
Native-currency values remain available when only FX is missing. If any holding
cannot be valued in ZAR, portfolio value, P&L, return, and sector allocation are
withheld rather than shown as a partial total. Total cost remains available when
all required exchange rates are known. The UI and CSV export display unavailable
values as `N/A`.

Stored alerts use a separate 256-bit browser-generated key in the `X-Alert-Key`
header. Only its SHA-256 fingerprint is stored on the server. Use `GET/POST
/api/alerts`, `DELETE /api/alerts/{alert_id}`, and `POST /api/alerts/check`; credentials
are never part of these URLs. Old client-key routes and body/query credentials no
longer grant access. Legacy alerts remain in the database but must be recreated,
since their ownership cannot safely be inferred. Clearing browser storage loses
access to anonymous alerts; use HTTPS in production.

Alert creation uses `alert_type` (`price`, `volume`, or `earnings`). Stateless
`POST /api/alerts/evaluate` accepts up to 50 rules using `type` and requires no
stored-alert credential. Price/volume rules require finite, nonnegative thresholds;
invalid types, conditions, tickers, and thresholds return HTTP 422. Malformed
persisted rules are skipped and reported in the evaluation response's `errors`.

Stored price alerts start armed. The first quote at or beyond the threshold fires
once; repeated qualifying quotes do not notify again. A quote on the opposite side
rearms the alert. `triggered` records the current latch, `armed` exposes its inverse,
and `last_triggered_at` retains the last notification time through rearming. Missing
quotes preserve the latch. Stateless evaluation callers must send the previous
result's `condition_met` as `was_triggered` on their next check to get the same
transition behavior. No database migration is needed for this state handling.

DCF inputs expose `fcf` in the listing's major trading currency (`fcf_currency`),
converted from `financialCurrency` before calculating per-share fair value.
`fcf_reporting`, `reporting_currency`, and `fx_rate` describe the original cash flow
and conversion. Missing currency metadata or FX rates withhold the comparison
instead of relabeling unconverted cash flows. Financial freedom uses dividend
payments in the last 12 calendar months ending today, in the dividend index's
timezone; old and future payments do not contribute. Dashboard exports send the
selected inflation rate so exported summary statistics match the displayed KPIs.

Schema changes are versioned with [Alembic](https://alembic.sqlalchemy.org/). Run commands from the `backend/` directory; the database URL is taken from `DATABASE_URL` (falls back to local SQLite).

```bash
cd backend
alembic upgrade head                          # apply all migrations
alembic revision --autogenerate -m "message"  # create a migration from model changes
alembic downgrade -1                           # roll back one migration
```

Production applies migrations automatically (`render.yaml` runs `alembic upgrade head` before starting the server).

> If you have an existing dev database that was created by the app's `create_all` safety net (so it has the `users` table but no `alembic_version`), baseline it once with `alembic stamp head` before applying new migrations — or just delete `users.db` and run `alembic upgrade head`.

## 🌐 Deployment

The project includes configurations for:
- **Netlify**: `netlify.toml` for frontend deployment.
- **Render**: `render.yaml` for backend deployment.

## 📄 License

This project is licensed under the terms found in the [LICENSE](LICENSE) file.


### Sector metadata administration

Startup loads the sector JSON without searching server directories for spreadsheets.
`POST /api/admin/sync-sectors` reloads that JSON and requires an
`Authorization: Bearer <SECTOR_ADMIN_TOKEN>` header over HTTPS. Set
`SECTOR_ADMIN_TOKEN` on the backend to a private random value (generate one with
`openssl rand -hex 32`); absent or shorter-than-32-character configuration disables
this endpoint. It accepts no query parameters and does not import spreadsheets.

To import a spreadsheet, run the local administrative CLI from `backend/`:

```bash
python sector_manager.py --import-excel /trusted/path/constituent_details.xlsx
```

Then restart the backend or call the authenticated reload endpoint on each running
instance. Existing public stock and sector read endpoints remain available.


Daily-feature selections are persisted in `daily_stock_picks`, including the
selected fallback ticker and company name. Yesterday's feature reads that saved
record, so provider recovery, a changed universe, or a backend restart cannot
replace the actual pick. Dates without a published record have no yesterday card.
Apply the new table with `alembic upgrade head` from `backend/` before deployment.

Frontend request-order regressions run with `npm run test:requests` from
`frontend/`. They exercise the actual analysis and valuation components with
out-of-order responses, including transports that ignore cancellation.

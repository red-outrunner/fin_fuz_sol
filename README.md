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

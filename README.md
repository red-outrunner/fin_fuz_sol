# Global Index Analyzer (Fuzile)

A sophisticated financial analysis platform designed for retail investors (with specific focus on JSE) and institutional analysts. Fuzile provides deep insights into stock performance, risk metrics, and future projections using statistical models and machine learning.

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

## 🌐 Deployment

The project includes configurations for:
- **Netlify**: `netlify.toml` for frontend deployment.
- **Render**: `render.yaml` for backend deployment.

## 📄 License

This project is licensed under the terms found in the [LICENSE](LICENSE) file.
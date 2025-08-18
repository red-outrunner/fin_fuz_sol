import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import random

# --- 1. Client Profile Analysis ---
class ClientProfile:
    """
    Analyzes and quantifies a client's investment profile.
    """
    def __init__(self, risk_tolerance, investment_horizon, liquidity_needs, income, portfolio_preferences, sector_preferences):
        """
        Initializes the client profile.
        
        Args:
            risk_tolerance (str): 'low', 'medium', 'high'
            investment_horizon (int): Years
            liquidity_needs (str): 'low', 'medium', 'high'
            income (float): Annual income
            portfolio_preferences (list): e.g., ['growth', 'value', 'ethical']
            sector_preferences (list): e.g., ['tech', 'healthcare', 'energy']
        """
        self.risk_tolerance = risk_tolerance
        self.investment_horizon = investment_horizon
        self.liquidity_needs = liquidity_needs
        self.income = income
        self.portfolio_preferences = portfolio_preferences
        self.sector_preferences = sector_preferences
        self.risk_score = self._calculate_risk_score()

    def _calculate_risk_score(self):
        """
        Converts client's qualitative risk tolerance into a numerical score.
        A higher score means a higher tolerance for risk.
        """
        score = 0
        # Risk Tolerance Mapping
        if self.risk_tolerance == 'high':
            score += 40
        elif self.risk_tolerance == 'medium':
            score += 20
        else: # low
            score += 5

        # Investment Horizon Impact
        if self.investment_horizon > 10: # Long-term
            score += 20
        elif self.investment_horizon > 5: # Medium-term
            score += 10

        # Liquidity Needs Impact (inverse relationship)
        if self.liquidity_needs == 'low':
            score += 15
        elif self.liquidity_needs == 'medium':
            score += 5
            
        # Income can also play a role, higher income might support more risk
        if self.income > 150000:
            score += 10

        # Normalize score to be between 0 and 100
        return min(max(score, 0), 100)

    def get_profile(self):
        """Returns a summary of the client's profile."""
        return {
            "risk_tolerance": self.risk_tolerance,
            "investment_horizon": self.investment_horizon,
            "liquidity_needs": self.liquidity_needs,
            "income": self.income,
            "portfolio_preferences": self.portfolio_preferences,
            "sector_preferences": self.sector_preferences,
            "risk_score": self.risk_score
        }

# --- 2. Market Data Ingestion & Processing ---
class MarketData:
    """
    Handles fetching and processing of market data.
    In a real application, this would connect to APIs like Alpha Vantage, Bloomberg, or Reuters.
    For this example, we'll simulate the data.
    """
    def __init__(self, assets):
        self.assets = assets
        self.historical_data = self._fetch_historical_data()
        self.macro_indicators = self._fetch_macro_indicators()

    def _fetch_historical_data(self):
        """Simulates fetching historical price data for a list of assets."""
        print("Fetching historical market data...")
        data = {}
        for asset in self.assets:
            # Simulate daily prices for the last 3 years
            dates = pd.date_range(end=pd.Timestamp.today(), periods=365 * 3)
            # Simulate price movements with some randomness
            price_movements = np.random.randn(len(dates)).cumsum()
            start_price = random.uniform(50, 500)
            prices = start_price + price_movements
            data[asset] = pd.Series(prices, index=dates)
        return pd.DataFrame(data)

    def _fetch_macro_indicators(self):
        """Simulates fetching macroeconomic indicators."""
        print("Fetching macroeconomic indicators...")
        dates = self.historical_data.index
        gdp_growth = pd.Series(np.random.uniform(1.5, 3.5, size=len(dates)), index=dates)
        inflation_rate = pd.Series(np.random.uniform(1.0, 4.0, size=len(dates)), index=dates)
        return pd.DataFrame({'GDP_Growth': gdp_growth, 'Inflation': inflation_rate})

    def get_data(self):
        return self.historical_data, self.macro_indicators

# --- 3. Predictive Modeling & Forecasting ---
class PredictiveModel:
    """
    Forecasts future asset performance using predictive models.
    This example uses a simple Linear Regression model.
    Real-world models: ARIMA, GARCH, LSTMs, Gradient Boosting.
    """
    def __init__(self, data):
        self.data = data
        self.models = {}

    def train(self):
        """Trains a model for each asset."""
        print("Training predictive models...")
        for asset in self.data.columns:
            df = self.data[[asset]].copy()
            # Feature engineering: use lagged prices to predict future price
            df['target'] = df[asset].shift(-30) # Predict 30 days ahead
            df.dropna(inplace=True)
            
            X = df[[asset]]
            y = df['target']
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            model = LinearRegression()
            model.fit(X_train, y_train)
            self.models[asset] = model
            print(f"  - Model for {asset} trained. Score: {model.score(X_test, y_test):.2f}")

    def forecast(self):
        """Generates future return projections."""
        print("Forecasting future asset performance...")
        forecasts = {}
        for asset, model in self.models.items():
            current_price = self.data[asset].iloc[-1]
            predicted_price = model.predict([[current_price]])[0]
            expected_return = (predicted_price - current_price) / current_price
            
            # Simulate volatility (standard deviation of daily returns)
            volatility = self.data[asset].pct_change().std() * np.sqrt(252) # Annualized
            
            forecasts[asset] = {
                "expected_return": expected_return,
                "volatility": volatility,
                "current_price": current_price
            }
        return forecasts

# --- 4. Portfolio Optimization ---
class PortfolioOptimizer:
    """
    Optimizes portfolio allocation based on client profile and forecasts.
    This example uses a simple risk-based allocation.
    Real-world models: Mean-Variance Optimization, Black-Litterman, Monte Carlo Simulation.
    """
    def __init__(self, client_profile, forecasts):
        self.client_profile = client_profile
        self.forecasts = forecasts

    def optimize(self):
        """
        Determines the optimal asset allocation.
        """
        print("Optimizing portfolio allocation...")
        risk_score = self.client_profile.risk_score
        
        # Simple allocation strategy based on risk score
        # Higher risk score -> more allocation to higher return/volatility assets
        
        # Categorize assets by expected return
        sorted_assets = sorted(self.forecasts.items(), key=lambda x: x[1]['expected_return'], reverse=True)
        
        allocations = {}
        
        # Define asset classes based on risk/return profile (simplified)
        high_risk_assets = [a[0] for a in sorted_assets[:2]]
        medium_risk_assets = [a[0] for a in sorted_assets[2:4]]
        low_risk_assets = [a[0] for a in sorted_assets[4:]]

        # Allocate based on risk score
        if risk_score > 70: # Aggressive
            allocations.update({asset: 0.30 for asset in high_risk_assets})
            allocations.update({asset: 0.15 for asset in medium_risk_assets})
            allocations.update({asset: 0.05 for asset in low_risk_assets})
        elif risk_score > 40: # Moderate
            allocations.update({asset: 0.15 for asset in high_risk_assets})
            allocations.update({asset: 0.25 for asset in medium_risk_assets})
            allocations.update({asset: 0.10 for asset in low_risk_assets})
        else: # Conservative
            allocations.update({asset: 0.05 for asset in high_risk_assets})
            allocations.update({asset: 0.15 for asset in medium_risk_assets})
            allocations.update({asset: 0.30 for asset in low_risk_assets})
            
        # Normalize to ensure sum is 100%
        total_allocation = sum(allocations.values())
        final_allocations = {asset: (alloc / total_allocation) for asset, alloc in allocations.items()}
        
        return final_allocations

# --- 5. Main Virtual CIO Orchestrator ---
class VirtualCIO:
    """
    The main class that orchestrates the entire process.
    """
    def __init__(self, client_info, market_assets):
        self.client_info = client_info
        self.market_assets = market_assets

    def generate_recommendations(self):
        """
        Runs the full pipeline from client analysis to portfolio recommendation.
        """
        print("--- Starting Virtual CIO Analysis ---")
        
        # 1. Analyze Client Profile
        client = ClientProfile(**self.client_info)
        profile = client.get_profile()
        print(f"\nAnalyzed Client Profile. Risk Score: {profile['risk_score']}")
        
        # 2. Get Market Data
        market = MarketData(self.market_assets)
        hist_data, _ = market.get_data()
        
        # 3. Train Model and Forecast
        model = PredictiveModel(hist_data)
        model.train()
        forecasts = model.forecast()
        
        # 4. Optimize Portfolio
        optimizer = PortfolioOptimizer(client, forecasts)
        allocations = optimizer.optimize()
        
        # 5. Generate Output
        portfolio_return = sum(allocations[asset] * forecasts[asset]['expected_return'] for asset in allocations)
        portfolio_volatility = sum(allocations[asset] * forecasts[asset]['volatility'] for asset in allocations) # Simplified
        
        recommendations = {
            "client_profile": profile,
            "suggested_investments": {
                asset: {
                    "allocation_percentage": f"{allocations[asset]*100:.2f}%",
                    "expected_return": f"{forecasts[asset]['expected_return']*100:.2f}%",
                    "volatility": f"{forecasts[asset]['volatility']*100:.2f}%"
                } for asset in allocations
            },
            "risk_adjusted_projections": {
                "projected_annual_return": f"{portfolio_return*100:.2f}%",
                "projected_annual_volatility": f"{portfolio_volatility*100:.2f}%"
            },
            "alerts": self._generate_alerts(forecasts)
        }
        
        print("\n--- Recommendations Generated ---")
        return recommendations

    def _generate_alerts(self, forecasts):
        """Generates alerts based on market conditions."""
        alerts = []
        high_volatility_assets = [asset for asset, data in forecasts.items() if data['volatility'] > 0.4]
        if high_volatility_assets:
            alerts.append(f"High volatility detected in: {', '.join(high_volatility_assets)}. Consider reviewing exposure.")
        
        # Example of a trend shift alert
        # In a real scenario, this would compare recent performance to historical trends
        if random.choice([True, False]):
             alerts.append("Emerging trend detected in the Technology sector. Potential for upward momentum.")

        if not alerts:
            alerts.append("Market conditions appear stable. No immediate alerts.")
            
        return alerts

# --- Example Usage ---
if __name__ == '__main__':
    # 1. Define Client Input
    client_input = {
        "risk_tolerance": "medium",
        "investment_horizon": 15,
        "liquidity_needs": "low",
        "income": 120000,
        "portfolio_preferences": ["growth", "ethical"],
        "sector_preferences": ["tech", "healthcare"]
    }
    
    # 2. Define Market Universe
    # In a real scenario, this would be a much larger, dynamic list
    market_universe = [
        'AAPL', 'GOOGL', 'MSFT', # Tech
        'JNJ', 'PFE', # Healthcare
        'XOM', # Energy
        'VTI', # Total Stock Market ETF
        'BND'  # Total Bond Market ETF
    ]
    
    # 3. Instantiate and Run the Virtual CIO
    virtual_cio = VirtualCIO(client_info=client_input, market_assets=market_universe)
    final_recommendations = virtual_cio.generate_recommendations()
    
    # 4. Print the Output
    import json
    print("\nFinal Investment Recommendations:")
    print(json.dumps(final_recommendations, indent=2))


#!/usr/bin/env python3
"""
Predictive Investment Decision Algorithm (PIDA) - Virtual CIO
Single file implementation for capital allocation decisions
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any
import warnings
import pprint

warnings.filterwarnings('ignore')

# Suppress scientific notation
pd.set_option('display.float_format', lambda x: '%.4f' % x)

class VirtualCIO:
    """
    Virtual Chief Investment Officer for automated capital allocation
    """
    
    def __init__(self):
        """Initializes the VirtualCIO instance."""
        self.client_profiles = {}
        self.market_data = {}
        self.asset_universe = self._initialize_asset_universe()
        self.optimization_history = {}
        
    def _initialize_asset_universe(self) -> Dict:
        """Initialize available investment assets with base characteristics."""
        return {
            'SPY': {'type': 'equity', 'sector': 'broad_market', 'risk': 1.0, 'expected_return': 0.08},
            'QQQ': {'type': 'equity', 'sector': 'technology', 'risk': 1.2, 'expected_return': 0.10},
            'IWM': {'type': 'equity', 'sector': 'small_cap', 'risk': 1.3, 'expected_return': 0.09},
            'EFA': {'type': 'equity', 'sector': 'international', 'risk': 1.1, 'expected_return': 0.07},
            'EEM': {'type': 'equity', 'sector': 'emerging_markets', 'risk': 1.5, 'expected_return': 0.12},
            'AGG': {'type': 'bond', 'sector': 'us_bonds', 'risk': 0.3, 'expected_return': 0.03},
            'LQD': {'type': 'bond', 'sector': 'corporate_bonds', 'risk': 0.5, 'expected_return': 0.04},
            'TLT': {'type': 'bond', 'sector': 'treasury_bonds', 'risk': 0.8, 'expected_return': 0.05},
            'VNQ': {'type': 'reit', 'sector': 'real_estate', 'risk': 0.9, 'expected_return': 0.06},
            'GLD': {'type': 'commodity', 'sector': 'precious_metals', 'risk': 1.0, 'expected_return': 0.05}
        }
    
    def set_client_profile(self, client_id: str, profile_data: Dict) -> None:
        """Set or update a client's profile."""
        self.client_profiles[client_id] = {
            'profile': profile_data,
            'timestamp': datetime.now()
        }
    
    def simulate_market_data(self) -> Dict:
        """Simulate real-time market data for demonstration purposes."""
        np.random.seed(int(datetime.now().timestamp()))  # Use current time for dynamic results
        
        # Simulate current market conditions
        market_conditions = {
            'volatility_regime': np.random.choice(['low', 'normal', 'high'], p=[0.3, 0.5, 0.2]),
            'market_trend': np.random.choice(['bull', 'sideways', 'bear'], p=[0.4, 0.4, 0.2]),
            'economic_cycle': np.random.choice(['expansion', 'peak', 'contraction', 'trough'], 
                                             p=[0.35, 0.25, 0.25, 0.15])
        }
        
        # Simulate asset correlations based on market regime
        correlation_matrix = self._generate_correlation_matrix(market_conditions)
        
        # Simulate current asset metrics
        asset_metrics = {}
        for asset, info in self.asset_universe.items():
            # Adjust returns based on market conditions
            base_return = info['expected_return']
            market_impact = self._calculate_market_impact(market_conditions, info['sector'])
            current_return = base_return * (1 + market_impact)
            
            # Adjust risk based on volatility regime
            vol_multiplier = {'low': 0.7, 'normal': 1.0, 'high': 1.5}[market_conditions['volatility_regime']]
            current_risk = info['risk'] * vol_multiplier
            
            asset_metrics[asset] = {
                'expected_return': max(0.001, current_return),  # Ensure a minimum positive return
                'risk': current_risk,
                'sharpe_ratio': current_return / current_risk if current_risk > 0 else 0,
                'sector': info['sector']
            }
        
        return {
            'market_conditions': market_conditions,
            'correlation_matrix': correlation_matrix,
            'asset_metrics': asset_metrics
        }
    
    def _generate_correlation_matrix(self, market_conditions: Dict) -> pd.DataFrame:
        """Generate a correlation matrix based on market conditions."""
        assets = list(self.asset_universe.keys())
        n_assets = len(assets)
        
        # Base correlation matrix
        base_corr = np.full((n_assets, n_assets), 0.3)
        np.fill_diagonal(base_corr, 1.0)
        
        # Adjust correlations based on market regime
        corr_adjustment = 0.0
        if market_conditions['volatility_regime'] == 'high':
            corr_adjustment = 0.3  # Higher correlations during stress
        elif market_conditions['market_trend'] == 'bear':
            corr_adjustment = 0.2  # Higher correlations in bear markets
        
        adjusted_corr = base_corr + np.full((n_assets, n_assets), corr_adjustment)
        adjusted_corr = np.minimum(adjusted_corr, 0.9)  # Cap maximum correlation at 0.9
        np.fill_diagonal(adjusted_corr, 1.0) # Ensure diagonal is 1.0
        
        return pd.DataFrame(adjusted_corr, index=assets, columns=assets)
    
    def _calculate_market_impact(self, market_conditions: Dict, sector: str) -> float:
        """Calculate the market's impact on specific sectors."""
        impact = 0.0
        
        # Market trend impact
        trend_impact = {'bull': 0.1, 'sideways': 0.0, 'bear': -0.1}[market_conditions['market_trend']]
        impact += trend_impact
        
        # Economic cycle impact by sector
        cycle_impacts = {
            'expansion': {'technology': 0.15, 'small_cap': 0.12, 'real_estate': 0.08, 'emerging_markets': 0.10, 'broad_market': 0.08},
            'peak': {'technology': 0.05, 'small_cap': 0.02, 'real_estate': 0.03, 'emerging_markets': -0.02, 'broad_market': 0.01},
            'contraction': {'technology': -0.08, 'small_cap': -0.12, 'real_estate': -0.05, 'emerging_markets': -0.15, 'broad_market': -0.08, 'us_bonds': 0.05, 'treasury_bonds': 0.06},
            'trough': {'technology': -0.05, 'small_cap': -0.08, 'real_estate': -0.03, 'emerging_markets': -0.10, 'broad_market': -0.05, 'us_bonds': 0.08, 'treasury_bonds': 0.09}
        }
        
        cycle = market_conditions['economic_cycle']
        sector_impact = cycle_impacts[cycle].get(sector, 0.0)
        impact += sector_impact
        
        return impact
    
    def calculate_client_risk_profile(self, client_data: Dict) -> Dict:
        """Calculate a comprehensive client risk profile."""
        # Risk tolerance scoring (1-10 scale)
        risk_tolerance = client_data.get('risk_tolerance', 5)
        
        # Investment horizon scoring
        horizon = client_data.get('investment_horizon', 10)  # years
        horizon_score = min(10, max(1, horizon / 2))  # Scale 1-10
        
        # Liquidity needs scoring
        liquidity_needs = client_data.get('liquidity_needs', 'moderate')
        liquidity_scores = {'high': 2, 'moderate': 5, 'low': 8}
        liquidity_score = liquidity_scores.get(liquidity_needs, 5)
        
        # Income requirements and stability
        income = client_data.get('income', 100000)
        income_stability = client_data.get('income_stability', 'stable')
        stability_multiplier = {'volatile': 0.7, 'stable': 1.0, 'highly_stable': 1.2}[income_stability]
        
        # Calculate composite risk score (1-10)
        composite_risk = (risk_tolerance * 0.4 + horizon_score * 0.3 + liquidity_score * 0.3) * stability_multiplier
        
        # Adjust for constraints
        if liquidity_needs == 'high':
            composite_risk = min(composite_risk, 4)  # Cap risk for high liquidity needs
        
        return {
            'risk_tolerance': risk_tolerance,
            'horizon_score': horizon_score,
            'liquidity_score': liquidity_score,
            'composite_risk': min(10, max(1, composite_risk)),
            'income_requirements': income,
            'preferred_sectors': client_data.get('sector_preferences', []),
            'asset_preferences': client_data.get('portfolio_preferences', [])
        }
    
    def optimize_portfolio(self, client_profile: Dict, market_data: Dict) -> Dict:
        """Optimize portfolio allocation using a simplified mean-variance approach."""
        risk_profile = self.calculate_client_risk_profile(client_profile)
        
        eligible_assets = self._filter_assets_by_preferences(risk_profile['preferred_sectors'], risk_profile['asset_preferences'])
        
        asset_metrics = market_data['asset_metrics']
        selected_assets = [asset for asset in eligible_assets if asset in asset_metrics]

        if len(selected_assets) < 2:
            selected_assets = list(self.asset_universe.keys()) # Fallback
        
        returns = np.array([asset_metrics[a]['expected_return'] for a in selected_assets])
        risks = np.array([asset_metrics[a]['risk'] for a in selected_assets])
        corr_matrix = market_data['correlation_matrix'].loc[selected_assets, selected_assets]
        
        weights = self._mean_variance_optimization(returns, risks, corr_matrix.values, risk_profile['composite_risk'])
        weights = self._apply_constraints(weights, risk_profile, selected_assets)
        
        expected_risk = self._calculate_portfolio_risk(weights, risks, corr_matrix.values)
        expected_return = np.sum(weights * returns)

        return {
            'assets': selected_assets,
            'weights': weights,
            'expected_return': expected_return,
            'expected_risk': expected_risk,
            'sharpe_ratio': expected_return / expected_risk if expected_risk > 0 else 0
        }
    
    def _filter_assets_by_preferences(self, preferred_sectors: List[str], asset_preferences: List[str]) -> List[str]:
        """Filter assets based on client preferences."""
        if not preferred_sectors and not asset_preferences:
            return list(self.asset_universe.keys())
        
        eligible_assets = set()
        for asset, info in self.asset_universe.items():
            if info['sector'] in preferred_sectors or info['type'] in asset_preferences:
                eligible_assets.add(asset)
        
        return list(eligible_assets) if eligible_assets else list(self.asset_universe.keys())
    
    def _mean_variance_optimization(self, returns: np.array, risks: np.array, 
                                  correlation_matrix: np.array, risk_tolerance: float) -> np.array:
        """Simplified mean-variance optimization based on risk-adjusted returns."""
        n_assets = len(returns)
        if n_assets == 0:
            return np.array([])

        # Adjust returns based on risk, favoring higher risk-adjusted returns
        risk_adjusted_returns = returns / (risks + 1e-8)
        
        # Scale weights by risk-adjusted returns and client risk tolerance
        weights = risk_adjusted_returns * (risk_tolerance / 5.0) # Normalize around a mid-point risk tolerance
        
        # Normalize weights to sum to 1
        weights = np.maximum(weights, 0)  # No short positions
        weight_sum = np.sum(weights)
        if weight_sum > 0:
            weights /= weight_sum
        else:
            weights = np.ones(n_assets) / n_assets # Fallback to equal weights
        
        return weights
    
    def _calculate_portfolio_risk(self, weights: np.array, risks: np.array, 
                                correlation_matrix: np.array) -> float:
        """Calculate portfolio risk (standard deviation)."""
        if len(weights) == 0:
            return 0.0
        std_devs = np.diag(risks)
        covariance_matrix = std_devs @ correlation_matrix @ std_devs
        portfolio_variance = weights.T @ covariance_matrix @ weights
        return np.sqrt(portfolio_variance)
    
    def _apply_constraints(self, weights: np.array, risk_profile: Dict, assets: List[str]) -> np.array:
        """Apply various portfolio constraints."""
        if len(weights) == 0:
            return np.array([])
            
        # Individual position limits (no single position > 30%)
        weights = np.minimum(weights, 0.30)
        
        # Sector concentration limits
        sector_weights = {}
        for i, asset in enumerate(assets):
            sector = self.asset_universe[asset]['sector']
            sector_weights[sector] = sector_weights.get(sector, 0) + weights[i]
        
        for sector, weight in sector_weights.items():
            if weight > 0.40:  # 40% sector limit
                adjustment_factor = 0.40 / weight
                for i, asset in enumerate(assets):
                    if self.asset_universe[asset]['sector'] == sector:
                        weights[i] *= adjustment_factor
        
        # Liquidity constraint for clients with high liquidity needs
        if risk_profile['liquidity_score'] <= 3:
            for i, asset in enumerate(assets):
                sector = self.asset_universe[asset]['sector']
                if sector in ['emerging_markets', 'small_cap', 'real_estate']:
                    weights[i] *= 0.7  # Reduce exposure to less liquid assets by 30%
        
        # Re-normalize weights to sum to 1
        weight_sum = np.sum(weights)
        if weight_sum > 0:
            weights /= weight_sum
        else:
            weights = np.ones(len(weights)) / len(weights)
        
        return weights
    
    def generate_investment_recommendations(self, client_id: str, client_data: Dict) -> Dict:
        """Generate complete investment recommendations for a client."""
        self.set_client_profile(client_id, client_data)
        market_data = self.simulate_market_data()
        portfolio = self.optimize_portfolio(client_data, market_data)
        recommendations = self._format_recommendations(portfolio, client_data, market_data)
        
        self.optimization_history[client_id] = {
            'timestamp': datetime.now(),
            'portfolio': portfolio,
            'recommendations': recommendations
        }
        
        return recommendations
    
    def _format_recommendations(self, portfolio: Dict, client_data: Dict, market_data: Dict) -> Dict:
        """Format recommendations into a user-friendly dictionary."""
        risk_profile = self.calculate_client_risk_profile(client_data)
        
        asset_details = []
        total_investment = client_data.get('total_investment', 1000000) # Default to $1M
        
        for i, asset in enumerate(portfolio['assets']):
            weight = portfolio['weights'][i]
            asset_details.append({
                'asset': asset,
                'name': self._get_asset_name(asset),
                'allocation_percentage': round(weight * 100, 2),
                'allocation_amount': f"${weight * total_investment:,.0f}",
                'expected_return': f"{market_data['asset_metrics'][asset]['expected_return']*100:.2f}%",
                'risk_level': f"{market_data['asset_metrics'][asset]['risk']:.2f}",
                'sector': self.asset_universe[asset]['sector']
            })
            
        return {
            'summary': {
                'client_risk_profile': f"Risk Level {risk_profile['composite_risk']:.1f}/10",
                'portfolio_expected_return': f"{portfolio['expected_return']*100:.2f}%",
                'portfolio_expected_risk': f"{portfolio['expected_risk']*100:.2f}%",
                'portfolio_sharpe_ratio': f"{portfolio['sharpe_ratio']:.3f}",
                'market_conditions': market_data['market_conditions']
            },
            'asset_allocations': asset_details,
            'scenarios': self._generate_scenario_analysis(portfolio),
            'alerts': self._generate_alerts(client_data, market_data, portfolio),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
    
    def _get_asset_name(self, asset: str) -> str:
        """Get the full name for an asset ticker."""
        names = {
            'SPY': 'S&P 500 ETF', 'QQQ': 'NASDAQ 100 ETF', 'IWM': 'Russell 2000 ETF',
            'EFA': 'Developed Markets ETF', 'EEM': 'Emerging Markets ETF',
            'AGG': 'US Aggregate Bond ETF', 'LQD': 'Corporate Bond ETF',
            'TLT': '20+ Year Treasury ETF', 'VNQ': 'Real Estate ETF', 'GLD': 'Gold ETF'
        }
        return names.get(asset, asset)
    
    def _generate_scenario_analysis(self, portfolio: Dict) -> Dict:
        """Generate a simple scenario analysis for the portfolio."""
        base_return = portfolio['expected_return']
        base_risk = portfolio['expected_risk']
        
        return {
            'bull_market': {'description': 'Strong market growth (+15%)', 'projected_return': f"{(base_return + 0.08)*100:.2f}%"},
            'bear_market': {'description': 'Significant market downturn (-20%)', 'projected_return': f"{(base_return - 0.12)*100:.2f}%"},
            'high_volatility': {'description': 'Market volatility increases by 50%', 'projected_risk': f"{(base_risk * 1.5)*100:.2f}%"}
        }
    
    def _generate_alerts(self, client_data: Dict, market_data: Dict, portfolio: Dict) -> List[Dict]:
        """Generate alerts based on market conditions and portfolio composition."""
        alerts = []
        market_conditions = market_data['market_conditions']
        risk_profile = self.calculate_client_risk_profile(client_data)

        # Market condition alerts
        if market_conditions['volatility_regime'] == 'high':
            alerts.append({'type': 'Market', 'message': 'High volatility detected. Consider defensive positioning.'})
        if market_conditions['market_trend'] == 'bear':
            alerts.append({'type': 'Market', 'message': 'Bear market trend identified. Portfolio may see drawdowns.'})
        if market_conditions['economic_cycle'] == 'contraction':
            alerts.append({'type': 'Market', 'message': 'Economic contraction phase. Cyclical assets may underperform.'})

        # Portfolio-specific alerts
        if portfolio['expected_risk'] > (risk_profile['composite_risk'] / 10):
            alerts.append({'type': 'Portfolio', 'message': f"Portfolio risk ({portfolio['expected_risk']:.2f}) exceeds client's target risk level ({risk_profile['composite_risk'] / 10:.2f})."})
        
        # Concentration alerts
        for asset in portfolio['assets']:
            weight = portfolio['weights'][portfolio['assets'].index(asset)]
            if weight > 0.25: # Alert if any single asset is > 25%
                alerts.append({'type': 'Concentration', 'message': f"High concentration in {asset} ({weight*100:.1f}%). Review exposure."})

        return alerts

def main():
    """Main function to demonstrate the VirtualCIO class."""
    
    # Create an instance of the Virtual CIO
    cio = VirtualCIO()
    
    # Define a sample client profile
    client_1_data = {
        'risk_tolerance': 7,               # Scale 1-10
        'investment_horizon': 20,          # In years
        'liquidity_needs': 'low',          # high, moderate, low
        'income': 150000,                  # Annual income
        'income_stability': 'highly_stable', # volatile, stable, highly_stable
        'sector_preferences': ['technology', 'broad_market'], # e.g., 'technology', 'real_estate'
        'portfolio_preferences': ['equity'], # e.g., 'equity', 'bond'
        'total_investment': 500000
    }
    
    # Generate investment recommendations for the client
    print("--- Generating Recommendations for Client 1 (Growth Oriented) ---")
    recommendations = cio.generate_investment_recommendations('client_1', client_1_data)
    
    # Pretty print the results
    pprint.pprint(recommendations)
    
    # Define a second, more conservative client
    client_2_data = {
        'risk_tolerance': 3,
        'investment_horizon': 5,
        'liquidity_needs': 'moderate',
        'income': 80000,
        'income_stability': 'stable',
        'sector_preferences': ['us_bonds', 'corporate_bonds'],
        'portfolio_preferences': ['bond'],
        'total_investment': 1200000
    }
    
    print("\n--- Generating Recommendations for Client 2 (Conservative) ---")
    recommendations_2 = cio.generate_investment_recommendations('client_2', client_2_data)
    pprint.pprint(recommendations_2)


if __name__ == "__main__":
    main()

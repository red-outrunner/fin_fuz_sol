import React from 'react';

const KPICards = ({ stats }) => {
  const metrics = [
    { label: 'CAGR', value: `${(stats.cagr * 100).toFixed(2)}%`, desc: 'Compound Annual Growth Rate' },
    { label: 'Volatility', value: `${(stats.volatility * 100).toFixed(2)}%`, desc: 'Annualized Standard Deviation' },
    { label: 'Sharpe Ratio', value: stats.sharpe_ratio.toFixed(2), desc: 'Risk-Adjusted Return (Rf=2%)' },
    { label: 'Sortino Ratio', value: stats.sortino_ratio ? stats.sortino_ratio.toFixed(2) : 'N/A', desc: 'Downside Risk-Adjusted Return', isPositive: true },
    { label: 'Max Drawdown', value: `${(stats.max_drawdown * 100).toFixed(2)}%`, desc: 'Max Peak-to-Trough Loss', isNegative: true },
  ];

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-10">
      {metrics.map((metric, index) => (
        <div key={index} className="bg-white p-6 rounded-sm border border-beige shadow-sm hover:shadow-md transition-all duration-300 group">
          <h3 className="text-xs font-bold text-slate-500 uppercase tracking-widest mb-2 group-hover:text-gold transition-colors">{metric.label}</h3>
          <p className={`text-3xl font-serif font-bold mb-1 ${metric.isNegative ? 'text-error' : 'text-navy'}`}>
            {metric.value}
          </p>
          <p className="text-xs text-slate-400 font-medium">{metric.desc}</p>
        </div>
      ))}
    </div>
  );
};

export default KPICards;

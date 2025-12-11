
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
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-5 gap-6 mb-12">
      {metrics.map((metric, index) => (
        <div
          key={index}
          className="bg-white p-6 rounded-lg shadow-soft border border-beige-dark/50 hover:shadow-lg hover:-translate-y-1 transition-all duration-300 group"
        >
          <div className="flex flex-col h-full justify-between">
            <div>
              <h3 className="text-[10px] font-bold text-slate-400 uppercase tracking-widest mb-3 group-hover:text-gold transition-colors">{metric.label}</h3>
              <p className={`text-3xl font-serif font-bold tracking-tight mb-2 ${metric.isNegative ? 'text-error' : 'text-navy'}`}>
                {metric.value}
              </p>
            </div>
            <p className="text-xs text-slate-400 font-medium leading-relaxed border-t border-slate-100 pt-3 mt-2">{metric.desc}</p>
          </div>
        </div>
      ))}
    </div>
  );
};

export default KPICards;

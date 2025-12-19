import React from 'react';
import { useAuth } from '../context/AuthContext';

const KPICards = ({ stats }) => {
  const { user } = useAuth();
  if (!stats) return null;
  const isPro = user?.tier === 'pro' || user?.tier === 'institutional';

  const metrics = [
    { label: 'CAGR', value: stats.cagr !== null ? `${(stats.cagr * 100).toFixed(2)}%` : 'N/A', desc: 'Compound Annual Growth Rate', isPremium: false },
    { label: 'Volatility', value: stats.volatility !== null ? `${(stats.volatility * 100).toFixed(2)}%` : 'N/A', desc: 'Annualized Standard Deviation', isPremium: true },
    { label: 'Sharpe Ratio', value: stats.sharpe_ratio !== null ? stats.sharpe_ratio.toFixed(2) : 'N/A', desc: 'Risk-Adjusted Return (Rf=2%)', isPremium: true },
    { label: 'Sortino Ratio', value: stats.sortino_ratio !== null ? stats.sortino_ratio.toFixed(2) : 'N/A', desc: 'Downside Risk-Adjusted Return', isPositive: true, isPremium: true },
    { label: 'Max Drawdown', value: stats.max_drawdown !== null ? `${(stats.max_drawdown * 100).toFixed(2)}%` : 'N/A', desc: 'Max Peak-to-Trough Loss', isNegative: true, isPremium: true },
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
              <div className="flex justify-between items-start mb-3">
                <h3 className="text-[10px] font-bold text-slate-400 uppercase tracking-widest group-hover:text-gold transition-colors">{metric.label}</h3>
                {metric.isPremium && !isPro && (
                  <span className="text-[8px] bg-gold/10 text-gold font-bold px-1.5 py-0.5 rounded leading-none uppercase">Pro</span>
                )}
              </div>
              <p className={`text-3xl font-serif font-bold tracking-tight mb-2 ${metric.isNegative ? 'text-error' : 'text-navy'} ${metric.isPremium && !isPro ? 'blur-sm select-none' : ''}`}>
                {metric.isPremium && !isPro ? '00.00%' : metric.value}
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

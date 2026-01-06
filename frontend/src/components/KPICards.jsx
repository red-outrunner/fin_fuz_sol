import React from 'react';
import { useAuth } from '../context/AuthContext';
import { TrendingUp, Activity, ShieldCheck, ArrowDownCircle, Target } from 'lucide-react';

const KPICards = ({ stats }) => {
  const { user } = useAuth();
  if (!stats) return null;
  const isPro = user?.tier === 'pro' || user?.tier === 'institutional';

  const metrics = [
    { label: 'CAGR', value: stats.cagr !== null ? `${(stats.cagr * 100).toFixed(2)}%` : 'N/A', desc: 'Compound Annual Growth Rate', isPremium: false, icon: TrendingUp },
    { label: 'Volatility', value: stats.volatility !== null ? `${(stats.volatility * 100).toFixed(2)}%` : 'N/A', desc: 'Annualized Risk Level', isPremium: true, icon: Activity },
    { label: 'Sharpe Ratio', value: stats.sharpe_ratio !== null ? stats.sharpe_ratio.toFixed(2) : 'N/A', desc: 'Performance vs Risk', isPremium: true, icon: ShieldCheck },
    { label: 'Sortino Ratio', value: stats.sortino_ratio !== null ? stats.sortino_ratio.toFixed(2) : 'N/A', desc: 'Downside Risk Efficiency', isPositive: true, isPremium: true, icon: Target },
    { label: 'Max Drawdown', value: stats.max_drawdown !== null ? `${(stats.max_drawdown * 100).toFixed(2)}%` : 'N/A', desc: 'Peak to Trough Loss', isNegative: true, isPremium: true, icon: ArrowDownCircle },
  ];

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-5 gap-6 mb-16">
      {metrics.map((metric, index) => {
        const Icon = metric.icon;
        return (
          <div
            key={index}
            className="card-premium p-7 group"
          >
            <div className="flex flex-col h-full justify-between">
              <div>
                <div className="flex justify-between items-start mb-6">
                  <div className={`p-2 rounded-lg transition-colors duration-500 ${metric.isNegative ? 'bg-red-50 text-red-500' : 'bg-gold/5 text-gold'}`}>
                    <Icon size={18} strokeWidth={2} />
                  </div>
                  {metric.isPremium && !isPro && (
                    <span className="text-[9px] bg-navy text-gold font-bold px-2 py-0.5 rounded-full leading-none uppercase tracking-tighter">Pro</span>
                  )}
                </div>
                <h3 className="text-[10px] font-bold text-slate-400 uppercase tracking-[0.2em] mb-2 group-hover:text-navy transition-colors">{metric.label}</h3>
                <p className={`text-3xl font-serif font-bold tracking-tight mb-1 ${metric.isNegative ? 'text-red-600' : 'text-navy'} ${metric.isPremium && !isPro ? 'blur-md select-none opacity-40' : ''}`}>
                  {metric.isPremium && !isPro ? '00.00%' : metric.value}
                </p>
              </div>
              <p className="text-[11px] text-slate-500 font-medium leading-relaxed border-t border-navy/5 pt-4 mt-4">{metric.desc}</p>
            </div>
          </div>
        );
      })}
    </div>
  );
};

export default KPICards;

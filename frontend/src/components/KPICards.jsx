import React from 'react';
import { useAuth } from '../context/AuthContext';
import { TrendingUp, Activity, ShieldCheck, ArrowDownCircle, Target } from 'lucide-react';
import InfoTip from './InfoTip';

const KPICards = ({ stats }) => {
  const { user } = useAuth();
  if (!stats) return null;
  const isPro = user?.tier === 'pro' || user?.tier === 'institutional';

  const metrics = [
    { label: 'CAGR', value: stats.cagr !== null ? `${(stats.cagr * 100).toFixed(2)}%` : 'N/A', desc: 'Compound Annual Growth Rate', isPremium: false, icon: TrendingUp,
      help: 'Average growth per year, with compounding. 10% means your money grew about 10% every year. Higher is better.' },
    { label: 'Volatility', value: stats.volatility !== null ? `${(stats.volatility * 100).toFixed(2)}%` : 'N/A', desc: 'Annualized Risk Level', isPremium: true, icon: Activity,
      help: 'How much the price jumps around in a year. High = wild ride, low = calm ride. Not bad by itself — but you must be able to stomach it.' },
    { label: 'Sharpe Ratio', value: stats.sharpe_ratio !== null ? stats.sharpe_ratio.toFixed(2) : 'N/A', desc: 'Performance vs Risk', isPremium: true, icon: ShieldCheck,
      help: 'Reward earned per unit of risk taken. Above 1 = good deal. Below 0.5 = you take a lot of risk for little reward.' },
    { label: 'Sortino Ratio', value: stats.sortino_ratio !== null ? stats.sortino_ratio.toFixed(2) : 'N/A', desc: 'Downside Risk Efficiency', isPositive: true, isPremium: true, icon: Target,
      help: 'Like Sharpe, but it only punishes downward swings — the ones that actually hurt. Above 1.5 is strong.' },
    { label: 'Max Drawdown', value: stats.max_drawdown !== null ? `${(stats.max_drawdown * 100).toFixed(2)}%` : 'N/A', desc: 'Peak to Trough Loss', isNegative: true, isPremium: true, icon: ArrowDownCircle,
      help: 'The worst fall from a high to a low in this period. -40% means at one point you would have lost 40% from the top. Ask: could I hold through that?' },
  ];

  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-5 gap-4 md:gap-6 mb-16">
      {metrics.map((metric, index) => {
        const Icon = metric.icon;
        return (
          <div
            key={index}
            className="card-premium p-6 group relative overflow-hidden"
          >
            {/* Subtle Texture Overlay could go here */}
            <div className="flex flex-col h-full justify-between relative z-10">
              <div>
                <div className="flex justify-between items-start mb-6">
                  <div className={`transition-colors duration-500 ${metric.isNegative ? 'text-red-800' : 'text-gold'}`}>
                    <Icon size={16} strokeWidth={1.5} />
                  </div>
                  {metric.isPremium && !isPro && (
                    <span className="text-[8px] bg-navy text-gold font-bold px-2 py-0.5 rounded-none leading-none uppercase tracking-widest border border-gold/20">Pro</span>
                  )}
                </div>
                <h3 className="text-[9px] font-bold text-slate-500 uppercase tracking-[0.25em] mb-3 group-hover:text-navy transition-colors flex items-center gap-1.5">
                  {metric.label}
                  <InfoTip align="left" title={metric.label}>{metric.help}</InfoTip>
                </h3>
                <p className={`text-2xl md:text-3xl font-serif font-bold tracking-tight mb-1 ${metric.isNegative ? 'text-red-900' : 'text-navy'} ${metric.isPremium && !isPro ? 'blur-lg select-none opacity-20' : ''}`}>
                  {metric.isPremium && !isPro ? '00.00%' : metric.value}
                </p>
              </div>
              <p className="text-[10px] text-slate-400 font-medium leading-relaxed mt-6 italic opacity-0 group-hover:opacity-100 transition-opacity duration-500">{metric.desc}</p>
            </div>

            {/* Border Accent */}
            <div className="absolute bottom-0 left-0 w-full h-0.5 bg-gold/0 group-hover:bg-gold/20 transition-all duration-700" />
          </div>
        );
      })}
    </div>
  );
};

export default KPICards;

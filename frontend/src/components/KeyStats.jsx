import React from 'react';
import InfoTip from './InfoTip';

const MetricRow = ({ label, value, format = 'text', currency = 'USD' }) => {
    let displayValue = value;
    if (value === null || value === undefined) displayValue = '-';
    else if (format === 'percent') displayValue = `${(value * 100).toFixed(2)}%`;
    else if (format === 'currency') {
        displayValue = new Intl.NumberFormat(currency === 'ZAR' ? 'en-ZA' : 'en-US', {
            style: 'currency',
            currency,
            notation: 'compact',
        }).format(value);
    } else if (format === 'number') displayValue = new Intl.NumberFormat('en-US', { notation: 'compact' }).format(value);

    return (
        <div className="flex justify-between items-center py-1.5 border-b border-white/5 last:border-0 hover:bg-white/5 px-1 rounded transition-colors">
            <span className="text-[10px] text-slate-500 uppercase tracking-wide">{label}</span>
            <span className="text-xs font-mono text-slate-200 tabular-nums">{displayValue}</span>
        </div>
    );
};

const KeyStats = ({ stats, ticker = '' }) => {
    const isJSE = ticker.includes('.JO') || ticker.startsWith('^J');
    const currency = isJSE ? 'ZAR' : 'USD';

    if (!stats) {
        return (
            <div className="h-full flex items-center justify-center p-8 text-slate-500 text-sm">
                Key statistics unavailable — run analysis first.
            </div>
        );
    }

    return (
        <div className="h-full flex flex-col bg-transparent">
            <div className="px-4 py-2 border-b border-white/10 flex items-center gap-2 shrink-0 sticky top-0 bg-[#0f172a] z-10">
                <h3 className="text-gold text-[10px] font-bold uppercase tracking-widest">Key Statistics</h3>
                <InfoTip dark align="left" title="Key Statistics">
                    Valuation, financials, and analyst targets in one grid. Scroll for insight cards below.
                </InfoTip>
            </div>

            <div className="p-4 grid grid-cols-1 md:grid-cols-3 gap-6 overflow-y-auto custom-scrollbar flex-1">
                <div>
                    <h4 className="text-[10px] text-gold/70 font-bold uppercase mb-2 tracking-widest">Valuation</h4>
                    <div className="space-y-0">
                        <MetricRow label="Market Cap" value={stats.valuation.market_cap} format="currency" currency={currency} />
                        <MetricRow label="P/E (Trailing)" value={stats.valuation.pe_ratio} format="number" />
                        <MetricRow label="P/E (Forward)" value={stats.valuation.forward_pe} format="number" />
                        <MetricRow label="PEG Ratio" value={stats.valuation.peg_ratio} format="number" />
                        <MetricRow label="Price/Book" value={stats.valuation.price_to_book} format="number" />
                        <MetricRow label="Div Yield" value={stats.valuation.dividend_yield} format="percent" />
                    </div>
                </div>

                <div>
                    <h4 className="text-[10px] text-gold/70 font-bold uppercase mb-2 tracking-widest">Financials</h4>
                    <div className="space-y-0">
                        <MetricRow label="Revenue" value={stats.financials.revenue} format="currency" currency={currency} />
                        <MetricRow label="Rev Growth" value={stats.financials.revenue_growth} format="percent" />
                        <MetricRow label="Gross Margin" value={stats.financials.gross_margins} format="percent" />
                        <MetricRow label="Op Margin" value={stats.financials.operating_margins} format="percent" />
                        <MetricRow label="Profit Margin" value={stats.financials.profit_margins} format="percent" />
                        <MetricRow label="EBITDA" value={stats.financials.ebitda} format="currency" currency={currency} />
                    </div>
                </div>

                <div>
                    <h4 className="text-[10px] text-gold/70 font-bold uppercase mb-2 tracking-widest">Trading & Analysts</h4>
                    <div className="space-y-0">
                        <MetricRow label="Beta" value={stats.trading.beta} format="number" />
                        <MetricRow label="Short Ratio" value={stats.trading.short_ratio} format="number" />
                        <MetricRow label="Target Mean" value={stats.trading.target_mean} format="currency" currency={currency} />
                        <MetricRow label="Target High" value={stats.trading.target_high} format="currency" currency={currency} />
                        <MetricRow label="Rec Mean (1-5)" value={momentScore(stats.trading.recommendation_mean)} />
                    </div>
                </div>

                {stats.insight && (
                    <div className="col-span-full mt-2 pt-4 border-t border-white/10">
                        <h4 className="text-[10px] text-gold font-bold uppercase mb-3 tracking-widest">Insight</h4>
                        <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
                            <div className="bg-white/5 p-3 rounded-lg border border-white/10">
                                <span className="block text-[9px] text-slate-500 uppercase tracking-widest mb-1">Market Rank</span>
                                <span className="text-sm font-bold text-cream">{stats.insight.rank}</span>
                            </div>
                            <div className="bg-white/5 p-3 rounded-lg border border-white/10">
                                <span className="block text-[9px] text-slate-500 uppercase tracking-widest mb-1">
                                    {stats.insight.burgers_label || 'Burger Index'}
                                </span>
                                <span className="text-sm font-bold text-cream">{stats.insight.burgers}</span>
                            </div>
                            <div className="bg-white/5 p-3 rounded-lg border border-white/10">
                                <span className="block text-[9px] text-slate-500 uppercase tracking-widest mb-1">Market Mood</span>
                                <span className="text-sm font-bold text-cream">{stats.insight.mood}</span>
                            </div>
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
};

const momentScore = (val) => {
    if (!val) return '-';
    let color = 'text-slate-200';
    if (val <= 2) color = 'text-green-400';
    else if (val >= 4) color = 'text-red-400';
    return <span className={`${color} font-bold`}>{val.toFixed(2)}</span>;
};

export default KeyStats;

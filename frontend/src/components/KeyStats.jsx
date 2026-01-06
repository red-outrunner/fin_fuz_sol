import React from 'react';

const MetricRow = ({ label, value, format = 'text' }) => {
    let displayValue = value;
    if (value === null || value === undefined) displayValue = '-';
    else if (format === 'percent') displayValue = `${(value * 100).toFixed(2)}%`;
    else if (format === 'currency') displayValue = new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD', notation: 'compact' }).format(value);
    else if (format === 'number') displayValue = new Intl.NumberFormat('en-US', { notation: 'compact' }).format(value);

    return (
        <div className="flex justify-between items-center py-1 border-b border-gray-800 last:border-0 hover:bg-gray-800/50 px-1 rounded">
            <span className="text-xs text-gray-500">{label}</span>
            <span className="text-sm font-mono text-gray-200">{displayValue}</span>
        </div>
    );
};

const KeyStats = ({ stats }) => {
    if (!stats) return null;

    return (
        <div className="bg-gray-900 rounded-lg p-0 shadow-lg h-full border border-gray-800">
            <div className="bg-gray-800 px-4 py-2 rounded-t-lg border-b border-gray-700">
                <h3 className="text-orange-400 text-sm font-semibold uppercase tracking-wider">Key Statistics</h3>
            </div>

            <div className="p-4 grid grid-cols-1 md:grid-cols-3 gap-6">

                {/* Valuation */}
                <div>
                    <h4 className="text-xs text-blue-400 font-bold uppercase mb-2">Valuation</h4>
                    <div className="space-y-1">
                        <MetricRow label="Market Cap" value={stats.valuation.market_cap} format="currency" />
                        <MetricRow label="P/E (Trailing)" value={stats.valuation.pe_ratio} format="number" />
                        <MetricRow label="P/E (Forward)" value={stats.valuation.forward_pe} format="number" />
                        <MetricRow label="PEG Ratio" value={stats.valuation.peg_ratio} format="number" />
                        <MetricRow label="Price/Book" value={stats.valuation.price_to_book} format="number" />
                        <MetricRow label="Div Yield" value={stats.valuation.dividend_yield} format="percent" />
                    </div>
                </div>

                {/* Financials */}
                <div>
                    <h4 className="text-xs text-green-400 font-bold uppercase mb-2">Financials</h4>
                    <div className="space-y-1">
                        <MetricRow label="Revenue" value={stats.financials.revenue} format="currency" />
                        <MetricRow label="Rev Growth" value={stats.financials.revenue_growth} format="percent" />
                        <MetricRow label="Gross Margin" value={stats.financials.gross_margins} format="percent" />
                        <MetricRow label="Op Margin" value={stats.financials.operating_margins} format="percent" />
                        <MetricRow label="Profit Margin" value={stats.financials.profit_margins} format="percent" />
                        <MetricRow label="EBITDA" value={stats.financials.ebitda} format="currency" />
                    </div>
                </div>


                {/* Trading / Analyst */}
                <div>
                    <h4 className="text-xs text-purple-400 font-bold uppercase mb-2">Trading & Analysts</h4>
                    <div className="space-y-1">
                        <MetricRow label="Beta" value={stats.trading.beta} format="number" />
                        <MetricRow label="Short Ratio" value={stats.trading.short_ratio} format="number" />
                        <MetricRow label="Target Mean" value={stats.trading.target_mean} format="currency" />
                        <MetricRow label="Target High" value={stats.trading.target_high} format="currency" />
                        <MetricRow label="Rec Mean (1-5)" value={momentScore(stats.trading.recommendation_mean)} />
                    </div>
                </div>

                {/* Insight & Trivia */}
                {stats.insight && (
                    <div className="col-span-full mt-4 pt-4 border-t border-gray-800">
                        <h4 className="text-xs text-gold font-bold uppercase mb-2 flex items-center gap-2">
                            <span>✨ Insight & Trivia</span>
                        </h4>
                        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                            <div className="bg-gray-800/50 p-3 rounded border border-gray-700">
                                <span className="block text-[10px] text-gray-500 uppercase tracking-widest mb-1">Market Rank</span>
                                <span className="text-sm font-bold text-gray-200">{stats.insight.rank}</span>
                            </div>
                            <div className="bg-gray-800/50 p-3 rounded border border-gray-700">
                                <span className="block text-[10px] text-gray-500 uppercase tracking-widest mb-1">Burger Index</span>
                                <span className="text-sm font-bold text-gray-200">{stats.insight.burgers}</span>
                            </div>
                            <div className="bg-gray-800/50 p-3 rounded border border-gray-700">
                                <span className="block text-[10px] text-gray-500 uppercase tracking-widest mb-1">Market Mood</span>
                                <span className="text-sm font-bold text-gray-200">{stats.insight.mood}</span>
                            </div>
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
};

// Helper for Rec Mean color/text (1 is Strong Buy, 5 is Sell)
const momentScore = (val) => {
    if (!val) return '-';
    let color = 'text-gray-200';
    if (val <= 2) color = 'text-green-500';
    else if (val >= 4) color = 'text-red-500';
    return <span className={`${color} font-bold`}>{val.toFixed(2)}</span>;
}

export default KeyStats;

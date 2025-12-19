import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { API_BASE_URL } from '../api';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

const Comparison = ({ ticker, startYear, endDate }) => {
    const [benchmarkStats, setBenchmarkStats] = useState({});

    // ... (existing state)

    const fetchComparisonData = async (tickersToFetch) => {
        setLoading(true);
        try {
            // 1. Fetch Seasonal comparison data
            const comparisonRes = await axios.post(`${API_BASE_URL}/api/compare`, {
                tickers: tickersToFetch,
                start_year: startYear,
                end_date: endDate
            });
            setComparisonData(comparisonRes.data);

            // 2. Fetch Full Stats for Benchmarking (Parallel)
            const statsPromises = tickersToFetch.map(t =>
                axios.post(`${API_BASE_URL}/api/analyze`, {
                    ticker: t,
                    start_year: startYear,
                    end_date: endDate,
                    inflation_rate: 0
                }).then(res => ({ ticker: t, stats: res.data.stats })).catch(err => ({ ticker: t, stats: null }))
            );

            const statsResults = await Promise.all(statsPromises);
            const newStats = {};
            statsResults.forEach(item => {
                if (item.stats) newStats[item.ticker] = item.stats;
            });
            setBenchmarkStats(newStats);

        } catch (err) {
            console.error(err);
            setError('Failed to fetch comparison data.');
        } finally {
            setLoading(false);
        }
    };

    // ... (useEffect and other handlers remain similar)

    // ... (Chart render logic)

    // Helper to find best value for highlighting
    const getBest = (metric) => {
        let bestVal = -Infinity;
        let bestTicker = null;
        Object.entries(benchmarkStats).forEach(([t, s]) => {
            if (!s) return;
            const val = (metric === 'volatility' || metric === 'max_drawdown') ? -s[metric] : s[metric]; // Invert for "lower is better"
            if (val > bestVal) {
                bestVal = val;
                bestTicker = t;
            }
        });
        return bestTicker;
    };

    return (
        <div className="space-y-12 animate-in fade-in duration-500">
            {/* Header */}
            <div>
                <h2 className="text-3xl font-serif font-bold text-navy title-font">Peer Benchmarking</h2>
                <div className="h-1 w-20 bg-gold mt-2 mb-4"></div>
                <p className="text-slate-500 text-sm max-w-2xl">
                    Compare historical seasonality and risk-adjusted return metrics against peers.
                </p>
            </div>

            <div className="flex gap-4 mb-8">
                <input
                    type="text"
                    value={comparisonTicker}
                    onChange={(e) => setComparisonTicker(e.target.value)}
                    placeholder="Enter ticker to compare (e.g. ^GSPC)"
                    className="border border-beige-dark/30 rounded-lg p-3 flex-1 focus:outline-none focus:border-gold bg-white shadow-soft font-mono"
                />
                <button
                    onClick={handleAddComparison}
                    className="bg-navy text-cream px-6 py-3 rounded-lg hover:bg-navy-light transition-all font-bold tracking-wide shadow-md hover:shadow-lg"
                >
                    Add Comparison
                </button>
            </div>

            {loading && <div className="p-12 text-center text-gold font-bold animate-pulse">Running Benchmark Analysis...</div>}
            {error && <p className="text-error font-bold bg-red-50 p-4 border-l-4 border-error rounded shadow-sm">{error}</p>}

            {/* Performance Matrix (Premium Feature) */}
            {Object.keys(benchmarkStats).length > 0 && (
                <div className="bg-white p-8 rounded-lg shadow-xl border border-gold/20 relative overflow-hidden">
                    <div className="absolute top-0 right-0 bg-gold text-navy text-[10px] font-bold px-3 py-1 uppercase tracking-widest rounded-bl-lg">
                        Premium Analysis
                    </div>

                    <h3 className="text-xl font-serif font-bold text-navy mb-8 flex items-center gap-3">
                        Head-to-Head Performance
                    </h3>

                    <div className="overflow-x-auto">
                        <table className="w-full text-left border-collapse">
                            <thead>
                                <tr>
                                    <th className="p-4 border-b-2 border-beige-dark text-slate-400 font-sans text-xs uppercase tracking-widest">Ticker</th>
                                    <th className="p-4 border-b-2 border-beige-dark text-slate-400 font-sans text-xs uppercase tracking-widest text-center">CAGR</th>
                                    <th className="p-4 border-b-2 border-beige-dark text-slate-400 font-sans text-xs uppercase tracking-widest text-center">Volatility</th>
                                    <th className="p-4 border-b-2 border-beige-dark text-slate-400 font-sans text-xs uppercase tracking-widest text-center">Sharpe Ratio</th>
                                    <th className="p-4 border-b-2 border-beige-dark text-slate-400 font-sans text-xs uppercase tracking-widest text-center">Max Drawdown</th>
                                </tr>
                            </thead>
                            <tbody>
                                {activeComparisons.map(t => {
                                    const s = benchmarkStats[t];
                                    if (!s) return null;
                                    const isBestCAGR = getBest('cagr') === t;
                                    const isBestVol = getBest('volatility') === t;
                                    const isBestSharpe = getBest('sharpe_ratio') === t;
                                    const isBestDD = getBest('max_drawdown') === t;

                                    return (
                                        <tr key={t} className="hover:bg-cream transition-colors border-b border-beige-light last:border-0">
                                            <td className="p-4 font-bold font-serif text-lg text-navy">{t}</td>

                                            <td className={`p-4 text-center ${isBestCAGR ? 'text-green-700 font-bold bg-green-50/50' : 'text-slate-600'}`}>
                                                {(s.cagr * 100).toFixed(2)}%
                                                {isBestCAGR && <span className="ml-2 text-xs text-green-600">★</span>}
                                            </td>

                                            <td className={`p-4 text-center ${isBestVol ? 'text-green-700 font-bold bg-green-50/50' : 'text-slate-600'}`}>
                                                {(s.volatility * 100).toFixed(2)}%
                                                {isBestVol && <span className="ml-2 text-xs text-green-600">★</span>}
                                            </td>

                                            <td className={`p-4 text-center ${isBestSharpe ? 'text-gold font-bold bg-amber-50/50' : 'text-slate-600'}`}>
                                                {s.sharpe_ratio.toFixed(2)}
                                                {isBestSharpe && <span className="ml-2 text-xs text-gold">★</span>}
                                            </td>

                                            <td className={`p-4 text-center ${isBestDD ? 'text-green-700 font-bold bg-green-50/50' : 'text-red-500'}`}>
                                                {(s.max_drawdown * 100).toFixed(2)}%
                                                {isBestDD && <span className="ml-2 text-xs text-green-600">★</span>}
                                            </td>
                                        </tr>
                                    );
                                })}
                            </tbody>
                        </table>
                    </div>
                </div>
            )}

            {/* Seasonality Chart */}
            <div className="h-96 w-full bg-white p-6 rounded-lg shadow-soft border border-beige-dark/20 mt-12">
                <h3 className="text-lg font-serif font-bold text-navy mb-6">Seasonal Returns (Monthly Avg)</h3>
                <ResponsiveContainer width="100%" height="90%">
                    <LineChart data={chartData}>
                        <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#E2E8F0" />
                        <XAxis dataKey="name" stroke="#94A3B8" tick={{ fill: '#64748B', fontSize: 12 }} />
                        <YAxis stroke="#94A3B8" tick={{ fill: '#64748B', fontSize: 12 }} tickFormatter={(val) => `${val}%`} />
                        <Tooltip
                            contentStyle={{ backgroundColor: '#FDFCF8', borderColor: '#E2E8F0', boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)' }}
                            itemStyle={{ fontSize: 12 }}
                        />
                        <Legend wrapperStyle={{ paddingTop: 20 }} />
                        {activeComparisons.map((t, index) => (
                            <Line
                                key={t}
                                type="monotone"
                                dataKey={t}
                                stroke={colors[index % colors.length]}
                                activeDot={{ r: 6, fill: colors[index % colors.length], stroke: '#fff', strokeWidth: 2 }}
                                strokeWidth={2}
                                dot={false}
                            />
                        ))}
                    </LineChart>
                </ResponsiveContainer>
            </div>

            {/* Correlation Matrix Section */}
            {correlationData && (
                <div className="mt-12 bg-white p-8 rounded-lg shadow-soft border border-beige-dark/20">
                    <h3 className="text-xl font-serif font-bold text-navy mb-6 flex items-center gap-3">
                        Correlation Matrix
                    </h3>
                    {/* ... (Correlation Matrix Rendering - reusing logic but improved styling if needed) ... */
                        /* For brevity, I'll keep the logic mostly same but ensure it uses the new design tokens */
                    }
                    <div className="overflow-x-auto">
                        {/* ... matrix grid ... */}
                        <div className="inline-block min-w-full">
                            <div className="grid border border-beige-light" style={{
                                gridTemplateColumns: `auto repeat(${correlationData.tickers.length}, minmax(100px, 1fr))`
                            }}>
                                {/* Header Row */}
                                <div className="p-3 bg-cream border-b border-r border-beige-light"></div>
                                {correlationData.tickers.map(t => (
                                    <div key={t} className="p-3 font-bold text-center text-xs text-navy border-b border-r border-beige-light bg-cream uppercase tracking-wider">
                                        {t}
                                    </div>
                                ))}

                                {/* Rows */}
                                {correlationData.tickers.map(rowTicker => (
                                    <React.Fragment key={rowTicker}>
                                        <div className="p-3 font-bold text-left text-xs text-navy border-b border-r border-beige-light bg-cream uppercase tracking-wider flex items-center">
                                            {rowTicker}
                                        </div>
                                        {correlationData.tickers.map(colTicker => {
                                            const cell = correlationData.matrix.find(
                                                item => item.x === colTicker && item.y === rowTicker
                                            );
                                            const val = cell ? cell.value : 0;

                                            // Improved Color Logic for Matrix
                                            let bg = 'bg-white';
                                            let text = 'text-slate-400';
                                            if (val > 0.8) { bg = 'bg-navy'; text = 'text-white'; }
                                            else if (val > 0.5) { bg = 'bg-navy/60'; text = 'text-white'; }
                                            else if (val > 0.2) { bg = 'bg-navy/30'; text = 'text-navy'; }
                                            else if (val < -0.2) { bg = 'bg-red-500/30'; text = 'text-red-800'; }

                                            return (
                                                <div key={`${rowTicker}-${colTicker}`} className={`p-4 text-center ${bg} ${text} text-sm border-b border-r border-beige-light transition-colors duration-300 font-mono`}>
                                                    {val.toFixed(2)}
                                                </div>
                                            );
                                        })}
                                    </React.Fragment>
                                ))}
                            </div>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
};

export default Comparison;

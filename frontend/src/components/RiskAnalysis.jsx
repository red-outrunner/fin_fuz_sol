import React, { useState } from 'react';
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, BarChart, Bar, Cell } from 'recharts';

const RiskAnalysis = ({ stats, data }) => {
    if (!stats || !data) return null;

    // Prepare Drawdown Data
    const drawdownData = stats.drawdown_series || [];

    // Helper for Gauge (Semi-circle using CSS/SVG for cleaner look than heavy libraries)
    const Gauge = ({ value, label, min, max, optimal, inverse = false }) => {
        // Normalize value to 0-1 range
        const normalized = Math.min(Math.max((value - min) / (max - min), 0), 1);

        // Color Logic
        let color = "text-slate-400";
        let barColor = "bg-slate-300";

        if (inverse) {
            // Lower is better (e.g. Drawdown)
            if (value < optimal) { color = "text-green-600"; barColor = "bg-green-600"; }
            else if (value < optimal * 2) { color = "text-yellow-600"; barColor = "bg-yellow-500"; }
            else { color = "text-red-500"; barColor = "bg-red-500"; }
        } else {
            // Higher is better (e.g. Sharpe)
            if (value > optimal) { color = "text-green-600"; barColor = "bg-green-600"; }
            else if (value > optimal / 2) { color = "text-yellow-600"; barColor = "bg-yellow-500"; }
            else { color = "text-red-500"; barColor = "bg-red-500"; }
        }

        return (
            <div className="bg-white p-6 rounded-lg shadow-soft border border-beige-dark/20 flex flex-col items-center justify-center relative overflow-hidden group hover:shadow-md transition-shadow">
                <div className={`absolute top-0 left-0 w-1 h-full ${barColor} opacity-0 group-hover:opacity-100 transition-opacity`}></div>
                <span className="text-slate-500 text-xs font-bold uppercase tracking-widest mb-3">{label}</span>

                <div className="relative flex items-end mb-2">
                    <span className={`text-5xl font-serif font-bold ${color}`}>{value.toFixed(2)}</span>
                </div>

                {/* Linear Meter */}
                <div className="w-full h-1.5 bg-slate-100 rounded-full mt-2 relative overflow-hidden">
                    <div
                        className={`h-full absolute top-0 left-0 rounded-full ${barColor} transition-all duration-1000 ease-out`}
                        style={{ width: `${normalized * 100}%` }}
                    ></div>
                </div>

                <div className="flex justify-between w-full text-[10px] text-slate-400 mt-2 font-mono">
                    <span>{min}</span>
                    <span className="text-navy/40">Target: {optimal}</span>
                    <span>{max}</span>
                </div>
            </div>
        );
    };

    return (
        <div className="space-y-8 animate-in fade-in duration-500">
            <div className="border-b border-navy/5 pb-6">
                <h2 className="text-3xl font-serif font-bold text-navy title-font">Risk Analysis</h2>
                <div className="h-1 w-20 bg-gold mt-2 mb-4"></div>
                <p className="text-slate-500 text-sm max-w-2xl">
                    Evaluation of portfolio stability using annualized risk metrics.
                    <span className="italic text-navy/60 ml-2">Sharpe &gt; 1.0 is considered good.</span>
                </p>
            </div>

            {/* Key Risk Cards */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                <Gauge value={stats.sharpe_ratio || 0} label="Sharpe Ratio" min={-1} max={3} optimal={1.0} />
                <Gauge value={stats.sortino_ratio || 0} label="Sortino Ratio" min={-1} max={5} optimal={1.5} />

                <div className="bg-white p-6 rounded-lg shadow-soft border border-beige-dark/20 flex flex-col items-center justify-center hover:shadow-md transition-shadow">
                    <span className="text-slate-500 text-xs font-bold uppercase tracking-widest mb-3">Annual Volatility</span>
                    <span className="text-5xl font-serif font-bold text-navy">{(stats.volatility * 100).toFixed(1)}<span className="text-lg text-slate-400 ml-1">%</span></span>
                    <span className="text-xs text-slate-400 mt-4 bg-slate-50 px-2 py-1 rounded">Standard Deviation</span>
                </div>

                <div className="bg-white p-6 rounded-lg shadow-soft border border-beige-dark/20 flex flex-col items-center justify-center hover:shadow-md transition-shadow">
                    <span className="text-slate-500 text-xs font-bold uppercase tracking-widest mb-3">Max Drawdown</span>
                    <span className="text-5xl font-serif font-bold text-red-500">{(stats.max_drawdown * 100).toFixed(1)}<span className="text-lg text-red-300 ml-1">%</span></span>
                    <span className="text-xs text-slate-400 mt-4 bg-red-50 text-red-400 px-2 py-1 rounded">Peak-to-Valley</span>
                </div>
            </div>

            {/* Drawdown Chart */}
            <div className="bg-white p-6 rounded-lg shadow-soft border border-beige-dark/20 h-96">
                <h3 className="text-lg font-serif font-bold text-navy mb-6">Historical Drawdown</h3>
                <ResponsiveContainer width="100%" height="85%">
                    <AreaChart data={drawdownData}>
                        <defs>
                            <linearGradient id="colorDrawdown" x1="0" y1="0" x2="0" y2="1">
                                <stop offset="5%" stopColor="#ef4444" stopOpacity={0.8} />
                                <stop offset="95%" stopColor="#ef4444" stopOpacity={0} />
                            </linearGradient>
                        </defs>
                        <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#E2E8F0" />
                        <XAxis dataKey="date" stroke="#94A3B8" tick={{ fontSize: 12 }} tickFormatter={(val) => val.substring(0, 4)} />
                        <YAxis stroke="#94A3B8" tick={{ fontSize: 12 }} tickFormatter={(val) => `${(val * 100).toFixed(0)}%`} />
                        <Tooltip
                            contentStyle={{ backgroundColor: '#FDFCF8', borderColor: '#E2E8F0' }}
                            formatter={(value) => [`${(value * 100).toFixed(2)}%`, 'Drawdown']}
                        />
                        <Area type="monotone" dataKey="value" stroke="#ef4444" fillOpacity={1} fill="url(#colorDrawdown)" />
                    </AreaChart>
                </ResponsiveContainer>
            </div>
        </div>
    );
};

export default RiskAnalysis;

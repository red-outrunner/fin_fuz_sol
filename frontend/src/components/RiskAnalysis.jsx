import React, { useState } from 'react';
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, BarChart, Bar, Cell } from 'recharts';

const RiskAnalysis = ({ stats, data }) => {
    if (!stats || !data) return null;

    // Prepare Drawdown Data
    const drawdownData = stats.drawdown_series || [];

    // Helper for Gauge (Simple textual representation for now, or CSS gauge)
    const Gauge = ({ value, label, min, max, optimal }) => {
        // Simple color logic
        let color = "text-slate-400";
        if (value > optimal) color = "text-green-500";
        else if (value > optimal / 2) color = "text-yellow-500";
        else color = "text-red-500";

        return (
            <div className="bg-white p-6 rounded-lg shadow-soft border border-beige-dark/20 flex flex-col items-center justify-center">
                <span className="text-slate-500 text-sm font-bold uppercase tracking-wider mb-2">{label}</span>
                <span className={`text-4xl font-serif font-bold ${color}`}>{value.toFixed(2)}</span>
                <div className="w-full bg-slate-100 h-2 mt-4 rounded-full overflow-hidden">
                    <div
                        className={`h-full ${color.replace('text', 'bg')}`}
                        style={{ width: `${Math.min(100, Math.max(0, ((value - min) / (max - min)) * 100))}%` }}
                    ></div>
                </div>
                <div className="flex justify-between w-full text-xs text-slate-400 mt-1">
                    <span>{min}</span>
                    <span>{max}</span>
                </div>
            </div>
        );
    };

    return (
        <div className="space-y-8 animate-in fade-in duration-500">
            <div className="border-b border-navy/5 pb-6">
                <h2 className="text-2xl font-serif font-bold text-navy border-l-4 border-gold pl-4 title-font">Risk Analysis</h2>
                <p className="text-slate-500 text-sm pl-5 mt-2">
                    Advanced risk metrics and drawdown analysis to evaluate portfolio stability.
                </p>
            </div>

            {/* Key Risk Cards */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                <Gauge value={stats.sharpe_ratio || 0} label="Sharpe Ratio" min={-1} max={3} optimal={1} />
                <Gauge value={stats.sortino_ratio || 0} label="Sortino Ratio" min={-1} max={4} optimal={1.5} />

                <div className="bg-white p-6 rounded-lg shadow-soft border border-beige-dark/20 flex flex-col items-center justify-center">
                    <span className="text-slate-500 text-sm font-bold uppercase tracking-wider mb-2">Annual Volatility</span>
                    <span className="text-4xl font-serif font-bold text-navy">{(stats.volatility * 100).toFixed(2)}%</span>
                    <span className="text-xs text-slate-400 mt-2">Std Dev of Returns</span>
                </div>

                <div className="bg-white p-6 rounded-lg shadow-soft border border-beige-dark/20 flex flex-col items-center justify-center">
                    <span className="text-slate-500 text-sm font-bold uppercase tracking-wider mb-2">Max Drawdown</span>
                    <span className="text-4xl font-serif font-bold text-red-500">{(stats.max_drawdown * 100).toFixed(2)}%</span>
                    <span className="text-xs text-slate-400 mt-2">Worst Peak-to-Valley</span>
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

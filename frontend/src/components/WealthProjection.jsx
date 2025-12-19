import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { API_BASE_URL } from '../api';
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine } from 'recharts';

const WealthProjection = ({ ticker, startYear, endDate }) => {
    const [projectionData, setProjectionData] = useState([]);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);

    useEffect(() => {
        const fetchProjection = async () => {
            setLoading(true);
            try {
                const response = await axios.post(`${API_BASE_URL}/api/projection`, {
                    ticker,
                    start_year: startYear,
                    end_date: endDate
                });
                setProjectionData(response.data);
            } catch (err) {
                console.error("Projection error:", err);
                setError("Failed to generate wealth projection.");
            } finally {
                setLoading(false);
            }
        };

        if (ticker) fetchProjection();
    }, [ticker, startYear, endDate]);

    if (loading) return <div className="p-12 text-center text-gold font-bold animate-pulse">Running Monte Carlo Simulation (1000 iterations)...</div>;
    if (error) return <div className="p-12 text-center text-red-500">{error}</div>;
    if (!projectionData || projectionData.length === 0) return null;

    // Calculate final values for summary
    const finalPoint = projectionData[projectionData.length - 1];
    const initialVal = 10000;
    const upside = ((finalPoint.p90 - initialVal) / initialVal) * 100;
    const base = ((finalPoint.p50 - initialVal) / initialVal) * 100;
    const downside = ((finalPoint.p10 - initialVal) / initialVal) * 100;

    return (
        <div className="space-y-12 animate-in fade-in duration-500">
            <div>
                <h2 className="text-3xl font-serif font-bold text-navy title-font">Wealth Projection</h2>
                <div className="h-1 w-20 bg-gold mt-2 mb-4"></div>
                <p className="text-slate-500 text-sm max-w-2xl">
                    Probabilistic forecasting using Monte Carlo simulations (1,000 runs) based on historical volatility and drift.
                    Projections assume a hypothetical $10,000 investment.
                </p>
            </div>

            <div className="bg-white p-8 rounded-lg shadow-xl border border-beige-dark/20 relative overflow-hidden">
                <div className="absolute top-0 right-0 bg-gold text-navy text-[10px] font-bold px-3 py-1 uppercase tracking-widest rounded-bl-lg">
                    Premium Analysis
                </div>

                <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8 text-center">
                    <div className="p-4 bg-slate-50/50 rounded border border-slate-100">
                        <span className="text-xs text-slate-400 uppercase tracking-widest block mb-1">Conservative (10th %ile)</span>
                        <span className="text-2xl font-bold font-serif text-slate-600">
                            {new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD', maximumFractionDigits: 0 }).format(finalPoint.p10)}
                        </span>
                        <span className={`block text-xs font-bold mt-1 ${downside >= 0 ? 'text-green-600' : 'text-red-500'}`}>
                            {downside >= 0 ? '+' : ''}{downside.toFixed(0)}%
                        </span>
                    </div>
                    <div className="p-4 bg-navy/5 rounded border border-navy/10 transform scale-105 shadow-sm">
                        <span className="text-xs text-navy uppercase tracking-widest block mb-1 font-bold">Base Case (Median)</span>
                        <span className="text-3xl font-bold font-serif text-navy">
                            {new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD', maximumFractionDigits: 0 }).format(finalPoint.p50)}
                        </span>
                        <span className={`block text-xs font-bold mt-1 ${base >= 0 ? 'text-green-600' : 'text-red-500'}`}>
                            {base >= 0 ? '+' : ''}{base.toFixed(0)}%
                        </span>
                    </div>
                    <div className="p-4 bg-green-50/50 rounded border border-green-100">
                        <span className="text-xs text-green-700 uppercase tracking-widest block mb-1">Bull Case (90th %ile)</span>
                        <span className="text-2xl font-bold font-serif text-green-700">
                            {new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD', maximumFractionDigits: 0 }).format(finalPoint.p90)}
                        </span>
                        <span className={`block text-xs font-bold mt-1 ${upside >= 0 ? 'text-green-600' : 'text-red-500'}`}>
                            {upside >= 0 ? '+' : ''}{upside.toFixed(0)}%
                        </span>
                    </div>
                </div>

                <div className="h-96 w-full">
                    <ResponsiveContainer width="100%" height="100%">
                        <AreaChart data={projectionData}>
                            <defs>
                                <linearGradient id="colorP90" x1="0" y1="0" x2="0" y2="1">
                                    <stop offset="5%" stopColor="#8C735A" stopOpacity={0.3} />
                                    <stop offset="95%" stopColor="#8C735A" stopOpacity={0} />
                                </linearGradient>
                                <linearGradient id="colorP50" x1="0" y1="0" x2="0" y2="1">
                                    <stop offset="5%" stopColor="#1A2433" stopOpacity={0.5} />
                                    <stop offset="95%" stopColor="#1A2433" stopOpacity={0} />
                                </linearGradient>
                                <linearGradient id="colorP10" x1="0" y1="0" x2="0" y2="1">
                                    <stop offset="5%" stopColor="#94A3B8" stopOpacity={0.2} />
                                    <stop offset="95%" stopColor="#94A3B8" stopOpacity={0} />
                                </linearGradient>
                            </defs>
                            <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#E2E8F0" />
                            <XAxis
                                dataKey="date"
                                stroke="#94A3B8"
                                tick={{ fontSize: 12 }}
                                tickFormatter={(val) => val.substring(0, 4)}
                                minTickGap={30}
                            />
                            <YAxis
                                stroke="#94A3B8"
                                tick={{ fontSize: 12 }}
                                tickFormatter={(val) => `$${val / 1000}k`}
                            />
                            <Tooltip
                                contentStyle={{ backgroundColor: '#FDFCF8', borderColor: '#E2E8F0' }}
                                formatter={(value, name) => [
                                    new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD', maximumFractionDigits: 0 }).format(value),
                                    name === 'p90' ? '90th Percentile' : name === 'p50' ? 'Median Outcome' : '10th Percentile'
                                ]}
                                labelStyle={{ color: '#1A2433', fontWeight: 'bold' }}
                            />
                            <Area
                                type="monotone"
                                dataKey="p90"
                                stroke="#8C735A"
                                strokeWidth={1}
                                strokeDasharray="5 5"
                                fillOpacity={1}
                                fill="url(#colorP90)"
                                name="p90"
                            />
                            <Area
                                type="monotone"
                                dataKey="p50"
                                stroke="#1A2433"
                                strokeWidth={3}
                                fillOpacity={1}
                                fill="url(#colorP50)"
                                name="p50"
                            />
                            <Area
                                type="monotone"
                                dataKey="p10"
                                stroke="#94A3B8"
                                strokeWidth={1}
                                strokeDasharray="5 5"
                                fillOpacity={1}
                                fill="url(#colorP10)"
                                name="p10"
                            />
                        </AreaChart>
                    </ResponsiveContainer>
                </div>

                <p className="text-[10px] text-slate-400 mt-4 text-center italic">
                    *Projections are based on mathematical models and are not guarantees of future performance.
                </p>
            </div>
        </div>
    );
};

export default WealthProjection;

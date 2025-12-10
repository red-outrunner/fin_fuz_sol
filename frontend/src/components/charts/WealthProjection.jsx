
import React, { useEffect, useState } from 'react';
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Line, ComposedChart } from 'recharts';
import axios from 'axios';

const WealthProjection = ({ ticker, startYear, endDate }) => {
    const [data, setData] = useState(null);
    const [loading, setLoading] = useState(false);

    useEffect(() => {
        const fetchProjection = async () => {
            setLoading(true);
            try {
                const response = await axios.post('http://localhost:8000/api/projection', {
                    ticker,
                    start_year: startYear,
                    end_date: endDate
                });
                setData(response.data);
            } catch (err) {
                console.error("Projection error", err);
            } finally {
                setLoading(false);
            }
        };
        fetchProjection();
    }, [ticker, startYear, endDate]);

    if (loading) return <div className="h-80 bg-white border border-beige rounded-sm flex items-center justify-center text-gold animate-pulse">Running Monte Carlo Simulation...</div>;
    if (!data) return null;

    return (
        <div className="bg-white p-6 rounded-sm border border-beige shadow-sm">
            <h3 className="text-xl font-serif font-bold text-navy mb-2">Future Wealth Projection (10 Years)</h3>
            <p className="text-sm text-slate-500 mb-6 font-sans">Monte Carlo simulation (1000 runs) showing potential future paths for a R10,000 investment.</p>

            <div className="h-80 w-full">
                <ResponsiveContainer width="100%" height="100%">
                    <ComposedChart data={data}>
                        <defs>
                            <linearGradient id="colorCone" x1="0" y1="0" x2="0" y2="1">
                                <stop offset="5%" stopColor="#C5A059" stopOpacity={0.2} />
                                <stop offset="95%" stopColor="#C5A059" stopOpacity={0} />
                            </linearGradient>
                        </defs>
                        <CartesianGrid strokeDasharray="3 3" stroke="#F0EBE0" vertical={false} />
                        <XAxis
                            dataKey="date"
                            stroke="#8C735A"
                            tickFormatter={(str) => str.substring(0, 4)}
                            minTickGap={50}
                        />
                        <YAxis
                            stroke="#8C735A"
                            tickFormatter={(val) => `R${(val / 1000).toFixed(0)}k`}
                        />
                        <Tooltip
                            contentStyle={{ backgroundColor: '#F9F7F2', borderColor: '#C5A059' }}
                            formatter={(value, name) => [
                                `R${value.toLocaleString(undefined, { maximumFractionDigits: 0 })}`,
                                name === 'p90' ? 'Optimistic (90th)' : name === 'p50' ? 'Median (50th)' : 'Conservative (10th)'
                            ]}
                        />

                        {/* We use Lines for the percentiles */}
                        <Line type="monotone" dataKey="p90" stroke="#4A7C59" strokeWidth={2} dot={false} strokeDasharray="5 5" name="p90" />
                        <Line type="monotone" dataKey="p50" stroke="#C5A059" strokeWidth={3} dot={false} name="p50" />
                        <Line type="monotone" dataKey="p10" stroke="#8C4A4A" strokeWidth={2} dot={false} strokeDasharray="5 5" name="p10" />
                    </ComposedChart>
                </ResponsiveContainer>
            </div>

            <div className="flex justify-center gap-6 mt-4 text-xs font-bold uppercase tracking-widest">
                <div className="flex items-center gap-2"><div className="w-3 h-1 bg-[#4A7C59] border-b border-dashed"></div> Optimistic</div>
                <div className="flex items-center gap-2"><div className="w-3 h-1 bg-gold"></div> Median</div>
                <div className="flex items-center gap-2"><div className="w-3 h-1 bg-[#8C4A4A] border-b border-dashed"></div> Conservative</div>
            </div>
        </div>
    );
};

export default WealthProjection;

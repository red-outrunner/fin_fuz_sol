
import React from 'react';
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

const WealthProjection = ({ projectionData }) => {
    if (!projectionData) return null;

    return (
        <div className="bg-white p-6 rounded-sm border border-beige shadow-sm">
            <h3 className="text-xl font-serif font-bold text-navy mb-2">10-Year Wealth Projection</h3>
            <p className="text-sm text-slate-500 mb-6 font-sans">Monte Carlo output: 10th, 50th, and 90th percentile outcomes for R10k invested today.</p>

            <div className="h-80 w-full">
                <ResponsiveContainer width="100%" height="100%">
                    <AreaChart data={projectionData}>
                        <defs>
                            <linearGradient id="colorP90" x1="0" y1="0" x2="0" y2="1">
                                <stop offset="5%" stopColor="#4A7C59" stopOpacity={0.1} />
                                <stop offset="95%" stopColor="#4A7C59" stopOpacity={0} />
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
                        {/* 90th Percentile Range (Top) */}
                        <Area
                            type="monotone"
                            dataKey="p90"
                            stroke="transparent"
                            fill="url(#colorP90)"
                        />
                        {/* Median Line */}
                        <Area
                            type="monotone"
                            dataKey="p50"
                            stroke="#C5A059"
                            strokeWidth={2}
                            fill="transparent"
                            name="p50"
                        />
                        {/* 10th Percentile (Bottom) used to mask? No, just render lines or stacked areas? 
                             Recharts Area doesn't support 'range' easily without trickery. 
                             Let's just plot lines for simplicity or use 'stackId' if we want bands.
                             Actually, plotting lines is cleaner for "Cone".
                         */}
                    </AreaChart>
                </ResponsiveContainer>
                {/* Re-implementing with Lines for clarity as Recharts Area ranges are tricky */}
            </div>
        </div>
    );
};
// Redoing above component to use Lines for clarity in next step

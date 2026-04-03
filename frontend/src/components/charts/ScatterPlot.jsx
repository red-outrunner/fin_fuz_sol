import React from 'react';
import { ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Label } from 'recharts';

const ScatterPlot = ({ data }) => {
    const { stats } = data;
    const scatterData = Object.keys(stats.month_avg).map(month => ({
        month: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][month - 1],
        return: stats.month_avg[month] * 100,
        risk: stats.std_dev[month] * 100,
        positiveRate: stats.positive_rate[month] * 100
    }));

    return (
        <div className="h-72 md:h-96 w-full">
            <ResponsiveContainer width="100%" height="100%">
                <ScatterChart
                    margin={{ top: 20, right: 30, bottom: 20, left: 0 }}
                >
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(26, 36, 51, 0.05)" vertical={false} />
                    <XAxis
                        type="number"
                        dataKey="risk"
                        name="Risk"
                        unit="%"
                        stroke="rgba(26, 36, 51, 0.2)"
                        tick={{ fill: '#1A2433', fontSize: 11, fontWeight: 500 }}
                        axisLine={false}
                        tickLine={false}
                    />
                    <YAxis
                        type="number"
                        dataKey="return"
                        name="Return"
                        unit="%"
                        stroke="rgba(26, 36, 51, 0.2)"
                        tick={{ fill: '#1A2433', fontSize: 11, fontWeight: 500 }}
                        axisLine={false}
                        tickLine={false}
                    />
                    <Tooltip cursor={{ strokeDasharray: '3 3', stroke: 'rgba(197, 160, 89, 0.4)' }} content={({ active, payload }) => {
                        if (active && payload && payload.length) {
                            const data = payload[0].payload;
                            return (
                                <div className="bg-[#1A2433] p-4 rounded-xl shadow-2xl border border-transparent animate-fade-in">
                                    <p className="text-[#C5A059] text-[10px] font-bold uppercase tracking-widest mb-2 pb-2 border-b border-white/10">{data.month} Analysis</p>
                                    <div className="space-y-1.5">
                                        <div className="flex justify-between items-center gap-8">
                                            <span className="text-slate-400 text-xs font-medium">Return</span>
                                            <span className={`text-xs font-bold ${data.return >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                                                {data.return >= 0 ? '+' : ''}{data.return.toFixed(2)}%
                                            </span>
                                        </div>
                                        <div className="flex justify-between items-center gap-8">
                                            <span className="text-slate-400 text-xs font-medium">Risk (σ)</span>
                                            <span className="text-[#F9F7F2] text-xs font-bold">{data.risk.toFixed(2)}%</span>
                                        </div>
                                        <div className="flex justify-between items-center gap-8">
                                            <span className="text-slate-400 text-xs font-medium">Win Rate</span>
                                            <span className="text-[#F9F7F2] text-xs font-bold">{data.positiveRate.toFixed(1)}%</span>
                                        </div>
                                    </div>
                                </div>
                            );
                        }
                        return null;
                    }} />
                    <Scatter
                        name="Monthly Risk/Return"
                        data={scatterData}
                        fill="#C5A059"
                        fillOpacity={0.8}
                        stroke="#1A2433"
                        strokeWidth={1}
                    />
                </ScatterChart>
            </ResponsiveContainer>
        </div>
    );
};

export default ScatterPlot;

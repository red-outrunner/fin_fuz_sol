
import React from 'react';
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

const DrawdownChart = ({ data }) => {
    // data is the array of {date, value} where value is e.g. -0.15 for -15%
    if (!data) return null;

    return (
        <div className="w-full">
            <div className="h-64 w-full">
                <ResponsiveContainer width="100%" height="100%">
                    <AreaChart data={data}>
                        <defs>
                            <linearGradient id="colorDrawdown" x1="0" y1="0" x2="0" y2="1">
                                <stop offset="5%" stopColor="#8C4A4A" stopOpacity={0.8} />
                                <stop offset="95%" stopColor="#8C4A4A" stopOpacity={0.1} />
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
                            tickFormatter={(val) => `${(val * 100).toFixed(0)}%`}
                        />
                        <Tooltip
                            contentStyle={{ backgroundColor: '#F9F7F2', borderColor: '#8C4A4A' }}
                            formatter={(value) => [`${(value * 100).toFixed(2)}%`, 'Drawdown']}
                        />
                        <Area
                            type="step"
                            dataKey="value"
                            stroke="#8C4A4A"
                            fill="url(#colorDrawdown)"
                        />
                    </AreaChart>
                </ResponsiveContainer>
            </div>
        </div>
    );
};

export default DrawdownChart;

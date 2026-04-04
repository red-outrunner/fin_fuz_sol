
import React from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts';

const AnnualReturns = ({ data }) => {
    // data: {year: 2020, value: 0.15}
    if (!data) return null;

    return (
        <div className="w-full">
            <div className="h-64 w-full">
                <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={data}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#F0EBE0" vertical={false} />
                        <XAxis
                            dataKey="year"
                            stroke="#8C735A"
                        />
                        <YAxis
                            stroke="#8C735A"
                            tickFormatter={(val) => `${(val * 100).toFixed(0)}%`}
                        />
                        <Tooltip
                            contentStyle={{ backgroundColor: '#F9F7F2', borderColor: '#C5A059' }}
                            cursor={{ fill: '#F0EBE0', opacity: 0.4 }}
                            formatter={(value) => [`${(value * 100).toFixed(2)}%`, 'Return']}
                        />
                        <Bar dataKey="value" radius={[2, 2, 0, 0]}>
                            {data.map((entry, index) => (
                                <Cell key={`cell-${index}`} fill={entry.value >= 0 ? '#4A7C59' : '#8C4A4A'} />
                            ))}
                        </Bar>
                    </BarChart>
                </ResponsiveContainer>
            </div>
        </div>
    );
};

export default AnnualReturns;

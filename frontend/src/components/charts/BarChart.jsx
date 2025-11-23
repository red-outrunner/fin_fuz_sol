import React from 'react';
import { BarChart as RechartsBarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, ReferenceLine, Cell } from 'recharts';

const BarChart = ({ data, metric = 'Mean' }) => {
    const { stats } = data;
    const chartData = Object.keys(stats.month_avg).map(month => ({
        name: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][month - 1],
        value: (metric === 'Mean' ? stats.month_avg[month] : stats.month_median[month]) * 100
    }));

    const overallAvg = stats.overall_avg * 100;

    return (
        <div className="h-96 w-full bg-white p-6 rounded-sm border border-beige shadow-sm">
            <ResponsiveContainer width="100%" height="100%">
                <RechartsBarChart
                    data={chartData}
                    margin={{ top: 20, right: 30, left: 20, bottom: 5 }}
                >
                    <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#F0EBE0" />
                    <XAxis dataKey="name" stroke="#8C735A" tick={{ fill: '#2C3E50', fontSize: 12, fontFamily: 'Inter' }} />
                    <YAxis label={{ value: 'Return (%)', angle: -90, position: 'insideLeft', fill: '#8C735A' }} stroke="#8C735A" tick={{ fill: '#2C3E50', fontSize: 12, fontFamily: 'Inter' }} />
                    <Tooltip
                        formatter={(value) => [`${value.toFixed(2)}%`, metric]}
                        contentStyle={{ backgroundColor: '#F9F7F2', borderColor: '#C5A059', fontFamily: 'Inter' }}
                        itemStyle={{ color: '#1A2433' }}
                    />
                    <Legend wrapperStyle={{ fontFamily: 'Inter', color: '#2C3E50' }} />
                    <ReferenceLine y={0} stroke="#2C3E50" />
                    <ReferenceLine y={overallAvg} stroke="#C5A059" strokeDasharray="3 3" label={{ value: 'Avg', fill: '#C5A059', fontSize: 12 }} />
                    <Bar dataKey="value" name={`${metric} Monthly Return`}>
                        {chartData.map((entry, index) => (
                            <Cell key={`cell-${index}`} fill={entry.value >= 0 ? '#4A7C59' : '#8C4A4A'} />
                        ))}
                    </Bar>
                </RechartsBarChart>
            </ResponsiveContainer>
        </div>
    );
};

export default BarChart;

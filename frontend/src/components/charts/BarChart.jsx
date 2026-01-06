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
        <div className="h-96 w-full">
            <ResponsiveContainer width="100%" height="100%">
                <RechartsBarChart
                    data={chartData}
                    margin={{ top: 20, right: 30, left: 0, bottom: 5 }}
                >
                    <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="rgba(26, 36, 51, 0.05)" />
                    <XAxis
                        dataKey="name"
                        stroke="rgba(26, 36, 51, 0.2)"
                        tick={{ fill: '#1A2433', fontSize: 11, fontWeight: 500 }}
                        axisLine={false}
                        tickLine={false}
                        dy={10}
                    />
                    <YAxis
                        stroke="rgba(26, 36, 51, 0.2)"
                        tick={{ fill: '#1A2433', fontSize: 11, fontWeight: 500 }}
                        axisLine={false}
                        tickLine={false}
                        tickFormatter={(value) => `${value}%`}
                    />
                    <Tooltip
                        cursor={{ fill: 'rgba(197, 160, 89, 0.05)' }}
                        formatter={(value) => [`${value.toFixed(2)}%`, metric === 'Mean' ? 'Average' : 'Median']}
                        contentStyle={{
                            backgroundColor: '#1A2433',
                            borderColor: 'transparent',
                            borderRadius: '12px',
                            boxShadow: '0 10px 15px -3px rgba(0, 0, 0, 0.1)',
                            padding: '12px'
                        }}
                        itemStyle={{ color: '#F9F7F2', fontSize: '12px', fontWeight: 600 }}
                        labelStyle={{ color: '#C5A059', fontSize: '10px', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.1em', marginBottom: '4px' }}
                    />
                    <ReferenceLine y={0} stroke="rgba(26, 36, 51, 0.1)" />
                    <ReferenceLine
                        y={overallAvg}
                        stroke="#C5A059"
                        strokeDasharray="4 4"
                        label={{ value: 'Portfolio Avg', fill: '#C5A059', fontSize: 10, fontWeight: 700, position: 'right', offset: 10 }}
                    />
                    <Bar dataKey="value" radius={[4, 4, 0, 0]} maxBarSize={40}>
                        {chartData.map((entry, index) => (
                            <Cell key={`cell-${index}`} fill={entry.value >= 0 ? '#C5A059' : '#1A2433'} fillOpacity={0.9} />
                        ))}
                    </Bar>
                </RechartsBarChart>
            </ResponsiveContainer>
        </div>
    );
};

export default BarChart;

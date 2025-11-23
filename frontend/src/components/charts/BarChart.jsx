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
                    margin={{ top: 20, right: 30, left: 20, bottom: 5 }}
                >
                    <CartesianGrid strokeDasharray="3 3" vertical={false} />
                    <XAxis dataKey="name" />
                    <YAxis label={{ value: 'Return (%)', angle: -90, position: 'insideLeft' }} />
                    <Tooltip formatter={(value) => [`${value.toFixed(2)}%`, metric]} />
                    <Legend />
                    <ReferenceLine y={0} stroke="#000" />
                    <ReferenceLine y={overallAvg} stroke="red" strokeDasharray="3 3" label="Avg" />
                    <Bar dataKey="value" name={`${metric} Monthly Return`}>
                        {chartData.map((entry, index) => (
                            <Cell key={`cell-${index}`} fill={entry.value >= 0 ? '#3b82f6' : '#ef4444'} />
                        ))}
                    </Bar>
                </RechartsBarChart>
            </ResponsiveContainer>
        </div>
    );
};

export default BarChart;

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
        <div className="h-96 w-full">
            <ResponsiveContainer width="100%" height="100%">
                <ScatterChart
                    margin={{ top: 20, right: 20, bottom: 20, left: 20 }}
                >
                    <CartesianGrid />
                    <XAxis type="number" dataKey="risk" name="Risk (Std Dev %)" unit="%" />
                    <YAxis type="number" dataKey="return" name="Average Return %" unit="%" />
                    <Tooltip cursor={{ strokeDasharray: '3 3' }} content={({ active, payload }) => {
                        if (active && payload && payload.length) {
                            const data = payload[0].payload;
                            return (
                                <div className="bg-white p-2 border border-slate-200 shadow rounded">
                                    <p className="font-bold">{data.month}</p>
                                    <p>Return: {data.return.toFixed(2)}%</p>
                                    <p>Risk: {data.risk.toFixed(2)}%</p>
                                    <p>Positive Rate: {data.positiveRate.toFixed(1)}%</p>
                                </div>
                            );
                        }
                        return null;
                    }} />
                    <Scatter name="Risk vs Return" data={scatterData} fill="#8884d8">
                        <Label dataKey="month" position="top" />
                    </Scatter>
                </ScatterChart>
            </ResponsiveContainer>
        </div>
    );
};

export default ScatterPlot;

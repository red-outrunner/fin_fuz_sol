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
        <div className="h-96 w-full bg-white p-6 rounded-sm border border-beige shadow-sm">
            <ResponsiveContainer width="100%" height="100%">
                <ScatterChart
                    margin={{ top: 20, right: 20, bottom: 20, left: 20 }}
                >
                    <CartesianGrid strokeDasharray="3 3" stroke="#F0EBE0" />
                    <XAxis type="number" dataKey="risk" name="Risk (Std Dev %)" unit="%" stroke="#8C735A" tick={{ fill: '#2C3E50' }} />
                    <YAxis type="number" dataKey="return" name="Average Return %" unit="%" stroke="#8C735A" tick={{ fill: '#2C3E50' }} />
                    <Tooltip cursor={{ strokeDasharray: '3 3' }} content={({ active, payload }) => {
                        if (active && payload && payload.length) {
                            const data = payload[0].payload;
                            return (
                                <div className="bg-cream p-3 border border-gold shadow-md rounded-sm">
                                    <p className="font-serif font-bold text-navy mb-2">{data.month}</p>
                                    <p className="text-sm text-slate-600">Return: <span className={data.return >= 0 ? 'text-success font-bold' : 'text-error font-bold'}>{data.return.toFixed(2)}%</span></p>
                                    <p className="text-sm text-slate-600">Risk: <span className="font-bold">{data.risk.toFixed(2)}%</span></p>
                                    <p className="text-sm text-slate-600">Positive Rate: {data.positiveRate.toFixed(1)}%</p>
                                </div>
                            );
                        }
                        return null;
                    }} />
                    <Scatter name="Risk vs Return" data={scatterData} fill="#1A2433">
                        <Label dataKey="month" position="top" style={{ fill: '#2C3E50', fontSize: 10, fontFamily: 'Inter' }} />
                    </Scatter>
                </ScatterChart>
            </ResponsiveContainer>
        </div>
    );
};

export default ScatterPlot;

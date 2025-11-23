import React from 'react';
import { ResponsiveContainer, XAxis, YAxis, Tooltip, CartesianGrid, ScatterChart, Scatter, Cell } from 'recharts';

const Heatmap = ({ data }) => {
    const { pivot_data } = data;
    const months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];

    // Transform pivot data for scatter plot heatmap simulation
    const heatmapData = [];
    pivot_data.forEach(row => {
        months.forEach((month, index) => {
            const monthNum = index + 1;
            if (row[monthNum] !== null && row[monthNum] !== undefined) {
                heatmapData.push({
                    year: row.year,
                    month: monthNum,
                    value: row[monthNum] * 100,
                    monthName: month
                });
            }
        });
    });

    const getColor = (value) => {
        if (value > 0) return `rgba(0, 255, 0, ${Math.min(value / 5, 1)})`;
        return `rgba(255, 0, 0, ${Math.min(Math.abs(value) / 5, 1)})`;
    };

    return (
        <div className="h-96 w-full">
            <ResponsiveContainer width="100%" height="100%">
                <ScatterChart
                    margin={{ top: 20, right: 20, bottom: 20, left: 20 }}
                >
                    <CartesianGrid />
                    <XAxis type="category" dataKey="monthName" allowDuplicatedCategory={false} />
                    <YAxis type="number" dataKey="year" domain={['auto', 'auto']} reversed />
                    <Tooltip cursor={{ strokeDasharray: '3 3' }} content={({ active, payload }) => {
                        if (active && payload && payload.length) {
                            const data = payload[0].payload;
                            return (
                                <div className="bg-white p-2 border border-slate-200 shadow rounded">
                                    <p className="font-bold">{data.monthName} {data.year}</p>
                                    <p className={data.value >= 0 ? 'text-green-600' : 'text-red-600'}>
                                        {data.value.toFixed(2)}%
                                    </p>
                                </div>
                            );
                        }
                        return null;
                    }} />
                    <Scatter data={heatmapData} shape="square">
                        {heatmapData.map((entry, index) => (
                            <Cell key={`cell - ${index} `} fill={entry.value >= 0 ? '#22c55e' : '#ef4444'} fillOpacity={Math.min(Math.abs(entry.value) / 10 + 0.3, 1)} />
                        ))}
                    </Scatter>
                </ScatterChart>
            </ResponsiveContainer>
        </div>
    );
};

export default Heatmap;

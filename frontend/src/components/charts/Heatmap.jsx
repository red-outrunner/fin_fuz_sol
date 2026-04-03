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

    const renderShape = (props) => {
        const { cx, cy, payload } = props;
        const value = payload.value;
        const textColor = Math.abs(value) > 5 ? '#F9F7F2' : '#1A2433'; // Cream text for dark blocks, Navy for light

        return (
            <g>
                <rect x={cx - 22} y={cy - 12} width={44} height={24} fill={props.fill} fillOpacity={props.fillOpacity} rx={2} stroke="#F0EBE0" strokeWidth={0.5} />
                <text x={cx} y={cy} dy={4} textAnchor="middle" fill={textColor} fontSize={10} fontFamily="Inter" fontWeight="500">
                    {value.toFixed(1)}
                </text>
            </g>
        );
    };

    return (
        <div className="h-[400px] md:h-[600px] w-full overflow-y-auto overflow-x-hidden bg-white p-2 md:p-6 rounded-sm border border-beige shadow-sm">
            <ResponsiveContainer width="100%" height={Math.max(400, heatmapData.length / 12 * 40)}>
                <ScatterChart
                    margin={{ top: 20, right: 20, bottom: 60, left: 20 }}
                >
                    <CartesianGrid strokeDasharray="3 3" stroke="#F0EBE0" />
                    <XAxis
                        type="category"
                        dataKey="monthName"
                        allowDuplicatedCategory={false}
                        domain={months}
                        stroke="#8C735A"
                        tick={{ fill: '#2C3E50', dy: 20 }}
                        ticks={months}
                        interval={0}
                    />
                    <YAxis type="number" dataKey="year" domain={['dataMin', 'dataMax']} reversed tickCount={heatmapData.length / 12} stroke="#8C735A" tick={{ fill: '#2C3E50' }} />
                    <Tooltip cursor={{ strokeDasharray: '3 3' }} content={({ active, payload }) => {
                        if (active && payload && payload.length) {
                            const data = payload[0].payload;
                            return (
                                <div className="bg-cream p-3 border border-gold shadow-md rounded-sm z-50">
                                    <p className="font-serif font-bold text-navy">{data.monthName} {data.year}</p>
                                    <p className={data.value >= 0 ? 'text-success font-bold' : 'text-error font-bold'}>
                                        {data.value.toFixed(2)}%
                                    </p>
                                </div>
                            );
                        }
                        return null;
                    }} />
                    <Scatter data={heatmapData} shape={renderShape}>
                        {heatmapData.map((entry, index) => (
                            <Cell key={`cell-${index}`} fill={entry.value >= 0 ? '#4A7C59' : '#8C4A4A'} fillOpacity={Math.min(Math.abs(entry.value) / 10 + 0.3, 1)} />
                        ))}
                    </Scatter>
                </ScatterChart>
            </ResponsiveContainer>
        </div>
    );
};

export default Heatmap;

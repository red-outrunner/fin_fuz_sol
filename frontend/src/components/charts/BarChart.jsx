import React from 'react';
import { BarChart as RechartsBarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine, Cell } from 'recharts';
import { useChartColors } from '../../utils/chartTheme';
import { buildMonthSeries } from '../../utils/monthSeasonality';

const BarChart = ({ data, metric = 'Mean' }) => {
    const colors = useChartColors();
    const { stats } = data;
    const field = metric === 'Mean' ? 'month_avg' : 'month_median';
    const chartData = buildMonthSeries(stats, field, true);
    const overallAvg = stats.overall_avg * 100;

    return (
        <div className="h-72 md:h-96 w-full">
            <ResponsiveContainer width="100%" height="100%">
                <RechartsBarChart
                    data={chartData}
                    margin={{ top: 20, right: 30, left: 0, bottom: 5 }}
                >
                    <CartesianGrid strokeDasharray="3 3" vertical={false} stroke={colors.gridColor} />
                    <XAxis
                        dataKey="name"
                        stroke={colors.axisColor}
                        tick={{ fill: colors.tickColor, fontSize: 11, fontWeight: 500 }}
                        axisLine={false}
                        tickLine={false}
                        dy={10}
                    />
                    <YAxis
                        stroke={colors.axisColor}
                        tick={{ fill: colors.tickColor, fontSize: 11, fontWeight: 500 }}
                        axisLine={false}
                        tickLine={false}
                        tickFormatter={(value) => `${value}%`}
                    />
                    <Tooltip
                        cursor={{ fill: colors.cursorBg, opacity: 0.4 }}
                        formatter={(value) => [`${value.toFixed(2)}%`, metric === 'Mean' ? 'Average' : 'Median']}
                        contentStyle={{
                            backgroundColor: colors.tooltipBg,
                            borderColor: colors.tooltipBorder,
                            borderRadius: '12px',
                            boxShadow: '0 10px 15px -3px rgba(0, 0, 0, 0.1)',
                            padding: '12px',
                        }}
                        itemStyle={{ color: colors.tooltipText, fontSize: '12px', fontWeight: 600 }}
                        labelStyle={{
                            color: colors.lineGold,
                            fontSize: '10px',
                            fontWeight: 700,
                            textTransform: 'uppercase',
                            letterSpacing: '0.1em',
                            marginBottom: '4px',
                        }}
                    />
                    <ReferenceLine y={0} stroke={colors.gridColor} />
                    <ReferenceLine
                        y={overallAvg}
                        stroke={colors.lineGold}
                        strokeDasharray="4 4"
                        label={{
                            value: `Avg ${overallAvg >= 0 ? '+' : ''}${overallAvg.toFixed(2)}%`,
                            fill: colors.lineGold,
                            fontSize: 10,
                            fontWeight: 700,
                            position: 'insideTopRight',
                        }}
                    />
                    <Bar dataKey="value" radius={[4, 4, 0, 0]} maxBarSize={40}>
                        {chartData.map((entry, index) => (
                            <Cell
                                key={`cell-${index}`}
                                fill={entry.value >= 0 ? colors.barPositiveGold : colors.barNegative}
                                fillOpacity={0.9}
                            />
                        ))}
                    </Bar>
                </RechartsBarChart>
            </ResponsiveContainer>
        </div>
    );
};

export default BarChart;

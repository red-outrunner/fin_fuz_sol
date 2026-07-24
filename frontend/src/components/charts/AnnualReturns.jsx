import React from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts';
import { getChartColors, isDarkMode } from '../../utils/chartTheme';

const AnnualReturns = ({ data }) => {
    const isDark = isDarkMode();
    const colors = getChartColors(isDark);
    
    if (!data) return null;

    return (
        <div className="w-full">
            <div className="h-64 w-full">
                <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={data}>
                        <CartesianGrid strokeDasharray="3 3" stroke={colors.gridColor} vertical={false} />
                        <XAxis
                            dataKey="year"
                            stroke={colors.axisColor}
                            tick={{ fill: colors.tickColor, fontSize: 12 }}
                        />
                        <YAxis
                            stroke={colors.axisColor}
                            tick={{ fill: colors.tickColor, fontSize: 12 }}
                            tickFormatter={(val) => `${(val * 100).toFixed(0)}%`}
                        />
                        <Tooltip
                            contentStyle={{ 
                                backgroundColor: colors.tooltipBg, 
                                borderColor: colors.tooltipBorder,
                                color: colors.tooltipText
                            }}
                            itemStyle={{ color: colors.tooltipText }}
                            cursor={{ fill: colors.cursorBg, opacity: 0.4 }}
                            formatter={(value) => [`${(value * 100).toFixed(2)}%`, 'Return']}
                        />
                        <Bar dataKey="value" radius={[2, 2, 0, 0]}>
                            {data.map((entry, index) => (
                                <Cell 
                                    key={`cell-${index}`} 
                                    fill={entry.value >= 0 ? colors.barPositive : colors.barNegative} 
                                />
                            ))}
                        </Bar>
                    </BarChart>
                </ResponsiveContainer>
            </div>
        </div>
    );
};

export default AnnualReturns;

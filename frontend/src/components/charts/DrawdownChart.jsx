import React from 'react';
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { getChartColors, isDarkMode } from '../../utils/chartTheme';

const DrawdownChart = ({ data }) => {
    const isDark = isDarkMode();
    const colors = getChartColors(isDark);
    
    if (!data) return null;

    return (
        <div className="w-full">
            <div className="h-64 w-full">
                <ResponsiveContainer width="100%" height="100%">
                    <AreaChart data={data}>
                        <defs>
                            <linearGradient id="colorDrawdown" x1="0" y1="0" x2="0" y2="1">
                                <stop offset="5%" stopColor={colors.barNegative} stopOpacity={0.8} />
                                <stop offset="95%" stopColor={colors.barNegative} stopOpacity={0.1} />
                            </linearGradient>
                        </defs>
                        <CartesianGrid strokeDasharray="3 3" stroke={colors.gridColor} vertical={false} />
                        <XAxis
                            dataKey="date"
                            stroke={colors.axisColor}
                            tick={{ fill: colors.tickColor, fontSize: 12 }}
                            tickFormatter={(str) => str.substring(0, 4)}
                            minTickGap={50}
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
                            formatter={(value) => [`${(value * 100).toFixed(2)}%`, 'Drawdown']}
                        />
                        <Area
                            type="step"
                            dataKey="value"
                            stroke={colors.barNegative}
                            fill="url(#colorDrawdown)"
                        />
                    </AreaChart>
                </ResponsiveContainer>
            </div>
        </div>
    );
};

export default DrawdownChart;

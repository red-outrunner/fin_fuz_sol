import React from 'react';
import { ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { useChartColors } from '../../utils/chartTheme';
import { buildScatterMonthSeries } from '../../utils/monthSeasonality';

const ScatterPlot = ({ data }) => {
    const colors = useChartColors();
    const { stats } = data;
    const scatterData = buildScatterMonthSeries(stats);

    return (
        <div className="h-72 md:h-96 w-full">
            <ResponsiveContainer width="100%" height="100%">
                <ScatterChart
                    margin={{ top: 20, right: 30, bottom: 20, left: 0 }}
                >
                    <CartesianGrid strokeDasharray="3 3" stroke={colors.gridColor} vertical={false} />
                    <XAxis
                        type="number"
                        dataKey="risk"
                        name="Risk"
                        unit="%"
                        stroke={colors.axisColor}
                        tick={{ fill: colors.tickColor, fontSize: 11, fontWeight: 500 }}
                        axisLine={false}
                        tickLine={false}
                    />
                    <YAxis
                        type="number"
                        dataKey="return"
                        name="Return"
                        unit="%"
                        stroke={colors.axisColor}
                        tick={{ fill: colors.tickColor, fontSize: 11, fontWeight: 500 }}
                        axisLine={false}
                        tickLine={false}
                    />
                    <Tooltip
                        cursor={{ strokeDasharray: '3 3', stroke: colors.lineGold }}
                        content={({ active, payload }) => {
                            if (active && payload && payload.length) {
                                const point = payload[0].payload;
                                return (
                                    <div
                                        className="p-4 rounded-xl shadow-2xl border animate-fade-in"
                                        style={{
                                            backgroundColor: colors.tooltipBg,
                                            borderColor: colors.tooltipBorder,
                                        }}
                                    >
                                        <p
                                            className="text-[10px] font-bold uppercase tracking-widest mb-2 pb-2 border-b"
                                            style={{ color: colors.lineGold, borderColor: colors.tooltipBorder }}
                                        >
                                            {point.month} Analysis
                                        </p>
                                        <div className="space-y-1.5">
                                            <div className="flex justify-between items-center gap-8">
                                                <span className="text-xs font-medium" style={{ color: colors.tickColor }}>Return</span>
                                                <span className={`text-xs font-bold ${point.return >= 0 ? 'text-green-500' : 'text-red-400'}`}>
                                                    {point.return >= 0 ? '+' : ''}{point.return.toFixed(2)}%
                                                </span>
                                            </div>
                                            <div className="flex justify-between items-center gap-8">
                                                <span className="text-xs font-medium" style={{ color: colors.tickColor }}>Risk (σ)</span>
                                                <span className="text-xs font-bold" style={{ color: colors.tooltipText }}>{point.risk.toFixed(2)}%</span>
                                            </div>
                                            <div className="flex justify-between items-center gap-8">
                                                <span className="text-xs font-medium" style={{ color: colors.tickColor }}>Win Rate</span>
                                                <span className="text-xs font-bold" style={{ color: colors.tooltipText }}>{point.positiveRate.toFixed(1)}%</span>
                                            </div>
                                        </div>
                                    </div>
                                );
                            }
                            return null;
                        }}
                    />
                    <Scatter
                        name="Monthly Risk/Return"
                        data={scatterData}
                        fill={colors.lineGold}
                        fillOpacity={0.8}
                        stroke={colors.tooltipText}
                        strokeWidth={1}
                    />
                </ScatterChart>
            </ResponsiveContainer>
        </div>
    );
};

export default ScatterPlot;

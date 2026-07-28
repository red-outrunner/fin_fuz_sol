import React from 'react';
import { Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ComposedChart, Area, Legend } from 'recharts';
import { useChartColors } from '../../utils/chartTheme';

/** Match wealth-index dates to moving-average dict keys (YYYY-MM-DD variants). */
const lookupSeries = (dict, date) => {
    if (!dict || !date) return null;
    if (dict[date] != null) return dict[date];
    const prefix = date.substring(0, 7);
    const key = Object.keys(dict).find((k) => k.startsWith(prefix));
    return key ? dict[key] : null;
};

/** Scale price MAs onto the wealth-index axis so 1yr/5yr track standard monthly MAs. */
const buildWealthWithMA = (wealthData, movingAverages) => {
    if (!wealthData?.length) return [];

    const prices = movingAverages?.prices;
    const ma12 = movingAverages?.ma_12;
    const ma60 = movingAverages?.ma_60;

    if (!prices) {
        return wealthData.map((p) => ({ ...p, sma1yr: null, sma5yr: null }));
    }

    return wealthData.map((point) => {
        const price = lookupSeries(prices, point.date);
        const m12 = lookupSeries(ma12, point.date);
        const m60 = lookupSeries(ma60, point.date);

        return {
            ...point,
            sma1yr: price && m12 && point.value ? point.value * (m12 / price) : null,
            sma5yr: price && m60 && point.value ? point.value * (m60 / price) : null,
        };
    });
};

const WealthChart = ({ data }) => {
    const colors = useChartColors();

    if (!data || !data.stats || !data.stats.wealth_index) {
        return <div className="p-6 text-center text-slate-500 dark:text-slate-400 italic">No wealth data available</div>;
    }

    const [showSMA12, setShowSMA12] = React.useState(false);
    const [showSMA60, setShowSMA60] = React.useState(false);

    const chartData = React.useMemo(
        () => buildWealthWithMA(data.stats.wealth_index, data.moving_averages),
        [data.stats.wealth_index, data.moving_averages],
    );

    return (
        <div className="w-full">
            <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3 mb-4">
                <p className="text-[10px] text-slate-400 italic max-w-md">
                    12-mo avg appears after 1 year of data; 60-mo avg after 5 years (standard price moving averages, scaled to R10,000 growth).
                </p>
                <div className="flex justify-end gap-4 shrink-0">
                    <label className="flex items-center gap-2 text-xs font-bold uppercase text-slate-500 dark:text-slate-400 cursor-pointer hover:text-gold dark:hover:text-gold">
                        <input type="checkbox" checked={showSMA12} onChange={(e) => setShowSMA12(e.target.checked)} className="accent-gold" />
                        12-mo Avg
                    </label>
                    <label className="flex items-center gap-2 text-xs font-bold uppercase text-slate-500 dark:text-slate-400 cursor-pointer hover:text-gold dark:hover:text-gold">
                        <input type="checkbox" checked={showSMA60} onChange={(e) => setShowSMA60(e.target.checked)} className="accent-[#4A7C59]" />
                        60-mo Avg
                    </label>
                </div>
            </div>
            <div className="h-96 w-full">
                <ResponsiveContainer width="100%" height="100%">
                    <ComposedChart data={chartData} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
                        <defs>
                            <linearGradient id="colorValue" x1="0" y1="0" x2="0" y2="1">
                                <stop offset="5%" stopColor={colors.lineGold} stopOpacity={colors.isDark ? 0.2 : 0.3} />
                                <stop offset="95%" stopColor={colors.lineGold} stopOpacity={0} />
                            </linearGradient>
                        </defs>
                        <CartesianGrid strokeDasharray="3 3" stroke={colors.gridColor} vertical={false} />
                        <XAxis
                            dataKey="date"
                            stroke={colors.axisColor}
                            tick={{ fill: colors.tickColor, fontSize: 12, fontFamily: 'Inter' }}
                            tickFormatter={(str) => str.substring(0, 4)}
                            minTickGap={50}
                        />
                        <YAxis
                            stroke={colors.axisColor}
                            tick={{ fill: colors.tickColor, fontSize: 12, fontFamily: 'Inter' }}
                            tickFormatter={(value) => `R${value.toLocaleString()}`}
                        />
                        <Tooltip
                            contentStyle={{
                                backgroundColor: colors.tooltipBg,
                                borderColor: colors.tooltipBorder,
                                color: colors.tooltipText,
                                fontFamily: 'Inter',
                            }}
                            itemStyle={{ color: colors.tooltipText }}
                            formatter={(value, name) => [
                                value != null ? `R${Number(value).toLocaleString(undefined, { maximumFractionDigits: 0 })}` : '—',
                                name,
                            ]}
                            labelFormatter={(label) => new Date(label).toLocaleDateString(undefined, { year: 'numeric', month: 'long' })}
                        />
                        {(showSMA12 || showSMA60) && (
                            <Legend wrapperStyle={{ paddingTop: 8, fontSize: 11, color: colors.tickColor }} />
                        )}
                        <Area
                            type="monotone"
                            dataKey="value"
                            name="Portfolio Value"
                            stroke={colors.areaStroke}
                            strokeWidth={2}
                            fillOpacity={1}
                            fill="url(#colorValue)"
                        />
                        {showSMA12 && (
                            <Line
                                type="monotone"
                                dataKey="sma1yr"
                                stroke={colors.lineGold}
                                strokeWidth={2}
                                dot={false}
                                name="12-mo Avg"
                                connectNulls={false}
                            />
                        )}
                        {showSMA60 && (
                            <Line
                                type="monotone"
                                dataKey="sma5yr"
                                stroke={colors.lineGreen}
                                strokeWidth={2}
                                dot={false}
                                name="60-mo Avg"
                                connectNulls={false}
                            />
                        )}
                    </ComposedChart>
                </ResponsiveContainer>
            </div>
        </div>
    );
};

export default WealthChart;

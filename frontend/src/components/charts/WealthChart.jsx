import React from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, AreaChart, Area } from 'recharts';
import { getChartColors, isDarkMode } from '../../utils/chartTheme';

const WealthChart = ({ data }) => {
    const isDark = isDarkMode();
    const colors = getChartColors(isDark);
    
    if (!data || !data.stats || !data.stats.wealth_index) {
        return <div className="p-6 text-center text-slate-500 dark:text-slate-400 italic">No wealth data available</div>;
    }

    const { stats } = data;
    const wealthData = stats.wealth_index;

    const [showSMA12, setShowSMA12] = React.useState(false);
    const [showSMA60, setShowSMA60] = React.useState(false);

    const calculateSMA = (data, window) => {
        return data.map((point, index, array) => {
            if (index < window - 1) return { ...point, [`sma${window}`]: null };
            const slice = array.slice(index - window + 1, index + 1);
            const sum = slice.reduce((acc, curr) => acc + curr.value, 0);
            return { ...point, [`sma${window}`]: sum / window };
        });
    };

    const dataWithSMA12 = React.useMemo(() => calculateSMA(wealthData, 12), [wealthData]);
    const finalData = React.useMemo(() => calculateSMA(dataWithSMA12, 60), [dataWithSMA12]);

    return (
        <div className="w-full">
            <div className="flex justify-end gap-4 mb-4">
                <label className="flex items-center gap-2 text-xs font-bold uppercase text-slate-500 dark:text-slate-400 cursor-pointer hover:text-gold dark:hover:text-gold">
                    <input type="checkbox" checked={showSMA12} onChange={e => setShowSMA12(e.target.checked)} className="accent-gold" />
                    Show 1yr Avg
                </label>
                <label className="flex items-center gap-2 text-xs font-bold uppercase text-slate-500 dark:text-slate-400 cursor-pointer hover:text-gold dark:hover:text-gold">
                    <input type="checkbox" checked={showSMA60} onChange={e => setShowSMA60(e.target.checked)} className="accent-[#4A7C59]" />
                    Show 5yr Avg
                </label>
            </div>
            <div className="h-96 w-full">
                <ResponsiveContainer width="100%" height="100%">
                    <AreaChart data={finalData} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
                        <defs>
                            <linearGradient id="colorValue" x1="0" y1="0" x2="0" y2="1">
                                <stop offset="5%" stopColor="#C5A059" stopOpacity={isDark ? 0.2 : 0.3} />
                                <stop offset="95%" stopColor="#C5A059" stopOpacity={0} />
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
                                fontFamily: 'Inter'
                            }}
                            itemStyle={{ color: colors.tooltipText }}
                            formatter={(value) => [`R${value.toLocaleString(undefined, { maximumFractionDigits: 0 })}`, 'Value']}
                            labelFormatter={(label) => new Date(label).toLocaleDateString(undefined, { year: 'numeric', month: 'long' })}
                        />
                        <Area
                            type="monotone"
                            dataKey="value"
                            stroke={colors.areaStroke}
                            strokeWidth={2}
                            fillOpacity={1}
                            fill="url(#colorValue)"
                        />
                        {showSMA12 && (
                            <Line type="monotone" dataKey="sma12" stroke={colors.lineGold} strokeWidth={2} dot={false} name="1yr MA" />
                        )}
                        {showSMA60 && (
                            <Line type="monotone" dataKey="sma60" stroke={colors.lineGreen} strokeWidth={2} dot={false} name="5yr MA" />
                        )}
                    </AreaChart>
                </ResponsiveContainer>
            </div>
        </div>
    );
};

export default WealthChart;

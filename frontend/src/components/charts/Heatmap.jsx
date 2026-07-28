import React from 'react';
import { useTheme } from '../../context/ThemeContext';
import { getHeatmapCellStyle } from '../../utils/chartTheme';
import { MONTH_LABELS } from '../../utils/monthSeasonality';

/**
 * Monthly returns matrix — calendar heatmap with theme-aware quiet-wealth palette.
 */
const Heatmap = ({ data }) => {
    const { isDark } = useTheme();
    const { pivot_data } = data;
    const gridTemplate = '44px repeat(12, minmax(0, 1fr))';

    const shellClass = isDark
        ? 'bg-[#0b1220] border-white/10'
        : 'bg-[#FDFCF8] border-navy/10';
    const stickyBg = isDark ? 'bg-[#0b1220]' : 'bg-[#FDFCF8]';
    const labelClass = isDark ? 'text-slate-500' : 'text-slate-600';
    const yearClass = isDark ? 'text-slate-400' : 'text-slate-600';

    const legendGradient = isDark
        ? 'linear-gradient(90deg, rgb(140,74,74), rgb(30,41,59), rgb(197,160,89))'
        : 'linear-gradient(90deg, rgb(140,74,74), rgb(226,217,200), rgb(197,160,89))';

    return (
        <div className={`w-full overflow-x-auto overflow-y-auto max-h-[400px] md:max-h-[600px] rounded-lg border shadow-sm ${shellClass}`}>
            <div className="min-w-[560px] md:min-w-0 p-3 md:p-4">
                <div className="flex items-center justify-between mb-3 px-0.5">
                    <p className={`text-[9px] font-bold uppercase tracking-[0.15em] ${labelClass}`}>
                        Monthly return · %
                    </p>
                    <div className="flex items-center gap-1.5">
                        <span className={`text-[9px] font-mono ${labelClass}`}>−10</span>
                        <div className="h-1.5 w-20 rounded-sm" style={{ background: legendGradient }} />
                        <span className={`text-[9px] font-mono ${labelClass}`}>+10</span>
                    </div>
                </div>

                <div
                    className={`grid gap-[3px] mb-[3px] sticky top-0 z-10 pb-1 ${stickyBg}`}
                    style={{ gridTemplateColumns: gridTemplate }}
                >
                    <div />
                    {MONTH_LABELS.map((m) => (
                        <div
                            key={m}
                            className={`text-center text-[9px] md:text-[10px] font-bold uppercase tracking-wider ${labelClass}`}
                        >
                            {m}
                        </div>
                    ))}
                </div>

                {pivot_data.map((row) => (
                    <div
                        key={row.year}
                        className="grid gap-[3px] mb-[3px]"
                        style={{ gridTemplateColumns: gridTemplate }}
                    >
                        <div className={`flex items-center justify-end pr-2 text-[10px] md:text-[11px] font-mono font-bold sticky left-0 z-10 ${yearClass} ${stickyBg}`}>
                            {row.year}
                        </div>
                        {MONTH_LABELS.map((m, idx) => {
                            const value = row[idx + 1];
                            const has = value !== null && value !== undefined;
                            return (
                                <div
                                    key={m}
                                    title={has ? `${m} ${row.year}: ${(value * 100).toFixed(2)}%` : `${m} ${row.year}: —`}
                                    className="flex items-center justify-center rounded-[2px] h-7 md:h-9 text-[9px] md:text-[10px] font-mono font-semibold tabular-nums select-none transition-transform hover:scale-[1.04] hover:z-10 hover:ring-1 hover:ring-gold/50"
                                    style={getHeatmapCellStyle(value, isDark)}
                                >
                                    {has ? (value * 100).toFixed(1) : '·'}
                                </div>
                            );
                        })}
                    </div>
                ))}
            </div>
        </div>
    );
};

export default Heatmap;

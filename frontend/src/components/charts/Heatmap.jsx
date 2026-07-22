import React from 'react';

/**
 * Monthly returns matrix — terminal-style calendar heatmap.
 */
const Heatmap = ({ data }) => {
    const { pivot_data } = data;
    const months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
    const gridTemplate = '44px repeat(12, minmax(0, 1fr))';

    const cellStyle = (value) => {
        if (value === null || value === undefined) {
            return { backgroundColor: 'rgba(148,163,184,0.08)', color: '#64748b' };
        }
        const v = value * 100;
        const intensity = Math.min(Math.abs(v) / 10, 1);
        if (v >= 0) {
            const a = 0.2 + intensity * 0.75;
            return { backgroundColor: `rgba(16, 185, 129, ${a})`, color: intensity > 0.45 ? '#ecfdf5' : '#064e3b' };
        }
        const a = 0.2 + intensity * 0.75;
        return { backgroundColor: `rgba(244, 63, 94, ${a})`, color: intensity > 0.45 ? '#fff1f2' : '#881337' };
    };

    return (
        <div className="w-full overflow-x-auto overflow-y-auto max-h-[400px] md:max-h-[600px] rounded-lg border border-slate-800/20 dark:border-white/10 bg-[#0b1220] dark:bg-[#020617] shadow-sm">
            <div className="min-w-[560px] md:min-w-0 p-3 md:p-4">
                <div className="flex items-center justify-between mb-3 px-0.5">
                    <p className="text-[9px] font-bold uppercase tracking-[0.15em] text-slate-500">
                        Monthly return · %
                    </p>
                    <div className="flex items-center gap-1.5">
                        <span className="text-[9px] text-slate-500 font-mono">−10</span>
                        <div
                            className="h-1.5 w-20 rounded-sm"
                            style={{
                                background: 'linear-gradient(90deg, rgb(244,63,94), rgb(30,41,59), rgb(16,185,129))',
                            }}
                        />
                        <span className="text-[9px] text-slate-500 font-mono">+10</span>
                    </div>
                </div>

                <div
                    className="grid gap-[3px] mb-[3px] sticky top-0 z-10 bg-[#0b1220] dark:bg-[#020617] pb-1"
                    style={{ gridTemplateColumns: gridTemplate }}
                >
                    <div />
                    {months.map((m) => (
                        <div
                            key={m}
                            className="text-center text-[9px] md:text-[10px] font-bold uppercase tracking-wider text-slate-500"
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
                        <div className="flex items-center justify-end pr-2 text-[10px] md:text-[11px] font-mono font-bold text-slate-400 sticky left-0 bg-[#0b1220] dark:bg-[#020617] z-10">
                            {row.year}
                        </div>
                        {months.map((m, idx) => {
                            const value = row[idx + 1];
                            const has = value !== null && value !== undefined;
                            return (
                                <div
                                    key={m}
                                    title={has ? `${m} ${row.year}: ${(value * 100).toFixed(2)}%` : `${m} ${row.year}: —`}
                                    className="flex items-center justify-center rounded-[2px] h-7 md:h-9 text-[9px] md:text-[10px] font-mono font-semibold tabular-nums select-none transition-transform hover:scale-[1.04] hover:z-10 hover:ring-1 hover:ring-gold/50"
                                    style={cellStyle(value)}
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

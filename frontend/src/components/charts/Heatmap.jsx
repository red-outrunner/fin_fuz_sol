import React from 'react';

const Heatmap = ({ data }) => {
    const { pivot_data } = data;
    const months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];

    // Grid template: a year-label column followed by 12 equal month columns.
    const gridTemplate = '40px repeat(12, minmax(0, 1fr))';

    const cellStyle = (value) => {
        if (value === null || value === undefined) {
            return { backgroundColor: '#F5F1E8' };
        }
        const v = value * 100;
        const intensity = Math.min(Math.abs(v) / 10 + 0.25, 1);
        const base = v >= 0 ? '74, 124, 89' : '140, 74, 74'; // success / error
        return { backgroundColor: `rgba(${base}, ${intensity})` };
    };

    const textColor = (value) => {
        const v = (value ?? 0) * 100;
        return Math.abs(v) > 5 ? '#F9F7F2' : '#1A2433'; // cream on dark cells, navy on light
    };

    return (
        <div className="w-full overflow-x-auto overflow-y-auto max-h-[400px] md:max-h-[600px] bg-white rounded-sm border border-beige shadow-sm">
            {/* min-width keeps cells readable on phones (horizontal scroll); on desktop it fills the card */}
            <div className="min-w-[560px] md:min-w-0 p-2 md:p-4">
                {/* Month header (sticky while scrolling vertically) */}
                <div
                    className="grid gap-[2px] mb-[2px] sticky top-0 z-10 bg-white"
                    style={{ gridTemplateColumns: gridTemplate }}
                >
                    <div className="sticky left-0 bg-white z-10" />
                    {months.map((m) => (
                        <div key={m} className="text-center text-[10px] md:text-xs font-semibold text-navy pb-1">
                            {m}
                        </div>
                    ))}
                </div>

                {/* One row per year */}
                {pivot_data.map((row) => (
                    <div
                        key={row.year}
                        className="grid gap-[2px] mb-[2px]"
                        style={{ gridTemplateColumns: gridTemplate }}
                    >
                        <div className="flex items-center justify-end pr-1.5 text-[10px] md:text-xs font-bold text-navy sticky left-0 bg-white z-10">
                            {row.year}
                        </div>
                        {months.map((m, idx) => {
                            const value = row[idx + 1];
                            const has = value !== null && value !== undefined;
                            return (
                                <div
                                    key={m}
                                    title={has ? `${m} ${row.year}: ${(value * 100).toFixed(2)}%` : `${m} ${row.year}: no data`}
                                    className="flex items-center justify-center rounded-[3px] h-7 md:h-9 text-[9px] md:text-[11px] font-medium tabular-nums select-none"
                                    style={{ ...cellStyle(value), color: textColor(value) }}
                                >
                                    {has ? (value * 100).toFixed(1) : ''}
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

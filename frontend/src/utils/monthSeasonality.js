export const MONTH_LABELS = [
    'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
    'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec',
];

/** Read a month-indexed stat map (keys may be numbers or strings from JSON). */
export function getMonthValue(stats, field, month) {
    if (!stats?.[field]) return null;
    const map = stats[field];
    const val = map[month] ?? map[String(month)];
    return val === undefined ? null : val;
}

/** Build Jan→Dec ordered series for charts and tables. */
export function buildMonthSeries(stats, field = 'month_avg', asPercent = false) {
    return Array.from({ length: 12 }, (_, i) => {
        const month = i + 1;
        const raw = getMonthValue(stats, field, month);
        return {
            month,
            name: MONTH_LABELS[i],
            value: raw === null ? null : (asPercent ? raw * 100 : raw),
            raw,
        };
    });
}

/** Build ordered scatter data for risk/return chart. */
export function buildScatterMonthSeries(stats) {
    return buildMonthSeries(stats, 'month_avg', true)
        .map(({ month, name, value: returnVal }) => {
            const std = getMonthValue(stats, 'std_dev', month);
            const pos = getMonthValue(stats, 'positive_rate', month);
            return {
                month: name,
                return: returnVal,
                risk: std !== null ? std * 100 : null,
                positiveRate: pos !== null ? pos * 100 : null,
            };
        })
        .filter((d) => d.return !== null && d.risk !== null);
}
